# this_file: src/vttiro/providers/gemini/transcriber.py
"""Google Gemini 2.0 Flash transcription provider.

This module implements the Gemini transcription provider using the new
VTTiro 2.0 architecture. Provides context-aware transcription with
WebVTT timing, speaker diarization, and safety filter handling.

Used by:
- Core orchestration for Gemini-based transcription
- Provider selection logic
- Testing infrastructure for Gemini functionality
"""

import asyncio
import os
import time
from pathlib import Path
from typing import Any

from loguru import logger

from vttiro.core.constants import GEMINI_MAX_FILE_SIZE_MB, WEBVTT_DEFAULT_CONFIDENCE
from vttiro.core.errors import (
    APIError,
    AuthenticationError,
    ContentFilterError,
    ProcessingError,
    create_debug_context,
    handle_provider_exception,
)
from vttiro.core.types import TranscriptionResult, TranscriptSegment

# Removed complex type validation - using simple validation in base class
from vttiro.providers.base import TranscriberABC
from vttiro.utils.api_keys import get_api_key_with_fallbacks
from vttiro.utils.logging import log_performance, log_provider_debug
from vttiro.utils.prompt import build_webvtt_prompt, optimize_prompt_for_provider
from vttiro.utils.timestamp import (
    distribute_words_over_duration,
    parse_webvtt_timestamp_line,
)

# Optional dependency handling
try:
    import google.generativeai as genai
    from google.generativeai.types import HarmBlockThreshold, HarmCategory

    GEMINI_AVAILABLE = True
except ImportError:
    genai = None
    HarmCategory = None
    HarmBlockThreshold = None
    GEMINI_AVAILABLE = False


class GeminiTranscriber(TranscriberABC):
    """Google Gemini 2.0 Flash transcription provider.

    Implements high-quality transcription using Google's Gemini 2.0 Flash model
    with WebVTT format output, context-aware prompting, and comprehensive
    safety filter handling.

    Features:
    - Direct WebVTT format output with precise timing
    - Context-aware transcription for improved accuracy
    - Speaker diarization support
    - Configurable safety settings
    - Robust error handling and fallbacks
    """

    def __init__(
        self,
        config_or_api_key=None,
        model: str = "gemini-2.5-flash",
        safety_settings: dict[str, str] | None = None,
    ):
        """Initialize Gemini transcriber.

        Args:
            config_or_api_key: VttiroConfig object or API key string
            model: Gemini model to use
            safety_settings: Custom safety settings override

        Raises:
            ImportError: If google-generativeai is not installed
            AuthenticationError: If API key is missing or invalid
        """
        if not GEMINI_AVAILABLE:
            msg = "Google GenerativeAI not available. Install with: uv add google-generativeai"
            raise ImportError(msg)

        # Handle both config object and direct api_key parameter
        api_key = None
        if config_or_api_key is not None:
            if hasattr(config_or_api_key, "engine"):
                # It's a VttiroConfig object
                # Extract model from config if not explicitly set
                if model == "gemini-2.5-flash":  # default value
                    model = getattr(config_or_api_key, "gemini_model", "gemini-2.5-flash")
                # No API key extraction from config - always use environment variables
                api_key = None
            else:
                # It's an API key string
                api_key = config_or_api_key

        # Get API key from parameter or environment with fallbacks
        self.api_key = get_api_key_with_fallbacks("gemini", api_key)
        if not self.api_key:
            msg = (
                "Gemini API key not provided. Set one of: VTTIRO_GEMINI_API_KEY, "
                "GEMINI_API_KEY, GOOGLE_API_KEY environment variables "
                "or pass api_key parameter."
            )
            raise AuthenticationError(
                msg,
                provider="gemini",
            )

        self.model = model

        # Configure client
        genai.configure(api_key=self.api_key)
        self.client = genai.GenerativeModel(model_name=model)

        # Default safety settings - block none except for harassment/hate
        self.safety_settings = safety_settings or {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        logger.debug(f"Initialized Gemini transcriber with model {model}")

    @property
    def provider_name(self) -> str:
        """Provider identification."""
        return "gemini"

    @property
    def supported_formats(self) -> list[str]:
        """Supported audio formats."""
        return ["mp3", "wav", "m4a", "flac", "webm", "mp4"]

    @property
    def max_file_size_mb(self) -> int:
        """Maximum file size in MB."""
        return GEMINI_MAX_FILE_SIZE_MB

    @property
    def supports_speaker_diarization(self) -> bool:
        """Whether speaker diarization is supported."""
        return True

    async def transcribe(self, audio_path: Path, **kwargs) -> TranscriptionResult:
        """Transcribe audio file using Gemini.

        Args:
            audio_path: Path to audio file
            **kwargs: Additional arguments (language, context, etc.)

        Returns:
            TranscriptionResult with segments and metadata

        Raises:
            ProcessingError: If transcription fails
            APIError: If API request fails
            ContentFilterError: If content is blocked by safety filters
        """
        start_time = time.time()

        try:
            # Validate file
            if not audio_path.exists():
                msg = f"Audio file not found: {audio_path}"
                raise ProcessingError(msg)

            file_size_mb = audio_path.stat().st_size / (1024 * 1024)
            if file_size_mb > self.max_file_size_mb:
                msg = f"File too large: {file_size_mb:.1f}MB > {self.max_file_size_mb}MB"
                raise ProcessingError(msg)

            log_provider_debug(
                "gemini",
                "file_processing",
                {"file_name": audio_path.name, "file_size_mb": file_size_mb},
            )

            # Build context-aware prompt
            prompt = build_webvtt_prompt(
                language=kwargs.get("language", "auto"),
                context=kwargs.get("context", {}),
                include_speaker_diarization=kwargs.get("speaker_diarization", True),
                include_emotions=kwargs.get("emotions", False),
                max_segment_duration=kwargs.get("max_segment_duration", 30.0),
            )

            # Optimize prompt for Gemini
            prompt = optimize_prompt_for_provider(prompt, "gemini")

            # Upload file and generate content
            logger.debug(f"Uploading {audio_path.name} to Gemini")
            audio_file = genai.upload_file(str(audio_path))

            # Wait for processing if needed
            while audio_file.state.name == "PROCESSING":
                logger.debug("Waiting for file processing...")
                await asyncio.sleep(1)
                audio_file = genai.get_file(audio_file.name)

            if audio_file.state.name != "ACTIVE":
                msg = f"File upload failed: {audio_file.state.name}"
                raise ProcessingError(msg)

            logger.debug("Sending transcription request to Gemini")

            # Generate content with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = await asyncio.to_thread(
                        self.client.generate_content,
                        [prompt, audio_file],
                        safety_settings=self.safety_settings,
                        generation_config=genai.types.GenerationConfig(
                            temperature=0.1,  # Low temperature for accuracy
                            max_output_tokens=8192,
                        ),
                    )
                    break
                except Exception as e:
                    if "429" in str(e) and attempt < max_retries - 1:
                        wait_time = 2**attempt
                        logger.warning(f"Rate limit hit, waiting {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        continue
                    raise

            # Clean up uploaded file
            try:
                genai.delete_file(audio_file.name)
                logger.debug("Cleaned up uploaded file")
            except Exception as e:
                logger.warning(f"Failed to delete uploaded file: {e}")

            # Check for content filtering
            if not response.text:
                if hasattr(response, "prompt_feedback"):
                    feedback = response.prompt_feedback
                    if feedback.block_reason:
                        msg = f"Content blocked by safety filters: {feedback.block_reason}"
                        raise ContentFilterError(
                            msg,
                            provider="gemini",
                            filter_type=str(feedback.block_reason),
                        )

                msg = "Empty response from Gemini (possible content filtering)"
                raise ProcessingError(msg)

            # Parse WebVTT response
            webvtt_content = response.text.strip()
            log_provider_debug("gemini", "response_received", {"response_length": len(webvtt_content)})

            # Parse segments from WebVTT
            segments = self._parse_webvtt_response(webvtt_content, audio_path)

            processing_time = time.time() - start_time
            log_performance(
                operation="gemini_transcription",
                duration=processing_time,
                metrics={
                    "file_size_mb": file_size_mb,
                    "segments_count": len(segments),
                },
            )

            return TranscriptionResult(
                segments=segments,
                provider="gemini",
                language=kwargs.get("language", "auto"),
                confidence=WEBVTT_DEFAULT_CONFIDENCE,
                metadata={
                    "model": self.model,
                    "file_size_mb": file_size_mb,
                    "processing_time": processing_time,
                    "raw_response": (webvtt_content[:1000] if kwargs.get("debug") else None),
                },
            )

        except Exception as e:
            context = create_debug_context(
                provider="gemini",
                model=self.model,
                file_path=str(audio_path),
                error=str(e),
            )
            raise handle_provider_exception(e, "gemini", context) from e

    def _parse_webvtt_response(self, webvtt_text: str, audio_path: Path) -> list[TranscriptSegment]:
        """Parse WebVTT format response into segments.

        Args:
            webvtt_text: Raw WebVTT format text from Gemini
            audio_path: Path to original audio file

        Returns:
            List of TranscriptSegment objects
        """
        segments = []
        lines = webvtt_text.strip().split("\n")

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Skip empty lines and WEBVTT header
            if not line or line == "WEBVTT":
                i += 1
                continue

            # Look for timestamp line (e.g., "00:00:00.000 --> 00:00:05.000")
            if "-->" in line:
                try:
                    start_time, end_time = parse_webvtt_timestamp_line(line)

                    # Collect text lines until next timestamp or end
                    text_lines = []
                    i += 1
                    while i < len(lines):
                        next_line = lines[i].strip()
                        if not next_line:
                            break
                        if "-->" in next_line:
                            # This is the next timestamp, don't consume it
                            break
                        text_lines.append(next_line)
                        i += 1

                    if text_lines:
                        text = " ".join(text_lines)

                        # Extract speaker info if present (e.g., "Speaker 1: Hello")
                        speaker = None
                        if ":" in text and text.split(":", 1)[0].strip().lower().startswith("speaker"):
                            speaker_part, text = text.split(":", 1)
                            speaker = speaker_part.strip()
                            text = text.strip()

                        # Distribute words over duration for word-level timing
                        distribute_words_over_duration(text, start_time, end_time)

                        segment = TranscriptSegment(
                            start=start_time,
                            end=end_time,
                            text=text,
                            confidence=WEBVTT_DEFAULT_CONFIDENCE,
                            speaker=speaker,
                        )
                        segments.append(segment)

                except Exception as e:
                    logger.warning(f"Failed to parse WebVTT line: {line} - {e}")
                    i += 1
            else:
                i += 1

        logger.debug(f"Parsed {len(segments)} segments from Gemini response")
        return segments

    async def health_check(self) -> dict[str, Any]:
        """Check provider health and capabilities.

        Returns:
            Dictionary with health status and capabilities
        """
        try:
            # Test API key by making a simple request
            models = genai.list_models()
            available_models = [m.name for m in models if "gemini" in m.name.lower()]

            return {
                "status": "healthy",
                "provider": "gemini",
                "available": True,
                "models": available_models,
                "max_file_size_mb": self.max_file_size_mb,
                "supported_formats": self.supported_formats,
                "supports_diarization": self.supports_speaker_diarization,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": "gemini",
                "available": False,
                "error": str(e),
                "models": [],
            }

    def estimate_cost(self, duration_minutes: float, **kwargs) -> dict[str, Any]:
        """Estimate transcription cost.

        Args:
            duration_minutes: Audio duration in minutes
            **kwargs: Additional parameters

        Returns:
            Cost estimation details
        """
        # Gemini 2.0 Flash pricing (as of Jan 2025)
        # Audio input: $0.075 per minute
        base_cost_per_minute = 0.075

        estimated_cost = duration_minutes * base_cost_per_minute

        return {
            "provider": "gemini",
            "model": self.model,
            "duration_minutes": duration_minutes,
            "estimated_cost_usd": estimated_cost,
            "cost_per_minute": base_cost_per_minute,
            "currency": "USD",
            "last_updated": "2025-01-01",
        }
