# this_file: src/vttiro/providers/openai/transcriber.py
"""OpenAI Whisper transcription provider.

This module implements the OpenAI transcription provider using the new
VTTiro 2.0 architecture. Provides high-quality transcription with
Whisper models, word-level timestamps, and cost-effective processing.

Used by:
- Core orchestration for OpenAI Whisper-based transcription
- Provider selection logic  
- Testing infrastructure for OpenAI functionality
"""

import asyncio
import os
import time
from pathlib import Path
from typing import Any

from ...core.errors import (
    APIError,
    AuthenticationError, 
    ProcessingError,
    handle_provider_exception,
)
from ...core.types import TranscriptionResult, TranscriptSegment
from ...utils.prompt import build_webvtt_prompt, optimize_prompt_for_provider
from ...utils.timestamp import parse_webvtt_timestamp_line, distribute_words_over_duration
# Removed complex type validation
from ..base import TranscriberABC

try:
    from loguru import logger
except ImportError:
    import logging as logger

# Optional dependency handling
try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI not available. Install with: uv add openai")
    openai = None
    OpenAI = None


class OpenAITranscriber(TranscriberABC):
    """OpenAI Whisper transcription provider.
    
    Implements high-quality transcription using OpenAI's Whisper models
    with word-level timestamps, multiple format support, and cost-effective
    processing for various audio types.
    
    Features:
    - Whisper-1 model for high accuracy transcription
    - Direct WebVTT format output with precise timing
    - Word-level timestamps and confidence scores
    - Multi-language support with auto-detection
    - Context-aware prompting for improved accuracy
    - Robust error handling and retries
    """
    
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "whisper-1"
    ):
        """Initialize OpenAI transcriber.
        
        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: OpenAI model to use (whisper-1)
            
        Raises:
            ImportError: If openai package is not installed
            AuthenticationError: If API key is missing or invalid
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI not available. Install with: uv add openai"
            )
        
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise AuthenticationError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter.",
                provider="openai"
            )
        
        self.model_name = model
        
        # Initialize OpenAI client
        try:
            self.client = OpenAI(api_key=self.api_key)
        except Exception as e:
            raise handle_provider_exception(e, "openai")
        
        # Model capabilities and supported languages
        self._supported_languages = [
            "en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh", 
            "ar", "hi", "tr", "pl", "nl", "sv", "da", "no", "fi", "he",
            "th", "vi", "id", "ms", "tl", "uk", "bg", "hr", "cs", "et",
            "lv", "lt", "sk", "sl", "mt", "is", "mk", "sq", "az", "hy",
            "ka", "be", "kk", "ky", "uz", "tg", "mn", "ne", "si", "my",
            "km", "lo", "bn", "te", "ta", "ml", "kn", "gu", "pa", "ur",
            "fa", "ps", "sw", "ha", "yo", "ig", "zu", "af", "am", "rw"
        ]
    
    
    async def transcribe(
        self,
        audio_path: Path,
        language: str | None = None,
        context: str | None = None,
        **kwargs: Any
    ) -> TranscriptionResult:
        """Transcribe audio using OpenAI Whisper.
        
        Args:
            audio_path: Path to audio file
            language: Language code (ISO 639-1) or None for auto-detection
            context: Additional context to improve transcription accuracy
            **kwargs: Additional parameters (temperature, etc.)
            
        Returns:
            TranscriptionResult with segments and metadata
            
        Raises:
            ProcessingError: If transcription fails
            AuthenticationError: If API key is invalid
        """
        logger.info(f"Starting OpenAI transcription: {audio_path}")
        start_time = time.time()
        
        # Validate audio file exists
        if not audio_path.exists():
            raise ProcessingError(
                f"Audio file not found: {audio_path}",
                file_path=str(audio_path)
            )
        
        # Prepare transcription parameters
        params = self._prepare_transcription_params(language, context, **kwargs)
        logger.debug(f"OpenAI transcription parameters: {params}")
        
        try:
            # Perform transcription
            response = await self._call_openai_api(audio_path, params)
            
            # Process response into segments
            segments = self._parse_response(response, params)
            
            # Calculate processing metrics
            processing_time = time.time() - start_time
            logger.info(f"OpenAI transcription completed in {processing_time:.2f}s")
            
            # Build metadata
            metadata = {
                "provider": "openai",
                "model": self.model_name,
                "processing_time": processing_time,
                "response_format": params.get("response_format", "json"),
                "language_detected": getattr(response, 'language', language),
                "timestamp_granularities": params.get("timestamp_granularities", [])
            }
            
            # Estimate confidence score
            confidence = self._estimate_confidence(response, segments)
            
            return TranscriptionResult(
                segments=segments,
                metadata=metadata,
                provider="openai",
                language=getattr(response, 'language', language),
                confidence=confidence
            )
            
        except openai.AuthenticationError as e:
            logger.error(f"OpenAI authentication failed: {e}")
            raise AuthenticationError(str(e), provider="openai") from e
        except openai.RateLimitError as e:
            logger.error(f"OpenAI rate limit exceeded: {e}")
            raise APIError(f"Rate limit exceeded: {e}", provider="openai") from e
        except openai.APIStatusError as e:
            logger.error(f"OpenAI API error: {e}")
            raise APIError(f"API error: {e}", provider="openai") from e
        except Exception as e:
            logger.error(f"OpenAI transcription failed: {e}")
            raise ProcessingError(f"Transcription failed: {e}", file_path=str(audio_path)) from e
    
    def _prepare_transcription_params(
        self, 
        language: str | None, 
        context: str | None,
        **kwargs: Any
    ) -> dict[str, Any]:
        """Prepare parameters for OpenAI Whisper API call.
        
        Args:
            language: Target language code
            context: Additional context for transcription
            **kwargs: Additional parameters
            
        Returns:
            Dictionary of API parameters
        """
        params = {
            "model": self.model_name,
            "response_format": "verbose_json",  # Get detailed response with timestamps
            "timestamp_granularities": ["word", "segment"]
        }
        
        # Language parameter (if specified and supported)
        if language and language != "auto" and language in self._supported_languages:
            params["language"] = language
        
        # Temperature for consistency (0.0 = deterministic)
        temperature = kwargs.get("temperature", 0.0)
        if 0.0 <= temperature <= 1.0:
            params["temperature"] = temperature
        
        # Context-aware prompting
        if context:
            prompt = self._build_context_prompt(context, language)
            if prompt:
                # OpenAI has a 224-token limit for prompts
                params["prompt"] = prompt[:224]
        
        return params
    
    def _build_context_prompt(self, context: str, language: str | None) -> str:
        """Build context-aware prompt for OpenAI Whisper.
        
        Args:
            context: Context information
            language: Target language
            
        Returns:
            Optimized prompt string
        """
        try:
            # Handle empty or None context
            if not context:
                return ""
            
            # Use VTTiro's prompt utilities
            context_dict = {"topic": context}
            
            base_prompt = build_webvtt_prompt(
                language=language,
                context=context_dict
            )
            
            # Optimize for OpenAI's prompt format
            optimized = optimize_prompt_for_provider(base_prompt, "openai")
            
            # Keep within OpenAI's token limit (approximately 224 tokens ≈ 896 chars)
            if len(optimized) > 896:
                optimized = optimized[:896].rsplit(' ', 1)[0]  # Cut at word boundary
            
            return optimized
            
        except Exception as e:
            logger.warning(f"Failed to build context prompt: {e}")
            return ""
    
    async def _call_openai_api(self, audio_path: Path, params: dict[str, Any]) -> Any:
        """Call OpenAI Whisper API with the audio file.
        
        Args:
            audio_path: Path to audio file
            params: API parameters
            
        Returns:
            OpenAI API response object
        """
        try:
            with open(audio_path, "rb") as audio_file:
                # Use asyncio.to_thread for I/O bound operation
                response = await asyncio.to_thread(
                    self.client.audio.transcriptions.create,
                    file=audio_file,
                    **params
                )
            return response
            
        except Exception as e:
            logger.error(f"OpenAI API call failed for {audio_path}: {e}")
            raise
    
    def _parse_response(self, response: Any, params: dict[str, Any]) -> list[TranscriptSegment]:
        """Parse OpenAI response into TranscriptSegment objects.
        
        Args:
            response: OpenAI API response
            params: Original request parameters
            
        Returns:
            List of TranscriptSegment objects
        """
        segments = []
        response_format = params.get("response_format", "json")
        
        try:
            if response_format == "verbose_json" and hasattr(response, 'segments'):
                # Parse segments from verbose JSON response
                for segment in response.segments:
                    # Extract basic segment info
                    text = getattr(segment, 'text', '').strip() if hasattr(segment, 'text') else ""
                    if not text:
                        continue
                    
                    start_time = float(getattr(segment, 'start', 0.0))
                    end_time = float(getattr(segment, 'end', start_time + 1.0))
                    
                    # Create segment (convert avg_logprob to confidence if available)
                    avg_logprob = getattr(segment, 'avg_logprob', None)
                    confidence = None
                    if avg_logprob is not None:
                        # Convert log probability to confidence score (0.0-1.0)
                        # Whisper log probs typically range from -3.0 to 0.0
                        confidence = max(0.0, min(1.0, (avg_logprob + 3.0) / 3.0))
                    
                    transcript_segment = TranscriptSegment(
                        start=start_time,
                        end=end_time,
                        text=text,
                        confidence=confidence
                    )
                    segments.append(transcript_segment)
                    
            elif hasattr(response, 'text'):
                # Fallback: single segment from basic response
                text = response.text.strip()
                if text:
                    segments.append(TranscriptSegment(
                        start=0.0,
                        end=1.0,  # Will need duration from audio file for accurate timing
                        text=text,
                        confidence=0.9
                    ))
            
            logger.debug(f"Parsed {len(segments)} segments from OpenAI response")
            return segments
            
        except Exception as e:
            logger.error(f"Failed to parse OpenAI response: {e}")
            return []
    
    def _estimate_confidence(self, response: Any, segments: list[TranscriptSegment]) -> float:
        """Estimate overall confidence score from OpenAI response.
        
        Args:
            response: OpenAI API response
            segments: Parsed transcript segments
            
        Returns:
            Overall confidence score (0.0-1.0)
        """
        try:
            if hasattr(response, 'segments') and response.segments:
                # Calculate average log probability from segments
                logprobs = []
                for segment in response.segments:
                    if hasattr(segment, 'avg_logprob') and segment.avg_logprob is not None:
                        logprobs.append(segment.avg_logprob)
                
                if logprobs:
                    avg_logprob = sum(logprobs) / len(logprobs)
                    # Convert log probability to confidence (approximate)
                    # Whisper log probs typically range from -3.0 to 0.0
                    confidence = max(0.0, min(1.0, (avg_logprob + 3.0) / 3.0))
                    return confidence
            
            # Fallback: high confidence for Whisper
            return 0.92
            
        except Exception as e:
            logger.warning(f"Failed to estimate confidence: {e}")
            return 0.90
    
    
    def estimate_cost(self, duration_seconds: float) -> float:
        """Estimate OpenAI Whisper transcription cost in USD.
        
        Args:
            duration_seconds: Audio duration in seconds
            
        Returns:
            Estimated cost in USD
            
        Raises:
            ValueError: If duration is invalid
        """
        if duration_seconds <= 0:
            raise ValueError("Duration must be positive")
        
        # OpenAI Whisper pricing: $0.006 per minute (as of 2024)
        minutes = duration_seconds / 60.0
        cost_per_minute = 0.006
        return minutes * cost_per_minute
    
    @property
    def provider_name(self) -> str:
        """Return provider name for identification."""
        return "openai"
    
    @property
    def supports_speaker_diarization(self) -> bool:
        """Return whether speaker diarization is supported."""
        return False  # Whisper doesn't include speaker diarization
    
    @property
    def supported_languages(self) -> list[str]:
        """Return list of supported language codes."""
        return self._supported_languages.copy()
    
    @property
    def model_info(self) -> dict[str, Any]:
        """Return information about the current model."""
        return {
            "name": self.model_name,
            "provider": "openai",
            "supports_streaming": False,
            "supports_word_timestamps": True,
            "supports_speaker_diarization": False,
            "max_file_size_mb": 25,  # OpenAI's limit
            "supported_formats": [
                "mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"
            ]
        }