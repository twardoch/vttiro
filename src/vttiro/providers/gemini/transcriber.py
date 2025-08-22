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

from ...core.errors import (
    APIError,
    AuthenticationError,
    ContentFilterError,
    ProcessingError,
    handle_provider_exception,
    create_debug_context,
)
from ...core.types import TranscriptionResult, TranscriptSegment
from ...utils.prompt import build_webvtt_prompt, optimize_prompt_for_provider
from ...utils.timestamp import parse_webvtt_timestamp_line, distribute_words_over_duration
from ...utils.api_keys import get_api_key_with_fallbacks
# Removed complex type validation - using simple validation in base class
from ..base import TranscriberABC

try:
    from loguru import logger
except ImportError:
    import logging as logger

# Optional dependency handling
try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("Google GenerativeAI not available. Install with: uv add google-generativeai")
    # Create dummy objects to avoid NameError
    genai = None
    HarmCategory = object()
    HarmBlockThreshold = object()


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
        api_key: str | None = None,
        model: str = "gemini-2.0-flash",
        safety_settings: dict[str, str] | None = None
    ):
        """Initialize Gemini transcriber.
        
        Args:
            api_key: Gemini API key (or set GEMINI_API_KEY env var)
            model: Gemini model to use
            safety_settings: Custom safety settings override
            
        Raises:
            ImportError: If google-generativeai is not installed
            AuthenticationError: If API key is missing or invalid
        """
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "Google GenerativeAI not available. Install with: uv add google-generativeai"
            )
        
        # Get API key from parameter or environment with fallbacks
        self.api_key = get_api_key_with_fallbacks("gemini", api_key)
        if not self.api_key:
            raise AuthenticationError(
                "Gemini API key not provided. Set one of: VTTIRO_GEMINI_API_KEY, "
                "GEMINI_API_KEY, GOOGLE_API_KEY environment variables "
                "or pass api_key parameter.",
                provider="gemini"
            )
        
        self.model_name = model
        
        # Configure Gemini API
        genai.configure(api_key=self.api_key)
        
        # Set up safety settings
        self._safety_settings = self._configure_safety_settings(safety_settings)
        
        # Initialize model with optimal settings for transcription
        try:
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config={
                    "temperature": 0.1,  # Low temperature for consistent transcription
                    "top_p": 0.8,
                    "top_k": 40,
                    "max_output_tokens": 8192,
                },
                safety_settings=self._safety_settings
            )
        except Exception as e:
            raise handle_provider_exception(e, "gemini")
        
        # Model capabilities
        self._supported_languages = [
            "en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh", 
            "ar", "hi", "bn", "te", "mr", "ta", "ur", "gu", "kn", "ml",
            "pa", "ne", "si", "my", "th", "vi", "id", "ms", "tl", "sw"
        ]
    
    def _configure_safety_settings(
        self, 
        custom_settings: dict[str, str] | None = None
    ) -> dict[HarmCategory, HarmBlockThreshold]:
        """Configure safety settings for Gemini transcription.
        
        For transcription use cases, defaults to minimal blocking since:
        1. Audio content is typically legitimate speech/interviews
        2. False positives can block valid transcription tasks
        3. Users can override via environment variables if needed
        
        Args:
            custom_settings: Custom safety settings override
            
        Returns:
            Safety settings dictionary for Gemini
        """
        # Default to minimal blocking for transcription
        defaults = {
            "harassment": "none",
            "hate_speech": "none", 
            "sexually_explicit": "none",
            "dangerous_content": "none",
        }
        
        # Apply custom settings if provided
        if custom_settings:
            defaults.update(custom_settings)
        
        # Check for environment variable overrides
        env_overrides = {
            "harassment": os.getenv("GEMINI_SAFETY_HARASSMENT", defaults["harassment"]),
            "hate_speech": os.getenv("GEMINI_SAFETY_HATE_SPEECH", defaults["hate_speech"]),
            "sexually_explicit": os.getenv("GEMINI_SAFETY_SEXUALLY_EXPLICIT", defaults["sexually_explicit"]),
            "dangerous_content": os.getenv("GEMINI_SAFETY_DANGEROUS_CONTENT", defaults["dangerous_content"]),
        }
        
        # Map string values to HarmBlockThreshold enums
        threshold_map = {
            "none": HarmBlockThreshold.BLOCK_NONE,
            "low": HarmBlockThreshold.BLOCK_ONLY_HIGH,
            "medium": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            "high": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        }
        
        category_map = {
            "harassment": HarmCategory.HARM_CATEGORY_HARASSMENT,
            "hate_speech": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            "sexually_explicit": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            "dangerous_content": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        }
        
        # Build final safety settings
        safety_settings = {}
        for setting_name, threshold_str in env_overrides.items():
            threshold_str = threshold_str.lower()
            if threshold_str in threshold_map and setting_name in category_map:
                safety_settings[category_map[setting_name]] = threshold_map[threshold_str]
                logger.debug(f"Safety setting for {setting_name}: {threshold_str}")
            else:
                logger.warning(f"Invalid safety threshold '{threshold_str}' for {setting_name}")
                safety_settings[category_map[setting_name]] = HarmBlockThreshold.BLOCK_NONE
        
        return safety_settings
    
    async def transcribe(
        self,
        audio_path: Path,
        language: str | None = None,
        context: str | None = None,
        **kwargs: Any
    ) -> TranscriptionResult:
        """Transcribe audio using Gemini 2.0 Flash.
        
        Args:
            audio_path: Path to audio file
            language: Target language code (ISO 639-1)
            context: Additional context for improved accuracy
            **kwargs: Additional provider-specific parameters
            
        Returns:
            TranscriptionResult with segments and metadata
            
        Raises:
            ProcessingError: If audio file processing fails
            APIError: If Gemini API call fails
            ContentFilterError: If content is blocked by safety filters
            AuthenticationError: If API authentication fails
        """
        start_time = time.time()
        
        logger.info(f"Starting Gemini transcription: {audio_path}")
        
        try:
            # Validate audio file exists (basic check first)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # Validate audio file format
            self.validate_audio_file(audio_path)
            
            # Upload audio file to Gemini
            audio_file = await self._upload_audio_file(audio_path)
            
            # Generate context-aware prompt
            prompt = self._generate_transcription_prompt(language, context, **kwargs)
            
            # Perform transcription
            response = await self._transcribe_with_gemini(audio_file, prompt)
            
            # Handle safety filter blocking
            self._check_safety_filters(response)
            
            # Extract WebVTT content from response
            webvtt_content = self._extract_response_text(response)
            
            processing_time = time.time() - start_time
            
            # Log response for debugging
            self._log_response_debug(webvtt_content)
            
            # Parse WebVTT content to extract segments
            segments = self._parse_webvtt_response(webvtt_content)
            
            # Calculate overall confidence
            confidence = self._estimate_confidence(response, segments)
            
            logger.info(f"Gemini transcription completed in {processing_time:.2f}s")
            
            return TranscriptionResult(
                segments=segments,
                metadata={
                    "provider": "gemini",
                    "model": self.model_name,
                    "language": language or "auto",
                    "context_used": bool(context),
                    "processing_time": processing_time,
                    "safety_ratings": getattr(response, 'safety_ratings', []),
                    "finish_reason": self._get_finish_reason(response),
                    "webvtt_content": webvtt_content,
                    "native_timing": True,
                    **kwargs
                },
                provider="gemini",
                language=language,
                confidence=confidence
            )
            
        except Exception as e:
            if isinstance(e, (APIError, ContentFilterError, AuthenticationError, ProcessingError, FileNotFoundError)):
                raise
            else:
                raise handle_provider_exception(e, "gemini")
    
    def estimate_cost(self, duration_seconds: float) -> float:
        """Estimate Gemini transcription cost in USD.
        
        Args:
            duration_seconds: Audio duration in seconds
            
        Returns:
            Estimated cost in USD
            
        Raises:
            ValueError: If duration is invalid
        """
        if duration_seconds <= 0:
            raise ValueError("Duration must be positive")
        
        # Gemini 2.0 Flash pricing (estimated based on current rates)
        # Audio processing cost approximately $1.20 per hour
        hours = duration_seconds / 3600.0
        cost_per_hour = 1.20
        return hours * cost_per_hour
    
    @property
    def provider_name(self) -> str:
        """Return provider name for identification."""
        return "gemini"
    
    @property
    def supports_speaker_diarization(self) -> bool:
        """Return whether speaker diarization is supported."""
        return True
    
    @property
    def supported_languages(self) -> list[str]:
        """Return list of supported language codes."""
        return self._supported_languages.copy()
    
    async def _upload_audio_file(self, audio_path: Path) -> Any:
        """Upload audio file to Gemini for processing.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Gemini file object
            
        Raises:
            ProcessingError: If file upload fails
        """
        try:
            logger.debug(f"Uploading audio file to Gemini: {audio_path}")
            
            # Use asyncio.to_thread for the blocking Gemini upload call
            audio_file = await asyncio.to_thread(
                genai.upload_file, 
                path=str(audio_path)
            )
            
            logger.debug(f"Audio file uploaded successfully: {audio_file.display_name}")
            return audio_file
            
        except Exception as e:
            raise ProcessingError(
                f"Failed to upload audio file to Gemini: {e}",
                file_path=str(audio_path)
            )
    
    def _generate_transcription_prompt(
        self, 
        language: str | None, 
        context: str | None,
        **kwargs: Any
    ) -> str:
        """Generate optimized prompt for Gemini transcription.
        
        Args:
            language: Target language code
            context: Additional context
            **kwargs: Additional prompt parameters
            
        Returns:
            Optimized prompt string
        """
        logger.debug("Generating WebVTT prompt for Gemini transcription")
        
        # Extract prompt-relevant kwargs
        context_dict = {}
        if context:
            context_dict["topic"] = context
        
        # Build base WebVTT prompt
        base_prompt = build_webvtt_prompt(
            language=language,
            context=context_dict,
            include_speaker_diarization=kwargs.get("include_speaker_diarization", True),
            include_emotions=kwargs.get("include_emotions", False),
            max_segment_duration=kwargs.get("max_segment_duration", 30.0)
        )
        
        # Add Gemini-specific optimizations
        enhanced_prompt = optimize_prompt_for_provider(base_prompt, "gemini")
        
        logger.debug(f"Generated prompt ({len(enhanced_prompt)} chars) for Gemini")
        return enhanced_prompt
    
    async def _transcribe_with_gemini(self, audio_file: Any, prompt: str) -> Any:
        """Perform transcription using Gemini API.
        
        Args:
            audio_file: Uploaded Gemini file object
            prompt: Transcription prompt
            
        Returns:
            Gemini response object
            
        Raises:
            APIError: If API call fails
        """
        try:
            # Create content parts with prompt and audio file
            content_parts = [prompt, audio_file]
            
            # Generate content using the model
            response = await asyncio.to_thread(
                self.model.generate_content,
                content_parts
            )
            
            return response
            
        except Exception as e:
            # Create debugging context for better error reporting
            debug_context = create_debug_context(
                operation="generate_content_with_audio",
                provider="gemini",
                file_path=str(audio_path) if audio_path else None,
                model=self.model,
                language=language
            )
            
            # Check for specific error types with enhanced messages
            if hasattr(e, 'status_code'):
                status_code = getattr(e, 'status_code')
                if status_code == 401:
                    raise AuthenticationError(
                        "Gemini API authentication failed. Your API key may be invalid, expired, or missing.",
                        provider="gemini",
                        details={**debug_context, "status_code": status_code}
                    )
                elif status_code == 403:
                    raise AuthenticationError(
                        "Gemini API access forbidden. Check if your API key has the required permissions.",
                        provider="gemini",
                        details={**debug_context, "status_code": status_code}
                    )
                elif status_code == 429:
                    raise APIError(
                        "Gemini rate limit exceeded. You've made too many requests too quickly.",
                        provider="gemini",
                        status_code=status_code,
                        details={**debug_context, "retry_suggested": True}
                    )
                elif status_code >= 500:
                    raise APIError(
                        f"Gemini server error (HTTP {status_code}). This is not your fault - Google's servers are having issues.",
                        provider="gemini",
                        status_code=status_code,
                        details={**debug_context, "server_error": True}
                    )
            
            # Check for specific error messages in exception text
            error_str = str(e).lower()
            if "quota" in error_str or "limit" in error_str:
                raise APIError(
                    "Gemini API quota exceeded. You've reached your usage limits for this time period.",
                    provider="gemini",
                    details={**debug_context, "quota_exceeded": True}
                )
            elif "safety" in error_str or "blocked" in error_str:
                raise ContentFilterError(
                    "Content blocked by Gemini's safety filters. The audio may contain sensitive material.",
                    provider="gemini",
                    details={**debug_context, "safety_blocked": True}
                )
            elif "file" in error_str and "format" in error_str:
                raise ProcessingError(
                    f"Gemini doesn't support this audio format. Try converting to WAV, MP3, or M4A.",
                    file_path=str(audio_path) if audio_path else None,
                    details={**debug_context, "format_error": True}
                )
            
            # Generic API error with comprehensive debugging info
            raise APIError(
                f"Gemini API call failed unexpectedly: {e}",
                provider="gemini",
                details={**debug_context, "original_error": str(e), "exception_type": type(e).__name__}
            )
    
    def _check_safety_filters(self, response: Any) -> None:
        """Check if response was blocked by safety filters.
        
        Args:
            response: Gemini response object
            
        Raises:
            ContentFilterError: If content was blocked
        """
        if not hasattr(response, 'candidates') or not response.candidates:
            # No candidates could indicate safety blocking
            raise ContentFilterError(
                "Gemini returned no response candidates. This usually means the audio content "
                "was blocked by safety filters before processing could begin.",
                provider="gemini", 
                details=create_debug_context(
                    operation="safety_filter_check",
                    provider="gemini",
                    no_candidates=True,
                    response_type=type(response).__name__
                )
            )
        
        candidate = response.candidates[0]
        finish_reason = getattr(candidate, 'finish_reason', None)
        
        if finish_reason == 2:  # SAFETY filter blocking
            # Extract blocked categories with enhanced details
            safety_ratings = getattr(response, 'safety_ratings', [])
            blocked_categories = []
            safety_details = {"finish_reason": finish_reason}
            
            for rating in safety_ratings:
                if hasattr(rating, 'blocked') and rating.blocked:
                    category_name = getattr(rating, 'category', 'unknown')
                    blocked_categories.append(str(category_name))
                    
                    # Add probability info for debugging
                    probability = getattr(rating, 'probability', None)
                    if probability is not None:
                        safety_details[f'{category_name}_probability'] = str(probability)
            
            # Create detailed error message with specific guidance
            if blocked_categories:
                categories_str = ', '.join(blocked_categories)
                message = (
                    f"Gemini safety filters blocked transcription due to detected content: {categories_str}. "
                    f"This typically indicates the audio contains violence, hate speech, harassment, "
                    f"or other potentially harmful content. Consider using OpenAI or AssemblyAI instead, "
                    f"as they have different content policies."
                )
            else:
                message = (
                    f"Gemini safety filters blocked transcription for unspecified safety reasons. "
                    f"The audio may contain sensitive content. Try using a different provider "
                    f"(OpenAI or AssemblyAI) which may have more permissive policies."
                )
            
            raise ContentFilterError(
                message,
                provider="gemini",
                blocked_categories=blocked_categories,
                details={**safety_details, **create_debug_context(
                    operation="safety_filter_blocking",
                    provider="gemini"
                )}
            )
        
        elif finish_reason is not None and finish_reason != 1:  # 1 = STOP (normal)
            # Map finish reasons to human-readable explanations
            finish_reason_map = {
                0: "FINISH_REASON_UNSPECIFIED",
                1: "STOP (normal completion)",
                2: "SAFETY (content blocked)",
                3: "RECITATION (copyright/citation issues)",
                4: "OTHER"
            }
            
            reason_desc = finish_reason_map.get(finish_reason, f"Unknown reason ({finish_reason})")
            
            if finish_reason == 3:  # RECITATION
                error_msg = (
                    f"Gemini blocked transcription due to potential copyright or citation issues (RECITATION). "
                    f"The audio may contain copyrighted material. Try using a different provider."
                )
            else:
                error_msg = (
                    f"Gemini transcription failed with unexpected finish reason: {reason_desc}. "
                    f"This indicates an unusual response from the API."
                )
                
            raise APIError(
                error_msg,
                provider="gemini",
                details=create_debug_context(
                    operation="unexpected_finish_reason",
                    provider="gemini",
                    finish_reason=finish_reason,
                    finish_reason_desc=reason_desc
                )
            )
    
    def _extract_response_text(self, response: Any) -> str:
        """Extract text content from Gemini response.
        
        Args:
            response: Gemini response object
            
        Returns:
            Response text content
            
        Raises:
            APIError: If text extraction fails
        """
        try:
            return response.text.strip()
        except Exception as e:
            raise APIError(
                f"Failed to extract text from Gemini response: {e}",
                provider="gemini"
            )
    
    def _log_response_debug(self, webvtt_content: str) -> None:
        """Log response content for debugging.
        
        Args:
            webvtt_content: WebVTT content to log
        """
        try:
            # Simple debug logging without level checking
            if webvtt_content:
                content_preview = webvtt_content[:500]
                if len(webvtt_content) > 500:
                    content_preview += "..."
                logger.debug(f"Gemini WebVTT response ({len(webvtt_content)} chars):\n{content_preview}")
            else:
                logger.warning("Gemini returned empty response!")
        except Exception:
            # Ignore logging errors
            pass
    
    def _parse_webvtt_response(self, webvtt_content: str) -> list[TranscriptSegment]:
        """Parse WebVTT content into TranscriptSegment objects.
        
        Args:
            webvtt_content: Raw WebVTT content from Gemini
            
        Returns:
            List of TranscriptSegment objects
        """
        import re
        
        logger.debug("Parsing WebVTT response from Gemini")
        
        segments = []
        lines = webvtt_content.strip().split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip WEBVTT header and empty lines
            if line.startswith('WEBVTT') or not line:
                i += 1
                continue
            
            # Check for timestamp line
            timestamp_result = parse_webvtt_timestamp_line(line)
            if timestamp_result:
                start_time, end_time = timestamp_result
                
                # Collect cue text from following lines
                cue_text_parts = []
                i += 1
                
                while i < len(lines) and lines[i].strip():
                    cue_line = lines[i].strip()
                    
                    # Handle speaker tags <v Speaker>
                    speaker_match = re.match(r'<v\s+([^>]+)>(.*)', cue_line)
                    if speaker_match:
                        speaker_name, speaker_text = speaker_match.groups()
                        cue_text_parts.append(speaker_text.strip())
                        # TODO: Store speaker information when we support it
                    else:
                        cue_text_parts.append(cue_line)
                    
                    i += 1
                
                # Join cue text
                cue_text = ' '.join(cue_text_parts).strip()
                
                if cue_text:
                    try:
                        segment = TranscriptSegment(
                            start=start_time,
                            end=end_time,
                            text=cue_text,
                            confidence=0.95  # High confidence for Gemini WebVTT
                        )
                        segments.append(segment)
                    except ValueError as e:
                        logger.warning(f"Invalid segment data: {e}")
                        continue
            else:
                i += 1
        
        logger.debug(f"Parsed {len(segments)} segments from WebVTT")
        
        if not segments:
            logger.warning("No segments extracted from WebVTT response")
            logger.debug(f"Original WebVTT content:\n{webvtt_content}")
        
        return segments
    
    def _estimate_confidence(self, response: Any, segments: list[TranscriptSegment]) -> float:
        """Estimate overall confidence score.
        
        Args:
            response: Gemini response object
            segments: Parsed transcript segments
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        base_confidence = 0.90  # High baseline for Gemini
        
        # Adjust based on response characteristics
        total_text = ' '.join(segment.text for segment in segments)
        
        if len(total_text) < 10:
            base_confidence -= 0.10  # Very short responses might be incomplete
        elif any(char in total_text for char in ["[", "]", "(", ")"]):
            base_confidence -= 0.05  # Bracketed text might indicate uncertainty
        
        # Check finish reason
        finish_reason = self._get_finish_reason(response)
        if finish_reason != "complete":
            base_confidence -= 0.15
        
        return max(0.0, min(1.0, base_confidence))
    
    def _get_finish_reason(self, response: Any) -> str:
        """Extract finish reason from response.
        
        Args:
            response: Gemini response object
            
        Returns:
            Finish reason string
        """
        try:
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                finish_reason = getattr(candidate, 'finish_reason', 1)
                return "complete" if finish_reason == 1 else str(finish_reason)
            return "complete"
        except Exception:
            return "unknown"