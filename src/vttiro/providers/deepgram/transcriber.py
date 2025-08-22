# this_file: src/vttiro/providers/deepgram/transcriber.py
"""Deepgram Nova-3 transcription provider.

This module implements the Deepgram transcription provider using the new
VTTiro 2.0 architecture. Provides fast, accurate transcription with
Nova-3 model, speaker diarization, and advanced features.

Used by:
- Core orchestration for Deepgram-based transcription
- Provider selection logic
- Testing infrastructure for Deepgram functionality
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
from ...utils.timestamp import distribute_words_over_duration
from ...utils.type_validation import type_validated, validate_audio_path, validate_language_code
from ..base import TranscriberABC

try:
    from loguru import logger
except ImportError:
    import logging as logger

# Optional dependency handling
try:
    from deepgram import DeepgramClient, PrerecordedOptions
    DEEPGRAM_AVAILABLE = True
except ImportError:
    DEEPGRAM_AVAILABLE = False
    logger.warning("Deepgram SDK not available. Install with: uv add deepgram-sdk")
    DeepgramClient = None
    PrerecordedOptions = None


class DeepgramTranscriber(TranscriberABC):
    """Deepgram Nova-3 transcription provider.
    
    Implements fast, accurate transcription using Deepgram's Nova-3 model
    with speaker diarization, word-level timestamps, and advanced features
    for high-speed multilingual transcription.
    
    Features:
    - Nova-3 model for optimal speed/accuracy balance
    - Speaker diarization and identification  
    - Word-level timestamps with confidence scores
    - Keyword boosting and context awareness
    - Smart formatting and punctuation
    - Multi-language support with auto-detection
    """
    
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "nova-3",
        enable_speaker_diarization: bool = True,
        tier: str = "nova"
    ):
        """Initialize Deepgram transcriber.
        
        Args:
            api_key: Deepgram API key (or set DEEPGRAM_API_KEY env var)
            model: Deepgram model to use (nova-3, nova-2, enhanced, base, whisper-cloud)
            enable_speaker_diarization: Whether to enable speaker identification
            tier: Deepgram tier to use (nova, enhanced, base)
            
        Raises:
            ImportError: If deepgram-sdk package is not installed
            AuthenticationError: If API key is missing or invalid
        """
        if not DEEPGRAM_AVAILABLE:
            raise ImportError(
                "Deepgram SDK not available. Install with: uv add deepgram-sdk"
            )
        
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv("DEEPGRAM_API_KEY")
        if not self.api_key:
            raise AuthenticationError(
                "Deepgram API key not provided. Set DEEPGRAM_API_KEY environment variable "
                "or pass api_key parameter.",
                provider="deepgram"
            )
        
        self.model_name = model
        self.enable_speaker_diarization = enable_speaker_diarization
        self.tier = tier
        
        # Initialize Deepgram client
        try:
            self.client = DeepgramClient(self.api_key)
        except Exception as e:
            raise handle_provider_exception(e, "deepgram")
        
        # Model capabilities and supported languages
        self._supported_languages = [
            "en", "en-US", "en-GB", "en-AU", "en-NZ", "en-IN",
            "es", "es-ES", "es-419", "fr", "fr-FR", "fr-CA", 
            "de", "de-DE", "it", "it-IT", "pt", "pt-BR", "pt-PT",
            "pl", "pl-PL", "nl", "nl-NL", "tr", "tr-TR", "ru", "ru-RU",
            "ar", "ar-SA", "zh", "zh-CN", "zh-TW", "ja", "ja-JP", 
            "ko", "ko-KR", "hi", "hi-IN", "th", "th-TH", "vi", "vi-VN",
            "uk", "uk-UA", "cs", "cs-CZ", "da", "da-DK", "fi", "fi-FI",
            "no", "no-NO", "sv", "sv-SE", "he", "he-IL", "id", "id-ID",
            "ms", "ms-MY", "ro", "ro-RO", "sk", "sk-SK", "bg", "bg-BG"
        ]
    
    @type_validated
    async def transcribe(
        self,
        audio_path: Path,
        language: str | None = None,
        context: str | None = None,
        **kwargs: Any
    ) -> TranscriptionResult:
        """Transcribe audio using Deepgram Nova-3.
        
        Args:
            audio_path: Path to audio file
            language: Language code (ISO 639-1) or None for auto-detection
            context: Additional context to improve transcription accuracy
            **kwargs: Additional parameters (diarize, keywords, etc.)
            
        Returns:
            TranscriptionResult with segments and metadata
            
        Raises:
            ProcessingError: If transcription fails
            AuthenticationError: If API key is invalid
        """
        logger.info(f"Starting Deepgram transcription: {audio_path}")
        start_time = time.time()
        
        # Validate audio file exists
        if not audio_path.exists():
            raise ProcessingError(
                f"Audio file not found: {audio_path}",
                file_path=str(audio_path)
            )
        
        # Prepare transcription options
        options = self._prepare_transcription_options(language, context, **kwargs)
        logger.debug(f"Deepgram transcription options: {options}")
        
        try:
            # Perform transcription
            response = await self._call_deepgram_api(audio_path, options)
            
            # Process response into segments
            segments = self._parse_response(response)
            
            # Calculate processing metrics
            processing_time = time.time() - start_time
            logger.info(f"Deepgram transcription completed in {processing_time:.2f}s")
            
            # Extract response data
            result = response.results if hasattr(response, 'results') else None
            alternative = None
            if result and result.channels and result.channels[0].alternatives:
                alternative = result.channels[0].alternatives[0]
            
            # Build metadata
            metadata = {
                "provider": "deepgram",
                "model": self.model_name,
                "tier": self.tier,
                "processing_time": processing_time,
                "request_id": getattr(response, 'request_id', None),
                "model_info": getattr(result, 'model_info', {}) if result else {},
                "channels": len(result.channels) if result and result.channels else 0,
                "alternatives": len(result.channels[0].alternatives) if result and result.channels and result.channels[0].alternatives else 0,
                "language_detected": self._detect_language(result, language),
                "diarization_enabled": options.get("diarize", False),
                "keywords_used": len(options.get("keywords", [])),
                "smart_format": options.get("smart_format", False)
            }
            
            # Estimate confidence score
            confidence = self._estimate_confidence(alternative, segments)
            
            return TranscriptionResult(
                segments=segments,
                metadata=metadata,
                provider="deepgram",
                language=self._detect_language(result, language),
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Deepgram transcription failed: {e}")
            raise ProcessingError(f"Transcription failed: {e}", file_path=str(audio_path)) from e
    
    def _prepare_transcription_options(
        self, 
        language: str | None, 
        context: str | None,
        **kwargs: Any
    ) -> dict[str, Any]:
        """Prepare options for Deepgram transcription.
        
        Args:
            language: Target language code
            context: Additional context for transcription
            **kwargs: Additional configuration parameters
            
        Returns:
            Dictionary of API configuration parameters
        """
        # Build base configuration for optimal performance
        options = {
            # Model selection
            "model": self.model_name,
            "version": "latest",
            "tier": self.tier,
            
            # Language settings
            "language": self._normalize_language_code(language) if language and language != "auto" else None,
            "detect_language": language is None or language == "auto",
            
            # Accuracy features
            "punctuate": True,
            "profanity_filter": False,  # Preserve original content
            "redact": [],  # Don't redact anything for accuracy
            
            # Timing and structure
            "paragraphs": True,
            "utterances": True,
            "utt_split": 0.8,  # Split utterances at 0.8s pauses
            
            # Advanced features
            "diarize": kwargs.get("diarize", self.enable_speaker_diarization),
            "multichannel": False,  # Audio is mono from processing
            "alternatives": 1,  # Just the best alternative for speed
            "numerals": True,  # Convert numbers to numerals
            "search": kwargs.get("search", []),
            "replace": kwargs.get("replace", []),
            "keywords": [],  # Will be populated from context
            
            # Performance optimizations
            "smart_format": True,  # Apply intelligent formatting
            "filler_words": False,  # Remove filler words for cleaner output
            "encoding": "linear16",  # Optimal encoding
            "sample_rate": 16000,  # Standard sample rate
        }
        
        # Add context-specific enhancements
        if context:
            # Build keywords from context
            keywords = self._build_keywords(context)
            if keywords:
                options["keywords"] = keywords
                options["keyword_boost"] = "high"
                logger.debug(f"Using Deepgram keywords: {keywords}")
        
        # Remove None values for clean API call
        return {k: v for k, v in options.items() if v is not None}
    
    def _normalize_language_code(self, language: str) -> str:
        """Normalize language code to Deepgram format.
        
        Args:
            language: Input language code
            
        Returns:
            Normalized language code
        """
        # Map common language codes to Deepgram specific codes
        language_mapping = {
            "en": "en-US",
            "es": "es",
            "fr": "fr",
            "de": "de",
            "it": "it",
            "pt": "pt",
            "pl": "pl",
            "nl": "nl",
            "tr": "tr",
            "ru": "ru",
            "ar": "ar",
            "zh": "zh-CN",
            "ja": "ja",
            "ko": "ko",
            "hi": "hi-IN",
            "th": "th",
            "vi": "vi"
        }
        
        return language_mapping.get(language, language)
    
    def _build_keywords(self, context: str | None) -> list[str]:
        """Build keywords from context for better recognition.
        
        Args:
            context: Context string
            
        Returns:
            List of keywords for boosting
        """
        if not context:
            return []
        
        keywords = set()
        
        # Extract potential technical terms, names, and important words
        words = context.replace('-', ' ').replace('_', ' ').split()
        for word in words:
            word = word.strip('.,!?";:()[]{}')
            # Include words that are likely proper nouns or technical terms
            # Check for: title case, all caps, has digits, or mixed case
            has_upper = any(c.isupper() for c in word)
            has_lower = any(c.islower() for c in word)
            is_mixed_case = has_upper and has_lower and word[0].isupper()
            
            if len(word) > 3 and (word.istitle() or word.isupper() or any(c.isdigit() for c in word) or is_mixed_case):
                keywords.add(word)
        
        # Add domain-specific keywords for technical content
        if any(term in context.lower() for term in ['tech', 'programming', 'coding', 'software']):
            tech_keywords = {
                'API', 'SDK', 'GitHub', 'Python', 'JavaScript', 'React',
                'Node.js', 'AWS', 'Azure', 'Docker', 'Kubernetes', 'AI', 'ML'
            }
            keywords.update(tech_keywords)
        elif any(term in context.lower() for term in ['business', 'marketing', 'finance']):
            business_keywords = {
                'ROI', 'KPI', 'SaaS', 'B2B', 'B2C', 'CRM', 'CEO', 'CFO'
            }
            keywords.update(business_keywords)
                
        return list(keywords)[:25]  # Limit to 25 keywords for performance
    
    async def _call_deepgram_api(self, audio_path: Path, options: dict[str, Any]) -> Any:
        """Call Deepgram API with the audio file.
        
        Args:
            audio_path: Path to audio file
            options: Transcription options
            
        Returns:
            Deepgram API response
        """
        try:
            # Create PrerecordedOptions from dict
            prerecorded_options = PrerecordedOptions(**options)
            
            # Read audio file
            with open(audio_path, 'rb') as audio_file:
                buffer_data = audio_file.read()
            
            # Use asyncio.to_thread for I/O bound operation
            response = await asyncio.to_thread(
                self.client.listen.rest.v("1").transcribe_file,
                {"buffer": buffer_data},
                prerecorded_options
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Deepgram API call failed for {audio_path}: {e}")
            raise
    
    def _parse_response(self, response: Any) -> list[TranscriptSegment]:
        """Parse Deepgram response into TranscriptSegment objects.
        
        Args:
            response: Deepgram API response
            
        Returns:
            List of TranscriptSegment objects
        """
        segments = []
        
        try:
            # Get the result from response
            result = getattr(response, 'results', None)
            if not result or not result.channels:
                return segments
            
            channel = result.channels[0]
            if not channel.alternatives:
                return segments
            
            alternative = channel.alternatives[0]
            
            # Check if we have utterances (best for segmentation)
            if hasattr(alternative, 'paragraphs') and alternative.paragraphs:
                paragraphs = alternative.paragraphs.paragraphs
                for paragraph in paragraphs:
                    if hasattr(paragraph, 'sentences'):
                        for sentence in paragraph.sentences:
                            text = getattr(sentence, 'text', '').strip()
                            if text:
                                segments.append(TranscriptSegment(
                                    start=getattr(sentence, 'start', 0.0),
                                    end=getattr(sentence, 'end', 0.0),
                                    text=text,
                                    confidence=getattr(sentence, 'confidence', 0.9)
                                ))
            
            # Fallback: use words to create segments
            elif hasattr(alternative, 'words') and alternative.words:
                # Group words into segments by natural breaks
                current_segment_words = []
                current_speaker = None
                
                for word in alternative.words:
                    word_text = getattr(word, 'word', '')
                    word_start = getattr(word, 'start', 0.0)
                    word_end = getattr(word, 'end', 0.0)
                    word_confidence = getattr(word, 'confidence', 0.9)
                    word_speaker = getattr(word, 'speaker', None)
                    
                    # Start new segment if speaker changes or if we have too many words
                    if (word_speaker != current_speaker and current_segment_words) or len(current_segment_words) >= 50:
                        if current_segment_words:
                            segments.append(self._create_segment_from_words(
                                current_segment_words, current_speaker
                            ))
                        current_segment_words = []
                        current_speaker = word_speaker
                    
                    # Add word to current segment
                    current_segment_words.append({
                        'text': word_text,
                        'start': word_start,
                        'end': word_end,
                        'confidence': word_confidence,
                        'speaker': word_speaker
                    })
                    
                    if current_speaker is None:
                        current_speaker = word_speaker
                
                # Add final segment
                if current_segment_words:
                    segments.append(self._create_segment_from_words(
                        current_segment_words, current_speaker
                    ))
            
            # Final fallback: use basic transcript with estimated timing
            elif hasattr(alternative, 'transcript') and alternative.transcript:
                text = alternative.transcript.strip()
                if text:
                    segments.append(TranscriptSegment(
                        start=0.0,
                        end=1.0,  # Will need duration from metadata
                        text=text,
                        confidence=getattr(alternative, 'confidence', 0.9)
                    ))
            
            logger.debug(f"Parsed {len(segments)} segments from Deepgram response")
            return segments
            
        except Exception as e:
            logger.error(f"Failed to parse Deepgram response: {e}")
            return []
    
    def _create_segment_from_words(self, words: list[dict], speaker: str | None) -> TranscriptSegment:
        """Create a transcript segment from a list of words.
        
        Args:
            words: List of word dictionaries
            speaker: Speaker identifier
            
        Returns:
            TranscriptSegment object
        """
        if not words:
            return TranscriptSegment(start=0.0, end=0.0, text="", confidence=0.0)
        
        # Combine text
        text = ' '.join(word['text'] for word in words).strip()
        
        # Get timing
        start_time = words[0]['start']
        end_time = words[-1]['end']
        
        # Calculate average confidence
        confidences = [word['confidence'] for word in words if word['confidence'] is not None]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.9
        
        return TranscriptSegment(
            start=start_time,
            end=end_time,
            text=text,
            speaker=f"Speaker {speaker}" if speaker is not None else None,
            confidence=avg_confidence
        )
    
    def _detect_language(self, result: Any, specified_language: str | None) -> str:
        """Detect or return language from Deepgram response.
        
        Args:
            result: Deepgram result object
            specified_language: User-specified language
            
        Returns:
            Language code
        """
        if specified_language and specified_language != "auto":
            return specified_language
            
        # Try to get detected language from Deepgram response
        if result and hasattr(result, 'channels') and result.channels:
            channel = result.channels[0]
            if hasattr(channel, 'detected_language'):
                return channel.detected_language
            if hasattr(channel, 'alternatives') and channel.alternatives:
                alt = channel.alternatives[0]
                if hasattr(alt, 'language'):
                    return alt.language
                    
        return "en"  # Default fallback
    
    def _estimate_confidence(self, alternative: Any, segments: list[TranscriptSegment]) -> float:
        """Estimate overall confidence score from Deepgram response.
        
        Args:
            alternative: Deepgram alternative object
            segments: Parsed transcript segments
            
        Returns:
            Overall confidence score (0.0-1.0)
        """
        try:
            # Use alternative-level confidence if available
            if alternative and hasattr(alternative, 'confidence') and alternative.confidence is not None:
                return float(alternative.confidence)
            
            # Calculate from segment confidences
            if segments:
                confidences = [seg.confidence for seg in segments if seg.confidence is not None]
                if confidences:
                    return sum(confidences) / len(confidences)
            
            # High confidence for Deepgram Nova-3
            return 0.92
            
        except Exception as e:
            logger.warning(f"Failed to estimate confidence: {e}")
            return 0.88
    
    @type_validated
    def estimate_cost(self, duration_seconds: float) -> float:
        """Estimate Deepgram transcription cost in USD.
        
        Args:
            duration_seconds: Audio duration in seconds
            
        Returns:
            Estimated cost in USD
            
        Raises:
            ValueError: If duration is invalid
        """
        if duration_seconds <= 0:
            raise ValueError("Duration must be positive")
        
        # Deepgram pricing varies by tier and model
        # Nova-3: ~$0.0043 per minute
        # Enhanced: ~$0.0025 per minute
        # Base: ~$0.0015 per minute
        cost_per_minute = {
            "nova": 0.0043,
            "enhanced": 0.0025,
            "base": 0.0015
        }
        
        minutes = duration_seconds / 60.0
        base_cost = minutes * cost_per_minute.get(self.tier, 0.0043)
        
        # Additional costs for advanced features
        if self.enable_speaker_diarization:
            base_cost += minutes * 0.0010  # Diarization add-on
        
        return base_cost
    
    @property
    def provider_name(self) -> str:
        """Return provider name for identification."""
        return "deepgram"
    
    @property
    def supports_speaker_diarization(self) -> bool:
        """Return whether speaker diarization is supported."""
        return True
    
    @property
    def supported_languages(self) -> list[str]:
        """Return list of supported language codes."""
        return self._supported_languages.copy()
    
    @property
    def model_info(self) -> dict[str, Any]:
        """Return information about the current model."""
        return {
            "name": self.model_name,
            "provider": "deepgram",
            "tier": self.tier,
            "supports_streaming": True,
            "supports_word_timestamps": True,
            "supports_speaker_diarization": True,
            "supports_language_detection": True,
            "supports_smart_formatting": True,
            "max_file_size_mb": 2000,  # Deepgram's limit
            "supported_formats": [
                "mp3", "mp4", "wav", "flac", "m4a", "wma", "ogg", "webm", 
                "aac", "amr", "3gp", "opus"
            ]
        }