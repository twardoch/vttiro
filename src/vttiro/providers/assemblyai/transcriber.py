# this_file: src/vttiro/providers/assemblyai/transcriber.py
"""AssemblyAI Universal-2 transcription provider.

This module implements the AssemblyAI transcription provider using the new
VTTiro 2.0 architecture. Provides high-accuracy transcription with
Universal-2 model, speaker diarization, and advanced features.

Used by:
- Core orchestration for AssemblyAI-based transcription
- Provider selection logic
- Testing infrastructure for AssemblyAI functionality
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
    import assemblyai as aai
    ASSEMBLYAI_AVAILABLE = True
except ImportError:
    ASSEMBLYAI_AVAILABLE = False
    logger.warning("AssemblyAI not available. Install with: uv add assemblyai")
    aai = None


class AssemblyAITranscriber(TranscriberABC):
    """AssemblyAI Universal-2 transcription provider.
    
    Implements high-accuracy transcription using AssemblyAI's Universal-2 model
    with speaker diarization, word-level timestamps, and advanced features
    for maximum accuracy transcription.
    
    Features:
    - Universal-2 model for highest accuracy
    - Speaker diarization and identification  
    - Word-level timestamps with confidence scores
    - Custom vocabulary and context awareness
    - Entity detection and auto-highlights
    - Multi-language support with auto-detection
    """
    
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "universal-2",
        enable_speaker_diarization: bool = True,
        max_speakers: int | None = None
    ):
        """Initialize AssemblyAI transcriber.
        
        Args:
            api_key: AssemblyAI API key (or set ASSEMBLYAI_API_KEY env var)
            model: AssemblyAI model to use (universal-2, universal-1, nano, best)
            enable_speaker_diarization: Whether to enable speaker identification
            max_speakers: Maximum number of speakers to identify
            
        Raises:
            ImportError: If assemblyai package is not installed
            AuthenticationError: If API key is missing or invalid
        """
        if not ASSEMBLYAI_AVAILABLE:
            raise ImportError(
                "AssemblyAI not available. Install with: uv add assemblyai"
            )
        
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv("ASSEMBLYAI_API_KEY")
        if not self.api_key:
            raise AuthenticationError(
                "AssemblyAI API key not provided. Set ASSEMBLYAI_API_KEY environment variable "
                "or pass api_key parameter.",
                provider="assemblyai"
            )
        
        self.model_name = model
        self.enable_speaker_diarization = enable_speaker_diarization
        self.max_speakers = max_speakers
        
        # Configure AssemblyAI API
        aai.settings.api_key = self.api_key
        
        # Initialize transcriber
        try:
            self.transcriber = aai.Transcriber()
        except Exception as e:
            raise handle_provider_exception(e, "assemblyai")
        
        # Model capabilities and supported languages
        self._supported_languages = [
            "en", "es", "fr", "de", "it", "pt", "pl", "nl", "tr", "ru",
            "ar", "zh", "ja", "ko", "hi", "th", "vi", "uk", "cs", "da",
            "fi", "no", "sv", "he", "id", "ms", "ro", "sk", "bg", "hr",
            "ca", "eu", "gl", "mt", "cy", "ga", "is", "lv", "lt", "mk",
            "sl", "et", "sq", "bs", "hr", "sr", "me", "af", "sw", "zu"
        ]
    
    @type_validated
    async def transcribe(
        self,
        audio_path: Path,
        language: str | None = None,
        context: str | None = None,
        **kwargs: Any
    ) -> TranscriptionResult:
        """Transcribe audio using AssemblyAI Universal-2.
        
        Args:
            audio_path: Path to audio file
            language: Language code (ISO 639-1) or None for auto-detection
            context: Additional context to improve transcription accuracy
            **kwargs: Additional parameters (speaker_labels, etc.)
            
        Returns:
            TranscriptionResult with segments and metadata
            
        Raises:
            ProcessingError: If transcription fails
            AuthenticationError: If API key is invalid
        """
        logger.info(f"Starting AssemblyAI transcription: {audio_path}")
        start_time = time.time()
        
        # Validate audio file exists
        if not audio_path.exists():
            raise ProcessingError(
                f"Audio file not found: {audio_path}",
                file_path=str(audio_path)
            )
        
        # Prepare transcription configuration
        config = self._prepare_transcription_config(language, context, **kwargs)
        logger.debug(f"AssemblyAI transcription config: {config}")
        
        try:
            # Perform transcription
            transcript = await self._call_assemblyai_api(audio_path, config)
            
            # Process response into segments
            segments = self._parse_response(transcript)
            
            # Calculate processing metrics
            processing_time = time.time() - start_time
            logger.info(f"AssemblyAI transcription completed in {processing_time:.2f}s")
            
            # Build metadata
            metadata = {
                "provider": "assemblyai",
                "model": self.model_name,
                "processing_time": processing_time,
                "transcript_id": getattr(transcript, 'id', None),
                "status": getattr(transcript, 'status', None),
                "audio_duration": getattr(transcript, 'audio_duration', None),
                "confidence": getattr(transcript, 'confidence', None),
                "language_detected": getattr(transcript, 'language_code', language),
                "speaker_labels_enabled": config.get("speaker_labels", False),
                "auto_highlights": getattr(transcript, 'auto_highlights_result', None),
                "entities": getattr(transcript, 'entities', None)
            }
            
            # Estimate confidence score
            confidence = self._estimate_confidence(transcript, segments)
            
            return TranscriptionResult(
                segments=segments,
                metadata=metadata,
                provider="assemblyai",
                language=getattr(transcript, 'language_code', language),
                confidence=confidence
            )
            
        except aai.types.TranscriptError as e:
            logger.error(f"AssemblyAI transcript error: {e}")
            raise ProcessingError(f"AssemblyAI transcript error: {e}", file_path=str(audio_path)) from e
        except Exception as e:
            logger.error(f"AssemblyAI transcription failed: {e}")
            raise ProcessingError(f"Transcription failed: {e}", file_path=str(audio_path)) from e
    
    def _prepare_transcription_config(
        self, 
        language: str | None, 
        context: str | None,
        **kwargs: Any
    ) -> dict[str, Any]:
        """Prepare configuration for AssemblyAI transcription.
        
        Args:
            language: Target language code
            context: Additional context for transcription
            **kwargs: Additional configuration parameters
            
        Returns:
            Dictionary of API configuration parameters
        """
        # Build base configuration for maximum accuracy
        config = aai.TranscriptionConfig(
            speech_model=aai.SpeechModel.universal_2 if self.model_name == "universal-2" else aai.SpeechModel.best,
            language_code=language if language and language != "auto" and language in self._supported_languages else None,
            punctuate=True,
            format_text=True,
            
            # Advanced features for accuracy
            speaker_labels=kwargs.get("speaker_labels", self.enable_speaker_diarization),
            speakers_expected=kwargs.get("speakers_expected", self.max_speakers),
            auto_highlights=True,  # Identify key phrases
            entity_detection=True,  # Detect names, organizations, etc.
            
            # Quality settings
            filter_profanity=False,  # Preserve original content
            redact_pii=False,       # Don't redact for accuracy
            confidence_threshold=0.0,  # Include all transcription
            
            # Audio processing
            boost_param=aai.types.WordBoost.high if context else aai.types.WordBoost.default,
            word_boost=self._build_custom_vocabulary(context) if context else None
        )
        
        return config
    
    def _build_custom_vocabulary(self, context: str | None) -> list[str]:
        """Build custom vocabulary from context for better recognition.
        
        Args:
            context: Context string
            
        Returns:
            List of custom vocabulary words
        """
        if not context:
            return []
        
        vocabulary = set()
        
        # Extract potential technical terms, names, and important words
        words = context.replace('-', ' ').replace('_', ' ').split()
        for word in words:
            word = word.strip('.,!?";:()[]{}')
            # Include words that are likely proper nouns or technical terms
            # Check for: title case, all caps, has digits, or mixed case (like TensorFlow, PyTorch)
            has_upper = any(c.isupper() for c in word)
            has_lower = any(c.islower() for c in word)
            is_mixed_case = has_upper and has_lower and word[0].isupper()
            
            if len(word) > 2 and (word.istitle() or word.isupper() or any(c.isdigit() for c in word) or is_mixed_case):
                vocabulary.add(word)
        
        return list(vocabulary)[:100]  # Limit to 100 words for performance
    
    async def _call_assemblyai_api(self, audio_path: Path, config: aai.TranscriptionConfig) -> Any:
        """Call AssemblyAI API with the audio file.
        
        Args:
            audio_path: Path to audio file
            config: Transcription configuration
            
        Returns:
            AssemblyAI transcript object
        """
        try:
            # Use asyncio.to_thread for I/O bound operation
            transcript = await asyncio.to_thread(
                self.transcriber.transcribe,
                str(audio_path),
                config=config
            )
            return transcript
            
        except Exception as e:
            logger.error(f"AssemblyAI API call failed for {audio_path}: {e}")
            raise
    
    def _parse_response(self, transcript: Any) -> list[TranscriptSegment]:
        """Parse AssemblyAI response into TranscriptSegment objects.
        
        Args:
            transcript: AssemblyAI transcript object
            
        Returns:
            List of TranscriptSegment objects
        """
        segments = []
        
        try:
            # Check if transcript has words (word-level timestamps)
            if hasattr(transcript, 'words') and transcript.words:
                # Group words into segments by speaker or natural breaks
                current_segment_words = []
                current_speaker = None
                segment_start = None
                
                for word in transcript.words:
                    # Get word details
                    word_text = getattr(word, 'text', '')
                    word_start = getattr(word, 'start', 0) / 1000.0  # Convert ms to seconds
                    word_end = getattr(word, 'end', 0) / 1000.0
                    word_confidence = getattr(word, 'confidence', 0.9)
                    word_speaker = getattr(word, 'speaker', None)
                    
                    # Start new segment if speaker changes or if we have too many words
                    if (word_speaker != current_speaker and current_segment_words) or len(current_segment_words) >= 50:
                        # Create segment from accumulated words
                        if current_segment_words:
                            segments.append(self._create_segment_from_words(
                                current_segment_words, current_speaker
                            ))
                        current_segment_words = []
                        current_speaker = word_speaker
                        segment_start = word_start
                    
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
                    if segment_start is None:
                        segment_start = word_start
                
                # Add final segment
                if current_segment_words:
                    segments.append(self._create_segment_from_words(
                        current_segment_words, current_speaker
                    ))
            
            # Fallback: use utterances if available
            elif hasattr(transcript, 'utterances') and transcript.utterances:
                for utterance in transcript.utterances:
                    text = getattr(utterance, 'text', '').strip()
                    if not text:
                        continue
                    
                    start_time = getattr(utterance, 'start', 0) / 1000.0
                    end_time = getattr(utterance, 'end', 0) / 1000.0
                    confidence = getattr(utterance, 'confidence', 0.9)
                    speaker = getattr(utterance, 'speaker', None)
                    
                    segments.append(TranscriptSegment(
                        start=start_time,
                        end=end_time,
                        text=text,
                        speaker=f"Speaker {speaker}" if speaker else None,
                        confidence=confidence
                    ))
            
            # Final fallback: use basic text with estimated timing
            elif hasattr(transcript, 'text') and transcript.text:
                text = transcript.text.strip()
                if text:
                    segments.append(TranscriptSegment(
                        start=0.0,
                        end=1.0,  # Will need duration from metadata
                        text=text,
                        confidence=getattr(transcript, 'confidence', 0.9)
                    ))
            
            logger.debug(f"Parsed {len(segments)} segments from AssemblyAI response")
            return segments
            
        except Exception as e:
            logger.error(f"Failed to parse AssemblyAI response: {e}")
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
            speaker=f"Speaker {speaker}" if speaker else None,
            confidence=avg_confidence
        )
    
    def _estimate_confidence(self, transcript: Any, segments: list[TranscriptSegment]) -> float:
        """Estimate overall confidence score from AssemblyAI response.
        
        Args:
            transcript: AssemblyAI transcript object
            segments: Parsed transcript segments
            
        Returns:
            Overall confidence score (0.0-1.0)
        """
        try:
            # Use transcript-level confidence if available
            if hasattr(transcript, 'confidence') and transcript.confidence is not None:
                return float(transcript.confidence)
            
            # Calculate from segment confidences
            if segments:
                confidences = [seg.confidence for seg in segments if seg.confidence is not None]
                if confidences:
                    return sum(confidences) / len(confidences)
            
            # High confidence for AssemblyAI Universal-2
            return 0.94
            
        except Exception as e:
            logger.warning(f"Failed to estimate confidence: {e}")
            return 0.90
    
    @type_validated
    def estimate_cost(self, duration_seconds: float) -> float:
        """Estimate AssemblyAI transcription cost in USD.
        
        Args:
            duration_seconds: Audio duration in seconds
            
        Returns:
            Estimated cost in USD
            
        Raises:
            ValueError: If duration is invalid
        """
        if duration_seconds <= 0:
            raise ValueError("Duration must be positive")
        
        # AssemblyAI pricing: $0.00037 per second for Core models
        # Universal-2 estimated at $0.00074 per second
        cost_per_second = 0.00074 if self.model_name == "universal-2" else 0.00037
        
        base_cost = duration_seconds * cost_per_second
        
        # Speaker diarization adds ~50% cost
        if self.enable_speaker_diarization:
            base_cost *= 1.5
        
        return base_cost
    
    @property
    def provider_name(self) -> str:
        """Return provider name for identification."""
        return "assemblyai"
    
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
            "provider": "assemblyai",
            "supports_streaming": False,
            "supports_word_timestamps": True,
            "supports_speaker_diarization": True,
            "supports_auto_highlights": True,
            "supports_entity_detection": True,
            "max_file_size_mb": 500,  # AssemblyAI's limit
            "supported_formats": [
                "mp3", "mp4", "wav", "flac", "m4a", "wma", "ogg", "webm", "3gp"
            ]
        }