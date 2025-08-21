#!/usr/bin/env python3
# this_file: src/vttiro/models/assemblyai.py
"""AssemblyAI Universal-2 transcription engine for maximum accuracy."""

import asyncio
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

try:
    from loguru import logger
except ImportError:
    import logging as logger

try:
    import assemblyai as aai
    ASSEMBLYAI_AVAILABLE = True
except ImportError:
    ASSEMBLYAI_AVAILABLE = False
    logger.warning("AssemblyAI not available. Install with: uv add assemblyai")
    # Create dummy objects to avoid NameError
    aai = object()
    aai.settings = object()
    aai.Transcriber = object
    aai.TranscriptionConfig = object
    aai.TranscriptStatus = object()
    aai.SpeechModel = object()
    aai.SpeechModel.best = "best"

from vttiro.core.transcription import TranscriptionEngine
from vttiro.core.config import VttiroConfig, TranscriptionResult
from vttiro.models.base import AssemblyAIModel


class AssemblyAITranscriber(TranscriptionEngine):
    """AssemblyAI Universal-2 transcription engine optimized for maximum accuracy."""
    
    def __init__(self, config: VttiroConfig, model: AssemblyAIModel = AssemblyAIModel.UNIVERSAL_2):
        super().__init__(config)
        self.model_variant = model
        
        if not ASSEMBLYAI_AVAILABLE:
            raise ImportError("AssemblyAI not available. Install with: uv add assemblyai")
        
        # Configure AssemblyAI API
        api_key = config.transcription.assemblyai_api_key
        if not api_key:
            raise ValueError("AssemblyAI API key not configured. Set ASSEMBLYAI_API_KEY environment variable.")
            
        aai.settings.api_key = api_key
        
        # Initialize transcriber with optimal settings
        self.transcriber = aai.Transcriber()
        
        # Model capabilities - AssemblyAI Universal-2 supports many languages
        self._supported_languages = [
            "en", "es", "fr", "de", "it", "pt", "pl", "nl", "tr", "ru",
            "ar", "zh", "ja", "ko", "hi", "th", "vi", "uk", "cs", "da",
            "fi", "no", "sv", "he", "id", "ms", "ro", "sk", "bg", "hr"
        ]
        
    @property
    def name(self) -> str:
        return f"assemblyai/{self.model_variant.value}"
        
    async def transcribe(
        self, 
        audio_path: Path, 
        language: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> TranscriptionResult:
        """Transcribe audio using AssemblyAI Universal-2 for maximum accuracy."""
        start_time = time.time()
        
        logger.info(f"Transcribing with AssemblyAI: {audio_path}")
        
        try:
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # Configure transcription settings
            config = self._build_transcription_config(language, context)
            
            # Perform transcription
            transcript = await self._transcribe_with_assemblyai(audio_path, config)
            
            processing_time = time.time() - start_time
            
            # Extract results
            transcribed_text = transcript.text or ""
            confidence = transcript.confidence or 0.0
            
            # Extract word-level timestamps
            word_timestamps = self._extract_word_timestamps(transcript)
            
            logger.info(f"AssemblyAI transcription completed in {processing_time:.2f}s")
            
            return TranscriptionResult(
                text=transcribed_text,
                confidence=confidence,
                word_timestamps=word_timestamps,
                processing_time=processing_time,
                model_name=self.model_variant.value,
                language=language or transcript.language_code or "auto",
                metadata={
                    "engine": "assemblyai",
                    "model_version": self.model_variant.value, 
                    "transcript_id": transcript.id,
                    "status": transcript.status.value,
                    "audio_duration": getattr(transcript, 'audio_duration', None),
                    "acoustic_model": getattr(transcript, 'acoustic_model', self.model_variant.value),
                    "language_model": getattr(transcript, 'language_model', 'default'),
                    "confidence_threshold": config.get('confidence_threshold', 0.0),
                    "context_used": bool(context),
                    "features_used": self._get_enabled_features(config)
                }
            )
            
        except Exception as e:
            logger.error(f"AssemblyAI transcription failed for {audio_path}: {e}")
            raise
            
    def _build_transcription_config(
        self, 
        language: Optional[str], 
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build AssemblyAI transcription configuration."""
        
        # Base configuration for maximum accuracy
        config = {
            # Core settings
            "speech_model": self._get_speech_model(),  # Use specified model
            "language_code": language if language and language != "auto" else None,
            "punctuate": True,
            "format_text": True,
            
            # Advanced features for accuracy
            "speaker_labels": self.config.diarization.enabled,
            "speakers_expected": self.config.diarization.max_speakers if self.config.diarization.enabled else None,
            "auto_highlights": True,  # Identify key phrases
            "entity_detection": True,  # Detect names, organizations, etc.
            "sentiment_analysis": False,  # Not needed for transcription
            "auto_chapters": False,    # Not needed for segments
            
            # Quality settings
            "filter_profanity": False,  # Preserve original content
            "redact_pii": False,       # Don't redact for accuracy
            "confidence_threshold": 0.0,  # Include all transcription
            
            # Audio processing
            "audio_start_from": None,
            "audio_end_at": None,
            "word_boost": [],  # Will be populated from context
            "boost_param": "high"  # High boost for custom vocabulary
        }
        
        # Add context-specific enhancements
        if context:
            # Build custom vocabulary from context
            vocabulary = self._build_custom_vocabulary(context)
            if vocabulary:
                config["word_boost"] = vocabulary
                logger.debug(f"Using custom vocabulary: {vocabulary}")
                
            # Adjust speaker expectations based on content
            if context.get('video_title'):
                title = context['video_title'].lower()
                if any(word in title for word in ['interview', 'conversation', 'discussion']):
                    config["speakers_expected"] = min(4, config.get("speakers_expected", 2))
                elif any(word in title for word in ['lecture', 'presentation', 'speech']):
                    config["speakers_expected"] = 1
                    
        return config
    
    def _get_speech_model(self) -> str:
        """Get the appropriate AssemblyAI speech model based on model variant."""
        # Map our model enum values to AssemblyAI speech model values
        model_mapping = {
            AssemblyAIModel.UNIVERSAL_1: "universal-1",
            AssemblyAIModel.UNIVERSAL_2: "universal-2", 
            AssemblyAIModel.NANO: "nano",
            AssemblyAIModel.BEST: "best"
        }
        
        # Return mapped value or fall back to model string value
        return model_mapping.get(self.model_variant, self.model_variant.value)
        
    def _build_custom_vocabulary(self, context: Dict[str, Any]) -> List[str]:
        """Build custom vocabulary from video context for better recognition."""
        vocabulary = set()
        
        # Extract names and brands from video metadata
        if context.get('video_uploader'):
            uploader = context['video_uploader']
            # Split by common separators and add individual words
            words = uploader.replace('_', ' ').replace('-', ' ').split()
            vocabulary.update(word.strip() for word in words if len(word) > 2)
            
        if context.get('video_title'):
            title = context['video_title']
            # Extract potential proper nouns (capitalized words)
            words = title.split()
            for word in words:
                if word and word[0].isupper() and len(word) > 2:
                    # Remove punctuation
                    clean_word = ''.join(c for c in word if c.isalnum())
                    if clean_word:
                        vocabulary.add(clean_word)
                        
        # Common technical/domain terms that might be mispronounced
        tech_terms = {
            'API', 'SDK', 'AI', 'ML', 'GPU', 'CPU', 'SaaS', 'API',
            'GitHub', 'Python', 'JavaScript', 'React', 'Node.js',
            'AWS', 'Azure', 'Google Cloud', 'Docker', 'Kubernetes'
        }
        
        # Add tech terms if video seems technical
        if context.get('video_title') or context.get('video_description'):
            text = (context.get('video_title', '') + ' ' + 
                   context.get('video_description', '')).lower()
            if any(term in text for term in ['tech', 'coding', 'programming', 'software', 'development']):
                vocabulary.update(tech_terms)
                
        # Limit vocabulary size to avoid API limits
        return list(vocabulary)[:50] if vocabulary else []
        
    async def _transcribe_with_assemblyai(
        self, 
        audio_path: Path, 
        config: Dict[str, Any]
    ) -> Any:
        """Perform transcription using AssemblyAI API."""
        try:
            # Create TranscriptionConfig object
            transcription_config = aai.TranscriptionConfig(**config)
            
            # Upload and transcribe
            logger.debug(f"Starting AssemblyAI transcription with config: {config}")
            
            transcript = await asyncio.to_thread(
                self.transcriber.transcribe,
                str(audio_path),
                config=transcription_config
            )
            
            if transcript.status == aai.TranscriptStatus.error:
                raise RuntimeError(f"AssemblyAI transcription failed: {transcript.error}")
                
            return transcript
            
        except Exception as e:
            logger.error(f"AssemblyAI API call failed: {e}")
            raise
            
    def _extract_word_timestamps(self, transcript: Any) -> List[Dict[str, Any]]:
        """Extract word-level timestamps from AssemblyAI response."""
        word_timestamps = []
        
        if hasattr(transcript, 'words') and transcript.words:
            for word_obj in transcript.words:
                word_timestamps.append({
                    "word": word_obj.text,
                    "start": word_obj.start / 1000.0,  # Convert ms to seconds
                    "end": word_obj.end / 1000.0,
                    "confidence": word_obj.confidence
                })
        else:
            # Fallback: estimate timestamps if word-level data not available
            words = transcript.text.split() if transcript.text else []
            if words and hasattr(transcript, 'audio_duration') and transcript.audio_duration:
                duration = transcript.audio_duration / 1000.0  # Convert ms to seconds
                for i, word in enumerate(words):
                    start_time = (i / len(words)) * duration
                    end_time = ((i + 1) / len(words)) * duration
                    
                    word_timestamps.append({
                        "word": word,
                        "start": start_time,
                        "end": end_time,
                        "confidence": transcript.confidence or 0.95
                    })
                    
        return word_timestamps
        
    def _get_enabled_features(self, config: Dict[str, Any]) -> List[str]:
        """Get list of enabled AssemblyAI features for metadata."""
        features = []
        
        feature_mapping = {
            'speaker_labels': 'speaker_diarization',
            'auto_highlights': 'key_phrase_detection',
            'entity_detection': 'named_entity_recognition',
            'sentiment_analysis': 'sentiment_analysis',
            'auto_chapters': 'chapter_detection',
            'filter_profanity': 'profanity_filter',
            'redact_pii': 'pii_redaction'
        }
        
        for config_key, feature_name in feature_mapping.items():
            if config.get(config_key):
                features.append(feature_name)
                
        return features
        
    def estimate_cost(self, duration_seconds: float) -> float:
        """Estimate AssemblyAI transcription cost in USD."""
        # AssemblyAI pricing: ~$0.00037 per second for Core models
        # Universal-2 might have premium pricing, estimated at $0.00074 per second
        cost_per_second = 0.00074
        
        # Additional costs for advanced features
        base_cost = duration_seconds * cost_per_second
        
        # Speaker diarization adds ~50% cost
        if self.config.diarization.enabled:
            base_cost *= 1.5
            
        return base_cost
        
    def get_supported_languages(self) -> List[str]:
        """Return list of supported language codes."""
        return self._supported_languages.copy()