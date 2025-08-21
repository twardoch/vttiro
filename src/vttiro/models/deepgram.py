#!/usr/bin/env python3
# this_file: src/vttiro/models/deepgram.py
"""Deepgram Nova-3 transcription engine optimized for speed and real-time processing."""

import asyncio
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

try:
    from loguru import logger
except ImportError:
    import logging as logger

try:
    from deepgram import DeepgramClient, PrerecordedOptions
    DEEPGRAM_AVAILABLE = True
except ImportError:
    DEEPGRAM_AVAILABLE = False
    logger.warning("Deepgram SDK not available. Install with: uv add deepgram-sdk")
    # Create dummy classes to avoid NameError
    DeepgramClient = None
    PrerecordedOptions = object

from vttiro.core.transcription import TranscriptionEngine
from vttiro.core.config import VttiroConfig, TranscriptionResult
from vttiro.models.base import DeepgramModel


class DeepgramTranscriber(TranscriptionEngine):
    """Deepgram Nova-3 transcription engine optimized for speed and multilingual support."""
    
    def __init__(self, config: VttiroConfig, model: DeepgramModel = DeepgramModel.NOVA_3):
        super().__init__(config)
        self.model_variant = model
        
        if not DEEPGRAM_AVAILABLE:
            raise ImportError("Deepgram SDK not available. Install with: uv add deepgram-sdk")
        
        # Configure Deepgram API
        api_key = config.transcription.deepgram_api_key
        if not api_key:
            raise ValueError("Deepgram API key not configured. Set DEEPGRAM_API_KEY environment variable.")
            
        self.client = DeepgramClient(api_key)
        
        # Deepgram Nova-3 supports 30+ languages
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
        
    @property
    def name(self) -> str:
        return f"deepgram/{self.model_variant.value}"
        
    async def transcribe(
        self, 
        audio_path: Path, 
        language: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> TranscriptionResult:
        """Transcribe audio using Deepgram Nova-3 for fast, accurate results."""
        start_time = time.time()
        
        logger.info(f"Transcribing with Deepgram: {audio_path}")
        
        try:
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # Configure transcription options
            options = self._build_transcription_options(language, context)
            
            # Perform transcription
            response = await self._transcribe_with_deepgram(audio_path, options)
            
            processing_time = time.time() - start_time
            
            # Extract results from response
            result = response.results
            if not result.channels or not result.channels[0].alternatives:
                raise RuntimeError("No transcription results returned from Deepgram")
                
            alternative = result.channels[0].alternatives[0]
            transcribed_text = alternative.transcript
            confidence = alternative.confidence or 0.0
            
            # Extract word-level timestamps
            word_timestamps = self._extract_word_timestamps(alternative)
            
            # Detect language if not specified
            detected_language = self._detect_language(result, language)
            
            logger.info(f"Deepgram transcription completed in {processing_time:.2f}s")
            
            return TranscriptionResult(
                text=transcribed_text,
                confidence=confidence,
                word_timestamps=word_timestamps,
                processing_time=processing_time,
                model_name=self.model_variant.value,
                language=detected_language,
                metadata={
                    "engine": "deepgram",
                    "model_version": self.model_variant.value,
                    "model_info": getattr(result, 'model_info', {}),
                    "request_id": getattr(response, 'request_id', None),
                    "channels": len(result.channels) if result.channels else 0,
                    "alternatives": len(result.channels[0].alternatives) if result.channels and result.channels[0].alternatives else 0,
                    "context_used": bool(context),
                    "features_used": self._get_enabled_features(options),
                    "processing_duration": getattr(result, 'duration', None)
                }
            )
            
        except Exception as e:
            logger.error(f"Deepgram transcription failed for {audio_path}: {e}")
            raise
            
    def _build_transcription_options(
        self, 
        language: Optional[str], 
        context: Optional[Dict[str, Any]]
    ) -> PrerecordedOptions:
        """Build Deepgram transcription options for optimal performance."""
        
        # Base options for specified model
        options_dict = {
            # Model selection
            "model": self._get_deepgram_model(),  # Use specified model
            "version": "latest",
            
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
            "diarize": self.config.diarization.enabled,
            "multichannel": False,  # Our audio is mono from segmentation
            "alternatives": 1,  # Just the best alternative for speed
            "numerals": True,  # Convert numbers to numerals
            "search": [],  # No search terms by default
            "replace": [],  # No replacements by default
            "keywords": [],  # Will be populated from context
            
            # Performance optimizations
            "tier": "nova",  # Use Nova tier for best accuracy
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
                options_dict["keywords"] = keywords
                options_dict["keyword_boost"] = "high"
                logger.debug(f"Using Deepgram keywords: {keywords}")
                
            # Adjust diarization settings based on context
            if context.get('video_title'):
                title = context['video_title'].lower()
                if any(word in title for word in ['interview', 'conversation', 'podcast']):
                    options_dict["diarize"] = True
                    options_dict["diarize_version"] = "2023-09-07"  # Latest diarization version
                elif any(word in title for word in ['lecture', 'presentation', 'monologue']):
                    options_dict["diarize"] = False  # Single speaker likely
                    
        # Remove None values
        options_dict = {k: v for k, v in options_dict.items() if v is not None}
        
        return PrerecordedOptions(**options_dict)
    
    def _get_deepgram_model(self) -> str:
        """Get the appropriate Deepgram model based on model variant."""
        # Map our model enum values to Deepgram model values
        model_mapping = {
            DeepgramModel.NOVA_2: "nova-2",
            DeepgramModel.NOVA_3: "nova-3", 
            DeepgramModel.ENHANCED: "enhanced",
            DeepgramModel.BASE: "base",
            DeepgramModel.WHISPER_CLOUD: "whisper-cloud"
        }
        
        # Return mapped value or fall back to model string value
        return model_mapping.get(self.model_variant, self.model_variant.value)
        
    def _normalize_language_code(self, language: str) -> str:
        """Normalize language code to Deepgram format."""
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
        
    def _build_keywords(self, context: Dict[str, Any]) -> List[str]:
        """Build keywords from video context for better recognition."""
        keywords = set()
        
        # Extract important terms from video metadata
        if context.get('video_uploader'):
            uploader = context['video_uploader']
            # Clean and extract meaningful words
            words = uploader.replace('_', ' ').replace('-', ' ').split()
            keywords.update(word.strip() for word in words if len(word) > 2)
            
        if context.get('video_title'):
            title = context['video_title']
            # Extract potential proper nouns and important terms
            words = title.split()
            for word in words:
                if word and len(word) > 3:
                    # Remove punctuation
                    clean_word = ''.join(c for c in word if c.isalnum())
                    if clean_word and clean_word[0].isupper():
                        keywords.add(clean_word)
                        
        # Add domain-specific keywords based on content
        if context.get('video_description'):
            desc = context['video_description'].lower()
            if any(term in desc for term in ['tech', 'programming', 'coding', 'software']):
                tech_keywords = [
                    'API', 'SDK', 'GitHub', 'Python', 'JavaScript', 'React',
                    'Node.js', 'AWS', 'Azure', 'Docker', 'Kubernetes', 'AI', 'ML'
                ]
                keywords.update(tech_keywords)
            elif any(term in desc for term in ['business', 'marketing', 'finance']):
                business_keywords = [
                    'ROI', 'KPI', 'SaaS', 'B2B', 'B2C', 'CRM', 'CEO', 'CFO'
                ]
                keywords.update(business_keywords)
                
        # Limit keywords to avoid API limits  
        return list(keywords)[:25] if keywords else []
        
    async def _transcribe_with_deepgram(
        self, 
        audio_path: Path, 
        options: PrerecordedOptions
    ) -> Any:
        """Perform transcription using Deepgram API."""
        try:
            # Read audio file
            with open(audio_path, 'rb') as audio_file:
                buffer_data = audio_file.read()
                
            logger.debug(f"Starting Deepgram transcription with options: {options}")
            
            # Call Deepgram API
            response = await asyncio.to_thread(
                self.client.listen.rest.v("1").transcribe_file,
                {"buffer": buffer_data},
                options
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Deepgram API call failed: {e}")
            raise
            
    def _extract_word_timestamps(self, alternative: Any) -> List[Dict[str, Any]]:
        """Extract word-level timestamps from Deepgram response."""
        word_timestamps = []
        
        if hasattr(alternative, 'words') and alternative.words:
            for word_obj in alternative.words:
                word_timestamps.append({
                    "word": word_obj.word,
                    "start": word_obj.start,
                    "end": word_obj.end,
                    "confidence": word_obj.confidence,
                    "punctuated_word": getattr(word_obj, 'punctuated_word', word_obj.word)
                })
        else:
            # Fallback: estimate timestamps from transcript
            words = alternative.transcript.split() if alternative.transcript else []
            if words:
                # Use rough estimation based on speech rate
                estimated_duration = len(words) / 3.0  # ~3 words per second
                
                for i, word in enumerate(words):
                    start_time = (i / len(words)) * estimated_duration
                    end_time = ((i + 1) / len(words)) * estimated_duration
                    
                    word_timestamps.append({
                        "word": word,
                        "start": start_time,
                        "end": end_time,
                        "confidence": alternative.confidence or 0.95
                    })
                    
        return word_timestamps
        
    def _detect_language(self, result: Any, specified_language: Optional[str]) -> str:
        """Detect or return language from Deepgram response."""
        if specified_language and specified_language != "auto":
            return specified_language
            
        # Try to get detected language from Deepgram response
        if hasattr(result, 'channels') and result.channels:
            channel = result.channels[0]
            if hasattr(channel, 'detected_language'):
                return channel.detected_language
            if hasattr(channel, 'alternatives') and channel.alternatives:
                alt = channel.alternatives[0]
                if hasattr(alt, 'language'):
                    return alt.language
                    
        return "en"  # Default fallback
        
    def _get_enabled_features(self, options: PrerecordedOptions) -> List[str]:
        """Get list of enabled Deepgram features for metadata."""
        features = []
        
        # Map options to feature names
        if getattr(options, 'diarize', False):
            features.append('speaker_diarization')
        if getattr(options, 'punctuate', False):
            features.append('punctuation')
        if getattr(options, 'paragraphs', False):
            features.append('paragraph_detection')
        if getattr(options, 'utterances', False):
            features.append('utterance_segmentation')
        if getattr(options, 'detect_language', False):
            features.append('language_detection')
        if getattr(options, 'smart_format', False):
            features.append('smart_formatting')
        if getattr(options, 'numerals', False):
            features.append('numeral_formatting')
        if getattr(options, 'keywords', []):
            features.append('keyword_boosting')
            
        return features
        
    def estimate_cost(self, duration_seconds: float) -> float:
        """Estimate Deepgram transcription cost in USD."""
        # Deepgram Nova-2/3 pricing: ~$0.0043 per minute for base features
        minutes = duration_seconds / 60.0
        base_cost = minutes * 0.0043
        
        # Additional costs for advanced features
        if self.config.diarization.enabled:
            base_cost += minutes * 0.0010  # Diarization add-on
            
        return base_cost
        
    def get_supported_languages(self) -> List[str]:
        """Return list of supported language codes."""
        return self._supported_languages.copy()