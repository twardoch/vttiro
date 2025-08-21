#!/usr/bin/env python3
# this_file: src/vttiro/models/openai.py
"""OpenAI Audio API transcription engine with Whisper and GPT-4o support."""

import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List, AsyncGenerator
import time
import io
import re

try:
    from loguru import logger
except ImportError:
    import logging as logger

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI not available. Install with: uv add openai")
    OpenAI = None

from vttiro.core.transcription import TranscriptionEngine
from vttiro.core.config import VttiroConfig, TranscriptionResult
from vttiro.models.base import OpenAIModel, get_model_capabilities
from vttiro.core.prompts import WebVTTPromptGenerator, PromptTemplate


class OpenAITranscriber(TranscriptionEngine):
    """OpenAI Audio API transcription engine with Whisper-1 and GPT-4o support."""
    
    def __init__(self, config: VttiroConfig, model: OpenAIModel = OpenAIModel.GPT_4O_TRANSCRIBE):
        try:
            logger.debug(f"Starting OpenAI transcriber initialization with model: {model.value}")
            super().__init__(config)
            self.model_variant = model
            
            # Check OpenAI availability with detailed logging
            logger.debug(f"Checking OpenAI package availability: {OPENAI_AVAILABLE}")
            if not OPENAI_AVAILABLE:
                error_msg = "OpenAI not available. Install with: uv add openai"
                logger.error(error_msg)
                raise ImportError(error_msg)
            
            # Configure OpenAI API with enhanced validation
            logger.debug("Retrieving OpenAI API key from configuration")
            api_key = config.transcription.openai_api_key
            if not api_key:
                error_msg = "OpenAI API key not configured. Set OPENAI_API_KEY environment variable."
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Validate API key format (basic validation)
            if not api_key.startswith('sk-'):
                logger.warning(f"API key format may be invalid (doesn't start with 'sk-'): {api_key[:10]}...")
            
            logger.debug(f"Creating OpenAI client with API key: {api_key[:10]}...")
            try:
                self.client = OpenAI(api_key=api_key)
                logger.debug("OpenAI client created successfully")
            except Exception as e:
                error_msg = f"Failed to create OpenAI client: {e}"
                logger.error(error_msg)
                raise ValueError(error_msg) from e
            
            # Test API connectivity with a simple call
            logger.debug("Testing OpenAI API connectivity")
            try:
                # Simple test to validate API key and connectivity
                models = self.client.models.list()
                logger.debug(f"API connectivity test successful, found {len(models.data)} models")
            except Exception as e:
                error_msg = f"OpenAI API connectivity test failed: {e}"
                logger.error(error_msg)
                raise ConnectionError(error_msg) from e
            
            # Model-specific configuration
            logger.debug(f"Getting model configuration for: {model.value}")
            try:
                self.model_config = self._get_model_config(model)
                logger.debug(f"Model configuration loaded: {self.model_config}")
            except Exception as e:
                error_msg = f"Failed to get model configuration: {e}"
                logger.error(error_msg)
                raise ValueError(error_msg) from e
            
            # Language support mapping
            logger.debug("Loading supported languages")
            try:
                self._supported_languages = self._get_supported_languages(model)
                logger.debug(f"Supported languages: {self._supported_languages}")
            except Exception as e:
                logger.warning(f"Failed to load supported languages, using fallback: {e}")
                self._supported_languages = {"en", "es", "fr", "de", "it", "pt"}
            
            # Initialize WebVTT prompt generator for GPT-4o models
            if model in [OpenAIModel.GPT_4O_TRANSCRIBE, OpenAIModel.GPT_4O_MINI_TRANSCRIBE]:
                logger.debug("Initializing WebVTT prompt generator for GPT-4o model")
                try:
                    self.prompt_generator = WebVTTPromptGenerator(
                        include_examples=True,
                        include_diarization=True,
                        include_emotions=True,
                        template=PromptTemplate.SPEAKER_DIARIZATION
                    )
                    logger.debug("WebVTT prompt generator initialized successfully")
                except Exception as e:
                    error_msg = f"Failed to initialize prompt generator: {e}"
                    logger.error(error_msg)
                    raise ValueError(error_msg) from e
            else:
                logger.debug("Using Whisper-1 model, no prompt generator needed")
                self.prompt_generator = None
            
            logger.info(f"OpenAI transcriber initialized successfully: {model.value}")
            
        except Exception as e:
            error_msg = f"OpenAI transcriber initialization failed: {e}"
            logger.error(error_msg)
            # Re-raise with more context
            raise type(e)(error_msg) from e
    
    @property
    def name(self) -> str:
        return f"openai/{self.model_variant.value}"
    
    def _get_model_config(self, model: OpenAIModel) -> Dict[str, Any]:
        """Get model-specific configuration."""
        configs = {
            OpenAIModel.WHISPER_1: {
                "supports_vtt": True,
                "supports_verbose_json": True,
                "supports_streaming": False,
                "supports_prompting": True,
                "supports_all_params": True,
                "default_response_format": "verbose_json"
            },
            OpenAIModel.GPT_4O_TRANSCRIBE: {
                "supports_vtt": False,
                "supports_verbose_json": False,
                "supports_streaming": True,
                "supports_prompting": True,
                "supports_all_params": False,
                "default_response_format": "json"
            },
            OpenAIModel.GPT_4O_MINI_TRANSCRIBE: {
                "supports_vtt": False,
                "supports_verbose_json": False,
                "supports_streaming": True,
                "supports_prompting": True,
                "supports_all_params": False,
                "default_response_format": "json"
            }
        }
        return configs[model]
    
    def _get_supported_languages(self, model: OpenAIModel) -> set[str]:
        """Get supported languages for the model."""
        try:
            capability = get_model_capabilities(model.value)
            return capability.language_support
        except ValueError:
            # Fallback to basic language set
            return {"en", "es", "fr", "de", "it", "pt"}
    
    async def transcribe(
        self, 
        audio_path: Path, 
        language: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> TranscriptionResult:
        """Transcribe audio using OpenAI Audio API."""
        start_time = time.time()
        
        logger.info(f"Transcribing with OpenAI {self.model_variant.value}: {audio_path}")
        
        try:
            # Validate file size and prepare for upload
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            file_size = audio_path.stat().st_size
            if file_size > 25 * 1024 * 1024:  # 25MB limit
                raise ValueError(f"File too large: {file_size / (1024*1024):.1f}MB. OpenAI limit is 25MB.")
            
            # Prepare transcription parameters
            transcription_params = self._prepare_transcription_params(language, context)
            
            # Perform transcription
            response = await self._transcribe_with_openai(audio_path, transcription_params)
            
            # Process response based on model and format
            result = self._process_openai_response(response, transcription_params, start_time)
            
            logger.info(f"OpenAI transcription completed in {result.processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"OpenAI transcription failed for {audio_path}: {e}")
            raise
    
    def _prepare_transcription_params(
        self, 
        language: Optional[str], 
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Prepare parameters for OpenAI transcription."""
        params = {
            "model": self.model_variant.value
        }
        
        # Response format selection
        if self.model_config["supports_vtt"] and self.config.output.default_format == "webvtt":
            params["response_format"] = "vtt"
        elif self.model_config["supports_verbose_json"]:
            params["response_format"] = "verbose_json"
        else:
            params["response_format"] = self.model_config["default_response_format"]
        
        # Language parameter
        if language and language != "auto" and language in self._supported_languages:
            params["language"] = language
        
        # Model-specific parameters
        if self.model_config["supports_all_params"]:
            # Whisper-1 supports full parameter set
            if context:
                params["timestamp_granularities"] = ["word", "segment"]
            
        # Prompting for GPT-4o models (using prompt parameter)
        if (self.model_config["supports_prompting"] and 
            self.model_variant in [OpenAIModel.GPT_4O_TRANSCRIBE, OpenAIModel.GPT_4O_MINI_TRANSCRIBE]):
            
            # Generate WebVTT-format prompt for GPT-4o models
            if self.prompt_generator:
                prompt = self.prompt_generator.generate_webvtt_prompt(language, context)
                params["prompt"] = prompt[:224]  # OpenAI prompt limit
        
        # Whisper-1 specific prompting (different approach)
        elif (self.model_variant == OpenAIModel.WHISPER_1 and context):
            # For Whisper-1, use prompt for context and proper noun guidance
            context_prompt = self._generate_whisper_prompt(context)
            if context_prompt:
                params["prompt"] = context_prompt[:224]  # 224 token limit
        
        return params
    
    async def _transcribe_with_openai(
        self, 
        audio_path: Path, 
        params: Dict[str, Any]
    ) -> Any:
        """Perform transcription using OpenAI Audio API."""
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
            logger.error(f"OpenAI API call failed: {e}")
            raise
    
    def _process_openai_response(
        self, 
        response: Any, 
        params: Dict[str, Any], 
        start_time: float
    ) -> TranscriptionResult:
        """Process OpenAI API response into TranscriptionResult."""
        processing_time = time.time() - start_time
        response_format = params.get("response_format", "json")
        
        # Handle different response formats
        if response_format == "vtt":
            # Direct WebVTT response from Whisper-1
            webvtt_content = response
            transcribed_text, word_timestamps = self._parse_webvtt_response(webvtt_content)
            metadata = {
                "format": "webvtt",
                "webvtt_content": webvtt_content,
                "native_timing": True
            }
            
        elif response_format == "verbose_json":
            # Detailed JSON response with segments and words
            transcribed_text = response.text
            word_timestamps = self._extract_word_timestamps(response)
            metadata = {
                "format": "verbose_json",
                "segments": getattr(response, 'segments', []),
                "native_timing": True
            }
            
        else:
            # Basic JSON/text response
            if hasattr(response, 'text'):
                transcribed_text = response.text
            else:
                transcribed_text = str(response)
            
            word_timestamps = []
            metadata = {
                "format": response_format,
                "native_timing": False
            }
        
        # Common metadata
        metadata.update({
            "engine": "openai",
            "model": self.model_variant.value,
            "response_format": response_format,
            "processing_time": processing_time
        })
        
        return TranscriptionResult(
            text=transcribed_text,
            confidence=0.90,  # OpenAI doesn't provide confidence scores
            word_timestamps=word_timestamps,
            processing_time=processing_time,
            model_name=self.model_variant.value,
            language=params.get("language", "auto"),
            metadata=metadata,
            raw_response={
                "type": "openai.Transcription",
                "text": getattr(response, 'text', transcribed_text),
                "language": getattr(response, 'language', None),
                "duration": getattr(response, 'duration', None),
                "words": getattr(response, 'words', []),
                "segments": getattr(response, 'segments', []),
                "response_format": response_format,
                "model": self.model_variant.value,
                "original_response": str(response) if hasattr(response, '__dict__') else str(response)
            }
        )
    
    def _parse_webvtt_response(self, webvtt_content: str) -> tuple[str, List[Dict[str, Any]]]:
        """Parse WebVTT content from Whisper-1 vtt format response."""
        if not webvtt_content:
            return "", []
        
        # Extract plain text and timestamps
        text_parts = []
        word_timestamps = []
        
        # Split into cues
        cue_pattern = r'\n(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})\n(.*?)(?=\n\n|\n\d{2}:|\Z)'
        matches = re.findall(cue_pattern, webvtt_content, re.DOTALL)
        
        for start_str, end_str, text in matches:
            # Convert timestamps to seconds
            start_seconds = self._timestamp_to_seconds(start_str)
            end_seconds = self._timestamp_to_seconds(end_str)
            
            # Clean text
            clean_text = re.sub(r'<[^>]*>', '', text).strip()
            text_parts.append(clean_text)
            
            # Split text into words for word timestamps
            words = clean_text.split()
            if words:
                word_duration = (end_seconds - start_seconds) / len(words)
                for i, word in enumerate(words):
                    word_start = start_seconds + (i * word_duration)
                    word_end = start_seconds + ((i + 1) * word_duration)
                    
                    word_timestamps.append({
                        "word": word,
                        "start": word_start,
                        "end": word_end,
                        "confidence": 0.90
                    })
        
        full_text = ' '.join(text_parts)
        return full_text, word_timestamps
    
    def _timestamp_to_seconds(self, timestamp_str: str) -> float:
        """Convert HH:MM:SS.mmm timestamp to seconds."""
        try:
            parts = timestamp_str.split(':')
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds_parts = parts[2].split('.')
            seconds = int(seconds_parts[0])
            milliseconds = int(seconds_parts[1])
            
            total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000
            return total_seconds
        except (ValueError, IndexError):
            return 0.0
    
    def _extract_word_timestamps(self, response: Any) -> List[Dict[str, Any]]:
        """Extract word-level timestamps from verbose_json response."""
        word_timestamps = []
        
        if hasattr(response, 'segments'):
            for segment in response.segments:
                if hasattr(segment, 'words'):
                    for word_data in segment.words:
                        word_timestamps.append({
                            "word": word_data.word,
                            "start": word_data.start,
                            "end": word_data.end,
                            "confidence": getattr(word_data, 'confidence', 0.90)
                        })
        
        return word_timestamps
    
    def _generate_whisper_prompt(self, context: Dict[str, Any]) -> Optional[str]:
        """Generate context-aware prompt for Whisper-1 model."""
        prompt_parts = []
        
        # Add video title for proper noun recognition
        if context.get('video_title'):
            title_words = context['video_title'].split()[:10]  # Limit words
            prompt_parts.extend(title_words)
        
        # Add known terminology from description
        if context.get('video_description'):
            desc = context['video_description'][:100]  # Limit length
            # Extract proper nouns and technical terms
            # Simple approach: capitalize words, acronyms
            proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', desc)
            acronyms = re.findall(r'\b[A-Z]{2,}\b', desc)
            prompt_parts.extend(proper_nouns[:5])
            prompt_parts.extend(acronyms[:3])
        
        if prompt_parts:
            return ', '.join(prompt_parts[:20])  # Stay within token limits
        
        return None
    
    def estimate_cost(self, duration_seconds: float) -> float:
        """Estimate processing cost in USD."""
        try:
            capability = get_model_capabilities(self.model_variant.value)
            if capability.cost_per_minute:
                duration_minutes = duration_seconds / 60
                return capability.cost_per_minute * duration_minutes
        except ValueError:
            pass
        return 0.0
    
    def get_supported_languages(self) -> List[str]:
        """Return list of supported language codes."""
        return sorted(list(self._supported_languages))
    
    # Streaming support for GPT-4o models (optional)
    async def transcribe_streaming(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[TranscriptionResult, None]:
        """Stream transcription results for GPT-4o models."""
        if not self.model_config["supports_streaming"]:
            # Fall back to regular transcription
            result = await self.transcribe(audio_path, language, context)
            yield result
            return
        
        # For now, fall back to regular transcription
        # Full streaming implementation would require OpenAI streaming API integration
        logger.info("Streaming not yet implemented, falling back to batch transcription")
        result = await self.transcribe(audio_path, language, context)
        yield result