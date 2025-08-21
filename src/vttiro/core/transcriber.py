#!/usr/bin/env python3
# this_file: src/vttiro/core/transcriber.py
"""Main Transcriber class that orchestrates the entire transcription pipeline."""

from pathlib import Path
from typing import Optional, Union, Dict, Any, List
import asyncio
import uuid

try:
    from loguru import logger
except ImportError:
    import logging as logger

from vttiro.core.config import VttiroConfig, TranscriptionResult
from vttiro.config.enhanced import EnhancedVttiroConfig
# Hot reload removed for simplification
# Monitoring removed for simplification
from vttiro.core.transcription import TranscriptionEngine, TranscriptionEnsemble, MockTranscriptionEngine
from vttiro.processing.video import VideoProcessor, AudioChunk

# Import enhanced error handling and resilience framework
from vttiro.utils import (
    VttiroError,
    ConfigurationError,
    ValidationError,
    TranscriptionError,
    ProcessingError,
    ModelError,
    OutputGenerationError,
    create_api_resilience_manager,
    create_error,
)

# Import AI model transcribers (with fallback to None if dependencies missing)
try:
    from vttiro.models import GeminiTranscriber, AssemblyAITranscriber, DeepgramTranscriber
except ImportError:
    GeminiTranscriber = None
    AssemblyAITranscriber = None 
    DeepgramTranscriber = None


class Transcriber:
    """Main transcriber class that orchestrates the entire transcription pipeline.
    
    Enhanced with comprehensive error handling, circuit breaker protection,
    and resilient retry mechanisms for production-grade reliability.
    """
    
    def __init__(self, config: Optional[Union[VttiroConfig, EnhancedVttiroConfig]] = None):
        """Initialize transcriber with configuration.
        
        Args:
            config: Configuration object (VttiroConfig or EnhancedVttiroConfig).
                   Defaults to loading enhanced configuration from default location.
            
        Raises:
            ConfigurationError: If configuration is invalid or missing required settings
        """
        try:
            # Prioritize enhanced configuration system
            if config is None:
                # Load enhanced configuration directly (hot-reload removed for simplification)
                self.enhanced_config = EnhancedVttiroConfig()
                # Create legacy config for backward compatibility
                self.config = self._create_legacy_config_from_enhanced(self.enhanced_config)
            elif isinstance(config, EnhancedVttiroConfig):
                self.enhanced_config = config
                self.config = self._create_legacy_config_from_enhanced(config)
            else:
                # Legacy VttiroConfig provided
                self.config = config
                self.config.update_from_env()
                self.enhanced_config = None
                logger.warning(
                    "Using legacy VttiroConfig. Consider upgrading to EnhancedVttiroConfig for improved features."
                )
        except Exception as e:
            raise ConfigurationError(
                "Failed to load or validate configuration",
                cause=e,
                context={"config_provided": config is not None, "config_type": type(config).__name__ if config else "None"}
            )
        
        # Initialize resilience managers for different operations
        # Use enhanced config settings if available
        if self.enhanced_config and self.enhanced_config.models.circuit_breaker_enabled:
            self.api_resilience = create_api_resilience_manager(
                "transcription_api",
                failure_threshold=self.enhanced_config.models.failure_threshold,
                recovery_timeout=self.enhanced_config.models.recovery_timeout,
                max_retries=self.enhanced_config.api.max_retries,
                retry_delay=self.enhanced_config.api.retry_delay
            )
            self.processing_resilience = create_api_resilience_manager(
                "processing",
                failure_threshold=self.enhanced_config.models.failure_threshold,
                recovery_timeout=self.enhanced_config.models.recovery_timeout,
                max_retries=self.enhanced_config.api.max_retries,
                retry_delay=self.enhanced_config.api.retry_delay
            )
        else:
            # Use default settings for legacy config
            self.api_resilience = create_api_resilience_manager("transcription_api")
            self.processing_resilience = create_api_resilience_manager("processing")
        
        try:
            # Initialize video processor with error handling
            # Pass enhanced config if available for better features
            processor_config = self.enhanced_config if self.enhanced_config else self.config
            self.video_processor = VideoProcessor(processor_config)
        except Exception as e:
            raise ConfigurationError(
                "Failed to initialize video processor",
                cause=e,
                context={
                    "config_type": type(processor_config).__name__,
                    "enhanced_config_available": self.enhanced_config is not None
                }
            )
        
        # Initialize transcription engines with comprehensive error handling
        self.engines: List[TranscriptionEngine] = []
        self._initialize_engines()
        
        # Create ensemble with error handling
        try:
            self.ensemble = TranscriptionEnsemble(self.engines, self.config)
        except Exception as e:
            raise ConfigurationError(
                "Failed to initialize transcription ensemble",
                cause=e,
                context={"available_engines": len(self.engines)}
            )
        
        # Initialize monitoring integration
        # Monitoring removed for simplification
        
        # Log configuration summary for debugging and verification
        self.log_configuration_summary()
        
    def _initialize_engines(self) -> None:
        """Initialize available transcription engines based on configuration and API keys.
        
        Uses comprehensive error handling and graceful degradation to ensure
        at least one engine is always available for transcription.
        
        Raises:
            ConfigurationError: If no engines can be initialized
        """
        self.engines = []
        initialization_errors = []
        
        # Try to initialize AI model engines based on available API keys and config
        
        # 1. Gemini 2.0 Flash (Priority 1 - Best context understanding)
        if (hasattr(self.config.transcription, 'gemini_api_key') and 
            self.config.transcription.gemini_api_key and 
            GeminiTranscriber and 
            self.config.transcription.preferred_model in ["auto", "gemini"]):
            try:
                gemini_engine = GeminiTranscriber(self.config)
                self.engines.append(gemini_engine)
                logger.info("Initialized Gemini 2.0 Flash transcription engine")
            except Exception as e:
                error = ModelError(
                    "Failed to initialize Gemini engine",
                    model_name="gemini-2.0-flash",
                    cause=e,
                    context={"api_key_present": bool(self.config.transcription.gemini_api_key)}
                )
                initialization_errors.append(error)
                logger.warning(f"Failed to initialize Gemini engine: {error}")
                
        # 2. AssemblyAI Universal-2 (Priority 2 - Maximum accuracy)
        if (hasattr(self.config.transcription, 'assemblyai_api_key') and 
            self.config.transcription.assemblyai_api_key and 
            AssemblyAITranscriber and 
            self.config.transcription.preferred_model in ["auto", "assemblyai"]):
            try:
                assemblyai_engine = AssemblyAITranscriber(self.config)
                self.engines.append(assemblyai_engine)
                logger.info("Initialized AssemblyAI Universal-2 transcription engine")
            except Exception as e:
                error = ModelError(
                    "Failed to initialize AssemblyAI engine",
                    model_name="assemblyai-universal-2",
                    cause=e,
                    context={"api_key_present": bool(self.config.transcription.assemblyai_api_key)}
                )
                initialization_errors.append(error)
                logger.warning(f"Failed to initialize AssemblyAI engine: {error}")
                
        # 3. Deepgram Nova-3 (Priority 3 - Speed optimized)
        if (hasattr(self.config.transcription, 'deepgram_api_key') and 
            self.config.transcription.deepgram_api_key and 
            DeepgramTranscriber and 
            self.config.transcription.preferred_model in ["auto", "deepgram"]):
            try:
                deepgram_engine = DeepgramTranscriber(self.config)
                self.engines.append(deepgram_engine)
                logger.info("Initialized Deepgram Nova-3 transcription engine")
            except Exception as e:
                error = ModelError(
                    "Failed to initialize Deepgram engine",
                    model_name="deepgram-nova-3",
                    cause=e,
                    context={"api_key_present": bool(self.config.transcription.deepgram_api_key)}
                )
                initialization_errors.append(error)
                logger.warning(f"Failed to initialize Deepgram engine: {error}")
                
        # Fallback to mock engine if no real engines available
        if not self.engines:
            try:
                mock_engine = MockTranscriptionEngine(self.config)
                self.engines.append(mock_engine)
                logger.warning("No AI transcription engines available, using mock engine for testing")
            except Exception as e:
                raise ConfigurationError(
                    "Failed to initialize any transcription engines, including fallback mock engine",
                    cause=e,
                    context={
                        "initialization_errors": [error.to_dict() for error in initialization_errors],
                        "available_models": [cls.__name__ for cls in [GeminiTranscriber, AssemblyAITranscriber, DeepgramTranscriber] if cls is not None]
                    }
                )
            
        logger.info(
            f"Successfully initialized {len(self.engines)} transcription engines: {[e.name for e in self.engines]}",
            extra={"initialization_errors_count": len(initialization_errors)}
        )
        
    def _create_legacy_config_from_enhanced(self, enhanced_config: EnhancedVttiroConfig) -> VttiroConfig:
        """Create a legacy VttiroConfig from EnhancedVttiroConfig for backward compatibility.
        
        Args:
            enhanced_config: Enhanced configuration object
            
        Returns:
            Legacy VttiroConfig object
        """
        from vttiro.core.config import (
            TranscriptionConfig, ProcessingConfig, DiarizationConfig,
            EmotionConfig, OutputConfig, YouTubeConfig
        )
        
        # Map enhanced config to legacy config structure
        transcription_config = TranscriptionConfig(
            default_model=enhanced_config.models.default_provider,
            preferred_model=enhanced_config.models.default_provider,
            ensemble_enabled=len(enhanced_config.models.fallback_providers) > 0,
            confidence_threshold=0.8,  # Default value
            language=None,  # Will be set per request
            gemini_api_key=enhanced_config.api.gemini_api_key,
            assemblyai_api_key=enhanced_config.api.assemblyai_api_key,
            deepgram_api_key=enhanced_config.api.deepgram_api_key
        )
        
        processing_config = ProcessingConfig(
            chunk_duration=enhanced_config.processing.chunk_duration,
            overlap_duration=enhanced_config.processing.overlap_duration,
            max_duration=enhanced_config.processing.max_duration,
            sample_rate=enhanced_config.processing.sample_rate,
            prefer_integer_seconds=enhanced_config.processing.prefer_integer_seconds,
            energy_threshold_percentile=enhanced_config.processing.energy_threshold_percentile,
            min_energy_window=enhanced_config.processing.min_energy_window
        )
        
        diarization_config = DiarizationConfig(
            enabled=enhanced_config.features.speaker_diarization,
            min_speakers=None,
            max_speakers=10,
            threshold=0.7,
            huggingface_token=getattr(enhanced_config.api, 'huggingface_token', None)
        )
        
        emotion_config = EmotionConfig(
            enabled=enhanced_config.features.emotion_detection,
            audio_weight=0.6,
            text_weight=0.4,
            confidence_threshold=0.5,
            cultural_adaptation=True
        )
        
        output_config = OutputConfig(
            default_format=enhanced_config.output.default_format,
            max_chars_per_line=enhanced_config.output.max_chars_per_line,
            max_lines_per_cue=enhanced_config.output.max_lines_per_cue,
            max_cue_duration=enhanced_config.output.max_cue_duration,
            reading_speed_wpm=enhanced_config.output.reading_speed_wpm,
            wcag_compliance=enhanced_config.output.wcag_compliance,
            include_sound_descriptions=enhanced_config.output.include_sound_descriptions
        )
        
        youtube_config = YouTubeConfig(
            enabled=enhanced_config.features.youtube_integration,
            client_secrets_file=None,
            quota_limit=10000,
            auto_upload=enhanced_config.features.auto_upload_subtitles
        )
        
        # Create legacy config
        legacy_config = VttiroConfig(
            transcription=transcription_config,
            processing=processing_config,
            diarization=diarization_config,
            emotion=emotion_config,
            output=output_config,
            youtube=youtube_config,
            verbose=enhanced_config.monitoring.log_level == "DEBUG",
            temp_dir=enhanced_config.processing.temp_directory
        )
        
        return legacy_config
    
    def get_config_value(self, path: str, default=None):
        """Get configuration value with fallback to legacy config.
        
        Args:
            path: Dot-separated path to configuration value (e.g., 'api.timeout_seconds')
            default: Default value if not found
            
        Returns:
            Configuration value or default
        """
        # Try enhanced config first
        if self.enhanced_config:
            try:
                value = self.enhanced_config
                for part in path.split('.'):
                    value = getattr(value, part)
                return value
            except AttributeError:
                pass
        
        # Fallback to legacy config
        try:
            value = self.config
            for part in path.split('.'):
                value = getattr(value, part)
            return value
        except AttributeError:
            return default
    
    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a feature is enabled in configuration.
        
        Args:
            feature: Feature name
            
        Returns:
            True if feature is enabled, False otherwise
        """
        if self.enhanced_config:
            return getattr(self.enhanced_config.features, feature, False)
        
        # Legacy config feature mapping
        feature_mapping = {
            'speaker_diarization': 'diarization.enabled',
            'emotion_detection': 'emotion.enabled',
            'youtube_integration': 'youtube.enabled'
        }
        
        if feature in feature_mapping:
            return self.get_config_value(feature_mapping[feature], False)
        
        return False
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate current configuration and return health status.
        
        Returns:
            Dictionary with validation results and recommendations
        """
        validation_results = {
            'status': 'healthy',
            'warnings': [],
            'errors': [],
            'recommendations': [],
            'config_type': 'enhanced' if self.enhanced_config else 'legacy',
            'features_enabled': {}
        }
        
        try:
            # Check API key availability
            api_keys = {}
            if self.enhanced_config:
                api_keys = {
                    'gemini': bool(self.enhanced_config.api.gemini_api_key),
                    'assemblyai': bool(self.enhanced_config.api.assemblyai_api_key),
                    'deepgram': bool(self.enhanced_config.api.deepgram_api_key),
                }
            else:
                api_keys = {
                    'gemini': bool(self.config.transcription.gemini_api_key),
                    'assemblyai': bool(self.config.transcription.assemblyai_api_key),
                    'deepgram': bool(self.config.transcription.deepgram_api_key),
                }
            
            available_keys = sum(api_keys.values())
            if available_keys == 0:
                validation_results['errors'].append("No API keys configured")
                validation_results['status'] = 'error'
            elif available_keys == 1:
                validation_results['warnings'].append("Only one API key configured - no fallback available")
                validation_results['status'] = 'warning'
            
            # Check feature configuration
            if self.enhanced_config:
                validation_results['features_enabled'] = {
                    'speaker_diarization': self.enhanced_config.features.speaker_diarization,
                    'emotion_detection': self.enhanced_config.features.emotion_detection,
                    'auto_segmentation': self.enhanced_config.features.auto_segmentation,
                    'context_aware_prompting': self.enhanced_config.features.context_aware_prompting,
                    'youtube_integration': self.enhanced_config.features.youtube_integration,
                    'circuit_breaker': self.enhanced_config.models.circuit_breaker_enabled,
                    'monitoring': self.enhanced_config.monitoring.enabled,
                    'hot_reload': self.enhanced_config.monitoring.hot_reload_enabled
                }
                
                # Recommendations for enhanced config
                
                if not self.enhanced_config.models.circuit_breaker_enabled:
                    validation_results['recommendations'].append("Enable circuit breaker for better resilience")
                
                if self.enhanced_config.environment.value == "production" and not self.enhanced_config.security.encryption_enabled:
                    validation_results['errors'].append("Encryption must be enabled in production")
                    validation_results['status'] = 'error'
                    
            else:
                # Legacy config validation
                validation_results['recommendations'].append("Upgrade to EnhancedVttiroConfig for improved features")
                validation_results['features_enabled'] = {
                    'speaker_diarization': self.config.diarization.enabled,
                    'emotion_detection': self.config.emotion.enabled,
                    'youtube_integration': self.config.youtube.enabled
                }
            
            # Engine validation
            if len(self.engines) == 0:
                validation_results['errors'].append("No transcription engines initialized")
                validation_results['status'] = 'error'
            elif len(self.engines) == 1 and self.engines[0].name == "mock":
                validation_results['warnings'].append("Only mock engine available - real transcription not possible")
                validation_results['status'] = 'warning'
            
        except Exception as e:
            validation_results['errors'].append(f"Configuration validation failed: {e}")
            validation_results['status'] = 'error'
        
        return validation_results
    
    def log_configuration_summary(self) -> None:
        """Log a summary of current configuration for debugging."""
        validation = self.validate_configuration()
        
        logger.info(
            f"Transcriber configuration: {validation['config_type']} config, status: {validation['status']}",
            extra={
                "config_type": validation['config_type'],
                "status": validation['status'],
                "engines_count": len(self.engines),
                "features_enabled": validation['features_enabled']
            }
        )
        
        if validation['warnings']:
            for warning in validation['warnings']:
                logger.warning(f"Configuration warning: {warning}")
        
        if validation['errors']:
            for error in validation['errors']:
                logger.error(f"Configuration error: {error}")
        
        if validation['recommendations']:
            for rec in validation['recommendations']:
                logger.info(f"Configuration recommendation: {rec}")
        
    async def transcribe(
        self,
        source: Union[str, Path],
        output: Optional[Union[str, Path]] = None,
        language: Optional[str] = None,
        **kwargs
    ) -> str:
        """Transcribe video or audio source to WebVTT format.
        
        Enhanced with comprehensive error handling, correlation tracking,
        resilient operation, and integrated monitoring for production-grade reliability.
        
        Args:
            source: Video URL (YouTube, etc.) or local file path
            output: Output file path (optional, auto-generated if not provided)
            language: Language code (optional, auto-detected if not provided)
            **kwargs: Additional parameters for transcription
            
        Returns:
            Path to generated subtitle file
            
        Raises:
            ValidationError: If source is invalid or unsupported
            ProcessingError: If video/audio processing fails
            TranscriptionError: If transcription pipeline fails
            OutputGenerationError: If WebVTT generation fails
        """
        # Generate correlation ID for request tracking
        correlation_id = str(uuid.uuid4())
        
        # Get primary provider for metrics (use first available engine)
        primary_provider = self.engines[0].name if self.engines else "unknown"
        
        logger.info(
            f"Starting transcription of: {source}",
            extra={"correlation_id": correlation_id, "source": str(source)}
        )
        
        # Simplified transcription without monitoring complexity
        return await self._perform_transcription(
            source=source,
            output=output,
            language=language,
            correlation_id=correlation_id,
            **kwargs
        )
    
    async def _perform_transcription(
        self,
        source: Union[str, Path],
        output: Optional[Union[str, Path]],
        language: Optional[str],
        correlation_id: str,
        **kwargs
    ) -> str:
        """Perform the actual transcription operation.
        
        Args:
            source: Video source
            output: Output path
            language: Language code
            correlation_id: Correlation ID for tracking
            **kwargs: Additional parameters
            
        Returns:
            Path to generated subtitle file
        """
        try:
            # Step 1: Video processing and audio extraction with segmentation
            logger.info(
                "Processing video source and extracting audio segments",
                extra={"correlation_id": correlation_id}
            )
            
            # Trace video processing
            with self.tracer.trace_processing_operation(
                operation_name="video_processing",
                input_type="video",
                correlation_id=correlation_id
            ) as processing_span:
                
                # Use processing resilience for video operations
                async def process_video():
                    return await self.video_processor.process_source(
                        source,
                        extract_audio=True,
                        segment_audio=True
                    )
                
                try:
                    video_result = await self.processing_resilience.execute(
                        process_video,
                        correlation_id=correlation_id
                    )
                    
                    # Add processing metrics to trace
                    if hasattr(video_result, 'metadata') and video_result.metadata:
                        processing_span.set_attribute("vttiro.audio_duration", video_result.metadata.duration_seconds)
                        processing_span.set_attribute("vttiro.segments_created", len(video_result.segments))
                        
                except Exception as e:
                    raise ProcessingError(
                        f"Failed to process video source: {source}",
                        correlation_id=correlation_id,
                        file_path=str(source),
                        processing_stage="video_processing",
                        cause=e
                    )
            
            # Extract metadata and segments
            metadata = video_result.metadata
            audio_chunks = video_result.segments
            
            # Update metrics context with audio duration
            if metrics_ctx and metadata:
                metrics_ctx.set_audio_duration(metadata.duration_seconds)
            
            logger.info(
                f"Video processed: {len(audio_chunks)} audio segments created",
                extra={
                    "correlation_id": correlation_id,
                    "segments_count": len(audio_chunks),
                    "duration": metadata.duration_seconds
                }
            )
            logger.info(f"Total duration: {metadata.duration_seconds:.1f}s")
            
            # Step 2: Transcribe each audio chunk with resilience
            logger.info(
                "Starting transcription of audio segments",
                extra={"correlation_id": correlation_id}
            )
            transcription_results = []
            
            for i, chunk in enumerate(audio_chunks):
                segment_correlation_id = f"{correlation_id}-seg-{i+1:03d}"
                
                logger.info(
                    f"Transcribing segment {i+1}/{len(audio_chunks)} ({chunk.start_time:.1f}s-{chunk.end_time:.1f}s)",
                    extra={
                        "correlation_id": segment_correlation_id,
                        "segment_index": i+1,
                        "total_segments": len(audio_chunks)
                    }
                )
                
                # Prepare context from metadata for factual prompting
                context = {
                    'video_title': metadata.title,
                    'video_description': metadata.description,
                    'video_uploader': metadata.uploader,
                    'segment_index': i,
                    'total_segments': len(audio_chunks),
                    'start_time': chunk.start_time,
                    'end_time': chunk.end_time,
                    'correlation_id': segment_correlation_id,
                    **kwargs
                }
                
                # Transcribe the chunk with API resilience
                async def transcribe_chunk():
                    return await self.ensemble.transcribe(
                        chunk.audio_file,
                        language=language,
                        context=context
                    )
                
                try:
                    chunk_result = await self.api_resilience.execute(
                        transcribe_chunk,
                        correlation_id=segment_correlation_id
                    )
                except Exception as e:
                    # Log error but continue with other segments
                    error = TranscriptionError(
                        f"Failed to transcribe segment {i+1}/{len(audio_chunks)}",
                        correlation_id=segment_correlation_id,
                        cause=e,
                        context={
                            "segment_index": i+1,
                            "total_segments": len(audio_chunks),
                            "start_time": chunk.start_time,
                            "end_time": chunk.end_time,
                            "audio_file": str(chunk.audio_file)
                        }
                    )
                    logger.error(f"Segment transcription failed: {error}")
                    
                    # Create empty result for failed segment to maintain timeline
                    from vttiro.core.config import TranscriptionResult
                    chunk_result = TranscriptionResult(
                        text="[Transcription failed for this segment]",
                        confidence=0.0,
                        language=language or "unknown",
                        start_time=chunk.start_time,
                        end_time=chunk.end_time,
                        metadata={"error": str(error), "correlation_id": segment_correlation_id}
                    )
                
                # Adjust timestamps to global timeline
                chunk_result.start_time = chunk.start_time
                chunk_result.end_time = chunk.end_time
                transcription_results.append(chunk_result)
                
                logger.info(
                    f"Segment {i+1} transcribed: {len(chunk_result.text)} characters",
                    extra={
                        "correlation_id": segment_correlation_id,
                        "characters": len(chunk_result.text),
                        "confidence": getattr(chunk_result, 'confidence', 0.0)
                    }
                )
            
            # Step 3: Combine results and generate WebVTT
            logger.info(
                "Generating WebVTT from transcription results",
                extra={"correlation_id": correlation_id}
            )
            
            # Generate output path
            try:
                if output is None:
                    if isinstance(source, str) and source.startswith(('http://', 'https://')):
                        # Use video title if available, otherwise default name
                        safe_title = self._sanitize_filename(metadata.title) if metadata.title else "transcript"
                        output = f"{safe_title}.vtt"
                    else:
                        source_path = Path(source)
                        output = f"{source_path.stem}.vtt"
                        
                output_path = Path(output)
                
                # Generate comprehensive WebVTT content
                webvtt_content = self._generate_comprehensive_webvtt(
                    transcription_results, 
                    metadata,
                    include_metadata=kwargs.get('include_metadata', True)
                )
                
                # Write to file with error handling
                try:
                    output_path.write_text(webvtt_content, encoding='utf-8')
                except Exception as e:
                    raise OutputGenerationError(
                        f"Failed to write WebVTT file: {output_path}",
                        output_format="webvtt",
                        output_path=str(output_path),
                        correlation_id=correlation_id,
                        cause=e
                    )
                
                # Basic completion logging without metrics complexity
                total_characters = sum(len(r.text) for r in transcription_results)
                
                logger.info(
                    f"Transcription completed: {output_path}",
                    extra={
                        "correlation_id": correlation_id,
                        "output_path": str(output_path),
                        "total_characters": total_characters,
                        "segments_processed": len(transcription_results)
                    }
                )
                logger.info(f"Total transcription length: {total_characters} characters")
                
                return str(output_path)
                
            except OutputGenerationError:
                # Re-raise output generation errors
                raise
            except Exception as e:
                raise OutputGenerationError(
                    "Failed to generate WebVTT output",
                    output_format="webvtt",
                    correlation_id=correlation_id,
                    cause=e,
                    context={
                        "source": str(source),
                        "segments_count": len(transcription_results)
                    }
                )
            
        except (ValidationError, ProcessingError, TranscriptionError, OutputGenerationError):
            # Re-raise known vttiro errors with correlation context
            raise
        except Exception as e:
            # Wrap unexpected errors
            error = create_error(
                TranscriptionError,
                f"Unexpected error during transcription of {source}",
                correlation_id=correlation_id,
                context={
                    "source": str(source),
                    "error_type": type(e).__name__
                },
                cause=e
            )
            logger.error(f"Transcription failed: {error}")
            raise error
        
    def _generate_webvtt(self, result: TranscriptionResult) -> str:
        """Generate WebVTT content from transcription result.
        
        Args:
            result: Transcription result
            
        Returns:
            WebVTT formatted string
        """
        # Basic WebVTT generation
        # TODO: Implement proper WebVTT generation with:
        # - Precise timing from word_timestamps
        # - Speaker labels (if available)
        # - Emotion indicators (if available)
        # - Proper line breaking and formatting
        
        webvtt_lines = ["WEBVTT", ""]
        
        # For now, create a simple cue
        webvtt_lines.extend([
            "00:00.000 --> 00:05.000",
            result.text,
            ""
        ])
        
        # Add metadata as comments
        if result.metadata:
            webvtt_lines.insert(1, f"NOTE Model: {result.model_name}")
            webvtt_lines.insert(2, f"NOTE Confidence: {result.confidence:.2f}")
            webvtt_lines.insert(3, "")
            
        return "\n".join(webvtt_lines)
    
    def _generate_comprehensive_webvtt(
        self, 
        results: List[TranscriptionResult], 
        metadata,
        include_metadata: bool = True
    ) -> str:
        """Generate comprehensive WebVTT content from multiple transcription results.
        
        Args:
            results: List of transcription results from segments
            metadata: Video metadata
            include_metadata: Whether to include metadata in WebVTT
            
        Returns:
            WebVTT formatted string
        """
        webvtt_lines = ["WEBVTT"]
        
        # Add metadata as NOTE comments if enabled
        if include_metadata and metadata:
            webvtt_lines.extend([
                "",
                f"NOTE Title: {metadata.title or 'Unknown'}",
                f"NOTE Duration: {metadata.duration_seconds:.1f}s",
                f"NOTE Uploader: {metadata.uploader or 'Unknown'}",
                f"NOTE Processed: {len(results)} segments"
            ])
            
        webvtt_lines.append("")
        
        # Generate cues from transcription results
        for i, result in enumerate(results):
            # Format timestamps
            start_timestamp = self._seconds_to_webvtt_time(result.start_time)
            end_timestamp = self._seconds_to_webvtt_time(result.end_time)
            
            # Add cue
            webvtt_lines.extend([
                f"{start_timestamp} --> {end_timestamp}",
                result.text.strip(),
                ""
            ])
            
        return "\n".join(webvtt_lines)
    
    def _seconds_to_webvtt_time(self, seconds: float) -> str:
        """Convert seconds to WebVTT timestamp format (HH:MM:SS.mmm).
        
        Args:
            seconds: Time in seconds
            
        Returns:
            WebVTT formatted timestamp
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename by removing invalid characters.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename safe for filesystem
        """
        import re
        
        # Remove invalid characters and replace with underscores
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Remove multiple consecutive underscores
        sanitized = re.sub(r'_{2,}', '_', sanitized)
        
        # Trim and limit length
        sanitized = sanitized.strip('_').strip()
        if len(sanitized) > 100:
            sanitized = sanitized[:100].rstrip('_')
            
        return sanitized or "transcript"
        
    def get_available_models(self) -> List[str]:
        """Get list of available transcription models."""
        return [engine.name for engine in self.engines]
        
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages across all engines."""
        languages = set()
        for engine in self.engines:
            languages.update(engine.get_supported_languages())
        return sorted(languages)
        
    def estimate_cost(self, source: Union[str, Path], duration: Optional[float] = None) -> float:
        """Estimate transcription cost.
        
        Args:
            source: Video source
            duration: Duration in seconds (optional, will be detected if not provided)
            
        Returns:
            Estimated cost in USD
        """
        # TODO: Detect duration from source if not provided
        if duration is None:
            duration = 300  # Default 5 minutes for estimation
            
        return self.ensemble.estimate_cost(duration)
        
    async def batch_transcribe(
        self,
        sources: List[Union[str, Path]],
        output_dir: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> List[str]:
        """Batch transcribe multiple sources with enhanced error handling.
        
        Enhanced with comprehensive error tracking, partial failure tolerance,
        and detailed progress reporting for production batch operations.
        
        Args:
            sources: List of video sources
            output_dir: Output directory (optional)
            **kwargs: Additional parameters for transcription
            
        Returns:
            List of generated subtitle file paths (successful transcriptions only)
            
        Raises:
            ValidationError: If sources list is empty or invalid
            ProcessingError: If output directory cannot be created
        """
        # Generate correlation ID for batch operation
        batch_correlation_id = str(uuid.uuid4())
        
        if not sources:
            raise ValidationError(
                "Sources list cannot be empty",
                correlation_id=batch_correlation_id,
                context={"sources_count": 0}
            )
        
        logger.info(
            f"Starting batch transcription of {len(sources)} sources",
            extra={
                "correlation_id": batch_correlation_id,
                "sources_count": len(sources),
                "output_dir": str(output_dir) if output_dir else "current_directory"
            }
        )
        
        # Setup output directory with error handling
        try:
            output_dir = Path(output_dir) if output_dir else Path.cwd()
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ProcessingError(
                f"Failed to create output directory: {output_dir}",
                correlation_id=batch_correlation_id,
                file_path=str(output_dir),
                processing_stage="directory_creation",
                cause=e
            )
        
        results = []
        failed_sources = []
        
        for i, source in enumerate(sources):
            source_correlation_id = f"{batch_correlation_id}-src-{i+1:03d}"
            
            logger.info(
                f"Processing {i+1}/{len(sources)}: {source}",
                extra={
                    "correlation_id": source_correlation_id,
                    "source_index": i+1,
                    "total_sources": len(sources)
                }
            )
            
            # Generate output path with error handling
            try:
                if isinstance(source, str) and source.startswith(('http://', 'https://')):
                    output_file = output_dir / f"transcript_{i+1:03d}.vtt"
                else:
                    source_path = Path(source)
                    output_file = output_dir / f"{source_path.stem}.vtt"
            except Exception as e:
                error_info = {
                    "source": str(source),
                    "source_index": i+1,
                    "error": str(e)
                }
                failed_sources.append(error_info)
                logger.error(
                    f"Failed to generate output path for {source}: {e}",
                    extra={"correlation_id": source_correlation_id}
                )
                continue
                
            try:
                # Pass correlation ID to individual transcription
                result_path = await self.transcribe(
                    source, 
                    output_file, 
                    correlation_id=source_correlation_id,
                    **kwargs
                )
                results.append(result_path)
                
                logger.info(
                    f"Successfully transcribed {source} -> {result_path}",
                    extra={
                        "correlation_id": source_correlation_id,
                        "output_path": str(result_path)
                    }
                )
                
            except Exception as e:
                error_info = {
                    "source": str(source),
                    "source_index": i+1,
                    "output_file": str(output_file),
                    "error": str(e),
                    "error_type": type(e).__name__
                }
                failed_sources.append(error_info)
                
                logger.error(
                    f"Failed to transcribe {source}: {e}",
                    extra={
                        "correlation_id": source_correlation_id,
                        "error_type": type(e).__name__
                    }
                )
                continue
                
        # Log comprehensive batch results
        success_rate = len(results) / len(sources) * 100
        logger.info(
            f"Batch transcription completed: {len(results)}/{len(sources)} successful ({success_rate:.1f}%)",
            extra={
                "correlation_id": batch_correlation_id,
                "successful": len(results),
                "failed": len(failed_sources),
                "total": len(sources),
                "success_rate": success_rate
            }
        )
        
        # Log failed sources for debugging
        if failed_sources:
            logger.warning(
                f"Failed to transcribe {len(failed_sources)} sources",
                extra={
                    "correlation_id": batch_correlation_id,
                    "failed_sources": failed_sources
                }
            )
        
        return results