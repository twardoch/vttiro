#!/usr/bin/env python3
# this_file: src/vttiro/core/file_transcriber.py
"""Simple file transcriber for local audio/video files.

This module provides the core missing functionality - transcribing local
audio/video files to WebVTT subtitles using AI models.
"""

import asyncio
import uuid
import time
import json
from pathlib import Path
from typing import Optional, Union, List, Dict, Any

try:
    from loguru import logger
except ImportError:
    import logging as logger

from vttiro.core.config import VttiroConfig, TranscriptionResult
from vttiro.config.enhanced import EnhancedVttiroConfig
from vttiro.processing.simple_audio import SimpleAudioProcessor
from vttiro.output.simple_webvtt import SimpleWebVTTGenerator
from vttiro.utils.types import SimpleTranscriptSegment
from vttiro.models import (
    TranscriptionEngine,
    get_default_model,
    validate_engine_model_combination,
)
from vttiro.utils.exceptions import (
    VttiroError,
    ProcessingError,
    TranscriptionError,
    ValidationError,
)
from vttiro.monitoring import get_performance_monitor
from vttiro.utils.config_validation import validate_startup_configuration


class FileTranscriber:
    """Simple file transcriber for local audio/video files.

    Provides the essential missing functionality to transcribe local media files
    to WebVTT subtitles. Supports common formats and basic error handling.
    """

    def __init__(
        self, config: Optional[Union[VttiroConfig, EnhancedVttiroConfig]] = None
    ):
        """Initialize the file transcriber.

        Args:
            config: Configuration object (creates default if None)
        """
        if config is None:
            self.config = VttiroConfig()
            # Load environment variables automatically
            self.config.update_from_env()
        elif isinstance(config, EnhancedVttiroConfig):
            # Convert enhanced config to simple config for compatibility
            self.config = self._convert_enhanced_config(config)
        else:
            self.config = config

        # Initialize audio processor
        self.audio_processor = SimpleAudioProcessor()
        
        # Initialize performance monitor
        self.performance_monitor = get_performance_monitor()
        
        # Validate configuration at startup
        is_valid, critical_errors = validate_startup_configuration(self.config)
        if not is_valid:
            error_msg = "Configuration validation failed with critical errors: " + "; ".join(critical_errors)
            logger.error(error_msg)
            for error in critical_errors:
                logger.error(f"  - {error}")
            logger.error("Use 'vttiro config_health' to diagnose configuration issues")
            raise ValidationError(error_msg)
        
        logger.info("FileTranscriber initialized with validated configuration")

    def _convert_enhanced_config(
        self, enhanced_config: EnhancedVttiroConfig
    ) -> VttiroConfig:
        """Convert enhanced config to simple config for backward compatibility.

        Args:
            enhanced_config: Enhanced configuration object

        Returns:
            Simple VttiroConfig object
        """
        # Create basic config first
        config = VttiroConfig()
        
        # Update transcription settings
        if enhanced_config.api.gemini_api_key:
            config.transcription.gemini_api_key = str(enhanced_config.api.gemini_api_key)
        if enhanced_config.api.assemblyai_api_key:
            config.transcription.assemblyai_api_key = str(enhanced_config.api.assemblyai_api_key)
        if enhanced_config.api.deepgram_api_key:
            config.transcription.deepgram_api_key = str(enhanced_config.api.deepgram_api_key)
        
        # Update other settings
        config.transcription.preferred_model = enhanced_config.models.default_provider
        config.processing.chunk_duration = enhanced_config.processing.chunk_duration
        config.output.max_chars_per_line = enhanced_config.output.max_chars_per_line
        config.verbose = enhanced_config.verbose or False
        
        return config

    def validate_file(self, file_path: Path) -> bool:
        """Validate input file format and existence.

        Args:
            file_path: Path to the input file

        Returns:
            True if file is valid and supported

        Raises:
            ValidationError: If file is invalid or unsupported
        """
        return self.audio_processor.validate_file(file_path)

    def _extract_audio(self, video_path: Path, keep_audio: bool = False) -> Path:
        """Extract audio from video file using the audio processor.

        Args:
            video_path: Path to video file
            keep_audio: Save audio file next to video with same basename, reuse existing

        Returns:
            Path to extracted audio file

        Raises:
            ProcessingError: If audio extraction fails
        """
        return self.audio_processor.extract_audio(video_path, keep_audio=keep_audio)

    def _create_transcriber(self, engine: str, model: str):
        """Create appropriate transcriber based on engine and model.

        Args:
            engine: AI engine name (gemini, assemblyai, deepgram)
            model: Specific model within engine

        Returns:
            Transcriber instance for the specified engine/model

        Raises:
            ValidationError: If engine is not supported
        """
        if engine == "gemini":
            from vttiro.models.gemini import GeminiTranscriber
            from vttiro.models.base import GeminiModel

            model_enum = GeminiModel(model)
            return GeminiTranscriber(self.config, model_enum)
        elif engine == "assemblyai":
            from vttiro.models.assemblyai import AssemblyAITranscriber
            from vttiro.models.base import AssemblyAIModel

            model_enum = AssemblyAIModel(model)
            return AssemblyAITranscriber(self.config, model_enum)
        elif engine == "deepgram":
            from vttiro.models.deepgram import DeepgramTranscriber
            from vttiro.models.base import DeepgramModel

            model_enum = DeepgramModel(model)
            return DeepgramTranscriber(self.config, model_enum)
        elif engine == "openai":
            logger.debug(f"Creating OpenAI transcriber for engine={engine}, model={model}")
            try:
                from vttiro.models.openai import OpenAITranscriber
                from vttiro.models.base import OpenAIModel
                logger.debug("OpenAI imports successful")
                
                logger.debug(f"Converting model string '{model}' to OpenAIModel enum")
                model_enum = OpenAIModel(model)
                logger.debug(f"Model enum created: {model_enum}")
                
                logger.debug("Creating OpenAITranscriber instance")
                transcriber = OpenAITranscriber(self.config, model_enum)
                logger.info(f"OpenAI transcriber created successfully: {transcriber.name}")
                return transcriber
            except Exception as e:
                error_msg = f"Failed to create OpenAI transcriber: {e}"
                logger.error(error_msg)
                raise
        else:
            raise ValidationError(f"Unsupported engine: {engine}")

    async def _transcribe_with_retry(
        self,
        transcriber,
        audio_path: Path,
        correlation_id: str,
        max_retries: int = 3,
        timeout_seconds: int = 300,
    ):
        """Transcribe with retry logic and timeout handling.

        Args:
            transcriber: The transcriber instance to use
            audio_path: Path to the audio file
            correlation_id: Request correlation ID for logging
            max_retries: Maximum number of retry attempts
            timeout_seconds: Timeout for each transcription attempt

        Returns:
            TranscriptionResult

        Raises:
            TranscriptionError: If all retries fail
            TimeoutError: If transcription times out
        """
        last_exception = None

        for attempt in range(max_retries):
            try:
                logger.info(
                    f"[{correlation_id}] Transcription attempt {attempt + 1}/{max_retries}"
                )

                # Add timeout to the transcription call
                result = await asyncio.wait_for(
                    transcriber.transcribe(audio_path), timeout=timeout_seconds
                )

                # CRITICAL: Validate transcription result quality
                if not result.text or not result.text.strip():
                    logger.error(f"[{correlation_id}] CRITICAL: Transcription returned empty text on attempt {attempt + 1}")
                    logger.error(f"[{correlation_id}] Result metadata: {result.metadata}")
                    if result.word_timestamps:
                        logger.error(f"[{correlation_id}] Word timestamps available: {len(result.word_timestamps)} words")
                    else:
                        logger.error(f"[{correlation_id}] No word timestamps available")
                    
                    # Treat empty results as a failure for retry
                    if attempt < max_retries - 1:
                        logger.warning(f"[{correlation_id}] Retrying due to empty transcription result...")
                        continue
                    else:
                        logger.error(f"[{correlation_id}] FINAL ATTEMPT: Accepting empty transcription result")

                logger.info(
                    f"[{correlation_id}] Transcription successful on attempt {attempt + 1}"
                )
                
                # Log transcription quality info
                text_length = len(result.text.strip()) if result.text else 0
                word_count = len(result.word_timestamps) if result.word_timestamps else 0
                logger.info(f"[{correlation_id}] Quality: {text_length} chars, {word_count} words, confidence={result.confidence:.3f}")
                
                return result

            except asyncio.TimeoutError as e:
                last_exception = e
                logger.warning(
                    f"[{correlation_id}] Transcription timed out on attempt {attempt + 1}"
                )
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # Exponential backoff: 2s, 4s, 6s
                    logger.info(
                        f"[{correlation_id}] Retrying in {wait_time} seconds..."
                    )
                    await asyncio.sleep(wait_time)

            except (ConnectionError, OSError) as e:
                # Network-related errors that might be transient
                last_exception = e
                logger.warning(
                    f"[{correlation_id}] Network error on attempt {attempt + 1}: {e}"
                )
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    logger.info(
                        f"[{correlation_id}] Retrying in {wait_time} seconds..."
                    )
                    await asyncio.sleep(wait_time)

            except Exception as e:
                # Non-retryable errors
                logger.error(f"[{correlation_id}] Non-retryable error: {e}")
                raise TranscriptionError(f"Transcription failed: {e}")

        # All retries exhausted
        if isinstance(last_exception, asyncio.TimeoutError):
            raise TimeoutError(
                f"Transcription timed out after {max_retries} attempts (max {timeout_seconds}s each)"
            )
        else:
            raise TranscriptionError(
                f"Transcription failed after {max_retries} attempts: {last_exception}"
            )

    async def transcribe_file(
        self,
        file_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        engine: Optional[str] = None,
        model: Optional[str] = None,
        full_prompt: Optional[str] = None,
        xtra_prompt: Optional[str] = None,
        add_cues: bool = False,
        keep_audio: bool = False,
        raw: bool = False,
    ) -> Path:
        """Transcribe audio/video file to WebVTT subtitles.

        Args:
            file_path: Path to input audio/video file
            output_path: Path for output WebVTT file (optional)
            engine: AI engine to use (optional, defaults to 'gemini')
            model: Specific model within engine (optional, uses engine default)
            full_prompt: Replace default prompt entirely (file path or text, optional)
            xtra_prompt: Append to default/custom prompt (file path or text, optional)
            add_cues: Include cue identifiers in WebVTT output (default: False)
            keep_audio: Save audio file next to video with same basename, reuse existing (default: False)
            raw: Save complete raw AI model output as JSON alongside WebVTT file (default: False)

        Returns:
            Path to generated WebVTT file

        Raises:
            ValidationError: If input file is invalid
            ProcessingError: If audio extraction fails
            TranscriptionError: If transcription fails
        """
        # Generate correlation ID for request tracking
        correlation_id = str(uuid.uuid4())[:8]

        # Convert to Path objects
        input_path = Path(file_path)

        if output_path is None:
            output_path = input_path.with_suffix(".vtt")
        else:
            output_path = Path(output_path)

        logger.info(
            f"[{correlation_id}] Starting transcription: {input_path.name} -> {output_path.name}"
        )
        start_time = time.time()

        # Start performance monitoring for the entire transcription session
        session_metrics = self.performance_monitor.start_transcription(
            correlation_id=correlation_id,
            input_file=str(input_path),
            output_file=str(output_path),
            engine=engine or "gemini",
            model=model or "default"
        )

        try:
            # Validate input file
            self.validate_file(input_path)

            # Extract audio if needed - track this operation
            audio_extraction_op = self.performance_monitor.start_operation(
                correlation_id, "audio_extraction", {"keep_audio": keep_audio}
            )
            try:
                audio_path = self._extract_audio(input_path, keep_audio=keep_audio)
                # Record audio file size for performance analysis
                audio_size = audio_path.stat().st_size if audio_path.exists() else 0
                session_metrics.audio_file_size = audio_size
                self.performance_monitor.finish_operation(
                    audio_extraction_op, success=True, 
                    metadata={"audio_file_size": audio_size, "audio_file_path": str(audio_path)}
                )
            except Exception as e:
                self.performance_monitor.finish_operation(
                    audio_extraction_op, success=False, error_message=str(e)
                )
                raise

            # Validate and set engine/model
            if engine is None:
                engine = "gemini"  # Default engine

            # Validate engine
            try:
                engine_enum = TranscriptionEngine(engine)
            except ValueError:
                raise ValidationError(f"Invalid engine: {engine}")

            # Set default model if not specified
            if model is None:
                model = get_default_model(engine_enum)
            else:
                # Validate engine/model combination
                if not validate_engine_model_combination(engine, model):
                    raise ValidationError(
                        f"Invalid model '{model}' for engine '{engine}'"
                    )

            # Create appropriate transcriber based on engine
            transcriber = self._create_transcriber(engine, model)

            # Perform transcription with retry logic and timeout - track this operation
            logger.info(
                f"[{correlation_id}] Transcribing audio file: {audio_path.name} with {engine}/{model}"
            )
            
            ai_processing_op = self.performance_monitor.start_operation(
                correlation_id, "ai_processing", 
                {"engine": engine, "model": model, "audio_file": str(audio_path)}
            )
            try:
                result = await self._transcribe_with_retry(
                    transcriber, audio_path, correlation_id
                )
                
                # Extract quality metrics for performance tracking
                quality_metrics = {
                    "text_length": len(result.text) if result.text else 0,
                    "word_count": len(result.word_timestamps) if result.word_timestamps else 0,
                    "confidence": result.confidence,
                    "language": result.language,
                }
                
                self.performance_monitor.finish_operation(
                    ai_processing_op, success=True, metadata=quality_metrics
                )
            except Exception as e:
                self.performance_monitor.finish_operation(
                    ai_processing_op, success=False, error_message=str(e)
                )
                raise

            # Save result to WebVTT file - track this operation
            webvtt_generation_op = self.performance_monitor.start_operation(
                correlation_id, "webvtt_generation", 
                {"output_file": str(output_path)}
            )
            try:
                self._save_webvtt(result, output_path)
                
                # Save raw output if requested
                if raw:
                    self._save_raw_output(result, output_path)
                
                # Record output file size
                output_size = output_path.stat().st_size if output_path.exists() else 0
                
                self.performance_monitor.finish_operation(
                    webvtt_generation_op, success=True,
                    metadata={"output_file_size": output_size, "raw_output_saved": raw}
                )
            except Exception as e:
                self.performance_monitor.finish_operation(
                    webvtt_generation_op, success=False, error_message=str(e)
                )
                raise

            # Cleanup temporary files
            self.audio_processor.cleanup_temp_files()

            elapsed_time = time.time() - start_time
            
            # Complete performance monitoring for successful transcription
            transcription_quality = {
                "text_length": len(result.text) if result.text else 0,
                "word_count": len(result.word_timestamps) if result.word_timestamps else 0,
                "confidence": result.confidence,
                "language": result.language,
            }
            
            self.performance_monitor.finish_transcription(
                correlation_id, success=True, transcription_quality=transcription_quality
            )
            
            logger.info(
                f"[{correlation_id}] Transcription completed successfully: {output_path} (took {elapsed_time:.2f}s)"
            )
            return output_path

        except (ValidationError, ProcessingError, TranscriptionError) as e:
            elapsed_time = time.time() - start_time
            
            # Complete performance monitoring for failed transcription
            self.performance_monitor.finish_transcription(
                correlation_id, success=False, error_message=str(e)
            )
            
            logger.error(
                f"[{correlation_id}] Transcription failed after {elapsed_time:.2f}s: {e}"
            )
            raise
        except Exception as e:
            elapsed_time = time.time() - start_time
            
            # Complete performance monitoring for unexpected errors
            self.performance_monitor.finish_transcription(
                correlation_id, success=False, error_message=f"Unexpected error: {e}"
            )
            
            logger.error(
                f"[{correlation_id}] Unexpected error during transcription after {elapsed_time:.2f}s: {e}"
            )
            raise TranscriptionError(f"Transcription failed: {e}")

    def _save_webvtt(self, result: TranscriptionResult, output_path: Path) -> None:
        """Save transcription result to WebVTT file.

        Args:
            result: Transcription result object
            output_path: Path to output WebVTT file

        Raises:
            ProcessingError: If file writing fails
        """
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert TranscriptionResult to WebVTT segments
            segments = self._create_webvtt_segments(result)

            # Create WebVTT generator
            webvtt_generator = SimpleWebVTTGenerator()

            # Generate and save WebVTT content
            webvtt_generator.generate_webvtt(
                segments=segments, output_path=output_path, language=result.language
            )

            logger.debug(f"WebVTT file saved: {output_path}")

        except Exception as e:
            raise ProcessingError(f"Failed to save WebVTT file: {e}")

    def _save_raw_output(self, result: TranscriptionResult, webvtt_path: Path) -> None:
        """Save complete raw AI model output as JSON alongside WebVTT file.

        Args:
            result: Transcription result object containing raw output
            webvtt_path: Path to WebVTT file (used to determine JSON output path)

        Raises:
            ProcessingError: If file writing fails
        """
        try:
            # Create JSON file path with .vtt.json extension
            json_path = webvtt_path.with_suffix(".vtt.json")
            
            # Ensure output directory exists
            json_path.parent.mkdir(parents=True, exist_ok=True)

            # Create comprehensive raw output data
            raw_data = {
                "metadata": {
                    "transcription_timestamp": time.time(),
                    "engine": getattr(result, "engine", "unknown"),
                    "model": getattr(result, "model", "unknown"),
                    "language": result.language,
                    "confidence": result.confidence,
                    "processing_time": getattr(result, "processing_time", None),
                    "file_duration": getattr(result, "file_duration", None),
                    "vttiro_version": "1.0.1"  # Add version info
                },
                "transcription": {
                    "text": result.text,
                    "word_timestamps": result.word_timestamps,
                    "start_time": getattr(result, "start_time", None),
                    "end_time": getattr(result, "end_time", None),
                },
                "raw_response": getattr(result, "raw_response", None),
                "additional_metadata": result.metadata or {}
            }

            # Write JSON file with proper formatting
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(raw_data, f, indent=2, ensure_ascii=False, default=self._json_serializer)

            logger.debug(f"Raw output JSON saved: {json_path}")

        except Exception as e:
            raise ProcessingError(f"Failed to save raw output JSON: {e}")

    def _json_serializer(self, obj):
        """Custom JSON serializer for handling non-JSON-serializable objects."""
        # Handle common AI model response types
        if hasattr(obj, '__dict__'):
            return str(obj)
        elif hasattr(obj, '_asdict'):  # namedtuple
            return obj._asdict()
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif hasattr(obj, '__str__'):
            return str(obj)
        else:
            return f"<{type(obj).__name__} object>"

    def _create_webvtt_segments(
        self, result: TranscriptionResult
    ) -> List[SimpleTranscriptSegment]:
        """Convert TranscriptionResult to SimpleTranscriptSegment objects.

        Args:
            result: Transcription result with text and word timestamps

        Returns:
            List of SimpleTranscriptSegment objects for WebVTT generation
        """
        segments = []

        # If we have word-level timestamps, use them to create segments
        if result.word_timestamps and len(result.word_timestamps) > 0:
            # Group words into segments (aim for ~7 seconds per segment max)
            current_segment_words = []
            current_start_time = result.word_timestamps[0].get("start", 0.0)
            segment_duration_limit = 7.0  # seconds

            for word_data in result.word_timestamps:
                word_start = word_data.get("start", 0.0)
                word_end = word_data.get("end", word_start + 0.5)
                word_text = word_data.get("word", "")

                # If this segment would be too long, close current segment
                if (
                    word_start - current_start_time > segment_duration_limit
                    and len(current_segment_words) > 0
                ):

                    # Create segment from accumulated words
                    segment_text = " ".join(current_segment_words).strip()
                    if segment_text:
                        last_word = (
                            result.word_timestamps[
                                len(segments) * 10 + len(current_segment_words) - 1
                            ]
                            if segments or current_segment_words
                            else word_data
                        )
                        segment_end = last_word.get("end", word_start)

                        segments.append(
                            SimpleTranscriptSegment(
                                start_time=current_start_time,
                                end_time=segment_end,
                                text=segment_text,
                            )
                        )

                    # Start new segment
                    current_segment_words = [word_text]
                    current_start_time = word_start
                else:
                    current_segment_words.append(word_text)

            # Add final segment if there are remaining words
            if current_segment_words:
                segment_text = " ".join(current_segment_words).strip()
                if segment_text:
                    final_end = result.word_timestamps[-1].get(
                        "end", current_start_time + 1.0
                    )
                    segments.append(
                        SimpleTranscriptSegment(
                            start_time=current_start_time,
                            end_time=final_end,
                            text=segment_text,
                        )
                    )

        else:
            # Fallback: create single segment with full text
            # Use result start/end times or estimate based on text length
            start_time = getattr(result, "start_time", 0.0)
            end_time = getattr(result, "end_time", None)

            if end_time is None:
                # Estimate duration based on text length (average reading speed)
                words = len(result.text.split())
                estimated_duration = max(
                    words * 0.5, 2.0
                )  # ~0.5 seconds per word, minimum 2 seconds
                end_time = start_time + estimated_duration

            segments.append(
                SimpleTranscriptSegment(
                    start_time=start_time, end_time=end_time, text=result.text.strip()
                )
            )

        return segments

    def get_supported_formats(self) -> List[str]:
        """Get list of supported input formats.

        Returns:
            List of supported file extensions
        """
        return self.audio_processor.get_supported_formats()

    def __repr__(self) -> str:
        """String representation of FileTranscriber."""
        return f"FileTranscriber(model={self.config.preferred_model})"
