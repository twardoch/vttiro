# this_file: src/vttiro/core/transcriber.py
"""Simplified transcriber for VTTiro - focused on core transcription functionality.

This module provides a clean, simple interface for transcription without
over-engineered resilience patterns, circuit breakers, or excessive validation.

Key responsibilities:
- Audio file transcription using AI providers
- Basic error handling and retries
- VTT output generation

Used by:
- CLI interface for command-line transcription
- Direct API usage for programmatic access
"""

import asyncio
import time
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from vttiro.core.config import VttiroConfig
from vttiro.core.constants import MAX_TIMEOUT_SECONDS, MIN_TIMEOUT_SECONDS
from vttiro.core.errors import APIError, TranscriptionError, VttiroError
from vttiro.core.types import TranscriptionResult, TranscriptSegment
from vttiro.utils.logging import log_milestone, log_performance, log_timing


class Transcriber:
    """Simple transcriber for VTTiro.

    Provides a clean interface for transcription without excessive complexity.
    Focuses on the core function: converting audio to WebVTT subtitles.
    """

    def __init__(self, config: VttiroConfig | None = None):
        """Initialize transcriber with configuration.

        Args:
            config: VttiroConfig instance, uses defaults if None
        """
        self.config = config or VttiroConfig()

    async def transcribe(self, media_path: Path, output_path: Path | None = None, **kwargs: Any) -> TranscriptionResult:
        """Transcribe video/audio file to text with timing.

        Complete transcription pipeline:
        1. Extract audio from video (if needed) and chunk if large
        2. Transcribe each audio chunk separately
        3. Combine results with timestamp adjustment
        4. Generate VTT output if requested

        Args:
            media_path: Path to input video or audio file
            output_path: Optional output path for VTT file
            **kwargs: Additional parameters for transcription

        Returns:
            TranscriptionResult with segments and metadata

        Raises:
            VttiroError: Transcription failed
            FileNotFoundError: Input file not found
        """
        start_time = time.time()

        # Basic validation
        if not media_path.exists():
            msg = f"Media file not found: {media_path}"
            raise FileNotFoundError(msg)

        provider = self.config.engine or self.config.provider or "gemini"
        logger.info(f"Transcribing {media_path} using {provider}")

        # Log audio processing start
        log_milestone(
            "audio_processing_start",
            {
                "provider": provider,
                "file_path": str(media_path),
                "file_size_mb": f"{media_path.stat().st_size / (1024 * 1024):.1f}",
            },
        )

        # Initialize audio processor
        from vttiro.processing import create_audio_processor

        audio_processor = create_audio_processor(debug=getattr(self.config, "debug", False))

        work_dir = None
        try:
            # Process media file (extract audio, chunk if needed)
            with log_timing("audio_processing", {"num_chunks": "TBD"}):
                audio_chunks, work_dir = await audio_processor.process_media_file(media_path, output_path)

            logger.info(f"Processing {len(audio_chunks)} audio chunks")
            log_milestone("chunks_ready", {"chunk_count": len(audio_chunks)})

            # Transcribe all chunks
            chunk_results = []
            total_offset = 0.0

            for i, chunk_path in enumerate(audio_chunks):
                logger.info(f"Transcribing chunk {i + 1}/{len(audio_chunks)}: {chunk_path.name}")

                # Transcribe individual chunk with timing
                chunk_start_time = time.time()
                chunk_result = await self._transcribe_chunk(chunk_path, provider, **kwargs)
                chunk_duration = time.time() - chunk_start_time

                # Log chunk performance
                log_performance(
                    f"chunk_{i + 1}_transcription",
                    chunk_duration,
                    {
                        "provider": provider,
                        "chunk_file": chunk_path.name,
                        "segments_found": len(chunk_result.segments) if chunk_result else 0,
                    },
                )

                # Adjust timestamps by adding offset
                adjusted_segments = []
                for segment in chunk_result.segments:
                    adjusted_segment = TranscriptSegment(
                        start=segment.start + total_offset,
                        end=segment.end + total_offset,
                        text=segment.text,
                        confidence=segment.confidence,
                        speaker=segment.speaker,
                    )
                    adjusted_segments.append(adjusted_segment)

                chunk_results.append((adjusted_segments, chunk_result.metadata))

                # Update offset for next chunk (get duration from last segment)
                if adjusted_segments:
                    total_offset = adjusted_segments[-1].end

            # Combine all results
            combined_result = self._combine_chunk_results(chunk_results, provider, media_path)

            # Generate VTT output if requested
            if output_path:
                await self._generate_vtt(combined_result, output_path)

            # Add final metadata
            duration = time.time() - start_time
            combined_result.metadata.update(
                {
                    "transcription_duration": duration,
                    "provider": provider,
                    "media_path": str(media_path),
                    "chunks_processed": len(audio_chunks),
                }
            )

            return combined_result

        except Exception as e:
            if isinstance(e, VttiroError):
                raise

            # Convert to VttiroError with context
            error = TranscriptionError(
                f"Transcription failed with {provider}: {e!s}",
                attempts=1,
                details={"media_path": str(media_path), "provider": provider, "error": str(e)},
            )
            raise error from e

        finally:
            # Cleanup working directory unless debug mode
            if work_dir:
                audio_processor.cleanup_working_directory(work_dir)

    async def _transcribe_chunk(self, audio_path: Path, provider: str, **kwargs) -> TranscriptionResult:
        """Transcribe a single audio chunk with retry logic."""
        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                return await self._call_provider(audio_path, provider, **kwargs)

            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    delay = 2**attempt  # Simple exponential backoff
                    logger.warning(f"Chunk transcription attempt {attempt + 1} failed, retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {max_retries} attempts failed for chunk: {audio_path.name}")

        raise last_error or TranscriptionError(f"Chunk transcription failed after {max_retries} attempts")

    def _combine_chunk_results(self, chunk_results: list, provider: str, media_path: Path) -> TranscriptionResult:
        """Combine transcription results from multiple chunks."""
        all_segments = []
        combined_metadata = {"provider": provider, "media_path": str(media_path), "chunks": len(chunk_results)}

        # Combine all segments from all chunks
        for segments, metadata in chunk_results:
            all_segments.extend(segments)

            # Merge metadata (keep first chunk's metadata as base)
            if not combined_metadata.get("language") and metadata.get("language"):
                combined_metadata["language"] = metadata["language"]
            if not combined_metadata.get("confidence") and metadata.get("confidence"):
                combined_metadata["confidence"] = metadata["confidence"]

        # Calculate overall confidence
        if all_segments:
            confidences = [seg.confidence for seg in all_segments if seg.confidence is not None]
            if confidences:
                combined_metadata["confidence"] = sum(confidences) / len(confidences)

        # Create combined result
        result = TranscriptionResult(
            segments=all_segments,
            metadata=combined_metadata,
            provider=provider,
            language=combined_metadata.get("language"),
            confidence=combined_metadata.get("confidence", 0.0),
        )

        logger.info(f"Combined {len(all_segments)} segments from {len(chunk_results)} chunks")
        return result

    async def _call_provider(self, audio_path: Path, provider: str, **kwargs) -> TranscriptionResult:
        """Call the specified transcription provider."""

        # Import provider dynamically to avoid circular imports
        if provider == "gemini":
            from vttiro.providers.gemini.transcriber import GeminiTranscriber

            provider_instance = GeminiTranscriber(self.config)
        elif provider == "openai":
            from vttiro.providers.openai.transcriber import OpenAITranscriber

            provider_instance = OpenAITranscriber(self.config)
        elif provider == "assemblyai":
            from vttiro.providers.assemblyai.transcriber import AssemblyAITranscriber

            provider_instance = AssemblyAITranscriber(self.config)
        elif provider == "deepgram":
            from vttiro.providers.deepgram.transcriber import DeepgramTranscriber

            provider_instance = DeepgramTranscriber(self.config)
        else:
            msg = f"Unsupported provider: {provider}"
            raise ValueError(msg)

        # Call the provider's transcribe method
        return await provider_instance.transcribe(audio_path, **kwargs)

    async def _generate_vtt(self, result: TranscriptionResult, output_path: Path) -> None:
        """Generate VTT file from transcription result."""

        try:
            from vttiro.output.enhanced_webvtt import EnhancedWebVTTFormatter

            formatter = EnhancedWebVTTFormatter()
            vtt_content = formatter.format(result)

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Write VTT file
            output_path.write_text(vtt_content, encoding="utf-8")
            logger.info(f"VTT file written to {output_path}")

        except Exception as e:
            logger.error(f"Failed to generate VTT file: {e}")
            raise

    def get_supported_providers(self) -> list[str]:
        """Get list of available transcription providers."""
        return ["gemini", "openai", "assemblyai", "deepgram"]

    def validate_config(self) -> dict[str, Any]:
        """Basic configuration validation."""
        issues = []
        warnings = []

        # Check provider
        provider = self.config.engine or self.config.provider
        if provider and provider not in self.get_supported_providers():
            issues.append(f"Unsupported provider: {provider}")

        # Check timeout
        if self.config.timeout_seconds < MIN_TIMEOUT_SECONDS:
            warnings.append("Very short timeout may cause failures")
        elif self.config.timeout_seconds > MAX_TIMEOUT_SECONDS:
            warnings.append("Very long timeout - consider reducing")

        return {"valid": len(issues) == 0, "issues": issues, "warnings": warnings, "config": self.config.to_dict()}
