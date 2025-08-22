# this_file: src/vttiro/processing/audio.py
"""Audio extraction and processing for VTTiro.

This module handles:
- Audio extraction from video files using ffmpeg
- Audio chunking for large files (>20MB or >20min)
- Working directory management with caching
- Audio format conversion and optimization

Used by:
- Core transcription pipeline for audio preparation
- Provider-specific audio processing requirements
"""

import os
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)


class AudioProcessor:
    """Handles audio extraction and processing for transcription."""

    # Size and duration limits for chunking
    MAX_FILE_SIZE_MB = 20
    MAX_DURATION_MINUTES = 20

    def __init__(self, debug: bool = False):
        """Initialize audio processor.

        Args:
            debug: If True, preserve working directories for debugging
        """
        self.debug = debug

    def process_media_file(
        self, input_path: Path, output_path: Optional[Path] = None
    ) -> Tuple[List[Path], Path]:
        """Process video/audio file for transcription.

        Args:
            input_path: Path to input video or audio file
            output_path: Optional output path for VTT file (used for working dir)

        Returns:
            Tuple of (list of audio chunk paths, working directory)
        """
        # Create working directory
        work_dir = self._create_working_directory(input_path, output_path)

        # Extract audio if it's a video file
        audio_path = self._extract_audio(input_path, work_dir)

        # Check if chunking is needed
        chunks = self._chunk_audio_if_needed(audio_path, work_dir)

        logger.info(f"Audio processing complete: {len(chunks)} chunks in {work_dir}")
        return chunks, work_dir

    def cleanup_working_directory(self, work_dir: Path) -> None:
        """Clean up working directory unless debug mode is enabled."""
        if not self.debug and work_dir.exists():
            import shutil

            shutil.rmtree(work_dir)
            logger.debug(f"Cleaned up working directory: {work_dir}")

    def _create_working_directory(
        self, input_path: Path, output_path: Optional[Path]
    ) -> Path:
        """Create working directory based on input/output paths."""
        if output_path:
            # Use output file location and basename
            work_dir = output_path.parent / f"{output_path.stem}"
        else:
            # Use input file location and basename
            work_dir = input_path.parent / f"{input_path.stem}"

        work_dir.mkdir(exist_ok=True)
        logger.debug(f"Working directory: {work_dir}")
        return work_dir

    def _extract_audio(self, input_path: Path, work_dir: Path) -> Path:
        """Extract audio from video or convert audio format."""
        audio_path = work_dir / f"{input_path.stem}.mp3"

        # Check if audio already exists (caching)
        if audio_path.exists():
            logger.info(f"Using cached audio: {audio_path}")
            return audio_path

        # Check if input is already audio
        audio_extensions = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"}
        if input_path.suffix.lower() in audio_extensions:
            # Copy or convert audio file
            self._convert_audio(input_path, audio_path)
        else:
            # Extract audio from video
            self._extract_audio_from_video(input_path, audio_path)

        logger.info(f"Audio extracted: {audio_path}")
        return audio_path

    def _extract_audio_from_video(self, video_path: Path, audio_path: Path) -> None:
        """Extract audio from video using ffmpeg."""
        cmd = [
            "ffmpeg",
            "-i",
            str(video_path),
            "-vn",  # No video
            "-acodec",
            "mp3",
            "-ab",
            "128k",  # Audio bitrate
            "-ar",
            "44100",  # Sample rate
            "-y",  # Overwrite output
            str(audio_path),
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.debug(f"FFmpeg extraction successful: {audio_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg failed: {e.stderr}")
            raise RuntimeError(f"Failed to extract audio from {video_path}: {e.stderr}")
        except FileNotFoundError:
            raise RuntimeError(
                "ffmpeg not found. Please install ffmpeg to process video files."
            )

    def _convert_audio(self, input_path: Path, output_path: Path) -> None:
        """Convert audio to standardized MP3 format."""
        cmd = [
            "ffmpeg",
            "-i",
            str(input_path),
            "-acodec",
            "mp3",
            "-ab",
            "128k",  # Audio bitrate
            "-ar",
            "44100",  # Sample rate
            "-y",  # Overwrite output
            str(output_path),
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.debug(f"Audio conversion successful: {output_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Audio conversion failed: {e.stderr}")
            raise RuntimeError(f"Failed to convert audio {input_path}: {e.stderr}")

    def _chunk_audio_if_needed(self, audio_path: Path, work_dir: Path) -> List[Path]:
        """Chunk audio file if it exceeds size or duration limits."""
        # Check file size
        file_size_mb = audio_path.stat().st_size / (1024 * 1024)

        # Get audio duration
        duration_seconds = self._get_audio_duration(audio_path)
        duration_minutes = duration_seconds / 60

        logger.info(f"Audio stats: {file_size_mb:.1f}MB, {duration_minutes:.1f}min")

        # Check if chunking is needed
        if (
            file_size_mb <= self.MAX_FILE_SIZE_MB
            and duration_minutes <= self.MAX_DURATION_MINUTES
        ):
            logger.info("Audio within limits, no chunking needed")
            return [audio_path]

        # Calculate chunk duration
        max_chunk_duration = min(
            self.MAX_DURATION_MINUTES * 60,  # 20 minutes in seconds
            duration_seconds / 2,  # Split in half if needed
        )

        return self._split_audio(audio_path, work_dir, max_chunk_duration)

    def _get_audio_duration(self, audio_path: Path) -> float:
        """Get audio duration in seconds using ffprobe."""
        cmd = [
            "ffprobe",
            "-i",
            str(audio_path),
            "-show_entries",
            "format=duration",
            "-v",
            "quiet",
            "-of",
            "csv=p=0",
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            duration = float(result.stdout.strip())
            return duration
        except (subprocess.CalledProcessError, ValueError) as e:
            logger.warning(
                f"Could not get audio duration, using file size estimate: {e}"
            )
            # Fallback: estimate based on file size (rough approximation)
            file_size_mb = audio_path.stat().st_size / (1024 * 1024)
            return file_size_mb * 60  # Very rough estimate: 1MB ≈ 1 minute

    def _split_audio(
        self, audio_path: Path, work_dir: Path, chunk_duration: float
    ) -> List[Path]:
        """Split audio into chunks at low-energy points."""
        chunks = []
        total_duration = self._get_audio_duration(audio_path)

        # Find optimal split points (simple implementation - split at regular intervals)
        split_points = []
        current_time = 0
        chunk_index = 0

        while current_time < total_duration:
            if current_time + chunk_duration >= total_duration:
                # Last chunk - take remaining duration
                break

            # Find low-energy point near target split time
            target_split = current_time + chunk_duration
            optimal_split = self._find_low_energy_point(audio_path, target_split)
            split_points.append(optimal_split)
            current_time = optimal_split

        # Create chunks
        start_time = 0
        for i, end_time in enumerate(split_points + [total_duration]):
            chunk_path = work_dir / f"chunk_{i:03d}.mp3"

            # Check if chunk already exists (caching)
            if not chunk_path.exists():
                self._extract_audio_segment(
                    audio_path, chunk_path, start_time, end_time
                )

            chunks.append(chunk_path)
            start_time = end_time

        logger.info(f"Created {len(chunks)} audio chunks")
        return chunks

    def _find_low_energy_point(self, audio_path: Path, target_time: float) -> float:
        """Find a low-energy point near target time for optimal splitting.

        Simple implementation: round to nearest second and look for silence.
        """
        # Round to full seconds as requested
        rounded_time = round(target_time)

        # For now, just return the rounded time
        # TODO: Implement actual energy analysis if needed
        return float(rounded_time)

    def _extract_audio_segment(
        self, input_path: Path, output_path: Path, start_time: float, end_time: float
    ) -> None:
        """Extract a segment of audio using ffmpeg."""
        duration = end_time - start_time

        cmd = [
            "ffmpeg",
            "-i",
            str(input_path),
            "-ss",
            str(start_time),  # Start time
            "-t",
            str(duration),  # Duration
            "-acodec",
            "mp3",
            "-y",  # Overwrite output
            str(output_path),
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.debug(
                f"Audio segment extracted: {output_path} ({start_time:.1f}s-{end_time:.1f}s)"
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Audio segment extraction failed: {e.stderr}")
            raise RuntimeError(
                f"Failed to extract segment {start_time}-{end_time}: {e.stderr}"
            )


def create_audio_processor(debug: bool = False) -> AudioProcessor:
    """Create an AudioProcessor instance.

    Args:
        debug: Enable debug mode (preserve working directories)

    Returns:
        AudioProcessor instance
    """
    return AudioProcessor(debug=debug)
