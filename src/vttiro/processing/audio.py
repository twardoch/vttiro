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

import asyncio
import json
import os
import shutil
import subprocess
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psutil
from loguru import logger

from vttiro.core.errors import ProcessingError, create_debug_context

# Memory management constants
MEMORY_LIMITS = {
    "max_memory_usage": 0.8,  # Max 80% of available memory
    "chunk_size_mb": 25,  # Default chunk size
    "streaming_threshold": 100 * 1024 * 1024,  # 100MB - use streaming above this
    "cleanup_threshold": 500 * 1024 * 1024,  # 500MB - aggressive cleanup above this
}

# Audio processing parameters
AUDIO_CONFIG = {
    "sample_rate": 16000,  # Standard rate for speech recognition
    "channels": 1,  # Mono for transcription
    "format": "mp3",  # Output format
    "bitrate": "64k",  # Balanced quality/size
}


class MemoryManager:
    """Memory usage monitoring and management."""

    def __init__(self):
        self.process = psutil.Process()
        self.initial_memory = self.get_memory_usage()

    def get_memory_usage(self) -> dict[str, float]:
        """Get current memory usage statistics."""
        memory_info = self.process.memory_info()
        virtual_memory = psutil.virtual_memory()

        return {
            "process_mb": memory_info.rss / (1024 * 1024),
            "system_available_mb": virtual_memory.available / (1024 * 1024),
            "system_percent": virtual_memory.percent,
            "memory_limit_mb": virtual_memory.total * MEMORY_LIMITS["max_memory_usage"] / (1024 * 1024),
        }

    def check_memory_pressure(self) -> tuple[bool, str]:
        """Check if memory usage is approaching limits."""
        stats = self.get_memory_usage()

        if stats["system_percent"] > 85:
            return True, f"System memory usage high: {stats['system_percent']:.1f}%"

        if stats["process_mb"] > stats["memory_limit_mb"]:
            return True, f"Process memory limit exceeded: {stats['process_mb']:.1f}MB"

        return False, ""

    def suggest_optimization(self, file_size_mb: float) -> dict[str, Any]:
        """Suggest memory optimization strategies."""
        stats = self.get_memory_usage()

        suggestions = {
            "use_streaming": file_size_mb > MEMORY_LIMITS["streaming_threshold"] / (1024 * 1024),
            "reduce_chunk_size": stats["system_percent"] > 70,
            "enable_cleanup": file_size_mb > MEMORY_LIMITS["cleanup_threshold"] / (1024 * 1024),
            "parallel_processing": stats["system_available_mb"] > 1000,
            "current_stats": stats,
        }

        return suggestions


class ProgressTracker:
    """Progress tracking with ETA calculations."""

    def __init__(self, total_work: int, description: str = "Processing"):
        self.total_work = total_work
        self.completed_work = 0
        self.start_time = time.time()
        self.description = description
        self.last_update = 0

    def update(self, work_done: int, message: str | None = None):
        """Update progress and log if significant change."""
        self.completed_work = min(work_done, self.total_work)

        # Only log significant updates (every 5% or 5 seconds)
        current_time = time.time()
        progress_percent = (self.completed_work / self.total_work) * 100

        if current_time - self.last_update > 5.0 or progress_percent - (self.last_update * 100) >= 5:
            eta = self._calculate_eta()
            logger.info(f"{self.description}: {progress_percent:.1f}% complete{eta}")
            if message:
                logger.debug(message)
            self.last_update = current_time

    def _calculate_eta(self) -> str:
        """Calculate estimated time to completion."""
        if self.completed_work == 0:
            return ""

        elapsed = time.time() - self.start_time
        rate = self.completed_work / elapsed
        remaining = self.total_work - self.completed_work

        if rate > 0:
            eta_seconds = remaining / rate
            if eta_seconds < 60:
                return f" (ETA: {eta_seconds:.0f}s)"
            return f" (ETA: {eta_seconds / 60:.1f}m)"

        return ""


class AudioProcessor:
    """Advanced audio processing with memory management and streaming support."""

    def __init__(self):
        self.memory_manager = MemoryManager()
        self.temp_directories = []
        self.executor = ThreadPoolExecutor(max_workers=2)  # Conservative threading

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup_all()
        self.executor.shutdown(wait=True)

    async def process_media_file(
        self, file_path: Path, output_name: str | None = None, progress_callback: Callable | None = None
    ) -> tuple[list[Path], Path]:
        """Process media file with memory-optimized approach.

        Args:
            file_path: Input media file path
            output_name: Custom output name (defaults to input filename)
            progress_callback: Optional callback for progress updates

        Returns:
            Tuple of (audio_chunk_paths, working_directory)
        """
        logger.info(f"Starting audio processing: {file_path}")

        # Memory assessment
        file_size = file_path.stat().st_size
        file_size_mb = file_size / (1024 * 1024)
        optimization = self.memory_manager.suggest_optimization(file_size_mb)

        logger.debug(f"File size: {file_size_mb:.1f}MB, Memory optimization: {optimization}")

        # Create working directory
        working_dir = self._create_working_directory(file_path, output_name)

        try:
            # Progress tracking setup
            progress = ProgressTracker(100, f"Processing {file_path.name}")

            # Step 1: Extract audio (40% of work)
            audio_path = await self._extract_audio_optimized(
                file_path, working_dir, optimization, lambda p: progress.update(int(p * 0.4))
            )

            # Step 2: Analyze audio properties (10% of work)
            audio_info = await self._analyze_audio(audio_path)
            progress.update(50, f"Audio analysis: {audio_info['duration']:.1f}s")

            # Step 3: Chunk audio if needed (50% of work)
            chunk_paths = await self._chunk_audio_optimized(
                audio_path, audio_info, optimization, lambda p: progress.update(50 + int(p * 0.5))
            )

            progress.update(100, f"Completed: {len(chunk_paths)} chunks")

            return chunk_paths, working_dir

        except Exception as e:
            # Cleanup on failure
            await self._cleanup_directory(working_dir)
            raise ProcessingError(
                f"Audio processing failed: {e}",
                file_path=str(file_path),
                stage="audio_processing",
                details=create_debug_context(str(file_path), error=str(e)),
            )

    def _create_working_directory(self, file_path: Path, output_name: str | None = None) -> Path:
        """Create optimized working directory with cleanup tracking."""
        if not output_name:
            output_name = file_path.stem

        # Create unique directory name to avoid conflicts
        timestamp = int(time.time())
        dir_name = f"{output_name}.{timestamp}"
        working_dir = file_path.parent / dir_name

        working_dir.mkdir(exist_ok=True)
        self.temp_directories.append(working_dir)

        logger.debug(f"Created working directory: {working_dir}")
        return working_dir

    async def _extract_audio_optimized(
        self, file_path: Path, working_dir: Path, optimization: dict[str, Any], progress_callback: Callable
    ) -> Path:
        """Extract audio with memory-optimized ffmpeg settings."""
        output_path = working_dir / f"{file_path.stem}.{AUDIO_CONFIG['format']}"

        # Build ffmpeg command with optimization
        cmd = [
            "ffmpeg",
            "-i",
            str(file_path),
            "-vn",  # No video
            "-acodec",
            "mp3",
            "-ar",
            str(AUDIO_CONFIG["sample_rate"]),
            "-ac",
            str(AUDIO_CONFIG["channels"]),
            "-ab",
            AUDIO_CONFIG["bitrate"],
            "-y",  # Overwrite output
        ]

        # Memory optimization flags
        if optimization["use_streaming"]:
            cmd.extend(["-f", "mp3", "-movflags", "+faststart"])

        if optimization["reduce_chunk_size"]:
            cmd.extend(["-bufsize", "512k"])

        cmd.append(str(output_path))

        try:
            # Run extraction with progress monitoring
            await self._run_ffmpeg_with_progress(cmd, progress_callback)

            if not output_path.exists() or output_path.stat().st_size == 0:
                raise ProcessingError(
                    "Audio extraction produced no output", file_path=str(file_path), stage="audio_extraction"
                )

            logger.info(f"Audio extracted: {output_path} ({output_path.stat().st_size / (1024 * 1024):.1f}MB)")
            return output_path

        except subprocess.CalledProcessError as e:
            raise ProcessingError(f"FFmpeg extraction failed: {e}", file_path=str(file_path), stage="audio_extraction")

    async def _run_ffmpeg_with_progress(self, cmd: list[str], progress_callback: Callable):
        """Run ffmpeg command with progress monitoring."""
        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        # Simple progress simulation (ffmpeg progress parsing would be complex)
        for i in range(0, 101, 5):
            if process.returncode is not None:
                break
            progress_callback(i)
            await asyncio.sleep(0.1)

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd, stderr.decode())

    async def _analyze_audio(self, audio_path: Path) -> dict[str, Any]:
        """Analyze audio properties for optimal chunking."""
        cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", str(audio_path)]

        try:
            result = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()

            if result.returncode != 0:
                raise ProcessingError("Failed to analyze audio properties")

            data = json.loads(stdout.decode())

            format_info = data.get("format", {})
            duration = float(format_info.get("duration", 0))
            size_mb = audio_path.stat().st_size / (1024 * 1024)

            return {
                "duration": duration,
                "size_mb": size_mb,
                "bitrate": format_info.get("bit_rate"),
                "needs_chunking": size_mb > MEMORY_LIMITS["chunk_size_mb"] or duration > 300,  # 5 minutes
            }

        except Exception as e:
            logger.warning(f"Audio analysis failed, using defaults: {e}")
            # Fallback to basic file size analysis
            size_mb = audio_path.stat().st_size / (1024 * 1024)
            return {
                "duration": size_mb * 8,  # Rough estimate
                "size_mb": size_mb,
                "bitrate": "unknown",
                "needs_chunking": size_mb > MEMORY_LIMITS["chunk_size_mb"],
            }

    async def _chunk_audio_optimized(
        self, audio_path: Path, audio_info: dict[str, Any], optimization: dict[str, Any], progress_callback: Callable
    ) -> list[Path]:
        """Create audio chunks with memory-optimized approach."""
        if not audio_info["needs_chunking"]:
            logger.info(f"Audio within limits, no chunking needed: {audio_info['size_mb']:.1f}MB")
            progress_callback(100)
            return [audio_path]

        # Calculate optimal chunk parameters
        duration = audio_info["duration"]
        target_chunk_mb = MEMORY_LIMITS["chunk_size_mb"]
        if optimization["reduce_chunk_size"]:
            target_chunk_mb = target_chunk_mb * 0.6  # Reduce by 40%

        # Estimate chunk duration based on file size ratio
        chunk_duration = (target_chunk_mb / audio_info["size_mb"]) * duration
        chunk_duration = max(60, min(300, chunk_duration))  # Between 1-5 minutes

        num_chunks = int(duration / chunk_duration) + 1
        chunk_paths = []

        logger.info(f"Chunking audio: {num_chunks} chunks of ~{chunk_duration:.0f}s each")

        # Create chunks with progress tracking
        for i in range(num_chunks):
            start_time = i * chunk_duration
            chunk_path = audio_path.parent / f"{audio_path.stem}_chunk_{i:03d}.mp3"

            cmd = [
                "ffmpeg",
                "-i",
                str(audio_path),
                "-ss",
                str(start_time),
                "-t",
                str(chunk_duration),
                "-c",
                "copy",  # Stream copy for speed
                "-y",
                str(chunk_path),
            ]

            try:
                await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
                )

                if chunk_path.exists() and chunk_path.stat().st_size > 1024:  # At least 1KB
                    chunk_paths.append(chunk_path)
                    progress_callback(int((i + 1) / num_chunks * 100))

                    # Memory pressure check
                    pressure, msg = self.memory_manager.check_memory_pressure()
                    if pressure:
                        logger.warning(f"Memory pressure detected: {msg}")
                        if optimization["enable_cleanup"]:
                            await self._cleanup_temp_files(audio_path.parent)

            except Exception as e:
                logger.warning(f"Failed to create chunk {i}: {e}")

        logger.info(f"Created {len(chunk_paths)} audio chunks")
        return chunk_paths

    async def _cleanup_temp_files(self, directory: Path):
        """Cleanup temporary files to free memory."""
        logger.debug(f"Cleaning up temporary files in {directory}")

        for temp_file in directory.glob("*.tmp"):
            try:
                temp_file.unlink()
            except Exception as e:
                logger.debug(f"Could not remove temp file {temp_file}: {e}")

    async def _cleanup_directory(self, directory: Path):
        """Clean up entire working directory."""
        try:
            if directory.exists():
                shutil.rmtree(directory)
                logger.debug(f"Cleaned up directory: {directory}")
        except Exception as e:
            logger.warning(f"Failed to cleanup directory {directory}: {e}")

    def cleanup_all(self):
        """Clean up all temporary directories created by this processor."""
        logger.debug(f"Cleaning up {len(self.temp_directories)} temporary directories")

        for directory in self.temp_directories:
            try:
                if directory.exists():
                    shutil.rmtree(directory)
            except Exception as e:
                logger.warning(f"Failed to cleanup {directory}: {e}")

        self.temp_directories.clear()

    def get_memory_stats(self) -> dict[str, Any]:
        """Get current memory usage statistics."""
        return self.memory_manager.get_memory_usage()

    def cleanup_working_directory(self, working_dir: Path) -> None:
        """Clean up a specific working directory.

        Args:
            working_dir: Path to working directory to cleanup
        """
        try:
            if working_dir and working_dir.exists():
                import shutil

                shutil.rmtree(working_dir)
                logger.debug(f"Cleaned up working directory: {working_dir}")
                # Remove from tracking list if present
                if working_dir in self.temp_directories:
                    self.temp_directories.remove(working_dir)
        except Exception as e:
            logger.warning(f"Failed to cleanup working directory {working_dir}: {e}")


def create_audio_processor(debug: bool = False) -> AudioProcessor:
    """Create and configure an AudioProcessor instance.

    Args:
        debug: Enable debug mode for more verbose logging and temp file preservation

    Returns:
        Configured AudioProcessor instance
    """
    processor = AudioProcessor()
    if debug:
        logger.debug("AudioProcessor created in debug mode")
    return processor
