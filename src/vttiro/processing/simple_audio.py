#!/usr/bin/env python3
# this_file: src/vttiro/processing/simple_audio.py
"""Simple audio processing pipeline for vttiro.

This module provides basic audio extraction and validation functionality
without the complexity of the previous over-engineered processing pipeline.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Set
import shutil

try:
    from loguru import logger
except ImportError:
    import logging as logger

from vttiro.utils.exceptions import ProcessingError, ValidationError


class SimpleAudioProcessor:
    """Simple audio processor for extracting audio from video files.
    
    Provides basic audio extraction using ffmpeg with minimal complexity
    and good error handling. Focuses on the essential use case of getting
    audio from video files for transcription.
    """
    
    # Supported input formats
    SUPPORTED_FORMATS = {
        # Video formats
        '.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v', '.flv',
        # Audio formats (no processing needed)
        '.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac'
    }
    
    # Audio formats that don't need extraction
    AUDIO_FORMATS = {'.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac'}
    
    def __init__(self):
        """Initialize the simple audio processor."""
        self.temp_files: list[Path] = []
        
        # Quality and size thresholds
        self.max_file_size_gb = 10.0  # Maximum input file size
        self.large_file_threshold_mb = 500  # Warn about large files
        self.max_chunk_size_mb = 20  # Maximum chunk size for processing
        
        logger.debug("SimpleAudioProcessor initialized with enhanced validation")
    
    def validate_file(self, file_path: Path) -> bool:
        """Validate input file format and comprehensive properties.
        
        Args:
            file_path: Path to the input file
            
        Returns:
            True if file is valid and supported
            
        Raises:
            ValidationError: If file is invalid or unsupported
        """
        # Basic file existence and type validation
        if not file_path.exists():
            raise ValidationError(f"File not found: {file_path}")
        
        if not file_path.is_file():
            raise ValidationError(f"Path is not a file: {file_path}")
        
        # File size validation with detailed feedback
        file_size_bytes = file_path.stat().st_size
        if file_size_bytes == 0:
            raise ValidationError(f"File is empty: {file_path}")
        
        file_size_mb = file_size_bytes / (1024**2)
        file_size_gb = file_size_mb / 1024
        
        # Size warnings and limits (Task 3: Enhanced validation)
        if file_size_gb > self.max_file_size_gb:
            raise ValidationError(
                f"File too large: {file_size_gb:.1f}GB "
                f"(maximum: {self.max_file_size_gb:.1f}GB)"
            )
        elif file_size_mb > self.large_file_threshold_mb:
            logger.warning(
                f"Large file detected: {file_size_mb:.1f}MB - "
                f"processing may take significant time and memory"
            )
            
            # Suggest alternatives for very large files
            if file_size_mb > 1000:  # >1GB
                logger.warning(
                    "Consider splitting large files or using dedicated transcription services "
                    "for files over 1GB for better performance"
                )
        
        # Format validation
        file_ext = file_path.suffix.lower()
        if file_ext not in self.SUPPORTED_FORMATS:
            supported = ', '.join(sorted(self.SUPPORTED_FORMATS))
            raise ValidationError(
                f"Unsupported format: {file_ext}. "
                f"Supported formats: {supported}"
            )
        
        # File accessibility check
        try:
            with open(file_path, 'rb') as f:
                # Try to read first few bytes to ensure file is accessible
                f.read(1024)
        except PermissionError:
            raise ValidationError(f"Permission denied reading file: {file_path}")
        except OSError as e:
            raise ValidationError(f"Cannot read file {file_path}: {e}")
        
        # Enhanced format detection and validation
        try:
            audio_info = self._probe_file_quality(file_path)
            if audio_info:
                self._log_quality_assessment(file_path, audio_info)
        except Exception as e:
            # Don't fail validation if probe fails, just log warning
            logger.warning(f"Could not analyze file quality: {e}")
        
        logger.debug(f"File validation passed: {file_path} ({file_size_mb:.1f}MB)")
        return True
    
    def is_audio_file(self, file_path: Path) -> bool:
        """Check if file is already in audio format.
        
        Args:
            file_path: Path to check
            
        Returns:
            True if file is already audio format
        """
        return file_path.suffix.lower() in self.AUDIO_FORMATS
    
    def extract_audio(self, input_path: Path, output_path: Optional[Path] = None, keep_audio: bool = False) -> Path:
        """Extract audio from video file using ffmpeg.
        
        Args:
            input_path: Path to input video/audio file
            output_path: Path for extracted audio (optional, creates temp file if None)
            keep_audio: Save audio file next to video with same basename, reuse existing if found
            
        Returns:
            Path to extracted audio file
            
        Raises:
            ValidationError: If input file is invalid
            ProcessingError: If audio extraction fails
        """
        # Validate input file first
        self.validate_file(input_path)
        
        # If already audio, return as-is
        if self.is_audio_file(input_path):
            logger.debug(f"File is already audio format: {input_path}")
            return input_path
        
        # Create output path if not provided
        if output_path is None:
            if keep_audio:
                # Save audio next to video with same basename
                output_path = input_path.parent / f"{input_path.stem}.mp3"
                
                # Check if audio file already exists
                if output_path.exists():
                    logger.info(f"Reusing existing audio file: {output_path}")
                    # Validate the existing file is accessible and non-empty
                    if output_path.stat().st_size > 0:
                        return output_path
                    else:
                        logger.warning(f"Existing audio file is empty, will recreate: {output_path}")
            else:
                # Use temporary file
                temp_dir = Path(tempfile.mkdtemp())
                output_path = temp_dir / f"{input_path.stem}.mp3"
                self.temp_files.append(output_path)
                self.temp_files.append(temp_dir)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Build ffmpeg command for audio extraction
        cmd = [
            'ffmpeg',
            '-i', str(input_path),
            '-vn',  # No video stream
            '-acodec', 'libmp3lame',  # MP3 format with LAME encoder
            '-b:a', '128k',  # 128kbps bitrate for good quality/size balance
            '-ar', '16000',  # 16kHz sample rate (optimal for speech AI models)
            '-ac', '1',  # Mono audio (sufficient for transcription)
            '-y',  # Overwrite output file if exists
            '-loglevel', 'warning',  # Reduce ffmpeg verbosity
            str(output_path)
        ]
        
        logger.info(f"Extracting audio: {input_path.name} -> {output_path.name}")
        logger.debug(f"ffmpeg command: {' '.join(cmd)}")
        
        try:
            # Run ffmpeg with timeout
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout for large files
            )
            
            # Verify output file was created and is not empty
            if not output_path.exists():
                raise ProcessingError("ffmpeg completed but no output file was created")
            
            if output_path.stat().st_size == 0:
                raise ProcessingError("ffmpeg created empty output file")
            
            # Validate output size and quality (Task 3: Enhanced validation)
            self.validate_output_size(output_path)
            
            # Log success with quality info
            file_size_mb = output_path.stat().st_size / (1024**2)
            if keep_audio:
                logger.info(f"Audio extraction successful (kept): {output_path.name} ({file_size_mb:.1f}MB)")
            else:
                logger.info(f"Audio extraction successful: {output_path.name} ({file_size_mb:.1f}MB)")
            
            # Additional quality check on extracted audio
            try:
                extracted_info = self._probe_file_quality(output_path)
                if extracted_info:
                    logger.debug(f"Extracted audio quality validated: {extracted_info.get('codec_name', 'unknown')} "
                               f"@ {extracted_info.get('sample_rate', 'unknown')}Hz")
            except Exception as e:
                logger.debug(f"Could not validate extracted audio quality: {e}")
            
            return output_path
            
        except subprocess.CalledProcessError as e:
            error_msg = f"ffmpeg failed with return code {e.returncode}"
            if e.stderr:
                error_msg += f": {e.stderr.strip()}"
            logger.error(error_msg)
            raise ProcessingError(f"Audio extraction failed: {error_msg}")
            
        except subprocess.TimeoutExpired:
            error_msg = "ffmpeg timed out after 10 minutes"
            logger.error(error_msg)
            raise ProcessingError(f"Audio extraction failed: {error_msg}")
            
        except Exception as e:
            error_msg = f"Unexpected error during audio extraction: {e}"
            logger.error(error_msg)
            raise ProcessingError(error_msg)
    
    def _probe_file_quality(self, file_path: Path) -> Optional[dict]:
        """Probe file for quality information using ffprobe.
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            Dictionary with quality information, or None if probe fails
        """
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_streams',
                '-select_streams', 'a:0',  # First audio stream
                str(file_path)
            ]
            
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=30  # Quick probe timeout
            )
            
            import json
            probe_data = json.loads(result.stdout)
            
            if 'streams' in probe_data and probe_data['streams']:
                stream = probe_data['streams'][0]
                
                return {
                    'codec_name': stream.get('codec_name', 'unknown'),
                    'sample_rate': int(stream.get('sample_rate', 0)) if stream.get('sample_rate') else None,
                    'channels': int(stream.get('channels', 0)) if stream.get('channels') else None,
                    'bit_rate': int(stream.get('bit_rate', 0)) if stream.get('bit_rate') else None,
                    'duration': float(stream.get('duration', 0)) if stream.get('duration') else None,
                    'codec_long_name': stream.get('codec_long_name', 'Unknown codec')
                }
            
            return None
            
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError, KeyError):
            return None
    
    def _log_quality_assessment(self, file_path: Path, audio_info: dict) -> None:
        """Log quality assessment and warnings for the audio file.
        
        Args:
            file_path: Path to the file
            audio_info: Audio information from probe
        """
        try:
            codec = audio_info.get('codec_name', 'unknown')
            sample_rate = audio_info.get('sample_rate')
            channels = audio_info.get('channels')
            bit_rate = audio_info.get('bit_rate')
            duration = audio_info.get('duration')
            
            # Log basic info
            info_parts = [f"codec: {codec}"]
            if sample_rate:
                info_parts.append(f"sample_rate: {sample_rate}Hz")
            if channels:
                channel_desc = "mono" if channels == 1 else f"{channels}ch"
                info_parts.append(f"channels: {channel_desc}")
            if duration:
                info_parts.append(f"duration: {duration:.1f}s")
            
            logger.debug(f"Audio quality: {file_path.name} - {', '.join(info_parts)}")
            
            # Quality warnings
            if sample_rate and sample_rate < 16000:
                logger.warning(f"Low sample rate detected: {sample_rate}Hz - may affect transcription quality")
                logger.warning("Consider using source material with at least 16kHz sample rate")
            
            if sample_rate and sample_rate > 48000:
                logger.info(f"High sample rate: {sample_rate}Hz - will be downsampled to 16kHz for efficiency")
            
            if bit_rate and bit_rate < 64000:  # <64kbps
                logger.warning(f"Low bit rate: {bit_rate//1000}kbps - may affect transcription accuracy")
            
            if channels and channels > 2:
                logger.info(f"Multi-channel audio ({channels}ch) - will be converted to mono for transcription")
            
            # Duration-based warnings
            if duration:
                duration_minutes = duration / 60
                if duration_minutes > 120:  # 2 hours
                    logger.warning(f"Very long audio: {duration_minutes:.1f} minutes - consider splitting for better performance")
                elif duration_minutes > 60:  # 1 hour
                    logger.info(f"Long audio file: {duration_minutes:.1f} minutes - processing may take some time")
            
        except Exception as e:
            logger.debug(f"Error in quality assessment logging: {e}")
    
    def validate_output_size(self, output_path: Path) -> bool:
        """Validate that extracted audio is within reasonable size limits.
        
        Args:
            output_path: Path to the extracted audio file
            
        Returns:
            True if size is acceptable
            
        Raises:
            ProcessingError: If output is too large or suspicious
        """
        try:
            if not output_path.exists():
                raise ProcessingError("Output file does not exist")
            
            file_size_mb = output_path.stat().st_size / (1024**2)
            
            # Check for oversized audio output
            if file_size_mb > self.max_chunk_size_mb:
                logger.warning(
                    f"Extracted audio is large: {file_size_mb:.1f}MB "
                    f"(>chunk limit: {self.max_chunk_size_mb}MB)"
                )
                logger.warning("Consider splitting the source material for better processing")
                
                # Don't fail, but suggest chunking
                if file_size_mb > self.max_chunk_size_mb * 2:  # Really oversized
                    logger.error(f"Audio file too large for processing: {file_size_mb:.1f}MB")
                    raise ProcessingError(
                        f"Extracted audio exceeds processing limit: {file_size_mb:.1f}MB "
                        f"(max: {self.max_chunk_size_mb * 2}MB)"
                    )
            
            return True
            
        except Exception as e:
            raise ProcessingError(f"Output validation failed: {e}")
    
    def cleanup_temp_files(self, force: bool = False) -> None:
        """Clean up temporary files created during processing.
        
        Args:
            force: Force cleanup even if files might still be in use
        """
        cleaned_count = 0
        errors = []
        
        for temp_path in self.temp_files.copy():
            try:
                if temp_path.exists():
                    if temp_path.is_file():
                        temp_path.unlink()
                        cleaned_count += 1
                    elif temp_path.is_dir():
                        shutil.rmtree(temp_path)
                        cleaned_count += 1
                
                # Remove from tracking list
                self.temp_files.remove(temp_path)
                
            except Exception as e:
                error_msg = f"Failed to cleanup {temp_path}: {e}"
                errors.append(error_msg)
                logger.warning(error_msg)
                
                # Remove from list anyway if force cleanup
                if force and temp_path in self.temp_files:
                    self.temp_files.remove(temp_path)
        
        if cleaned_count > 0:
            logger.debug(f"Cleaned up {cleaned_count} temporary files")
        
        if errors and not force:
            logger.warning(f"{len(errors)} temp files could not be cleaned up")
    
    def get_supported_formats(self) -> Set[str]:
        """Get set of supported file formats.
        
        Returns:
            Set of supported file extensions (including the dot)
        """
        return self.SUPPORTED_FORMATS.copy()
    
    def get_audio_info(self, file_path: Path) -> dict:
        """Get basic audio file information using ffprobe.
        
        Args:
            file_path: Path to audio/video file
            
        Returns:
            Dictionary with audio information
            
        Raises:
            ProcessingError: If ffprobe fails
        """
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_streams',
            '-select_streams', 'a:0',  # First audio stream
            str(file_path)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            import json
            data = json.loads(result.stdout)
            
            if not data.get('streams'):
                raise ProcessingError("No audio streams found in file")
            
            stream = data['streams'][0]
            
            return {
                'codec': stream.get('codec_name', 'unknown'),
                'duration': float(stream.get('duration', 0)),
                'sample_rate': int(stream.get('sample_rate', 0)),
                'channels': int(stream.get('channels', 0)),
                'bit_rate': int(stream.get('bit_rate', 0))
            }
            
        except subprocess.CalledProcessError as e:
            raise ProcessingError(f"Failed to get audio info: {e}")
        except json.JSONDecodeError:
            raise ProcessingError("Failed to parse ffprobe output")
        except Exception as e:
            raise ProcessingError(f"Unexpected error getting audio info: {e}")
    
    def split_large_audio(
        self, 
        audio_path: Path, 
        chunk_duration_minutes: int = 10,
        overlap_seconds: float = 1.0
    ) -> list[Path]:
        """Split large audio files into manageable chunks with natural boundaries.
        
        Args:
            audio_path: Path to the audio file to split
            chunk_duration_minutes: Target duration for each chunk in minutes
            overlap_seconds: Overlap between chunks to avoid cutting words
            
        Returns:
            List of paths to audio chunk files
            
        Raises:
            ProcessingError: If splitting fails
        """
        try:
            # Get audio duration first
            audio_info = self._probe_file_quality(audio_path)
            if not audio_info or not audio_info.get('duration'):
                raise ProcessingError("Cannot determine audio duration for splitting")
            
            total_duration = audio_info['duration']
            chunk_duration = chunk_duration_minutes * 60  # Convert to seconds
            
            # If file is small enough, no splitting needed
            if total_duration <= chunk_duration * 1.2:  # 20% buffer
                logger.debug(f"File duration {total_duration:.1f}s within chunk size, no splitting needed")
                return [audio_path]
            
            # Calculate chunk parameters
            num_chunks = int((total_duration - overlap_seconds) / (chunk_duration - overlap_seconds)) + 1
            chunk_paths = []
            
            logger.info(f"Splitting audio into {num_chunks} chunks of ~{chunk_duration_minutes} minutes each")
            
            # Create temp directory for chunks
            temp_dir = Path(tempfile.mkdtemp())
            self.temp_files.append(temp_dir)
            
            for i in range(num_chunks):
                # Calculate start and end times with smart boundary detection
                start_time = max(0, i * (chunk_duration - overlap_seconds))
                end_time = min(total_duration, start_time + chunk_duration)
                
                # For the last chunk, take everything remaining
                if i == num_chunks - 1:
                    end_time = total_duration
                
                # Skip if chunk would be too short
                if end_time - start_time < 30:  # Less than 30 seconds
                    logger.debug(f"Skipping short chunk {i+1}: {end_time - start_time:.1f}s")
                    continue
                
                # Create chunk file
                chunk_path = temp_dir / f"{audio_path.stem}_chunk_{i+1:03d}.mp3"
                
                # Use ffmpeg to extract chunk with fade to avoid clicks
                cmd = [
                    'ffmpeg',
                    '-i', str(audio_path),
                    '-ss', str(start_time),  # Start time
                    '-t', str(end_time - start_time),  # Duration
                    '-acodec', 'libmp3lame',  # MP3 format with LAME encoder
                    '-b:a', '128k',  # 128kbps bitrate for good quality/size balance
                    '-ar', '16000',  # 16kHz sample rate
                    '-ac', '1',  # Mono
                    '-af', f'afade=in:st=0:d=0.1,afade=out:st={end_time - start_time - 0.1}:d=0.1',  # Fade in/out
                    '-y',  # Overwrite
                    '-loglevel', 'warning',
                    str(chunk_path)
                ]
                
                logger.debug(f"Creating chunk {i+1}/{num_chunks}: {start_time:.1f}s - {end_time:.1f}s")
                
                result = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout per chunk
                )
                
                # Verify chunk was created
                if chunk_path.exists() and chunk_path.stat().st_size > 0:
                    chunk_paths.append(chunk_path)
                    self.temp_files.append(chunk_path)
                    
                    chunk_size_mb = chunk_path.stat().st_size / (1024**2)
                    logger.debug(f"Chunk {i+1} created: {chunk_path.name} ({chunk_size_mb:.1f}MB)")
                else:
                    logger.warning(f"Failed to create chunk {i+1}")
            
            if not chunk_paths:
                raise ProcessingError("No audio chunks were successfully created")
            
            logger.info(f"Audio splitting complete: {len(chunk_paths)} chunks created")
            return chunk_paths
            
        except subprocess.CalledProcessError as e:
            error_msg = f"ffmpeg failed during audio splitting: {e.stderr if e.stderr else 'unknown error'}"
            logger.error(error_msg)
            raise ProcessingError(error_msg)
        except Exception as e:
            error_msg = f"Audio splitting failed: {e}"
            logger.error(error_msg)
            raise ProcessingError(error_msg)
    
    def __del__(self):
        """Cleanup on destruction."""
        self.cleanup_temp_files()
    
    def __repr__(self) -> str:
        """String representation."""
        return f"SimpleAudioProcessor(temp_files={len(self.temp_files)})"