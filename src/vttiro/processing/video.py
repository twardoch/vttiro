#!/usr/bin/env python3
# this_file: src/vttiro/processing/video.py
"""Video processing and audio extraction using yt-dlp."""

import os
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from urllib.parse import urlparse

try:
    import yt_dlp
    from loguru import logger
    import ffmpeg
    from pydantic import BaseModel, Field
except ImportError as e:
    logger.error(f"Missing required dependency: {e}")
    raise ImportError("Required dependencies missing. Install with: uv pip install --system vttiro[local]")

from vttiro.core.config import VttiroConfig

# Import advanced segmentation engine (optional)
try:
    from vttiro.segmentation import SegmentationEngine, SegmentationConfig, AudioSegment as AdvancedAudioSegment
    ADVANCED_SEGMENTATION_AVAILABLE = True
except ImportError:
    ADVANCED_SEGMENTATION_AVAILABLE = False
    SegmentationEngine = None
    AdvancedAudioSegment = None


@dataclass
class VideoMetadata:
    """Metadata extracted from video source."""
    
    url: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    duration: Optional[float] = None  # Duration in seconds
    language: Optional[str] = None
    uploader: Optional[str] = None
    upload_date: Optional[str] = None
    view_count: Optional[int] = None
    like_count: Optional[int] = None
    channel: Optional[str] = None
    channel_id: Optional[str] = None
    thumbnail: Optional[str] = None
    categories: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    format_info: Optional[Dict[str, Any]] = None
    existing_subtitles: Optional[List[str]] = None
    quality_metrics: Optional[Dict[str, float]] = None


@dataclass
class AudioChunk:
    """Represents a segmented audio chunk."""
    
    chunk_id: str
    start_time: float  # Start time in seconds
    end_time: float    # End time in seconds
    duration: float    # Chunk duration in seconds
    file_path: Path
    energy_stats: Optional[Dict[str, float]] = None
    quality_metrics: Optional[Dict[str, float]] = None


class VideoProcessor:
    """Enhanced video processing with yt-dlp integration and smart segmentation."""
    
    def __init__(self, config: VttiroConfig, temp_dir: Optional[Path] = None):
        """Initialize video processor.
        
        Args:
            config: vttiro configuration
            temp_dir: Temporary directory for processing (optional)
        """
        self.config = config
        self.temp_dir = Path(temp_dir) if temp_dir else Path.cwd() / "temp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize advanced segmentation engine if available
        self.advanced_segmentation_engine = None
        if ADVANCED_SEGMENTATION_AVAILABLE:
            try:
                self.advanced_segmentation_engine = SegmentationEngine(config)
                logger.info("Advanced segmentation engine initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize advanced segmentation: {e}")
                logger.info("Falling back to basic energy-based segmentation")
        
        # Configure yt-dlp options
        self.yt_dlp_opts = {
            'outtmpl': str(self.temp_dir / '%(title)s.%(ext)s'),
            'format': 'bestaudio/best',
            'extract_flat': False,
            'writethumbnail': True,
            'writeinfojson': True,
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitlesformat': 'vtt',
            'ignoreerrors': True,
            'no_warnings': False,
            'extractaudio': True,
            'audioformat': 'wav',
            'audioquality': '192',
        }
        
    async def process_source(
        self,
        source: Union[str, Path],
        extract_audio: bool = True,
        segment_audio: bool = True
    ) -> Dict[str, Any]:
        """Process video source and extract audio with metadata.
        
        Args:
            source: Video URL or local file path
            extract_audio: Whether to extract audio
            segment_audio: Whether to segment audio into chunks
            
        Returns:
            Dictionary containing metadata, audio chunks, and processing info
        """
        logger.info(f"Processing video source: {source}")
        
        try:
            # Extract metadata
            metadata = await self.extract_metadata(source)
            
            # Check duration limits
            if metadata.duration and metadata.duration > self.config.processing.max_duration:
                raise ValueError(
                    f"Video duration {metadata.duration}s exceeds maximum "
                    f"{self.config.processing.max_duration}s"
                )
            
            result = {
                'source': str(source),
                'metadata': metadata,
                'audio_file': None,
                'audio_chunks': [],
                'processing_time': 0
            }
            
            if extract_audio:
                # Extract audio
                audio_file = await self.extract_audio(source, metadata)
                result['audio_file'] = audio_file
                
                if segment_audio and audio_file:
                    # Segment audio into chunks
                    chunks = await self.segment_audio(audio_file, metadata)
                    result['audio_chunks'] = chunks
                    
            return result
            
        except Exception as e:
            logger.error(f"Error processing {source}: {e}")
            raise
            
    async def extract_metadata(self, source: Union[str, Path]) -> VideoMetadata:
        """Extract comprehensive metadata from video source.
        
        Args:
            source: Video URL or local file path
            
        Returns:
            VideoMetadata object with extracted information
        """
        logger.debug(f"Extracting metadata from: {source}")
        
        if self._is_url(str(source)):
            return await self._extract_metadata_from_url(str(source))
        else:
            return await self._extract_metadata_from_file(Path(source))
            
    async def extract_audio(
        self,
        source: Union[str, Path],
        metadata: Optional[VideoMetadata] = None
    ) -> Optional[Path]:
        """Extract audio from video source.
        
        Args:
            source: Video URL or local file path
            metadata: Optional metadata for optimization
            
        Returns:
            Path to extracted audio file
        """
        logger.debug(f"Extracting audio from: {source}")
        
        try:
            if self._is_url(str(source)):
                audio_file = await self._extract_audio_from_url(str(source))
            else:
                audio_file = await self._extract_audio_from_file(Path(source))
                
            if audio_file and audio_file.exists():
                # Assess audio quality
                quality_metrics = await self._assess_audio_quality(audio_file)
                logger.debug(f"Audio quality metrics: {quality_metrics}")
                
                # Apply preprocessing if needed
                if self._should_preprocess_audio(quality_metrics):
                    audio_file = await self._preprocess_audio(audio_file, quality_metrics)
                    
            return audio_file
            
        except Exception as e:
            logger.error(f"Error extracting audio: {e}")
            raise
            
    async def segment_audio(
        self,
        audio_file: Path,
        metadata: Optional[VideoMetadata] = None,
        chunk_duration: Optional[int] = None
    ) -> List[AudioChunk]:
        """Segment audio into intelligent chunks using energy-based analysis.
        
        Args:
            audio_file: Path to audio file
            metadata: Optional metadata for context
            chunk_duration: Override default chunk duration
            
        Returns:
            List of AudioChunk objects
        """
        logger.debug(f"Segmenting audio: {audio_file}")
        
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
            
        # Use configured chunk duration or override
        max_chunk_duration = chunk_duration or self.config.processing.chunk_duration
        overlap_duration = self.config.processing.overlap_duration
        
        try:
            # Try advanced segmentation first if available
            if self.advanced_segmentation_engine:
                logger.info("Using advanced segmentation engine")
                
                # Prepare metadata for advanced segmentation
                segmentation_metadata = {}
                if metadata:
                    segmentation_metadata = {
                        'title': metadata.title,
                        'description': metadata.description,
                        'uploader': metadata.uploader,
                        'duration': metadata.duration,
                        'language': metadata.language,
                        'categories': metadata.categories,
                        'tags': metadata.tags
                    }
                    
                try:
                    # Use advanced segmentation engine
                    advanced_segments = await self.advanced_segmentation_engine.segment_audio(
                        audio_file, segmentation_metadata
                    )
                    
                    # Convert advanced segments to traditional AudioChunk format
                    chunks = []
                    for segment in advanced_segments:
                        chunk_id = segment.chunk_id or f"{audio_file.stem}_chunk_{len(chunks):03d}"
                        
                        # Extract chunk to separate file if needed
                        chunk_file = await self._extract_audio_chunk(
                            audio_file, segment.start_time, segment.end_time, chunk_id
                        )
                        
                        chunks.append(AudioChunk(
                            chunk_id=chunk_id,
                            start_time=segment.start_time,
                            end_time=segment.end_time,
                            duration=segment.duration,
                            audio_file=chunk_file,
                            overlap_start=0,  # Advanced engine handles overlap internally
                            overlap_end=0,
                            energy_stats=segment.energy_stats,
                            quality_metrics=segment.quality_metrics
                        ))
                        
                    logger.info(f"Advanced segmentation created {len(chunks)} chunks")
                    return chunks
                    
                except Exception as e:
                    logger.warning(f"Advanced segmentation failed, falling back to basic: {e}")
            
            # Fallback to basic energy-based segmentation
            logger.info("Using basic energy-based segmentation")
            
            # Load audio for analysis
            import librosa
            import numpy as np
            
            audio, sr = librosa.load(str(audio_file), sr=self.config.processing.sample_rate)
            total_duration = len(audio) / sr
            
            logger.info(f"Audio loaded: {total_duration:.2f}s at {sr}Hz")
            
            # Perform energy-based segmentation
            segment_boundaries = await self._compute_energy_based_segments(
                audio, sr, max_chunk_duration
            )
            
            # Create audio chunks
            chunks = []
            for i, (start_time, end_time) in enumerate(segment_boundaries):
                chunk_id = f"{audio_file.stem}_chunk_{i:03d}"
                
                # Add overlap except for first/last chunks
                actual_start = max(0, start_time - (overlap_duration if i > 0 else 0))
                actual_end = min(total_duration, end_time + (overlap_duration if i < len(segment_boundaries) - 1 else 0))
                
                # Extract chunk to separate file
                chunk_file = await self._extract_audio_chunk(
                    audio_file, actual_start, actual_end, chunk_id
                )
                
                if chunk_file:
                    # Calculate energy statistics for this chunk
                    chunk_audio = audio[int(actual_start * sr):int(actual_end * sr)]
                    energy_stats = self._calculate_energy_stats(chunk_audio)
                    
                    chunk = AudioChunk(
                        chunk_id=chunk_id,
                        start_time=actual_start,
                        end_time=actual_end,
                        duration=actual_end - actual_start,
                        file_path=chunk_file,
                        energy_stats=energy_stats
                    )
                    chunks.append(chunk)
                    
            logger.info(f"Created {len(chunks)} audio chunks")
            return chunks
            
        except ImportError:
            # Fallback to simple time-based segmentation if librosa not available
            logger.warning("librosa not available, using simple time-based segmentation")
            return await self._simple_time_based_segmentation(audio_file, max_chunk_duration)
            
        except Exception as e:
            logger.error(f"Error segmenting audio: {e}")
            raise
            
    async def _extract_metadata_from_url(self, url: str) -> VideoMetadata:
        """Extract metadata from video URL using yt-dlp."""
        
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            try:
                info = ydl.extract_info(url, download=False)
                
                return VideoMetadata(
                    url=url,
                    title=info.get('title'),
                    description=info.get('description'),
                    duration=info.get('duration'),
                    language=info.get('language'),
                    uploader=info.get('uploader'),
                    upload_date=info.get('upload_date'),
                    view_count=info.get('view_count'),
                    like_count=info.get('like_count'),
                    channel=info.get('channel'),
                    channel_id=info.get('channel_id'),
                    thumbnail=info.get('thumbnail'),
                    categories=info.get('categories', []),
                    tags=info.get('tags', []),
                    format_info={'available_formats': len(info.get('formats', []))},
                    existing_subtitles=list(info.get('subtitles', {}).keys()) + 
                                    list(info.get('automatic_captions', {}).keys())
                )
                
            except Exception as e:
                logger.error(f"Error extracting metadata from URL: {e}")
                raise
                
    async def _extract_metadata_from_file(self, file_path: Path) -> VideoMetadata:
        """Extract metadata from local video file using ffprobe."""
        
        try:
            probe = ffmpeg.probe(str(file_path))
            
            # Extract video stream info
            video_streams = [s for s in probe['streams'] if s['codec_type'] == 'video']
            audio_streams = [s for s in probe['streams'] if s['codec_type'] == 'audio']
            
            duration = float(probe.get('format', {}).get('duration', 0))
            
            return VideoMetadata(
                url=str(file_path),
                title=file_path.stem,
                duration=duration,
                format_info={
                    'video_streams': len(video_streams),
                    'audio_streams': len(audio_streams),
                    'format_name': probe.get('format', {}).get('format_name'),
                    'size': probe.get('format', {}).get('size')
                }
            )
            
        except Exception as e:
            logger.error(f"Error extracting metadata from file: {e}")
            # Return basic metadata
            return VideoMetadata(
                url=str(file_path),
                title=file_path.stem
            )
            
    async def _extract_audio_from_url(self, url: str) -> Optional[Path]:
        """Extract audio from URL using yt-dlp."""
        
        output_file = self.temp_dir / f"audio_{hash(url)}.wav"
        
        opts = {
            **self.yt_dlp_opts,
            'outtmpl': str(output_file.with_suffix('.%(ext)s')),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }]
        }
        
        with yt_dlp.YoutubeDL(opts) as ydl:
            try:
                ydl.download([url])
                
                # Find the extracted audio file
                for file in self.temp_dir.glob(f"audio_{hash(url)}.*"):
                    if file.suffix in ['.wav', '.mp3', '.m4a']:
                        return file
                        
                return None
                
            except Exception as e:
                logger.error(f"Error extracting audio from URL: {e}")
                raise
                
    async def _extract_audio_from_file(self, file_path: Path) -> Optional[Path]:
        """Extract audio from local video file using ffmpeg."""
        
        output_file = self.temp_dir / f"{file_path.stem}_audio.wav"
        
        try:
            (
                ffmpeg
                .input(str(file_path))
                .output(str(output_file), acodec='pcm_s16le', ar=self.config.processing.sample_rate)
                .overwrite_output()
                .run(quiet=True)
            )
            
            return output_file if output_file.exists() else None
            
        except Exception as e:
            logger.error(f"Error extracting audio from file: {e}")
            raise
            
    async def _compute_energy_based_segments(
        self,
        audio: 'np.ndarray',
        sr: int,
        max_chunk_duration: int
    ) -> List[tuple[float, float]]:
        """Compute segment boundaries using energy-based analysis."""
        
        import numpy as np
        import librosa
        
        # Calculate frame-level energy features
        hop_length = 512
        frame_length = 2048
        
        # RMS energy
        rms_energy = librosa.feature.rms(
            y=audio,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]
        
        # Convert frames to time
        frame_times = librosa.frames_to_time(
            np.arange(len(rms_energy)),
            sr=sr,
            hop_length=hop_length
        )
        
        # Find low-energy regions
        energy_threshold = np.percentile(rms_energy, self.config.processing.energy_threshold_percentile)
        low_energy_mask = rms_energy < energy_threshold
        
        # Find segments
        segments = []
        current_start = 0
        
        for i in range(len(frame_times)):
            current_time = frame_times[i]
            
            # Check if we need to create a segment
            should_segment = (
                (current_time - current_start) >= max_chunk_duration or  # Max duration reached
                (low_energy_mask[i] and (current_time - current_start) >= 60)  # Low energy + min duration
            )
            
            if should_segment:
                # Prefer integer seconds
                if self.config.processing.prefer_integer_seconds:
                    segment_end = round(current_time)
                else:
                    segment_end = current_time
                    
                segments.append((current_start, segment_end))
                current_start = segment_end
                
        # Add final segment
        if current_start < len(audio) / sr:
            final_end = len(audio) / sr
            if self.config.processing.prefer_integer_seconds:
                final_end = round(final_end)
            segments.append((current_start, final_end))
            
        logger.debug(f"Computed {len(segments)} segments using energy analysis")
        return segments
        
    async def _extract_audio_chunk(
        self,
        audio_file: Path,
        start_time: float,
        end_time: float,
        chunk_id: str
    ) -> Optional[Path]:
        """Extract a specific time range from audio file."""
        
        chunk_file = self.temp_dir / f"{chunk_id}.wav"
        
        try:
            (
                ffmpeg
                .input(str(audio_file), ss=start_time, t=(end_time - start_time))
                .output(str(chunk_file), acodec='pcm_s16le')
                .overwrite_output()
                .run(quiet=True)
            )
            
            return chunk_file if chunk_file.exists() else None
            
        except Exception as e:
            logger.error(f"Error extracting audio chunk: {e}")
            return None
            
    def _calculate_energy_stats(self, audio: 'np.ndarray') -> Dict[str, float]:
        """Calculate energy statistics for audio segment."""
        
        import numpy as np
        
        # Calculate RMS energy
        rms = np.sqrt(np.mean(audio ** 2))
        
        # Calculate peak energy
        peak = np.max(np.abs(audio))
        
        # Calculate energy distribution
        energy_percentiles = np.percentile(np.abs(audio), [10, 25, 50, 75, 90])
        
        return {
            'rms_energy': float(rms),
            'peak_energy': float(peak),
            'energy_p10': float(energy_percentiles[0]),
            'energy_p25': float(energy_percentiles[1]),
            'energy_median': float(energy_percentiles[2]),
            'energy_p75': float(energy_percentiles[3]),
            'energy_p90': float(energy_percentiles[4]),
            'dynamic_range': float(peak / (rms + 1e-8))
        }
        
    async def _assess_audio_quality(self, audio_file: Path) -> Dict[str, float]:
        """Assess audio quality metrics."""
        
        try:
            # Use ffprobe to get basic audio info
            probe = ffmpeg.probe(str(audio_file))
            audio_stream = next(s for s in probe['streams'] if s['codec_type'] == 'audio')
            
            sample_rate = int(audio_stream.get('sample_rate', 0))
            bit_rate = int(audio_stream.get('bit_rate', 0))
            
            return {
                'sample_rate': sample_rate,
                'bit_rate': bit_rate,
                'quality_score': min(1.0, (sample_rate / 44100) * (bit_rate / 320000))
            }
            
        except Exception as e:
            logger.warning(f"Could not assess audio quality: {e}")
            return {'quality_score': 0.5}  # Default moderate quality
            
    def _should_preprocess_audio(self, quality_metrics: Dict[str, float]) -> bool:
        """Determine if audio preprocessing is needed."""
        
        quality_score = quality_metrics.get('quality_score', 0.5)
        return quality_score < 0.7
        
    async def _preprocess_audio(self, audio_file: Path, quality_metrics: Dict[str, float]) -> Path:
        """Apply audio preprocessing to improve quality."""
        
        processed_file = self.temp_dir / f"{audio_file.stem}_processed.wav"
        
        try:
            # Apply noise reduction and normalization
            (
                ffmpeg
                .input(str(audio_file))
                .filter('highpass', f=80)  # Remove low-frequency noise
                .filter('lowpass', f=8000)  # Remove high-frequency noise
                .filter('dynaudnorm')  # Dynamic audio normalization
                .output(str(processed_file), acodec='pcm_s16le')
                .overwrite_output()
                .run(quiet=True)
            )
            
            return processed_file if processed_file.exists() else audio_file
            
        except Exception as e:
            logger.warning(f"Audio preprocessing failed: {e}")
            return audio_file
            
    async def _simple_time_based_segmentation(
        self,
        audio_file: Path,
        max_chunk_duration: int
    ) -> List[AudioChunk]:
        """Fallback simple time-based segmentation."""
        
        # Get audio duration
        try:
            probe = ffmpeg.probe(str(audio_file))
            duration = float(probe['format']['duration'])
        except:
            logger.warning("Could not determine audio duration")
            duration = max_chunk_duration
            
        chunks = []
        current_time = 0
        chunk_index = 0
        
        while current_time < duration:
            end_time = min(current_time + max_chunk_duration, duration)
            
            # Prefer integer seconds
            if self.config.processing.prefer_integer_seconds:
                end_time = round(end_time)
                
            chunk_id = f"{audio_file.stem}_chunk_{chunk_index:03d}"
            chunk_file = await self._extract_audio_chunk(
                audio_file, current_time, end_time, chunk_id
            )
            
            if chunk_file:
                chunk = AudioChunk(
                    chunk_id=chunk_id,
                    start_time=current_time,
                    end_time=end_time,
                    duration=end_time - current_time,
                    file_path=chunk_file
                )
                chunks.append(chunk)
                
            current_time = end_time
            chunk_index += 1
            
        return chunks
        
    def _is_url(self, source: str) -> bool:
        """Check if source is a URL."""
        
        try:
            result = urlparse(source)
            return all([result.scheme, result.netloc])
        except:
            return False
            
    def cleanup_temp_files(self) -> None:
        """Clean up temporary files."""
        
        try:
            for file in self.temp_dir.glob("*"):
                if file.is_file():
                    file.unlink()
            logger.debug("Cleaned up temporary files")
        except Exception as e:
            logger.warning(f"Error cleaning up temp files: {e}")


# Export classes for easy import
__all__ = ['VideoProcessor', 'VideoMetadata', 'AudioChunk']