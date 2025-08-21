#!/usr/bin/env python3
# this_file: src/vttiro/processing/optimized_audio.py
"""Memory-efficient and optimized audio processing with streaming and chunking."""

import asyncio
import numpy as np
import gc
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union, AsyncIterator, Dict, Any
from dataclasses import dataclass
import time
import psutil
from contextlib import contextmanager

try:
    from loguru import logger
except ImportError:
    import logging as logger

try:
    import librosa
    import soundfile as sf
    import scipy.signal
    AUDIO_LIBS_AVAILABLE = True
except ImportError:
    AUDIO_LIBS_AVAILABLE = False
    logger.warning("Audio processing libraries not available. Install with: pip install librosa soundfile scipy")

from vttiro.utils.exceptions import ProcessingError, ResourceError


@dataclass
class AudioSegment:
    """Represents an audio segment with metadata."""
    data: np.ndarray
    sample_rate: int
    start_time: float
    end_time: float
    channel_count: int = 1
    bit_depth: int = 16
    
    @property
    def duration(self) -> float:
        """Get duration in seconds."""
        return self.end_time - self.start_time
    
    @property
    def sample_count(self) -> int:
        """Get number of samples."""
        return len(self.data)
    
    @property
    def memory_usage_mb(self) -> float:
        """Get memory usage in MB."""
        return self.data.nbytes / (1024 * 1024)
    
    def normalize(self, target_db: float = -23.0) -> 'AudioSegment':
        """Normalize audio to target loudness.
        
        Args:
            target_db: Target loudness in dB
            
        Returns:
            Normalized audio segment
        """
        if not AUDIO_LIBS_AVAILABLE:
            return self
        
        # Calculate RMS and normalize
        rms = np.sqrt(np.mean(self.data ** 2))
        if rms > 0:
            target_rms = 10 ** (target_db / 20)
            normalized_data = self.data * (target_rms / rms)
            # Prevent clipping
            normalized_data = np.clip(normalized_data, -1.0, 1.0)
        else:
            normalized_data = self.data
        
        return AudioSegment(
            data=normalized_data,
            sample_rate=self.sample_rate,
            start_time=self.start_time,
            end_time=self.end_time,
            channel_count=self.channel_count,
            bit_depth=self.bit_depth
        )
    
    def apply_bandpass_filter(self, low_freq: float = 80.0, high_freq: float = 8000.0) -> 'AudioSegment':
        """Apply bandpass filter to improve speech recognition.
        
        Args:
            low_freq: Low frequency cutoff in Hz
            high_freq: High frequency cutoff in Hz
            
        Returns:
            Filtered audio segment
        """
        if not AUDIO_LIBS_AVAILABLE:
            return self
        
        try:
            # Design bandpass filter
            nyquist = self.sample_rate / 2
            low_normalized = low_freq / nyquist
            high_normalized = min(high_freq / nyquist, 0.99)
            
            sos = scipy.signal.butter(4, [low_normalized, high_normalized], btype='band', output='sos')
            filtered_data = scipy.signal.sosfilt(sos, self.data)
            
            return AudioSegment(
                data=filtered_data,
                sample_rate=self.sample_rate,
                start_time=self.start_time,
                end_time=self.end_time,
                channel_count=self.channel_count,
                bit_depth=self.bit_depth
            )
        except Exception as e:
            logger.warning(f"Filter application failed: {e}")
            return self
    
    def reduce_noise(self, noise_gate_db: float = -40.0) -> 'AudioSegment':
        """Apply simple noise reduction.
        
        Args:
            noise_gate_db: Noise gate threshold in dB
            
        Returns:
            Noise-reduced audio segment
        """
        # Simple noise gate based on amplitude
        threshold = 10 ** (noise_gate_db / 20)
        noise_reduced_data = np.where(np.abs(self.data) > threshold, self.data, 0)
        
        return AudioSegment(
            data=noise_reduced_data,
            sample_rate=self.sample_rate,
            start_time=self.start_time,
            end_time=self.end_time,
            channel_count=self.channel_count,
            bit_depth=self.bit_depth
        )


class MemoryEfficientAudioLoader:
    """Memory-efficient audio loader with streaming capabilities."""
    
    def __init__(self, chunk_size_seconds: float = 30.0, overlap_seconds: float = 1.0):
        """Initialize audio loader.
        
        Args:
            chunk_size_seconds: Size of each chunk in seconds
            overlap_seconds: Overlap between chunks in seconds
        """
        if not AUDIO_LIBS_AVAILABLE:
            raise ProcessingError("Audio processing libraries not available")
        
        self.chunk_size_seconds = chunk_size_seconds
        self.overlap_seconds = overlap_seconds
    
    @contextmanager
    def memory_monitor(self, operation_name: str):
        """Context manager for monitoring memory usage.
        
        Args:
            operation_name: Name of the operation being monitored
        """
        start_memory = psutil.virtual_memory().used
        start_time = time.perf_counter()
        
        try:
            yield
        finally:
            end_memory = psutil.virtual_memory().used
            end_time = time.perf_counter()
            
            memory_increase = (end_memory - start_memory) / (1024 * 1024)  # MB
            duration = end_time - start_time
            
            logger.debug(
                f"{operation_name}: {duration:.2f}s, memory delta: {memory_increase:.1f}MB"
            )
    
    def get_audio_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Get audio file information with caching.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Audio file metadata
        """
        file_path = Path(file_path)
        
        try:
            with sf.SoundFile(str(file_path)) as audio_file:
                return {
                    'duration': len(audio_file) / audio_file.samplerate,
                    'sample_rate': audio_file.samplerate,
                    'channels': audio_file.channels,
                    'frames': len(audio_file),
                    'format': audio_file.format,
                    'subtype': audio_file.subtype,
                    'file_size_mb': file_path.stat().st_size / (1024 * 1024)
                }
        except Exception as e:
            raise ProcessingError(f"Failed to get audio info: {e}", cause=e)
    
    def stream_audio_chunks(
        self, 
        file_path: Union[str, Path],
        target_sample_rate: Optional[int] = None
    ) -> Iterator[AudioSegment]:
        """Stream audio file in chunks for memory-efficient processing.
        
        Args:
            file_path: Path to audio file
            target_sample_rate: Target sample rate for resampling
            
        Yields:
            Audio segments
        """
        file_path = Path(file_path)
        
        with self.memory_monitor(f"stream_audio_chunks: {file_path.name}"):
            try:
                # Get file info first
                info = self.get_audio_info(file_path)
                original_sample_rate = info['sample_rate']
                duration = info['duration']
                
                # Calculate chunk parameters
                chunk_samples = int(self.chunk_size_seconds * original_sample_rate)
                overlap_samples = int(self.overlap_seconds * original_sample_rate)
                
                logger.info(
                    f"Streaming audio: {file_path.name}, "
                    f"duration: {duration:.1f}s, "
                    f"sr: {original_sample_rate}Hz, "
                    f"chunk_size: {self.chunk_size_seconds}s"
                )
                
                # Stream chunks
                with sf.SoundFile(str(file_path)) as audio_file:
                    current_time = 0.0
                    chunk_index = 0
                    
                    while current_time < duration:
                        # Calculate chunk boundaries
                        start_sample = int(current_time * original_sample_rate)
                        end_sample = min(start_sample + chunk_samples, len(audio_file))
                        
                        # Seek to start position
                        audio_file.seek(start_sample)
                        
                        # Read chunk
                        chunk_data = audio_file.read(end_sample - start_sample)
                        
                        if len(chunk_data) == 0:
                            break
                        
                        # Convert to mono if needed
                        if len(chunk_data.shape) > 1:
                            chunk_data = np.mean(chunk_data, axis=1)
                        
                        # Resample if needed
                        if target_sample_rate and target_sample_rate != original_sample_rate:
                            chunk_data = librosa.resample(
                                chunk_data, 
                                orig_sr=original_sample_rate, 
                                target_sr=target_sample_rate
                            )
                            sample_rate = target_sample_rate
                        else:
                            sample_rate = original_sample_rate
                        
                        # Calculate time boundaries
                        chunk_duration = len(chunk_data) / sample_rate
                        end_time = current_time + chunk_duration
                        
                        # Create audio segment
                        segment = AudioSegment(
                            data=chunk_data.astype(np.float32),
                            sample_rate=sample_rate,
                            start_time=current_time,
                            end_time=end_time,
                            channel_count=1
                        )
                        
                        yield segment
                        
                        # Update position (with overlap)
                        current_time += self.chunk_size_seconds - self.overlap_seconds
                        chunk_index += 1
                        
                        # Force garbage collection periodically
                        if chunk_index % 10 == 0:
                            gc.collect()
                
            except Exception as e:
                raise ProcessingError(f"Failed to stream audio chunks: {e}", cause=e)
    
    async def stream_audio_chunks_async(
        self, 
        file_path: Union[str, Path],
        target_sample_rate: Optional[int] = None
    ) -> AsyncIterator[AudioSegment]:
        """Async version of stream_audio_chunks.
        
        Args:
            file_path: Path to audio file
            target_sample_rate: Target sample rate for resampling
            
        Yields:
            Audio segments
        """
        loop = asyncio.get_event_loop()
        
        # Run streaming in thread pool to avoid blocking
        def sync_generator():
            return self.stream_audio_chunks(file_path, target_sample_rate)
        
        generator = await loop.run_in_executor(None, sync_generator)
        
        for segment in generator:
            yield segment
            # Allow other coroutines to run
            await asyncio.sleep(0)
    
    def load_audio_optimized(
        self, 
        file_path: Union[str, Path],
        target_sample_rate: int = 16000,
        max_duration: Optional[float] = None,
        apply_preprocessing: bool = True
    ) -> AudioSegment:
        """Load audio file with optimizations.
        
        Args:
            file_path: Path to audio file
            target_sample_rate: Target sample rate
            max_duration: Maximum duration to load (None for full file)
            apply_preprocessing: Whether to apply preprocessing
            
        Returns:
            Loaded and processed audio segment
        """
        file_path = Path(file_path)
        
        with self.memory_monitor(f"load_audio_optimized: {file_path.name}"):
            try:
                # Simple audio loading without caching
                
                # Get file info
                info = self.get_audio_info(file_path)
                original_sample_rate = info['sample_rate']
                duration = info['duration']
                
                # Calculate load parameters
                if max_duration:
                    load_duration = min(max_duration, duration)
                    load_samples = int(load_duration * original_sample_rate)
                else:
                    load_duration = duration
                    load_samples = None
                
                # Load audio
                data, sr = librosa.load(
                    str(file_path),
                    sr=target_sample_rate,
                    mono=True,
                    duration=load_duration,
                    dtype=np.float32
                )
                
                # Create segment
                segment = AudioSegment(
                    data=data,
                    sample_rate=sr,
                    start_time=0.0,
                    end_time=len(data) / sr,
                    channel_count=1
                )
                
                # Apply preprocessing if requested
                if apply_preprocessing:
                    segment = self.apply_preprocessing(segment)
                
                logger.info(
                    f"Loaded audio: {file_path.name}, "
                    f"duration: {segment.duration:.1f}s, "
                    f"memory: {segment.memory_usage_mb:.1f}MB"
                )
                
                return segment
                
            except Exception as e:
                raise ProcessingError(f"Failed to load audio: {e}", cause=e)
    
    def apply_preprocessing(self, segment: AudioSegment) -> AudioSegment:
        """Apply standard preprocessing to audio segment.
        
        Args:
            segment: Input audio segment
            
        Returns:
            Preprocessed audio segment
        """
        try:
            # Apply preprocessing pipeline
            processed = segment
            
            # 1. Normalize loudness
            processed = processed.normalize(target_db=-23.0)
            
            # 2. Apply bandpass filter for speech
            processed = processed.apply_bandpass_filter(low_freq=80.0, high_freq=8000.0)
            
            # 3. Reduce noise
            processed = processed.reduce_noise(noise_gate_db=-40.0)
            
            logger.debug(f"Applied preprocessing to {segment.duration:.1f}s segment")
            return processed
            
        except Exception as e:
            logger.warning(f"Preprocessing failed: {e}")
            return segment


class EnergyBasedSegmenter:
    """Energy-based audio segmentation for intelligent chunking."""
    
    def __init__(
        self,
        frame_length: int = 2048,
        hop_length: int = 512,
        energy_threshold: float = 0.01,
        min_silence_duration: float = 0.5,
        min_segment_duration: float = 5.0,
        max_segment_duration: float = 30.0
    ):
        """Initialize energy-based segmenter.
        
        Args:
            frame_length: Frame length for energy analysis
            hop_length: Hop length for energy analysis
            energy_threshold: Energy threshold for silence detection
            min_silence_duration: Minimum silence duration for split
            min_segment_duration: Minimum segment duration
            max_segment_duration: Maximum segment duration
        """
        if not AUDIO_LIBS_AVAILABLE:
            raise ProcessingError("Audio processing libraries not available")
        
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.energy_threshold = energy_threshold
        self.min_silence_duration = min_silence_duration
        self.min_segment_duration = min_segment_duration
        self.max_segment_duration = max_segment_duration
    
    def calculate_energy(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Calculate short-time energy of audio signal.
        
        Args:
            audio_data: Audio data
            sample_rate: Sample rate
            
        Returns:
            Energy values per frame
        """
        # Calculate RMS energy per frame
        rms_energy = librosa.feature.rms(
            y=audio_data,
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )[0]
        
        return rms_energy
    
    def find_silence_regions(
        self, 
        energy: np.ndarray, 
        sample_rate: int
    ) -> List[Tuple[float, float]]:
        """Find silence regions in audio based on energy.
        
        Args:
            energy: Energy values per frame
            sample_rate: Sample rate
            
        Returns:
            List of (start_time, end_time) tuples for silence regions
        """
        # Convert frame indices to time
        frame_times = librosa.frames_to_time(
            np.arange(len(energy)),
            sr=sample_rate,
            hop_length=self.hop_length
        )
        
        # Find silence frames
        silence_mask = energy < self.energy_threshold
        
        # Find continuous silence regions
        silence_regions = []
        in_silence = False
        silence_start = 0.0
        
        for i, (is_silence, time) in enumerate(zip(silence_mask, frame_times)):
            if is_silence and not in_silence:
                # Start of silence
                silence_start = time
                in_silence = True
            elif not is_silence and in_silence:
                # End of silence
                silence_duration = time - silence_start
                if silence_duration >= self.min_silence_duration:
                    silence_regions.append((silence_start, time))
                in_silence = False
        
        # Handle case where audio ends in silence
        if in_silence:
            silence_duration = frame_times[-1] - silence_start
            if silence_duration >= self.min_silence_duration:
                silence_regions.append((silence_start, frame_times[-1]))
        
        return silence_regions
    
    def segment_audio(self, segment: AudioSegment) -> List[AudioSegment]:
        """Segment audio based on energy analysis.
        
        Args:
            segment: Input audio segment
            
        Returns:
            List of audio segments
        """
        try:
            # Calculate energy
            energy = self.calculate_energy(segment.data, segment.sample_rate)
            
            # Find silence regions
            silence_regions = self.find_silence_regions(energy, segment.sample_rate)
            
            if not silence_regions:
                # No suitable silence found, split by max duration
                return self._split_by_duration(segment)
            
            # Create segments based on silence regions
            segments = []
            current_start = segment.start_time
            
            for silence_start, silence_end in silence_regions:
                # Adjust silence times to absolute time
                abs_silence_start = segment.start_time + silence_start
                abs_silence_end = segment.start_time + silence_end
                
                # Create segment before silence
                if abs_silence_start - current_start >= self.min_segment_duration:
                    # Extract data for this segment
                    start_sample = int((current_start - segment.start_time) * segment.sample_rate)
                    end_sample = int((abs_silence_start - segment.start_time) * segment.sample_rate)
                    
                    segment_data = segment.data[start_sample:end_sample]
                    
                    if len(segment_data) > 0:
                        audio_segment = AudioSegment(
                            data=segment_data,
                            sample_rate=segment.sample_rate,
                            start_time=current_start,
                            end_time=abs_silence_start,
                            channel_count=segment.channel_count,
                            bit_depth=segment.bit_depth
                        )
                        segments.append(audio_segment)
                
                # Update start for next segment
                current_start = abs_silence_end
                
                # Check if we need to split long segments
                if abs_silence_start - (segments[-1].start_time if segments else segment.start_time) > self.max_segment_duration:
                    break
            
            # Add remaining audio
            if current_start < segment.end_time:
                start_sample = int((current_start - segment.start_time) * segment.sample_rate)
                segment_data = segment.data[start_sample:]
                
                if len(segment_data) > 0:
                    audio_segment = AudioSegment(
                        data=segment_data,
                        sample_rate=segment.sample_rate,
                        start_time=current_start,
                        end_time=segment.end_time,
                        channel_count=segment.channel_count,
                        bit_depth=segment.bit_depth
                    )
                    segments.append(audio_segment)
            
            # Fallback to duration-based splitting if no good segments
            if not segments:
                segments = self._split_by_duration(segment)
            
            logger.debug(
                f"Segmented {segment.duration:.1f}s audio into {len(segments)} segments "
                f"using {len(silence_regions)} silence regions"
            )
            
            return segments
            
        except Exception as e:
            logger.warning(f"Energy-based segmentation failed: {e}")
            return self._split_by_duration(segment)
    
    def _split_by_duration(self, segment: AudioSegment) -> List[AudioSegment]:
        """Split audio segment by fixed duration as fallback.
        
        Args:
            segment: Input audio segment
            
        Returns:
            List of audio segments
        """
        segments = []
        current_time = segment.start_time
        
        while current_time < segment.end_time:
            end_time = min(current_time + self.max_segment_duration, segment.end_time)
            
            # Extract data for this segment
            start_sample = int((current_time - segment.start_time) * segment.sample_rate)
            end_sample = int((end_time - segment.start_time) * segment.sample_rate)
            
            segment_data = segment.data[start_sample:end_sample]
            
            if len(segment_data) > 0:
                audio_segment = AudioSegment(
                    data=segment_data,
                    sample_rate=segment.sample_rate,
                    start_time=current_time,
                    end_time=end_time,
                    channel_count=segment.channel_count,
                    bit_depth=segment.bit_depth
                )
                segments.append(audio_segment)
            
            current_time = end_time
        
        return segments


class OptimizedAudioProcessor:
    """High-level optimized audio processor combining all optimization techniques."""
    
    def __init__(
        self,
        chunk_size_seconds: float = 30.0,
        target_sample_rate: int = 16000,
        max_memory_mb: float = 512.0
    ):
        """Initialize optimized audio processor.
        
        Args:
            chunk_size_seconds: Chunk size for processing
            target_sample_rate: Target sample rate
            max_memory_mb: Maximum memory usage
        """
        self.chunk_size_seconds = chunk_size_seconds
        self.target_sample_rate = target_sample_rate
        self.max_memory_mb = max_memory_mb
        
        self.loader = MemoryEfficientAudioLoader(chunk_size_seconds)
        self.segmenter = EnergyBasedSegmenter(
            max_segment_duration=chunk_size_seconds
        )
        
        # Memory monitoring
        self._memory_usage_mb = 0.0
        self._peak_memory_mb = 0.0
    
    def check_memory_usage(self) -> None:
        """Check and enforce memory limits."""
        current_memory = psutil.virtual_memory().used / (1024 * 1024)
        self._memory_usage_mb = current_memory
        self._peak_memory_mb = max(self._peak_memory_mb, current_memory)
        
        available_memory = psutil.virtual_memory().available / (1024 * 1024)
        
        if available_memory < self.max_memory_mb:
            gc.collect()  # Force garbage collection
            
            # Re-check after GC
            available_memory = psutil.virtual_memory().available / (1024 * 1024)
            if available_memory < self.max_memory_mb:
                raise ResourceError(
                    f"Insufficient memory: {available_memory:.1f}MB available, "
                    f"{self.max_memory_mb:.1f}MB required"
                )
    
    async def process_audio_file(
        self, 
        file_path: Union[str, Path],
        apply_preprocessing: bool = True,
        use_energy_segmentation: bool = True
    ) -> List[AudioSegment]:
        """Process audio file with all optimizations.
        
        Args:
            file_path: Path to audio file
            apply_preprocessing: Whether to apply preprocessing
            use_energy_segmentation: Whether to use energy-based segmentation
            
        Returns:
            List of processed audio segments
        """
        file_path = Path(file_path)
        
        logger.info(f"Processing audio file: {file_path.name}")
        
        try:
            # Check memory before starting
            self.check_memory_usage()
            
            segments = []
            
            # Stream and process chunks
            async for chunk in self.loader.stream_audio_chunks_async(
                file_path, self.target_sample_rate
            ):
                # Apply preprocessing if requested
                if apply_preprocessing:
                    chunk = self.loader.apply_preprocessing(chunk)
                
                # Apply energy-based segmentation if requested
                if use_energy_segmentation:
                    chunk_segments = self.segmenter.segment_audio(chunk)
                    segments.extend(chunk_segments)
                else:
                    segments.append(chunk)
                
                # Check memory periodically
                if len(segments) % 10 == 0:
                    self.check_memory_usage()
            
            logger.info(
                f"Processed {file_path.name}: {len(segments)} segments, "
                f"peak memory: {self._peak_memory_mb:.1f}MB"
            )
            
            return segments
            
        except Exception as e:
            raise ProcessingError(f"Failed to process audio file: {e}", cause=e)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics.
        
        Returns:
            Performance statistics
        """
        return {
            "memory_usage_mb": self._memory_usage_mb,
            "peak_memory_mb": self._peak_memory_mb,
            "target_sample_rate": self.target_sample_rate,
            "chunk_size_seconds": self.chunk_size_seconds,
            "available_memory_mb": psutil.virtual_memory().available / (1024 * 1024),
        }