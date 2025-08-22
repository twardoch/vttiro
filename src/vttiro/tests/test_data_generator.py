# this_file: src/vttiro/tests/test_data_generator.py
"""Automated test data generation framework for VTTiro testing.

This module provides utilities for generating synthetic audio and video files
for consistent testing across different environments. Generates files with
known properties for validation testing, provider testing, and CI/CD pipelines.

Key features:
- Synthetic audio generation with configurable duration, format, and properties
- Synthetic video generation with audio tracks for video processing tests
- Metadata generation for testing transcription accuracy expectations
- Deterministic generation for reproducible test results
- Multiple format support (WAV, MP3, MP4, etc.)

Used by:
- CI/CD pipeline for consistent test data
- Provider testing with known audio content
- Performance benchmarking with standardized files
- Integration testing across different file formats
"""

import wave
import struct
import math
import tempfile
import json
import hashlib
import io
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict

try:
    from moviepy.editor import AudioFileClip, ColorClip, CompositeVideoClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False

try:
    from pydub import AudioSegment
    from pydub.generators import Sine, Sawtooth, Square, Triangle
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False


@dataclass
class TestAudioMetadata:
    """Metadata for generated test audio files."""
    
    duration_seconds: float
    sample_rate: int
    channels: int
    bit_depth: int
    format: str
    content_type: str  # 'silence', 'tone', 'speech', 'music'
    frequency: Optional[float] = None  # For tone content
    expected_transcript: Optional[str] = None  # For speech content
    file_size_bytes: Optional[int] = None
    checksum: Optional[str] = None


@dataclass 
class TestVideoMetadata:
    """Metadata for generated test video files."""
    
    duration_seconds: float
    width: int
    height: int
    fps: int
    format: str
    has_audio: bool
    audio_metadata: Optional[TestAudioMetadata] = None
    file_size_bytes: Optional[int] = None
    checksum: Optional[str] = None


class SyntheticAudioGenerator:
    """Generator for synthetic audio files with known properties."""
    
    def __init__(self, temp_dir: Optional[Path] = None):
        """Initialize audio generator.
        
        Args:
            temp_dir: Directory for temporary files, uses system temp if None
        """
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir())
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_silence(
        self,
        duration: float = 5.0,
        sample_rate: int = 44100,
        channels: int = 1,
        bit_depth: int = 16,
        format: str = "wav",
        filename_prefix: str = None
    ) -> Tuple[Path, TestAudioMetadata]:
        """Generate silent audio file.
        
        Args:
            duration: Duration in seconds
            sample_rate: Sample rate in Hz
            channels: Number of audio channels
            bit_depth: Bit depth (16 or 24)
            format: Output format ('wav', 'mp3', 'm4a')
            
        Returns:
            Tuple of (file_path, metadata)
        """
        if filename_prefix:
            filename = f"{filename_prefix}.{format}"
        else:
            filename = f"silence_{duration}s_{sample_rate}hz_{channels}ch_{bit_depth}bit.{format}"
        file_path = self.temp_dir / filename
        
        # Generate raw silence data
        num_samples = int(duration * sample_rate)
        silence_data = [0] * num_samples * channels
        
        if format.lower() == 'wav':
            self._write_wav_file(file_path, silence_data, sample_rate, channels, bit_depth)
        else:
            # Use pydub for other formats if available
            if PYDUB_AVAILABLE:
                audio = AudioSegment.silent(duration=int(duration * 1000), frame_rate=sample_rate)
                if channels == 2:
                    audio = audio.set_channels(2)
                audio.export(str(file_path), format=format.lower())
            else:
                raise RuntimeError(f"pydub required for {format} format generation")
        
        metadata = TestAudioMetadata(
            duration_seconds=duration,
            sample_rate=sample_rate,
            channels=channels,
            bit_depth=bit_depth,
            format=format,
            content_type='silence',
            file_size_bytes=file_path.stat().st_size,
            checksum=self._calculate_checksum(file_path)
        )
        
        return file_path, metadata
    
    def generate_tone(
        self,
        frequency: float = 440.0,
        duration: float = 5.0,
        sample_rate: int = 44100,
        channels: int = 1,
        bit_depth: int = 16,
        format: str = "wav",
        waveform: str = "sine",
        filename_prefix: str = None
    ) -> Tuple[Path, TestAudioMetadata]:
        """Generate tone audio file.
        
        Args:
            frequency: Frequency in Hz
            duration: Duration in seconds
            sample_rate: Sample rate in Hz
            channels: Number of audio channels
            bit_depth: Bit depth (16 or 24)
            format: Output format ('wav', 'mp3', 'm4a')
            waveform: Waveform type ('sine', 'square', 'sawtooth', 'triangle')
            
        Returns:
            Tuple of (file_path, metadata)
        """
        if filename_prefix:
            filename = f"{filename_prefix}.{format}"
        else:
            filename = f"tone_{frequency}hz_{duration}s_{waveform}_{sample_rate}hz_{channels}ch.{format}"
        file_path = self.temp_dir / filename
        
        if format.lower() == 'wav':
            # Generate tone using pure Python
            num_samples = int(duration * sample_rate)
            tone_data = []
            
            for i in range(num_samples):
                t = i / sample_rate
                if waveform == 'sine':
                    sample = math.sin(2 * math.pi * frequency * t)
                elif waveform == 'square':
                    sample = 1.0 if math.sin(2 * math.pi * frequency * t) >= 0 else -1.0
                elif waveform == 'sawtooth':
                    sample = 2 * (t * frequency - math.floor(t * frequency + 0.5))
                elif waveform == 'triangle':
                    sample = 2 * abs(2 * (t * frequency - math.floor(t * frequency + 0.5))) - 1
                else:
                    sample = math.sin(2 * math.pi * frequency * t)  # Default to sine
                
                # Convert to integer range and duplicate for channels
                int_sample = int(sample * 32767)
                for _ in range(channels):
                    tone_data.append(int_sample)
            
            self._write_wav_file(file_path, tone_data, sample_rate, channels, bit_depth)
        
        else:
            # Use pydub for other formats if available
            if PYDUB_AVAILABLE:
                if waveform == 'sine':
                    audio = Sine(frequency).to_audio_segment(duration=int(duration * 1000))
                elif waveform == 'square':
                    audio = Square(frequency).to_audio_segment(duration=int(duration * 1000))
                elif waveform == 'sawtooth':
                    audio = Sawtooth(frequency).to_audio_segment(duration=int(duration * 1000))
                elif waveform == 'triangle':
                    audio = Triangle(frequency).to_audio_segment(duration=int(duration * 1000))
                else:
                    audio = Sine(frequency).to_audio_segment(duration=int(duration * 1000))
                
                if channels == 2:
                    audio = audio.set_channels(2)
                audio.export(str(file_path), format=format.lower())
            else:
                raise RuntimeError(f"pydub required for {format} format generation")
        
        metadata = TestAudioMetadata(
            duration_seconds=duration,
            sample_rate=sample_rate,
            channels=channels,
            bit_depth=bit_depth,
            format=format,
            content_type='tone',
            frequency=frequency,
            file_size_bytes=file_path.stat().st_size,
            checksum=self._calculate_checksum(file_path)
        )
        
        return file_path, metadata
    
    def generate_speech_simulation(
        self,
        transcript: str = "This is a test audio file for VTTiro transcription testing.",
        duration: float = 5.0,
        sample_rate: int = 44100,
        format: str = "wav",
        filename_prefix: str = None
    ) -> Tuple[Path, TestAudioMetadata]:
        """Generate speech-like audio simulation.
        
        This creates a synthetic audio file that simulates speech patterns
        using multiple tones at speech-like frequencies. While not actual
        speech, it provides a known expected transcript for testing.
        
        Args:
            transcript: Expected transcript text
            duration: Duration in seconds
            sample_rate: Sample rate in Hz
            format: Output format
            
        Returns:
            Tuple of (file_path, metadata)
        """
        if filename_prefix:
            filename = f"{filename_prefix}.{format}"
        else:
            filename = f"speech_sim_{len(transcript)}chars_{duration}s.{format}"
        file_path = self.temp_dir / filename
        
        # Generate speech-like frequencies (human voice range: 85-255 Hz fundamental)
        base_frequencies = [120, 150, 180, 200, 240]  # Simulate formants
        num_samples = int(duration * sample_rate)
        speech_data = []
        
        for i in range(num_samples):
            t = i / sample_rate
            # Combine multiple frequencies to simulate speech
            sample = 0.0
            for freq in base_frequencies:
                # Add some variation to simulate speech patterns
                freq_mod = freq * (1.0 + 0.1 * math.sin(2 * math.pi * 2 * t))
                sample += 0.2 * math.sin(2 * math.pi * freq_mod * t)
            
            # Add envelope to simulate word boundaries
            envelope = 0.5 * (1 + math.sin(2 * math.pi * 0.5 * t))
            sample *= envelope
            
            # Convert to integer
            int_sample = int(sample * 16383)  # Slightly quieter than max
            speech_data.append(int_sample)
        
        if format.lower() == 'wav':
            self._write_wav_file(file_path, speech_data, sample_rate, 1, 16)
        else:
            if PYDUB_AVAILABLE:
                # Convert to pydub format
                audio_data = struct.pack('<' + 'h' * len(speech_data), *speech_data)
                audio = AudioSegment(
                    audio_data,
                    frame_rate=sample_rate,
                    sample_width=2,
                    channels=1
                )
                audio.export(str(file_path), format=format.lower())
            else:
                raise RuntimeError(f"pydub required for {format} format generation")
        
        metadata = TestAudioMetadata(
            duration_seconds=duration,
            sample_rate=sample_rate,
            channels=1,
            bit_depth=16,
            format=format,
            content_type='speech',
            expected_transcript=transcript,
            file_size_bytes=file_path.stat().st_size,
            checksum=self._calculate_checksum(file_path)
        )
        
        return file_path, metadata
    
    def _write_wav_file(
        self,
        file_path: Path,
        audio_data: List[int],
        sample_rate: int,
        channels: int,
        bit_depth: int
    ):
        """Write audio data to WAV file."""
        with wave.open(str(file_path), 'wb') as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(bit_depth // 8)
            wav_file.setframerate(sample_rate)
            
            # Pack audio data
            if bit_depth == 16:
                packed_data = struct.pack('<' + 'h' * len(audio_data), *audio_data)
            elif bit_depth == 24:
                # 24-bit is more complex, pack as 32-bit and trim
                packed_data = b''
                for sample in audio_data:
                    # Convert 16-bit to 24-bit
                    sample_24 = sample * 256
                    packed_data += struct.pack('<i', sample_24)[:3]
            else:
                raise ValueError(f"Unsupported bit depth: {bit_depth}")
            
            wav_file.writeframes(packed_data)
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()


class SyntheticVideoGenerator:
    """Generator for synthetic video files with known properties."""
    
    def __init__(self, temp_dir: Optional[Path] = None):
        """Initialize video generator.
        
        Args:
            temp_dir: Directory for temporary files
        """
        if not MOVIEPY_AVAILABLE:
            raise RuntimeError("moviepy required for video generation")
        
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir())
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.audio_generator = SyntheticAudioGenerator(temp_dir)
    
    def generate_video_with_audio(
        self,
        duration: float = 5.0,
        width: int = 640,
        height: int = 480,
        fps: int = 24,
        audio_type: str = "tone",
        audio_frequency: float = 440.0,
        format: str = "mp4"
    ) -> Tuple[Path, TestVideoMetadata]:
        """Generate video file with synthetic audio track.
        
        Args:
            duration: Duration in seconds
            width: Video width in pixels
            height: Video height in pixels
            fps: Frames per second
            audio_type: Type of audio ('tone', 'silence', 'speech')
            audio_frequency: Audio frequency for tone type
            format: Output format ('mp4', 'avi', 'mov')
            
        Returns:
            Tuple of (file_path, metadata)
        """
        filename = f"video_{width}x{height}_{fps}fps_{audio_type}_{duration}s.{format}"
        file_path = self.temp_dir / filename
        
        # Generate audio track
        if audio_type == "tone":
            audio_path, audio_metadata = self.audio_generator.generate_tone(
                frequency=audio_frequency,
                duration=duration,
                format="wav"
            )
        elif audio_type == "silence":
            audio_path, audio_metadata = self.audio_generator.generate_silence(
                duration=duration,
                format="wav"
            )
        elif audio_type == "speech":
            audio_path, audio_metadata = self.audio_generator.generate_speech_simulation(
                duration=duration,
                format="wav"
            )
        else:
            raise ValueError(f"Unknown audio type: {audio_type}")
        
        # Create video with solid color and audio
        color_clip = ColorClip(size=(width, height), color=(100, 100, 200), duration=duration)
        audio_clip = AudioFileClip(str(audio_path))
        
        video = color_clip.set_audio(audio_clip)
        video.write_videofile(str(file_path), fps=fps, verbose=False, logger=None)
        
        # Clean up temporary audio file
        audio_path.unlink()
        
        video_metadata = TestVideoMetadata(
            duration_seconds=duration,
            width=width,
            height=height,
            fps=fps,
            format=format,
            has_audio=True,
            audio_metadata=audio_metadata,
            file_size_bytes=file_path.stat().st_size,
            checksum=self.audio_generator._calculate_checksum(file_path)
        )
        
        return file_path, video_metadata


class TestDataManager:
    """Manager for test data generation and caching."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize test data manager.
        
        Args:
            cache_dir: Directory for caching generated files
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir()) / "vttiro_test_data"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.audio_generator = SyntheticAudioGenerator(self.cache_dir)
        if MOVIEPY_AVAILABLE:
            self.video_generator = SyntheticVideoGenerator(self.cache_dir)
        else:
            self.video_generator = None
        
        self.metadata_file = self.cache_dir / "test_data_metadata.json"
        self.metadata = self._load_metadata()
    
    def get_test_audio(
        self,
        test_name: str,
        content_type: str = "tone",
        duration: float = 5.0,
        format: str = "wav",
        **kwargs
    ) -> Tuple[Path, TestAudioMetadata]:
        """Get or generate test audio file.
        
        Args:
            test_name: Unique name for this test case
            content_type: Type of content ('tone', 'silence', 'speech')
            duration: Duration in seconds
            format: Audio format
            **kwargs: Additional parameters for generation
            
        Returns:
            Tuple of (file_path, metadata)
        """
        # Create comprehensive cache key including all parameters
        param_hash = hashlib.md5(str(sorted(kwargs.items())).encode()).hexdigest()[:8]
        cache_key = f"{test_name}_{content_type}_{duration}s_{format}_{param_hash}"
        
        # Check if cached file exists and is valid
        if cache_key in self.metadata:
            cached_path = Path(self.metadata[cache_key]["file_path"])
            if cached_path.exists():
                metadata_dict = self.metadata[cache_key]["metadata"]
                metadata = TestAudioMetadata(**metadata_dict)
                return cached_path, metadata
        
        # Generate new file with unique filename based on cache key
        if content_type == "tone":
            file_path, metadata = self.audio_generator.generate_tone(
                duration=duration, format=format, filename_prefix=cache_key, **kwargs
            )
        elif content_type == "silence":
            file_path, metadata = self.audio_generator.generate_silence(
                duration=duration, format=format, filename_prefix=cache_key, **kwargs
            )
        elif content_type == "speech":
            file_path, metadata = self.audio_generator.generate_speech_simulation(
                duration=duration, format=format, filename_prefix=cache_key, **kwargs
            )
        else:
            raise ValueError(f"Unknown content type: {content_type}")
        
        # Cache metadata
        self.metadata[cache_key] = {
            "file_path": str(file_path),
            "metadata": asdict(metadata)
        }
        self._save_metadata()
        
        return file_path, metadata
    
    def get_test_video(
        self,
        test_name: str,
        duration: float = 5.0,
        format: str = "mp4",
        **kwargs
    ) -> Tuple[Path, TestVideoMetadata]:
        """Get or generate test video file.
        
        Args:
            test_name: Unique name for this test case
            duration: Duration in seconds
            format: Video format
            **kwargs: Additional parameters for generation
            
        Returns:
            Tuple of (file_path, metadata)
        """
        if not self.video_generator:
            raise RuntimeError("Video generation not available (moviepy not installed)")
        
        cache_key = f"{test_name}_video_{duration}s_{format}"
        
        # Check if cached file exists and is valid
        if cache_key in self.metadata:
            cached_path = Path(self.metadata[cache_key]["file_path"])
            if cached_path.exists():
                metadata_dict = self.metadata[cache_key]["metadata"]
                # Reconstruct nested audio metadata
                if metadata_dict.get("audio_metadata"):
                    metadata_dict["audio_metadata"] = TestAudioMetadata(**metadata_dict["audio_metadata"])
                metadata = TestVideoMetadata(**metadata_dict)
                return cached_path, metadata
        
        # Generate new file
        file_path, metadata = self.video_generator.generate_video_with_audio(
            duration=duration, format=format, **kwargs
        )
        
        # Cache metadata
        metadata_dict = asdict(metadata)
        self.metadata[cache_key] = {
            "file_path": str(file_path),
            "metadata": metadata_dict
        }
        self._save_metadata()
        
        return file_path, metadata
    
    def cleanup_cache(self, max_age_days: int = 7):
        """Clean up old cached files.
        
        Args:
            max_age_days: Maximum age of files to keep
        """
        import time
        current_time = time.time()
        cutoff_time = current_time - (max_age_days * 24 * 3600)
        
        files_to_remove = []
        for cache_key, data in self.metadata.items():
            file_path = Path(data["file_path"])
            if file_path.exists():
                file_mtime = file_path.stat().st_mtime
                if file_mtime < cutoff_time:
                    file_path.unlink()
                    files_to_remove.append(cache_key)
        
        # Remove from metadata
        for cache_key in files_to_remove:
            del self.metadata[cache_key]
        
        self._save_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load metadata from cache file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {}
    
    def _save_metadata(self):
        """Save metadata to cache file."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except IOError:
            pass  # Ignore save errors


# Convenience functions for common test cases
def get_test_data_manager() -> TestDataManager:
    """Get global test data manager instance."""
    global _test_data_manager
    if '_test_data_manager' not in globals():
        _test_data_manager = TestDataManager()
    return _test_data_manager


def generate_provider_test_files() -> Dict[str, Dict[str, Path]]:
    """Generate standard test files for all providers.
    
    Returns:
        Dictionary mapping provider names to test file paths
    """
    manager = get_test_data_manager()
    test_files = {}
    
    # Standard test cases for each provider
    test_cases = [
        ("small_tone", {"content_type": "tone", "duration": 2.0, "format": "wav"}),
        ("medium_speech", {"content_type": "speech", "duration": 10.0, "format": "mp3"}),
        ("large_silence", {"content_type": "silence", "duration": 30.0, "format": "m4a"}),
    ]
    
    providers = ["openai", "gemini", "assemblyai", "deepgram"]
    
    for provider in providers:
        test_files[provider] = {}
        for test_name, params in test_cases:
            file_path, metadata = manager.get_test_audio(
                f"{provider}_{test_name}", **params
            )
            test_files[provider][test_name] = file_path
    
    return test_files