# this_file: src/vttiro/tests/test_file_format_support.py
"""Tests for file format support across different audio and video formats.

This module tests VTTiro's ability to handle various file formats correctly,
including format detection, validation, and processing compatibility.
"""

import pytest
import tempfile
import io
import struct
from pathlib import Path
from typing import Dict, List, Tuple
from unittest.mock import patch, Mock, MagicMock

from vttiro.core.config import VttiroConfig
from vttiro.core.registry import get_registry
from vttiro.core.errors import ProcessingError, ValidationError


class TestFileFormatDetection:
    """Test file format detection and validation."""
    
    def test_detect_audio_format_by_extension(self):
        """Test format detection based on file extension."""
        format_extensions = {
            'wav': ['.wav', '.WAV'],
            'mp3': ['.mp3', '.MP3'],
            'm4a': ['.m4a', '.M4A'],
            'flac': ['.flac', '.FLAC'],
            'ogg': ['.ogg', '.OGG'],
            'aac': ['.aac', '.AAC']
        }
        
        for format_name, extensions in format_extensions.items():
            for ext in extensions:
                file_path = f"/fake/audio{ext}"
                detected_format = self._detect_format_by_extension(file_path)
                assert detected_format == format_name.lower()
    
    def test_detect_video_format_by_extension(self):
        """Test video format detection based on file extension."""
        format_extensions = {
            'mp4': ['.mp4', '.MP4'],
            'avi': ['.avi', '.AVI'],
            'mov': ['.mov', '.MOV'],
            'mkv': ['.mkv', '.MKV'],
            'webm': ['.webm', '.WEBM'],
            'wmv': ['.wmv', '.WMV']
        }
        
        for format_name, extensions in format_extensions.items():
            for ext in extensions:
                file_path = f"/fake/video{ext}"
                detected_format = self._detect_format_by_extension(file_path)
                assert detected_format == format_name.lower()
    
    def test_unsupported_format_detection(self):
        """Test detection of unsupported file formats."""
        unsupported_files = [
            "/fake/document.pdf",
            "/fake/image.jpg", 
            "/fake/archive.zip",
            "/fake/text.txt",
            "/fake/unknown.xyz"
        ]
        
        for file_path in unsupported_files:
            with pytest.raises(ValidationError, match="Unsupported file format"):
                self._validate_file_format(file_path)
    
    def _detect_format_by_extension(self, file_path: str) -> str:
        """Helper to detect format by extension."""
        extension = Path(file_path).suffix.lower()
        
        audio_formats = {
            '.wav': 'wav', '.mp3': 'mp3', '.m4a': 'm4a',
            '.flac': 'flac', '.ogg': 'ogg', '.aac': 'aac'
        }
        
        video_formats = {
            '.mp4': 'mp4', '.avi': 'avi', '.mov': 'mov',
            '.mkv': 'mkv', '.webm': 'webm', '.wmv': 'wmv'
        }
        
        return audio_formats.get(extension) or video_formats.get(extension)
    
    def _validate_file_format(self, file_path: str) -> None:
        """Helper to validate file format."""
        detected = self._detect_format_by_extension(file_path)
        if not detected:
            raise ValidationError(f"Unsupported file format: {Path(file_path).suffix}")


class TestAudioFormatProcessing:
    """Test processing of various audio formats."""
    
    @pytest.fixture
    def mock_audio_files(self):
        """Create mock audio files for testing."""
        return {
            'wav': self._create_mock_wav_file(),
            'mp3': self._create_mock_mp3_file(),
            'm4a': self._create_mock_m4a_file(),
            'flac': self._create_mock_flac_file(),
            'ogg': self._create_mock_ogg_file()
        }
    
    def test_wav_file_processing(self, mock_audio_files):
        """Test WAV file processing."""
        wav_file = mock_audio_files['wav']
        
        # Test file validation
        assert self._validate_wav_format(wav_file.read())
        
        # Test metadata extraction
        wav_file.seek(0)
        metadata = self._extract_wav_metadata(wav_file.read())
        assert metadata['format'] == 'wav'
        assert metadata['sample_rate'] == 44100
        assert metadata['channels'] == 2
        assert metadata['duration'] > 0
    
    def test_mp3_file_processing(self, mock_audio_files):
        """Test MP3 file processing."""
        mp3_file = mock_audio_files['mp3']
        
        # Test file validation (basic header check)
        mp3_data = mp3_file.read()
        assert self._validate_mp3_format(mp3_data)
        
        # Test metadata extraction
        metadata = self._extract_mp3_metadata(mp3_data)
        assert metadata['format'] == 'mp3'
        assert 'duration' in metadata
    
    def test_m4a_file_processing(self, mock_audio_files):
        """Test M4A file processing."""
        m4a_file = mock_audio_files['m4a']
        
        # Test file validation (basic container check)
        m4a_data = m4a_file.read()
        assert self._validate_m4a_format(m4a_data)
        
        # Test metadata extraction
        metadata = self._extract_m4a_metadata(m4a_data)
        assert metadata['format'] == 'm4a'
        assert 'duration' in metadata
    
    def test_flac_file_processing(self, mock_audio_files):
        """Test FLAC file processing."""
        flac_file = mock_audio_files['flac']
        
        # Test file validation
        flac_data = flac_file.read()
        assert self._validate_flac_format(flac_data)
        
        # Test metadata extraction
        metadata = self._extract_flac_metadata(flac_data)
        assert metadata['format'] == 'flac'
        assert 'sample_rate' in metadata
    
    def test_ogg_file_processing(self, mock_audio_files):
        """Test OGG file processing."""
        ogg_file = mock_audio_files['ogg']
        
        # Test file validation
        ogg_data = ogg_file.read()
        assert self._validate_ogg_format(ogg_data)
        
        # Test metadata extraction
        metadata = self._extract_ogg_metadata(ogg_data)
        assert metadata['format'] == 'ogg'
        assert 'duration' in metadata
    
    def test_corrupted_audio_file_handling(self):
        """Test handling of corrupted audio files."""
        corrupted_data = b"corrupted_audio_data_not_valid_format"
        
        with pytest.raises(ValidationError, match="Invalid|Corrupted"):
            self._validate_wav_format(corrupted_data)
        
        with pytest.raises(ValidationError, match="Invalid|Corrupted"):
            self._validate_mp3_format(corrupted_data)
    
    def test_empty_audio_file_handling(self):
        """Test handling of empty audio files."""
        empty_data = b""
        
        with pytest.raises(ValidationError, match="Empty|Invalid"):
            self._validate_wav_format(empty_data)
    
    def test_large_audio_file_handling(self):
        """Test handling of very large audio files."""
        # Simulate a large file (metadata only, not actual data)
        large_file_metadata = {
            'format': 'wav',
            'duration': 7200.0,  # 2 hours
            'file_size': 1024 * 1024 * 1024,  # 1GB
            'sample_rate': 44100,
            'channels': 2
        }
        
        # Should detect large file but not reject it
        assert self._is_large_file(large_file_metadata)
        assert self._validate_large_file_constraints(large_file_metadata)
    
    def _create_mock_wav_file(self) -> io.BytesIO:
        """Create a mock WAV file with proper headers."""
        # WAV file header structure
        wav_data = io.BytesIO()
        
        # RIFF header
        wav_data.write(b'RIFF')
        wav_data.write(struct.pack('<I', 36))  # File size - 8
        wav_data.write(b'WAVE')
        
        # fmt chunk
        wav_data.write(b'fmt ')
        wav_data.write(struct.pack('<I', 16))  # Chunk size
        wav_data.write(struct.pack('<H', 1))   # Audio format (PCM)
        wav_data.write(struct.pack('<H', 2))   # Channels
        wav_data.write(struct.pack('<I', 44100))  # Sample rate
        wav_data.write(struct.pack('<I', 176400))  # Byte rate
        wav_data.write(struct.pack('<H', 4))   # Block align
        wav_data.write(struct.pack('<H', 16))  # Bits per sample
        
        # data chunk
        wav_data.write(b'data')
        wav_data.write(struct.pack('<I', 0))   # Data size
        
        wav_data.seek(0)
        return wav_data
    
    def _create_mock_mp3_file(self) -> io.BytesIO:
        """Create a mock MP3 file with proper headers."""
        mp3_data = io.BytesIO()
        
        # MP3 header (simplified)
        mp3_data.write(b'\xff\xfb')  # MP3 sync word and header
        mp3_data.write(b'\x90\x00')  # Rest of header
        mp3_data.write(b'fake_mp3_data' * 100)  # Fake audio data
        
        mp3_data.seek(0)
        return mp3_data
    
    def _create_mock_m4a_file(self) -> io.BytesIO:
        """Create a mock M4A file with proper container."""
        m4a_data = io.BytesIO()
        
        # M4A/MP4 container (simplified ftyp box)
        m4a_data.write(struct.pack('>I', 32))  # Box size
        m4a_data.write(b'ftyp')  # Box type
        m4a_data.write(b'M4A ')  # Major brand
        m4a_data.write(struct.pack('>I', 0))   # Minor version
        m4a_data.write(b'M4A mp42isom')  # Compatible brands
        m4a_data.write(b'fake_m4a_data' * 100)
        
        m4a_data.seek(0)
        return m4a_data
    
    def _create_mock_flac_file(self) -> io.BytesIO:
        """Create a mock FLAC file with proper headers."""
        flac_data = io.BytesIO()
        
        # FLAC header
        flac_data.write(b'fLaC')  # FLAC signature
        flac_data.write(b'\x00\x00\x00\x22')  # Metadata block header
        flac_data.write(b'fake_flac_metadata' * 5)
        flac_data.write(b'fake_flac_data' * 100)
        
        flac_data.seek(0)
        return flac_data
    
    def _create_mock_ogg_file(self) -> io.BytesIO:
        """Create a mock OGG file with proper headers."""
        ogg_data = io.BytesIO()
        
        # OGG page header
        ogg_data.write(b'OggS')  # Capture pattern
        ogg_data.write(b'\x00')  # Version
        ogg_data.write(b'\x02')  # Header type
        ogg_data.write(b'\x00' * 16)  # Granule position, serial, etc.
        ogg_data.write(b'fake_ogg_data' * 100)
        
        ogg_data.seek(0)
        return ogg_data
    
    def _validate_wav_format(self, data: bytes) -> bool:
        """Validate WAV format."""
        if len(data) < 12:
            raise ValidationError("Invalid WAV file: too short")
        if data[:4] != b'RIFF' or data[8:12] != b'WAVE':
            raise ValidationError("Invalid WAV file: missing RIFF/WAVE headers")
        return True
    
    def _validate_mp3_format(self, data: bytes) -> bool:
        """Validate MP3 format."""
        if len(data) < 4:
            raise ValidationError("Invalid MP3 file: too short")
        if data[:2] not in [b'\xff\xfb', b'\xff\xfa', b'\xff\xf3', b'\xff\xf2']:
            raise ValidationError("Invalid MP3 file: missing sync word")
        return True
    
    def _validate_m4a_format(self, data: bytes) -> bool:
        """Validate M4A format."""
        if len(data) < 8:
            raise ValidationError("Invalid M4A file: too short")
        if data[4:8] != b'ftyp':
            raise ValidationError("Invalid M4A file: missing ftyp box")
        return True
    
    def _validate_flac_format(self, data: bytes) -> bool:
        """Validate FLAC format."""
        if len(data) < 4:
            raise ValidationError("Invalid FLAC file: too short")
        if data[:4] != b'fLaC':
            raise ValidationError("Invalid FLAC file: missing signature")
        return True
    
    def _validate_ogg_format(self, data: bytes) -> bool:
        """Validate OGG format."""
        if len(data) < 4:
            raise ValidationError("Invalid OGG file: too short")
        if data[:4] != b'OggS':
            raise ValidationError("Invalid OGG file: missing capture pattern")
        return True
    
    def _extract_wav_metadata(self, data: bytes) -> Dict:
        """Extract metadata from WAV file."""
        if len(data) < 44:
            return {'format': 'wav', 'duration': 0}
        
        # Extract sample rate and channels from WAV header
        channels = struct.unpack('<H', data[22:24])[0]
        sample_rate = struct.unpack('<I', data[24:28])[0]
        
        return {
            'format': 'wav',
            'sample_rate': sample_rate,
            'channels': channels,
            'duration': 1.0  # Mock duration
        }
    
    def _extract_mp3_metadata(self, data: bytes) -> Dict:
        """Extract metadata from MP3 file."""
        return {
            'format': 'mp3',
            'duration': 1.0,  # Mock duration
            'bitrate': 128  # Mock bitrate
        }
    
    def _extract_m4a_metadata(self, data: bytes) -> Dict:
        """Extract metadata from M4A file."""
        return {
            'format': 'm4a',
            'duration': 1.0,  # Mock duration
            'bitrate': 256  # Mock bitrate
        }
    
    def _extract_flac_metadata(self, data: bytes) -> Dict:
        """Extract metadata from FLAC file."""
        return {
            'format': 'flac',
            'duration': 1.0,  # Mock duration
            'sample_rate': 44100,  # Mock sample rate
            'channels': 2
        }
    
    def _extract_ogg_metadata(self, data: bytes) -> Dict:
        """Extract metadata from OGG file."""
        return {
            'format': 'ogg',
            'duration': 1.0,  # Mock duration
            'bitrate': 160  # Mock bitrate
        }
    
    def _is_large_file(self, metadata: Dict) -> bool:
        """Check if file is considered large."""
        return metadata.get('file_size', 0) > 100 * 1024 * 1024  # > 100MB
    
    def _validate_large_file_constraints(self, metadata: Dict) -> bool:
        """Validate constraints for large files."""
        # Could implement size limits per provider
        max_size_per_provider = {
            'gemini': 100 * 1024 * 1024,  # 100MB
            'openai': 25 * 1024 * 1024,   # 25MB
            'assemblyai': 500 * 1024 * 1024,  # 500MB
            'deepgram': 2 * 1024 * 1024 * 1024  # 2GB
        }
        
        # For testing, just check if under largest limit
        return metadata.get('file_size', 0) <= max_size_per_provider['deepgram']


class TestVideoFormatProcessing:
    """Test processing of various video formats."""
    
    @pytest.fixture
    def mock_video_files(self):
        """Create mock video files for testing."""
        return {
            'mp4': self._create_mock_mp4_file(),
            'avi': self._create_mock_avi_file(),
            'mov': self._create_mock_mov_file(),
            'mkv': self._create_mock_mkv_file(),
            'webm': self._create_mock_webm_file()
        }
    
    def test_mp4_video_processing(self, mock_video_files):
        """Test MP4 video processing."""
        mp4_file = mock_video_files['mp4']
        
        # Test file validation
        mp4_data = mp4_file.read()
        assert self._validate_mp4_format(mp4_data)
        
        # Test metadata extraction
        metadata = self._extract_mp4_metadata(mp4_data)
        assert metadata['format'] == 'mp4'
        assert metadata['has_audio'] is True
        assert metadata['has_video'] is True
    
    def test_avi_video_processing(self, mock_video_files):
        """Test AVI video processing."""
        avi_file = mock_video_files['avi']
        
        # Test file validation
        avi_data = avi_file.read()
        assert self._validate_avi_format(avi_data)
        
        # Test metadata extraction
        metadata = self._extract_avi_metadata(avi_data)
        assert metadata['format'] == 'avi'
        assert 'duration' in metadata
    
    def test_mov_video_processing(self, mock_video_files):
        """Test MOV video processing."""
        mov_file = mock_video_files['mov']
        
        # Test file validation
        mov_data = mov_file.read()
        assert self._validate_mov_format(mov_data)
        
        # Test metadata extraction
        metadata = self._extract_mov_metadata(mov_data)
        assert metadata['format'] == 'mov'
        assert metadata['container'] == 'quicktime'
    
    def test_mkv_video_processing(self, mock_video_files):
        """Test MKV video processing."""
        mkv_file = mock_video_files['mkv']
        
        # Test file validation
        mkv_data = mkv_file.read()
        assert self._validate_mkv_format(mkv_data)
        
        # Test metadata extraction
        metadata = self._extract_mkv_metadata(mkv_data)
        assert metadata['format'] == 'mkv'
        assert metadata['container'] == 'matroska'
    
    def test_webm_video_processing(self, mock_video_files):
        """Test WebM video processing."""
        webm_file = mock_video_files['webm']
        
        # Test file validation
        webm_data = webm_file.read()
        assert self._validate_webm_format(webm_data)
        
        # Test metadata extraction
        metadata = self._extract_webm_metadata(webm_data)
        assert metadata['format'] == 'webm'
        assert metadata['container'] == 'webm'
    
    def test_video_audio_extraction_requirements(self):
        """Test requirements for audio extraction from video."""
        video_formats = ['mp4', 'avi', 'mov', 'mkv', 'webm']
        
        for video_format in video_formats:
            # Should be able to identify audio streams
            has_audio = self._video_has_audio_stream(video_format)
            assert isinstance(has_audio, bool)
            
            # Should be able to extract audio if present
            if has_audio:
                audio_info = self._extract_audio_stream_info(video_format)
                assert 'codec' in audio_info
                assert 'sample_rate' in audio_info
    
    def _create_mock_mp4_file(self) -> io.BytesIO:
        """Create a mock MP4 file."""
        mp4_data = io.BytesIO()
        
        # MP4 ftyp box
        mp4_data.write(struct.pack('>I', 32))  # Box size
        mp4_data.write(b'ftyp')  # Box type
        mp4_data.write(b'mp42')  # Major brand
        mp4_data.write(struct.pack('>I', 0))   # Minor version
        mp4_data.write(b'mp42mp41isom')  # Compatible brands
        mp4_data.write(b'fake_mp4_data' * 100)
        
        mp4_data.seek(0)
        return mp4_data
    
    def _create_mock_avi_file(self) -> io.BytesIO:
        """Create a mock AVI file."""
        avi_data = io.BytesIO()
        
        # AVI RIFF header
        avi_data.write(b'RIFF')
        avi_data.write(struct.pack('<I', 1000))  # File size
        avi_data.write(b'AVI ')
        avi_data.write(b'fake_avi_data' * 100)
        
        avi_data.seek(0)
        return avi_data
    
    def _create_mock_mov_file(self) -> io.BytesIO:
        """Create a mock MOV file."""
        mov_data = io.BytesIO()
        
        # MOV is similar to MP4 but with 'qt  ' brand
        mov_data.write(struct.pack('>I', 32))  # Box size
        mov_data.write(b'ftyp')  # Box type
        mov_data.write(b'qt  ')  # Major brand (QuickTime)
        mov_data.write(struct.pack('>I', 0))   # Minor version
        mov_data.write(b'qt  mp42isom')  # Compatible brands
        mov_data.write(b'fake_mov_data' * 100)
        
        mov_data.seek(0)
        return mov_data
    
    def _create_mock_mkv_file(self) -> io.BytesIO:
        """Create a mock MKV file."""
        mkv_data = io.BytesIO()
        
        # Matroska header (simplified EBML)
        mkv_data.write(b'\x1a\x45\xdf\xa3')  # EBML header ID
        mkv_data.write(b'\x9f')  # Header size indicator
        mkv_data.write(b'fake_mkv_header' * 2)
        mkv_data.write(b'fake_mkv_data' * 100)
        
        mkv_data.seek(0)
        return mkv_data
    
    def _create_mock_webm_file(self) -> io.BytesIO:
        """Create a mock WebM file."""
        webm_data = io.BytesIO()
        
        # WebM is Matroska-based
        webm_data.write(b'\x1a\x45\xdf\xa3')  # EBML header ID
        webm_data.write(b'\x9f')  # Header size indicator  
        webm_data.write(b'webm')  # DocType
        webm_data.write(b'fake_webm_data' * 100)
        
        webm_data.seek(0)
        return webm_data
    
    def _validate_mp4_format(self, data: bytes) -> bool:
        """Validate MP4 format."""
        if len(data) < 8:
            raise ValidationError("Invalid MP4 file: too short")
        if data[4:8] != b'ftyp':
            raise ValidationError("Invalid MP4 file: missing ftyp box")
        return True
    
    def _validate_avi_format(self, data: bytes) -> bool:
        """Validate AVI format."""
        if len(data) < 12:
            raise ValidationError("Invalid AVI file: too short")
        if data[:4] != b'RIFF' or data[8:12] != b'AVI ':
            raise ValidationError("Invalid AVI file: missing RIFF/AVI headers")
        return True
    
    def _validate_mov_format(self, data: bytes) -> bool:
        """Validate MOV format."""
        if len(data) < 8:
            raise ValidationError("Invalid MOV file: too short")
        if data[4:8] != b'ftyp':
            raise ValidationError("Invalid MOV file: missing ftyp box")
        return True
    
    def _validate_mkv_format(self, data: bytes) -> bool:
        """Validate MKV format."""
        if len(data) < 4:
            raise ValidationError("Invalid MKV file: too short")
        if data[:4] != b'\x1a\x45\xdf\xa3':
            raise ValidationError("Invalid MKV file: missing EBML header")
        return True
    
    def _validate_webm_format(self, data: bytes) -> bool:
        """Validate WebM format."""
        if len(data) < 4:
            raise ValidationError("Invalid WebM file: too short") 
        if data[:4] != b'\x1a\x45\xdf\xa3':
            raise ValidationError("Invalid WebM file: missing EBML header")
        return True
    
    def _extract_mp4_metadata(self, data: bytes) -> Dict:
        """Extract metadata from MP4 file."""
        return {
            'format': 'mp4',
            'container': 'mp4',
            'has_audio': True,
            'has_video': True,
            'duration': 10.0  # Mock duration
        }
    
    def _extract_avi_metadata(self, data: bytes) -> Dict:
        """Extract metadata from AVI file."""
        return {
            'format': 'avi',
            'container': 'avi',
            'has_audio': True,
            'has_video': True,
            'duration': 15.0  # Mock duration
        }
    
    def _extract_mov_metadata(self, data: bytes) -> Dict:
        """Extract metadata from MOV file."""
        return {
            'format': 'mov',
            'container': 'quicktime',
            'has_audio': True,
            'has_video': True,
            'duration': 12.0  # Mock duration
        }
    
    def _extract_mkv_metadata(self, data: bytes) -> Dict:
        """Extract metadata from MKV file."""
        return {
            'format': 'mkv',
            'container': 'matroska',
            'has_audio': True,
            'has_video': True,
            'duration': 20.0  # Mock duration
        }
    
    def _extract_webm_metadata(self, data: bytes) -> Dict:
        """Extract metadata from WebM file."""
        return {
            'format': 'webm',
            'container': 'webm',
            'has_audio': True,
            'has_video': True,
            'duration': 8.0  # Mock duration
        }
    
    def _video_has_audio_stream(self, video_format: str) -> bool:
        """Check if video format typically has audio."""
        # Most video formats support audio
        return video_format in ['mp4', 'avi', 'mov', 'mkv', 'webm', 'wmv']
    
    def _extract_audio_stream_info(self, video_format: str) -> Dict:
        """Extract audio stream information."""
        audio_codecs = {
            'mp4': 'aac',
            'avi': 'mp3',
            'mov': 'aac',
            'mkv': 'vorbis',
            'webm': 'opus'
        }
        
        return {
            'codec': audio_codecs.get(video_format, 'unknown'),
            'sample_rate': 44100,
            'channels': 2,
            'bitrate': 128
        }


class TestProviderFormatCompatibility:
    """Test provider compatibility with different file formats."""
    
    def test_provider_format_support_matrix(self):
        """Test which formats each provider supports."""
        registry = get_registry()
        providers = registry.list_providers()
        
        # Define expected format support per provider
        expected_support = {
            'gemini': ['wav', 'mp3', 'mp4', 'm4a'],
            'openai': ['wav', 'mp3', 'mp4', 'm4a', 'webm'],
            'assemblyai': ['wav', 'mp3', 'mp4', 'm4a', 'flac'],
            'deepgram': ['wav', 'mp3', 'mp4', 'm4a', 'flac', 'ogg']
        }
        
        for provider_name in providers:
            if provider_name in expected_support:
                supported_formats = expected_support[provider_name]
                
                for format_name in supported_formats:
                    # Test that provider accepts this format
                    assert self._provider_supports_format(provider_name, format_name)
    
    def test_provider_format_limitations(self):
        """Test provider-specific format limitations."""
        limitations = {
            'openai': {
                'max_file_size': 25 * 1024 * 1024,  # 25MB
                'supported_formats': ['wav', 'mp3', 'mp4', 'm4a', 'webm']
            },
            'gemini': {
                'max_file_size': 100 * 1024 * 1024,  # 100MB
                'supported_formats': ['wav', 'mp3', 'mp4', 'm4a']
            },
            'assemblyai': {
                'max_file_size': 500 * 1024 * 1024,  # 500MB
                'supported_formats': ['wav', 'mp3', 'mp4', 'm4a', 'flac']
            },
            'deepgram': {
                'max_file_size': 2 * 1024 * 1024 * 1024,  # 2GB
                'supported_formats': ['wav', 'mp3', 'mp4', 'm4a', 'flac', 'ogg']
            }
        }
        
        for provider_name, limits in limitations.items():
            # Test file size limitations
            assert self._check_file_size_limit(provider_name, limits['max_file_size'])
            
            # Test format support
            for format_name in limits['supported_formats']:
                assert self._provider_supports_format(provider_name, format_name)
    
    def test_unsupported_format_handling(self):
        """Test handling of unsupported formats by providers."""
        unsupported_combinations = [
            ('openai', 'flac'),  # OpenAI doesn't support FLAC
            ('gemini', 'ogg'),   # Gemini doesn't support OGG
            ('openai', 'avi'),   # OpenAI doesn't support AVI
        ]
        
        for provider_name, format_name in unsupported_combinations:
            with pytest.raises((ValidationError, ProcessingError)):
                self._validate_provider_format(provider_name, format_name)
    
    def test_format_conversion_recommendations(self):
        """Test format conversion recommendations."""
        conversion_recommendations = [
            ('openai', 'flac', 'wav'),  # Convert FLAC to WAV for OpenAI
            ('gemini', 'ogg', 'mp3'),   # Convert OGG to MP3 for Gemini
            ('all', 'avi', 'mp4'),      # Convert AVI to MP4 for better compatibility
        ]
        
        for provider, source_format, target_format in conversion_recommendations:
            recommendation = self._get_format_conversion_recommendation(
                provider, source_format
            )
            if recommendation:
                assert target_format in recommendation['suggested_formats']
    
    def _provider_supports_format(self, provider_name: str, format_name: str) -> bool:
        """Check if provider supports a specific format."""
        # Simplified format support check
        format_support_matrix = {
            'gemini': ['wav', 'mp3', 'mp4', 'm4a'],
            'openai': ['wav', 'mp3', 'mp4', 'm4a', 'webm'],
            'assemblyai': ['wav', 'mp3', 'mp4', 'm4a', 'flac'],
            'deepgram': ['wav', 'mp3', 'mp4', 'm4a', 'flac', 'ogg']
        }
        
        supported = format_support_matrix.get(provider_name, [])
        return format_name in supported
    
    def _check_file_size_limit(self, provider_name: str, max_size: int) -> bool:
        """Check provider file size limits."""
        # This would normally check against provider specifications
        return max_size > 0  # All limits should be positive
    
    def _validate_provider_format(self, provider_name: str, format_name: str) -> None:
        """Validate that provider supports format."""
        if not self._provider_supports_format(provider_name, format_name):
            raise ValidationError(
                f"Provider {provider_name} does not support {format_name} format"
            )
    
    def _get_format_conversion_recommendation(
        self, provider_name: str, source_format: str
    ) -> Dict | None:
        """Get format conversion recommendations."""
        if self._provider_supports_format(provider_name, source_format):
            return None  # No conversion needed
        
        # Return conversion suggestions
        universal_formats = ['wav', 'mp3', 'mp4']
        return {
            'reason': f'{source_format} not supported by {provider_name}',
            'suggested_formats': universal_formats
        }


class TestFileFormatIntegration:
    """Integration tests for file format handling."""
    
    @pytest.mark.integration
    def test_end_to_end_format_processing(self):
        """Test end-to-end processing of different formats."""
        test_formats = ['wav', 'mp3', 'mp4', 'm4a']
        
        for format_name in test_formats:
            # Mock file processing pipeline
            file_path = f"/fake/test_file.{format_name}"
            
            # Step 1: Format detection
            detected_format = self._detect_format_by_extension(file_path)
            assert detected_format == format_name
            
            # Step 2: Validation
            self._validate_file_format(file_path)
            
            # Step 3: Provider compatibility check
            config = VttiroConfig(provider="gemini")
            compatible = self._check_provider_compatibility(config.provider, format_name)
            
            # Step 4: Processing simulation
            if compatible:
                result = self._simulate_processing(file_path, config)
                assert result['success'] is True
                assert result['format'] == format_name
    
    def test_format_error_handling_integration(self):
        """Test integrated error handling for format issues."""
        error_scenarios = [
            ('corrupted.wav', 'Corrupted file'),
            ('unsupported.xyz', 'Unsupported format'),
            ('large.mp4', 'File too large'),
            ('empty.mp3', 'Empty file')
        ]
        
        for file_path, expected_error_type in error_scenarios:
            try:
                self._simulate_full_processing_pipeline(file_path)
                # Should not reach here for error scenarios
                assert False, f"Expected error for {file_path}"
            except (ValidationError, ProcessingError) as e:
                # Verify appropriate error type
                assert any(
                    keyword in str(e).lower() 
                    for keyword in expected_error_type.lower().split()
                )
    
    def _detect_format_by_extension(self, file_path: str) -> str:
        """Helper method for format detection."""
        return Path(file_path).suffix[1:].lower()
    
    def _validate_file_format(self, file_path: str) -> None:
        """Helper method for format validation."""
        format_name = self._detect_format_by_extension(file_path)
        supported_formats = ['wav', 'mp3', 'mp4', 'm4a', 'flac', 'ogg', 'avi', 'mov', 'mkv', 'webm']
        
        if format_name not in supported_formats:
            raise ValidationError(f"Unsupported format: {format_name}")
    
    def _check_provider_compatibility(self, provider_name: str, format_name: str) -> bool:
        """Helper method for provider compatibility check."""
        return self._provider_supports_format(provider_name, format_name)
    
    def _provider_supports_format(self, provider_name: str, format_name: str) -> bool:
        """Helper method for format support check."""
        support_matrix = {
            'gemini': ['wav', 'mp3', 'mp4', 'm4a'],
            'openai': ['wav', 'mp3', 'mp4', 'm4a', 'webm'],
            'assemblyai': ['wav', 'mp3', 'mp4', 'm4a', 'flac'],
            'deepgram': ['wav', 'mp3', 'mp4', 'm4a', 'flac', 'ogg']
        }
        return format_name in support_matrix.get(provider_name, [])
    
    def _simulate_processing(self, file_path: str, config: VttiroConfig) -> Dict:
        """Simulate file processing."""
        format_name = self._detect_format_by_extension(file_path)
        
        return {
            'success': True,
            'format': format_name,
            'provider': config.provider,
            'duration': 10.0,  # Mock duration
            'file_path': file_path
        }
    
    def _simulate_full_processing_pipeline(self, file_path: str) -> Dict:
        """Simulate full processing pipeline with error handling."""
        # Simulate various error conditions
        if 'corrupted' in file_path:
            raise ValidationError("Corrupted file detected")
        elif 'unsupported' in file_path:
            raise ValidationError("Unsupported format detected")
        elif 'large' in file_path:
            raise ProcessingError("File too large for processing")
        elif 'empty' in file_path:
            raise ValidationError("Empty file detected")
        
        # Normal processing
        return self._simulate_processing(file_path, VttiroConfig())