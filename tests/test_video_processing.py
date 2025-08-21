#!/usr/bin/env python3
# this_file: tests/test_video_processing.py
"""Unit tests for video processing functionality."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
from pathlib import Path

from vttiro.processing.video import VideoProcessor, AudioChunk
from vttiro.core.config import VttiroConfig
from vttiro.utils.exceptions import ProcessingError, ValidationError


class TestVideoProcessor:
    """Test VideoProcessor class functionality."""
    
    @pytest.fixture
    def video_processor(self, mock_config):
        """Create VideoProcessor instance for testing."""
        return VideoProcessor(mock_config)
    
    @pytest.fixture
    def mock_youtube_dl(self):
        """Mock youtube-dl/yt-dlp functionality."""
        mock_dl = MagicMock()
        mock_dl.extract_info.return_value = {
            'title': 'Test Video',
            'description': 'A test video for unit testing',
            'uploader': 'Test User',
            'duration': 120.0,
            'url': 'https://example.com/test-video',
            'thumbnail': 'https://example.com/thumbnail.jpg',
            'upload_date': '20240101',
            'view_count': 1000,
            'like_count': 50,
            'tags': ['test', 'video']
        }
        return mock_dl
    
    @pytest.fixture
    def mock_ffmpeg(self):
        """Mock FFmpeg functionality."""
        mock_ffmpeg = MagicMock()
        mock_ffmpeg.input.return_value = mock_ffmpeg
        mock_ffmpeg.output.return_value = mock_ffmpeg
        mock_ffmpeg.run.return_value = None
        mock_ffmpeg.probe.return_value = {
            'streams': [{
                'codec_type': 'audio',
                'duration': '120.0',
                'sample_rate': '16000',
                'channels': 1
            }]
        }
        return mock_ffmpeg
    
    def test_video_processor_initialization(self, video_processor, mock_config):
        """Test VideoProcessor initialization."""
        assert video_processor.config == mock_config
        assert hasattr(video_processor, 'temp_dir')
        assert hasattr(video_processor, 'max_workers')
    
    @pytest.mark.asyncio
    async def test_process_youtube_url(self, video_processor, mock_youtube_dl, mock_ffmpeg):
        """Test processing YouTube URL."""
        url = "https://www.youtube.com/watch?v=test123"
        
        with patch('yt_dlp.YoutubeDL', return_value=mock_youtube_dl), \
             patch('ffmpeg.input', return_value=mock_ffmpeg), \
             patch('ffmpeg.probe', return_value=mock_ffmpeg.probe.return_value), \
             patch.object(video_processor, '_extract_audio_segments') as mock_extract:
            
            mock_extract.return_value = [
                AudioChunk(
                    audio_file=Path("/tmp/chunk_0.wav"),
                    start_time=0.0,
                    end_time=30.0,
                    duration=30.0,
                    sample_rate=16000
                ),
                AudioChunk(
                    audio_file=Path("/tmp/chunk_1.wav"),
                    start_time=30.0,
                    end_time=60.0,
                    duration=30.0,
                    sample_rate=16000
                )
            ]
            
            result = await video_processor.process_source(
                url,
                extract_audio=True,
                segment_audio=True
            )
            
            # Verify metadata extraction
            assert result.metadata.title == "Test Video"
            assert result.metadata.duration_seconds == 120.0
            assert result.metadata.uploader == "Test User"
            
            # Verify audio segments
            assert len(result.segments) == 2
            assert result.segments[0].start_time == 0.0
            assert result.segments[1].start_time == 30.0
    
    @pytest.mark.asyncio
    async def test_process_local_file(self, video_processor, mock_ffmpeg, temp_dir):
        """Test processing local video file."""
        # Create mock video file
        video_file = temp_dir / "test_video.mp4"
        video_file.write_bytes(b"mock video data")
        
        with patch('ffmpeg.input', return_value=mock_ffmpeg), \
             patch('ffmpeg.probe', return_value=mock_ffmpeg.probe.return_value), \
             patch.object(video_processor, '_extract_audio_segments') as mock_extract:
            
            mock_extract.return_value = [
                AudioChunk(
                    audio_file=Path("/tmp/chunk_0.wav"),
                    start_time=0.0,
                    end_time=60.0,
                    duration=60.0,
                    sample_rate=16000
                )
            ]
            
            result = await video_processor.process_source(
                str(video_file),
                extract_audio=True,
                segment_audio=True
            )
            
            # Should have basic metadata for local file
            assert result.metadata.title == "test_video"
            assert result.metadata.duration_seconds == 120.0
            assert len(result.segments) == 1
    
    @pytest.mark.asyncio
    async def test_process_source_invalid_url(self, video_processor):
        """Test processing invalid URL."""
        with pytest.raises(ValidationError, match="Invalid source URL or file path"):
            await video_processor.process_source("invalid://url")
    
    @pytest.mark.asyncio
    async def test_process_source_missing_file(self, video_processor):
        """Test processing non-existent file."""
        with pytest.raises(ValidationError, match="File not found"):
            await video_processor.process_source("/path/to/nonexistent/file.mp4")
    
    @pytest.mark.asyncio
    async def test_audio_extraction_failure(self, video_processor, mock_youtube_dl):
        """Test handling of audio extraction failure."""
        url = "https://www.youtube.com/watch?v=test123"
        
        with patch('yt_dlp.YoutubeDL', return_value=mock_youtube_dl), \
             patch('ffmpeg.input', side_effect=Exception("FFmpeg failed")):
            
            with pytest.raises(ProcessingError, match="Failed to extract audio"):
                await video_processor.process_source(url, extract_audio=True)
    
    def test_audio_segmentation_energy_based(self, video_processor):
        """Test energy-based audio segmentation."""
        # Mock audio data with varying energy levels
        mock_audio_data = [0.1, 0.8, 0.9, 0.2, 0.1, 0.7, 0.8, 0.1]  # 8 samples
        sample_rate = 16000
        
        with patch.object(video_processor, '_load_audio_data', return_value=mock_audio_data):
            segments = video_processor._segment_by_energy(
                audio_file=Path("/tmp/test.wav"),
                target_duration=2.0,  # 2 seconds per segment
                sample_rate=sample_rate
            )
            
            # Should create segments based on energy patterns
            assert len(segments) > 0
            
            # Each segment should have valid properties
            for segment in segments:
                assert isinstance(segment, AudioChunk)
                assert segment.start_time >= 0.0
                assert segment.end_time > segment.start_time
                assert segment.duration > 0.0
                assert segment.sample_rate == sample_rate
    
    def test_audio_segmentation_fixed_duration(self, video_processor):
        """Test fixed-duration audio segmentation."""
        audio_file = Path("/tmp/test_120s.wav")
        total_duration = 120.0  # 2 minutes
        chunk_duration = 30.0   # 30 seconds per chunk
        
        with patch.object(video_processor, '_get_audio_duration', return_value=total_duration):
            segments = video_processor._segment_by_duration(
                audio_file=audio_file,
                chunk_duration=chunk_duration
            )
            
            # Should create 4 segments (120s / 30s = 4)
            assert len(segments) == 4
            
            # Verify segment timing
            for i, segment in enumerate(segments):
                expected_start = i * chunk_duration
                expected_end = min((i + 1) * chunk_duration, total_duration)
                
                assert segment.start_time == expected_start
                assert segment.end_time == expected_end
                assert segment.duration == expected_end - expected_start
    
    def test_audio_metadata_extraction(self, video_processor, mock_ffmpeg):
        """Test audio metadata extraction."""
        audio_file = Path("/tmp/test.wav")
        
        with patch('ffmpeg.probe', return_value=mock_ffmpeg.probe.return_value):
            metadata = video_processor._extract_audio_metadata(audio_file)
            
            assert metadata['duration'] == 120.0
            assert metadata['sample_rate'] == 16000
            assert metadata['channels'] == 1
    
    @pytest.mark.asyncio
    async def test_parallel_audio_processing(self, video_processor, mock_ffmpeg):
        """Test parallel processing of audio segments."""
        segments = [
            AudioChunk(
                audio_file=Path(f"/tmp/chunk_{i}.wav"),
                start_time=i * 30.0,
                end_time=(i + 1) * 30.0,
                duration=30.0,
                sample_rate=16000
            )
            for i in range(4)
        ]
        
        async def mock_process_segment(segment):
            """Mock segment processing."""
            await asyncio.sleep(0.1)  # Simulate processing time
            return segment
        
        with patch.object(video_processor, '_process_audio_segment', side_effect=mock_process_segment):
            processed_segments = await video_processor._process_segments_parallel(segments)
            
            assert len(processed_segments) == 4
            assert all(isinstance(seg, AudioChunk) for seg in processed_segments)
    
    def test_youtube_metadata_extraction(self, video_processor, mock_youtube_dl):
        """Test YouTube metadata extraction."""
        url = "https://www.youtube.com/watch?v=test123"
        
        with patch('yt_dlp.YoutubeDL', return_value=mock_youtube_dl):
            metadata = video_processor._extract_youtube_metadata(url)
            
            assert metadata.title == "Test Video"
            assert metadata.description == "A test video for unit testing"
            assert metadata.uploader == "Test User"
            assert metadata.duration_seconds == 120.0
            assert metadata.url == "https://example.com/test-video"
            assert metadata.view_count == 1000
    
    def test_local_file_metadata_extraction(self, video_processor, mock_ffmpeg, temp_dir):
        """Test local file metadata extraction."""
        video_file = temp_dir / "test_video.mp4"
        video_file.write_bytes(b"mock video data")
        
        with patch('ffmpeg.probe', return_value=mock_ffmpeg.probe.return_value):
            metadata = video_processor._extract_local_metadata(str(video_file))
            
            assert metadata.title == "test_video"
            assert metadata.duration_seconds == 120.0
            assert "mp4" in metadata.url.lower()
    
    @pytest.mark.asyncio
    async def test_cleanup_temp_files(self, video_processor, temp_dir):
        """Test cleanup of temporary files."""
        # Create mock temporary files
        temp_files = []
        for i in range(3):
            temp_file = temp_dir / f"temp_audio_{i}.wav"
            temp_file.write_bytes(b"mock audio data")
            temp_files.append(temp_file)
        
        # Register files for cleanup
        video_processor._temp_files.extend(temp_files)
        
        # Verify files exist
        for temp_file in temp_files:
            assert temp_file.exists()
        
        # Cleanup
        await video_processor._cleanup_temp_files()
        
        # Verify files are removed
        for temp_file in temp_files:
            assert not temp_file.exists()
    
    def test_error_handling_invalid_audio_format(self, video_processor):
        """Test error handling for invalid audio format."""
        audio_file = Path("/tmp/invalid.xyz")
        
        with patch('ffmpeg.probe', side_effect=Exception("Invalid format")):
            with pytest.raises(ProcessingError, match="Failed to analyze audio"):
                video_processor._extract_audio_metadata(audio_file)
    
    @pytest.mark.asyncio
    async def test_memory_efficient_processing(self, video_processor, mock_config):
        """Test memory-efficient processing for large files."""
        # Configure for memory efficiency
        mock_config.processing.memory_limit_mb = 1024
        mock_config.transcription.chunk_duration_seconds = 15  # Smaller chunks
        
        url = "https://www.youtube.com/watch?v=long_video"
        
        with patch.object(video_processor, '_extract_youtube_metadata') as mock_meta, \
             patch.object(video_processor, '_extract_audio_segments') as mock_extract:
            
            # Mock large video metadata
            mock_metadata = MagicMock()
            mock_metadata.duration_seconds = 3600.0  # 1 hour
            mock_meta.return_value = mock_metadata
            
            # Mock many small segments for memory efficiency
            mock_segments = [
                AudioChunk(
                    audio_file=Path(f"/tmp/chunk_{i}.wav"),
                    start_time=i * 15.0,
                    end_time=(i + 1) * 15.0,
                    duration=15.0,
                    sample_rate=16000
                )
                for i in range(240)  # 240 segments of 15s each = 1 hour
            ]
            mock_extract.return_value = mock_segments
            
            result = await video_processor.process_source(
                url,
                extract_audio=True,
                segment_audio=True
            )
            
            # Should handle large number of segments efficiently
            assert len(result.segments) == 240
            assert result.metadata.duration_seconds == 3600.0


class TestAudioChunk:
    """Test AudioChunk data class."""
    
    def test_audio_chunk_creation(self):
        """Test AudioChunk creation with valid parameters."""
        chunk = AudioChunk(
            audio_file=Path("/tmp/test.wav"),
            start_time=0.0,
            end_time=30.0,
            duration=30.0,
            sample_rate=16000
        )
        
        assert chunk.audio_file == Path("/tmp/test.wav")
        assert chunk.start_time == 0.0
        assert chunk.end_time == 30.0
        assert chunk.duration == 30.0
        assert chunk.sample_rate == 16000
    
    def test_audio_chunk_validation(self):
        """Test AudioChunk validation."""
        # Test invalid time range
        with pytest.raises(ValidationError, match="start_time cannot be greater than end_time"):
            AudioChunk(
                audio_file=Path("/tmp/test.wav"),
                start_time=30.0,
                end_time=10.0,  # Invalid
                duration=20.0,
                sample_rate=16000
            )
        
        # Test negative duration
        with pytest.raises(ValidationError, match="duration must be positive"):
            AudioChunk(
                audio_file=Path("/tmp/test.wav"),
                start_time=0.0,
                end_time=30.0,
                duration=-5.0,  # Invalid
                sample_rate=16000
            )
        
        # Test invalid sample rate
        with pytest.raises(ValidationError, match="sample_rate must be positive"):
            AudioChunk(
                audio_file=Path("/tmp/test.wav"),
                start_time=0.0,
                end_time=30.0,
                duration=30.0,
                sample_rate=0  # Invalid
            )
    
    def test_audio_chunk_properties(self):
        """Test AudioChunk computed properties."""
        chunk = AudioChunk(
            audio_file=Path("/tmp/test.wav"),
            start_time=5.0,
            end_time=35.0,
            duration=30.0,
            sample_rate=16000
        )
        
        # Test computed properties
        assert chunk.duration_ms == 30000  # 30 seconds in milliseconds
        assert chunk.sample_count == 480000  # 30s * 16000 samples/s
        assert chunk.size_bytes > 0  # Should estimate file size


class TestVideoProcessingIntegration:
    """Integration tests for video processing pipeline."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_video_processing_pipeline(self, mock_config, temp_dir):
        """Test complete video processing pipeline."""
        # This test would require actual video/audio files
        # For now, we'll mock the entire pipeline
        
        processor = VideoProcessor(mock_config)
        
        with patch.object(processor, '_extract_youtube_metadata') as mock_meta, \
             patch.object(processor, '_extract_audio_segments') as mock_segments:
            
            # Mock realistic processing result
            mock_metadata = MagicMock()
            mock_metadata.title = "Integration Test Video"
            mock_metadata.duration_seconds = 180.0
            mock_meta.return_value = mock_metadata
            
            mock_audio_segments = [
                AudioChunk(
                    audio_file=temp_dir / f"segment_{i}.wav",
                    start_time=i * 30.0,
                    end_time=(i + 1) * 30.0,
                    duration=30.0,
                    sample_rate=16000
                )
                for i in range(6)  # 6 segments of 30s each
            ]
            mock_segments.return_value = mock_audio_segments
            
            result = await processor.process_source(
                "https://example.com/test-video",
                extract_audio=True,
                segment_audio=True
            )
            
            # Verify complete processing
            assert result.metadata.title == "Integration Test Video"
            assert result.metadata.duration_seconds == 180.0
            assert len(result.segments) == 6
            
            # Verify all segments have correct timing
            for i, segment in enumerate(result.segments):
                assert segment.start_time == i * 30.0
                assert segment.end_time == (i + 1) * 30.0
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_large_file_processing_performance(self, mock_config):
        """Test processing performance with large files."""
        # This test measures performance characteristics
        # In practice, would use actual large files
        
        processor = VideoProcessor(mock_config)
        
        # Mock processing of 1-hour video
        large_duration = 3600.0  # 1 hour
        segment_count = int(large_duration / 30)  # 30s segments
        
        start_time = time.perf_counter()
        
        with patch.object(processor, '_extract_youtube_metadata') as mock_meta, \
             patch.object(processor, '_extract_audio_segments') as mock_segments:
            
            mock_metadata = MagicMock()
            mock_metadata.duration_seconds = large_duration
            mock_meta.return_value = mock_metadata
            
            mock_segments.return_value = [
                AudioChunk(
                    audio_file=Path(f"/tmp/segment_{i}.wav"),
                    start_time=i * 30.0,
                    end_time=(i + 1) * 30.0,
                    duration=30.0,
                    sample_rate=16000
                )
                for i in range(segment_count)
            ]
            
            result = await processor.process_source(
                "https://example.com/large-video",
                extract_audio=True,
                segment_audio=True
            )
            
            end_time = time.perf_counter()
            processing_time = end_time - start_time
            
            # Performance assertions
            assert len(result.segments) == segment_count
            assert processing_time < 30.0  # Should complete in under 30 seconds
            
            # Calculate processing efficiency
            efficiency_ratio = large_duration / processing_time
            assert efficiency_ratio > 100  # Should be >100x faster than real-time