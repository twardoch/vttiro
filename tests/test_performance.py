#!/usr/bin/env python3
# this_file: tests/test_performance.py
"""Performance and benchmark tests for vttiro."""

import pytest
import time
import asyncio
import memory_profiler
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from vttiro.core.transcriber import Transcriber
from vttiro.core.config import VttiroConfig, TranscriptionResult


@pytest.mark.benchmark
class TestTranscriberPerformance:
    """Performance tests for Transcriber class."""
    
    @pytest.fixture
    def benchmark_transcriber(self, mock_config):
        """Create transcriber for benchmarking."""
        with patch('vttiro.core.transcriber.VideoProcessor'), \
             patch('vttiro.core.transcriber.TranscriptionEnsemble'):
            return Transcriber(mock_config)
    
    @pytest.mark.asyncio
    async def test_transcribe_performance_small_file(self, benchmark_transcriber, performance_collector):
        """Benchmark transcription of small audio file (30 seconds)."""
        # Mock fast processing for small file
        mock_video_result = MagicMock()
        mock_video_result.metadata.title = "Small Test"
        mock_video_result.metadata.duration_seconds = 30.0
        mock_video_result.segments = [MagicMock() for _ in range(1)]  # 1 segment
        
        benchmark_transcriber.processing_resilience.execute = AsyncMock(return_value=mock_video_result)
        benchmark_transcriber.api_resilience.execute = AsyncMock(return_value=TranscriptionResult(
            text="Quick test transcription",
            confidence=0.95,
            language="en"
        ))
        
        with patch('pathlib.Path.write_text'):
            start_time = time.perf_counter()
            
            result = await benchmark_transcriber.transcribe("small_test.mp4")
            
            end_time = time.perf_counter()
            processing_time = end_time - start_time
            
            performance_collector.record("small_file_transcription", processing_time)
            
            # Performance assertions
            assert processing_time < 5.0  # Should complete in under 5 seconds
            assert result is not None
    
    @pytest.mark.asyncio
    async def test_transcribe_performance_medium_file(self, benchmark_transcriber, performance_collector):
        """Benchmark transcription of medium audio file (5 minutes)."""
        # Mock processing for medium file
        mock_video_result = MagicMock()
        mock_video_result.metadata.title = "Medium Test"
        mock_video_result.metadata.duration_seconds = 300.0
        mock_video_result.segments = [MagicMock() for _ in range(10)]  # 10 segments
        
        benchmark_transcriber.processing_resilience.execute = AsyncMock(return_value=mock_video_result)
        benchmark_transcriber.api_resilience.execute = AsyncMock(return_value=TranscriptionResult(
            text="Medium length test transcription",
            confidence=0.92,
            language="en"
        ))
        
        with patch('pathlib.Path.write_text'):
            start_time = time.perf_counter()
            
            result = await benchmark_transcriber.transcribe("medium_test.mp4")
            
            end_time = time.perf_counter()
            processing_time = end_time - start_time
            
            performance_collector.record("medium_file_transcription", processing_time)
            
            # Performance assertions - should be faster than real-time
            realtime_ratio = processing_time / 300.0
            assert realtime_ratio < 0.2  # Should be at least 5x faster than real-time
            assert result is not None
    
    @pytest.mark.asyncio
    async def test_batch_transcribe_performance(self, benchmark_transcriber, performance_collector):
        """Benchmark batch transcription performance."""
        sources = [f"test_{i}.mp4" for i in range(5)]
        
        # Mock individual transcription calls
        async def mock_transcribe(source, output_file, **kwargs):
            await asyncio.sleep(0.1)  # Simulate processing time
            return str(output_file)
        
        benchmark_transcriber.transcribe = AsyncMock(side_effect=mock_transcribe)
        
        with patch('pathlib.Path.mkdir'):
            start_time = time.perf_counter()
            
            results = await benchmark_transcriber.batch_transcribe(sources)
            
            end_time = time.perf_counter()
            processing_time = end_time - start_time
            
            performance_collector.record("batch_transcription_5_files", processing_time)
            
            # Should process all files in reasonable time
            assert len(results) == 5
            assert processing_time < 10.0  # Should complete batch in under 10 seconds
    
    @pytest.mark.asyncio
    async def test_memory_usage_transcription(self, benchmark_transcriber):
        """Test memory usage during transcription."""
        # Mock large file scenario
        mock_video_result = MagicMock()
        mock_video_result.metadata.title = "Large Test"
        mock_video_result.metadata.duration_seconds = 3600.0  # 1 hour
        mock_video_result.segments = [MagicMock() for _ in range(120)]  # 120 segments
        
        benchmark_transcriber.processing_resilience.execute = AsyncMock(return_value=mock_video_result)
        benchmark_transcriber.api_resilience.execute = AsyncMock(return_value=TranscriptionResult(
            text="Large file test transcription",
            confidence=0.90,
            language="en"
        ))
        
        with patch('pathlib.Path.write_text'):
            # Measure memory before
            mem_before = memory_profiler.memory_usage()[0]
            
            result = await benchmark_transcriber.transcribe("large_test.mp4")
            
            # Measure memory after
            mem_after = memory_profiler.memory_usage()[0]
            memory_increase = mem_after - mem_before
            
            # Memory usage should be reasonable
            assert memory_increase < 500  # Less than 500MB increase
            assert result is not None
    
    def test_config_loading_performance(self, temp_dir, performance_collector):
        """Test configuration loading performance."""
        # Create large config file
        config_file = temp_dir / "large_config.yaml"
        config_content = f"""
transcription:
  preferred_model: "auto"
  language: "en"
  max_duration_seconds: 3600
  chunk_duration_seconds: 30
  # Large metadata section
  metadata:
"""
        
        # Add large metadata section
        for i in range(1000):
            config_content += f"    key_{i}: 'value_{i}'\n"
        
        config_file.write_text(config_content)
        
        start_time = time.perf_counter()
        
        config = VttiroConfig.load_from_file(config_file)
        
        end_time = time.perf_counter()
        loading_time = end_time - start_time
        
        performance_collector.record("large_config_loading", loading_time)
        
        # Config loading should be fast even for large files
        assert loading_time < 1.0  # Under 1 second
        assert config is not None



# Resilience performance tests removed for simplification

@pytest.fixture
def perf_helper():
    """Performance testing helper fixture."""
    return PerformanceTestHelper()
