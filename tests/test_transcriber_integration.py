#!/usr/bin/env python3
# this_file: tests/test_transcriber_integration.py
"""Integration tests for enhanced Transcriber class with error handling."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from vttiro.core.transcriber import Transcriber
from vttiro.core.config import VttiroConfig
from vttiro.utils.exceptions import (
    ValidationError,
    ProcessingError,
    TranscriptionError,
    OutputGenerationError,
)


class TestTranscriberIntegration:
    """Test Transcriber class integration with error handling framework."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for testing."""
        config = MagicMock(spec=VttiroConfig)
        config.transcription = MagicMock()
        config.transcription.preferred_model = "auto"
        config.transcription.gemini_api_key = None
        config.transcription.assemblyai_api_key = None
        config.transcription.deepgram_api_key = None
        return config
    
    @pytest.fixture
    def transcriber(self, mock_config):
        """Create Transcriber instance with mocked dependencies."""
        with patch('vttiro.core.transcriber.VideoProcessor'), \
             patch('vttiro.core.transcriber.TranscriptionEnsemble'):
            transcriber = Transcriber(mock_config)
            return transcriber
    
    def test_transcriber_initialization_with_error_handling(self, mock_config):
        """Test transcriber initialization includes error handling components."""
        with patch('vttiro.core.transcriber.VideoProcessor'), \
             patch('vttiro.core.transcriber.TranscriptionEnsemble'):
            transcriber = Transcriber(mock_config)
            
            # Verify resilience managers are initialized
            assert hasattr(transcriber, 'api_resilience')
            assert hasattr(transcriber, 'processing_resilience')
            assert transcriber.api_resilience is not None
            assert transcriber.processing_resilience is not None
    
    def test_transcriber_initialization_failure(self):
        """Test transcriber initialization failure handling."""
        with patch('vttiro.core.config.VttiroConfig.load_default', 
                   side_effect=Exception("Config load failed")):
            with pytest.raises(ConfigurationError, match="Failed to load or validate configuration"):
                Transcriber()
    
    def test_engine_initialization_with_all_failures(self, mock_config):
        """Test engine initialization when all AI engines fail."""
        with patch('vttiro.core.transcriber.VideoProcessor'), \
             patch('vttiro.core.transcriber.TranscriptionEnsemble'), \
             patch('vttiro.core.transcriber.GeminiTranscriber', None), \
             patch('vttiro.core.transcriber.AssemblyAITranscriber', None), \
             patch('vttiro.core.transcriber.DeepgramTranscriber', None), \
             patch('vttiro.core.transcriber.MockTranscriptionEngine') as mock_engine:
            
            mock_engine.return_value = MagicMock()
            transcriber = Transcriber(mock_config)
            
            # Should fall back to mock engine
            assert len(transcriber.engines) == 1
    
    @pytest.mark.asyncio
    async def test_transcribe_with_correlation_id(self, transcriber):
        """Test transcribe method generates correlation IDs."""
        # Mock dependencies
        mock_video_result = MagicMock()
        mock_video_result.metadata = MagicMock()
        mock_video_result.metadata.title = "Test Video"
        mock_video_result.metadata.duration_seconds = 60.0
        mock_video_result.segments = []
        
        transcriber.processing_resilience.execute = AsyncMock(return_value=mock_video_result)
        
        with patch('pathlib.Path.write_text'), \
             patch('uuid.uuid4', return_value=MagicMock()):
            
            result = await transcriber.transcribe("test_source.mp4")
            
            # Verify correlation ID was used
            assert transcriber.processing_resilience.execute.called
            call_args = transcriber.processing_resilience.execute.call_args
            assert 'correlation_id' in call_args.kwargs
    
    @pytest.mark.asyncio
    async def test_transcribe_processing_error(self, transcriber):
        """Test transcribe method handles processing errors."""
        transcriber.processing_resilience.execute = AsyncMock(
            side_effect=Exception("Video processing failed")
        )
        
        with pytest.raises(ProcessingError, match="Failed to process video source"):
            await transcriber.transcribe("test_source.mp4")
    
    @pytest.mark.asyncio
    async def test_transcribe_output_generation_error(self, transcriber):
        """Test transcribe method handles output generation errors."""
        # Mock successful video processing
        mock_video_result = MagicMock()
        mock_video_result.metadata = MagicMock()
        mock_video_result.metadata.title = "Test Video"
        mock_video_result.metadata.duration_seconds = 60.0
        mock_video_result.segments = []
        
        transcriber.processing_resilience.execute = AsyncMock(return_value=mock_video_result)
        
        # Mock file write failure
        with patch('pathlib.Path.write_text', side_effect=PermissionError("Write failed")):
            with pytest.raises(OutputGenerationError, match="Failed to write WebVTT file"):
                await transcriber.transcribe("test_source.mp4")
    
    @pytest.mark.asyncio
    async def test_batch_transcribe_validation_error(self, transcriber):
        """Test batch transcribe validates input."""
        with pytest.raises(ValidationError, match="Sources list cannot be empty"):
            await transcriber.batch_transcribe([])
    
    @pytest.mark.asyncio
    async def test_batch_transcribe_partial_failures(self, transcriber):
        """Test batch transcribe handles partial failures gracefully."""
        sources = ["source1.mp4", "source2.mp4", "source3.mp4"]
        
        # Mock transcribe method to fail on second source
        async def mock_transcribe(source, output_file, **kwargs):
            if "source2" in str(source):
                raise TranscriptionError("Transcription failed")
            return str(output_file)
        
        transcriber.transcribe = AsyncMock(side_effect=mock_transcribe)
        
        with patch('pathlib.Path.mkdir'):
            results = await transcriber.batch_transcribe(sources, output_dir="/tmp/test")
            
            # Should succeed for 2 out of 3 sources
            assert len(results) == 2
            assert transcriber.transcribe.call_count == 3
    
    def test_correlation_id_propagation_in_batch(self, transcriber):
        """Test correlation IDs are properly propagated in batch operations."""
        # This test would verify that correlation IDs flow through the system
        # In a real implementation, we'd check logs or mock the correlation tracking
        pass
    
    def test_circuit_breaker_integration(self, transcriber):
        """Test circuit breaker integration with transcriber."""
        # Verify circuit breaker is configured for API calls
        assert transcriber.api_resilience is not None
        
        # Test circuit breaker state reporting
        state = transcriber.api_resilience.get_state()
        assert "circuit_breaker" in state
        assert state["circuit_breaker"]["state"] == "closed"
    
    def test_retry_manager_integration(self, transcriber):
        """Test retry manager integration with transcriber."""
        # Verify retry manager is configured
        assert transcriber.api_resilience is not None
        
        # Test retry configuration
        state = transcriber.api_resilience.get_state()
        assert "retry_config" in state
        assert state["retry_config"]["max_attempts"] == 3


class TestTranscriberErrorScenarios:
    """Test specific error scenarios and recovery patterns."""
    
    @pytest.mark.asyncio
    async def test_segment_transcription_failure_graceful_degradation(self):
        """Test graceful degradation when segment transcription fails."""
        # This would test that failed segments get placeholder text
        # and the transcription continues with other segments
        pass
    
    @pytest.mark.asyncio
    async def test_api_rate_limit_handling(self):
        """Test API rate limit error handling and retry logic."""
        # This would test that rate limit errors trigger appropriate delays
        # and retry behavior as configured in the resilience framework
        pass
    
    @pytest.mark.asyncio
    async def test_network_timeout_recovery(self):
        """Test network timeout recovery with circuit breaker."""
        # This would test that network timeouts are handled by the circuit breaker
        # and the system gracefully degrades or recovers
        pass