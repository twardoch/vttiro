# this_file: src/vttiro/tests/test_openai_provider.py
"""Unit tests for OpenAI Whisper transcription provider."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from typing import Any

import pytest

from ..core.errors import AuthenticationError, ProcessingError, APIError
from ..core.types import TranscriptSegment

# Test if OpenAI is available
try:
    from ..providers.openai.transcriber import OpenAITranscriber, OPENAI_AVAILABLE
except ImportError:
    OPENAI_AVAILABLE = False


@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
class TestOpenAITranscriber:
    """Test OpenAITranscriber class."""
    
    def test_missing_api_key_raises_error(self):
        """Test that missing API key raises AuthenticationError."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(AuthenticationError, match="OpenAI API key not provided"):
                OpenAITranscriber()
    
    @patch('vttiro.providers.openai.transcriber.OpenAI')
    def test_initialization_with_api_key(self, mock_client_class):
        """Test successful initialization with API key."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'sk-test-key'}):
            transcriber = OpenAITranscriber()
            
            assert transcriber.provider_name == "openai"
            assert transcriber.supports_speaker_diarization is False
            assert len(transcriber.supported_languages) > 0
            assert transcriber.model_name == "whisper-1"
            mock_client_class.assert_called_once_with(api_key='sk-test-key')
    
    @patch('vttiro.providers.openai.transcriber.OpenAI')
    def test_initialization_with_parameter_api_key(self, mock_client_class):
        """Test initialization with API key as parameter."""
        transcriber = OpenAITranscriber(api_key="sk-param-key")
        
        assert transcriber.api_key == "sk-param-key"
        mock_client_class.assert_called_once_with(api_key='sk-param-key')
    
    @patch('vttiro.providers.openai.transcriber.OpenAI')
    def test_custom_model_setting(self, mock_client_class):
        """Test custom model configuration."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'sk-test-key'}):
            transcriber = OpenAITranscriber(model="whisper-1")
            
            assert transcriber.model_name == "whisper-1"
    
    def test_estimate_cost(self):
        """Test cost estimation functionality."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'sk-test-key'}):
            with patch('vttiro.providers.openai.transcriber.OpenAI'):
                transcriber = OpenAITranscriber()
                
                # Test normal duration
                cost = transcriber.estimate_cost(3600.0)  # 1 hour = 60 minutes
                expected_cost = 60 * 0.006  # $0.006 per minute
                assert cost == expected_cost
                
                # Test 30 minutes (1800 seconds)
                cost = transcriber.estimate_cost(1800.0)
                expected_cost = 30 * 0.006
                assert cost == expected_cost
                
                # Test invalid duration
                with pytest.raises(ValueError, match="Duration must be positive"):
                    transcriber.estimate_cost(0.0)
    
    @patch('vttiro.providers.openai.transcriber.OpenAI')
    def test_supported_languages(self, mock_client_class):
        """Test supported languages list."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'sk-test-key'}):
            transcriber = OpenAITranscriber()
            
            languages = transcriber.supported_languages
            assert isinstance(languages, list)
            assert len(languages) > 50  # Whisper supports many languages
            assert "en" in languages
            assert "es" in languages
            assert "fr" in languages
    
    @patch('vttiro.providers.openai.transcriber.OpenAI')
    def test_model_info(self, mock_client_class):
        """Test model information property."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'sk-test-key'}):
            transcriber = OpenAITranscriber()
            
            info = transcriber.model_info
            assert info["name"] == "whisper-1"
            assert info["provider"] == "openai"
            assert info["supports_word_timestamps"] is True
            assert info["supports_speaker_diarization"] is False
            assert info["max_file_size_mb"] == 25
            assert "mp3" in info["supported_formats"]
            assert "wav" in info["supported_formats"]
    
    @patch('vttiro.providers.openai.transcriber.OpenAI')
    @pytest.mark.asyncio
    async def test_transcribe_file_not_found(self, mock_client_class):
        """Test transcription with non-existent file."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'sk-test-key'}):
            transcriber = OpenAITranscriber()
            
            non_existent_file = Path("/nonexistent/file.wav")
            
            with pytest.raises(ProcessingError, match="Audio file not found"):
                await transcriber.transcribe(non_existent_file)
    
    @patch('vttiro.providers.openai.transcriber.OpenAI')
    @pytest.mark.asyncio
    async def test_transcribe_success(self, mock_client_class):
        """Test successful transcription."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'sk-test-key'}):
            # Create mock response with segments
            mock_segment1 = Mock()
            mock_segment1.start = 0.0
            mock_segment1.end = 5.0
            mock_segment1.text = "Hello world, this is a test."
            mock_segment1.avg_logprob = -0.2
            
            mock_segment2 = Mock()
            mock_segment2.start = 5.0
            mock_segment2.end = 10.0
            mock_segment2.text = "Thank you for listening."
            mock_segment2.avg_logprob = -0.1
            
            mock_response = Mock()
            mock_response.text = "Hello world, this is a test. Thank you for listening."
            mock_response.language = "en"
            mock_response.segments = [mock_segment1, mock_segment2]
            
            # Setup client mock
            mock_client = Mock()
            mock_transcribe = AsyncMock(return_value=mock_response)
            
            # Mock the transcription call using asyncio.to_thread pattern
            with patch('asyncio.to_thread', new=mock_transcribe):
                mock_client_class.return_value = mock_client
                mock_client.audio.transcriptions.create = Mock()
                
                transcriber = OpenAITranscriber()
                
                # Create temporary audio file
                with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_file:
                    audio_path = Path(tmp_file.name)
                    
                    result = await transcriber.transcribe(audio_path, language="en")
                    
                    # Verify result structure
                    assert result.provider == "openai"
                    assert result.language == "en"
                    assert len(result.segments) == 2
                    
                    # Check first segment
                    first_segment = result.segments[0]
                    assert first_segment.start == 0.0
                    assert first_segment.end == 5.0
                    assert "Hello world" in first_segment.text
                    # avg_logprob of -0.2 converts to confidence: (-0.2 + 3.0) / 3.0 = 0.933
                    assert abs(first_segment.confidence - 0.933) < 0.01
                    
                    # Check metadata
                    assert result.metadata["provider"] == "openai"
                    assert result.metadata["model"] == "whisper-1"
                    assert result.metadata["language_detected"] == "en"
    
    @patch('vttiro.providers.openai.transcriber.OpenAI')
    @pytest.mark.asyncio 
    async def test_transcribe_authentication_error(self, mock_client_class):
        """Test handling of authentication errors."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'sk-test-key'}):
            # Import the actual exception classes  
            from vttiro.providers.openai.transcriber import openai
            
            # Setup client to raise authentication error
            mock_client = Mock()
            mock_client.audio.transcriptions.create.side_effect = openai.AuthenticationError(
                message="Invalid API key",
                response=Mock(),
                body={}
            )
            mock_client_class.return_value = mock_client
            
            transcriber = OpenAITranscriber()
            
            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_file:
                audio_path = Path(tmp_file.name)
                
                with patch('asyncio.to_thread') as mock_to_thread:
                    mock_to_thread.side_effect = openai.AuthenticationError(
                        message="Invalid API key",
                        response=Mock(),
                        body={}
                    )
                    
                    with pytest.raises(AuthenticationError) as exc_info:
                        await transcriber.transcribe(audio_path)
                    
                    assert exc_info.value.provider == "openai"
    
    @patch('vttiro.providers.openai.transcriber.OpenAI')
    @pytest.mark.asyncio
    async def test_transcribe_rate_limit_error(self, mock_client_class):
        """Test handling of rate limit errors."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'sk-test-key'}):
            from vttiro.providers.openai.transcriber import openai
            
            transcriber = OpenAITranscriber()
            
            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_file:
                audio_path = Path(tmp_file.name)
                
                with patch('asyncio.to_thread') as mock_to_thread:
                    mock_to_thread.side_effect = openai.RateLimitError(
                        message="Rate limit exceeded",
                        response=Mock(),
                        body={}
                    )
                    
                    with pytest.raises(APIError, match="Rate limit exceeded") as exc_info:
                        await transcriber.transcribe(audio_path)
                    
                    assert exc_info.value.provider == "openai"
    
    @patch('vttiro.providers.openai.transcriber.OpenAI')
    @pytest.mark.asyncio
    async def test_transcribe_api_error(self, mock_client_class):
        """Test handling of general API errors."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'sk-test-key'}):
            from vttiro.providers.openai.transcriber import openai
            
            transcriber = OpenAITranscriber()
            
            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_file:
                audio_path = Path(tmp_file.name)
                
                with patch('asyncio.to_thread') as mock_to_thread:
                    mock_to_thread.side_effect = openai.APIStatusError(
                        message="Server error",
                        response=Mock(status_code=500),
                        body={}
                    )
                    
                    with pytest.raises(APIError, match="API error") as exc_info:
                        await transcriber.transcribe(audio_path)
                    
                    assert exc_info.value.provider == "openai"
    
    @patch('vttiro.providers.openai.transcriber.OpenAI')
    @pytest.mark.asyncio
    async def test_transcribe_with_context(self, mock_client_class):
        """Test transcription with context prompting."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'sk-test-key'}):
            # Create mock response
            mock_response = Mock()
            mock_response.text = "Hello world"
            mock_response.language = "en"
            mock_response.segments = []
            
            transcriber = OpenAITranscriber()
            
            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_file:
                audio_path = Path(tmp_file.name)
                
                with patch('asyncio.to_thread', return_value=mock_response):
                    result = await transcriber.transcribe(
                        audio_path, 
                        language="en",
                        context="This is a technical presentation about AI"
                    )
                    
                    # Verify context was processed
                    assert result.provider == "openai"
                    assert result.language == "en"
    
    @patch('vttiro.providers.openai.transcriber.OpenAI')
    def test_prepare_transcription_params(self, mock_client_class):
        """Test transcription parameter preparation."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'sk-test-key'}):
            transcriber = OpenAITranscriber()
            
            # Test basic parameters
            params = transcriber._prepare_transcription_params("en", None)
            assert params["model"] == "whisper-1"
            assert params["language"] == "en"
            assert params["response_format"] == "verbose_json"
            assert "timestamp_granularities" in params
            
            # Test with context
            params = transcriber._prepare_transcription_params("en", "AI presentation")
            assert "prompt" in params
            assert len(params["prompt"]) <= 224  # OpenAI limit
            
            # Test with temperature
            params = transcriber._prepare_transcription_params("en", None, temperature=0.2)
            assert params["temperature"] == 0.2
    
    @patch('vttiro.providers.openai.transcriber.OpenAI')
    def test_build_context_prompt(self, mock_client_class):
        """Test context prompt building."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'sk-test-key'}):
            transcriber = OpenAITranscriber()
            
            # Test with context
            prompt = transcriber._build_context_prompt("AI presentation about machine learning", "en")
            assert isinstance(prompt, str)
            assert len(prompt) <= 896  # Character limit for token approximation
            
            # Test with None context
            prompt = transcriber._build_context_prompt(None, "en")
            assert prompt == ""
    
    @patch('vttiro.providers.openai.transcriber.OpenAI')
    def test_estimate_confidence(self, mock_client_class):
        """Test confidence estimation from response."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'sk-test-key'}):
            transcriber = OpenAITranscriber()
            
            # Test with segment log probabilities
            mock_segment = Mock()
            mock_segment.avg_logprob = -0.5
            
            mock_response = Mock()
            mock_response.segments = [mock_segment]
            
            confidence = transcriber._estimate_confidence(mock_response, [])
            assert 0.0 <= confidence <= 1.0
            
            # Test fallback
            mock_response_no_segments = Mock()
            mock_response_no_segments.segments = []
            
            confidence = transcriber._estimate_confidence(mock_response_no_segments, [])
            assert confidence == 0.92


@pytest.mark.skipif(OPENAI_AVAILABLE, reason="OpenAI available, testing import error")
def test_import_error_handling():
    """Test graceful handling when OpenAI is not available."""
    # This test only runs when OpenAI is NOT available
    assert not OPENAI_AVAILABLE