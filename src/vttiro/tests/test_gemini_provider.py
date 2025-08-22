# this_file: src/vttiro/tests/test_gemini_provider.py
"""Unit tests for Gemini transcription provider."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from typing import Any

import pytest

from ..core.errors import AuthenticationError, ContentFilterError, APIError
from ..core.types import TranscriptSegment

# Test if Gemini is available
try:
    from ..providers.gemini.transcriber import GeminiTranscriber, GEMINI_AVAILABLE
except ImportError:
    GEMINI_AVAILABLE = False


@pytest.mark.skipif(not GEMINI_AVAILABLE, reason="Google GenerativeAI not available")
class TestGeminiTranscriber:
    """Test GeminiTranscriber class."""
    
    def test_missing_api_key_raises_error(self):
        """Test that missing API key raises AuthenticationError."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(AuthenticationError, match="Gemini API key not provided"):
                GeminiTranscriber()
    
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_initialization_with_api_key(self, mock_model, mock_configure):
        """Test successful initialization with API key."""
        with patch.dict('os.environ', {'GEMINI_API_KEY': 'test-key'}):
            transcriber = GeminiTranscriber()
            
            assert transcriber.provider_name == "gemini"
            assert transcriber.supports_speaker_diarization is True
            assert len(transcriber.supported_languages) > 0
            mock_configure.assert_called_once_with(api_key='test-key')
    
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_initialization_with_parameter_api_key(self, mock_model, mock_configure):
        """Test initialization with API key as parameter."""
        transcriber = GeminiTranscriber(api_key="param-key")
        
        assert transcriber.api_key == "param-key"
        mock_configure.assert_called_once_with(api_key='param-key')
    
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_custom_model_setting(self, mock_model, mock_configure):
        """Test custom model configuration."""
        with patch.dict('os.environ', {'GEMINI_API_KEY': 'test-key'}):
            transcriber = GeminiTranscriber(model="gemini-pro")
            
            assert transcriber.model_name == "gemini-pro"
    
    def test_estimate_cost(self):
        """Test cost estimation functionality."""
        with patch.dict('os.environ', {'GEMINI_API_KEY': 'test-key'}):
            with patch('google.generativeai.configure'), \
                 patch('google.generativeai.GenerativeModel'):
                transcriber = GeminiTranscriber()
                
                # Test normal duration
                cost = transcriber.estimate_cost(3600.0)  # 1 hour
                assert cost == 1.20  # $1.20 per hour
                
                # Test 30 minutes
                cost = transcriber.estimate_cost(1800.0)
                assert cost == 0.60
                
                # Test invalid duration
                with pytest.raises(ValueError, match="Duration must be positive"):
                    transcriber.estimate_cost(0.0)
    
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_safety_settings_configuration(self, mock_model, mock_configure):
        """Test safety settings configuration."""
        with patch.dict('os.environ', {'GEMINI_API_KEY': 'test-key'}):
            # Test with custom safety settings
            custom_settings = {"harassment": "high", "hate_speech": "medium"}
            transcriber = GeminiTranscriber(safety_settings=custom_settings)
            
            # Should initialize without error
            assert transcriber.provider_name == "gemini"
    
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    @pytest.mark.asyncio
    async def test_transcribe_file_not_found(self, mock_model, mock_configure):
        """Test transcription with non-existent file."""
        with patch.dict('os.environ', {'GEMINI_API_KEY': 'test-key'}):
            transcriber = GeminiTranscriber()
            
            non_existent_file = Path("/nonexistent/file.wav")
            
            with pytest.raises(FileNotFoundError, match="Audio file not found"):
                await transcriber.transcribe(non_existent_file)
    
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    @patch('google.generativeai.upload_file')
    @pytest.mark.asyncio
    async def test_transcribe_success(self, mock_upload, mock_model_class, mock_configure):
        """Test successful transcription."""
        with patch.dict('os.environ', {'GEMINI_API_KEY': 'test-key'}):
            # Create mock response
            mock_response = Mock()
            mock_response.text = """WEBVTT

00:00:01.000 --> 00:00:05.000
Hello world, this is a test.

00:00:05.500 --> 00:00:10.000
<v Speaker 2>Thank you for listening."""
            
            mock_response.candidates = [Mock()]
            mock_response.candidates[0].finish_reason = 1  # STOP (normal completion)
            mock_response.safety_ratings = []
            
            # Setup mocks
            mock_audio_file = Mock()
            mock_audio_file.display_name = "test.wav"
            mock_upload.return_value = mock_audio_file
            
            mock_model = Mock()
            mock_model.generate_content = Mock(return_value=mock_response)
            mock_model_class.return_value = mock_model
            
            transcriber = GeminiTranscriber()
            
            # Create temporary audio file
            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_file:
                audio_path = Path(tmp_file.name)
                
                result = await transcriber.transcribe(audio_path, language="en")
                
                # Verify result structure
                assert result.provider == "gemini"
                assert result.language == "en"
                assert len(result.segments) == 2
                
                # Check first segment
                first_segment = result.segments[0]
                assert first_segment.start == 1.0
                assert first_segment.end == 5.0
                assert "Hello world" in first_segment.text
                assert first_segment.confidence == 0.95
                
                # Check metadata
                assert result.metadata["provider"] == "gemini"
                assert result.metadata["language"] == "en"
                assert "webvtt_content" in result.metadata
    
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    @patch('google.generativeai.upload_file')
    @pytest.mark.asyncio
    async def test_transcribe_safety_filter_blocking(self, mock_upload, mock_model_class, mock_configure):
        """Test handling of safety filter blocking."""
        with patch.dict('os.environ', {'GEMINI_API_KEY': 'test-key'}):
            # Create mock response with safety blocking
            mock_response = Mock()
            mock_response.candidates = [Mock()]
            mock_response.candidates[0].finish_reason = 2  # SAFETY filter
            
            mock_rating = Mock()
            mock_rating.blocked = True
            mock_rating.category = "HARM_CATEGORY_HARASSMENT"
            mock_response.safety_ratings = [mock_rating]
            
            # Setup mocks
            mock_audio_file = Mock()
            mock_upload.return_value = mock_audio_file
            
            mock_model = Mock()
            mock_model.generate_content = Mock(return_value=mock_response)
            mock_model_class.return_value = mock_model
            
            transcriber = GeminiTranscriber()
            
            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_file:
                audio_path = Path(tmp_file.name)
                
                with pytest.raises(ContentFilterError) as exc_info:
                    await transcriber.transcribe(audio_path)
                
                assert exc_info.value.provider == "gemini"
                assert "HARM_CATEGORY_HARASSMENT" in exc_info.value.blocked_categories
    
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    @patch('google.generativeai.upload_file')
    @pytest.mark.asyncio
    async def test_transcribe_upload_failure(self, mock_upload, mock_model_class, mock_configure):
        """Test handling of upload failure."""
        with patch.dict('os.environ', {'GEMINI_API_KEY': 'test-key'}):
            # Make upload fail
            mock_upload.side_effect = Exception("Upload failed")
            
            mock_model_class.return_value = Mock()
            transcriber = GeminiTranscriber()
            
            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_file:
                audio_path = Path(tmp_file.name)
                
                with pytest.raises(Exception) as exc_info:
                    await transcriber.transcribe(audio_path)
                
                # Should be wrapped in a VTTiro error
                assert "upload" in str(exc_info.value).lower()
    
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    @patch('google.generativeai.upload_file')
    @pytest.mark.asyncio
    async def test_transcribe_api_failure(self, mock_upload, mock_model_class, mock_configure):
        """Test handling of API call failure."""
        with patch.dict('os.environ', {'GEMINI_API_KEY': 'test-key'}):
            # Setup upload success but API failure
            mock_audio_file = Mock()
            mock_upload.return_value = mock_audio_file
            
            mock_model = Mock()
            api_error = Exception("API call failed")
            api_error.status_code = 500
            mock_model.generate_content = Mock(side_effect=api_error)
            mock_model_class.return_value = mock_model
            
            transcriber = GeminiTranscriber()
            
            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_file:
                audio_path = Path(tmp_file.name)
                
                with pytest.raises(APIError) as exc_info:
                    await transcriber.transcribe(audio_path)
                
                assert exc_info.value.provider == "gemini"
    
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_prompt_generation(self, mock_model, mock_configure):
        """Test prompt generation functionality."""
        with patch.dict('os.environ', {'GEMINI_API_KEY': 'test-key'}):
            transcriber = GeminiTranscriber()
            
            # Test basic prompt generation
            prompt = transcriber._generate_transcription_prompt("en", "Meeting recording")
            
            assert "WEBVTT" in prompt
            assert "EN" in prompt.upper() or "english" in prompt.lower()
            assert "meeting" in prompt.lower()
            assert "gemini" in prompt.lower()  # Should have Gemini-specific optimizations
    
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_webvtt_parsing(self, mock_model, mock_configure):
        """Test WebVTT content parsing."""
        with patch.dict('os.environ', {'GEMINI_API_KEY': 'test-key'}):
            transcriber = GeminiTranscriber()
            
            webvtt_content = """WEBVTT

00:00:01.000 --> 00:00:05.000
Hello world, this is a test.

00:00:05.500 --> 00:00:10.000
<v Speaker 2>Thank you for listening.

00:00:10.500 --> 00:00:15.000
Final segment here."""
            
            segments = transcriber._parse_webvtt_response(webvtt_content)
            
            assert len(segments) == 3
            
            # Check first segment
            assert segments[0].start == 1.0
            assert segments[0].end == 5.0
            assert "Hello world" in segments[0].text
            
            # Check second segment (with speaker tag)
            assert segments[1].start == 5.5
            assert segments[1].end == 10.0
            assert "Thank you" in segments[1].text
            
            # Check third segment
            assert segments[2].start == 10.5
            assert segments[2].end == 15.0
            assert "Final segment" in segments[2].text
    
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_empty_webvtt_parsing(self, mock_model, mock_configure):
        """Test parsing of empty or invalid WebVTT."""
        with patch.dict('os.environ', {'GEMINI_API_KEY': 'test-key'}):
            transcriber = GeminiTranscriber()
            
            # Test empty content
            segments = transcriber._parse_webvtt_response("")
            assert len(segments) == 0
            
            # Test malformed content
            segments = transcriber._parse_webvtt_response("Not valid WebVTT content")
            assert len(segments) == 0
            
            # Test WebVTT header only
            segments = transcriber._parse_webvtt_response("WEBVTT\n\n")
            assert len(segments) == 0


@pytest.mark.skipif(GEMINI_AVAILABLE, reason="Testing import failure behavior")
class TestGeminiImportFailure:
    """Test behavior when Gemini dependencies are not available."""
    
    def test_import_error_on_initialization(self):
        """Test that ImportError is raised when dependencies are missing."""
        # This test only runs when GEMINI_AVAILABLE is False
        with pytest.raises(ImportError, match="Google GenerativeAI not available"):
            from ..providers.gemini.transcriber import GeminiTranscriber
            GeminiTranscriber(api_key="test-key")