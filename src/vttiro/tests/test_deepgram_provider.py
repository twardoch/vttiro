# this_file: src/vttiro/tests/test_deepgram_provider.py
"""Tests for the Deepgram transcription provider.

This module provides comprehensive tests for the DeepgramTranscriber class,
ensuring proper functionality, error handling, and integration with the
VTTiro 2.0 architecture.

Test coverage includes:
- Provider initialization and configuration
- Transcription functionality with various options
- Cost estimation for different models and features
- Error handling and edge cases
- Mock-based testing without external dependencies
"""

import os
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from vttiro.core.errors import AuthenticationError, ProcessingError
from vttiro.core.types import TranscriptionResult, TranscriptSegment

# Import availability check
try:
    from vttiro.providers.deepgram.transcriber import DEEPGRAM_AVAILABLE, DeepgramTranscriber
except ImportError:
    DEEPGRAM_AVAILABLE = False
    DeepgramTranscriber = None


class TestDeepgramTranscriber:
    """Test suite for DeepgramTranscriber functionality."""
    
    def test_missing_api_key_raises_error(self):
        """Test that missing API key raises AuthenticationError."""
        if not DEEPGRAM_AVAILABLE:
            pytest.skip("Deepgram not available")
            
        with patch.dict('os.environ', {}, clear=True):
            # Remove any existing API key
            if 'DEEPGRAM_API_KEY' in os.environ:
                del os.environ['DEEPGRAM_API_KEY']
            
            with patch('vttiro.providers.deepgram.transcriber.DeepgramClient'):
                with pytest.raises(AuthenticationError, match="Deepgram API key not provided"):
                    DeepgramTranscriber()
    
    @patch('vttiro.providers.deepgram.transcriber.DeepgramClient')
    def test_initialization_with_api_key(self, mock_client):
        """Test successful initialization with API key from environment."""
        if not DEEPGRAM_AVAILABLE:
            pytest.skip("Deepgram not available")
            
        with patch.dict('os.environ', {'DEEPGRAM_API_KEY': 'test-key'}):
            transcriber = DeepgramTranscriber()
            
            assert transcriber.api_key == 'test-key'
            assert transcriber.model_name == "nova-3"
            assert transcriber.enable_speaker_diarization is True
            assert transcriber.tier == "nova"
            mock_client.assert_called_once_with('test-key')
    
    @patch('vttiro.providers.deepgram.transcriber.DeepgramClient')
    def test_initialization_with_parameter_api_key(self, mock_client):
        """Test initialization with API key passed as parameter."""
        if not DEEPGRAM_AVAILABLE:
            pytest.skip("Deepgram not available")
            
        transcriber = DeepgramTranscriber(api_key="param-key")
        
        assert transcriber.api_key == "param-key"
        mock_client.assert_called_once_with("param-key")
    
    @patch('vttiro.providers.deepgram.transcriber.DeepgramClient')
    def test_custom_model_setting(self, mock_client):
        """Test initialization with custom model settings."""
        if not DEEPGRAM_AVAILABLE:
            pytest.skip("Deepgram not available")
            
        with patch.dict('os.environ', {'DEEPGRAM_API_KEY': 'test-key'}):
            transcriber = DeepgramTranscriber(
                model="nova-2",
                enable_speaker_diarization=False,
                tier="enhanced"
            )
            
            assert transcriber.model_name == "nova-2"
            assert transcriber.enable_speaker_diarization is False
            assert transcriber.tier == "enhanced"
    
    def test_estimate_cost(self):
        """Test cost estimation functionality."""
        with patch.dict('os.environ', {'DEEPGRAM_API_KEY': 'test-key'}):
            with patch('vttiro.providers.deepgram.transcriber.DeepgramClient'):
                # Test with speaker diarization disabled for base cost
                transcriber_no_diarization = DeepgramTranscriber(enable_speaker_diarization=False)
                cost = transcriber_no_diarization.estimate_cost(3600.0)  # 1 hour = 60 minutes
                expected_cost = 60 * 0.0043  # $0.0043 per minute for nova tier
                assert cost == expected_cost
                
                # Test with speaker diarization enabled (default)
                transcriber = DeepgramTranscriber()  # Default has diarization enabled
                cost_with_diarization = transcriber.estimate_cost(3600.0)
                expected_with_diarization = expected_cost + (60 * 0.0010)  # Diarization add-on
                assert cost_with_diarization == expected_with_diarization
                
                # Test invalid duration
                with pytest.raises(ValueError, match="Duration must be positive"):
                    transcriber.estimate_cost(0.0)
    
    @patch('vttiro.providers.deepgram.transcriber.DeepgramClient')
    def test_estimate_cost_different_tiers(self, mock_client):
        """Test cost estimation for different tiers."""
        with patch.dict('os.environ', {'DEEPGRAM_API_KEY': 'test-key'}):
            # Test enhanced tier (cheaper)
            transcriber_enhanced = DeepgramTranscriber(tier="enhanced", enable_speaker_diarization=False)
            cost_enhanced = transcriber_enhanced.estimate_cost(3600.0)
            expected_enhanced = 60 * 0.0025  # Lower cost per minute
            assert cost_enhanced == expected_enhanced
            
            # Test base tier (cheapest)
            transcriber_base = DeepgramTranscriber(tier="base", enable_speaker_diarization=False)
            cost_base = transcriber_base.estimate_cost(3600.0)
            expected_base = 60 * 0.0015  # Lowest cost per minute
            assert cost_base == expected_base
    
    @patch('vttiro.providers.deepgram.transcriber.DeepgramClient')
    def test_supported_languages(self, mock_client):
        """Test supported languages property."""
        with patch.dict('os.environ', {'DEEPGRAM_API_KEY': 'test-key'}):
            transcriber = DeepgramTranscriber()
            languages = transcriber.supported_languages
            
            assert isinstance(languages, list)
            assert "en" in languages
            assert "en-US" in languages
            assert "es" in languages
            assert "fr" in languages
            assert "de" in languages
            assert "zh-CN" in languages
            assert len(languages) > 30  # Deepgram supports many languages
    
    @patch('vttiro.providers.deepgram.transcriber.DeepgramClient')
    def test_model_info(self, mock_client):
        """Test model info property."""
        with patch.dict('os.environ', {'DEEPGRAM_API_KEY': 'test-key'}):
            transcriber = DeepgramTranscriber()
            info = transcriber.model_info
            
            assert info["name"] == "nova-3"
            assert info["provider"] == "deepgram"
            assert info["tier"] == "nova"
            assert info["supports_streaming"] is True
            assert info["supports_word_timestamps"] is True
            assert info["supports_speaker_diarization"] is True
            assert info["supports_language_detection"] is True
            assert info["max_file_size_mb"] == 2000
            assert "mp3" in info["supported_formats"]
            assert "wav" in info["supported_formats"]
    
    @patch('vttiro.providers.deepgram.transcriber.DeepgramClient')
    @pytest.mark.asyncio
    async def test_transcribe_file_not_found(self, mock_client):
        """Test transcription with non-existent file."""
        with patch.dict('os.environ', {'DEEPGRAM_API_KEY': 'test-key'}):
            transcriber = DeepgramTranscriber()
            
            non_existent_path = Path("/non/existent/file.wav")
            
            with pytest.raises(ProcessingError, match="Audio file not found"):
                await transcriber.transcribe(non_existent_path)
    
    @patch('vttiro.providers.deepgram.transcriber.DeepgramClient')
    @pytest.mark.asyncio
    async def test_transcribe_success(self, mock_client):
        """Test successful transcription with word-level data."""
        with patch.dict('os.environ', {'DEEPGRAM_API_KEY': 'test-key'}):
            # Create mock word objects
            mock_word1 = Mock()
            mock_word1.word = "Hello"
            mock_word1.start = 1.0
            mock_word1.end = 1.5
            mock_word1.confidence = 0.95
            mock_word1.speaker = 0
            
            mock_word2 = Mock()
            mock_word2.word = "world"
            mock_word2.start = 1.5
            mock_word2.end = 2.0
            mock_word2.confidence = 0.93
            mock_word2.speaker = 0
            
            mock_word3 = Mock()
            mock_word3.word = "Thank"
            mock_word3.start = 5.0
            mock_word3.end = 5.3
            mock_word3.confidence = 0.97
            mock_word3.speaker = 1
            
            mock_word4 = Mock()
            mock_word4.word = "you"
            mock_word4.start = 5.3
            mock_word4.end = 5.6
            mock_word4.confidence = 0.96
            mock_word4.speaker = 1
            
            # Create mock alternative - exclude paragraphs to force words fallback
            mock_alternative = Mock()
            mock_alternative.transcript = "Hello world Thank you"
            mock_alternative.confidence = 0.95
            mock_alternative.words = [mock_word1, mock_word2, mock_word3, mock_word4]
            # Ensure paragraphs attribute is None to skip that path
            mock_alternative.paragraphs = None
            
            # Create mock channel and result
            mock_channel = Mock()
            mock_channel.alternatives = [mock_alternative]
            mock_channel.detected_language = "en"
            
            mock_result = Mock()
            mock_result.channels = [mock_channel]
            mock_result.model_info = {"name": "nova-3", "version": "2024-01"}
            
            # Create mock response
            mock_response = Mock()
            mock_response.results = mock_result
            mock_response.request_id = "req-123"
            
            # Mock the API call
            transcriber = DeepgramTranscriber()
            transcriber._call_deepgram_api = AsyncMock(return_value=mock_response)
            
            # Create a temporary test file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                test_path = Path(f.name)
                f.write(b"fake audio data")
            
            try:
                # Perform transcription
                result = await transcriber.transcribe(test_path, language="en")
                
                # Verify result structure
                assert isinstance(result, TranscriptionResult)
                assert result.provider == "deepgram"
                assert result.language == "en"
                assert len(result.segments) == 2  # Two segments (different speakers)
                
                # Check segments
                first_segment = result.segments[0]
                assert first_segment.start == 1.0
                assert first_segment.end == 2.0
                assert "Hello world" in first_segment.text
                assert first_segment.speaker == "Speaker 0"
                
                second_segment = result.segments[1]
                assert second_segment.start == 5.0
                assert second_segment.end == 5.6
                assert "Thank you" in second_segment.text
                assert second_segment.speaker == "Speaker 1"
                
                # Check metadata
                assert result.metadata["provider"] == "deepgram"
                assert result.metadata["model"] == "nova-3"
                assert result.metadata["request_id"] == "req-123"
                assert result.metadata["language_detected"] == "en"
                
            finally:
                # Clean up
                test_path.unlink()
    
    @patch('vttiro.providers.deepgram.transcriber.DeepgramClient')
    @pytest.mark.asyncio
    async def test_transcribe_with_paragraphs_fallback(self, mock_client):
        """Test transcription fallback to paragraphs when words not available."""
        with patch.dict('os.environ', {'DEEPGRAM_API_KEY': 'test-key'}):
            # Create mock sentence objects
            mock_sentence1 = Mock()
            mock_sentence1.text = "Hello world, this is a test."
            mock_sentence1.start = 1.0
            mock_sentence1.end = 5.0
            mock_sentence1.confidence = 0.95
            
            mock_sentence2 = Mock()
            mock_sentence2.text = "Thank you for listening."
            mock_sentence2.start = 5.5
            mock_sentence2.end = 8.0
            mock_sentence2.confidence = 0.93
            
            # Create mock paragraph
            mock_paragraph = Mock()
            mock_paragraph.sentences = [mock_sentence1, mock_sentence2]
            
            mock_paragraphs = Mock()
            mock_paragraphs.paragraphs = [mock_paragraph]
            
            # Create mock alternative without words
            mock_alternative = Mock()
            mock_alternative.transcript = "Hello world, this is a test. Thank you for listening."
            mock_alternative.confidence = 0.94
            mock_alternative.paragraphs = mock_paragraphs
            # No words attribute
            
            mock_channel = Mock()
            mock_channel.alternatives = [mock_alternative]
            
            mock_result = Mock()
            mock_result.channels = [mock_channel]
            
            mock_response = Mock()
            mock_response.results = mock_result
            mock_response.request_id = "req-456"
            
            # Mock the API call
            transcriber = DeepgramTranscriber()
            transcriber._call_deepgram_api = AsyncMock(return_value=mock_response)
            
            # Create a temporary test file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                test_path = Path(f.name)
                f.write(b"fake audio data")
            
            try:
                # Perform transcription
                result = await transcriber.transcribe(test_path)
                
                # Verify result structure
                assert isinstance(result, TranscriptionResult)
                assert len(result.segments) == 2  # Two sentences
                
                # Check segments
                first_segment = result.segments[0]
                assert "Hello world, this is a test." in first_segment.text
                assert first_segment.start == 1.0
                assert first_segment.end == 5.0
                
                second_segment = result.segments[1]
                assert "Thank you for listening." in second_segment.text
                assert second_segment.start == 5.5
                assert second_segment.end == 8.0
                
            finally:
                # Clean up
                test_path.unlink()
    
    @patch('vttiro.providers.deepgram.transcriber.DeepgramClient')
    @pytest.mark.asyncio
    async def test_transcribe_with_context(self, mock_client):
        """Test transcription with context for keyword boosting."""
        with patch.dict('os.environ', {'DEEPGRAM_API_KEY': 'test-key'}):
            # Mock successful response - exclude paragraphs and words to force basic transcript fallback
            mock_alternative = Mock()
            mock_alternative.transcript = "Using TensorFlow for machine learning"
            mock_alternative.confidence = 0.96
            # No words or paragraphs - will use basic transcript fallback
            mock_alternative.paragraphs = None
            # Explicitly disable words attribute to force final fallback
            del mock_alternative.words
            
            mock_channel = Mock()
            mock_channel.alternatives = [mock_alternative]
            
            mock_result = Mock()
            mock_result.channels = [mock_channel]
            
            mock_response = Mock()
            mock_response.results = mock_result
            
            transcriber = DeepgramTranscriber()
            transcriber._call_deepgram_api = AsyncMock(return_value=mock_response)
            
            # Create a temporary test file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                test_path = Path(f.name)
                f.write(b"fake audio data")
            
            try:
                # Test with context containing technical terms
                context = "AI presentation about TensorFlow and PyTorch"
                result = await transcriber.transcribe(test_path, context=context)
                
                # Verify the transcription worked
                assert isinstance(result, TranscriptionResult)
                assert "TensorFlow" in result.segments[0].text
                
            finally:
                # Clean up
                test_path.unlink()
    
    @patch('vttiro.providers.deepgram.transcriber.DeepgramClient')
    def test_prepare_transcription_options(self, mock_client):
        """Test transcription options preparation."""
        with patch.dict('os.environ', {'DEEPGRAM_API_KEY': 'test-key'}):
            transcriber = DeepgramTranscriber()
            
            # Test basic options
            options = transcriber._prepare_transcription_options("en", None)
            
            assert options["model"] == "nova-3"
            assert options["tier"] == "nova"
            assert options["language"] == "en-US"  # Normalized
            assert options["punctuate"] is True
            assert options["diarize"] is True  # Default enabled
            assert options["smart_format"] is True
            
            # Test with context
            context = "TensorFlow machine learning"
            options_with_context = transcriber._prepare_transcription_options("fr", context)
            
            assert options_with_context["language"] == "fr"
            assert "keywords" in options_with_context
            assert "TensorFlow" in options_with_context["keywords"]
    
    @patch('vttiro.providers.deepgram.transcriber.DeepgramClient')
    def test_build_keywords(self, mock_client):
        """Test keyword building from context."""
        with patch.dict('os.environ', {'DEEPGRAM_API_KEY': 'test-key'}):
            transcriber = DeepgramTranscriber()
            
            # Test with technical context
            keywords = transcriber._build_keywords("TensorFlow and PyTorch for machine learning")
            assert isinstance(keywords, list)
            assert "TensorFlow" in keywords
            assert "PyTorch" in keywords
            
            # Test with empty context
            keywords = transcriber._build_keywords("")
            assert keywords == []
            
            # Test with business context
            business_context = "programming and software development with APIs"
            keywords = transcriber._build_keywords(business_context)
            # Should include some tech keywords due to 'programming' trigger
            assert len(keywords) > 0
    
    @patch('vttiro.providers.deepgram.transcriber.DeepgramClient')
    def test_normalize_language_code(self, mock_client):
        """Test language code normalization."""
        with patch.dict('os.environ', {'DEEPGRAM_API_KEY': 'test-key'}):
            transcriber = DeepgramTranscriber()
            
            # Test common mappings
            assert transcriber._normalize_language_code("en") == "en-US"
            assert transcriber._normalize_language_code("zh") == "zh-CN"
            assert transcriber._normalize_language_code("hi") == "hi-IN"
            
            # Test already specific codes
            assert transcriber._normalize_language_code("en-GB") == "en-GB"
            assert transcriber._normalize_language_code("fr-CA") == "fr-CA"
            
            # Test unknown codes (pass through)
            assert transcriber._normalize_language_code("xx-YY") == "xx-YY"
    
    @patch('vttiro.providers.deepgram.transcriber.DeepgramClient')
    def test_estimate_confidence(self, mock_client):
        """Test confidence estimation from response."""
        with patch.dict('os.environ', {'DEEPGRAM_API_KEY': 'test-key'}):
            transcriber = DeepgramTranscriber()
            
            # Test with alternative confidence
            mock_alternative = Mock()
            mock_alternative.confidence = 0.87
            
            confidence = transcriber._estimate_confidence(mock_alternative, [])
            assert confidence == 0.87
            
            # Test fallback without alternative confidence
            mock_alternative_no_conf = Mock()
            mock_alternative_no_conf.confidence = None
            
            segments = [
                TranscriptSegment(start=0, end=1, text="test", confidence=0.9),
                TranscriptSegment(start=1, end=2, text="test2", confidence=0.8)
            ]
            
            confidence = transcriber._estimate_confidence(mock_alternative_no_conf, segments)
            assert abs(confidence - 0.85) < 0.001  # Average of segment confidences
            
            # Test final fallback (no alternative, no segments)
            confidence = transcriber._estimate_confidence(None, [])
            assert confidence == 0.92  # Default high confidence for Deepgram


@pytest.mark.skipif(DEEPGRAM_AVAILABLE, reason="Deepgram available, testing import error")
def test_import_error_handling():
    """Test graceful handling when Deepgram SDK is not available."""
    # This test only runs when deepgram-sdk is not installed
    with pytest.raises(ImportError, match="Deepgram SDK not available"):
        DeepgramTranscriber(api_key="test-key")