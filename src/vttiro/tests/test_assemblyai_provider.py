# this_file: src/vttiro/tests/test_assemblyai_provider.py
"""Unit tests for AssemblyAI transcription provider."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from typing import Any

import pytest

from ..core.errors import AuthenticationError, ProcessingError, APIError
from ..core.types import TranscriptSegment
from .conftest import MockProviderMixin

# Test if AssemblyAI is available
try:
    from ..providers.assemblyai.transcriber import AssemblyAITranscriber, ASSEMBLYAI_AVAILABLE
except ImportError:
    ASSEMBLYAI_AVAILABLE = False


@pytest.mark.skipif(not ASSEMBLYAI_AVAILABLE, reason="AssemblyAI not available")
class TestAssemblyAITranscriber(MockProviderMixin):
    """Test AssemblyAITranscriber class.
    
    Uses MockProviderMixin for shared test utilities and session-scoped
    API key fixtures from conftest.py for improved performance.
    """
    
    def test_missing_api_key_raises_error(self):
        """Test that missing API key raises AuthenticationError."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(AuthenticationError, match="AssemblyAI API key not provided"):
                AssemblyAITranscriber()
    
    @patch('vttiro.providers.assemblyai.transcriber.aai')
    def test_initialization_with_api_key(self, mock_aai):
        """Test successful initialization with API key."""
        mock_aai.Transcriber.return_value = Mock()
        
        # API key already set by session fixture - no need for patch.dict
        transcriber = AssemblyAITranscriber()
        
        assert transcriber.provider_name == "assemblyai"
        assert transcriber.supports_speaker_diarization is True
        assert len(transcriber.supported_languages) > 0
        assert transcriber.model_name == "universal-2"
        mock_aai.Transcriber.assert_called_once()
    
    @patch('vttiro.providers.assemblyai.transcriber.aai')
    def test_initialization_with_parameter_api_key(self, mock_aai):
        """Test initialization with API key as parameter."""
        mock_aai.Transcriber.return_value = Mock()
        
        transcriber = AssemblyAITranscriber(api_key="param-key")
        
        assert transcriber.api_key == "param-key"
        assert mock_aai.settings.api_key == "param-key"
    
    @patch('vttiro.providers.assemblyai.transcriber.aai')
    def test_custom_model_setting(self, mock_aai):
        """Test custom model configuration."""
        mock_aai.Transcriber.return_value = Mock()
        
        # API key set by session fixture
        transcriber = AssemblyAITranscriber(model="universal-1")
        
        assert transcriber.model_name == "universal-1"
    
    def test_estimate_cost(self):
        """Test cost estimation functionality."""
        # API key set by session fixture, no environment patching needed
        with patch('vttiro.providers.assemblyai.transcriber.aai'):
            # Test with speaker diarization disabled for base cost
            transcriber_no_diarization = AssemblyAITranscriber(enable_speaker_diarization=False)
            cost = transcriber_no_diarization.estimate_cost(3600.0)  # 1 hour
            expected_cost = 3600 * 0.00074  # $0.00074 per second
            assert cost == expected_cost
            
            # Test with speaker diarization enabled (default)
            transcriber = AssemblyAITranscriber()  # Default has diarization enabled
            cost_with_diarization = transcriber.estimate_cost(3600.0)
            expected_with_diarization = expected_cost * 1.5  # 50% additional cost
            assert cost_with_diarization == expected_with_diarization
            
            # Test invalid duration
            with pytest.raises(ValueError, match="Duration must be positive"):
                transcriber.estimate_cost(0.0)
    
    @patch('vttiro.providers.assemblyai.transcriber.aai')
    def test_estimate_cost_different_models(self, mock_aai):
        """Test cost estimation for different models."""
        mock_aai.Transcriber.return_value = Mock()
        
        with patch.dict('os.environ', {'ASSEMBLYAI_API_KEY': 'test-key'}):
            # Test universal-1 model (cheaper)
            transcriber_u1 = AssemblyAITranscriber(model="universal-1", enable_speaker_diarization=False)
            cost_u1 = transcriber_u1.estimate_cost(3600.0)
            expected_u1 = 3600 * 0.00037  # Lower cost per second
            assert cost_u1 == expected_u1
            
            # Test universal-2 model (premium)
            transcriber_u2 = AssemblyAITranscriber(model="universal-2", enable_speaker_diarization=False)
            cost_u2 = transcriber_u2.estimate_cost(3600.0)
            expected_u2 = 3600 * 0.00074  # Higher cost per second
            assert cost_u2 == expected_u2
    
    @patch('vttiro.providers.assemblyai.transcriber.aai')
    def test_supported_languages(self, mock_aai):
        """Test supported languages list."""
        mock_aai.Transcriber.return_value = Mock()
        
        with patch.dict('os.environ', {'ASSEMBLYAI_API_KEY': 'test-key'}):
            transcriber = AssemblyAITranscriber()
            
            languages = transcriber.supported_languages
            assert isinstance(languages, list)
            assert len(languages) > 30  # AssemblyAI supports many languages
            assert "en" in languages
            assert "es" in languages
            assert "fr" in languages
            assert "zh" in languages
    
    @patch('vttiro.providers.assemblyai.transcriber.aai')
    def test_model_info(self, mock_aai):
        """Test model information property."""
        mock_aai.Transcriber.return_value = Mock()
        
        with patch.dict('os.environ', {'ASSEMBLYAI_API_KEY': 'test-key'}):
            transcriber = AssemblyAITranscriber()
            
            info = transcriber.model_info
            assert info["name"] == "universal-2"
            assert info["provider"] == "assemblyai"
            assert info["supports_word_timestamps"] is True
            assert info["supports_speaker_diarization"] is True
            assert info["supports_auto_highlights"] is True
            assert info["supports_entity_detection"] is True
            assert info["max_file_size_mb"] == 500
            assert "mp3" in info["supported_formats"]
            assert "wav" in info["supported_formats"]
    
    @patch('vttiro.providers.assemblyai.transcriber.aai')
    @pytest.mark.asyncio
    async def test_transcribe_file_not_found(self, mock_aai):
        """Test transcription with non-existent file."""
        mock_aai.Transcriber.return_value = Mock()
        
        with patch.dict('os.environ', {'ASSEMBLYAI_API_KEY': 'test-key'}):
            transcriber = AssemblyAITranscriber()
            
            non_existent_file = Path("/nonexistent/file.wav")
            
            with pytest.raises(ProcessingError, match="Audio file not found"):
                await transcriber.transcribe(non_existent_file)
    
    @patch('vttiro.providers.assemblyai.transcriber.aai')
    @pytest.mark.asyncio
    async def test_transcribe_success(self, mock_aai):
        """Test successful transcription."""
        with patch.dict('os.environ', {'ASSEMBLYAI_API_KEY': 'test-key'}):
            # Create mock word objects
            mock_word1 = Mock()
            mock_word1.text = "Hello"
            mock_word1.start = 1000  # 1 second in ms
            mock_word1.end = 1500
            mock_word1.confidence = 0.95
            mock_word1.speaker = "A"
            
            mock_word2 = Mock()
            mock_word2.text = "world"
            mock_word2.start = 1500
            mock_word2.end = 2000
            mock_word2.confidence = 0.93
            mock_word2.speaker = "A"
            
            mock_word3 = Mock()
            mock_word3.text = "Thank"
            mock_word3.start = 5000
            mock_word3.end = 5300
            mock_word3.confidence = 0.97
            mock_word3.speaker = "B"
            
            mock_word4 = Mock()
            mock_word4.text = "you"
            mock_word4.start = 5300
            mock_word4.end = 5600
            mock_word4.confidence = 0.96
            mock_word4.speaker = "B"
            
            # Create mock transcript
            mock_transcript = Mock()
            mock_transcript.id = "transcript-123"
            mock_transcript.status = "completed"
            mock_transcript.text = "Hello world Thank you"
            mock_transcript.confidence = 0.95
            mock_transcript.language_code = "en"
            mock_transcript.audio_duration = 6.0
            mock_transcript.words = [mock_word1, mock_word2, mock_word3, mock_word4]
            mock_transcript.auto_highlights_result = None
            mock_transcript.entities = None
            
            # Setup transcriber mock
            mock_transcriber_instance = Mock()
            mock_transcriber_instance.transcribe = Mock(return_value=mock_transcript)
            mock_aai.Transcriber.return_value = mock_transcriber_instance
            mock_aai.TranscriptionConfig.return_value = Mock()
            mock_aai.SpeechModel.universal_2 = "universal-2"
            
            transcriber = AssemblyAITranscriber()
            
            # Create temporary audio file
            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_file:
                audio_path = Path(tmp_file.name)
                
                with patch('asyncio.to_thread', return_value=mock_transcript):
                    result = await transcriber.transcribe(audio_path, language="en")
                    
                    # Verify result structure
                    assert result.provider == "assemblyai"
                    assert result.language == "en"
                    assert len(result.segments) == 2  # Two different speakers
                    
                    # Check first segment (Speaker A)
                    first_segment = result.segments[0]
                    assert first_segment.start == 1.0
                    assert first_segment.end == 2.0
                    assert "Hello world" in first_segment.text
                    assert first_segment.speaker == "Speaker A"
                    assert first_segment.confidence == 0.94  # Average of word confidences
                    
                    # Check second segment (Speaker B)
                    second_segment = result.segments[1]
                    assert second_segment.start == 5.0
                    assert second_segment.end == 5.6
                    assert "Thank you" in second_segment.text
                    assert second_segment.speaker == "Speaker B"
                    
                    # Check metadata
                    assert result.metadata["provider"] == "assemblyai"
                    assert result.metadata["model"] == "universal-2"
                    assert result.metadata["transcript_id"] == "transcript-123"
                    assert result.metadata["language_detected"] == "en"
    
    @patch('vttiro.providers.assemblyai.transcriber.aai')
    @pytest.mark.asyncio
    async def test_transcribe_with_utterances_fallback(self, mock_aai):
        """Test transcription fallback to utterances when words not available."""
        with patch.dict('os.environ', {'ASSEMBLYAI_API_KEY': 'test-key'}):
            # Create mock utterance objects
            mock_utterance1 = Mock()
            mock_utterance1.text = "Hello world, this is a test."
            mock_utterance1.start = 1000  # 1 second in ms
            mock_utterance1.end = 5000
            mock_utterance1.confidence = 0.95
            mock_utterance1.speaker = "A"
            
            mock_utterance2 = Mock()
            mock_utterance2.text = "Thank you for listening."
            mock_utterance2.start = 5500
            mock_utterance2.end = 8000
            mock_utterance2.confidence = 0.93
            mock_utterance2.speaker = "B"
            
            # Create mock transcript without words but with utterances
            mock_transcript = Mock()
            mock_transcript.id = "transcript-456"
            mock_transcript.text = "Hello world, this is a test. Thank you for listening."
            mock_transcript.confidence = 0.94
            mock_transcript.language_code = "en"
            mock_transcript.words = None  # No word-level timestamps
            mock_transcript.utterances = [mock_utterance1, mock_utterance2]
            
            mock_transcriber_instance = Mock()
            mock_transcriber_instance.transcribe = Mock(return_value=mock_transcript)
            mock_aai.Transcriber.return_value = mock_transcriber_instance
            mock_aai.TranscriptionConfig.return_value = Mock()
            
            transcriber = AssemblyAITranscriber()
            
            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_file:
                audio_path = Path(tmp_file.name)
                
                with patch('asyncio.to_thread', return_value=mock_transcript):
                    result = await transcriber.transcribe(audio_path)
                    
                    # Verify result structure
                    assert result.provider == "assemblyai"
                    assert len(result.segments) == 2
                    
                    # Check segments from utterances
                    first_segment = result.segments[0]
                    assert first_segment.start == 1.0
                    assert first_segment.end == 5.0
                    assert first_segment.text == "Hello world, this is a test."
                    assert first_segment.speaker == "Speaker A"
                    
                    second_segment = result.segments[1]
                    assert second_segment.start == 5.5
                    assert second_segment.end == 8.0
                    assert second_segment.text == "Thank you for listening."
                    assert second_segment.speaker == "Speaker B"
    
    @patch('vttiro.providers.assemblyai.transcriber.aai')
    @pytest.mark.asyncio
    async def test_transcribe_with_context(self, mock_aai):
        """Test transcription with context prompting."""
        with patch.dict('os.environ', {'ASSEMBLYAI_API_KEY': 'test-key'}):
            mock_transcript = Mock()
            mock_transcript.text = "Hello world"
            mock_transcript.confidence = 0.95
            mock_transcript.language_code = "en"
            mock_transcript.words = []
            mock_transcript.utterances = None
            
            mock_transcriber_instance = Mock()
            mock_transcriber_instance.transcribe = Mock(return_value=mock_transcript)
            mock_aai.Transcriber.return_value = mock_transcriber_instance
            mock_aai.TranscriptionConfig.return_value = Mock()
            
            transcriber = AssemblyAITranscriber()
            
            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_file:
                audio_path = Path(tmp_file.name)
                
                with patch('asyncio.to_thread', return_value=mock_transcript):
                    result = await transcriber.transcribe(
                        audio_path, 
                        language="en",
                        context="This is a technical presentation about AI and Machine Learning"
                    )
                    
                    # Verify context was processed
                    assert result.provider == "assemblyai"
                    assert result.language == "en"
    
    @patch('vttiro.providers.assemblyai.transcriber.aai')
    def test_prepare_transcription_config(self, mock_aai):
        """Test transcription configuration preparation."""
        mock_aai.Transcriber.return_value = Mock()
        mock_aai.TranscriptionConfig.return_value = Mock()
        mock_aai.SpeechModel.universal_2 = "universal-2"
        mock_aai.types.WordBoost.high = "high"
        mock_aai.types.WordBoost.default = "default"
        
        with patch.dict('os.environ', {'ASSEMBLYAI_API_KEY': 'test-key'}):
            transcriber = AssemblyAITranscriber()
            
            # Test basic parameters
            config = transcriber._prepare_transcription_config("en", None)
            assert isinstance(config, Mock)  # Mocked TranscriptionConfig
            
            # Test with context
            config = transcriber._prepare_transcription_config("en", "AI presentation with TensorFlow")
            assert isinstance(config, Mock)
    
    @patch('vttiro.providers.assemblyai.transcriber.aai')
    def test_build_custom_vocabulary(self, mock_aai):
        """Test custom vocabulary building."""
        mock_aai.Transcriber.return_value = Mock()
        
        with patch.dict('os.environ', {'ASSEMBLYAI_API_KEY': 'test-key'}):
            transcriber = AssemblyAITranscriber()
            
            # Test with context  
            vocab = transcriber._build_custom_vocabulary("AI presentation with TensorFlow and PyTorch")
            assert isinstance(vocab, list)
            assert "TensorFlow" in vocab
            assert "PyTorch" in vocab
            # Note: "AI" is not included because it's only 2 characters long
            
            # Test with empty context
            vocab = transcriber._build_custom_vocabulary("")
            assert vocab == []
            
            # Test with None context
            vocab = transcriber._build_custom_vocabulary(None)
            assert vocab == []
    
    @patch('vttiro.providers.assemblyai.transcriber.aai')
    def test_estimate_confidence(self, mock_aai):
        """Test confidence estimation from response."""
        mock_aai.Transcriber.return_value = Mock()
        
        with patch.dict('os.environ', {'ASSEMBLYAI_API_KEY': 'test-key'}):
            transcriber = AssemblyAITranscriber()
            
            # Test with transcript confidence
            mock_transcript = Mock()
            mock_transcript.confidence = 0.87
            
            confidence = transcriber._estimate_confidence(mock_transcript, [])
            assert confidence == 0.87
            
            # Test fallback without transcript confidence
            mock_transcript_no_conf = Mock()
            mock_transcript_no_conf.confidence = None
            
            segments = [
                TranscriptSegment(start=0, end=1, text="test", confidence=0.9),
                TranscriptSegment(start=1, end=2, text="test2", confidence=0.8)
            ]
            
            confidence = transcriber._estimate_confidence(mock_transcript_no_conf, segments)
            assert abs(confidence - 0.85) < 0.001  # Average of segment confidences
            
            # Test final fallback (no confidence, no segments)
            mock_transcript_fallback = Mock()
            mock_transcript_fallback.confidence = None
            confidence = transcriber._estimate_confidence(mock_transcript_fallback, [])
            assert confidence == 0.94  # Default high confidence for AssemblyAI

@pytest.mark.skipif(ASSEMBLYAI_AVAILABLE, reason="AssemblyAI available, testing import error")
def test_import_error_handling():
    """Test graceful handling when AssemblyAI is not available."""
    # This test only runs when AssemblyAI is NOT available
    assert not ASSEMBLYAI_AVAILABLE