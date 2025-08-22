# this_file: src/vttiro/tests/test_providers_base.py
"""Unit tests for provider base class.

Tests for TranscriberABC abstract base class, including ABC compliance,
validation methods, and contract enforcement.
"""

import tempfile
from pathlib import Path
from typing import Any

import pytest

from ..core.types import TranscriptionResult, TranscriptSegment
from ..providers.base import TranscriberABC


class MockTranscriber(TranscriberABC):
    """Mock transcriber for testing ABC implementation."""
    
    def __init__(self, provider_name: str = "mock"):
        self._provider_name = provider_name
    
    async def transcribe(
        self,
        audio_path: Path,
        language: str | None = None,
        context: str | None = None,
        **kwargs: Any
    ) -> TranscriptionResult:
        """Mock transcribe implementation."""
        segments = [
            TranscriptSegment(
                start=0.0,
                end=5.0,
                text="Mock transcription",
                confidence=0.95
            )
        ]
        
        return TranscriptionResult(
            segments=segments,
            metadata={"language": language, "context": context, **kwargs},
            provider=self._provider_name,
            language=language
        )
    
    def estimate_cost(self, duration_seconds: float) -> float:
        """Mock cost estimation."""
        if duration_seconds <= 0:
            raise ValueError("Duration must be positive")
        return duration_seconds * 0.01  # $0.01 per second
    
    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return self._provider_name


class IncompleteTranscriber(TranscriberABC):
    """Incomplete transcriber for testing ABC enforcement."""
    
    async def transcribe(self, audio_path: Path, **kwargs) -> TranscriptionResult:
        """Only implement transcribe method."""
        pass
    
    # Missing estimate_cost and provider_name implementations


class TestTranscriberABC:
    """Test TranscriberABC abstract base class."""
    
    def test_cannot_instantiate_abc_directly(self):
        """Test that ABC cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            TranscriberABC()
    
    def test_incomplete_implementation_fails(self):
        """Test that incomplete implementations fail to instantiate."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteTranscriber()
    
    def test_complete_implementation_succeeds(self):
        """Test that complete implementations can be instantiated."""
        transcriber = MockTranscriber("test")
        assert isinstance(transcriber, TranscriberABC)
        assert transcriber.provider_name == "test"
    
    @pytest.mark.asyncio
    async def test_transcribe_method_contract(self):
        """Test transcribe method contract."""
        transcriber = MockTranscriber("test")
        
        # Create temporary audio file for testing
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_file:
            audio_path = Path(tmp_file.name)
            
            result = await transcriber.transcribe(
                audio_path,
                language="en",
                context="test context"
            )
            
            assert isinstance(result, TranscriptionResult)
            assert result.provider == "test"
            assert result.language == "en"
            assert len(result.segments) == 1
            assert result.metadata["language"] == "en"
            assert result.metadata["context"] == "test context"
    
    def test_estimate_cost_method_contract(self):
        """Test estimate_cost method contract."""
        transcriber = MockTranscriber()
        
        # Valid duration
        cost = transcriber.estimate_cost(60.0)
        assert cost == 0.60  # 60 seconds * $0.01
        
        # Invalid duration should raise error
        with pytest.raises(ValueError, match="Duration must be positive"):
            transcriber.estimate_cost(0.0)
    
    def test_default_capabilities(self):
        """Test default capability properties."""
        transcriber = MockTranscriber()
        
        # Default capabilities should be False/empty
        assert transcriber.supports_speaker_diarization is False
        assert transcriber.supports_streaming is False
        assert transcriber.supported_languages == []
    
    def test_validate_audio_file_success(self):
        """Test audio file validation with valid file."""
        transcriber = MockTranscriber()
        
        # Create temporary audio file
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_file:
            audio_path = Path(tmp_file.name)
            
            # Should not raise any exceptions
            transcriber.validate_audio_file(audio_path)
    
    def test_validate_audio_file_not_found(self):
        """Test audio file validation with non-existent file."""
        transcriber = MockTranscriber()
        audio_path = Path("/nonexistent/file.wav")
        
        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            transcriber.validate_audio_file(audio_path)
    
    def test_validate_audio_file_not_file(self):
        """Test audio file validation with directory instead of file."""
        transcriber = MockTranscriber()
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            dir_path = Path(tmp_dir)
            
            with pytest.raises(ValueError, match="Path is not a file"):
                transcriber.validate_audio_file(dir_path)
    
    def test_validate_audio_file_unsupported_format(self):
        """Test audio file validation with unsupported format."""
        transcriber = MockTranscriber()
        
        with tempfile.NamedTemporaryFile(suffix=".txt") as tmp_file:
            text_path = Path(tmp_file.name)
            
            with pytest.raises(ValueError, match="Unsupported audio format"):
                transcriber.validate_audio_file(text_path)
    
    def test_supported_audio_formats(self):
        """Test that common audio formats are supported."""
        transcriber = MockTranscriber()
        
        supported_formats = [".wav", ".mp3", ".m4a", ".mp4", ".flac", ".ogg"]
        
        for ext in supported_formats:
            with tempfile.NamedTemporaryFile(suffix=ext) as tmp_file:
                audio_path = Path(tmp_file.name)
                
                # Should not raise exceptions for supported formats
                transcriber.validate_audio_file(audio_path)


class TestProviderCapabilities:
    """Test provider capability system."""
    
    def test_custom_capabilities(self):
        """Test overriding default capabilities."""
        
        class AdvancedTranscriber(MockTranscriber):
            @property
            def supports_speaker_diarization(self) -> bool:
                return True
            
            @property 
            def supports_streaming(self) -> bool:
                return True
            
            @property
            def supported_languages(self) -> list[str]:
                return ["en", "es", "fr"]
        
        transcriber = AdvancedTranscriber()
        
        assert transcriber.supports_speaker_diarization is True
        assert transcriber.supports_streaming is True
        assert transcriber.supported_languages == ["en", "es", "fr"]