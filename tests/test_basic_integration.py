#!/usr/bin/env python3
# this_file: tests/test_basic_integration.py
"""Basic integration tests for VTTiro core functionality."""

import pytest

from vttiro.cli import VttiroCLI
from vttiro.core.config import VttiroConfig
from vttiro.core.errors import VttiroError
from vttiro.core.transcriber import Transcriber
from vttiro.core.types import TranscriptionResult
from vttiro.processing.audio import AudioProcessor, create_audio_processor
from vttiro.providers import (
    ASSEMBLYAI_AVAILABLE,
    DEEPGRAM_AVAILABLE,
    GEMINI_AVAILABLE,
    OPENAI_AVAILABLE,
    AssemblyAITranscriber,
    DeepgramTranscriber,
    GeminiTranscriber,
    OpenAITranscriber,
)
from vttiro.utils.api_keys import get_api_key_with_fallbacks
from vttiro.utils.input_validation import validate_file_path


class TestBasicIntegration:
    """Basic integration tests for core VTTiro components."""

    def test_basic_imports(self):
        """Test that core modules import correctly."""
        # All imports should succeed without errors
        assert VttiroConfig is not None
        assert Transcriber is not None
        assert TranscriptionResult is not None
        assert VttiroError is not None
        assert validate_file_path is not None
        assert AudioProcessor is not None
        assert create_audio_processor is not None

    def test_config_creation(self):
        """Test basic configuration creation."""
        config = VttiroConfig()
        assert config is not None
        assert hasattr(config, "engine")
        assert hasattr(config, "model")

    def test_transcriber_creation(self):
        """Test basic transcriber creation."""
        config = VttiroConfig()
        transcriber = Transcriber(config)
        assert transcriber is not None
        assert transcriber.config == config

    def test_input_validator_function(self):
        """Test input validator function import."""
        assert validate_file_path is not None

    def test_audio_processor_factory(self):
        """Test audio processor factory function."""
        processor = create_audio_processor(debug=True)
        assert processor is not None
        assert isinstance(processor, AudioProcessor)

    def test_transcription_result_creation(self):
        """Test TranscriptionResult creation with basic fields."""
        result = TranscriptionResult(segments=[], metadata={}, provider="test", language="en")
        assert result is not None
        assert result.language == "en"
        assert result.provider == "test"
        assert result.duration() == 0.0
        assert len(result.segments) == 0

    def test_error_hierarchy(self):
        """Test that error hierarchy works correctly."""
        error = VttiroError("test error", "TEST_ERROR")
        assert error is not None
        assert str(error) == "[TEST_ERROR] test error"
        assert error.error_code == "TEST_ERROR"

        # Test that VttiroError can be caught as VttiroError
        with pytest.raises(VttiroError, match="test error"):
            raise error

    @pytest.mark.asyncio
    async def test_audio_processor_basic_functionality(self):
        """Test audio processor basic functionality without actual files."""
        processor = create_audio_processor()

        # Test memory stats retrieval
        stats = processor.get_memory_stats()
        assert isinstance(stats, dict)
        assert "process_mb" in stats
        assert "system_available_mb" in stats
        assert "system_percent" in stats

    def test_cli_import(self):
        """Test that CLI module imports correctly."""
        assert VttiroCLI is not None

    def test_providers_import(self):
        """Test that provider modules import correctly."""
        # These should import correctly even if None
        assert GeminiTranscriber is not None or not GEMINI_AVAILABLE
        assert OpenAITranscriber is not None or not OPENAI_AVAILABLE
        assert DeepgramTranscriber is not None or not DEEPGRAM_AVAILABLE
        assert AssemblyAITranscriber is not None or not ASSEMBLYAI_AVAILABLE


class TestSmokeFunctionality:
    """Smoke tests to ensure basic functionality works without external dependencies."""

    def test_config_defaults(self):
        """Test that config has reasonable defaults."""
        config = VttiroConfig()

        # Test that basic attributes exist and have sensible defaults
        assert hasattr(config, "engine")
        assert hasattr(config, "model")
        assert hasattr(config, "output_format")

        # Defaults should be set
        assert config.engine is not None
        assert config.output_format is not None
        # Note: model can be None as it's set dynamically based on engine

    def test_api_key_utilities(self):
        """Test API key utility functions."""
        # Test function exists and handles None gracefully
        result = get_api_key_with_fallbacks("GEMINI", None)
        # Should return None when no API keys are set, but not crash
        assert result is None or isinstance(result, str)
