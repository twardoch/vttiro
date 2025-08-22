#!/usr/bin/env python3
# this_file: tests/test_config.py
"""Unit tests for vttiro configuration system."""

from pathlib import Path
from unittest.mock import patch

import pytest

from vttiro.core.config import VttiroConfig
from vttiro.core.errors import ValidationError
from vttiro.core.types import TranscriptionResult


class TestVttiroConfig:
    """Test VttiroConfig class functionality."""

    def test_config_creation_with_defaults(self):
        """Test configuration creation with default values."""
        config = VttiroConfig()

        # Test default values are set
        assert config.transcription is not None
        assert config.processing is not None
        assert config.output is not None

        # Test default transcription settings
        assert config.transcription.preferred_model == "auto"
        assert config.transcription.language == "auto"
        assert config.transcription.max_duration_seconds == 3600
        assert config.transcription.chunk_duration_seconds == 30

    def test_config_validation_success(self):
        """Test successful configuration validation."""
        config = VttiroConfig()

        # Should not raise any exceptions
        config.validate()

    def test_config_validation_invalid_duration(self):
        """Test configuration validation with invalid duration."""
        config = VttiroConfig()
        config.transcription.max_duration_seconds = -1

        with pytest.raises(ValidationError, match="max_duration_seconds must be positive"):
            config.validate()

    def test_config_validation_invalid_chunk_size(self):
        """Test configuration validation with invalid chunk size."""
        config = VttiroConfig()
        config.transcription.chunk_duration_seconds = 0

        with pytest.raises(ValidationError, match="chunk_duration_seconds must be positive"):
            config.validate()

    def test_config_update_from_env(self, monkeypatch):
        """Test configuration update from environment variables."""
        # Set environment variables
        monkeypatch.setenv("VTTIRO_PREFERRED_MODEL", "gemini")
        monkeypatch.setenv("VTTIRO_LANGUAGE", "es")
        monkeypatch.setenv("VTTIRO_MAX_DURATION", "7200")
        monkeypatch.setenv("VTTIRO_GEMINI_API_KEY", "test_key_123")

        config = VttiroConfig()
        config.update_from_env()

        assert config.transcription.preferred_model == "gemini"
        assert config.transcription.language == "es"
        assert config.transcription.max_duration_seconds == 7200
        assert config.transcription.gemini_api_key == "test_key_123"

    def test_config_update_from_env_invalid_values(self, monkeypatch):
        """Test configuration update with invalid environment values."""
        monkeypatch.setenv("VTTIRO_MAX_DURATION", "invalid")

        config = VttiroConfig()

        with pytest.raises(ValidationError, match="Invalid environment variable"):
            config.update_from_env()

    def test_config_load_from_file(self, temp_dir):
        """Test loading configuration from file."""
        config_file = temp_dir / "test_config.yaml"
        config_content = """
transcription:
  preferred_model: "assemblyai"
  language: "fr"
  max_duration_seconds: 1800
  assemblyai_api_key: "test_assemblyai_key"

processing:
  max_workers: 8
  memory_limit_mb: 4096

output:
  format: "srt"
  include_metadata: false
"""
        config_file.write_text(config_content)

        config = VttiroConfig.load_from_file(config_file)

        assert config.transcription.preferred_model == "assemblyai"
        assert config.transcription.language == "fr"
        assert config.transcription.max_duration_seconds == 1800
        assert config.processing.max_workers == 8
        assert config.output.format == "srt"

    def test_config_load_from_file_not_found(self):
        """Test loading configuration from non-existent file."""
        with pytest.raises(ValidationError, match="Configuration file not found"):
            VttiroConfig.load_from_file("non_existent.yaml")

    def test_config_load_from_file_invalid_yaml(self, temp_dir):
        """Test loading configuration from invalid YAML file."""
        config_file = temp_dir / "invalid_config.yaml"
        config_file.write_text("invalid: yaml: content: [")

        with pytest.raises(ValidationError, match="Failed to parse configuration"):
            VttiroConfig.load_from_file(config_file)

    def test_config_save_to_file(self, temp_dir):
        """Test saving configuration to file."""
        config = VttiroConfig()
        config.transcription.preferred_model = "deepgram"
        config.transcription.language = "de"

        config_file = temp_dir / "saved_config.yaml"
        config.save_to_file(config_file)

        assert config_file.exists()

        # Load back and verify
        loaded_config = VttiroConfig.load_from_file(config_file)
        assert loaded_config.transcription.preferred_model == "deepgram"
        assert loaded_config.transcription.language == "de"

    def test_config_load_default_locations(self, temp_dir, monkeypatch):
        """Test loading configuration from default locations."""
        # Mock home directory
        monkeypatch.setenv("HOME", str(temp_dir))

        # Create config file in default location
        config_dir = temp_dir / ".config" / "vttiro"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "config.yaml"

        config_content = """
transcription:
  preferred_model: "auto"
  language: "en"
"""
        config_file.write_text(config_content)

        with patch("vttiro.core.config.VttiroConfig._get_default_config_paths") as mock_paths:
            mock_paths.return_value = [config_file]
            config = VttiroConfig.load_default()

            assert config.transcription.preferred_model == "auto"
            assert config.transcription.language == "en"

    def test_config_api_key_validation(self):
        """Test API key validation."""
        config = VttiroConfig()

        # Test with no API keys - should use mock engine
        assert not config.has_valid_api_keys()

        # Test with valid API key
        config.transcription.gemini_api_key = "valid_key_123"
        assert config.has_valid_api_keys()

    def test_config_to_dict(self):
        """Test configuration serialization to dictionary."""
        config = VttiroConfig()
        config.transcription.preferred_model = "gemini"
        config.transcription.gemini_api_key = "secret_key"

        config_dict = config.to_dict()

        assert config_dict["transcription"]["preferred_model"] == "gemini"
        # API keys should be masked in serialization
        assert config_dict["transcription"]["gemini_api_key"] == "***masked***"

    def test_config_from_dict(self):
        """Test configuration creation from dictionary."""
        config_dict = {
            "transcription": {"preferred_model": "assemblyai", "language": "es", "max_duration_seconds": 1800},
            "processing": {"max_workers": 6},
        }

        config = VttiroConfig.from_dict(config_dict)

        assert config.transcription.preferred_model == "assemblyai"
        assert config.transcription.language == "es"
        assert config.transcription.max_duration_seconds == 1800
        assert config.processing.max_workers == 6

    def test_config_merge(self):
        """Test configuration merging."""
        base_config = VttiroConfig()
        base_config.transcription.preferred_model = "gemini"
        base_config.transcription.language = "en"

        override_config = VttiroConfig()
        override_config.transcription.language = "fr"  # Override language
        override_config.processing.max_workers = 8  # Add new setting

        merged = base_config.merge(override_config)

        assert merged.transcription.preferred_model == "gemini"  # From base
        assert merged.transcription.language == "fr"  # Overridden
        assert merged.processing.max_workers == 8  # From override


class TestTranscriptionResult:
    """Test TranscriptionResult data class."""

    def test_transcription_result_creation(self):
        """Test TranscriptionResult creation with required fields."""
        result = TranscriptionResult(text="Test transcription", confidence=0.95, language="en")

        assert result.text == "Test transcription"
        assert result.confidence == 0.95
        assert result.language == "en"
        assert result.start_time == 0.0
        assert result.end_time == 0.0
        assert result.word_timestamps == []
        assert result.speaker_labels == []
        assert result.emotions == []
        assert result.metadata == {}

    def test_transcription_result_with_all_fields(self):
        """Test TranscriptionResult creation with all fields."""
        word_timestamps = [{"word": "Hello", "start": 0.0, "end": 0.5}, {"word": "world", "start": 0.6, "end": 1.0}]

        speaker_labels = [{"speaker": "Speaker 1", "start": 0.0, "end": 1.0}]

        emotions = [{"emotion": "neutral", "confidence": 0.8, "start": 0.0, "end": 1.0}]

        metadata = {"model": "test_model", "processing_time": 1.5, "correlation_id": "test-123"}

        result = TranscriptionResult(
            text="Hello world",
            confidence=0.92,
            language="en",
            start_time=0.0,
            end_time=1.0,
            word_timestamps=word_timestamps,
            speaker_labels=speaker_labels,
            emotions=emotions,
            metadata=metadata,
        )

        assert result.text == "Hello world"
        assert result.confidence == 0.92
        assert result.language == "en"
        assert result.start_time == 0.0
        assert result.end_time == 1.0
        assert result.word_timestamps == word_timestamps
        assert result.speaker_labels == speaker_labels
        assert result.emotions == emotions
        assert result.metadata == metadata

    def test_transcription_result_validation(self):
        """Test TranscriptionResult validation."""
        # Test invalid confidence
        with pytest.raises(ValidationError, match="Confidence must be between 0 and 1"):
            TranscriptionResult(
                text="Test",
                confidence=1.5,  # Invalid
                language="en",
            )

        # Test invalid time range
        with pytest.raises(ValidationError, match="start_time cannot be greater than end_time"):
            TranscriptionResult(
                text="Test",
                confidence=0.9,
                language="en",
                start_time=5.0,
                end_time=3.0,  # Invalid
            )

    def test_transcription_result_to_dict(self):
        """Test TranscriptionResult serialization."""
        result = TranscriptionResult(
            text="Test transcription", confidence=0.95, language="en", metadata={"model": "test"}
        )

        result_dict = result.to_dict()

        assert result_dict["text"] == "Test transcription"
        assert result_dict["confidence"] == 0.95
        assert result_dict["language"] == "en"
        assert result_dict["metadata"]["model"] == "test"

    def test_transcription_result_from_dict(self):
        """Test TranscriptionResult creation from dictionary."""
        result_dict = {
            "text": "Test transcription",
            "confidence": 0.88,
            "language": "es",
            "start_time": 1.0,
            "end_time": 6.0,
            "metadata": {"source": "test"},
        }

        result = TranscriptionResult.from_dict(result_dict)

        assert result.text == "Test transcription"
        assert result.confidence == 0.88
        assert result.language == "es"
        assert result.start_time == 1.0
        assert result.end_time == 6.0
        assert result.metadata["source"] == "test"

    def test_transcription_result_duration_property(self):
        """Test TranscriptionResult duration property."""
        result = TranscriptionResult(text="Test", confidence=0.9, language="en", start_time=2.5, end_time=7.8)

        assert result.duration == 5.3

    def test_transcription_result_word_count_property(self):
        """Test TranscriptionResult word count property."""
        result = TranscriptionResult(
            text="This is a test transcription with seven words", confidence=0.9, language="en"
        )

        assert result.word_count == 8


class TestConfigurationIntegration:
    """Test configuration integration with other components."""

    def test_config_environment_precedence(self, monkeypatch, temp_dir):
        """Test that environment variables take precedence over file config."""
        # Create config file
        config_file = temp_dir / "config.yaml"
        config_content = """
transcription:
  preferred_model: "assemblyai"
  language: "en"
"""
        config_file.write_text(config_content)

        # Set environment variable
        monkeypatch.setenv("VTTIRO_PREFERRED_MODEL", "gemini")

        config = VttiroConfig.load_from_file(config_file)
        config.update_from_env()

        # Environment should override file
        assert config.transcription.preferred_model == "gemini"
        assert config.transcription.language == "en"  # From file

    def test_config_with_multiple_api_keys(self):
        """Test configuration with multiple API keys."""
        config = VttiroConfig()
        config.transcription.gemini_api_key = "gemini_key"
        config.transcription.assemblyai_api_key = "assemblyai_key"
        config.transcription.deepgram_api_key = "deepgram_key"

        assert config.has_valid_api_keys()
        assert config.get_available_models() == ["gemini", "assemblyai", "deepgram"]

    def test_config_security_key_masking(self):
        """Test that API keys are properly masked in logs/serialization."""
        config = VttiroConfig()
        config.transcription.gemini_api_key = "very_secret_key_123456"

        # Test string representation doesn't expose keys
        config_str = str(config)
        assert "very_secret_key_123456" not in config_str
        assert "***masked***" in config_str

        # Test dict serialization masks keys
        config_dict = config.to_dict()
        assert config_dict["transcription"]["gemini_api_key"] == "***masked***"
