# this_file: src/vttiro/tests/test_config.py
"""Unit tests for configuration management.

Tests for VttiroConfig Pydantic model, including validation,
defaults, and configuration loading.
"""

import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from ..core.config import VttiroConfig


class TestVttiroConfig:
    """Test VttiroConfig Pydantic model."""
    
    def test_default_configuration(self):
        """Test creating config with default values."""
        config = VttiroConfig()
        
        assert config.provider == "gemini"
        assert config.language is None
        assert config.context is None
        assert config.output_format == "webvtt"
        assert config.output_path is None
        assert config.enable_speaker_diarization is False
        assert config.enable_emotion_detection is False
        assert config.audio_preprocessing is True
        assert config.max_segment_duration == 30.0
        assert config.gemini_model == "gemini-2.0-flash"
        assert config.openai_model == "whisper-1"
        assert config.timeout_seconds == 300.0
        assert config.max_retries == 3
        assert config.verbose is False
        assert config.dry_run is False
    
    def test_custom_configuration(self):
        """Test creating config with custom values."""
        config = VttiroConfig(
            provider="openai",
            language="es",
            context="Technical presentation",
            output_format="srt",
            enable_speaker_diarization=True,
            max_segment_duration=45.0,
            verbose=True
        )
        
        assert config.provider == "openai"
        assert config.language == "es"
        assert config.context == "Technical presentation"
        assert config.output_format == "srt"
        assert config.enable_speaker_diarization is True
        assert config.max_segment_duration == 45.0
        assert config.verbose is True
    
    def test_invalid_provider_raises_error(self):
        """Test that invalid provider raises validation error."""
        with pytest.raises(ValidationError, match="Input should be"):
            VttiroConfig(provider="invalid_provider")
    
    def test_invalid_output_format_raises_error(self):
        """Test that invalid output format raises validation error.""" 
        with pytest.raises(ValidationError, match="Input should be"):
            VttiroConfig(output_format="invalid_format")
    
    def test_invalid_max_segment_duration_raises_error(self):
        """Test that invalid max_segment_duration raises validation error."""
        # Too small
        with pytest.raises(ValidationError, match="Input should be greater than 0"):
            VttiroConfig(max_segment_duration=0.0)
        
        # Too large
        with pytest.raises(ValidationError, match="Input should be less than or equal to 120"):
            VttiroConfig(max_segment_duration=150.0)
    
    def test_invalid_timeout_raises_error(self):
        """Test that invalid timeout raises validation error."""
        with pytest.raises(ValidationError, match="Input should be greater than 0"):
            VttiroConfig(timeout_seconds=-10.0)
    
    def test_invalid_max_retries_raises_error(self):
        """Test that invalid max_retries raises validation error."""
        # Too small
        with pytest.raises(ValidationError, match="Input should be greater than or equal to 0"):
            VttiroConfig(max_retries=-1)
        
        # Too large
        with pytest.raises(ValidationError, match="Input should be less than or equal to 10"):
            VttiroConfig(max_retries=15)
    
    def test_language_code_validation(self):
        """Test language code validation."""
        # Valid 2-letter codes
        config = VttiroConfig(language="en")
        assert config.language == "en"
        
        config = VttiroConfig(language="es")
        assert config.language == "es"
        
        # Valid 5-letter locale codes
        config = VttiroConfig(language="en-US")
        assert config.language == "en-US"
        
        # Invalid length
        with pytest.raises(ValidationError, match="Invalid language code format"):
            VttiroConfig(language="english")
    
    def test_output_path_validation_and_creation(self):
        """Test output path validation and directory creation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Valid path in existing directory
            output_path = Path(tmp_dir) / "output.vtt"
            config = VttiroConfig(output_path=output_path)
            assert config.output_path == output_path
            
            # Path with non-existent parent directory (should be created)
            nested_path = Path(tmp_dir) / "nested" / "dir" / "output.vtt"
            config = VttiroConfig(output_path=nested_path)
            assert config.output_path == nested_path
            assert nested_path.parent.exists()
    
    def test_get_provider_config(self):
        """Test provider-specific configuration extraction."""
        # Gemini config
        config = VttiroConfig(provider="gemini", gemini_model="custom-model")
        provider_config = config.get_provider_config()
        assert provider_config == {"model": "custom-model"}
        
        # OpenAI config
        config = VttiroConfig(provider="openai", openai_model="whisper-large")
        provider_config = config.get_provider_config()
        assert provider_config == {"model": "whisper-large"}
        
        # Provider without specific config
        config = VttiroConfig(provider="assemblyai")
        provider_config = config.get_provider_config()
        assert provider_config == {}
    
    def test_to_dict_conversion(self):
        """Test conversion to dictionary."""
        config = VttiroConfig(
            provider="openai",
            language="en",
            verbose=True
        )
        
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["provider"] == "openai"
        assert config_dict["language"] == "en"
        assert config_dict["verbose"] is True
        assert "output_path" in config_dict  # Should be None but present
    
    def test_to_dict_with_path(self):
        """Test dictionary conversion with Path object."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "output.vtt"
            config = VttiroConfig(output_path=output_path)
            
            config_dict = config.to_dict()
            
            # Path should be converted to string
            assert isinstance(config_dict["output_path"], str)
            assert config_dict["output_path"] == str(output_path)
    
    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            VttiroConfig(unknown_field="value")
    
    def test_from_file_json(self):
        """Test loading configuration from JSON file."""
        config_data = {
            "provider": "openai",
            "language": "fr", 
            "verbose": True,
            "max_segment_duration": 45.0
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            import json
            json.dump(config_data, f)
            config_path = Path(f.name)
        
        try:
            config = VttiroConfig.from_file(config_path)
            
            assert config.provider == "openai"
            assert config.language == "fr"
            assert config.verbose is True
            assert config.max_segment_duration == 45.0
            # Other fields should have defaults
            assert config.output_format == "webvtt"
        finally:
            config_path.unlink()
    
    def test_from_file_not_found(self):
        """Test loading from non-existent file raises error."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            VttiroConfig.from_file(Path("/nonexistent/config.json"))
    
    def test_assignment_validation(self):
        """Test that assignment validation works."""
        config = VttiroConfig()
        
        # Valid assignment
        config.provider = "openai"
        assert config.provider == "openai"
        
        # Invalid assignment should raise error
        with pytest.raises(ValidationError):
            config.provider = "invalid"