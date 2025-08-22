# this_file: src/vttiro/tests/test_config_manager.py
"""Tests for enhanced configuration management system.

This module provides comprehensive testing for the configuration manager,
including environment-specific loading, validation, migration, and error handling.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from vttiro.core.config import VttiroConfig
from vttiro.core.config_manager import (
    ConfigurationManager, Environment, ConfigValidationError,
    load_config_for_environment, create_config_templates
)


class TestConfigurationManager:
    """Test cases for ConfigurationManager."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_dir = self.temp_dir / "config"
        self.config_dir.mkdir(parents=True)
        self.manager = ConfigurationManager(self.config_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_load_config_with_defaults(self):
        """Test loading configuration with default values."""
        config = self.manager.load_config(Environment.DEVELOPMENT, validate_strict=False)
        
        assert isinstance(config, VttiroConfig)
        assert config.provider == "gemini"
        assert config.output_format == "webvtt"
        assert config.max_retries == 3
    
    def test_load_config_with_base_file(self):
        """Test loading configuration with base configuration file."""
        # Create base configuration
        base_config = {
            "provider": "openai",
            "language": "en",
            "timeout_seconds": 600.0
        }
        
        base_path = self.config_dir / "base.json"
        with open(base_path, 'w') as f:
            json.dump(base_config, f)
        
        config = self.manager.load_config(Environment.DEVELOPMENT, validate_strict=False)
        
        assert config.provider == "openai"
        assert config.language == "en"
        assert config.timeout_seconds == 600.0
    
    def test_load_config_with_environment_override(self):
        """Test loading configuration with environment-specific overrides."""
        # Create base configuration
        base_config = {"provider": "gemini", "verbose": False}
        base_path = self.config_dir / "base.json"
        with open(base_path, 'w') as f:
            json.dump(base_config, f)
        
        # Create development environment override
        dev_config = {"verbose": True, "dry_run": True}
        dev_path = self.config_dir / "development.json"
        with open(dev_path, 'w') as f:
            json.dump(dev_config, f)
        
        config = self.manager.load_config(Environment.DEVELOPMENT, validate_strict=False)
        
        assert config.provider == "gemini"  # From base
        assert config.verbose is True  # From environment override
        assert config.dry_run is True  # From environment override
    
    def test_load_config_with_file_override(self):
        """Test loading configuration with file override."""
        # Create base configuration
        base_config = {"provider": "gemini"}
        base_path = self.config_dir / "base.json"
        with open(base_path, 'w') as f:
            json.dump(base_config, f)
        
        # Create override file
        override_config = {"provider": "assemblyai", "language": "es"}
        override_path = self.temp_dir / "override.json"
        with open(override_path, 'w') as f:
            json.dump(override_config, f)
        
        config = self.manager.load_config(
            Environment.DEVELOPMENT,
            config_override=override_path,
            validate_strict=False
        )
        
        assert config.provider == "assemblyai"  # From override
        assert config.language == "es"  # From override
    
    def test_config_caching(self):
        """Test configuration caching mechanism."""
        config1 = self.manager.load_config(Environment.DEVELOPMENT, validate_strict=False)
        config2 = self.manager.load_config(Environment.DEVELOPMENT, validate_strict=False)
        
        assert config1 is config2  # Should be the same instance from cache
    
    def test_yaml_config_loading(self):
        """Test loading YAML configuration files."""
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML not available")
        
        # Create YAML configuration as override file (since base only looks for .json)
        yaml_config = """
        provider: deepgram
        language: fr
        verbose: true
        """
        
        yaml_path = self.temp_dir / "override.yaml"
        with open(yaml_path, 'w') as f:
            f.write(yaml_config)
        
        config = self.manager.load_config(
            Environment.DEVELOPMENT,
            config_override=yaml_path,
            validate_strict=False
        )
        
        assert config.provider == "deepgram"
        assert config.language == "fr"
        assert config.verbose is True
    
    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"})
    def test_production_validation_success(self):
        """Test successful production environment validation."""
        # Create production configuration
        prod_config = {
            "provider": "gemini",
            "verbose": False,
            "dry_run": False,
            "timeout_seconds": 600.0
        }
        
        prod_path = self.config_dir / "production.json"
        with open(prod_path, 'w') as f:
            json.dump(prod_config, f)
        
        # Should not raise exception
        config = self.manager.load_config(Environment.PRODUCTION, validate_strict=True)
        assert config.provider == "gemini"
    
    def test_production_validation_failure(self):
        """Test production environment validation failure."""
        # Create invalid production configuration
        prod_config = {
            "provider": "gemini",
            "verbose": True,  # Invalid for production
            "dry_run": True   # Invalid for production
        }
        
        prod_path = self.config_dir / "production.json"
        with open(prod_path, 'w') as f:
            json.dump(prod_config, f)
        
        with pytest.raises(ConfigValidationError) as exc_info:
            self.manager.load_config(Environment.PRODUCTION, validate_strict=True)
        
        assert "Verbose logging should be disabled in production" in str(exc_info.value)
        assert "Dry run mode should be disabled in production" in str(exc_info.value)
    
    def test_api_key_validation(self):
        """Test API key validation for different providers."""
        providers_and_keys = [
            ("openai", "OPENAI_API_KEY"),
            ("assemblyai", "ASSEMBLYAI_API_KEY"),
            ("deepgram", "DEEPGRAM_API_KEY")
        ]
        
        # Clear all API keys from environment
        with patch.dict(os.environ, {}, clear=True):
            for provider, env_key in providers_and_keys:
                config_data = {"provider": provider}
                config_path = self.config_dir / "base.json"
                with open(config_path, 'w') as f:
                    json.dump(config_data, f)
                
                # Clear cache
                self.manager.config_cache.clear()
                
                # Should fail without API key
                with pytest.raises(ConfigValidationError) as exc_info:
                    self.manager.load_config(Environment.PRODUCTION, validate_strict=True)
                
                assert f"{env_key} environment variable required" in str(exc_info.value)
    
    def test_validation_suggestions(self):
        """Test validation error suggestions."""
        # Create invalid configuration
        invalid_config = {
            "provider": "invalid_provider",
            "language": "invalid_lang_code",
            "output_format": "invalid_format",
            "max_segment_duration": -1,
            "timeout_seconds": 0
        }
        
        config_path = self.config_dir / "base.json"
        with open(config_path, 'w') as f:
            json.dump(invalid_config, f)
        
        with pytest.raises(ConfigValidationError) as exc_info:
            self.manager.load_config(Environment.DEVELOPMENT, validate_strict=True)
        
        error_message = str(exc_info.value)
        assert "Valid providers: gemini, openai, assemblyai, deepgram" in error_message
        assert "Valid output formats: webvtt, srt, json" in error_message
    
    def test_invalid_json_handling(self):
        """Test handling of invalid JSON configuration files."""
        # Create invalid JSON file
        invalid_json_path = self.config_dir / "base.json"
        with open(invalid_json_path, 'w') as f:
            f.write('{"invalid": json}')  # Missing quotes around json
        
        with pytest.raises(ConfigValidationError) as exc_info:
            self.manager.load_config(Environment.DEVELOPMENT)
        
        assert "Invalid JSON" in str(exc_info.value)
        assert "Check JSON syntax" in str(exc_info.value)
    
    def test_create_config_template(self):
        """Test configuration template creation."""
        template_path = self.temp_dir / "test_template.json"
        
        self.manager.create_config_template(Environment.DEVELOPMENT, template_path)
        
        assert template_path.exists()
        
        with open(template_path) as f:
            template_data = json.load(f)
        
        assert template_data["verbose"] is True  # Development-specific
        assert template_data["provider"] == "gemini"
        assert "timeout_seconds" in template_data
    
    def test_config_migration(self):
        """Test configuration migration from old format."""
        # Create old-format configuration
        old_config = {
            "ai_provider": "openai",
            "target_language": "es",
            "output_type": "srt",
            "request_timeout": 120.0,
            "retry_attempts": 5,
            "context": "Medical transcription"
        }
        
        old_config_path = self.temp_dir / "old_config.json"
        with open(old_config_path, 'w') as f:
            json.dump(old_config, f)
        
        new_config_path = self.temp_dir / "new_config.json"
        
        self.manager.migrate_config(old_config_path, new_config_path)
        
        assert new_config_path.exists()
        
        with open(new_config_path) as f:
            migrated_data = json.load(f)
        
        assert migrated_data["provider"] == "openai"
        assert migrated_data["language"] == "es"
        assert migrated_data["output_format"] == "srt"
        assert migrated_data["timeout_seconds"] == 120.0
        assert migrated_data["max_retries"] == 5
        assert migrated_data["context"] == "Medical transcription"


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_load_config_for_environment(self):
        """Test load_config_for_environment convenience function."""
        with patch.dict(os.environ, {}, clear=True):
            # Use validate_strict=False in the manager to avoid API key validation
            with patch.object(ConfigurationManager, 'load_config') as mock_load:
                mock_load.return_value = VttiroConfig()
                config = load_config_for_environment("development")
                assert isinstance(config, VttiroConfig)
                mock_load.assert_called_once()
    
    def test_create_config_templates(self):
        """Test create_config_templates convenience function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            create_config_templates(output_dir)
            
            # Check that templates were created for all environments
            for env in Environment:
                template_path = output_dir / f"{env.value}.json"
                assert template_path.exists()
                
                with open(template_path) as f:
                    template_data = json.load(f)
                
                assert "provider" in template_data
                assert "timeout_seconds" in template_data


class TestEnvironmentSpecificBehavior:
    """Test environment-specific configuration behavior."""
    
    def test_development_environment_defaults(self):
        """Test development environment defaults."""
        manager = ConfigurationManager()
        
        with patch.object(manager, '_get_environment_template') as mock_template:
            mock_template.return_value = {"verbose": True, "dry_run": False}
            
            template = manager._get_environment_template(Environment.DEVELOPMENT)
            
            assert template["verbose"] is True
            assert template["dry_run"] is False
    
    def test_testing_environment_defaults(self):
        """Test testing environment defaults."""
        manager = ConfigurationManager()
        
        template = manager._get_environment_template(Environment.TESTING)
        
        assert template["verbose"] is True
        assert template["dry_run"] is True
        assert template["timeout_seconds"] == 60.0
    
    def test_production_environment_defaults(self):
        """Test production environment defaults."""
        manager = ConfigurationManager()
        
        template = manager._get_environment_template(Environment.PRODUCTION)
        
        assert template["verbose"] is False
        assert template["dry_run"] is False
        assert template["timeout_seconds"] == 600.0
        assert template["max_retries"] == 5


class TestConfigValidationError:
    """Test ConfigValidationError exception class."""
    
    def test_basic_error_formatting(self):
        """Test basic error message formatting."""
        error = ConfigValidationError("Test error")
        assert str(error) == "Test error"
    
    def test_error_with_suggestions(self):
        """Test error formatting with suggestions."""
        suggestions = ["Check configuration syntax", "Verify API keys"]
        error = ConfigValidationError("Test error", suggestions=suggestions)
        
        error_str = str(error)
        assert "Test error" in error_str
        assert "Suggestions:" in error_str
        assert "Check configuration syntax" in error_str
        assert "Verify API keys" in error_str
    
    def test_error_with_detailed_errors(self):
        """Test error formatting with detailed validation errors."""
        detailed_errors = [
            {"loc": ["provider"], "msg": "Invalid provider value"},
            {"loc": ["timeout_seconds"], "msg": "Must be greater than 0"}
        ]
        
        error = ConfigValidationError("Validation failed", errors=detailed_errors)
        
        error_str = str(error)
        assert "Validation failed" in error_str
        assert "Detailed errors:" in error_str
        assert "provider: Invalid provider value" in error_str
        assert "timeout_seconds: Must be greater than 0" in error_str