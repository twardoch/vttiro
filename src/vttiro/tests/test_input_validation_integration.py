# this_file: src/vttiro/tests/test_input_validation_integration.py
"""Tests for input validation integration across the codebase.

This module tests that the comprehensive input validation system
is properly integrated into the main components:
- CLI interface validation
- Core transcriber validation  
- Provider base class validation
- End-to-end validation flow

Used by:
- CI/CD pipeline for validation coverage
- Integration testing of validation flow
- Regression testing for validation changes
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import os

from ..cli import VttiroCLI
from ..core.transcriber import Transcriber
from ..core.config import VttiroConfig
from ..providers.base import TranscriberABC
from ..utils.input_validation import InputValidator, ProviderInputSanitizer


class TestInputValidationIntegration:
    """Test input validation integration across components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.valid_audio_file = Path(self.temp_dir) / "test.mp3"
        self.valid_audio_file.write_bytes(b"fake mp3 content")
        
        self.invalid_file = Path(self.temp_dir) / "test.invalid"
        self.invalid_file.write_bytes(b"invalid content")
        
        self.nonexistent_file = Path(self.temp_dir) / "nonexistent.mp3"
        
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_cli_validation_integration(self):
        """Test that CLI properly integrates input validation."""
        cli = VttiroCLI()
        
        # Test valid input
        with patch('builtins.print'):  # Suppress console output
            # Should not raise exception for valid inputs
            try:
                cli.transcribe(
                    input_path=str(self.valid_audio_file),
                    provider="gemini",
                    language="en",
                    dry_run=True
                )
            except Exception:
                pytest.fail("CLI validation failed for valid inputs")
    
    def test_cli_validation_catches_invalid_file(self):
        """Test that CLI validation catches invalid files."""
        cli = VttiroCLI()
        
        with patch('builtins.print'):  # Suppress console output
            # Test nonexistent file
            cli.transcribe(
                input_path=str(self.nonexistent_file),
                provider="gemini"
            )
            # Should handle gracefully without crashing
            
            # Test invalid format
            cli.transcribe(
                input_path=str(self.invalid_file),
                provider="gemini"
            )
            # Should handle gracefully without crashing
    
    def test_cli_validation_catches_invalid_provider(self):
        """Test that CLI validation catches invalid providers."""
        cli = VttiroCLI()
        
        with patch('builtins.print'):  # Suppress console output
            cli.transcribe(
                input_path=str(self.valid_audio_file),
                provider="invalid_provider"
            )
            # Should handle gracefully without crashing
    
    def test_transcriber_validation_integration(self):
        """Test that core Transcriber integrates validation."""
        config = VttiroConfig(provider="gemini", language="en")
        transcriber = Transcriber(config)
        
        # Test config validation
        validation_result = transcriber.validate_config()
        assert validation_result["valid"] is True
        assert len(validation_result["issues"]) == 0
    
    def test_transcriber_validation_catches_issues(self):
        """Test that Transcriber validation catches configuration issues."""
        config = VttiroConfig(provider="invalid_provider", language="invalid")
        transcriber = Transcriber(config)
        
        # Test config validation
        validation_result = transcriber.validate_config()
        assert validation_result["valid"] is False
        assert len(validation_result["issues"]) > 0
        
        # Check specific issues
        issues_text = " ".join(validation_result["issues"])
        assert "provider" in issues_text.lower()
        assert "language" in issues_text.lower()
    
    def test_provider_base_validation_integration(self):
        """Test that provider base class integrates validation."""
        
        class MockProvider(TranscriberABC):
            @property
            def provider_name(self) -> str:
                return "mock"
            
            async def transcribe(self, audio_path, language=None, context=None, **kwargs):
                return Mock()
            
            def estimate_cost(self, duration_seconds: float) -> float:
                return 0.0
        
        provider = MockProvider()
        
        # Test valid file validation
        provider.validate_audio_file(self.valid_audio_file)  # Should not raise
        
        # Test invalid file validation
        with pytest.raises((FileNotFoundError, ValueError)):
            provider.validate_audio_file(self.nonexistent_file)
        
        with pytest.raises(ValueError):
            provider.validate_audio_file(self.invalid_file)
    
    def test_provider_size_limit_validation(self):
        """Test that provider size limits are enforced."""
        
        class MockProvider(TranscriberABC):
            @property
            def provider_name(self) -> str:
                return "openai"  # Has 25MB limit
            
            async def transcribe(self, audio_path, language=None, context=None, **kwargs):
                return Mock()
            
            def estimate_cost(self, duration_seconds: float) -> float:
                return 0.0
        
        provider = MockProvider()
        
        # Create a large file that exceeds OpenAI's 25MB limit
        large_file = Path(self.temp_dir) / "large.mp3"
        large_file.write_bytes(b"x" * (26 * 1024 * 1024))  # 26MB
        
        # Should raise validation error for oversized file
        with pytest.raises(ValueError, match="exceeds maximum"):
            provider.validate_audio_file(large_file)
    
    def test_end_to_end_validation_flow(self):
        """Test complete validation flow from CLI to provider."""
        
        # Mock a provider to avoid actual API calls
        with patch('vttiro.core.transcriber.Transcriber.transcribe') as mock_transcribe:
            mock_transcribe.return_value = Mock()
            
            cli = VttiroCLI()
            
            # Test successful validation flow
            with patch('builtins.print'):  # Suppress console output
                cli.transcribe(
                    input_path=str(self.valid_audio_file),
                    provider="gemini",
                    language="en",
                    dry_run=True,
                    context="test context"
                )
                # Should complete without exceptions
    
    def test_kwargs_sanitization(self):
        """Test that provider-specific kwargs are sanitized."""
        validator = InputValidator()
        provider_sanitizer = ProviderInputSanitizer(validator)
        
        # Test Gemini kwargs sanitization
        gemini_inputs = {
            "model": "gemini-2.0-flash",
            "temperature": 0.7,
            "invalid_param": "should_be_removed",
            "max_tokens": 1000
        }
        is_valid, sanitized, warnings = provider_sanitizer.sanitize_for_provider("gemini", gemini_inputs)
        
        assert is_valid
        assert "model" in sanitized
        assert "temperature" in sanitized
        
        # Test OpenAI kwargs sanitization
        openai_inputs = {
            "model": "whisper-1",
            "response_format": "json",
            "temperature": 0.5,
            "invalid_param": "should_be_removed"
        }
        is_valid, sanitized, warnings = provider_sanitizer.sanitize_for_provider("openai", openai_inputs)
        
        assert is_valid
        assert "model" in sanitized
        assert "response_format" in sanitized
    
    def test_language_code_validation_integration(self):
        """Test language code validation across components."""
        validator = InputValidator()
        
        # Test valid language codes
        for lang in ["en", "es", "fr", "de", "ja", "zh"]:
            result = validator.validate_language_code(lang)
            assert result.is_valid
        
        # Test invalid language codes
        for invalid_lang in ["english", "esp", "invalid", "en-US", ""]:
            result = validator.validate_language_code(invalid_lang)
            assert not result.is_valid
            assert "language code" in result.error_message.lower()
    
    def test_numeric_parameter_validation(self):
        """Test numeric parameter validation in configuration."""
        validator = InputValidator()
        
        # Test valid numeric parameters
        result = validator.validate_numeric_parameter(1024, min_value=0)
        assert result.is_valid
        
        result = validator.validate_numeric_parameter(0.7, min_value=0.0, max_value=1.0)
        assert result.is_valid
        
        # Test invalid numeric parameters
        result = validator.validate_numeric_parameter("invalid", min_value=0)
        assert not result.is_valid
        assert "number" in result.error_message.lower()
        
        result = validator.validate_numeric_parameter(-1, min_value=0)
        assert not result.is_valid
        
        result = validator.validate_numeric_parameter(1.5, min_value=0.0, max_value=1.0)
        assert not result.is_valid
    
    def test_output_path_validation_integration(self):
        """Test output path validation across components."""
        validator = InputValidator()
        
        # Test valid output path
        valid_output = Path(self.temp_dir) / "output.vtt"
        result = validator.validate_output_path(valid_output)
        assert result.is_valid
        
        # Test output path in non-writable directory
        if os.name != 'nt':  # Skip on Windows due to permission complexity
            readonly_dir = Path(self.temp_dir) / "readonly"
            readonly_dir.mkdir()
            readonly_dir.chmod(0o444)  # Read-only
            
            readonly_output = readonly_dir / "output.vtt"
            result = validator.validate_output_path(readonly_output)
            # Should either be invalid or raise exception during validation
            
            # Cleanup
            readonly_dir.chmod(0o755)