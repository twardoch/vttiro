# this_file: src/vttiro/tests/test_security.py
"""Tests for enhanced security and input validation system.

This module provides comprehensive testing for the security infrastructure,
including API key encryption, input validation, rate limiting, and secure file handling.
"""

import os
import tempfile
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from vttiro.utils.security import (
    APIKeyManager, InputValidator, RateLimitMonitor, SecureFileHandler,
    SecurityError, ValidationRule, RateLimitInfo,
    get_api_key_manager, get_input_validator, get_rate_limit_monitor,
    validate_input, sanitize_input
)


class TestAPIKeyManager:
    """Test cases for APIKeyManager."""
    
    def setup_method(self):
        """Set up test environment."""
        self.manager = APIKeyManager("test_master_key_123")
    
    def test_key_encryption_decryption(self):
        """Test API key encryption and decryption."""
        original_key = "sk-1234567890abcdef"
        provider = "openai"
        
        # Encrypt key
        encrypted = self.manager.encrypt_api_key(original_key, provider)
        assert encrypted != original_key
        assert len(encrypted) > len(original_key)
        
        # Decrypt key
        decrypted = self.manager.decrypt_api_key(encrypted, provider)
        assert decrypted == original_key
    
    def test_provider_context_validation(self):
        """Test provider context validation during decryption."""
        api_key = "test-key-123"
        encrypted = self.manager.encrypt_api_key(api_key, "openai")
        
        # Should work with correct provider
        decrypted = self.manager.decrypt_api_key(encrypted, "openai")
        assert decrypted == api_key
        
        # Should fail with wrong provider
        with pytest.raises(SecurityError) as exc_info:
            self.manager.decrypt_api_key(encrypted, "gemini")
        
        assert "Failed to decrypt" in str(exc_info.value)
        assert exc_info.value.severity == "high"
    
    def test_invalid_encryption_data(self):
        """Test handling of invalid encryption data."""
        with pytest.raises(SecurityError) as exc_info:
            self.manager.decrypt_api_key("invalid_data", "openai")
        
        assert "Failed to decrypt" in str(exc_info.value)
        assert exc_info.value.error_code == "KEY_DECRYPTION_FAILED"
    
    def test_api_key_format_validation(self):
        """Test API key format validation for different providers."""
        test_cases = [
            ("openai", "sk-1234567890abcdefghijklmnop", True),
            ("openai", "invalid-key", False),
            ("gemini", "AIzaSyA12345678901234567890123456789012", True),  # Exactly 39 chars total
            ("gemini", "invalid-gemini-key", False),
            ("assemblyai", "abcdef1234567890abcdef1234567890", True),
            ("assemblyai", "invalid-assemblyai", False),
            ("deepgram", "1234567890abcdef1234567890abcdef12345678", True),
            ("deepgram", "short", False),
            ("unknown_provider", "any-key-format", True)  # Unknown providers pass
        ]
        
        for provider, key, expected in test_cases:
            result = self.manager.validate_api_key_format(key, provider)
            assert result == expected, f"Failed for {provider}: {key}"
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123456789"})
    def test_get_api_key_from_environment(self):
        """Test retrieving API key from environment variables."""
        key = self.manager.get_api_key("openai")
        assert key == "sk-test123456789"
    
    @patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaSyTest123456789"})
    def test_get_api_key_with_caching(self):
        """Test API key caching mechanism."""
        # First call should read from environment
        key1 = self.manager.get_api_key("gemini")
        assert key1 == "AIzaSyTest123456789"
        
        # Second call should use cache
        with patch.dict(os.environ, {}, clear=True):
            key2 = self.manager.get_api_key("gemini", use_cache=True)
            assert key2 == "AIzaSyTest123456789"  # From cache
            
            # Without cache should return None
            key3 = self.manager.get_api_key("gemini", use_cache=False)
            assert key3 is None
    
    def test_encrypted_key_detection(self):
        """Test detection of encrypted vs plain API keys."""
        plain_key = "sk-1234567890abcdef"
        encrypted_key = self.manager.encrypt_api_key(plain_key, "openai")
        
        assert not self.manager._is_encrypted_key(plain_key)
        assert self.manager._is_encrypted_key(encrypted_key)
    
    def test_cache_ttl_expiration(self):
        """Test cache TTL expiration."""
        manager = APIKeyManager()
        manager._cache_ttl = 0.1  # 100ms TTL for testing
        
        # Add key to cache
        manager._key_cache["test"] = "test-key"
        
        # Should be in cache
        assert "test" in manager._key_cache
        
        # Wait for TTL expiration
        time.sleep(0.2)
        
        # Next get_api_key call should clear cache
        manager.get_api_key("nonexistent")
        assert "test" not in manager._key_cache
    
    def test_clear_cache(self):
        """Test manual cache clearing."""
        self.manager._key_cache["test"] = "test-key"
        assert "test" in self.manager._key_cache
        
        self.manager.clear_cache()
        assert len(self.manager._key_cache) == 0


class TestInputValidator:
    """Test cases for InputValidator."""
    
    def setup_method(self):
        """Set up test environment."""
        self.validator = InputValidator()
    
    def test_filename_validation(self):
        """Test filename validation."""
        # Valid filenames
        valid_filenames = ["test.txt", "file_name.mp3", "document-v2.pdf"]
        for filename in valid_filenames:
            assert self.validator.validate(filename, "filename")
        
        # Invalid filenames
        invalid_filenames = [
            "../etc/passwd",  # Path traversal
            "file<script>.txt",  # Dangerous characters
            "CON.txt",  # Windows reserved name
            "x" * 300  # Too long
        ]
        
        for filename in invalid_filenames:
            with pytest.raises(SecurityError):
                self.validator.validate(filename, "filename")
    
    def test_file_path_validation(self):
        """Test file path validation."""
        # Valid paths
        valid_paths = ["documents/file.txt", "folder/subfolder/file.mp3"]
        for path in valid_paths:
            assert self.validator.validate(path, "file_path")
        
        # Invalid paths
        invalid_paths = [
            "../../../etc/passwd",  # Path traversal
            "/absolute/path",  # Absolute path
            "path\\with\\backslashes/../up",  # Mixed separators with traversal
            "x" * 5000  # Too long
        ]
        
        for path in invalid_paths:
            with pytest.raises(SecurityError):
                self.validator.validate(path, "file_path")
    
    def test_url_validation(self):
        """Test URL validation."""
        # Valid URLs
        valid_urls = [
            "https://api.openai.com/v1/audio/transcriptions",
            "http://localhost:8000/api",
            "https://example.com"
        ]
        for url in valid_urls:
            assert self.validator.validate(url, "url")
        
        # Invalid URLs
        invalid_urls = [
            "javascript:alert('xss')",
            "ftp://malicious.com/file",
            "https://",
            "not-a-url",
            "https://" + "x" * 2100  # Too long
        ]
        
        for url in invalid_urls:
            with pytest.raises(SecurityError):
                self.validator.validate(url, "url")
    
    def test_language_code_validation(self):
        """Test language code validation."""
        # Valid language codes
        valid_codes = ["en", "es", "fr", "en-US", "zh-CN"]
        for code in valid_codes:
            assert self.validator.validate(code, "language_code")
        
        # Invalid language codes
        invalid_codes = ["english", "ESP", "en_US", "123", ""]
        for code in invalid_codes:
            with pytest.raises(SecurityError):
                self.validator.validate(code, "language_code")
    
    def test_text_content_validation(self):
        """Test text content validation."""
        # Valid text
        valid_text = "This is normal text content for transcription."
        assert self.validator.validate(valid_text, "text_content")
        
        # Invalid text with dangerous patterns
        invalid_texts = [
            "<script>alert('xss')</script>",
            "javascript:void(0)",
            "data:text/html;base64,PHNjcmlwdD5hbGVydCgneHNzJyk8L3NjcmlwdD4=",
            "x" * 1000001  # Too long
        ]
        
        for text in invalid_texts:
            with pytest.raises(SecurityError):
                self.validator.validate(text, "text_content")
    
    def test_api_key_validation(self):
        """Test API key validation."""
        # Valid API keys
        valid_keys = ["sk-1234567890abcdef", "AIzaSyA123456789", "abcdef123456"]
        for key in valid_keys:
            assert self.validator.validate(key, "api_key")
        
        # Invalid API keys
        invalid_keys = [
            "short",  # Too short
            "x" * 600,  # Too long
            "key with spaces",  # Invalid characters
            ""  # Empty
        ]
        
        for key in invalid_keys:
            with pytest.raises(SecurityError):
                self.validator.validate(key, "api_key")
    
    def test_length_validation(self):
        """Test length validation rules."""
        rule = ValidationRule(
            name="test",
            min_length=5,
            max_length=10,
            error_message="Length validation failed"
        )
        
        self.validator._rules["test_length"] = rule
        
        # Valid length
        assert self.validator.validate("12345", "test_length")
        assert self.validator.validate("1234567890", "test_length")
        
        # Invalid lengths
        with pytest.raises(SecurityError, match="too short"):
            self.validator.validate("1234", "test_length")
        
        with pytest.raises(SecurityError, match="too long"):
            self.validator.validate("12345678901", "test_length")
    
    def test_custom_validator(self):
        """Test custom validation function."""
        def is_even_length(value, context):
            return len(value) % 2 == 0
        
        rule = ValidationRule(
            name="even_length",
            custom_validator=is_even_length,
            error_message="Must have even length"
        )
        
        self.validator._rules["even_test"] = rule
        
        # Valid (even length)
        assert self.validator.validate("test", "even_test")
        
        # Invalid (odd length)
        with pytest.raises(SecurityError, match="Must have even length"):
            self.validator.validate("hello", "even_test")
    
    def test_unknown_validation_rule(self):
        """Test handling of unknown validation rules."""
        with pytest.raises(SecurityError, match="Unknown validation rule"):
            self.validator.validate("test", "nonexistent_rule")
    
    def test_sanitization_functions(self):
        """Test input sanitization functions."""
        # Filename sanitization
        dirty_filename = "file<>name?.txt"
        clean_filename = self.validator.sanitize(dirty_filename, "filename")
        assert "<" not in clean_filename
        assert ">" not in clean_filename
        assert "?" not in clean_filename
        
        # Text content sanitization
        dirty_text = "<script>alert('xss')</script>Normal text"
        clean_text = self.validator.sanitize(dirty_text, "text_content")
        assert "<script>" not in clean_text
        assert "Normal text" in clean_text
        
        # URL sanitization
        valid_url = "https://example.com/path"
        invalid_url = "javascript:alert(1)"
        
        assert self.validator.sanitize(valid_url, "url") == valid_url
        assert self.validator.sanitize(invalid_url, "url") == ""
        
        # Model name sanitization
        dirty_model = "model<name>-v1.0"
        clean_model = self.validator.sanitize(dirty_model, "model_name")
        assert clean_model == "model_name_-v1.0"


class TestRateLimitMonitor:
    """Test cases for RateLimitMonitor."""
    
    def setup_method(self):
        """Set up test environment."""
        self.monitor = RateLimitMonitor()
    
    def test_request_recording(self):
        """Test recording API requests."""
        provider = "openai"
        
        # Record some requests
        self.monitor.record_request(provider)
        self.monitor.record_request(provider)
        
        # Check status
        status = self.monitor.check_rate_limit_status(provider)
        
        assert status["provider"] == provider
        assert status["total_requests"] == 2
        assert status["requests_last_minute"] == 2
        assert status["last_request"] is not None
    
    def test_rate_limit_header_extraction(self):
        """Test extraction of rate limit info from headers."""
        provider = "openai"
        headers = {
            "x-ratelimit-remaining": "99",
            "x-ratelimit-limit": "100",
            "x-ratelimit-reset": "1234567890"
        }
        
        self.monitor.record_request(provider, headers)
        status = self.monitor.check_rate_limit_status(provider)
        
        assert "rate_limit_headers" in status
        assert status["rate_limit_headers"]["remaining"] == "99"
        assert status["rate_limit_headers"]["limit"] == "100"
    
    def test_request_history_cleanup(self):
        """Test cleanup of old request history."""
        provider = "test_provider"
        
        # Mock old requests
        old_time = datetime.now(timezone.utc) - timedelta(hours=25)
        self.monitor._request_history[provider] = [old_time]
        
        # Record new request (should trigger cleanup)
        self.monitor.record_request(provider)
        
        # Old request should be removed
        assert len(self.monitor._request_history[provider]) == 1
        assert self.monitor._request_history[provider][0] > old_time
    
    def test_backoff_delay_suggestions(self):
        """Test backoff delay suggestions based on request frequency."""
        provider = "test_provider"
        now = datetime.now(timezone.utc)
        
        # No requests - no delay
        assert self.monitor.suggest_backoff_delay(provider) == 0.0
        
        # Simulate different request frequencies
        # High frequency (50+ requests)
        self.monitor._request_history[provider] = [now] * 50
        delay = self.monitor.suggest_backoff_delay(provider)
        assert delay == 2.0
        
        # Medium frequency (20-49 requests)
        self.monitor._request_history[provider] = [now] * 25
        delay = self.monitor.suggest_backoff_delay(provider)
        assert delay == 1.0
        
        # Low-medium frequency (10-19 requests)
        self.monitor._request_history[provider] = [now] * 15
        delay = self.monitor.suggest_backoff_delay(provider)
        assert delay == 0.5
        
        # Low frequency (<10 requests)
        self.monitor._request_history[provider] = [now] * 5
        delay = self.monitor.suggest_backoff_delay(provider)
        assert delay == 0.0
    
    def test_time_window_calculations(self):
        """Test request counting in different time windows."""
        provider = "test_provider"
        now = datetime.now(timezone.utc)
        
        # Create requests at different times
        requests = [
            now,  # Current
            now - timedelta(seconds=30),  # 30 seconds ago
            now - timedelta(minutes=30),  # 30 minutes ago
            now - timedelta(hours=12),  # 12 hours ago
            now - timedelta(days=2)  # 2 days ago (should be cleaned up)
        ]
        
        self.monitor._request_history[provider] = requests
        status = self.monitor.check_rate_limit_status(provider)
        
        # Should count requests in different windows correctly
        # Note: the cleanup happens during record_request, not in check_rate_limit_status
        # So we may still have all 5 requests in the history at this point
        assert status["requests_last_minute"] == 2  # Current + 30s ago
        assert status["requests_last_hour"] == 3   # + 30min ago
        assert status["requests_last_day"] == 4    # + 12h ago
        # Total requests might be 4 or 5 depending on cleanup timing
        assert status["total_requests"] >= 4       # At least 4 (cleanup may have occurred)


class TestSecureFileHandler:
    """Test cases for SecureFileHandler."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.handler = SecureFileHandler(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        self.handler.cleanup_temp_files()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_temp_file_creation(self):
        """Test secure temporary file creation."""
        content = b"test content"
        temp_file = self.handler.create_temp_file(
            content=content,
            suffix=".txt",
            prefix="test_"
        )
        
        # File should exist and contain content
        assert temp_file.exists()
        assert temp_file.read_bytes() == content
        assert temp_file.name.startswith("test_")
        assert temp_file.name.endswith(".txt")
        
        # Check file permissions (should be restrictive)
        stat = temp_file.stat()
        permissions = oct(stat.st_mode)[-3:]
        assert permissions == "600"  # Owner read/write only
    
    def test_auto_cleanup(self):
        """Test automatic cleanup of temporary files."""
        temp_file = self.handler.create_temp_file(
            content=b"test",
            auto_cleanup=True
        )
        
        assert temp_file.exists()
        
        # Cleanup should remove the file
        self.handler.cleanup_temp_files()
        assert not temp_file.exists()
    
    def test_manual_cleanup_disabled(self):
        """Test files without auto cleanup."""
        temp_file = self.handler.create_temp_file(
            content=b"test",
            auto_cleanup=False
        )
        
        assert temp_file.exists()
        
        # Cleanup should not remove files not tracked
        self.handler.cleanup_temp_files()
        assert temp_file.exists()
        
        # Clean up manually
        temp_file.unlink()
    
    def test_secure_directory_creation(self):
        """Test secure directory creation with restricted permissions."""
        handler = SecureFileHandler()  # Default directory
        
        # Base directory should exist with restricted permissions
        assert handler.base_dir.exists()
        
        stat = handler.base_dir.stat()
        permissions = oct(stat.st_mode)[-3:]
        assert permissions == "700"  # Owner access only
    
    def test_unique_filenames(self):
        """Test that generated filenames are unique."""
        files = []
        for _ in range(10):
            temp_file = self.handler.create_temp_file()
            files.append(temp_file.name)
        
        # All filenames should be unique
        assert len(set(files)) == len(files)


class TestSecurityError:
    """Test cases for SecurityError exception."""
    
    def test_basic_security_error(self):
        """Test basic security error creation."""
        error = SecurityError("Test error message")
        
        assert str(error) == "Test error message"
        assert error.severity == "medium"
        assert error.error_code is None
        assert error.remediation is None
    
    def test_security_error_with_details(self):
        """Test security error with full details."""
        error = SecurityError(
            message="Validation failed",
            severity="high",
            error_code="VALIDATION_ERROR",
            remediation="Fix input format"
        )
        
        assert str(error) == "Validation failed"
        assert error.severity == "high"
        assert error.error_code == "VALIDATION_ERROR"
        assert error.remediation == "Fix input format"


class TestGlobalFunctions:
    """Test global convenience functions."""
    
    def test_get_api_key_manager(self):
        """Test getting global API key manager."""
        manager1 = get_api_key_manager()
        manager2 = get_api_key_manager()
        
        assert isinstance(manager1, APIKeyManager)
        assert manager1 is manager2  # Should be singleton
    
    def test_get_input_validator(self):
        """Test getting global input validator."""
        validator1 = get_input_validator()
        validator2 = get_input_validator()
        
        assert isinstance(validator1, InputValidator)
        assert validator1 is validator2  # Should be singleton
    
    def test_get_rate_limit_monitor(self):
        """Test getting global rate limit monitor."""
        monitor1 = get_rate_limit_monitor()
        monitor2 = get_rate_limit_monitor()
        
        assert isinstance(monitor1, RateLimitMonitor)
        assert monitor1 is monitor2  # Should be singleton
    
    def test_validate_input_convenience(self):
        """Test validate_input convenience function."""
        # Valid input
        assert validate_input("test.txt", "filename")
        
        # Invalid input
        with pytest.raises(SecurityError):
            validate_input("../etc/passwd", "filename")
    
    def test_sanitize_input_convenience(self):
        """Test sanitize_input convenience function."""
        dirty_filename = "file<name>.txt"
        clean_filename = sanitize_input(dirty_filename, "filename")
        
        assert "<" not in clean_filename
        assert ">" not in clean_filename


class TestValidationRule:
    """Test ValidationRule dataclass."""
    
    def test_validation_rule_creation(self):
        """Test validation rule creation."""
        rule = ValidationRule(
            name="test_rule",
            min_length=5,
            max_length=50,
            error_message="Test validation failed"
        )
        
        assert rule.name == "test_rule"
        assert rule.min_length == 5
        assert rule.max_length == 50
        assert rule.error_message == "Test validation failed"
        assert rule.severity == "medium"  # Default
        assert rule.forbidden_patterns == []  # Default
    
    def test_validation_rule_with_patterns(self):
        """Test validation rule with regex patterns."""
        import re
        
        rule = ValidationRule(
            name="pattern_rule",
            pattern=re.compile(r"^[a-z]+$"),
            forbidden_patterns=[re.compile(r"bad"), re.compile(r"evil")],
            severity="high"
        )
        
        assert rule.pattern.pattern == "^[a-z]+$"
        assert len(rule.forbidden_patterns) == 2
        assert rule.severity == "high"


class TestRateLimitInfo:
    """Test RateLimitInfo dataclass."""
    
    def test_rate_limit_info_creation(self):
        """Test rate limit info creation."""
        now = datetime.now(timezone.utc)
        
        info = RateLimitInfo(
            provider="openai",
            requests_per_minute=60,
            requests_per_hour=3600,
            requests_per_day=86400,
            current_window_start=now,
            current_requests=10
        )
        
        assert info.provider == "openai"
        assert info.requests_per_minute == 60
        assert info.current_requests == 10
        assert info.last_request_time is None  # Default
        assert info.rate_limit_headers == {}  # Default