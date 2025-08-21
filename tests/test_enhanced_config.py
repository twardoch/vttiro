#!/usr/bin/env python3
# this_file: tests/test_enhanced_config.py
"""Tests for the enhanced configuration system."""

import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
import yaml

from src.vttiro.config.enhanced import (
    EnhancedVttiroConfig, SecureApiConfig, ProcessingConfig, 
    ValidationConfig, CachingConfig, MonitoringConfig, 
    SecretManager, ConfigValidationError, EnvironmentType
)
# Migration and hot-reload modules removed for simplification


class TestSecretManager:
    """Test the SecretManager class."""
    
    def test_generate_key(self):
        """Test key generation."""
        manager = SecretManager()
        key = manager._generate_key()
        assert len(key) == 44  # Base64 encoded 32-byte key
        
        # Generate another key to ensure they're different
        key2 = manager._generate_key()
        assert key != key2
    
    def test_encrypt_decrypt(self):
        """Test encryption and decryption."""
        manager = SecretManager()
        original_text = "test-api-key-12345"
        
        encrypted = manager.encrypt(original_text)
        assert encrypted != original_text
        assert isinstance(encrypted, str)
        
        decrypted = manager.decrypt(encrypted)
        assert decrypted == original_text
    
    def test_encrypt_decrypt_with_custom_key(self):
        """Test encryption/decryption with custom key."""
        custom_key = SecretManager._generate_key()
        manager = SecretManager(encryption_key=custom_key)
        
        original_text = "custom-key-test"
        encrypted = manager.encrypt(original_text)
        decrypted = manager.decrypt(encrypted)
        
        assert decrypted == original_text
    
    def test_decrypt_invalid_data(self):
        """Test decryption with invalid data."""
        manager = SecretManager()
        
        with pytest.raises(ValueError):
            manager.decrypt("invalid-encrypted-data")


class TestSecureApiConfig:
    """Test the SecureApiConfig class."""
    
    def test_valid_config(self):
        """Test valid API configuration."""
        config = SecureApiConfig(
            gemini_api_key="test-key",
            timeout_seconds=60,
            max_retries=3
        )
        
        assert config.gemini_api_key == "test-key"
        assert config.timeout_seconds == 60
        assert config.max_retries == 3
    
    def test_timeout_validation(self):
        """Test timeout validation."""
        with pytest.raises(ConfigValidationError):
            SecureApiConfig(timeout_seconds=0)
        
        with pytest.raises(ConfigValidationError):
            SecureApiConfig(timeout_seconds=601)
    
    def test_retry_validation(self):
        """Test retry validation."""
        with pytest.raises(ConfigValidationError):
            SecureApiConfig(max_retries=-1)
        
        with pytest.raises(ConfigValidationError):
            SecureApiConfig(max_retries=11)
    
    def test_rate_limit_validation(self):
        """Test rate limit validation."""
        with pytest.raises(ConfigValidationError):
            SecureApiConfig(rate_limit_per_minute=0)


class TestProcessingConfig:
    """Test the ProcessingConfig class."""
    
    def test_valid_config(self):
        """Test valid processing configuration."""
        config = ProcessingConfig(
            chunk_duration=300,
            overlap_duration=15,
            max_workers=4
        )
        
        assert config.chunk_duration == 300
        assert config.overlap_duration == 15
        assert config.max_workers == 4
    
    def test_chunk_duration_validation(self):
        """Test chunk duration validation."""
        with pytest.raises(ConfigValidationError):
            ProcessingConfig(chunk_duration=0)
        
        with pytest.raises(ConfigValidationError):
            ProcessingConfig(chunk_duration=3601)
    
    def test_overlap_validation(self):
        """Test overlap duration validation."""
        with pytest.raises(ConfigValidationError):
            ProcessingConfig(overlap_duration=-1)
        
        # Test overlap greater than chunk duration
        with pytest.raises(ConfigValidationError):
            ProcessingConfig(chunk_duration=60, overlap_duration=120)


class TestValidationConfig:
    """Test the ValidationConfig class."""
    
    def test_valid_config(self):
        """Test valid validation configuration."""
        config = ValidationConfig(
            max_file_size_mb=1024,
            allowed_domains=["example.com"],
            allowed_file_types=[".mp4", ".avi"]
        )
        
        assert config.max_file_size_mb == 1024
        assert "example.com" in config.allowed_domains
        assert ".mp4" in config.allowed_file_types
    
    def test_file_size_validation(self):
        """Test file size validation."""
        with pytest.raises(ConfigValidationError):
            ValidationConfig(max_file_size_mb=0)
        
        with pytest.raises(ConfigValidationError):
            ValidationConfig(max_file_size_mb=10241)
    
    def test_domain_validation(self):
        """Test domain validation."""
        # Valid domains
        config = ValidationConfig(allowed_domains=["example.com", "*.youtube.com"])
        assert len(config.allowed_domains) == 2
        
        # Invalid domain
        with pytest.raises(ConfigValidationError):
            ValidationConfig(allowed_domains=[""])


class TestCachingConfig:
    """Test the CachingConfig class."""
    
    def test_valid_config(self):
        """Test valid caching configuration."""
        config = CachingConfig(
            memory_cache_size=100,
            disk_cache_size_mb=1024,
            redis_enabled=True
        )
        
        assert config.memory_cache_size == 100
        assert config.disk_cache_size_mb == 1024
        assert config.redis_enabled is True
    
    def test_cache_size_validation(self):
        """Test cache size validation."""
        with pytest.raises(ConfigValidationError):
            CachingConfig(memory_cache_size=0)
        
        with pytest.raises(ConfigValidationError):
            CachingConfig(disk_cache_size_mb=-1)


class TestMonitoringConfig:
    """Test the MonitoringConfig class."""
    
    def test_valid_config(self):
        """Test valid monitoring configuration."""
        config = MonitoringConfig(
            log_level="INFO",
            metrics_port=9090,
            health_check_port=8080
        )
        
        assert config.log_level == "INFO"
        assert config.metrics_port == 9090
        assert config.health_check_port == 8080
    
    def test_log_level_validation(self):
        """Test log level validation."""
        # Valid log levels
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            config = MonitoringConfig(log_level=level)
            assert config.log_level == level
        
        # Invalid log level
        with pytest.raises(ConfigValidationError):
            MonitoringConfig(log_level="INVALID")
    
    def test_port_validation(self):
        """Test port validation."""
        with pytest.raises(ConfigValidationError):
            MonitoringConfig(metrics_port=0)
        
        with pytest.raises(ConfigValidationError):
            MonitoringConfig(health_check_port=65536)


class TestEnhancedVttiroConfig:
    """Test the main enhanced configuration class."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = EnhancedVttiroConfig()
        
        assert config.config_version == "2.0"
        assert config.environment == EnvironmentType.DEVELOPMENT
        assert isinstance(config.api, SecureApiConfig)
        assert isinstance(config.processing, ProcessingConfig)
        assert isinstance(config.validation, ValidationConfig)
        assert isinstance(config.caching, CachingConfig)
        assert isinstance(config.monitoring, MonitoringConfig)
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid configuration
        config_data = {
            "config_version": "2.0",
            "environment": "production",
            "api": {
                "timeout_seconds": 120,
                "max_retries": 5
            },
            "processing": {
                "chunk_duration": 600,
                "max_workers": 8
            }
        }
        
        config = EnhancedVttiroConfig(**config_data)
        assert config.environment == EnvironmentType.PRODUCTION
        assert config.api.timeout_seconds == 120
        assert config.processing.chunk_duration == 600
    
    def test_environment_type_validation(self):
        """Test environment type validation."""
        # Valid environments
        for env in ["development", "testing", "staging", "production"]:
            config = EnhancedVttiroConfig(environment=env)
            assert config.environment.value == env
        
        # Invalid environment
        with pytest.raises(ConfigValidationError):
            EnhancedVttiroConfig(environment="invalid")
    
    def test_save_and_load(self):
        """Test saving and loading configuration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_path = Path(f.name)
        
        try:
            # Create and save configuration
            original_config = EnhancedVttiroConfig(
                environment="production",
                api=SecureApiConfig(timeout_seconds=120),
                processing=ProcessingConfig(chunk_duration=600)
            )
            original_config.save_to_file(config_path)
            
            # Load configuration
            loaded_config = EnhancedVttiroConfig.load_from_file(config_path)
            
            assert loaded_config.environment == EnvironmentType.PRODUCTION
            assert loaded_config.api.timeout_seconds == 120
            assert loaded_config.processing.chunk_duration == 600
        
        finally:
            config_path.unlink()
    
    def test_environment_specific_validation(self):
        """Test environment-specific validation."""
        # Production should require encryption
        with pytest.raises(ConfigValidationError):
            EnhancedVttiroConfig(
                environment="production",
                security={"encryption_enabled": False}
            )
        
        # Development allows disabled encryption
        config = EnhancedVttiroConfig(
            environment="development",
            security={"encryption_enabled": False}
        )
        assert config.security.encryption_enabled is False


# Tests for migration and hot-reload functionality removed for simplification


# Integration tests
class TestConfigurationIntegration:
    """Integration tests for the complete configuration system."""
    
    def test_full_configuration_lifecycle(self):
        """Test complete configuration lifecycle with templates."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            
            # Create configuration from template
            template_path = Path("src/vttiro/config/templates/development.yaml")
            if template_path.exists():
                # Load development template
                with open(template_path, 'r') as f:
                    template_config = yaml.safe_load(f)
                
                # Save as test configuration
                with open(config_path, 'w') as f:
                    yaml.dump(template_config, f)
                
                # Load and validate configuration
                config = EnhancedVttiroConfig.load_from_file(config_path)
                assert config.config_version == "2.0"
                assert config.environment == EnvironmentType.DEVELOPMENT
                
                # Migration functionality removed for simplification
    
    def test_secret_management_integration(self):
        """Test secret management integration."""
        manager = SecretManager()
        
        # Test with API configuration
        api_config = SecureApiConfig(
            gemini_api_key="secret-key-123",
            assemblyai_api_key="another-secret"
        )
        
        # Encrypt secrets
        encrypted_gemini = manager.encrypt(api_config.gemini_api_key)
        encrypted_assembly = manager.encrypt(api_config.assemblyai_api_key)
        
        assert encrypted_gemini != api_config.gemini_api_key
        assert encrypted_assembly != api_config.assemblyai_api_key
        
        # Decrypt secrets
        decrypted_gemini = manager.decrypt(encrypted_gemini)
        decrypted_assembly = manager.decrypt(encrypted_assembly)
        
        assert decrypted_gemini == api_config.gemini_api_key
        assert decrypted_assembly == api_config.assemblyai_api_key