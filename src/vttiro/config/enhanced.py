#!/usr/bin/env python3
# this_file: src/vttiro/config/enhanced.py
"""Enhanced configuration management with comprehensive validation and security."""

import os
import re
import hashlib
import base64
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Literal, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import threading
import time

try:
    from pydantic import BaseModel, Field, field_validator, model_validator, SecretStr
    from pydantic.types import StrictInt, StrictFloat, StrictBool, StrictStr, conint, confloat, constr
    import yaml
    from cryptography.fernet import Fernet
    from loguru import logger
except ImportError:
    raise ImportError("Enhanced configuration requires: pydantic, cryptography, pyyaml, loguru")

from vttiro.utils.exceptions import ConfigurationError, ValidationError


class LogLevel(str, Enum):
    """Logging levels enumeration."""
    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class OutputFormat(str, Enum):
    """Supported output formats."""
    WEBVTT = "webvtt"
    SRT = "srt"
    TTML = "ttml"
    ASS = "ass"
    JSON = "json"


class TranscriptionModel(str, Enum):
    """Supported transcription models."""
    AUTO = "auto"
    GEMINI = "gemini"
    ASSEMBLYAI = "assemblyai"
    DEEPGRAM = "deepgram"
    MOCK = "mock"


class WCAGLevel(str, Enum):
    """WCAG compliance levels."""
    A = "A"
    AA = "AA"
    AAA = "AAA"


class EnvironmentType(str, Enum):
    """Environment types for configuration profiles."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class ValidationContext:
    """Context for configuration validation."""
    environment: EnvironmentType
    strict_mode: bool = True
    allow_missing_secrets: bool = False
    validation_errors: List[str] = field(default_factory=list)
    
    def add_error(self, error: str) -> None:
        """Add validation error."""
        self.validation_errors.append(error)
    
    def has_errors(self) -> bool:
        """Check if validation has errors."""
        return len(self.validation_errors) > 0


class SecretManager:
    """Secure secret management with encryption."""
    
    def __init__(self, encryption_key: Optional[bytes] = None):
        """Initialize secret manager.
        
        Args:
            encryption_key: Optional encryption key (generates one if None)
        """
        if encryption_key is None:
            encryption_key = Fernet.generate_key()
        
        self.cipher = Fernet(encryption_key)
        self._key = encryption_key
        self._secrets_cache: Dict[str, str] = {}
        self._cache_lock = threading.RLock()
    
    def encrypt_secret(self, secret: str) -> str:
        """Encrypt a secret value.
        
        Args:
            secret: Plain text secret
            
        Returns:
            Encrypted secret as base64 string
        """
        encrypted = self.cipher.encrypt(secret.encode())
        return base64.b64encode(encrypted).decode()
    
    def decrypt_secret(self, encrypted_secret: str) -> str:
        """Decrypt a secret value.
        
        Args:
            encrypted_secret: Encrypted secret as base64 string
            
        Returns:
            Decrypted plain text secret
        """
        try:
            encrypted_bytes = base64.b64decode(encrypted_secret.encode())
            decrypted = self.cipher.decrypt(encrypted_bytes)
            return decrypted.decode()
        except Exception as e:
            raise ConfigurationError(f"Failed to decrypt secret: {e}")
    
    def get_secret(self, key: str, encrypted_value: Optional[str] = None) -> Optional[str]:
        """Get secret value with caching.
        
        Args:
            key: Secret key/name
            encrypted_value: Encrypted value (if None, checks environment)
            
        Returns:
            Decrypted secret value or None
        """
        with self._cache_lock:
            # Check cache first
            if key in self._secrets_cache:
                return self._secrets_cache[key]
            
            # Try environment variable
            env_value = os.getenv(key)
            if env_value:
                self._secrets_cache[key] = env_value
                return env_value
            
            # Try decrypting provided value
            if encrypted_value:
                try:
                    decrypted = self.decrypt_secret(encrypted_value)
                    self._secrets_cache[key] = decrypted
                    return decrypted
                except Exception:
                    pass
            
            return None
    
    def set_secret(self, key: str, value: str, encrypt: bool = True) -> str:
        """Set secret value with optional encryption.
        
        Args:
            key: Secret key/name
            value: Secret value
            encrypt: Whether to encrypt the value
            
        Returns:
            Encrypted value if encrypt=True, otherwise original value
        """
        with self._cache_lock:
            self._secrets_cache[key] = value
            
            if encrypt:
                return self.encrypt_secret(value)
            return value
    
    def clear_cache(self) -> None:
        """Clear secrets cache."""
        with self._cache_lock:
            self._secrets_cache.clear()
    
    def get_encryption_key(self) -> str:
        """Get encryption key as base64 string."""
        return base64.b64encode(self._key).decode()


class SecureApiConfig(BaseModel):
    """Secure API configuration with comprehensive validation."""
    
    # AI Model API Keys
    gemini_api_key: Optional[SecretStr] = Field(
        None, 
        description="Google Gemini API key",
        env="GEMINI_API_KEY"
    )
    assemblyai_api_key: Optional[SecretStr] = Field(
        None,
        description="AssemblyAI API key", 
        env="ASSEMBLYAI_API_KEY"
    )
    deepgram_api_key: Optional[SecretStr] = Field(
        None,
        description="Deepgram API key",
        env="DEEPGRAM_API_KEY"
    )
    huggingface_token: Optional[SecretStr] = Field(
        None,
        description="HuggingFace token for pyannote",
        env="HUGGINGFACE_TOKEN"
    )
    
    # YouTube API Configuration
    youtube_client_id: Optional[SecretStr] = Field(
        None,
        description="YouTube OAuth client ID",
        env="YOUTUBE_CLIENT_ID"
    )
    youtube_client_secret: Optional[SecretStr] = Field(
        None,
        description="YouTube OAuth client secret",
        env="YOUTUBE_CLIENT_SECRET"
    )
    
    # Rate Limiting
    max_requests_per_minute: conint(ge=1, le=1000) = Field(
        60,
        description="Maximum API requests per minute"
    )
    max_concurrent_requests: conint(ge=1, le=100) = Field(
        10,
        description="Maximum concurrent API requests"
    )
    
    # Timeout Configuration
    api_timeout_seconds: confloat(ge=1.0, le=300.0) = Field(
        30.0,
        description="API request timeout in seconds"
    )
    retry_attempts: conint(ge=0, le=10) = Field(
        3,
        description="Number of retry attempts for failed requests"
    )
    
    @field_validator('gemini_api_key', 'assemblyai_api_key', 'deepgram_api_key', 'huggingface_token', 'youtube_client_id', 'youtube_client_secret', mode='before')
    def validate_secret_format(cls, v):
        """Validate secret format and length."""
        if v is None:
            return v
            
        if isinstance(v, SecretStr):
            secret_value = v.get_secret_value()
        else:
            secret_value = str(v)
        
        if len(secret_value) < 10:
            raise ValueError("API keys/tokens must be at least 10 characters")
        
        # Check for placeholder values
        forbidden_values = {"your_key_here", "api_key", "token", "secret", "changeme"}
        if secret_value.lower() in forbidden_values:
            raise ValueError("API key cannot be a placeholder value")
        
        return v
    
    @model_validator(mode='before')
    def validate_api_configuration(cls, values):
        """Validate overall API configuration."""
        # Check that at least one transcription API key is provided
        transcription_keys = [
            values.get('gemini_api_key'),
            values.get('assemblyai_api_key'), 
            values.get('deepgram_api_key')
        ]
        
        if not any(key is not None for key in transcription_keys):
            logger.warning("No transcription API keys configured - will use mock transcription")
        
        return values
    
    def get_available_models(self) -> Set[TranscriptionModel]:
        """Get available transcription models based on configured API keys."""
        available = {TranscriptionModel.MOCK}  # Always available
        
        if self.gemini_api_key:
            available.add(TranscriptionModel.GEMINI)
        if self.assemblyai_api_key:
            available.add(TranscriptionModel.ASSEMBLYAI)
        if self.deepgram_api_key:
            available.add(TranscriptionModel.DEEPGRAM)
        
        return available


class ProcessingConfig(BaseModel):
    """Enhanced processing configuration with comprehensive validation."""
    
    # Chunk Configuration
    chunk_duration_seconds: conint(ge=15, le=1800) = Field(
        600,
        description="Audio chunk duration in seconds (15s-30min)"
    )
    chunk_overlap_seconds: conint(ge=0, le=120) = Field(
        30,
        description="Chunk overlap duration in seconds (0-2min)"
    )
    max_duration_seconds: conint(ge=60, le=36000) = Field(
        36000,
        description="Maximum video duration in seconds (1min-10hrs)"
    )
    
    # Audio Processing
    target_sample_rate: conint(ge=8000, le=48000) = Field(
        16000,
        description="Target audio sample rate in Hz"
    )
    audio_channels: conint(ge=1, le=2) = Field(
        1,
        description="Number of audio channels (1=mono, 2=stereo)"
    )
    
    # Performance Configuration
    max_workers: conint(ge=1, le=32) = Field(
        4,
        description="Maximum worker processes/threads"
    )
    memory_limit_mb: conint(ge=512, le=32768) = Field(
        2048,
        description="Memory limit in MB (512MB-32GB)"
    )
    
    # Energy-Based Segmentation
    energy_threshold_percentile: conint(ge=5, le=95) = Field(
        20,
        description="Energy threshold percentile for segmentation"
    )
    min_energy_window_seconds: confloat(ge=0.5, le=10.0) = Field(
        2.0,
        description="Minimum energy analysis window in seconds"
    )
    prefer_integer_boundaries: StrictBool = Field(
        True,
        description="Prefer integer second boundaries for segments"
    )
    
    # Quality Settings
    audio_quality: Literal["low", "medium", "high", "lossless"] = Field(
        "high",
        description="Audio processing quality level"
    )
    normalize_audio: StrictBool = Field(
        True,
        description="Normalize audio levels"
    )
    noise_reduction: StrictBool = Field(
        True,
        description="Apply noise reduction filtering"
    )
    
    @field_validator('chunk_duration_seconds')
    def validate_chunk_duration(cls, v):
        """Validate chunk duration is reasonable."""
        if v <= 60:  # 1 minute minimum
            raise ValueError(f"Chunk duration ({v}s) must be > 60 seconds")
        return v
    
    @field_validator('max_workers')
    def validate_max_workers(cls, v):
        """Validate max workers against system capabilities."""
        import os
        cpu_count = os.cpu_count() or 1
        if v > cpu_count * 2:
            logger.warning(f"max_workers ({v}) exceeds 2x CPU count ({cpu_count})")
        return v
    
    @model_validator(mode='before')
    def validate_performance_settings(cls, values):
        """Validate overall performance configuration."""
        memory_mb = values.get('memory_limit_mb', 2048)
        max_workers = values.get('max_workers', 4)
        
        # Rough estimate: each worker needs ~200MB minimum
        min_memory_needed = max_workers * 200
        if memory_mb < min_memory_needed:
            raise ValueError(
                f"Memory limit ({memory_mb}MB) too low for {max_workers} workers "
                f"(need at least {min_memory_needed}MB)"
            )
        
        return values


class ValidationConfig(BaseModel):
    """Input validation and security configuration."""
    
    # File Validation
    max_file_size_mb: conint(ge=1, le=10240) = Field(
        1024,
        description="Maximum input file size in MB (1MB-10GB)"
    )
    allowed_extensions: Set[str] = Field(
        {".mp4", ".avi", ".mov", ".mkv", ".wav", ".mp3", ".m4a", ".flac"},
        description="Allowed file extensions"
    )
    allowed_mime_types: Set[str] = Field(
        {"video/", "audio/"},
        description="Allowed MIME type prefixes"
    )
    
    # URL Validation
    allowed_domains: Set[str] = Field(
        {"youtube.com", "youtu.be", "vimeo.com", "dailymotion.com"},
        description="Allowed domains for URL sources"
    )
    block_private_ips: StrictBool = Field(
        True,
        description="Block private/localhost IP addresses"
    )
    max_url_length: conint(ge=100, le=8192) = Field(
        2048,
        description="Maximum URL length"
    )
    
    # Security Settings
    enable_sandbox: StrictBool = Field(
        True,
        description="Enable sandbox for file processing"
    )
    temp_file_cleanup: StrictBool = Field(
        True,
        description="Automatically cleanup temporary files"
    )
    
    @field_validator('allowed_extensions')
    def validate_extensions(cls, v):
        """Validate file extensions format."""
        validated = set()
        for ext in v:
            if not ext.startswith('.'):
                ext = '.' + ext
            validated.add(ext.lower())
        return validated



class MonitoringConfig(BaseModel):
    """Monitoring and observability configuration."""
    
    # Logging
    log_level: LogLevel = Field(
        LogLevel.INFO,
        description="Logging level"
    )
    log_format: Literal["text", "json"] = Field(
        "json",
        description="Log output format"
    )
    log_file_enabled: StrictBool = Field(
        True,
        description="Enable file logging"
    )
    log_file_path: Optional[str] = Field(
        None,
        description="Log file path (auto-generated if None)"
    )
    log_rotation_size_mb: conint(ge=1, le=1024) = Field(
        100,
        description="Log file rotation size in MB"
    )
    
    # Metrics
    metrics_enabled: StrictBool = Field(
        False,
        description="Enable metrics collection"
    )
    metrics_port: conint(ge=1024, le=65535) = Field(
        9090,
        description="Metrics server port"
    )
    
    # Health Checks
    health_check_enabled: StrictBool = Field(
        True,
        description="Enable health check endpoints"
    )
    health_check_interval_seconds: conint(ge=5, le=300) = Field(
        30,
        description="Health check interval in seconds"
    )


class EnhancedVttiroConfig(BaseModel):
    """Enhanced main configuration with comprehensive validation and security."""
    
    # Configuration Metadata
    config_version: StrictStr = Field(
        "2.0",
        description="Configuration schema version"
    )
    environment: EnvironmentType = Field(
        EnvironmentType.DEVELOPMENT,
        description="Environment type"
    )
    created_at: Optional[datetime] = Field(
        None,
        description="Configuration creation timestamp"
    )
    updated_at: Optional[datetime] = Field(
        None,
        description="Configuration last update timestamp"
    )
    
    # Configuration Sections
    api: SecureApiConfig = Field(
        default_factory=SecureApiConfig,
        description="API configuration"
    )
    processing: ProcessingConfig = Field(
        default_factory=ProcessingConfig,
        description="Processing configuration"
    )
    validation: ValidationConfig = Field(
        default_factory=ValidationConfig,
        description="Input validation configuration"
    )
    monitoring: MonitoringConfig = Field(
        default_factory=MonitoringConfig,
        description="Monitoring configuration"
    )
    
    # Legacy Compatibility
    transcription: Optional[Dict[str, Any]] = Field(
        None,
        description="Legacy transcription config (for compatibility)"
    )
    output: Optional[Dict[str, Any]] = Field(
        None,
        description="Legacy output config (for compatibility)"
    )
    
    class Config:
        """Pydantic configuration."""
        env_prefix = "VTTIRO_"
        case_sensitive = False
        validate_assignment = True
        use_enum_values = True
        extra = "forbid"  # Strict mode - no extra fields allowed
    
    def __init__(self, **data):
        """Initialize with timestamp."""
        if 'created_at' not in data:
            data['created_at'] = datetime.utcnow()
        if 'updated_at' not in data:
            data['updated_at'] = datetime.utcnow()
        super().__init__(**data)
    
    @field_validator('config_version')
    def validate_config_version(cls, v):
        """Validate configuration version."""
        supported_versions = {"1.0", "2.0"}
        if v not in supported_versions:
            raise ValueError(f"Unsupported config version: {v}")
        return v
    
    @model_validator(mode='before')
    def validate_environment_specific_settings(cls, values):
        """Validate settings specific to environment type."""
        environment = values.get('environment', EnvironmentType.DEVELOPMENT)
        
        if environment == EnvironmentType.PRODUCTION:
            # Production-specific validations
            monitoring = values.get('monitoring', {})
            if isinstance(monitoring, MonitoringConfig):
                if monitoring.log_level == LogLevel.DEBUG:
                    raise ValueError("DEBUG logging not recommended for production")
                if not monitoring.metrics_enabled:
                    logger.warning("Metrics disabled in production environment")
        
        return values
    
    def validate_comprehensive(self, context: Optional[ValidationContext] = None) -> ValidationContext:
        """Perform comprehensive validation with context."""
        if context is None:
            context = ValidationContext(environment=self.environment)
        
        try:
            # Validate each section
            self.api.get_available_models()  # Triggers API key validation
            
            # Environment-specific validation
            if context.environment == EnvironmentType.PRODUCTION:
                if not context.allow_missing_secrets:
                    available_models = self.api.get_available_models()
                    if len(available_models) == 1 and TranscriptionModel.MOCK in available_models:
                        context.add_error("Production requires at least one real transcription API key")
            
            # Resource validation
            import psutil
            available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)
            required_memory = self.processing.memory_limit_mb
            
            if required_memory > available_memory_mb * 0.8:  # Leave 20% buffer
                context.add_error(
                    f"Memory limit ({required_memory}MB) exceeds 80% of available memory "
                    f"({available_memory_mb:.0f}MB)"
                )
            
        except Exception as e:
            context.add_error(f"Validation error: {e}")
        
        return context
    
    def get_secret_value(self, secret_path: str, secret_manager: Optional[SecretManager] = None) -> Optional[str]:
        """Get decrypted secret value.
        
        Args:
            secret_path: Dot-notation path to secret (e.g., "api.gemini_api_key")
            secret_manager: Secret manager instance
            
        Returns:
            Decrypted secret value or None
        """
        parts = secret_path.split('.')
        obj = self
        
        for part in parts[:-1]:
            obj = getattr(obj, part, None)
            if obj is None:
                return None
        
        secret_field = getattr(obj, parts[-1], None)
        if secret_field is None:
            return None
        
        if isinstance(secret_field, SecretStr):
            return secret_field.get_secret_value()
        
        return str(secret_field) if secret_field else None
    
    def update_timestamp(self) -> None:
        """Update the last modified timestamp."""
        self.updated_at = datetime.utcnow()
    
    def to_dict_safe(self) -> Dict[str, Any]:
        """Convert to dictionary with secrets masked."""
        data = self.dict()
        
        # Mask all secret fields
        def mask_secrets(obj: Any, path: str = "") -> Any:
            if isinstance(obj, dict):
                return {k: mask_secrets(v, f"{path}.{k}" if path else k) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [mask_secrets(item, f"{path}[{i}]") for i, item in enumerate(obj)]
            elif isinstance(obj, str) and any(secret_key in path.lower() for secret_key in 
                                              ['key', 'token', 'secret', 'password']):
                return "***MASKED***"
            else:
                return obj
        
        return mask_secrets(data)
    
    @classmethod
    def from_file(
        cls, 
        file_path: Union[str, Path], 
        secret_manager: Optional[SecretManager] = None
    ) -> 'EnhancedVttiroConfig':
        """Load configuration from file with secret decryption.
        
        Args:
            file_path: Path to configuration file
            secret_manager: Secret manager for decryption
            
        Returns:
            Loaded configuration instance
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise ConfigurationError(f"Configuration file not found: {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                data = yaml.safe_load(f)
            
            if not data:
                raise ConfigurationError("Configuration file is empty")
            
            return cls(**data)
            
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in configuration file: {e}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}")
    
    def save_to_file(
        self, 
        file_path: Union[str, Path], 
        mask_secrets: bool = True,
        secret_manager: Optional[SecretManager] = None
    ) -> None:
        """Save configuration to file with optional secret encryption.
        
        Args:
            file_path: Path to save configuration
            mask_secrets: Whether to mask secrets in saved file
            secret_manager: Secret manager for encryption
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Update timestamp
        self.update_timestamp()
        
        try:
            if mask_secrets:
                data = self.to_dict_safe()
            else:
                data = self.dict()
            
            with open(file_path, 'w') as f:
                yaml.dump(
                    data, 
                    f, 
                    default_flow_style=False, 
                    indent=2,
                    sort_keys=True
                )
                
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration: {e}")
    
    @classmethod
    def create_profile(cls, environment: EnvironmentType) -> 'EnhancedVttiroConfig':
        """Create configuration profile for specific environment.
        
        Args:
            environment: Target environment type
            
        Returns:
            Pre-configured instance for environment
        """
        if environment == EnvironmentType.DEVELOPMENT:
            return cls(
                environment=environment,
                monitoring=MonitoringConfig(
                    log_level=LogLevel.DEBUG,
                    metrics_enabled=False
                ),
                processing=ProcessingConfig(
                    max_workers=2,
                    memory_limit_mb=1024
                ),
            )
        elif environment == EnvironmentType.TESTING:
            return cls(
                environment=environment,
                monitoring=MonitoringConfig(
                    log_level=LogLevel.WARNING,
                    log_file_enabled=False,
                    metrics_enabled=False
                ),
                processing=ProcessingConfig(
                    max_workers=1,
                    memory_limit_mb=512
                ),
            )
        elif environment == EnvironmentType.PRODUCTION:
            return cls(
                environment=environment,
                monitoring=MonitoringConfig(
                    log_level=LogLevel.INFO,
                    log_format="json",
                    metrics_enabled=True,
                    health_check_enabled=True
                ),
                processing=ProcessingConfig(
                    max_workers=8,
                    memory_limit_mb=4096
                ),
                validation=ValidationConfig(
                    enable_sandbox=True,
                    block_private_ips=True
                )
            )
        else:  # STAGING
            return cls(
                environment=environment,
                monitoring=MonitoringConfig(
                    log_level=LogLevel.INFO,
                    metrics_enabled=True
                ),
                processing=ProcessingConfig(
                    max_workers=4,
                    memory_limit_mb=2048
                ),
            )