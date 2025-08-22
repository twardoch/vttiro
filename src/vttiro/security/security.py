# this_file: src/vttiro/utils/security.py
"""Enhanced security and input validation utilities for VTTiro.

This module provides comprehensive security features including:
- API key encryption and secure storage
- Input sanitization and validation
- Rate limiting compliance monitoring
- Security headers and request validation
- Secure temporary file handling

Used by:
- All provider implementations for secure API key handling
- Input validation across the entire application
- Core orchestration for security compliance
- CLI for secure credential management
"""

import base64
import hashlib
import hmac
import os
import re
import secrets
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Pattern, Union
from urllib.parse import urlparse

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class SecurityError(Exception):
    """Security-related error with severity levels."""
    
    def __init__(
        self,
        message: str,
        severity: str = "medium",
        error_code: str = None,
        remediation: str = None
    ):
        super().__init__(message)
        self.severity = severity
        self.error_code = error_code
        self.remediation = remediation


class APIKeyManager:
    """Secure API key management with encryption at rest."""
    
    def __init__(self, master_key: Optional[str] = None):
        """Initialize API key manager.
        
        Args:
            master_key: Master encryption key (if None, generates from system info)
        """
        self._fernet = self._initialize_encryption(master_key)
        self._key_cache: Dict[str, str] = {}
        self._last_cache_clear = time.time()
        self._cache_ttl = 3600  # 1 hour cache TTL
    
    def _initialize_encryption(self, master_key: Optional[str] = None) -> Fernet:
        """Initialize encryption with master key."""
        if master_key:
            key_bytes = master_key.encode()
        else:
            # Generate key from system information for reproducibility
            system_info = f"{os.getlogin()}{os.path.dirname(__file__)}"
            key_bytes = system_info.encode()
        
        # Derive encryption key using PBKDF2
        salt = b"vttiro_salt_v1"  # Fixed salt for reproducibility
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(key_bytes))
        return Fernet(key)
    
    def encrypt_api_key(self, api_key: str, provider: str) -> str:
        """Encrypt API key for secure storage.
        
        Args:
            api_key: Plain text API key
            provider: Provider name for context
            
        Returns:
            Encrypted API key as base64 string
            
        Raises:
            SecurityError: Encryption failed
        """
        try:
            # Add provider context to prevent key reuse across providers
            contextualized_key = f"{provider}:{api_key}"
            encrypted = self._fernet.encrypt(contextualized_key.encode())
            return base64.urlsafe_b64encode(encrypted).decode()
        except Exception as e:
            raise SecurityError(
                f"Failed to encrypt API key for {provider}",
                severity="high",
                error_code="KEY_ENCRYPTION_FAILED",
                remediation="Check encryption configuration and system permissions"
            ) from e
    
    def decrypt_api_key(self, encrypted_key: str, provider: str) -> str:
        """Decrypt API key from secure storage.
        
        Args:
            encrypted_key: Encrypted API key as base64 string
            provider: Provider name for context
            
        Returns:
            Decrypted API key
            
        Raises:
            SecurityError: Decryption failed
        """
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_key.encode())
            decrypted = self._fernet.decrypt(encrypted_bytes).decode()
            
            # Verify provider context
            if not decrypted.startswith(f"{provider}:"):
                raise SecurityError(
                    f"API key context mismatch for provider {provider}",
                    severity="high",
                    error_code="KEY_CONTEXT_MISMATCH"
                )
            
            return decrypted[len(f"{provider}:"):]
        except Exception as e:
            raise SecurityError(
                f"Failed to decrypt API key for {provider}",
                severity="high",
                error_code="KEY_DECRYPTION_FAILED",
                remediation="Verify encrypted key format and encryption configuration"
            ) from e
    
    def get_api_key(self, provider: str, use_cache: bool = True) -> Optional[str]:
        """Get API key for provider with caching.
        
        Args:
            provider: Provider name
            use_cache: Whether to use cached keys
            
        Returns:
            API key or None if not found
        """
        # Clear cache if TTL expired
        if time.time() - self._last_cache_clear > self._cache_ttl:
            self._key_cache.clear()
            self._last_cache_clear = time.time()
        
        # Check cache first
        if use_cache and provider in self._key_cache:
            return self._key_cache[provider]
        
        # Try environment variable
        env_var_patterns = [
            f"{provider.upper()}_API_KEY",
            f"{provider}_API_KEY",
            f"VTTIRO_{provider.upper()}_API_KEY"
        ]
        
        for env_var in env_var_patterns:
            key = os.getenv(env_var)
            if key:
                # Check if key is encrypted (base64 + encrypted format)
                if self._is_encrypted_key(key):
                    try:
                        key = self.decrypt_api_key(key, provider)
                    except SecurityError:
                        continue  # Try next pattern
                
                if use_cache:
                    self._key_cache[provider] = key
                return key
        
        return None
    
    def _is_encrypted_key(self, key: str) -> bool:
        """Check if key appears to be encrypted."""
        try:
            # Encrypted keys are base64 encoded and have specific length patterns
            decoded = base64.urlsafe_b64decode(key.encode())
            return len(decoded) > 50  # Encrypted keys are much longer
        except Exception:
            return False
    
    def validate_api_key_format(self, api_key: str, provider: str) -> bool:
        """Validate API key format for specific provider.
        
        Args:
            api_key: API key to validate
            provider: Provider name
            
        Returns:
            True if format is valid
        """
        # Provider-specific validation patterns
        patterns = {
            "openai": r"^sk-[a-zA-Z0-9]{20,}$",
            "gemini": r"^AIza[a-zA-Z0-9_-]{35}$",
            "assemblyai": r"^[a-f0-9]{32}$",
            "deepgram": r"^[a-f0-9]{40}$"
        }
        
        pattern = patterns.get(provider.lower())
        if not pattern:
            return True  # Unknown provider, assume valid
        
        return bool(re.match(pattern, api_key))
    
    def clear_cache(self):
        """Clear API key cache."""
        self._key_cache.clear()
        self._last_cache_clear = time.time()


@dataclass
class ValidationRule:
    """Input validation rule definition."""
    name: str
    pattern: Optional[Pattern[str]] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    allowed_chars: Optional[str] = None
    forbidden_patterns: List[Pattern[str]] = field(default_factory=list)
    custom_validator: Optional[callable] = None
    error_message: str = "Validation failed"
    severity: str = "medium"


class InputValidator:
    """Comprehensive input validation and sanitization."""
    
    def __init__(self):
        """Initialize input validator with security rules."""
        self._rules = self._initialize_validation_rules()
        self._sanitizers = self._initialize_sanitizers()
    
    def _initialize_validation_rules(self) -> Dict[str, ValidationRule]:
        """Initialize validation rules for different input types."""
        return {
            "filename": ValidationRule(
                name="filename",
                pattern=re.compile(r"^[a-zA-Z0-9._-]+$"),
                max_length=255,
                forbidden_patterns=[
                    re.compile(r"\.\."),  # Path traversal
                    re.compile(r"[<>:\"|?*]"),  # Windows forbidden chars
                    re.compile(r"^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(\.|$)", re.IGNORECASE)  # Windows reserved
                ],
                error_message="Filename contains invalid characters or patterns",
                severity="high"
            ),
            "file_path": ValidationRule(
                name="file_path",
                max_length=4096,
                forbidden_patterns=[
                    re.compile(r"\.\."),  # Path traversal
                    re.compile(r"[<>:\"|?*]"),  # Forbidden chars
                    re.compile(r"^[/\\]"),  # Absolute paths
                ],
                error_message="File path contains dangerous patterns",
                severity="high"
            ),
            "url": ValidationRule(
                name="url",
                pattern=re.compile(r"^https?://[a-zA-Z0-9.-]+(:[0-9]+)?(/[a-zA-Z0-9._/-]*)?$"),
                max_length=2048,
                error_message="Invalid URL format",
                severity="medium"
            ),
            "language_code": ValidationRule(
                name="language_code",
                pattern=re.compile(r"^[a-z]{2}(-[A-Z]{2})?$"),
                error_message="Invalid language code format (use ISO 639-1)",
                severity="low"
            ),
            "provider_name": ValidationRule(
                name="provider_name",
                pattern=re.compile(r"^[a-z][a-z0-9_-]*$"),
                max_length=50,
                error_message="Invalid provider name format",
                severity="medium"
            ),
            "model_name": ValidationRule(
                name="model_name",
                pattern=re.compile(r"^[a-zA-Z0-9._-]+$"),
                max_length=100,
                forbidden_patterns=[
                    re.compile(r"[<>\"'&]"),  # Script injection chars
                ],
                error_message="Invalid model name format",
                severity="medium"
            ),
            "text_content": ValidationRule(
                name="text_content",
                max_length=1000000,  # 1MB limit
                forbidden_patterns=[
                    re.compile(r"<script[^>]*>", re.IGNORECASE),  # Script tags
                    re.compile(r"javascript:", re.IGNORECASE),  # JavaScript URLs
                    re.compile(r"data:.*base64", re.IGNORECASE),  # Data URLs
                ],
                error_message="Text content contains potentially dangerous patterns",
                severity="high"
            ),
            "api_key": ValidationRule(
                name="api_key",
                min_length=8,
                max_length=500,
                pattern=re.compile(r"^[a-zA-Z0-9._-]+$"),
                error_message="Invalid API key format",
                severity="high"
            )
        }
    
    def _initialize_sanitizers(self) -> Dict[str, callable]:
        """Initialize sanitization functions."""
        return {
            "filename": self._sanitize_filename,
            "text_content": self._sanitize_text_content,
            "url": self._sanitize_url,
            "model_name": self._sanitize_model_name
        }
    
    def validate(self, value: Any, rule_name: str, context: str = None) -> bool:
        """Validate input against specified rule.
        
        Args:
            value: Value to validate
            rule_name: Name of validation rule
            context: Additional context for validation
            
        Returns:
            True if validation passes
            
        Raises:
            SecurityError: Validation failed
        """
        if rule_name not in self._rules:
            raise SecurityError(
                f"Unknown validation rule: {rule_name}",
                severity="low",
                error_code="UNKNOWN_VALIDATION_RULE"
            )
        
        rule = self._rules[rule_name]
        str_value = str(value) if value is not None else ""
        
        # Length validation
        if rule.min_length is not None and len(str_value) < rule.min_length:
            raise SecurityError(
                f"Input too short: minimum {rule.min_length} characters required",
                severity=rule.severity,
                error_code="INPUT_TOO_SHORT",
                remediation=f"Provide at least {rule.min_length} characters"
            )
        
        if rule.max_length is not None and len(str_value) > rule.max_length:
            raise SecurityError(
                f"Input too long: maximum {rule.max_length} characters allowed",
                severity=rule.severity,
                error_code="INPUT_TOO_LONG",
                remediation=f"Limit input to {rule.max_length} characters"
            )
        
        # Pattern validation
        if rule.pattern and not rule.pattern.match(str_value):
            raise SecurityError(
                rule.error_message,
                severity=rule.severity,
                error_code="PATTERN_VALIDATION_FAILED",
                remediation="Check input format requirements"
            )
        
        # Forbidden pattern validation
        for forbidden_pattern in rule.forbidden_patterns:
            if forbidden_pattern.search(str_value):
                raise SecurityError(
                    f"Input contains forbidden pattern: {rule.error_message}",
                    severity=rule.severity,
                    error_code="FORBIDDEN_PATTERN_DETECTED",
                    remediation="Remove dangerous characters or patterns"
                )
        
        # Custom validation
        if rule.custom_validator and not rule.custom_validator(str_value, context):
            raise SecurityError(
                rule.error_message,
                severity=rule.severity,
                error_code="CUSTOM_VALIDATION_FAILED"
            )
        
        return True
    
    def sanitize(self, value: Any, sanitizer_name: str) -> str:
        """Sanitize input using specified sanitizer.
        
        Args:
            value: Value to sanitize
            sanitizer_name: Name of sanitizer function
            
        Returns:
            Sanitized string
        """
        if sanitizer_name not in self._sanitizers:
            return str(value)  # No sanitization available
        
        return self._sanitizers[sanitizer_name](str(value))
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe usage."""
        # Remove path separators and dangerous characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # Remove consecutive dots (path traversal)
        sanitized = re.sub(r'\.{2,}', '.', sanitized)
        # Limit length
        return sanitized[:255]
    
    def _sanitize_text_content(self, text: str) -> str:
        """Sanitize text content for safe processing."""
        # Remove script tags and dangerous patterns
        sanitized = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
        sanitized = re.sub(r'javascript:', '', sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r'data:.*?base64', '', sanitized, flags=re.IGNORECASE)
        return sanitized
    
    def _sanitize_url(self, url: str) -> str:
        """Sanitize URL for safe usage."""
        parsed = urlparse(url)
        if parsed.scheme not in ['http', 'https']:
            return ""  # Only allow HTTP/HTTPS
        return url
    
    def _sanitize_model_name(self, model_name: str) -> str:
        """Sanitize model name for safe usage."""
        # Only allow alphanumeric, dots, dashes, underscores
        return re.sub(r'[^a-zA-Z0-9._-]', '_', model_name)


@dataclass
class RateLimitInfo:
    """Rate limiting information for API monitoring."""
    provider: str
    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int
    current_window_start: datetime
    current_requests: int
    last_request_time: Optional[datetime] = None
    rate_limit_headers: Dict[str, str] = field(default_factory=dict)


class RateLimitMonitor:
    """Monitor and enforce rate limiting compliance."""
    
    def __init__(self):
        """Initialize rate limit monitor."""
        self._rate_limits: Dict[str, RateLimitInfo] = {}
        self._request_history: Dict[str, List[datetime]] = {}
    
    def record_request(
        self,
        provider: str,
        response_headers: Dict[str, str] = None
    ):
        """Record API request for rate limiting monitoring.
        
        Args:
            provider: Provider name
            response_headers: HTTP response headers
        """
        now = datetime.now(timezone.utc)
        
        # Initialize provider tracking if needed
        if provider not in self._request_history:
            self._request_history[provider] = []
        
        # Record request timestamp
        self._request_history[provider].append(now)
        
        # Clean old requests (keep last 24 hours)
        cutoff = now - timedelta(hours=24)
        self._request_history[provider] = [
            ts for ts in self._request_history[provider] if ts > cutoff
        ]
        
        # Extract rate limit info from headers
        if response_headers:
            self._update_rate_limit_info(provider, response_headers, now)
    
    def _update_rate_limit_info(
        self,
        provider: str,
        headers: Dict[str, str],
        timestamp: datetime
    ):
        """Update rate limit information from response headers."""
        # Common rate limit header patterns
        rate_limit_headers = {
            "x-ratelimit-remaining": "remaining",
            "x-ratelimit-limit": "limit",
            "x-ratelimit-reset": "reset",
            "retry-after": "retry_after",
            "x-ratelimit-requests": "requests",
            "x-ratelimit-tokens": "tokens"
        }
        
        extracted_headers = {}
        for header, key in rate_limit_headers.items():
            value = headers.get(header) or headers.get(header.title())
            if value:
                extracted_headers[key] = value
        
        if extracted_headers:
            if provider not in self._rate_limits:
                self._rate_limits[provider] = RateLimitInfo(
                    provider=provider,
                    requests_per_minute=60,  # Default assumption
                    requests_per_hour=3600,
                    requests_per_day=86400,
                    current_window_start=timestamp,
                    current_requests=1
                )
            
            self._rate_limits[provider].rate_limit_headers = extracted_headers
            self._rate_limits[provider].last_request_time = timestamp
    
    def check_rate_limit_status(self, provider: str) -> Dict[str, Any]:
        """Check current rate limit status for provider.
        
        Args:
            provider: Provider name
            
        Returns:
            Rate limit status information
        """
        if provider not in self._request_history:
            return {"status": "unknown", "provider": provider}
        
        now = datetime.now(timezone.utc)
        requests = self._request_history[provider]
        
        # Calculate requests in different time windows
        minute_ago = now - timedelta(minutes=1)
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(days=1)
        
        requests_last_minute = len([r for r in requests if r > minute_ago])
        requests_last_hour = len([r for r in requests if r > hour_ago])
        requests_last_day = len([r for r in requests if r > day_ago])
        
        status = {
            "provider": provider,
            "requests_last_minute": requests_last_minute,
            "requests_last_hour": requests_last_hour,
            "requests_last_day": requests_last_day,
            "total_requests": len(requests),
            "last_request": requests[-1].isoformat() if requests else None
        }
        
        # Add rate limit info if available
        if provider in self._rate_limits:
            limit_info = self._rate_limits[provider]
            status.update({
                "rate_limit_headers": limit_info.rate_limit_headers,
                "estimated_limits": {
                    "per_minute": limit_info.requests_per_minute,
                    "per_hour": limit_info.requests_per_hour,
                    "per_day": limit_info.requests_per_day
                }
            })
        
        return status
    
    def suggest_backoff_delay(self, provider: str) -> float:
        """Suggest backoff delay based on rate limiting.
        
        Args:
            provider: Provider name
            
        Returns:
            Suggested delay in seconds
        """
        if provider not in self._request_history:
            return 0.0
        
        now = datetime.now(timezone.utc)
        requests = self._request_history[provider]
        
        # Check recent request frequency
        minute_ago = now - timedelta(minutes=1)
        recent_requests = [r for r in requests if r > minute_ago]
        
        if len(recent_requests) >= 50:  # High frequency
            return 2.0
        elif len(recent_requests) >= 20:  # Medium frequency
            return 1.0
        elif len(recent_requests) >= 10:  # Low-medium frequency
            return 0.5
        
        return 0.0  # No delay needed


class SecureFileHandler:
    """Secure temporary file handling with automatic cleanup."""
    
    def __init__(self, base_dir: Optional[Path] = None):
        """Initialize secure file handler.
        
        Args:
            base_dir: Base directory for temporary files
        """
        self.base_dir = base_dir or Path(tempfile.gettempdir()) / "vttiro_secure"
        self.base_dir.mkdir(exist_ok=True, mode=0o700)  # Restricted permissions
        self._temp_files: List[Path] = []
    
    def create_temp_file(
        self,
        content: bytes = None,
        suffix: str = ".tmp",
        prefix: str = "vttiro_",
        auto_cleanup: bool = True
    ) -> Path:
        """Create secure temporary file.
        
        Args:
            content: Initial file content
            suffix: File suffix
            prefix: File prefix
            auto_cleanup: Whether to auto-cleanup on exit
            
        Returns:
            Path to temporary file
        """
        # Generate secure random filename
        random_name = secrets.token_hex(16)
        filename = f"{prefix}{random_name}{suffix}"
        file_path = self.base_dir / filename
        
        # Create file with restricted permissions
        file_path.touch(mode=0o600)
        
        if content:
            file_path.write_bytes(content)
        
        if auto_cleanup:
            self._temp_files.append(file_path)
        
        return file_path
    
    def cleanup_temp_files(self):
        """Clean up all temporary files."""
        for temp_file in self._temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except Exception:
                pass  # Ignore cleanup errors
        
        self._temp_files.clear()
    
    def __del__(self):
        """Cleanup on destruction."""
        self.cleanup_temp_files()


# Global instances
_api_key_manager: Optional[APIKeyManager] = None
_input_validator: Optional[InputValidator] = None
_rate_limit_monitor: Optional[RateLimitMonitor] = None


def get_api_key_manager() -> APIKeyManager:
    """Get global API key manager."""
    global _api_key_manager
    if _api_key_manager is None:
        _api_key_manager = APIKeyManager()
    return _api_key_manager


def get_input_validator() -> InputValidator:
    """Get global input validator."""
    global _input_validator
    if _input_validator is None:
        _input_validator = InputValidator()
    return _input_validator


def get_rate_limit_monitor() -> RateLimitMonitor:
    """Get global rate limit monitor."""
    global _rate_limit_monitor
    if _rate_limit_monitor is None:
        _rate_limit_monitor = RateLimitMonitor()
    return _rate_limit_monitor


def validate_input(value: Any, rule_name: str, context: str = None) -> bool:
    """Convenience function for input validation."""
    return get_input_validator().validate(value, rule_name, context)


def sanitize_input(value: Any, sanitizer_name: str) -> str:
    """Convenience function for input sanitization."""
    return get_input_validator().sanitize(value, sanitizer_name)