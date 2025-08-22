# this_file: src/vttiro/core/errors.py
"""Error handling and exception definitions for VTTiro 2.0.

This module defines typed exceptions and error handling patterns used
throughout the VTTiro system. Provides clear error categorization and
helpful error messages for debugging and user feedback.

Used by:
- Provider implementations for consistent error reporting
- Core orchestration for error handling and fallbacks
- CLI interface for user-friendly error messages
- Testing infrastructure for error validation
"""

from typing import Any

# Import version for dynamic version reporting
try:
    from .. import __version__
except ImportError:
    __version__: str = "unknown"


class VttiroError(Exception):
    """Base exception for all VTTiro-specific errors.
    
    Provides common structure for error handling and logging.
    All VTTiro exceptions should inherit from this class.
    """
    
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        """Initialize error with message and optional details.
        
        Args:
            message: Human-readable error message
            details: Optional dictionary with additional error context
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self) -> str:
        """Return formatted error message."""
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} (details: {details_str})"
        return self.message


class ConfigurationError(VttiroError):
    """Raised when configuration is invalid or incomplete.
    
    Examples:
    - Missing API keys
    - Invalid provider settings
    - Conflicting configuration options
    """
    pass


class ProviderError(VttiroError):
    """Base class for provider-specific errors.
    
    Used for errors that occur during interaction with transcription
    providers (API failures, authentication issues, etc.).
    """
    
    def __init__(
        self, 
        message: str, 
        provider: str,
        details: dict[str, Any] | None = None
    ):
        """Initialize provider error.
        
        Args:
            message: Error message
            provider: Provider name (e.g., "gemini", "openai")
            details: Additional error context
        """
        super().__init__(message, details)
        self.provider = provider


class APIError(ProviderError):
    """Raised when external API calls fail.
    
    Examples:
    - Network timeouts
    - HTTP error responses
    - Rate limiting
    """
    
    def __init__(
        self,
        message: str,
        provider: str,
        status_code: int | None = None,
        response_text: str | None = None,
        details: dict[str, Any] | None = None
    ):
        """Initialize API error.
        
        Args:
            message: Error message
            provider: Provider name
            status_code: HTTP status code if applicable
            response_text: Response body if available
            details: Additional error context
        """
        details = details or {}
        if status_code is not None:
            details["status_code"] = status_code
        if response_text is not None:
            details["response_text"] = response_text[:500]  # Truncate long responses
            
        super().__init__(message, provider, details)
        self.status_code = status_code
        self.response_text = response_text


class AuthenticationError(ProviderError):
    """Raised when authentication with provider fails.
    
    Examples:
    - Invalid API keys
    - Expired tokens
    - Insufficient permissions
    """
    pass


class QuotaExceededError(ProviderError):
    """Raised when provider quota/rate limits are exceeded.
    
    Examples:
    - Monthly usage limits reached
    - Rate limiting in effect
    - Billing issues
    """
    
    def __init__(
        self,
        message: str,
        provider: str,
        retry_after: int | None = None,
        details: dict[str, Any] | None = None
    ):
        """Initialize quota error.
        
        Args:
            message: Error message
            provider: Provider name
            retry_after: Seconds until retry is allowed
            details: Additional error context
        """
        details = details or {}
        if retry_after is not None:
            details["retry_after"] = retry_after
            
        super().__init__(message, provider, details)
        self.retry_after = retry_after


class ContentFilterError(ProviderError):
    """Raised when content is blocked by safety filters.
    
    Examples:
    - Gemini safety filter blocking
    - Content policy violations
    - Potentially harmful content detected
    """
    
    def __init__(
        self,
        message: str,
        provider: str,
        blocked_categories: list[str] | None = None,
        details: dict[str, Any] | None = None
    ):
        """Initialize content filter error.
        
        Args:
            message: Error message
            provider: Provider name
            blocked_categories: List of blocked content categories
            details: Additional error context
        """
        details = details or {}
        if blocked_categories:
            details["blocked_categories"] = blocked_categories
            
        super().__init__(message, provider, details)
        self.blocked_categories = blocked_categories or []


class ProcessingError(VttiroError):
    """Raised when audio/video processing fails.
    
    Examples:
    - File format not supported
    - Corrupted media files
    - Preprocessing failures
    """
    
    def __init__(
        self,
        message: str,
        file_path: str | None = None,
        details: dict[str, Any] | None = None
    ):
        """Initialize processing error.
        
        Args:
            message: Error message
            file_path: Path to problematic file
            details: Additional error context
        """
        details = details or {}
        if file_path:
            details["file_path"] = file_path
            
        super().__init__(message, details)
        self.file_path = file_path


class TranscriptionError(VttiroError):
    """Raised when transcription process fails.
    
    Examples:
    - Empty or invalid responses
    - Parsing failures
    - Quality issues
    """
    
    def __init__(
        self,
        message: str,
        provider: str | None = None,
        details: dict[str, Any] | None = None
    ):
        """Initialize transcription error.
        
        Args:
            message: Error message
            provider: Provider name if applicable
            details: Additional error context
        """
        details = details or {}
        if provider:
            details["provider"] = provider
            
        super().__init__(message, details)
        self.provider = provider


class OutputError(VttiroError):
    """Raised when output generation fails.
    
    Examples:
    - File write permissions
    - Invalid output formats
    - Template rendering issues
    """
    pass


class RateLimitError(ProviderError):
    """Raised when provider rate limits are exceeded.
    
    Examples:
    - Too many requests per second
    - Burst rate limits exceeded
    - Temporary throttling
    """
    
    def __init__(
        self,
        message: str,
        provider: str,
        retry_after: int | None = None,
        details: dict[str, Any] | None = None
    ):
        """Initialize rate limit error with optional retry timing.
        
        Args:
            message: Error description
            provider: Provider name
            retry_after: Seconds to wait before retrying (if known)
            details: Additional error context
        """
        details = details or {}
        if retry_after is not None:
            details["retry_after"] = retry_after
            
        super().__init__(message, provider, details)
        self.retry_after = retry_after


class ProviderUnavailableError(ProviderError):
    """Raised when a provider service is temporarily unavailable.
    
    Examples:
    - Service maintenance
    - Temporary outages
    - Regional availability issues
    """
    pass


class TimeoutError(VttiroError):
    """Raised when operations exceed configured timeout limits.
    
    Examples:
    - Network request timeouts
    - Processing timeouts
    - Provider response timeouts
    """
    
    def __init__(
        self,
        message: str,
        timeout_duration: float | None = None,
        details: dict[str, Any] | None = None
    ):
        """Initialize timeout error with duration info.
        
        Args:
            message: Error description
            timeout_duration: Duration that was exceeded (if known)
            details: Additional error context
        """
        details = details or {}
        if timeout_duration is not None:
            details["timeout_duration"] = timeout_duration
            
        super().__init__(message, details)
        self.timeout_duration = timeout_duration


def handle_provider_exception(exc: Exception, provider: str) -> VttiroError:
    """Convert generic exceptions to typed VTTiro errors.
    
    Args:
        exc: Original exception
        provider: Provider name
        
    Returns:
        Appropriate VttiroError subclass
    """
    # Convert common exceptions to typed errors
    if isinstance(exc, FileNotFoundError):
        return ProcessingError(
            f"Audio file not found: {exc}",
            details={"original_error": str(exc)}
        )
    
    elif isinstance(exc, PermissionError):
        return ProcessingError(
            f"Permission denied: {exc}",
            details={"original_error": str(exc)}
        )
    
    elif isinstance(exc, ValueError):
        if "api key" in str(exc).lower() or "authentication" in str(exc).lower():
            return AuthenticationError(
                f"Authentication failed for {provider}: {exc}",
                provider=provider,
                details={"original_error": str(exc)}
            )
        else:
            return TranscriptionError(
                f"Invalid data or parameters: {exc}",
                provider=provider,
                details={"original_error": str(exc)}
            )
    
    elif hasattr(exc, 'status_code'):
        # HTTP-like errors
        status_code = getattr(exc, 'status_code', None)
        
        if status_code == 401:
            return AuthenticationError(
                f"Authentication failed for {provider}",
                provider=provider,
                details={"status_code": status_code, "original_error": str(exc)}
            )
        elif status_code == 429:
            return QuotaExceededError(
                f"Rate limit exceeded for {provider}",
                provider=provider,
                details={"status_code": status_code, "original_error": str(exc)}
            )
        elif status_code and status_code >= 400:
            return APIError(
                f"API error from {provider}",
                provider=provider,
                status_code=status_code,
                details={"original_error": str(exc)}
            )
    
    # Fallback to generic provider error
    return ProviderError(
        f"Unexpected error from {provider}: {exc}",
        provider=provider,
        details={"original_error": str(exc), "exception_type": type(exc).__name__}
    )


def suggest_solutions(error: VttiroError) -> list[str]:
    """Provide helpful suggestions for resolving errors.
    
    Args:
        error: VTTiro error instance
        
    Returns:
        List of suggested solutions
    """
    suggestions = []
    
    if isinstance(error, AuthenticationError):
        provider = getattr(error, 'provider', 'unknown')
        env_var_map = {
            'gemini': 'GEMINI_API_KEY',
            'openai': 'OPENAI_API_KEY', 
            'assemblyai': 'ASSEMBLYAI_API_KEY',
            'deepgram': 'DEEPGRAM_API_KEY'
        }
        env_var = env_var_map.get(provider, f'{provider.upper()}_API_KEY')
        
        suggestions.extend([
            f"Set your {provider} API key: export {env_var}=your_key_here",
            f"Verify the API key is correct (not expired/revoked)",
            f"Check your {provider} account dashboard for key status",
            f"Ensure your {provider} account has sufficient credits/quota"
        ])
        
        if provider in env_var_map:
            suggestions.append(f"Get API key from: {_get_provider_url(provider)}")
    
    elif isinstance(error, QuotaExceededError):
        provider = getattr(error, 'provider', 'unknown')
        suggestions.extend([
            f"Check your {provider} usage dashboard for current limits",
            f"Wait for quota reset (typically hourly/daily/monthly)",
            f"Upgrade your {provider} plan for higher limits",
            f"Use smaller audio files or shorter segments",
            f"Configure VTTiro to use a different provider as fallback"
        ])
        
        if error.retry_after:
            suggestions.append(f"API suggests retrying after {error.retry_after} seconds")
    
    elif isinstance(error, ContentFilterError):
        provider = getattr(error, 'provider', 'unknown')
        categories = getattr(error, 'blocked_categories', [])
        
        suggestions.extend([
            f"Audio content blocked by {provider} safety filters",
            f"Try a different provider (OpenAI/AssemblyAI have different policies)",
            f"Review audio for sensitive content (hate speech, violence, etc.)",
            f"Consider pre-processing audio to remove problematic sections"
        ])
        
        if categories:
            suggestions.append(f"Blocked categories: {', '.join(categories)}")
        
        if provider == 'gemini':
            suggestions.append("Gemini has strict safety filters; try OpenAI for sensitive content")
    
    elif isinstance(error, ProcessingError):
        file_path = getattr(error, 'file_path', None)
        suggestions.extend([
            "Verify audio file exists and is readable",
            "Check file format is supported: WAV, MP3, M4A, FLAC, OGG",
            "Try converting to WAV format: ffmpeg -i input.mp3 output.wav",
            "Ensure file size is under provider limits (typically 25MB-200MB)",
            "Check file is not corrupted by playing it in media player"
        ])
        
        if file_path:
            suggestions.append(f"Problematic file: {file_path}")
            # Add file-specific suggestions based on extension
            if file_path.endswith(('.mp4', '.mov', '.avi')):
                suggestions.append("For video files, extract audio first: ffmpeg -i video.mp4 -vn audio.wav")
    
    elif isinstance(error, APIError):
        provider = getattr(error, 'provider', 'unknown')
        status_code = getattr(error, 'status_code', None)
        
        suggestions.extend([
            "Check your internet connection stability",
            f"Verify {provider} service status at {_get_status_url(provider)}",
            "Try again in a few minutes (temporary service issues)",
            "Use exponential backoff for retries",
            "Configure a different provider as backup"
        ])
        
        if status_code:
            if status_code == 503:
                suggestions.append("Service temporarily unavailable - try again shortly")
            elif status_code == 502:
                suggestions.append("Gateway error - provider may be experiencing issues")
            elif status_code >= 500:
                suggestions.append("Server error on provider side - not your fault")
    
    elif isinstance(error, ConfigurationError):
        suggestions.extend([
            "Check your VTTiro configuration file syntax",
            "Verify all required fields are provided",
            "Ensure provider names are spelled correctly",
            "Review configuration documentation for valid options"
        ])
    
    # Add general debugging suggestions
    suggestions.extend([
        "Enable verbose logging for more details: set LOG_LEVEL=DEBUG",
        "Check VTTiro documentation: https://github.com/twardoch/vttiro",
        "Report persistent issues on GitHub with error details"
    ])
    
    return suggestions


def _get_provider_url(provider: str) -> str:
    """Get provider signup/API key URL."""
    urls = {
        'gemini': 'https://aistudio.google.com/app/apikey',
        'openai': 'https://platform.openai.com/api-keys',
        'assemblyai': 'https://www.assemblyai.com/app/account',
        'deepgram': 'https://console.deepgram.com/project/_/keys'
    }
    return urls.get(provider, f'https://{provider}.com')


def _get_status_url(provider: str) -> str:
    """Get provider status page URL."""
    status_urls = {
        'gemini': 'https://status.cloud.google.com/',
        'openai': 'https://status.openai.com/',
        'assemblyai': 'https://status.assemblyai.com/',
        'deepgram': 'https://status.deepgram.com/'
    }
    return status_urls.get(provider, f'https://status.{provider}.com')


def format_error_for_user(error: VttiroError, include_suggestions: bool = True) -> str:
    """Format error message for end-user display.
    
    Args:
        error: VTTiro error instance
        include_suggestions: Whether to include solution suggestions
        
    Returns:
        Formatted error message with context and suggestions
    """
    lines = [f"❌ {error.message}"]
    
    # Add error details if available
    if error.details:
        important_details = {k: v for k, v in error.details.items() 
                           if k not in ['original_error', 'exception_type']}
        if important_details:
            lines.append(f"   Details: {', '.join(f'{k}={v}' for k, v in important_details.items())}")
    
    # Add suggestions
    if include_suggestions:
        suggestions = suggest_solutions(error)
        if suggestions:
            lines.append("\n💡 Suggestions:")
            for i, suggestion in enumerate(suggestions[:5], 1):  # Limit to 5 suggestions
                lines.append(f"   {i}. {suggestion}")
            
            if len(suggestions) > 5:
                lines.append(f"   ... and {len(suggestions) - 5} more suggestions")
    
    return '\n'.join(lines)


def create_debug_context(
    operation: str,
    provider: str | None = None,
    file_path: str | None = None,
    **kwargs
) -> dict[str, Any]:
    """Create debugging context for error reporting.
    
    Args:
        operation: Description of operation being performed
        provider: Provider name if applicable
        file_path: File being processed if applicable
        **kwargs: Additional context information
        
    Returns:
        Dictionary with debugging context
    """
    import os
    import time
    from pathlib import Path
    
    context = {
        'operation': operation,
        'timestamp': time.time(),
        'vttiro_version': __version__,
    }
    
    if provider:
        context['provider'] = provider
        # Add provider-specific debugging info
        env_vars = {
            'gemini': 'GEMINI_API_KEY',
            'openai': 'OPENAI_API_KEY',
            'assemblyai': 'ASSEMBLYAI_API_KEY', 
            'deepgram': 'DEEPGRAM_API_KEY'
        }
        if provider in env_vars:
            key_set = env_vars[provider] in os.environ
            context[f'{provider}_api_key_set'] = key_set
    
    if file_path:
        path = Path(file_path)
        context.update({
            'file_path': str(path),
            'file_exists': path.exists(),
            'file_size': path.stat().st_size if path.exists() else None,
            'file_suffix': path.suffix.lower()
        })
    
    # Add any additional context
    context.update(kwargs)
    
    return context