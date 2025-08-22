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

from pathlib import Path
from typing import Any, Optional

# Import version for dynamic version reporting
try:
    from vttiro import __version__

    VTTIRO_VERSION = __version__
except ImportError:
    VTTIRO_VERSION = "unknown"


class VttiroError(Exception):
    """Base exception for all VTTiro errors."""

    def __init__(
        self, message: str, error_code: str = "UNKNOWN", details: dict | None = None, guidance: str | None = None
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.guidance = guidance
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format error message with code and guidance."""
        formatted = f"[{self.error_code}] {self.message}"
        if self.guidance:
            formatted += f"\n💡 {self.guidance}"
        return formatted


class AuthenticationError(VttiroError):
    """Error related to API authentication."""

    def __init__(self, message: str, provider: str | None = None, details: dict | None = None):
        error_code = f"AUTH_ERROR_{provider.upper()}" if provider else "AUTH_ERROR"

        # Provider-specific guidance
        guidance_map = {
            "gemini": "Set VTTIRO_GEMINI_API_KEY, GEMINI_API_KEY, or GOOGLE_API_KEY environment variable",
            "openai": "Set VTTIRO_OPENAI_API_KEY or OPENAI_API_KEY environment variable",
            "deepgram": "Set VTTIRO_DEEPGRAM_API_KEY, DEEPGRAM_API_KEY, or DG_API_KEY environment variable",
            "assemblyai": "Set VTTIRO_ASSEMBLYAI_API_KEY, ASSEMBLYAI_API_KEY, or AAI_API_KEY environment variable",
        }

        guidance = guidance_map.get(provider, "Check API key configuration. Use 'vttiro apikeys' to debug.")
        super().__init__(message, error_code, details, guidance)


class ProcessingError(VttiroError):
    """Error during file processing or transcription."""

    def __init__(
        self, message: str, file_path: str | None = None, stage: str = "processing", details: dict | None = None
    ):
        self.file_path = file_path
        self.stage = stage
        error_code = f"PROCESSING_{stage.upper()}"

        # Stage-specific guidance
        guidance_map = {
            "audio_extraction": "Ensure FFmpeg is installed: 'brew install ffmpeg' (macOS) or 'apt install ffmpeg' (Ubuntu)",
            "chunking": "Try reducing max_segment_duration parameter or use smaller input files",
            "transcription": "Check network connection and API key validity",
            "output_generation": "Ensure output directory is writable",
        }

        guidance = guidance_map.get(stage, "Check file format compatibility and try again")
        if file_path:
            details = details or {}
            details["file_path"] = file_path

        super().__init__(message, error_code, details, guidance)


class APIError(VttiroError):
    """Error from external API calls."""

    def __init__(self, message: str, provider: str, status_code: int | None = None, details: dict | None = None):
        self.provider = provider
        self.status_code = status_code
        error_code = f"API_{provider.upper()}_{status_code}" if status_code else f"API_{provider.upper()}"

        # Status code specific guidance
        guidance_map = {
            400: "Check request parameters and file format compatibility",
            401: "Verify API key is valid and has sufficient permissions",
            403: "API key lacks required permissions or quota exceeded",
            404: "API endpoint not found - check provider configuration",
            429: "Rate limit exceeded - wait before retrying",
            500: "Provider server error - try again later",
            503: "Provider service unavailable - try different provider or wait",
        }

        guidance = guidance_map.get(status_code, f"Check {provider} API status and try again")
        super().__init__(message, error_code, details, guidance)


class ContentFilterError(VttiroError):
    """Error when content is filtered by AI provider."""

    def __init__(self, message: str, provider: str, details: dict | None = None):
        self.provider = provider
        error_code = f"CONTENT_FILTERED_{provider.upper()}"
        
        guidance = f"Content was rejected by {provider} safety filters. Try rephrasing input or use different provider."
        super().__init__(message, error_code, details, guidance)


class FileFormatError(VttiroError):
    """Error related to unsupported or invalid file formats."""

    def __init__(
        self,
        message: str,
        file_path: str | None = None,
        format_type: str | None = None,
        details: dict | None = None,
    ):
        self.file_path = file_path
        self.format_type = format_type
        error_code = f"FORMAT_{format_type.upper()}" if format_type else "FORMAT_UNSUPPORTED"

        guidance = "Supported formats: MP4, AVI, MOV, MKV, WebM, MP3, WAV, M4A, FLAC. Convert using FFmpeg if needed."
        if file_path:
            details = details or {}
            details["file_path"] = file_path

        super().__init__(message, error_code, details, guidance)


class ValidationError(VttiroError):
    """Error from input validation."""

    def __init__(self, message: str, field: str | None = None, value: Any | None = None, details: dict | None = None):
        self.field = field
        self.value = value
        error_code = f"VALIDATION_{field.upper()}" if field else "VALIDATION_ERROR"

        guidance = "Check input parameters and configuration. Use --help for valid options."
        super().__init__(message, error_code, details, guidance)


class TranscriptionError(VttiroError):
    """Error during the transcription process."""

    def __init__(self, message: str, attempts: int = 0, max_retries: int = 3, details: dict | None = None):
        self.attempts = attempts
        self.max_retries = max_retries
        error_code = "TRANSCRIPTION_FAILED"

        guidance = f"Transcription failed after {attempts}/{max_retries} attempts. Try different provider or check network connectivity."
        super().__init__(message, error_code, details, guidance)


def handle_provider_exception(exc: Exception, provider: str, context: dict | None = None) -> VttiroError:
    """Convert provider-specific exceptions to VttiroError with guidance.

    Args:
        exc: Original exception from provider
        provider: Provider name (gemini, openai, etc.)
        context: Additional context for error handling

    Returns:
        VttiroError with appropriate error code and guidance
    """
    context = context or {}
    exc_str = str(exc).lower()

    # Authentication errors
    if any(keyword in exc_str for keyword in ["api key", "authentication", "unauthorized", "invalid key"]):
        return AuthenticationError(
            f"{provider.title()} authentication failed: {exc}", provider=provider, details=context
        )

    # Rate limiting errors
    if any(keyword in exc_str for keyword in ["rate limit", "quota", "too many requests"]):
        return APIError(
            f"{provider.title()} rate limit exceeded: {exc}", provider=provider, status_code=429, details=context
        )

    # Network/connectivity errors
    if any(keyword in exc_str for keyword in ["network", "connection", "timeout", "dns"]):
        return APIError(f"Network error with {provider.title()}: {exc}", provider=provider, details=context)

    # File format errors
    if any(keyword in exc_str for keyword in ["format", "codec", "unsupported", "invalid file"]):
        return FileFormatError(f"File format error: {exc}", file_path=context.get("file_path"), details=context)

    # Generic processing error
    return ProcessingError(
        f"{provider.title()} processing error: {exc}", file_path=context.get("file_path"), details=context
    )


def create_debug_context(file_path: str | None = None, **kwargs) -> dict[str, Any]:
    """Create debug context for error reporting.

    Args:
        file_path: Path to file being processed
        **kwargs: Additional context information

    Returns:
        Context dictionary with debug information
    """
    context = {
        "vttiro_version": VTTIRO_VERSION,
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        **kwargs,
    }

    if file_path:
        path = Path(file_path)
        if path.exists():
            context.update(
                {"file_path": str(path), "file_size": path.stat().st_size, "file_extension": path.suffix.lower()}
            )

    return context
