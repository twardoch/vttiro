#!/usr/bin/env python3
# this_file: src/vttiro/utils/exceptions.py
"""Comprehensive exception hierarchy for vttiro with detailed error information."""

from typing import Optional, Dict, Any
import uuid
from datetime import datetime


class VttiroError(Exception):
    """Base exception class for all vttiro errors.
    
    Provides error correlation tracking, structured error information,
    and context preservation for debugging and monitoring.
    """
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        correlation_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        """Initialize VttiroError with comprehensive error information.
        
        Args:
            message: Human-readable error description
            error_code: Machine-readable error code for programmatic handling
            correlation_id: Request correlation ID for tracking across services
            context: Additional context information for debugging
            cause: Original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.context = context or {}
        self.cause = cause
        self.timestamp = datetime.utcnow()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for structured logging and serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
            "cause": str(self.cause) if self.cause else None
        }
        
    def __str__(self) -> str:
        """String representation with correlation ID for easy tracking."""
        return f"[{self.correlation_id[:8]}] {self.message}"


class ConfigurationError(VttiroError):
    """Errors related to configuration validation and setup.
    
    Used when:
    - Invalid configuration parameters
    - Missing required API keys or settings
    - Configuration file parsing errors
    - Environment setup issues
    """
    pass


class ValidationError(VttiroError):
    """Errors related to input validation and sanitization.
    
    Used when:
    - Invalid input parameters
    - File format validation failures
    - URL validation errors
    - Data type validation issues
    """
    pass


class TranscriptionError(VttiroError):
    """Base class for all transcription-related errors.
    
    Used when:
    - General transcription pipeline failures
    - Audio processing errors
    - Model loading issues
    - Result processing failures
    """
    pass


class APIError(TranscriptionError):
    """Base class for external API-related errors.
    
    Used when:
    - External service API failures
    - Authentication errors
    - Rate limiting issues
    - Service unavailability
    """
    
    def __init__(
        self,
        message: str,
        service_name: Optional[str] = None,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
        **kwargs
    ):
        """Initialize APIError with service-specific information.
        
        Args:
            message: Human-readable error description
            service_name: Name of the external service (e.g., 'gemini', 'assemblyai')
            status_code: HTTP status code if applicable
            response_body: Raw response body for debugging
            **kwargs: Additional VttiroError parameters
        """
        super().__init__(message, **kwargs)
        self.service_name = service_name
        self.status_code = status_code
        self.response_body = response_body
        
        # Add service-specific context
        if service_name:
            self.context["service_name"] = service_name
        if status_code:
            self.context["status_code"] = status_code
        if response_body:
            self.context["response_body"] = response_body[:1000]  # Limit size


class NetworkError(APIError):
    """Network-related errors for external service calls.
    
    Used when:
    - Connection timeouts
    - DNS resolution failures
    - Connection refused errors
    - Network unreachability
    """
    pass


class AuthenticationError(APIError):
    """Authentication and authorization errors.
    
    Used when:
    - Invalid API keys
    - Expired tokens
    - Insufficient permissions
    - Authentication service failures
    """
    pass


class RateLimitError(APIError):
    """Rate limiting errors from external services.
    
    Used when:
    - API rate limits exceeded
    - Quota exhausted
    - Temporary throttling
    - Service-imposed limits
    """
    
    def __init__(
        self,
        message: str,
        retry_after: Optional[int] = None,
        current_usage: Optional[int] = None,
        limit: Optional[int] = None,
        **kwargs
    ):
        """Initialize RateLimitError with rate limiting details.
        
        Args:
            message: Human-readable error description
            retry_after: Seconds to wait before retrying
            current_usage: Current usage count
            limit: Rate limit threshold
            **kwargs: Additional APIError parameters
        """
        super().__init__(message, **kwargs)
        self.retry_after = retry_after
        self.current_usage = current_usage
        self.limit = limit
        
        # Add rate limiting context
        if retry_after is not None:
            self.context["retry_after"] = retry_after
        if current_usage is not None:
            self.context["current_usage"] = current_usage
        if limit is not None:
            self.context["limit"] = limit


class ServiceUnavailableError(APIError):
    """Service unavailability errors.
    
    Used when:
    - External service maintenance
    - Service outages
    - Temporary service disruptions
    - Load balancer failures
    """
    pass


class ModelError(TranscriptionError):
    """AI model-related errors.
    
    Used when:
    - Model loading failures
    - Model inference errors
    - Unsupported model configurations
    - Model resource constraints
    """
    
    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        **kwargs
    ):
        """Initialize ModelError with model-specific information.
        
        Args:
            message: Human-readable error description
            model_name: Name of the AI model
            model_version: Version of the model
            **kwargs: Additional TranscriptionError parameters
        """
        super().__init__(message, **kwargs)
        self.model_name = model_name
        self.model_version = model_version
        
        # Add model-specific context
        if model_name:
            self.context["model_name"] = model_name
        if model_version:
            self.context["model_version"] = model_version


class ProcessingError(TranscriptionError):
    """Audio/video processing errors.
    
    Used when:
    - Audio extraction failures
    - Video download errors
    - File format issues
    - Codec problems
    """
    
    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        processing_stage: Optional[str] = None,
        **kwargs
    ):
        """Initialize ProcessingError with processing details.
        
        Args:
            message: Human-readable error description
            file_path: Path to file being processed
            processing_stage: Stage where error occurred
            **kwargs: Additional TranscriptionError parameters
        """
        super().__init__(message, **kwargs)
        self.file_path = file_path
        self.processing_stage = processing_stage
        
        # Add processing context
        if file_path:
            self.context["file_path"] = file_path
        if processing_stage:
            self.context["processing_stage"] = processing_stage


class SegmentationError(ProcessingError):
    """Audio segmentation-related errors.
    
    Used when:
    - Segmentation algorithm failures
    - Invalid segment boundaries
    - Energy analysis errors
    - Chunk reassembly issues
    """
    pass


class DiarizationError(TranscriptionError):
    """Speaker diarization errors.
    
    Used when:
    - Speaker identification failures
    - Diarization model errors
    - Embedding extraction issues
    - Clustering failures
    """
    pass


class EmotionDetectionError(TranscriptionError):
    """Emotion detection errors.
    
    Used when:
    - Emotion analysis failures
    - Model inference errors
    - Feature extraction issues
    - Classification failures
    """
    pass


class OutputGenerationError(VttiroError):
    """Output format generation errors.
    
    Used when:
    - WebVTT generation failures
    - Subtitle formatting errors
    - File writing issues
    - Format conversion problems
    """
    
    def __init__(
        self,
        message: str,
        output_format: Optional[str] = None,
        output_path: Optional[str] = None,
        **kwargs
    ):
        """Initialize OutputGenerationError with output details.
        
        Args:
            message: Human-readable error description
            output_format: Target output format (webvtt, srt, etc.)
            output_path: Output file path
            **kwargs: Additional VttiroError parameters
        """
        super().__init__(message, **kwargs)
        self.output_format = output_format
        self.output_path = output_path
        
        # Add output context
        if output_format:
            self.context["output_format"] = output_format
        if output_path:
            self.context["output_path"] = output_path


class SecurityError(VttiroError):
    """Security-related errors.
    
    Used when:
    - Input validation failures
    - Unauthorized access attempts
    - File system security violations
    - Malicious input detection
    """
    pass


class CacheError(VttiroError):
    """Caching system errors.
    
    Used when:
    - Cache connection failures
    - Cache key generation errors
    - Cache serialization issues
    - Cache invalidation problems
    """
    pass


class ResourceError(VttiroError):
    """Resource constraint and availability errors.
    
    Used when:
    - Insufficient memory
    - Disk space exhaustion
    - CPU resource limits
    - GPU unavailability
    """
    
    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        current_usage: Optional[float] = None,
        limit: Optional[float] = None,
        **kwargs
    ):
        """Initialize ResourceError with resource details.
        
        Args:
            message: Human-readable error description
            resource_type: Type of resource (memory, disk, cpu, gpu)
            current_usage: Current resource usage
            limit: Resource limit
            **kwargs: Additional VttiroError parameters
        """
        super().__init__(message, **kwargs)
        self.resource_type = resource_type
        self.current_usage = current_usage
        self.limit = limit
        
        # Add resource context
        if resource_type:
            self.context["resource_type"] = resource_type
        if current_usage is not None:
            self.context["current_usage"] = current_usage
        if limit is not None:
            self.context["limit"] = limit


# Convenience function for error creation with correlation ID propagation
def create_error(
    error_class: type,
    message: str,
    correlation_id: Optional[str] = None,
    **kwargs
) -> VttiroError:
    """Create an error instance with consistent correlation ID handling.
    
    Args:
        error_class: Exception class to instantiate
        message: Error message
        correlation_id: Request correlation ID
        **kwargs: Additional error-specific parameters
        
    Returns:
        Configured error instance
    """
    if not issubclass(error_class, VttiroError):
        raise ValueError(f"error_class must be a subclass of VttiroError, got {error_class}")
    
    return error_class(
        message=message,
        correlation_id=correlation_id,
        **kwargs
    )


# Error code mapping for programmatic error handling
ERROR_CODE_MAP = {
    "VTTIRO_CONFIGURATION": ConfigurationError,
    "VTTIRO_VALIDATION": ValidationError,
    "VTTIRO_TRANSCRIPTION": TranscriptionError,
    "VTTIRO_API": APIError,
    "VTTIRO_NETWORK": NetworkError,
    "VTTIRO_AUTH": AuthenticationError,
    "VTTIRO_RATE_LIMIT": RateLimitError,
    "VTTIRO_SERVICE_UNAVAILABLE": ServiceUnavailableError,
    "VTTIRO_MODEL": ModelError,
    "VTTIRO_PROCESSING": ProcessingError,
    "VTTIRO_SEGMENTATION": SegmentationError,
    "VTTIRO_DIARIZATION": DiarizationError,
    "VTTIRO_EMOTION": EmotionDetectionError,
    "VTTIRO_OUTPUT": OutputGenerationError,
    "VTTIRO_SECURITY": SecurityError,
    "VTTIRO_CACHE": CacheError,
    "VTTIRO_RESOURCE": ResourceError,
}


def from_error_code(error_code: str, message: str, **kwargs) -> VttiroError:
    """Create error instance from error code.
    
    Args:
        error_code: Machine-readable error code
        message: Human-readable error message
        **kwargs: Additional error-specific parameters
        
    Returns:
        Configured error instance
        
    Raises:
        ValueError: If error code is not recognized
    """
    if error_code not in ERROR_CODE_MAP:
        raise ValueError(f"Unknown error code: {error_code}")
    
    error_class = ERROR_CODE_MAP[error_code]
    return error_class(message=message, error_code=error_code, **kwargs)