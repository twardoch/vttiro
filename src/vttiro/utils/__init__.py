# this_file: src/vttiro/utils/__init__.py
"""Common utilities and helpers for vttiro package.

This module provides:
- Comprehensive exception hierarchy for error handling
- Basic error handling (resilience framework removed for simplification)
"""

from .exceptions import (
    # Base exceptions
    VttiroError,
    ConfigurationError,
    ValidationError,
    SecurityError,
    
    # Transcription-related exceptions
    TranscriptionError,
    APIError,
    NetworkError,
    AuthenticationError,
    RateLimitError,
    ServiceUnavailableError,
    ModelError,
    ProcessingError,
    SegmentationError,
    DiarizationError,
    EmotionDetectionError,
    
    # Output and resource exceptions
    OutputGenerationError,
    CacheError,
    ResourceError,
    
    # Utility functions
    create_error,
    from_error_code,
    ERROR_CODE_MAP,
)

# Resilience framework removed for simplification

__all__ = [
    # Exception classes
    "VttiroError",
    "ConfigurationError", 
    "ValidationError",
    "SecurityError",
    "TranscriptionError",
    "APIError",
    "NetworkError",
    "AuthenticationError",
    "RateLimitError",
    "ServiceUnavailableError",
    "ModelError",
    "ProcessingError",
    "SegmentationError",
    "DiarizationError",
    "EmotionDetectionError",
    "OutputGenerationError",
    "CacheError",
    "ResourceError",
    
    # Exception utilities
    "create_error",
    "from_error_code",
    "ERROR_CODE_MAP",
    
    # Resilience framework removed for simplification
]