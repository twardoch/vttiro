#!/usr/bin/env python3
# this_file: tests/test_error_handling.py
"""Unit tests for comprehensive error handling and resilience framework."""

import asyncio
import uuid
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from vttiro.core.errors import (
    APIError,
    AuthenticationError,
    FileFormatError,
    ProcessingError,
    TranscriptionError,
    ValidationError,
    VttiroError,
)

# Resilience framework imports removed for simplification


class TestExceptionHierarchy:
    """Test exception hierarchy and error creation."""

    def test_vttiro_error_creation(self):
        """Test VttiroError base exception creation."""
        correlation_id = str(uuid.uuid4())
        context = {"source": "test", "operation": "unit_test"}

        error = VttiroError(
            message="Test error", error_code="TEST_ERROR", correlation_id=correlation_id, context=context
        )

        assert str(error) == f"[{correlation_id[:8]}] Test error"
        assert error.message == "Test error"
        assert error.error_code == "TEST_ERROR"
        assert error.correlation_id == correlation_id
        assert error.context == context
        assert isinstance(error.timestamp, datetime)

    def test_error_serialization(self):
        """Test error to_dict serialization."""
        error = ConfigurationError(
            message="Invalid configuration", error_code="CONFIG_INVALID", context={"config_file": "test.yaml"}
        )

        error_dict = error.to_dict()

        assert error_dict["error_type"] == "ConfigurationError"
        assert error_dict["message"] == "Invalid configuration"
        assert error_dict["error_code"] == "CONFIG_INVALID"
        assert "correlation_id" in error_dict
        assert "timestamp" in error_dict
        assert error_dict["context"]["config_file"] == "test.yaml"

    def test_api_error_with_service_info(self):
        """Test APIError with service-specific information."""
        error = APIError(
            message="API call failed",
            service_name="gemini",
            status_code=429,
            response_body='{"error": "rate_limit_exceeded"}',
        )

        assert error.service_name == "gemini"
        assert error.status_code == 429
        assert error.response_body == '{"error": "rate_limit_exceeded"}'
        assert error.context["service_name"] == "gemini"
        assert error.context["status_code"] == 429

    def test_rate_limit_error_details(self):
        """Test RateLimitError with rate limiting details."""
        error = RateLimitError(
            message="Rate limit exceeded", service_name="assemblyai", retry_after=300, current_usage=1000, limit=1000
        )

        assert error.retry_after == 300
        assert error.current_usage == 1000
        assert error.limit == 1000
        assert error.context["retry_after"] == 300
        assert error.context["current_usage"] == 1000
        assert error.context["limit"] == 1000

    def test_model_error_details(self):
        """Test ModelError with model-specific information."""
        error = ModelError(message="Model inference failed", model_name="gemini-2.0-flash", model_version="1.0")

        assert error.model_name == "gemini-2.0-flash"
        assert error.model_version == "1.0"
        assert error.context["model_name"] == "gemini-2.0-flash"
        assert error.context["model_version"] == "1.0"

    def test_processing_error_details(self):
        """Test ProcessingError with processing details."""
        error = ProcessingError(
            message="Video processing failed", file_path="/path/to/video.mp4", processing_stage="audio_extraction"
        )

        assert error.file_path == "/path/to/video.mp4"
        assert error.processing_stage == "audio_extraction"
        assert error.context["file_path"] == "/path/to/video.mp4"
        assert error.context["processing_stage"] == "audio_extraction"

    def test_create_error_function(self):
        """Test error creation utility function."""
        correlation_id = str(uuid.uuid4())

        error = create_error(NetworkError, "Connection failed", correlation_id=correlation_id, service_name="deepgram")

        assert isinstance(error, NetworkError)
        assert error.message == "Connection failed"
        assert error.correlation_id == correlation_id
        assert error.service_name == "deepgram"

    def test_from_error_code_function(self):
        """Test error creation from error code."""
        error = from_error_code("VTTIRO_VALIDATION", "Input validation failed", context={"field": "language"})

        assert isinstance(error, ValidationError)
        assert error.message == "Input validation failed"
        assert error.error_code == "VTTIRO_VALIDATION"
        assert error.context["field"] == "language"

    def test_invalid_error_code(self):
        """Test error creation with invalid error code."""
        with pytest.raises(ValueError, match="Unknown error code"):
            from_error_code("INVALID_CODE", "Test message")


# Resilience framework tests removed for simplification
