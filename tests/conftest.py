#!/usr/bin/env python3
# this_file: tests/conftest.py
"""Shared test fixtures and configuration for vttiro test suite."""

import asyncio
import shutil
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from vttiro.core.config import VttiroConfig
from vttiro.core.types import TranscriptionResult


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def mock_config():
    """Create a mock VttiroConfig for testing."""
    config = MagicMock(spec=VttiroConfig)

    # Mock transcription settings
    config.transcription = MagicMock()
    config.transcription.preferred_model = "auto"
    config.transcription.gemini_api_key = None
    config.transcription.assemblyai_api_key = None
    config.transcription.deepgram_api_key = None
    config.transcription.max_duration_seconds = 3600
    config.transcription.chunk_duration_seconds = 30
    config.transcription.language = "auto"

    # Mock processing settings
    config.processing = MagicMock()
    config.processing.max_workers = 4
    config.processing.memory_limit_mb = 2048
    config.processing.temp_dir = "/tmp/vttiro"

    # Mock output settings
    config.output = MagicMock()
    config.output.format = "webvtt"
    config.output.include_metadata = True
    config.output.include_timestamps = True

    return config


@pytest.fixture
def mock_transcription_result():
    """Create a mock TranscriptionResult for testing."""
    return TranscriptionResult(
        text="This is a test transcription.",
        confidence=0.95,
        language="en",
        start_time=0.0,
        end_time=5.0,
        word_timestamps=[],
        speaker_labels=[],
        emotions=[],
        metadata={"model": "test_model", "processing_time": 1.5, "correlation_id": "test-123"},
    )


@pytest.fixture
def mock_video_metadata():
    """Create mock video metadata for testing."""
    return MagicMock(
        title="Test Video",
        description="A test video for unit testing",
        uploader="Test User",
        duration_seconds=120.0,
        url="https://example.com/test-video",
        thumbnail_url="https://example.com/thumbnail.jpg",
        upload_date="2024-01-01",
        view_count=1000,
        like_count=50,
        tags=["test", "video", "transcription"],
    )


@pytest.fixture
def mock_audio_chunks():
    """Create mock audio chunks for testing."""
    chunks = []
    for i in range(4):  # 4 chunks of 30 seconds each
        chunk = MagicMock()
        chunk.start_time = i * 30.0
        chunk.end_time = (i + 1) * 30.0
        chunk.audio_file = Path(f"/tmp/chunk_{i}.wav")
        chunk.duration = 30.0
        chunk.sample_rate = 16000
        chunks.append(chunk)
    return chunks


@pytest.fixture
def mock_video_processor():
    """Create a mock VideoProcessor for testing."""
    processor = AsyncMock()

    # Mock successful processing result
    result = MagicMock()
    result.metadata = MagicMock()
    result.metadata.title = "Test Video"
    result.metadata.duration_seconds = 120.0
    result.segments = []

    processor.process_source.return_value = result
    return processor


@pytest.fixture
def mock_transcription_engine():
    """Create a mock TranscriptionEngine for testing."""
    engine = AsyncMock()
    engine.name = "test_engine"
    engine.model_name = "test-model-v1"
    engine.supported_languages = ["en", "es", "fr", "de"]

    # Mock successful transcription
    engine.transcribe.return_value = TranscriptionResult(
        text="Test transcription result", confidence=0.9, language="en", start_time=0.0, end_time=5.0
    )

    engine.get_supported_languages.return_value = ["en", "es", "fr", "de"]
    engine.estimate_cost.return_value = 0.01

    return engine


@pytest.fixture
def mock_transcription_ensemble():
    """Create a mock TranscriptionEnsemble for testing."""
    ensemble = AsyncMock()

    # Mock successful transcription
    ensemble.transcribe.return_value = TranscriptionResult(
        text="Ensemble transcription result", confidence=0.95, language="en", start_time=0.0, end_time=5.0
    )

    ensemble.estimate_cost.return_value = 0.015
    return ensemble


@pytest.fixture
def sample_audio_file(temp_dir):
    """Create a sample audio file for testing."""
    audio_file = temp_dir / "test_audio.wav"
    # Create a dummy file (in real tests, you might generate actual audio)
    audio_file.write_bytes(b"dummy audio data")
    return audio_file


@pytest.fixture
def sample_video_file(temp_dir):
    """Create a sample video file for testing."""
    video_file = temp_dir / "test_video.mp4"
    # Create a dummy file (in real tests, you might generate actual video)
    video_file.write_bytes(b"dummy video data")
    return video_file


@pytest.fixture
def sample_webvtt_content():
    """Create sample WebVTT content for testing."""
    return """WEBVTT

NOTE Title: Test Video
NOTE Duration: 60.0s

00:00:00.000 --> 00:00:05.000
This is the first subtitle.

00:00:05.000 --> 00:00:10.000
This is the second subtitle.

00:00:10.000 --> 00:00:15.000
This is the third subtitle.
"""


@pytest.fixture
def api_key_config():
    """Create configuration with mock API keys for testing."""
    return {
        "gemini_api_key": "test_gemini_key_123",
        "assemblyai_api_key": "test_assemblyai_key_456",
        "deepgram_api_key": "test_deepgram_key_789",
    }


@pytest.fixture
def mock_network_responses():
    """Create mock network responses for API testing."""
    return {
        "gemini": {
            "success": {"candidates": [{"content": {"parts": [{"text": "Test transcription from Gemini"}]}}]},
            "error": {"error": {"code": 429, "message": "Rate limit exceeded", "status": "RESOURCE_EXHAUSTED"}},
        },
        "assemblyai": {
            "success": {
                "id": "test-123",
                "status": "completed",
                "text": "Test transcription from AssemblyAI",
                "confidence": 0.92,
                "words": [],
            },
            "error": {"error": "Insufficient funds"},
        },
        "deepgram": {
            "success": {
                "results": {
                    "channels": [
                        {"alternatives": [{"transcript": "Test transcription from Deepgram", "confidence": 0.88}]}
                    ]
                }
            },
            "error": {"err_code": "INVALID_AUTH", "err_msg": "Invalid authentication"},
        },
    }


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch, temp_dir):
    """Set up test environment with isolated temporary directories."""
    # Set environment variables for testing
    monkeypatch.setenv("VTTIRO_TEST_MODE", "true")
    monkeypatch.setenv("VTTIRO_TEMP_DIR", str(temp_dir))
    monkeypatch.setenv("VTTIRO_LOG_LEVEL", "DEBUG")

    # Ensure clean state for each test

    # Cleanup is handled by temp_dir fixture


class MockAPIResponse:
    """Mock API response for testing network operations."""

    def __init__(self, json_data: dict[str, Any], status_code: int = 200):
        self.json_data = json_data
        self.status_code = status_code
        self.text = str(json_data)
        self.headers = {"Content-Type": "application/json"}

    def json(self):
        return self.json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            msg = f"HTTP {self.status_code}"
            raise Exception(msg)


@pytest.fixture
def mock_api_response_factory():
    """Factory for creating mock API responses."""
    return MockAPIResponse


# Performance testing utilities
class PerformanceCollector:
    """Collect performance metrics during tests."""

    def __init__(self):
        self.metrics = {}

    def record(self, name: str, value: float, unit: str = "seconds"):
        """Record a performance metric."""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append({"value": value, "unit": unit})

    def get_average(self, name: str) -> float | None:
        """Get average value for a metric."""
        if name not in self.metrics:
            return None
        values = [m["value"] for m in self.metrics[name]]
        return sum(values) / len(values) if values else None


@pytest.fixture
def performance_collector():
    """Create a performance metrics collector."""
    return PerformanceCollector()


# Helper functions for test data generation
def generate_test_transcription_results(count: int = 5) -> list:
    """Generate test transcription results."""
    results = []
    for i in range(count):
        result = TranscriptionResult(
            text=f"Test transcription segment {i + 1}",
            confidence=0.8 + (i * 0.02),  # Varying confidence
            language="en",
            start_time=i * 30.0,
            end_time=(i + 1) * 30.0,
            metadata={"segment_id": i + 1},
        )
        results.append(result)
    return results


def generate_test_error_scenarios():
    """Generate common error scenarios for testing."""
    return [
        {"type": "network", "message": "Connection timeout"},
        {"type": "api", "message": "Rate limit exceeded"},
        {"type": "auth", "message": "Invalid API key"},
        {"type": "processing", "message": "Unsupported file format"},
        {"type": "validation", "message": "Invalid input parameters"},
    ]
