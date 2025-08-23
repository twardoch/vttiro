#!/usr/bin/env python3
# this_file: tests/factories.py
"""Test data factories for creating consistent test objects."""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import factory
import factory.fuzzy

from vttiro.core.config import VttiroConfig
from vttiro.core.types import TranscriptionResult


class VideoMetadataFactory(factory.Factory):
    """Factory for creating video metadata objects."""

    class Meta:
        model = dict

    title = factory.Faker("sentence", nb_words=4)
    description = factory.Faker("text", max_nb_chars=200)
    uploader = factory.Faker("user_name")
    duration_seconds = factory.fuzzy.FuzzyFloat(30.0, 3600.0)
    url = factory.Faker("url")
    thumbnail_url = factory.Faker("image_url")
    upload_date = factory.Faker("date")
    view_count = factory.fuzzy.FuzzyInteger(100, 1000000)
    like_count = factory.fuzzy.FuzzyInteger(10, 50000)
    tags = factory.LazyFunction(lambda: [factory.Faker("word").generate() for _ in range(3)])


class TranscriptionResultFactory(factory.Factory):
    """Factory for creating TranscriptionResult objects."""

    class Meta:
        model = TranscriptionResult

    text = factory.Faker("paragraph", nb_sentences=3)
    confidence = factory.fuzzy.FuzzyFloat(0.7, 1.0)
    language = factory.fuzzy.FuzzyChoice(["en", "es", "fr", "de", "it", "pt"])
    start_time = factory.fuzzy.FuzzyFloat(0.0, 100.0)
    end_time = factory.LazyAttribute(lambda obj: obj.start_time + factory.fuzzy.FuzzyFloat(5.0, 60.0).fuzz())
    word_timestamps = factory.LazyFunction(list)
    speaker_labels = factory.LazyFunction(list)
    emotions = factory.LazyFunction(list)
    metadata = factory.LazyFunction(dict)


class AudioChunkFactory(factory.Factory):
    """Factory for creating AudioChunk objects."""

    class Meta:
        model = dict  # We'll create dict and convert to AudioChunk

    audio_file = factory.LazyAttribute(lambda obj: Path(f"/tmp/chunk_{factory.fuzzy.FuzzyInteger(0, 999).fuzz()}.wav"))
    start_time = factory.fuzzy.FuzzyFloat(0.0, 1000.0)
    end_time = factory.LazyAttribute(lambda obj: obj.start_time + 30.0)
    duration = factory.LazyAttribute(lambda obj: obj.end_time - obj.start_time)
    sample_rate = factory.fuzzy.FuzzyChoice([16000, 22050, 44100, 48000])


class VttiroConfigFactory(factory.Factory):
    """Factory for creating VttiroConfig objects."""

    class Meta:
        model = dict  # We'll create dict and convert to VttiroConfig

    # Transcription settings
    transcription = factory.SubFactory(
        factory.DictFactory,
        {
            "preferred_model": factory.fuzzy.FuzzyChoice(["auto", "gemini", "assemblyai", "deepgram"]),
            "language": factory.fuzzy.FuzzyChoice(["auto", "en", "es", "fr", "de"]),
            "max_duration_seconds": factory.fuzzy.FuzzyInteger(300, 7200),
            "chunk_duration_seconds": factory.fuzzy.FuzzyChoice([15, 30, 45, 60]),
            "gemini_api_key": factory.Faker("password", length=32),
            "assemblyai_api_key": factory.Faker("password", length=32),
            "deepgram_api_key": factory.Faker("password", length=32),
        },
    )

    # Processing settings
    processing = factory.SubFactory(
        factory.DictFactory,
        {
            "max_workers": factory.fuzzy.FuzzyInteger(1, 8),
            "memory_limit_mb": factory.fuzzy.FuzzyChoice([1024, 2048, 4096, 8192]),
            "temp_dir": factory.LazyFunction(lambda: f"/tmp/vttiro_{factory.fuzzy.FuzzyInteger(1000, 9999).fuzz()}"),
        },
    )

    # Output settings
    output = factory.SubFactory(
        factory.DictFactory,
        {
            "format": factory.fuzzy.FuzzyChoice(["webvtt", "srt", "ass"]),
            "include_metadata": factory.Faker("boolean"),
            "include_timestamps": factory.Faker("boolean"),
        },
    )


class ErrorScenarioFactory(factory.Factory):
    """Factory for creating error test scenarios."""

    class Meta:
        model = dict

    error_type = factory.fuzzy.FuzzyChoice(["network", "api", "auth", "rate_limit", "processing", "validation"])
    message = factory.Faker("sentence")
    status_code = factory.fuzzy.FuzzyChoice([400, 401, 403, 429, 500, 502, 503])
    retry_after = factory.fuzzy.FuzzyInteger(1, 300)
    should_retry = factory.Faker("boolean")


class APIResponseFactory(factory.Factory):
    """Factory for creating mock API responses."""

    class Meta:
        model = dict

    service = factory.fuzzy.FuzzyChoice(["gemini", "assemblyai", "deepgram"])
    status_code = factory.fuzzy.FuzzyChoice([200, 201, 400, 401, 429, 500])
    response_time = factory.fuzzy.FuzzyFloat(0.1, 5.0)
    success = factory.LazyAttribute(lambda obj: obj.status_code < 400)


class PerformanceMetricsFactory(factory.Factory):
    """Factory for creating performance test metrics."""

    class Meta:
        model = dict

    operation = factory.fuzzy.FuzzyChoice(["transcription", "video_processing", "audio_segmentation", "api_call"])
    duration_seconds = factory.fuzzy.FuzzyFloat(0.1, 60.0)
    memory_usage_mb = factory.fuzzy.FuzzyFloat(50.0, 2048.0)
    cpu_usage_percent = factory.fuzzy.FuzzyFloat(10.0, 100.0)
    file_size_mb = factory.fuzzy.FuzzyFloat(1.0, 500.0)
    throughput_ratio = factory.fuzzy.FuzzyFloat(1.0, 20.0)  # x faster than real-time


class TestFileFactory(factory.Factory):
    """Factory for creating test file information."""

    class Meta:
        model = dict

    filename = factory.Faker("file_name")
    file_path = factory.LazyAttribute(lambda obj: Path(f"/tmp/test_{obj.filename}"))
    file_size = factory.fuzzy.FuzzyInteger(1024, 100 * 1024 * 1024)  # 1KB to 100MB
    file_type = factory.fuzzy.FuzzyChoice(["mp4", "wav", "mp3", "avi", "mkv"])
    duration = factory.fuzzy.FuzzyFloat(10.0, 3600.0)
    sample_rate = factory.fuzzy.FuzzyChoice([16000, 22050, 44100, 48000])
    channels = factory.fuzzy.FuzzyChoice([1, 2])


# Trait classes for specialized factories
class LargeFileFactory(TestFileFactory):
    """Factory for large test files."""

    file_size = factory.fuzzy.FuzzyInteger(100 * 1024 * 1024, 1024 * 1024 * 1024)  # 100MB to 1GB
    duration = factory.fuzzy.FuzzyFloat(1800.0, 7200.0)  # 30 minutes to 2 hours


class SmallFileFactory(TestFileFactory):
    """Factory for small test files."""

    file_size = factory.fuzzy.FuzzyInteger(1024, 10 * 1024 * 1024)  # 1KB to 10MB
    duration = factory.fuzzy.FuzzyFloat(10.0, 300.0)  # 10 seconds to 5 minutes


class HighQualityTranscriptionFactory(TranscriptionResultFactory):
    """Factory for high-quality transcription results."""

    confidence = factory.fuzzy.FuzzyFloat(0.9, 1.0)
    text = factory.Faker("paragraph", nb_sentences=5)


class LowQualityTranscriptionFactory(TranscriptionResultFactory):
    """Factory for low-quality transcription results."""

    confidence = factory.fuzzy.FuzzyFloat(0.5, 0.7)
    text = factory.Faker("sentence", nb_words=3)


class MultilingualTranscriptionFactory(TranscriptionResultFactory):
    """Factory for multilingual transcription scenarios."""

    language = factory.fuzzy.FuzzyChoice(["en", "es", "fr", "de", "it", "pt", "ja", "ko", "zh"])
    text = factory.LazyAttribute(lambda obj: f"[{obj.language}] " + factory.Faker("paragraph").generate())


class BatchProcessingScenarioFactory(factory.Factory):
    """Factory for batch processing test scenarios."""

    class Meta:
        model = dict

    source_count = factory.fuzzy.FuzzyInteger(2, 20)
    sources = factory.LazyAttribute(lambda obj: [f"source_{i}.mp4" for i in range(obj.source_count)])
    expected_success_rate = factory.fuzzy.FuzzyFloat(0.8, 1.0)
    max_processing_time = factory.fuzzy.FuzzyInteger(60, 600)


# Resilience scenario factories removed for simplification


# Utility functions for creating common test scenarios
def create_successful_transcription_batch(count: int = 5) -> list[TranscriptionResult]:
    """Create a batch of successful transcription results."""
    return [HighQualityTranscriptionFactory() for _ in range(count)]


def create_mixed_quality_transcription_batch(count: int = 10) -> list[TranscriptionResult]:
    """Create a batch with mixed quality transcription results."""
    results = []
    for i in range(count):
        if i % 3 == 0:
            results.append(LowQualityTranscriptionFactory())
        else:
            results.append(HighQualityTranscriptionFactory())
    return results


def create_error_cascade_scenario() -> list[dict[str, Any]]:
    """Create a scenario with cascading errors for resilience testing."""
    return [ErrorScenarioFactory(error_type="network", should_retry=True, retry_after=1) for _ in range(3)] + [
        ErrorScenarioFactory(error_type="api", should_retry=False, status_code=500)
    ]


def create_performance_test_files() -> list[dict[str, Any]]:
    """Create a variety of files for performance testing."""
    return [SmallFileFactory(), SmallFileFactory(), TestFileFactory(), TestFileFactory(), LargeFileFactory()]


def create_multilingual_test_batch() -> list[TranscriptionResult]:
    """Create multilingual transcription results for testing."""
    languages = ["en", "es", "fr", "de", "ja", "ko"]
    return [MultilingualTranscriptionFactory(language=lang) for lang in languages]


# Factory registry for easy access
FACTORIES = {
    "video_metadata": VideoMetadataFactory,
    "transcription_result": TranscriptionResultFactory,
    "audio_chunk": AudioChunkFactory,
    "vttiro_config": VttiroConfigFactory,
    "error_scenario": ErrorScenarioFactory,
    "api_response": APIResponseFactory,
    "performance_metrics": PerformanceMetricsFactory,
    "test_file": TestFileFactory,
    "large_file": LargeFileFactory,
    "small_file": SmallFileFactory,
    "high_quality_transcription": HighQualityTranscriptionFactory,
    "low_quality_transcription": LowQualityTranscriptionFactory,
    "multilingual_transcription": MultilingualTranscriptionFactory,
    "batch_scenario": BatchProcessingScenarioFactory,
    # Resilience scenario factories removed for simplification
}


def get_factory(name: str) -> factory.Factory:
    """Get a factory by name.

    Args:
        name: Name of the factory

    Returns:
        Factory class

    Raises:
        KeyError: If factory name is not found
    """
    if name not in FACTORIES:
        msg = f"Unknown factory: {name}. Available: {list(FACTORIES.keys())}"
        raise KeyError(msg)
    return FACTORIES[name]


def create_test_data(factory_name: str, count: int = 1, **kwargs) -> Any:
    """Create test data using a named factory.

    Args:
        factory_name: Name of the factory to use
        count: Number of objects to create
        **kwargs: Additional arguments to pass to the factory

    Returns:
        Single object if count=1, list of objects otherwise
    """
    factory_class = get_factory(factory_name)

    if count == 1:
        return factory_class(**kwargs)
    return [factory_class(**kwargs) for _ in range(count)]
