#!/usr/bin/env python3
# this_file: tests/test_property_based.py
"""Property-based tests using Hypothesis for edge case discovery."""

import pytest
from hypothesis import given, strategies as st, settings, assume, example
from hypothesis.stateful import RuleBasedStateMachine, Bundle, rule, initialize, invariant
import string
import tempfile
from pathlib import Path

from vttiro.core.config import VttiroConfig, TranscriptionResult
from vttiro.utils.exceptions import VttiroError, ValidationError
# Resilience framework imports removed for simplification


# Custom strategies for vttiro-specific data types
@st.composite
def transcription_text(draw):
    """Generate realistic transcription text."""
    # Common words and phrases found in transcriptions
    words = draw(st.lists(
        st.text(
            alphabet=string.ascii_letters + string.digits + " .,!?",
            min_size=1,
            max_size=20
        ).filter(lambda x: x.strip()),
        min_size=1,
        max_size=100
    ))
    return " ".join(words)


@st.composite
def confidence_score(draw):
    """Generate realistic confidence scores."""
    return draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))


@st.composite
def timestamp(draw):
    """Generate realistic timestamp values."""
    return draw(st.floats(min_value=0.0, max_value=86400.0, allow_nan=False, allow_infinity=False))


@st.composite
def language_code(draw):
    """Generate realistic language codes."""
    return draw(st.sampled_from([
        "en", "es", "fr", "de", "it", "pt", "ja", "ko", "zh", "ar", "hi", "ru"
    ]))


@st.composite
def transcription_result(draw):
    """Generate TranscriptionResult instances."""
    text = draw(transcription_text())
    confidence = draw(confidence_score())
    language = draw(language_code())
    start_time = draw(timestamp())
    end_time = draw(st.floats(min_value=start_time, max_value=start_time + 3600.0))
    
    return TranscriptionResult(
        text=text,
        confidence=confidence,
        language=language,
        start_time=start_time,
        end_time=end_time
    )


class TestTranscriptionResultProperties:
    """Property-based tests for TranscriptionResult."""
    
    @given(transcription_result())
    def test_transcription_result_invariants(self, result):
        """Test invariants that should always hold for TranscriptionResult."""
        # Text should not be None or empty after creation
        assert result.text is not None
        assert isinstance(result.text, str)
        
        # Confidence should be in valid range
        assert 0.0 <= result.confidence <= 1.0
        
        # Time ordering should be correct
        assert result.start_time <= result.end_time
        
        # Duration should be non-negative
        assert result.duration >= 0.0
        
        # Language should be a valid string
        assert isinstance(result.language, str)
        assert len(result.language) >= 2  # Minimum language code length
    
    @given(
        text=transcription_text(),
        confidence=confidence_score(),
        language=language_code(),
        start_time=timestamp()
    )
    def test_transcription_result_serialization_roundtrip(self, text, confidence, language, start_time):
        """Test that serialization and deserialization preserve data."""
        end_time = start_time + 30.0  # Add 30 seconds
        
        original = TranscriptionResult(
            text=text,
            confidence=confidence,
            language=language,
            start_time=start_time,
            end_time=end_time
        )
        
        # Serialize to dict and back
        serialized = original.to_dict()
        restored = TranscriptionResult.from_dict(serialized)
        
        # All fields should be preserved
        assert restored.text == original.text
        assert abs(restored.confidence - original.confidence) < 1e-10
        assert restored.language == original.language
        assert abs(restored.start_time - original.start_time) < 1e-10
        assert abs(restored.end_time - original.end_time) < 1e-10
    
    @given(st.lists(transcription_result(), min_size=1, max_size=20))
    def test_transcription_results_sorting(self, results):
        """Test that transcription results can be sorted by time."""
        # Sort by start time
        sorted_results = sorted(results, key=lambda r: r.start_time)
        
        # Verify sorting is correct
        for i in range(1, len(sorted_results)):
            assert sorted_results[i-1].start_time <= sorted_results[i].start_time
    
    @given(
        transcription_text(),
        confidence_score(),
        language_code()
    )
    @example("", 0.5, "en")  # Edge case: empty text
    @example("a" * 10000, 1.0, "en")  # Edge case: very long text
    def test_transcription_result_edge_cases(self, text, confidence, language):
        """Test edge cases for TranscriptionResult creation."""
        # Should handle various text lengths and content
        result = TranscriptionResult(
            text=text,
            confidence=confidence,
            language=language
        )
        
        assert result.text == text
        assert result.confidence == confidence
        assert result.language == language


class TestConfigurationProperties:
    """Property-based tests for configuration handling."""
    
    @given(
        st.integers(min_value=1, max_value=86400),  # max_duration_seconds
        st.integers(min_value=1, max_value=300),    # chunk_duration_seconds
        st.integers(min_value=1, max_value=16),     # max_workers
        st.integers(min_value=512, max_value=8192)  # memory_limit_mb
    )
    def test_config_validation_properties(self, max_duration, chunk_duration, max_workers, memory_limit):
        """Test configuration validation with various valid inputs."""
        config = VttiroConfig()
        config.transcription.max_duration_seconds = max_duration
        config.transcription.chunk_duration_seconds = chunk_duration
        config.processing.max_workers = max_workers
        config.processing.memory_limit_mb = memory_limit
        
        # Should not raise validation errors for valid inputs
        config.validate()
        
        # Properties should maintain their values
        assert config.transcription.max_duration_seconds == max_duration
        assert config.transcription.chunk_duration_seconds == chunk_duration
        assert config.processing.max_workers == max_workers
        assert config.processing.memory_limit_mb == memory_limit
    
    @given(
        st.text(alphabet=string.ascii_letters + string.digits + "_-", min_size=8, max_size=64)
    )
    def test_api_key_handling_properties(self, api_key):
        """Test API key handling with various key formats."""
        config = VttiroConfig()
        config.transcription.gemini_api_key = api_key
        
        # API key should be stored
        assert config.transcription.gemini_api_key == api_key
        
        # Should be masked in serialization
        config_dict = config.to_dict()
        assert config_dict["transcription"]["gemini_api_key"] == "***masked***"
        
        # Should detect as having valid keys
        assert config.has_valid_api_keys()
    
    @given(
        st.dictionaries(
            st.text(alphabet=string.ascii_lowercase + "_", min_size=1, max_size=20),
            st.one_of(
                st.text(min_size=0, max_size=100),
                st.integers(min_value=0, max_value=10000),
                st.floats(min_value=0.0, max_value=100.0, allow_nan=False),
                st.booleans()
            ),
            min_size=0,
            max_size=10
        )
    )
    def test_config_metadata_handling(self, metadata):
        """Test configuration metadata handling with arbitrary data."""
        config = VttiroConfig()
        
        # Should handle arbitrary metadata
        for key, value in metadata.items():
            setattr(config.transcription, key, value)
        
        # Should serialize without errors
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)


class TestErrorHandlingProperties:
    """Property-based tests for error handling framework."""
    
    @given(
        st.text(min_size=1, max_size=1000),
        st.text(alphabet=string.ascii_uppercase + "_", min_size=5, max_size=50).filter(
            lambda x: x.isidentifier()
        )
    )
    def test_error_creation_properties(self, message, error_code):
        """Test error creation with various messages and codes."""
        error = VttiroError(message=message, error_code=error_code)
        
        # Basic properties should be preserved
        assert error.message == message
        assert error.error_code == error_code
        assert error.correlation_id is not None
        assert len(error.correlation_id) > 0
        
        # String representation should contain message
        assert message in str(error)
        
        # Serialization should work
        error_dict = error.to_dict()
        assert error_dict["message"] == message
        assert error_dict["error_code"] == error_code
    
    # Resilience framework tests removed for simplification


class TestStatefulTranscriberBehavior(RuleBasedStateMachine):
    """Stateful testing for Transcriber behavior."""
    
    sources = Bundle("sources")
    configs = Bundle("configs")
    
    @initialize()
    def init_state(self):
        """Initialize the state machine."""
        self.transcription_count = 0
        self.errors_encountered = []
    
    @rule(target=configs)
    def create_config(self):
        """Create a configuration."""
        config = VttiroConfig()
        return config
    
    @rule(target=sources, 
          duration=st.integers(min_value=30, max_value=3600))
    def create_source(self, duration):
        """Create a mock source with given duration."""
        return f"test_source_{duration}s.mp4"
    
    @rule(config=configs, source=sources)
    def attempt_transcription(self, config, source):
        """Attempt to transcribe a source with given config."""
        try:
            # Mock transcription attempt
            self.transcription_count += 1
            
            # Simulate occasional failures
            if self.transcription_count % 7 == 0:  # Fail every 7th attempt
                raise ValidationError("Simulated validation error")
                
        except Exception as e:
            self.errors_encountered.append(type(e).__name__)
    
    @invariant()
    def transcription_count_non_negative(self):
        """Transcription count should never be negative."""
        assert self.transcription_count >= 0
    
    @invariant()
    def error_types_are_valid(self):
        """All encountered errors should be known types."""
        valid_error_types = [
            "ValidationError", "ConfigurationError", "TranscriptionError",
            "ProcessingError", "ModelError", "APIError"
        ]
        
        for error_type in self.errors_encountered:
            assert error_type in valid_error_types


class TestFileSystemProperties:
    """Property-based tests for file system operations."""
    
    @given(
        st.text(
            alphabet=string.ascii_letters + string.digits + "-_",
            min_size=1,
            max_size=50
        ).filter(lambda x: x not in [".", "..", "CON", "PRN", "AUX", "NUL"])  # Avoid reserved names
    )
    def test_filename_sanitization_properties(self, filename):
        """Test filename sanitization with various inputs."""
        from vttiro.core.transcriber import Transcriber
        
        # Create a mock transcriber to access sanitization method
        transcriber = Transcriber.__new__(Transcriber)
        
        sanitized = transcriber._sanitize_filename(filename)
        
        # Sanitized filename should be safe
        assert isinstance(sanitized, str)
        assert len(sanitized) > 0
        assert sanitized != "."
        assert sanitized != ".."
        
        # Should not contain dangerous characters
        dangerous_chars = '<>:"/\\|?*'
        for char in dangerous_chars:
            assert char not in sanitized
    
    @given(
        st.lists(
            st.text(min_size=1, max_size=100),
            min_size=1,
            max_size=20
        )
    )
    def test_webvtt_generation_properties(self, text_segments):
        """Test WebVTT generation with various text inputs."""
        from vttiro.core.transcriber import Transcriber
        
        # Create mock transcriber
        transcriber = Transcriber.__new__(Transcriber)
        
        # Create transcription results from text segments
        results = []
        for i, text in enumerate(text_segments):
            result = TranscriptionResult(
                text=text,
                confidence=0.9,
                language="en",
                start_time=i * 30.0,
                end_time=(i + 1) * 30.0
            )
            results.append(result)
        
        # Generate WebVTT
        mock_metadata = type('MockMetadata', (), {
            'title': 'Test Video',
            'duration_seconds': len(text_segments) * 30.0,
            'uploader': 'Test User'
        })()
        
        webvtt_content = transcriber._generate_comprehensive_webvtt(
            results, mock_metadata, include_metadata=True
        )
        
        # WebVTT should be valid
        assert isinstance(webvtt_content, str)
        assert webvtt_content.startswith("WEBVTT")
        
        # Should contain all text segments
        for text in text_segments:
            assert text in webvtt_content


# Test runner for property-based tests
TestStatefulTranscriberBehaviorTest = TestStatefulTranscriberBehavior.TestCase


if __name__ == "__main__":
    # Run property-based tests with custom settings
    pytest.main([
        __file__,
        "--hypothesis-show-statistics",
        "--hypothesis-verbosity=verbose"
    ])