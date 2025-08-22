# this_file: src/vttiro/tests/test_types.py
"""Unit tests for core data types.

Tests for TranscriptSegment and TranscriptionResult dataclasses,
including validation, edge cases, and helper methods.
"""

import pytest

from ..core.types import TranscriptSegment, TranscriptionResult


class TestTranscriptSegment:
    """Test TranscriptSegment dataclass."""
    
    def test_valid_segment_creation(self):
        """Test creating valid segment."""
        segment = TranscriptSegment(
            start=1.5,
            end=5.0,
            text="Hello world",
            speaker="Speaker 1",
            confidence=0.95
        )
        
        assert segment.start == 1.5
        assert segment.end == 5.0
        assert segment.text == "Hello world"
        assert segment.speaker == "Speaker 1"
        assert segment.confidence == 0.95
    
    def test_minimal_segment_creation(self):
        """Test creating segment with only required fields."""
        segment = TranscriptSegment(
            start=0.0,
            end=3.0,
            text="Test"
        )
        
        assert segment.start == 0.0
        assert segment.end == 3.0
        assert segment.text == "Test"
        assert segment.speaker is None
        assert segment.confidence is None
    
    def test_duration_calculation(self):
        """Test duration helper method."""
        segment = TranscriptSegment(start=2.0, end=7.5, text="Test")
        assert segment.duration() == 5.5
    
    def test_negative_start_time_raises_error(self):
        """Test that negative start time raises ValueError."""
        with pytest.raises(ValueError, match="Start time cannot be negative"):
            TranscriptSegment(start=-1.0, end=5.0, text="Test")
    
    def test_end_before_start_raises_error(self):
        """Test that end time before start time raises ValueError."""
        with pytest.raises(ValueError, match="End time .* cannot be before start time"):
            TranscriptSegment(start=5.0, end=3.0, text="Test")
    
    def test_empty_text_raises_error(self):
        """Test that empty text raises ValueError."""
        with pytest.raises(ValueError, match="Segment text cannot be empty"):
            TranscriptSegment(start=0.0, end=5.0, text="   ")
    
    def test_invalid_confidence_raises_error(self):
        """Test that confidence outside 0-1 range raises ValueError."""
        with pytest.raises(ValueError, match="Confidence score must be between"):
            TranscriptSegment(start=0.0, end=5.0, text="Test", confidence=1.5)
        
        with pytest.raises(ValueError, match="Confidence score must be between"):
            TranscriptSegment(start=0.0, end=5.0, text="Test", confidence=-0.1)


class TestTranscriptionResult:
    """Test TranscriptionResult dataclass."""
    
    def test_valid_result_creation(self):
        """Test creating valid transcription result."""
        segments = [
            TranscriptSegment(start=0.0, end=3.0, text="Hello"),
            TranscriptSegment(start=3.0, end=6.0, text="world")
        ]
        
        result = TranscriptionResult(
            segments=segments,
            metadata={"model": "test", "quality": "high"},
            provider="test_provider",
            language="en",
            confidence=0.92
        )
        
        assert len(result.segments) == 2
        assert result.metadata["model"] == "test"
        assert result.provider == "test_provider"
        assert result.language == "en"
        assert result.confidence == 0.92
    
    def test_minimal_result_creation(self):
        """Test creating result with only required fields."""
        segments = [TranscriptSegment(start=0.0, end=3.0, text="Test")]
        
        result = TranscriptionResult(
            segments=segments,
            metadata={},
            provider="test"
        )
        
        assert len(result.segments) == 1
        assert result.metadata == {}
        assert result.provider == "test"
        assert result.language is None
        assert result.confidence is None
    
    def test_duration_calculation(self):
        """Test duration helper method."""
        segments = [
            TranscriptSegment(start=0.0, end=3.0, text="One"),
            TranscriptSegment(start=3.0, end=8.5, text="Two")
        ]
        
        result = TranscriptionResult(
            segments=segments,
            metadata={},
            provider="test"
        )
        
        assert result.duration() == 8.5
    
    def test_empty_segments_duration(self):
        """Test duration calculation with empty segments."""
        result = TranscriptionResult(
            segments=[],
            metadata={},
            provider="test"
        )
        
        assert result.duration() == 0.0
    
    def test_word_count_calculation(self):
        """Test word count helper method."""
        segments = [
            TranscriptSegment(start=0.0, end=3.0, text="Hello world"),
            TranscriptSegment(start=3.0, end=6.0, text="This is a test")
        ]
        
        result = TranscriptionResult(
            segments=segments,
            metadata={},
            provider="test"
        )
        
        assert result.word_count() == 6  # "Hello world" (2) + "This is a test" (4)
    
    def test_invalid_segments_type_raises_error(self):
        """Test that non-list segments raises TypeError."""
        with pytest.raises(TypeError, match="Segments must be a list"):
            TranscriptionResult(
                segments="not a list",
                metadata={},
                provider="test"
            )
    
    def test_invalid_metadata_type_raises_error(self):
        """Test that non-dict metadata raises TypeError."""
        with pytest.raises(TypeError, match="Metadata must be a dictionary"):
            TranscriptionResult(
                segments=[],
                metadata="not a dict",
                provider="test"
            )
    
    def test_empty_provider_raises_error(self):
        """Test that empty provider name raises ValueError."""
        with pytest.raises(ValueError, match="Provider name cannot be empty"):
            TranscriptionResult(
                segments=[],
                metadata={},
                provider=""
            )
    
    def test_invalid_confidence_raises_error(self):
        """Test that confidence outside 0-1 range raises ValueError."""
        with pytest.raises(ValueError, match="Overall confidence must be between"):
            TranscriptionResult(
                segments=[],
                metadata={},
                provider="test",
                confidence=1.5
            )
    
    def test_overlapping_segments_raises_error(self):
        """Test that overlapping segments raise ValueError."""
        segments = [
            TranscriptSegment(start=0.0, end=5.0, text="First"),
            TranscriptSegment(start=3.0, end=8.0, text="Second")  # Overlaps with first
        ]
        
        with pytest.raises(ValueError, match="Segment .* overlaps with previous segment"):
            TranscriptionResult(
                segments=segments,
                metadata={},
                provider="test"
            )