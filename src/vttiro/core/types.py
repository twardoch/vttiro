# this_file: src/vttiro/core/types.py
"""Core data types for VTTiro transcription results.

This module defines the primary data structures used throughout VTTiro 2.0
for representing transcription results, segments, and related metadata.

These types serve as the contract between providers, processing modules,
and output formatters, ensuring consistent data flow across the system.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class TranscriptSegment:
    """A single segment of transcribed text with timing information.

    Represents a contiguous piece of audio that has been transcribed,
    with precise start and end timestamps. May include optional metadata
    like speaker identification and confidence scores.

    Used by:
    - All provider implementations to return segmented results
    - Output formatters (WebVTT, SRT) for timing-based formatting
    - Processing modules for segment-level analysis
    """

    start: float  # Start time in seconds from beginning of audio
    end: float  # End time in seconds from beginning of audio
    text: str  # Transcribed text content for this segment
    speaker: str | None = None  # Speaker identifier (e.g., "Speaker 1", "John")
    confidence: float | None = None  # Confidence score 0.0-1.0 for this segment

    def duration(self) -> float:
        """Calculate segment duration in seconds."""
        return self.end - self.start

    def __post_init__(self) -> None:
        """Validate segment timing and content."""
        if self.start < 0:
            raise ValueError(f"Start time cannot be negative: {self.start}")
        if self.end < self.start:
            raise ValueError(f"End time {self.end} cannot be before start time {self.start}")
        if not self.text.strip():
            raise ValueError("Segment text cannot be empty")
        if self.confidence is not None and not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence score must be between 0.0 and 1.0: {self.confidence}")


@dataclass
class TranscriptionResult:
    """Complete transcription result from a provider.

    Contains all segments produced by a transcription provider, along with
    metadata about the transcription process, provider used, and overall
    quality metrics.

    Used by:
    - Provider implementations as their return type
    - Core orchestration for result aggregation
    - Output modules for format conversion
    - Monitoring for quality tracking
    """

    segments: list[TranscriptSegment]  # All transcribed segments in chronological order
    metadata: dict[str, Any]  # Provider-specific metadata and settings
    provider: str  # Provider name (e.g., "gemini", "assemblyai")
    language: str | None = None  # Detected or specified language code (ISO 639-1)
    confidence: float | None = None  # Overall confidence score 0.0-1.0 for entire result

    def duration(self) -> float:
        """Calculate total transcription duration in seconds."""
        if not self.segments:
            return 0.0
        return max(segment.end for segment in self.segments)

    def word_count(self) -> int:
        """Count total words across all segments."""
        return sum(len(segment.text.split()) for segment in self.segments)

    def __post_init__(self) -> None:
        """Validate transcription result."""
        if not isinstance(self.segments, list):
            raise TypeError("Segments must be a list")
        if not isinstance(self.metadata, dict):
            raise TypeError("Metadata must be a dictionary")
        if not self.provider:
            raise ValueError("Provider name cannot be empty")
        if self.confidence is not None and not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Overall confidence must be between 0.0 and 1.0: {self.confidence}")

        # Validate segment ordering
        for i, segment in enumerate(self.segments):
            if i > 0 and segment.start < self.segments[i - 1].end:
                raise ValueError(f"Segment {i} overlaps with previous segment")
