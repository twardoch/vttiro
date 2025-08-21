#!/usr/bin/env python3
# this_file: src/vttiro/utils/types.py
"""Shared type definitions to avoid circular imports."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SimpleTranscriptSegment:
    """Simple transcript segment for WebVTT generation.
    
    Represents a single subtitle segment with start time, end time, and text.
    """
    start_time: float  # Start time in seconds
    end_time: float    # End time in seconds  
    text: str         # Subtitle text
    speaker: Optional[str] = None  # Speaker name (optional)