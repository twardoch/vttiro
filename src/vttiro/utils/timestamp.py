# this_file: src/vttiro/utils/timestamp.py
"""Timestamp parsing and formatting utilities.

This module provides functions for parsing and formatting timestamps
in various formats commonly used in transcription systems, including
WebVTT timestamps and other time representations.

Used by:
- Provider implementations for parsing API responses
- Output formatters for generating timed subtitles
- Processing modules for temporal analysis
"""

import re
from typing import Any

try:
    from loguru import logger
except ImportError:
    import logging as logger


def parse_timestamp(timestamp_str: str) -> float:
    """Parse WebVTT-style timestamp to seconds.
    
    Handles various timestamp formats including malformed ones sometimes
    returned by AI models:
    - Standard: HH:MM:SS.mmm (e.g., "00:05:30.250")
    - Short: MM:SS.mmm (e.g., "05:30.250")
    - Malformed: HH:MM:SSS (e.g., "00:05:700" -> interpreted as 5.7 seconds)
    
    Args:
        timestamp_str: Timestamp string in various formats
        
    Returns:
        Time in seconds as float
        
    Raises:
        ValueError: If timestamp format is completely invalid
        
    Examples:
        >>> parse_timestamp("00:05:30.250")
        330.25
        >>> parse_timestamp("05:30.250")
        330.25
        >>> parse_timestamp("00:05:700")  # Malformed but handled
        5.7
    """
    try:
        # Clean up the timestamp string
        timestamp_str = timestamp_str.strip()
        
        # Split by colon to handle different formats
        parts = timestamp_str.split(':')
        
        if len(parts) == 3:
            # HH:MM:SS.mmm or HH:MM:SSS format
            hours = int(parts[0])
            minutes = int(parts[1])
            
            # Handle seconds part - could be SS.mmm or SSS (malformed)
            seconds_part = parts[2]
            if '.' in seconds_part:
                # Standard format: SS.mmm
                sec_parts = seconds_part.split('.')
                seconds = int(sec_parts[0])
                # Handle milliseconds of varying length (1-3 digits)
                ms_str = sec_parts[1].ljust(3, '0')[:3]  # Pad or truncate to 3 digits
                milliseconds = int(ms_str)
            else:
                # Malformed format: handle as special case for Gemini-style timestamps
                raw_number = int(seconds_part)
                if raw_number >= 100:
                    # For numbers like 700, interpret as tenths: 700 -> 0.7 seconds
                    # This handles cases like "00:05:700" -> 5.7 total seconds
                    seconds = raw_number // 100  # First digit(s) as seconds
                    milliseconds = (raw_number % 100) * 10  # Remaining as milliseconds
                else:
                    # Small number, treat as seconds
                    seconds = raw_number
                    milliseconds = 0
                    
        elif len(parts) == 2:
            # MM:SS.mmm or MM:SSS format
            hours = 0
            minutes = int(parts[0])
            seconds_part = parts[1]
            
            if '.' in seconds_part:
                sec_parts = seconds_part.split('.')
                seconds = int(sec_parts[0])
                ms_str = sec_parts[1].ljust(3, '0')[:3]
                milliseconds = int(ms_str)
            else:
                # Handle MM:SSS format
                raw_number = int(seconds_part)
                if raw_number >= 100:
                    seconds = raw_number // 100
                    milliseconds = (raw_number % 100) * 10
                else:
                    seconds = raw_number
                    milliseconds = 0
                    
        elif len(parts) == 1:
            # Pure seconds format: "30.5" or "30"
            if '.' in timestamp_str:
                sec_parts = timestamp_str.split('.')
                hours = 0
                minutes = 0
                seconds = int(sec_parts[0])
                ms_str = sec_parts[1].ljust(3, '0')[:3]
                milliseconds = int(ms_str)
            else:
                hours = 0
                minutes = 0
                seconds = int(timestamp_str)
                milliseconds = 0
        else:
            raise ValueError(f"Unexpected timestamp format: '{timestamp_str}'")
        
        # Calculate total seconds
        total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0
        
        logger.debug(f"Parsed timestamp '{timestamp_str}' -> {total_seconds:.3f}s")
        return total_seconds
        
    except (ValueError, IndexError) as e:
        try:
            logger.warning(f"Failed to parse timestamp '{timestamp_str}': {e}")
        except:
            pass  # Ignore logging errors
        raise ValueError(f"Invalid timestamp format: '{timestamp_str}'") from e


def format_timestamp(seconds: float, format_type: str = "webvtt") -> str:
    """Format seconds as timestamp string.
    
    Args:
        seconds: Time in seconds
        format_type: Output format ("webvtt", "srt", "simple")
        
    Returns:
        Formatted timestamp string
        
    Examples:
        >>> format_timestamp(330.25, "webvtt")
        "00:05:30.250"
        >>> format_timestamp(330.25, "simple")
        "5:30.25"
    """
    if seconds < 0:
        raise ValueError("Timestamp cannot be negative")
    
    # Extract components
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    
    if format_type == "webvtt":
        # WebVTT format: HH:MM:SS.mmm
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    elif format_type == "srt":
        # SRT format: HH:MM:SS,mmm (note comma for milliseconds)
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace('.', ',')
    elif format_type == "simple":
        # Simple format: omit hours if zero
        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:04.1f}"
        else:
            return f"{minutes}:{secs:04.1f}"
    else:
        raise ValueError(f"Unknown format type: {format_type}")


def parse_webvtt_timestamp_line(line: str) -> tuple[float, float] | None:
    """Parse a WebVTT timestamp line (cue timing).
    
    Args:
        line: Line containing timestamp range (e.g., "00:05:30.250 --> 00:05:35.500")
        
    Returns:
        Tuple of (start_seconds, end_seconds) or None if not a timestamp line
        
    Examples:
        >>> parse_webvtt_timestamp_line("00:05:30.250 --> 00:05:35.500")
        (330.25, 335.5)
        >>> parse_webvtt_timestamp_line("This is not a timestamp")
        None
    """
    # Match WebVTT timestamp format with flexible arrow
    timestamp_pattern = r'([\d:\.]+)\s*-->\s*([\d:\.]+)'
    match = re.match(timestamp_pattern, line.strip())
    
    if not match:
        return None
    
    start_str, end_str = match.groups()
    
    try:
        start_seconds = parse_timestamp(start_str)
        end_seconds = parse_timestamp(end_str)
        
        if end_seconds <= start_seconds:
            logger.warning(f"Invalid timestamp range: {start_str} --> {end_str}")
            # Fix by adding minimal duration
            end_seconds = start_seconds + 0.1
        
        return start_seconds, end_seconds
        
    except ValueError as e:
        logger.warning(f"Failed to parse timestamp line '{line}': {e}")
        return None


def distribute_words_over_duration(
    words: list[str], 
    start_time: float, 
    end_time: float,
    base_confidence: float = 0.95
) -> list[dict[str, Any]]:
    """Distribute words evenly over a time duration.
    
    When exact word-level timing isn't available, this function creates
    estimated word timestamps by distributing words evenly across the
    available time span.
    
    Args:
        words: List of words to distribute
        start_time: Start time in seconds
        end_time: End time in seconds  
        base_confidence: Base confidence score for all words
        
    Returns:
        List of word timestamp dictionaries
        
    Examples:
        >>> words = ["Hello", "world"]
        >>> distribute_words_over_duration(words, 0.0, 2.0)
        [
            {"word": "Hello", "start": 0.0, "end": 1.0, "confidence": 0.95},
            {"word": "world", "start": 1.0, "end": 2.0, "confidence": 0.95}
        ]
    """
    if not words:
        return []
    
    if end_time <= start_time:
        logger.warning(f"Invalid time range: {start_time} to {end_time}")
        end_time = start_time + len(words) * 0.5  # Fallback: 0.5s per word
    
    duration = end_time - start_time
    word_timestamps = []
    
    for i, word in enumerate(words):
        # Calculate start and end times for this word
        word_start = start_time + (i / len(words)) * duration
        word_end = start_time + ((i + 1) / len(words)) * duration
        
        # Clean word of punctuation for better matching
        clean_word = re.sub(r'[^\w\'-]', '', word)
        
        if clean_word:  # Only add non-empty words
            word_timestamps.append({
                "word": word,
                "start": word_start,
                "end": word_end,
                "confidence": base_confidence
            })
    
    return word_timestamps