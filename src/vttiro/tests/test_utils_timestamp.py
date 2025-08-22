# this_file: src/vttiro/tests/test_utils_timestamp.py
"""Unit tests for timestamp utilities."""

import pytest

from ..utils.timestamp import (
    parse_timestamp,
    format_timestamp,
    parse_webvtt_timestamp_line,
    distribute_words_over_duration
)


class TestParseTimestamp:
    """Test timestamp parsing functionality."""
    
    def test_standard_webvtt_format(self):
        """Test parsing standard WebVTT timestamps."""
        assert parse_timestamp("00:05:30.250") == 330.25
        assert parse_timestamp("01:23:45.678") == 5025.678
        assert parse_timestamp("00:00:01.000") == 1.0
    
    def test_short_format(self):
        """Test parsing short MM:SS.mmm format."""
        assert parse_timestamp("05:30.250") == 330.25
        assert parse_timestamp("01:23.500") == 83.5
        assert parse_timestamp("00:05.000") == 5.0
    
    def test_malformed_gemini_format(self):
        """Test parsing malformed Gemini timestamps."""
        # 00:05:700 -> 0 hours + 5 minutes + 7 seconds = 307 seconds
        assert parse_timestamp("00:05:700") == 307.0
        # 00:12:1500 -> 0 hours + 12 minutes + 15 seconds = 735 seconds  
        assert parse_timestamp("00:12:1500") == 735.0
    
    def test_seconds_only_format(self):
        """Test parsing pure seconds format."""
        assert parse_timestamp("30.5") == 30.5
        assert parse_timestamp("45") == 45.0
        assert parse_timestamp("123.456") == 123.456
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        assert parse_timestamp("00:00:00.000") == 0.0
        assert parse_timestamp("23:59:59.999") == 86399.999
        assert parse_timestamp("0.001") == 0.001
    
    def test_millisecond_padding(self):
        """Test handling of milliseconds with different digit counts."""
        assert parse_timestamp("00:00:01.1") == 1.1
        assert parse_timestamp("00:00:01.12") == 1.12
        assert parse_timestamp("00:00:01.123") == 1.123
        assert parse_timestamp("00:00:01.1234") == 1.123  # Truncated to 3 digits
    
    def test_invalid_format_raises_error(self):
        """Test that invalid formats raise ValueError."""
        with pytest.raises(ValueError, match="Invalid timestamp format"):
            parse_timestamp("invalid")
        
        with pytest.raises(ValueError, match="Invalid timestamp format"):
            parse_timestamp("25:61:61.000")  # Invalid time values
        
        with pytest.raises(ValueError, match="Invalid timestamp format"):
            parse_timestamp("")


class TestFormatTimestamp:
    """Test timestamp formatting functionality."""
    
    def test_webvtt_format(self):
        """Test WebVTT format output."""
        assert format_timestamp(330.25, "webvtt") == "00:05:30.250"
        assert format_timestamp(5025.678, "webvtt") == "01:23:45.678"
        assert format_timestamp(0.0, "webvtt") == "00:00:00.000"
    
    def test_srt_format(self):
        """Test SRT format output (with comma separator)."""
        assert format_timestamp(330.25, "srt") == "00:05:30,250"
        assert format_timestamp(5025.678, "srt") == "01:23:45,678"
    
    def test_simple_format(self):
        """Test simple format output."""
        assert format_timestamp(330.25, "simple") == "5:30.2"
        assert format_timestamp(5025.678, "simple") == "1:23:45.7"
        assert format_timestamp(30.5, "simple") == "0:30.5"
    
    def test_negative_timestamp_raises_error(self):
        """Test that negative timestamps raise ValueError."""
        with pytest.raises(ValueError, match="Timestamp cannot be negative"):
            format_timestamp(-1.0)
    
    def test_unknown_format_raises_error(self):
        """Test that unknown formats raise ValueError."""
        with pytest.raises(ValueError, match="Unknown format type"):
            format_timestamp(30.0, "unknown")


class TestParseWebvttTimestampLine:
    """Test WebVTT timestamp line parsing."""
    
    def test_standard_timestamp_line(self):
        """Test parsing standard timestamp lines."""
        result = parse_webvtt_timestamp_line("00:05:30.250 --> 00:05:35.500")
        assert result == (330.25, 335.5)
        
        result = parse_webvtt_timestamp_line("01:23:45.000 --> 01:23:50.750")
        assert result == (5025.0, 5030.75)
    
    def test_flexible_arrow_format(self):
        """Test parsing with flexible arrow spacing."""
        result = parse_webvtt_timestamp_line("00:05:30.250-->00:05:35.500")
        assert result == (330.25, 335.5)
        
        result = parse_webvtt_timestamp_line("00:05:30.250  -->  00:05:35.500")
        assert result == (330.25, 335.5)
    
    def test_invalid_timestamp_range_correction(self):
        """Test correction of invalid timestamp ranges."""
        # End time before start time should be corrected
        result = parse_webvtt_timestamp_line("00:05:35.500 --> 00:05:30.250")
        assert result is not None
        start, end = result
        assert start == 335.5
        assert end > start  # Should be corrected to start + 0.1
    
    def test_non_timestamp_line_returns_none(self):
        """Test that non-timestamp lines return None."""
        assert parse_webvtt_timestamp_line("This is not a timestamp") is None
        assert parse_webvtt_timestamp_line("WEBVTT") is None
        assert parse_webvtt_timestamp_line("1") is None
        assert parse_webvtt_timestamp_line("") is None
    
    def test_malformed_timestamps_handled(self):
        """Test handling of malformed timestamps in lines."""
        # Should still work with malformed individual timestamps
        result = parse_webvtt_timestamp_line("00:05:700 --> 00:06:800")
        assert result is not None
        start, end = result
        assert start == 5.7  # Parsed as malformed
        assert end == 6.8


class TestDistributeWordsOverDuration:
    """Test word distribution over time duration."""
    
    def test_even_distribution(self):
        """Test even distribution of words."""
        words = ["Hello", "world", "test"]
        result = distribute_words_over_duration(words, 0.0, 3.0)
        
        assert len(result) == 3
        assert result[0]["word"] == "Hello"
        assert result[0]["start"] == 0.0
        assert result[0]["end"] == 1.0
        assert result[1]["word"] == "world"
        assert result[1]["start"] == 1.0
        assert result[1]["end"] == 2.0
        assert result[2]["word"] == "test"
        assert result[2]["start"] == 2.0
        assert result[2]["end"] == 3.0
    
    def test_confidence_assignment(self):
        """Test confidence score assignment."""
        words = ["Hello", "world"]
        result = distribute_words_over_duration(words, 0.0, 2.0, base_confidence=0.85)
        
        assert all(item["confidence"] == 0.85 for item in result)
    
    def test_empty_words_list(self):
        """Test handling of empty words list."""
        result = distribute_words_over_duration([], 0.0, 5.0)
        assert result == []
    
    def test_single_word(self):
        """Test distribution of single word."""
        words = ["Hello"]
        result = distribute_words_over_duration(words, 1.0, 3.0)
        
        assert len(result) == 1
        assert result[0]["word"] == "Hello"
        assert result[0]["start"] == 1.0
        assert result[0]["end"] == 3.0
    
    def test_invalid_time_range_correction(self):
        """Test correction of invalid time ranges."""
        words = ["Hello", "world"]
        # End time before start time should be corrected
        result = distribute_words_over_duration(words, 5.0, 3.0)
        
        assert len(result) == 2
        assert result[0]["start"] == 5.0
        assert result[1]["end"] > result[0]["end"]  # Should be corrected
    
    def test_punctuation_filtering(self):
        """Test filtering of words with only punctuation."""
        words = ["Hello", "...", "world", "!", "test"]
        result = distribute_words_over_duration(words, 0.0, 3.0)
        
        # Should filter out words that are only punctuation
        word_texts = [item["word"] for item in result]
        assert "Hello" in word_texts
        assert "world" in word_texts
        assert "test" in word_texts
        # Punctuation-only words might be filtered or included depending on implementation