#!/usr/bin/env python3
# this_file: src/vttiro/output/simple_webvtt.py
"""Simple WebVTT generation for vttiro.

This module provides clean, readable WebVTT output generation without
the complexity of the previous over-engineered output systems.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union
import re

try:
    from loguru import logger
except ImportError:
    import logging as logger

from vttiro.utils.exceptions import ProcessingError
from vttiro.utils.timestamp_utils import validate_webvtt_timestamps
from vttiro.utils.types import SimpleTranscriptSegment


class SimpleWebVTTGenerator:
    """Simple WebVTT generator for clean subtitle output.
    
    Generates properly formatted WebVTT files with good readability,
    proper line breaks, and clean timestamps. Focuses on simplicity
    and compliance with WebVTT standards.
    """
    
    def __init__(
        self,
        max_chars_per_line: int = 50,
        max_lines_per_cue: int = 2,
        max_cue_duration: float = 7.0,
        reading_speed_wpm: int = 160,
        include_cue_ids: bool = False,
        validate_timestamps: bool = True,
        auto_repair_timestamps: bool = True
    ):
        """Initialize the WebVTT generator.
        
        Args:
            max_chars_per_line: Maximum characters per line
            max_lines_per_cue: Maximum lines per subtitle cue
            max_cue_duration: Maximum duration for a single cue (seconds)
            reading_speed_wpm: Assumed reading speed in words per minute
            include_cue_ids: Include cue identifiers in WebVTT output (default: False)
            validate_timestamps: Validate timestamp ranges and sequences (default: True)
            auto_repair_timestamps: Automatically repair timestamp issues (default: True)
        """
        self.max_chars_per_line = max_chars_per_line
        self.max_lines_per_cue = max_lines_per_cue
        self.max_cue_duration = max_cue_duration
        self.reading_speed_wpm = reading_speed_wpm
        self.include_cue_ids = include_cue_ids
        self.validate_timestamps = validate_timestamps
        self.auto_repair_timestamps = auto_repair_timestamps
        
        logger.debug(f"SimpleWebVTTGenerator initialized with {max_chars_per_line} chars/line, "
                    f"cue_ids={include_cue_ids}, validate_timestamps={validate_timestamps}, "
                    f"auto_repair={auto_repair_timestamps}")
    
    def generate_webvtt(
        self, 
        segments: List[SimpleTranscriptSegment], 
        output_path: Union[str, Path],
        title: Optional[str] = None,
        language: Optional[str] = None
    ) -> Path:
        """Generate WebVTT file from transcript segments.
        
        Args:
            segments: List of transcript segments
            output_path: Output file path
            title: Optional title for the WebVTT file
            language: Optional language code (e.g., 'en', 'es')
            
        Returns:
            Path to generated WebVTT file
            
        Raises:
            ProcessingError: If generation fails
        """
        if not segments:
            raise ProcessingError("No transcript segments provided")
        
        output_path = Path(output_path)
        
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Process segments for better readability
            processed_segments = self._process_segments(segments)
            
            # Generate WebVTT content
            content = self._build_webvtt_content(processed_segments, title, language)
            
            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # CRITICAL: Check for zero cues and raise alarm
            if len(processed_segments) == 0:
                logger.error(f"CRITICAL: WebVTT generation produced 0 cues from {len(segments)} input segments!")
                logger.error("This indicates a serious transcription failure.")
                if segments:
                    logger.error("Input segments preview:")
                    for i, seg in enumerate(segments[:3]):
                        logger.error(f"  Segment {i+1}: '{seg.text}' [{seg.start_time:.3f}s - {seg.end_time:.3f}s]")
                    if len(segments) > 3:
                        logger.error(f"  ... and {len(segments) - 3} more segments")
                else:
                    logger.error("No input segments were provided!")
                
                # Still create the file but with warning content
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write("WEBVTT\n\nNOTE\nTranscription failed: No content was generated.\n")
                
                logger.warning(f"WebVTT file created with failure notice: {output_path}")
            else:
                logger.info(f"WebVTT file generated: {output_path} ({len(processed_segments)} cues)")
            
            return output_path
            
        except Exception as e:
            raise ProcessingError(f"Failed to generate WebVTT file: {e}")
    
    def _process_segments(self, segments: List[SimpleTranscriptSegment]) -> List[SimpleTranscriptSegment]:
        """Process segments for optimal readability and timing.
        
        Args:
            segments: Raw transcript segments
            
        Returns:
            Processed segments with improved timing and text formatting
        """
        processed = []
        
        for segment in segments:
            # Clean and format text
            clean_text = self._clean_text(segment.text)
            if not clean_text.strip():
                continue
            
            # Split long segments if needed
            split_segments = self._split_long_segment(segment, clean_text)
            processed.extend(split_segments)
        
        # Ensure no overlapping times (legacy method)
        processed = self._fix_timing_overlaps(processed)
        
        # Enhanced timestamp validation and repair (Issue 105 improvements)
        if self.validate_timestamps:
            logger.debug(f"Validating timestamps for {len(processed)} segments")
            is_valid, repaired_segments, validation_report = validate_webvtt_timestamps(
                processed, 
                auto_repair=self.auto_repair_timestamps,
                min_gap=0.1,  # 100ms minimum gap
                min_duration=0.5  # 500ms minimum duration
            )
            
            if not is_valid and self.auto_repair_timestamps:
                logger.info(f"Timestamp validation: {len(processed) - len(repaired_segments)} segments repaired")
                processed = repaired_segments
            elif not is_valid:
                logger.warning("Timestamp validation failed - consider enabling auto_repair")
                logger.warning(validation_report)
            else:
                logger.debug("All timestamps validated successfully")
        
        return processed
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for subtitle display.
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove common transcription artifacts
        text = re.sub(r'\[.*?\]', '', text)  # Remove [background noise] etc.
        text = re.sub(r'\(.*?\)', '', text)  # Remove (inaudible) etc.
        
        # Fix common punctuation issues
        text = re.sub(r'\s+([,.!?])', r'\1', text)  # Fix spacing before punctuation
        text = re.sub(r'([.!?])\s*([a-z])', r'\1 \2', text)  # Ensure space after sentence end
        
        # Capitalize first letter
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        
        return text.strip()
    
    def _split_long_segment(
        self, 
        segment: SimpleTranscriptSegment, 
        text: str
    ) -> List[SimpleTranscriptSegment]:
        """Split overly long segments for better readability.
        
        Args:
            segment: Original segment
            text: Cleaned text
            
        Returns:
            List of split segments
        """
        duration = segment.end_time - segment.start_time
        
        # Check if segment needs splitting
        if (duration <= self.max_cue_duration and 
            len(text) <= self.max_chars_per_line * self.max_lines_per_cue):
            # Format text with line breaks
            formatted_text = self._format_text_lines(text)
            return [SimpleTranscriptSegment(
                start_time=segment.start_time,
                end_time=segment.end_time,
                text=formatted_text,
                speaker=segment.speaker
            )]
        
        # Split text into chunks
        words = text.split()
        if len(words) <= 1:
            # Single word, don't split
            formatted_text = self._format_text_lines(text)
            return [SimpleTranscriptSegment(
                start_time=segment.start_time,
                end_time=segment.end_time,
                text=formatted_text,
                speaker=segment.speaker
            )]
        
        # Calculate optimal split points
        chunks = self._split_text_into_chunks(words)
        
        # Create segments with proportional timing
        result_segments = []
        total_words = len(words)
        
        for i, chunk in enumerate(chunks):
            chunk_words = len(chunk.split())
            proportion = chunk_words / total_words
            
            chunk_duration = duration * proportion
            chunk_start = segment.start_time + (i * duration / len(chunks))
            chunk_end = min(chunk_start + chunk_duration, segment.end_time)
            
            # Ensure minimum duration
            if chunk_end - chunk_start < 0.5:
                chunk_end = chunk_start + 0.5
            
            formatted_text = self._format_text_lines(chunk)
            result_segments.append(SimpleTranscriptSegment(
                start_time=chunk_start,
                end_time=chunk_end,
                text=formatted_text,
                speaker=segment.speaker
            ))
        
        return result_segments
    
    def _split_text_into_chunks(self, words: List[str]) -> List[str]:
        """Split words into optimal chunks for subtitles.
        
        Args:
            words: List of words
            
        Returns:
            List of text chunks
        """
        chunks = []
        current_chunk = []
        current_length = 0
        max_chunk_chars = self.max_chars_per_line * self.max_lines_per_cue
        
        for word in words:
            word_length = len(word) + 1  # +1 for space
            
            if current_length + word_length <= max_chunk_chars or not current_chunk:
                current_chunk.append(word)
                current_length += word_length
            else:
                # Start new chunk
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
        
        # Add final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _format_text_lines(self, text: str) -> str:
        """Format text with proper line breaks for readability.
        
        Args:
            text: Text to format
            
        Returns:
            Text with line breaks
        """
        if len(text) <= self.max_chars_per_line:
            return text
        
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + (1 if current_line else 0)  # +1 for space if not first word
            
            if current_length + word_length <= self.max_chars_per_line or not current_line:
                current_line.append(word)
                current_length += word_length
            else:
                # Start new line
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
                
                # Limit number of lines
                if len(lines) >= self.max_lines_per_cue:
                    break
        
        # Add final line
        if current_line and len(lines) < self.max_lines_per_cue:
            lines.append(' '.join(current_line))
        
        return '\n'.join(lines)
    
    def _fix_timing_overlaps(
        self, 
        segments: List[SimpleTranscriptSegment]
    ) -> List[SimpleTranscriptSegment]:
        """Fix overlapping timestamps between segments.
        
        Args:
            segments: List of segments
            
        Returns:
            Segments with fixed timing
        """
        if len(segments) <= 1:
            return segments
        
        fixed = []
        
        for i, segment in enumerate(segments):
            if i == 0:
                fixed.append(segment)
                continue
            
            prev_segment = fixed[-1]
            
            # Ensure no overlap
            if segment.start_time < prev_segment.end_time:
                # Adjust start time to avoid overlap
                gap = 0.1  # 100ms gap between segments
                new_start = prev_segment.end_time + gap
                
                # If this would make the segment too short, adjust the previous segment instead
                if segment.end_time - new_start < 0.5:
                    # Shorten previous segment
                    prev_segment.end_time = segment.start_time - gap
                    fixed[-1] = prev_segment
                else:
                    # Adjust current segment start time
                    segment = SimpleTranscriptSegment(
                        start_time=new_start,
                        end_time=segment.end_time,
                        text=segment.text,
                        speaker=segment.speaker
                    )
            
            fixed.append(segment)
        
        return fixed
    
    def _build_webvtt_content(
        self, 
        segments: List[SimpleTranscriptSegment],
        title: Optional[str] = None,
        language: Optional[str] = None
    ) -> str:
        """Build complete WebVTT file content.
        
        Args:
            segments: Processed segments
            title: Optional title
            language: Optional language code
            
        Returns:
            Complete WebVTT content
        """
        lines = ['WEBVTT']
        
        # Add optional headers
        if title:
            lines.append(f'Title: {title}')
        if language:
            lines.append(f'Language: {language}')
        
        lines.append('')  # Empty line after headers
        
        # Add cues
        for i, segment in enumerate(segments):
            # Add cue identifier only if enabled (Issue 105: make cue IDs optional)
            if self.include_cue_ids:
                cue_id = f"cue-{i+1:04d}"
                if segment.speaker:
                    cue_id += f"-{segment.speaker.lower().replace(' ', '-')}"
                lines.append(cue_id)
            
            # Add timestamp line
            start_ts = self._format_timestamp(segment.start_time)
            end_ts = self._format_timestamp(segment.end_time)
            lines.append(f'{start_ts} --> {end_ts}')
            
            # Add speaker name if available
            if segment.speaker:
                lines.append(f'<v {segment.speaker}>{segment.text}</v>')
            else:
                lines.append(segment.text)
            
            lines.append('')  # Empty line between cues
        
        return '\n'.join(lines)
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format timestamp for WebVTT (HH:MM:SS.mmm).
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted timestamp string
        """
        # Ensure non-negative
        seconds = max(0, seconds)
        
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        
        return f'{hours:02d}:{minutes:02d}:{secs:06.3f}'
    
    def estimate_reading_time(self, text: str) -> float:
        """Estimate reading time for text based on reading speed.
        
        Args:
            text: Text to analyze
            
        Returns:
            Estimated reading time in seconds
        """
        word_count = len(text.split())
        reading_time = (word_count / self.reading_speed_wpm) * 60
        return max(1.0, reading_time)  # Minimum 1 second
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"SimpleWebVTTGenerator(max_chars={self.max_chars_per_line}, "
                f"max_lines={self.max_lines_per_cue})")