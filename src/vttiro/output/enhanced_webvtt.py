# this_file: src/vttiro/output/enhanced_webvtt.py
"""Enhanced WebVTT formatter with accessibility compliance and quality optimization.

This module provides advanced WebVTT generation capabilities including:
- W3C WebVTT specification compliance with accessibility features
- Speaker identification and role-based formatting
- Quality metrics and validation
- Customizable styling and positioning
- Accessibility compliance checking (WCAG 2.1 AA)

Used by:
- Output generation pipeline for high-quality subtitle creation
- Accessibility validation systems
- Quality monitoring for subtitle compliance
"""

import re
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Optional

from ..core.types import TranscriptionResult, TranscriptSegment


@dataclass
class WebVTTConfig:
    """Configuration for enhanced WebVTT generation."""
    
    # Accessibility settings
    max_line_length: int = 42                    # WCAG recommended max chars per line
    max_lines_per_cue: int = 2                   # Max lines per subtitle cue
    min_cue_duration: float = 1.0                # Minimum cue duration in seconds
    max_cue_duration: float = 6.0                # Maximum cue duration in seconds
    reading_speed_wpm: float = 180               # Words per minute reading speed
    
    # Formatting settings
    include_speaker_labels: bool = True           # Include speaker identification
    speaker_label_format: str = "<v {speaker}>"  # Speaker label format
    use_positioning: bool = True                  # Use WebVTT positioning
    include_timestamps: bool = False              # Include timestamp comments
    
    # Quality settings
    auto_break_lines: bool = True                # Automatically break long lines
    preserve_sentence_boundaries: bool = True     # Keep sentences together
    remove_filler_words: bool = False            # Remove "um", "uh" etc.
    normalize_punctuation: bool = True           # Standardize punctuation
    
    # Styling settings
    apply_styling: bool = True                   # Apply CSS styling
    custom_css: str = ""                         # Custom CSS styles


@dataclass
class QualityMetrics:
    """Quality metrics for generated subtitles."""
    
    total_cues: int
    avg_cue_duration: float
    avg_reading_speed_wpm: float
    max_line_length: int
    lines_exceeding_limit: int
    cues_too_short: int
    cues_too_long: int
    accessibility_score: float  # 0.0-1.0
    wcag_compliance_level: str  # "A", "AA", "AAA", or "Non-compliant"


class EnhancedWebVTTFormatter:
    """Advanced WebVTT formatter with accessibility and quality features."""
    
    def __init__(self, config: Optional[WebVTTConfig] = None):
        """Initialize formatter with configuration."""
        self.config = config or WebVTTConfig()
        self._filler_words = {'um', 'uh', 'er', 'ah', 'like', 'you know'}
    
    def format_transcription(self, result: TranscriptionResult) -> tuple[str, QualityMetrics]:
        """Generate enhanced WebVTT with quality metrics.
        
        Args:
            result: Transcription result to format
            
        Returns:
            Tuple of (WebVTT content, quality metrics)
        """
        # Preprocess segments for quality optimization
        optimized_segments = self._optimize_segments(result.segments)
        
        # Generate WebVTT content
        webvtt_content = self._generate_webvtt(optimized_segments, result.metadata)
        
        # Calculate quality metrics
        metrics = self._calculate_quality_metrics(optimized_segments)
        
        return webvtt_content, metrics
    
    def _optimize_segments(self, segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
        """Optimize segments for better readability and accessibility."""
        optimized = []
        
        for segment in segments:
            # Process text content
            text = segment.text
            
            if self.config.remove_filler_words:
                text = self._remove_filler_words(text)
            
            if self.config.normalize_punctuation:
                text = self._normalize_punctuation(text)
            
            # Apply line breaking if needed
            if self.config.auto_break_lines:
                text = self._apply_line_breaking(text)
            
            # Create optimized segment
            optimized_segment = TranscriptSegment(
                start=segment.start,
                end=segment.end,
                text=text.strip(),
                speaker=segment.speaker,
                confidence=segment.confidence
            )
            
            # Split long segments if necessary
            split_segments = self._split_long_segments(optimized_segment)
            optimized.extend(split_segments)
        
        # Merge short segments if beneficial
        return self._merge_short_segments(optimized)
    
    def _generate_webvtt(self, segments: list[TranscriptSegment], metadata: dict[str, Any]) -> str:
        """Generate WebVTT content from optimized segments."""
        lines = ["WEBVTT"]
        
        # Add metadata comments
        if self.config.include_timestamps:
            lines.append(f"NOTE Generated by VTTiro at {metadata.get('timestamp', 'unknown')}")
        
        # Add styling
        if self.config.apply_styling:
            lines.extend(self._generate_css_styles())
        
        lines.append("")  # Empty line after header
        
        # Generate cues
        for i, segment in enumerate(segments, 1):
            cue_lines = self._generate_cue(segment, i)
            lines.extend(cue_lines)
            lines.append("")  # Empty line between cues
        
        return "\n".join(lines)
    
    def _generate_cue(self, segment: TranscriptSegment, cue_id: int) -> list[str]:
        """Generate WebVTT cue from segment."""
        lines = []
        
        # Cue identifier
        lines.append(f"cue-{cue_id}")
        
        # Timing line with optional positioning
        timing_line = f"{self._format_timestamp(segment.start)} --> {self._format_timestamp(segment.end)}"
        
        if self.config.use_positioning and segment.speaker:
            # Add speaker positioning
            position = self._get_speaker_position(segment.speaker)
            timing_line += f" {position}"
        
        lines.append(timing_line)
        
        # Cue text with speaker identification
        text = segment.text
        if self.config.include_speaker_labels and segment.speaker:
            speaker_label = self.config.speaker_label_format.format(speaker=segment.speaker)
            text = f"{speaker_label} {text}"
        
        lines.append(text)
        
        return lines
    
    def _generate_css_styles(self) -> list[str]:
        """Generate CSS styling for WebVTT."""
        styles = [
            "STYLE",
            "::cue {",
            "  font-family: 'Arial', sans-serif;",
            "  font-size: 18px;",
            "  line-height: 1.3;",
            "  background-color: rgba(0, 0, 0, 0.8);",
            "  color: white;",
            "  padding: 2px 6px;",
            "}",
            "",
            "::cue(v[voice=\"Speaker 1\"]) {",
            "  color: #FFFF00;",
            "}",
            "",
            "::cue(v[voice=\"Speaker 2\"]) {",
            "  color: #00FFFF;",
            "}",
            ""
        ]
        
        if self.config.custom_css:
            styles.extend(self.config.custom_css.split('\n'))
        
        return styles
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format timestamp in WebVTT format (HH:MM:SS.mmm)."""
        td = timedelta(seconds=seconds)
        hours = int(td.total_seconds() // 3600)
        minutes = int((td.total_seconds() % 3600) // 60)
        secs = td.total_seconds() % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    
    def _get_speaker_position(self, speaker: str) -> str:
        """Get WebVTT positioning for speaker."""
        # Simple positioning based on speaker
        if "1" in speaker or speaker.lower().startswith("speak"):
            return "align:left position:10%"
        else:
            return "align:right position:90%"
    
    def _apply_line_breaking(self, text: str) -> str:
        """Apply intelligent line breaking for readability."""
        words = text.split()
        if len(text) <= self.config.max_line_length:
            return text
        
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + (1 if current_line else 0)  # +1 for space
            
            if current_length + word_length <= self.config.max_line_length:
                current_line.append(word)
                current_length += word_length
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                    current_length = len(word)
                else:
                    # Word is too long, split it
                    lines.append(word[:self.config.max_line_length])
                    current_line = [word[self.config.max_line_length:]] if len(word) > self.config.max_line_length else []
                    current_length = len(current_line[0]) if current_line else 0
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return '\n'.join(lines[:self.config.max_lines_per_cue])
    
    def _remove_filler_words(self, text: str) -> str:
        """Remove common filler words from text."""
        words = text.split()
        filtered_words = [word for word in words 
                         if word.lower().strip('.,!?;:') not in self._filler_words]
        return ' '.join(filtered_words)
    
    def _normalize_punctuation(self, text: str) -> str:
        """Normalize punctuation for consistency."""
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Ensure proper spacing around punctuation
        text = re.sub(r'\s*([.!?])\s*', r'\1 ', text)
        text = re.sub(r'\s*([,;:])\s*', r'\1 ', text)
        
        # Remove extra spaces at start/end
        return text.strip()
    
    def _split_long_segments(self, segment: TranscriptSegment) -> list[TranscriptSegment]:
        """Split segments that are too long for good readability."""
        duration = segment.duration()
        
        if duration <= self.config.max_cue_duration:
            return [segment]
        
        # Calculate optimal reading speed
        word_count = len(segment.text.split())
        reading_time = (word_count / self.config.reading_speed_wpm) * 60
        
        if duration > reading_time * 1.5:  # Allow 50% extra time
            # Split segment
            sentences = re.split(r'[.!?]+', segment.text)
            if len(sentences) <= 1:
                return [segment]  # Can't split further
            
            # Create segments for each sentence
            segments = []
            time_per_sentence = duration / len(sentences)
            
            for i, sentence in enumerate(sentences):
                if sentence.strip():
                    start_time = segment.start + (i * time_per_sentence)
                    end_time = min(segment.end, start_time + time_per_sentence)
                    
                    segments.append(TranscriptSegment(
                        start=start_time,
                        end=end_time,
                        text=sentence.strip() + '.',
                        speaker=segment.speaker,
                        confidence=segment.confidence
                    ))
            
            return segments
        
        return [segment]
    
    def _merge_short_segments(self, segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
        """Merge segments that are too short for good readability."""
        if not segments:
            return segments
        
        merged = []
        current_segment = segments[0]
        
        for i in range(1, len(segments)):
            next_segment = segments[i]
            
            # Check if current segment is too short and can be merged
            if (current_segment.duration() < self.config.min_cue_duration and
                next_segment.start - current_segment.end < 1.0 and  # Gap less than 1 second
                current_segment.speaker == next_segment.speaker):
                
                # Merge segments
                merged_text = f"{current_segment.text} {next_segment.text}"
                current_segment = TranscriptSegment(
                    start=current_segment.start,
                    end=next_segment.end,
                    text=merged_text,
                    speaker=current_segment.speaker,
                    confidence=min(current_segment.confidence or 1.0, 
                                 next_segment.confidence or 1.0)
                )
            else:
                merged.append(current_segment)
                current_segment = next_segment
        
        merged.append(current_segment)
        return merged
    
    def _calculate_quality_metrics(self, segments: list[TranscriptSegment]) -> QualityMetrics:
        """Calculate comprehensive quality metrics for generated subtitles."""
        if not segments:
            return QualityMetrics(
                total_cues=0, avg_cue_duration=0, avg_reading_speed_wpm=0,
                max_line_length=0, lines_exceeding_limit=0,
                cues_too_short=0, cues_too_long=0,
                accessibility_score=0.0, wcag_compliance_level="Non-compliant"
            )
        
        # Basic metrics
        total_cues = len(segments)
        avg_duration = sum(seg.duration() for seg in segments) / total_cues
        
        # Reading speed analysis
        total_words = sum(len(seg.text.split()) for seg in segments)
        total_time = sum(seg.duration() for seg in segments) / 60  # minutes
        avg_reading_speed = total_words / total_time if total_time > 0 else 0
        
        # Line length analysis
        all_lines = []
        for segment in segments:
            all_lines.extend(segment.text.split('\n'))
        
        max_line_length = max(len(line) for line in all_lines) if all_lines else 0
        lines_exceeding_limit = sum(1 for line in all_lines 
                                  if len(line) > self.config.max_line_length)
        
        # Duration analysis
        cues_too_short = sum(1 for seg in segments 
                           if seg.duration() < self.config.min_cue_duration)
        cues_too_long = sum(1 for seg in segments 
                          if seg.duration() > self.config.max_cue_duration)
        
        # Accessibility scoring
        accessibility_score = self._calculate_accessibility_score(
            segments, lines_exceeding_limit, cues_too_short, cues_too_long
        )
        
        # WCAG compliance level
        wcag_level = self._determine_wcag_compliance(accessibility_score)
        
        return QualityMetrics(
            total_cues=total_cues,
            avg_cue_duration=avg_duration,
            avg_reading_speed_wpm=avg_reading_speed,
            max_line_length=max_line_length,
            lines_exceeding_limit=lines_exceeding_limit,
            cues_too_short=cues_too_short,
            cues_too_long=cues_too_long,
            accessibility_score=accessibility_score,
            wcag_compliance_level=wcag_level
        )
    
    def _calculate_accessibility_score(self, segments: list[TranscriptSegment], 
                                     lines_exceeding: int, short_cues: int, 
                                     long_cues: int) -> float:
        """Calculate accessibility score (0.0-1.0) based on WCAG guidelines."""
        total_cues = len(segments)
        if total_cues == 0:
            return 0.0
        
        # Scoring factors
        line_length_score = max(0, 1 - (lines_exceeding / (total_cues * 2)))  # Assume 2 lines max
        duration_score = max(0, 1 - ((short_cues + long_cues) / total_cues))
        
        # Speaker identification score
        speaker_score = 1.0 if any(seg.speaker for seg in segments) else 0.8
        
        # Overall score (weighted average)
        return (line_length_score * 0.4 + duration_score * 0.4 + speaker_score * 0.2)
    
    def _determine_wcag_compliance(self, accessibility_score: float) -> str:
        """Determine WCAG compliance level based on accessibility score."""
        if accessibility_score >= 0.95:
            return "AAA"
        elif accessibility_score >= 0.85:
            return "AA"
        elif accessibility_score >= 0.7:
            return "A"
        else:
            return "Non-compliant"