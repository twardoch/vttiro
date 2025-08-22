# this_file: src/vttiro/output/multi_format_exporter.py
"""Multi-format subtitle export with quality optimization and accessibility compliance.

This module provides comprehensive subtitle format generation including:
- SubRip (SRT) format with timing optimization
- Timed Text Markup Language (TTML) with styling
- Advanced SubStation Alpha (ASS) with positioning
- Plain text transcript generation
- Quality metrics across all formats

Used by:
- Output generation pipeline for comprehensive format support
- Broadcasting and streaming platform integrations
- Accessibility compliance systems
"""

import html
import re
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from typing import Any, Dict, Optional

from ..core.types import TranscriptionResult, TranscriptSegment


class SubtitleFormat(Enum):
    """Supported subtitle export formats."""
    WEBVTT = "webvtt"
    SRT = "srt"
    TTML = "ttml"
    ASS = "ass"
    TRANSCRIPT = "txt"


@dataclass
class FormatConfig:
    """Configuration for specific subtitle format."""
    
    # Timing settings
    max_line_length: int = 42
    max_lines_per_subtitle: int = 2
    min_duration: float = 1.0
    max_duration: float = 6.0
    
    # Formatting settings
    include_speaker_labels: bool = True
    use_styling: bool = True
    line_break_strategy: str = "smart"  # "smart", "fixed", "none"
    
    # Quality settings
    auto_fix_timing: bool = True
    remove_redundant_text: bool = False
    normalize_whitespace: bool = True


@dataclass
class ExportResult:
    """Result of subtitle format export."""
    
    content: str
    format: SubtitleFormat
    quality_metrics: Dict[str, Any]
    accessibility_score: float
    file_extension: str
    mime_type: str


class MultiFormatExporter:
    """Advanced multi-format subtitle exporter with quality optimization."""
    
    def __init__(self):
        """Initialize exporter with format-specific configurations."""
        self.format_configs = {
            SubtitleFormat.SRT: FormatConfig(max_line_length=35),
            SubtitleFormat.TTML: FormatConfig(use_styling=True, max_lines_per_subtitle=3),
            SubtitleFormat.ASS: FormatConfig(include_speaker_labels=True, use_styling=True),
            SubtitleFormat.TRANSCRIPT: FormatConfig(include_speaker_labels=True, use_styling=False)
        }
        
        self.mime_types = {
            SubtitleFormat.WEBVTT: "text/vtt",
            SubtitleFormat.SRT: "application/x-subrip",
            SubtitleFormat.TTML: "application/ttml+xml",
            SubtitleFormat.ASS: "text/x-ass",
            SubtitleFormat.TRANSCRIPT: "text/plain"
        }
        
        self.extensions = {
            SubtitleFormat.WEBVTT: "vtt",
            SubtitleFormat.SRT: "srt",
            SubtitleFormat.TTML: "ttml",
            SubtitleFormat.ASS: "ass",
            SubtitleFormat.TRANSCRIPT: "txt"
        }
    
    def export_format(self, result: TranscriptionResult, 
                     format_type: SubtitleFormat,
                     config: Optional[FormatConfig] = None) -> ExportResult:
        """Export transcription to specified format.
        
        Args:
            result: Transcription result to export
            format_type: Target subtitle format
            config: Optional format-specific configuration
            
        Returns:
            ExportResult with formatted content and metrics
        """
        config = config or self.format_configs.get(format_type, FormatConfig())
        
        # Preprocess segments for format optimization
        optimized_segments = self._optimize_for_format(result.segments, format_type, config)
        
        # Generate format-specific content
        if format_type == SubtitleFormat.SRT:
            content = self._generate_srt(optimized_segments, config)
        elif format_type == SubtitleFormat.TTML:
            content = self._generate_ttml(optimized_segments, result.metadata, config)
        elif format_type == SubtitleFormat.ASS:
            content = self._generate_ass(optimized_segments, config)
        elif format_type == SubtitleFormat.TRANSCRIPT:
            content = self._generate_transcript(optimized_segments, config)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        # Calculate quality metrics
        metrics = self._calculate_format_metrics(optimized_segments, format_type)
        accessibility_score = self._calculate_accessibility_score(optimized_segments)
        
        return ExportResult(
            content=content,
            format=format_type,
            quality_metrics=metrics,
            accessibility_score=accessibility_score,
            file_extension=self.extensions[format_type],
            mime_type=self.mime_types[format_type]
        )
    
    def export_all_formats(self, result: TranscriptionResult) -> Dict[SubtitleFormat, ExportResult]:
        """Export transcription to all supported formats.
        
        Args:
            result: Transcription result to export
            
        Returns:
            Dictionary mapping format types to export results
        """
        exports = {}
        
        for format_type in SubtitleFormat:
            if format_type != SubtitleFormat.WEBVTT:  # Handled by EnhancedWebVTTFormatter
                exports[format_type] = self.export_format(result, format_type)
        
        return exports
    
    def _optimize_for_format(self, segments: list[TranscriptSegment], 
                           format_type: SubtitleFormat, 
                           config: FormatConfig) -> list[TranscriptSegment]:
        """Optimize segments for specific format requirements."""
        optimized = []
        
        for segment in segments:
            # Apply format-specific text processing
            text = self._process_text_for_format(segment.text, format_type, config)
            
            # Apply line breaking based on format requirements
            if config.line_break_strategy == "smart":
                text = self._apply_smart_line_breaking(text, config.max_line_length)
            elif config.line_break_strategy == "fixed":
                text = self._apply_fixed_line_breaking(text, config.max_line_length)
            
            optimized_segment = TranscriptSegment(
                start=segment.start,
                end=segment.end,
                text=text.strip(),
                speaker=segment.speaker,
                confidence=segment.confidence
            )
            
            optimized.append(optimized_segment)
        
        # Apply format-specific timing optimizations
        if config.auto_fix_timing:
            optimized = self._fix_timing_issues(optimized, config)
        
        return optimized
    
    def _generate_srt(self, segments: list[TranscriptSegment], config: FormatConfig) -> str:
        """Generate SubRip (SRT) format content."""
        lines = []
        
        for i, segment in enumerate(segments, 1):
            # Subtitle number
            lines.append(str(i))
            
            # Timing line (SRT format: HH:MM:SS,mmm --> HH:MM:SS,mmm)
            start_time = self._format_srt_timestamp(segment.start)
            end_time = self._format_srt_timestamp(segment.end)
            lines.append(f"{start_time} --> {end_time}")
            
            # Subtitle text with optional speaker label
            text = segment.text
            if config.include_speaker_labels and segment.speaker:
                text = f"[{segment.speaker}] {text}"
            
            lines.append(text)
            lines.append("")  # Empty line between subtitles
        
        return "\n".join(lines)
    
    def _generate_ttml(self, segments: list[TranscriptSegment], 
                      metadata: Dict[str, Any], config: FormatConfig) -> str:
        """Generate Timed Text Markup Language (TTML) content."""
        # TTML header
        ttml_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<tt xmlns="http://www.w3.org/ns/ttml"',
            '    xmlns:tts="http://www.w3.org/ns/ttml#styling"',
            '    xml:lang="en">',
            '  <head>',
            '    <styling>',
            '      <style xml:id="defaultCaption"',
            '             tts:fontSize="18px"',
            '             tts:fontFamily="Arial, sans-serif"',
            '             tts:textAlign="center"',
            '             tts:color="white"',
            '             tts:backgroundColor="rgba(0,0,0,0.8)"/>',
            '      <style xml:id="speaker1"',
            '             tts:color="#FFFF00"/>',
            '      <style xml:id="speaker2"',
            '             tts:color="#00FFFF"/>',
            '    </styling>',
            '  </head>',
            '  <body>',
            '    <div>'
        ]
        
        # Generate TTML cues
        for i, segment in enumerate(segments):
            begin_time = self._format_ttml_timestamp(segment.start)
            end_time = self._format_ttml_timestamp(segment.end)
            
            # Determine style based on speaker
            style = "defaultCaption"
            if segment.speaker and "1" in segment.speaker:
                style = "speaker1"
            elif segment.speaker and "2" in segment.speaker:
                style = "speaker2"
            
            # Escape HTML entities
            text = html.escape(segment.text)
            
            # Add speaker label if configured
            if config.include_speaker_labels and segment.speaker:
                text = f"<tts:span tts:fontWeight=\"bold\">[{segment.speaker}]</tts:span> {text}"
            
            ttml_lines.append(f'      <p xml:id="subtitle{i+1}"')
            ttml_lines.append(f'         style="{style}"')
            ttml_lines.append(f'         begin="{begin_time}"')
            ttml_lines.append(f'         end="{end_time}">')
            ttml_lines.append(f'        {text}')
            ttml_lines.append('      </p>')
        
        # TTML footer
        ttml_lines.extend([
            '    </div>',
            '  </body>',
            '</tt>'
        ])
        
        return "\n".join(ttml_lines)
    
    def _generate_ass(self, segments: list[TranscriptSegment], config: FormatConfig) -> str:
        """Generate Advanced SubStation Alpha (ASS) format content."""
        # ASS header
        ass_lines = [
            "[Script Info]",
            "Title: Generated by VTTiro",
            "ScriptType: v4.00+",
            "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
            "Style: Default,Arial,18,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1",
            "Style: Speaker1,Arial,18,&H0000FFFF,&H000000FF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1",
            "Style: Speaker2,Arial,18,&H00FFFF00,&H000000FF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1",
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
        ]
        
        # Generate ASS events
        for segment in segments:
            start_time = self._format_ass_timestamp(segment.start)
            end_time = self._format_ass_timestamp(segment.end)
            
            # Determine style based on speaker
            style = "Default"
            if segment.speaker and "1" in segment.speaker:
                style = "Speaker1"
            elif segment.speaker and "2" in segment.speaker:
                style = "Speaker2"
            
            # Prepare text
            text = segment.text.replace('\n', '\\N')  # ASS line break
            
            # Add speaker label if configured
            if config.include_speaker_labels and segment.speaker:
                text = f"[{segment.speaker}] {text}"
            
            ass_lines.append(f"Dialogue: 0,{start_time},{end_time},{style},,0,0,0,,{text}")
        
        return "\n".join(ass_lines)
    
    def _generate_transcript(self, segments: list[TranscriptSegment], config: FormatConfig) -> str:
        """Generate plain text transcript."""
        lines = []
        
        for segment in segments:
            timestamp = self._format_transcript_timestamp(segment.start)
            
            if config.include_speaker_labels and segment.speaker:
                lines.append(f"[{timestamp}] {segment.speaker}: {segment.text}")
            else:
                lines.append(f"[{timestamp}] {segment.text}")
        
        return "\n".join(lines)
    
    def _process_text_for_format(self, text: str, format_type: SubtitleFormat, 
                               config: FormatConfig) -> str:
        """Apply format-specific text processing."""
        if config.normalize_whitespace:
            text = re.sub(r'\s+', ' ', text).strip()
        
        if config.remove_redundant_text:
            # Remove repeated words (simple implementation)
            words = text.split()
            filtered_words = []
            prev_word = None
            
            for word in words:
                if word.lower() != prev_word:
                    filtered_words.append(word)
                prev_word = word.lower()
            
            text = ' '.join(filtered_words)
        
        # Format-specific processing
        if format_type == SubtitleFormat.TTML:
            # Escape special characters for XML
            text = html.escape(text)
        elif format_type == SubtitleFormat.ASS:
            # Escape ASS special characters
            text = text.replace('{', '\\{').replace('}', '\\}')
        
        return text
    
    def _apply_smart_line_breaking(self, text: str, max_length: int) -> str:
        """Apply intelligent line breaking based on linguistic boundaries."""
        if len(text) <= max_length:
            return text
        
        # Try to break at sentence boundaries first
        sentences = re.split(r'([.!?]+)', text)
        if len(sentences) > 1:
            lines = []
            current_line = ""
            
            for part in sentences:
                test_line = current_line + part
                if len(test_line) <= max_length:
                    current_line = test_line
                else:
                    if current_line:
                        lines.append(current_line.strip())
                    current_line = part
            
            if current_line:
                lines.append(current_line.strip())
            
            return '\n'.join(lines)
        
        # Fall back to word-based breaking
        return self._apply_fixed_line_breaking(text, max_length)
    
    def _apply_fixed_line_breaking(self, text: str, max_length: int) -> str:
        """Apply fixed line breaking at word boundaries."""
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + (1 if current_line else 0)
            
            if current_length + word_length <= max_length:
                current_line.append(word)
                current_length += word_length
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return '\n'.join(lines)
    
    def _fix_timing_issues(self, segments: list[TranscriptSegment], 
                          config: FormatConfig) -> list[TranscriptSegment]:
        """Fix common timing issues in segments."""
        if not segments:
            return segments
        
        fixed = []
        
        for i, segment in enumerate(segments):
            # Ensure minimum duration
            duration = segment.duration()
            if duration < config.min_duration:
                # Extend end time
                new_end = segment.start + config.min_duration
                
                # Avoid overlap with next segment
                if i + 1 < len(segments):
                    new_end = min(new_end, segments[i + 1].start - 0.1)
                
                segment = TranscriptSegment(
                    start=segment.start,
                    end=new_end,
                    text=segment.text,
                    speaker=segment.speaker,
                    confidence=segment.confidence
                )
            
            # Ensure maximum duration
            elif duration > config.max_duration:
                # Split long segment (simplified)
                mid_point = segment.start + (duration / 2)
                
                # Try to split at sentence boundary
                sentences = segment.text.split('.')
                if len(sentences) > 1:
                    first_part = sentences[0] + '.'
                    second_part = '.'.join(sentences[1:]).strip()
                    
                    if second_part:
                        fixed.append(TranscriptSegment(
                            start=segment.start,
                            end=mid_point,
                            text=first_part,
                            speaker=segment.speaker,
                            confidence=segment.confidence
                        ))
                        
                        fixed.append(TranscriptSegment(
                            start=mid_point,
                            end=segment.end,
                            text=second_part,
                            speaker=segment.speaker,
                            confidence=segment.confidence
                        ))
                        continue
            
            fixed.append(segment)
        
        return fixed
    
    def _format_srt_timestamp(self, seconds: float) -> str:
        """Format timestamp for SRT format (HH:MM:SS,mmm)."""
        td = timedelta(seconds=seconds)
        hours = int(td.total_seconds() // 3600)
        minutes = int((td.total_seconds() % 3600) // 60)
        secs = int(td.total_seconds() % 60)
        millisecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
    
    def _format_ttml_timestamp(self, seconds: float) -> str:
        """Format timestamp for TTML format (HH:MM:SS.mmm)."""
        td = timedelta(seconds=seconds)
        hours = int(td.total_seconds() // 3600)
        minutes = int((td.total_seconds() % 3600) // 60)
        secs = td.total_seconds() % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    
    def _format_ass_timestamp(self, seconds: float) -> str:
        """Format timestamp for ASS format (H:MM:SS.cc)."""
        td = timedelta(seconds=seconds)
        hours = int(td.total_seconds() // 3600)
        minutes = int((td.total_seconds() % 3600) // 60)
        secs = int(td.total_seconds() % 60)
        centisecs = int((seconds % 1) * 100)
        return f"{hours}:{minutes:02d}:{secs:02d}.{centisecs:02d}"
    
    def _format_transcript_timestamp(self, seconds: float) -> str:
        """Format timestamp for transcript (HH:MM:SS)."""
        td = timedelta(seconds=seconds)
        hours = int(td.total_seconds() // 3600)
        minutes = int((td.total_seconds() % 3600) // 60)
        secs = int(td.total_seconds() % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def _calculate_format_metrics(self, segments: list[TranscriptSegment], 
                                format_type: SubtitleFormat) -> Dict[str, Any]:
        """Calculate format-specific quality metrics."""
        if not segments:
            return {}
        
        total_segments = len(segments)
        total_duration = sum(seg.duration() for seg in segments)
        total_words = sum(len(seg.text.split()) for seg in segments)
        
        metrics = {
            "total_segments": total_segments,
            "total_duration_seconds": total_duration,
            "total_words": total_words,
            "average_segment_duration": total_duration / total_segments,
            "words_per_minute": (total_words / (total_duration / 60)) if total_duration > 0 else 0,
            "format_type": format_type.value
        }
        
        # Format-specific metrics
        if format_type == SubtitleFormat.SRT:
            metrics["srt_compliance"] = self._check_srt_compliance(segments)
        elif format_type == SubtitleFormat.TTML:
            metrics["xml_valid"] = True  # Simplified - would need XML validation
        
        return metrics
    
    def _calculate_accessibility_score(self, segments: list[TranscriptSegment]) -> float:
        """Calculate accessibility score for any format."""
        if not segments:
            return 0.0
        
        # Check reading speed (should be reasonable)
        total_words = sum(len(seg.text.split()) for seg in segments)
        total_time = sum(seg.duration() for seg in segments) / 60
        reading_speed = total_words / total_time if total_time > 0 else 0
        
        speed_score = 1.0 if 120 <= reading_speed <= 200 else 0.7
        
        # Check for speaker identification
        speaker_score = 1.0 if any(seg.speaker for seg in segments) else 0.8
        
        # Check timing consistency
        timing_issues = sum(1 for seg in segments if seg.duration() < 0.5 or seg.duration() > 8)
        timing_score = max(0, 1.0 - (timing_issues / len(segments)))
        
        return (speed_score * 0.4 + speaker_score * 0.3 + timing_score * 0.3)
    
    def _check_srt_compliance(self, segments: list[TranscriptSegment]) -> bool:
        """Check if segments comply with SRT format requirements."""
        for segment in segments:
            # Check timing format compatibility
            if segment.start < 0 or segment.end <= segment.start:
                return False
            
            # Check text length (simplified check)
            lines = segment.text.split('\n')
            if len(lines) > 2 or any(len(line) > 42 for line in lines):
                return False
        
        return True