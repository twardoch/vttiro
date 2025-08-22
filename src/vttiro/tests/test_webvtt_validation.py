# this_file: src/vttiro/tests/test_webvtt_validation.py
"""Tests for WebVTT structure and timing validation.

This module tests WebVTT output compliance with the WebVTT specification,
including structure validation, timing accuracy, and format correctness.
"""

import pytest
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from vttiro.core.types import TranscriptionResult, TranscriptSegment


@dataclass
class WebVTTCue:
    """Represents a WebVTT cue for validation."""
    
    identifier: Optional[str]
    start_time: float
    end_time: float
    text: str
    settings: Dict[str, str]
    notes: List[str]


@dataclass
class WebVTTFile:
    """Represents a parsed WebVTT file for validation."""
    
    header: str
    metadata: Dict[str, Any]
    cues: List[WebVTTCue]
    notes: List[str]


class WebVTTValidator:
    """Validator for WebVTT file structure and content."""
    
    # WebVTT specification patterns
    WEBVTT_HEADER_PATTERN = re.compile(r'^WEBVTT(?:\s+.*)?$')
    TIME_PATTERN = re.compile(r'^(\d{2}):(\d{2}):(\d{2})\.(\d{3})$')
    TIME_RANGE_PATTERN = re.compile(
        r'^(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s+-->\s+(\d{2}):(\d{2}):(\d{2})\.(\d{3})(?:\s+(.*))?$'
    )
    CUE_SETTINGS_PATTERN = re.compile(r'(\w+):([^\s]+)')
    
    def __init__(self):
        """Initialize WebVTT validator."""
        self.errors = []
        self.warnings = []
    
    def validate_webvtt_content(self, content: str) -> Tuple[bool, List[str], List[str]]:
        """Validate WebVTT content structure and format.
        
        Args:
            content: WebVTT file content as string
            
        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        self.errors = []
        self.warnings = []
        
        try:
            webvtt_file = self.parse_webvtt(content)
            self._validate_structure(webvtt_file)
            self._validate_timing(webvtt_file)
            self._validate_text_content(webvtt_file)
            self._validate_cue_settings(webvtt_file)
        except Exception as e:
            self.errors.append(f"Parse error: {e}")
        
        return len(self.errors) == 0, self.errors, self.warnings
    
    def parse_webvtt(self, content: str) -> WebVTTFile:
        """Parse WebVTT content into structured format.
        
        Args:
            content: WebVTT file content
            
        Returns:
            Parsed WebVTT file structure
        """
        lines = content.strip().split('\n')
        if not lines:
            raise ValueError("Empty WebVTT content")
        
        # Parse header
        header_line = lines[0].strip()
        if not self.WEBVTT_HEADER_PATTERN.match(header_line):
            raise ValueError(f"Invalid WebVTT header: {header_line}")
        
        webvtt_file = WebVTTFile(
            header=header_line,
            metadata={},
            cues=[],
            notes=[]
        )
        
        # Parse content
        i = 1
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines
            if not line:
                i += 1
                continue
            
            # Parse NOTE blocks
            if line.startswith('NOTE'):
                note_lines = [line[4:].strip()]  # Remove 'NOTE' prefix
                i += 1
                while i < len(lines) and lines[i].strip():
                    note_lines.append(lines[i].strip())
                    i += 1
                webvtt_file.notes.append(' '.join(note_lines))
                continue
            
            # Parse cues
            cue = self._parse_cue(lines, i)
            if cue:
                webvtt_file.cues.append(cue[0])
                i = cue[1]
            else:
                i += 1
        
        return webvtt_file
    
    def _parse_cue(self, lines: List[str], start_index: int) -> Optional[Tuple[WebVTTCue, int]]:
        """Parse a single WebVTT cue.
        
        Args:
            lines: List of file lines
            start_index: Starting line index
            
        Returns:
            Tuple of (parsed cue, next line index) or None if not a cue
        """
        i = start_index
        identifier = None
        
        # Check if first line is identifier or timing
        first_line = lines[i].strip()
        timing_match = self.TIME_RANGE_PATTERN.match(first_line)
        
        if not timing_match:
            # First line is identifier
            identifier = first_line
            i += 1
            if i >= len(lines):
                return None
            timing_line = lines[i].strip()
            timing_match = self.TIME_RANGE_PATTERN.match(timing_line)
        else:
            timing_line = first_line
        
        if not timing_match:
            return None
        
        # Parse timing
        start_time = self._parse_time(timing_match.groups()[:4])
        end_time = self._parse_time(timing_match.groups()[4:8])
        settings_text = timing_match.group(9) or ""
        settings = self._parse_cue_settings(settings_text)
        
        # Parse cue text
        i += 1
        text_lines = []
        while i < len(lines) and lines[i].strip():
            text_lines.append(lines[i])
            i += 1
        
        cue = WebVTTCue(
            identifier=identifier,
            start_time=start_time,
            end_time=end_time,
            text='\n'.join(text_lines),
            settings=settings,
            notes=[]
        )
        
        return cue, i
    
    def _parse_time(self, time_groups: Tuple[str, str, str, str]) -> float:
        """Parse time components into seconds.
        
        Args:
            time_groups: Tuple of (hours, minutes, seconds, milliseconds)
            
        Returns:
            Time in seconds as float
        """
        hours, minutes, seconds, milliseconds = time_groups
        return (
            int(hours) * 3600 +
            int(minutes) * 60 +
            int(seconds) +
            int(milliseconds) / 1000.0
        )
    
    def _parse_cue_settings(self, settings_text: str) -> Dict[str, str]:
        """Parse cue settings from timing line.
        
        Args:
            settings_text: Settings portion of timing line
            
        Returns:
            Dictionary of setting name to value
        """
        settings = {}
        if settings_text:
            matches = self.CUE_SETTINGS_PATTERN.findall(settings_text)
            for name, value in matches:
                settings[name] = value
        return settings
    
    def _validate_structure(self, webvtt_file: WebVTTFile):
        """Validate WebVTT file structure.
        
        Args:
            webvtt_file: Parsed WebVTT file
        """
        # Check header
        if not webvtt_file.header.startswith('WEBVTT'):
            self.errors.append("Missing or invalid WEBVTT header")
        
        # Check for cues
        if not webvtt_file.cues:
            self.warnings.append("No cues found in WebVTT file")
        
        # Validate cue identifiers are unique if present
        identifiers = [cue.identifier for cue in webvtt_file.cues if cue.identifier]
        if len(identifiers) != len(set(identifiers)):
            self.errors.append("Duplicate cue identifiers found")
    
    def _validate_timing(self, webvtt_file: WebVTTFile):
        """Validate timing accuracy and constraints.
        
        Args:
            webvtt_file: Parsed WebVTT file
        """
        for i, cue in enumerate(webvtt_file.cues):
            cue_id = cue.identifier or f"cue-{i+1}"
            
            # Validate individual cue timing
            if cue.start_time < 0:
                self.errors.append(f"{cue_id}: Negative start time")
            
            if cue.end_time < 0:
                self.errors.append(f"{cue_id}: Negative end time")
            
            if cue.start_time >= cue.end_time:
                self.errors.append(f"{cue_id}: Start time >= end time")
            
            # Check for very short cues (might be unintentional)
            duration = cue.end_time - cue.start_time
            if duration < 0.1:
                self.warnings.append(f"{cue_id}: Very short duration ({duration:.3f}s)")
            
            # Check for very long cues (might impact readability)
            if duration > 10.0:
                self.warnings.append(f"{cue_id}: Very long duration ({duration:.1f}s)")
        
        # Validate sequential timing (overlaps and gaps)
        self._validate_timing_sequence(webvtt_file.cues)
    
    def _validate_timing_sequence(self, cues: List[WebVTTCue]):
        """Validate timing sequence between cues.
        
        Args:
            cues: List of cues to validate
        """
        for i in range(len(cues) - 1):
            current_cue = cues[i]
            next_cue = cues[i + 1]
            
            current_id = current_cue.identifier or f"cue-{i+1}"
            next_id = next_cue.identifier or f"cue-{i+2}"
            
            # Check for overlapping cues
            if current_cue.end_time > next_cue.start_time:
                overlap = current_cue.end_time - next_cue.start_time
                self.warnings.append(
                    f"Overlapping cues: {current_id} and {next_id} "
                    f"(overlap: {overlap:.3f}s)"
                )
            
            # Check for large gaps (might indicate missing content)
            gap = next_cue.start_time - current_cue.end_time
            if gap > 5.0:
                self.warnings.append(
                    f"Large gap between {current_id} and {next_id} "
                    f"(gap: {gap:.1f}s)"
                )
    
    def _validate_text_content(self, webvtt_file: WebVTTFile):
        """Validate text content in cues.
        
        Args:
            webvtt_file: Parsed WebVTT file
        """
        for i, cue in enumerate(webvtt_file.cues):
            cue_id = cue.identifier or f"cue-{i+1}"
            
            # Check for empty text
            if not cue.text.strip():
                self.warnings.append(f"{cue_id}: Empty cue text")
            
            # Check text length (readability)
            text_length = len(cue.text)
            if text_length > 200:
                self.warnings.append(
                    f"{cue_id}: Long text ({text_length} chars) may impact readability"
                )
            
            # Check for line breaks and formatting
            lines = cue.text.split('\n')
            if len(lines) > 3:
                self.warnings.append(f"{cue_id}: More than 3 lines may not display properly")
            
            # Validate HTML tags if present
            self._validate_cue_markup(cue.text, cue_id)
    
    def _validate_cue_markup(self, text: str, cue_id: str):
        """Validate HTML markup in cue text.
        
        Args:
            text: Cue text content
            cue_id: Cue identifier for error reporting
        """
        # Check for valid WebVTT tags
        valid_tags = {'b', 'i', 'u', 'c', 'v', 'lang', 'ruby', 'rt'}
        
        # Find all HTML-like tags
        tag_pattern = re.compile(r'<(/?)(\w+)(?:\s+[^>]*)?>')
        tags = tag_pattern.findall(text)
        
        for is_closing, tag_name in tags:
            if tag_name not in valid_tags:
                self.warnings.append(
                    f"{cue_id}: Non-standard WebVTT tag '<{tag_name}>'"
                )
        
        # Check for unmatched tags
        self._validate_tag_matching(text, cue_id)
    
    def _validate_tag_matching(self, text: str, cue_id: str):
        """Validate that HTML tags are properly matched.
        
        Args:
            text: Cue text content
            cue_id: Cue identifier for error reporting
        """
        tag_stack = []
        tag_pattern = re.compile(r'<(/?)(\w+)(?:\s+[^>]*)?>')
        
        for match in tag_pattern.finditer(text):
            is_closing = match.group(1) == '/'
            tag_name = match.group(2)
            
            if is_closing:
                if not tag_stack or tag_stack[-1] != tag_name:
                    self.errors.append(
                        f"{cue_id}: Unmatched closing tag '</{tag_name}>'"
                    )
                else:
                    tag_stack.pop()
            else:
                tag_stack.append(tag_name)
        
        # Check for unclosed tags
        for tag_name in tag_stack:
            self.errors.append(f"{cue_id}: Unclosed tag '<{tag_name}>'")
    
    def _validate_cue_settings(self, webvtt_file: WebVTTFile):
        """Validate cue settings.
        
        Args:
            webvtt_file: Parsed WebVTT file
        """
        valid_settings = {
            'vertical': ['rl', 'lr'],
            'line': None,  # Can be number or percentage
            'position': None,  # Can be number or percentage
            'size': None,  # Percentage
            'align': ['start', 'center', 'end', 'left', 'right']
        }
        
        for i, cue in enumerate(webvtt_file.cues):
            cue_id = cue.identifier or f"cue-{i+1}"
            
            for setting_name, setting_value in cue.settings.items():
                if setting_name not in valid_settings:
                    self.errors.append(
                        f"{cue_id}: Invalid cue setting '{setting_name}'"
                    )
                    continue
                
                allowed_values = valid_settings[setting_name]
                if allowed_values and setting_value not in allowed_values:
                    self.errors.append(
                        f"{cue_id}: Invalid value '{setting_value}' for setting '{setting_name}'"
                    )


class TestWebVTTStructureValidation:
    """Test WebVTT structure validation."""
    
    @pytest.fixture
    def validator(self):
        """Provide WebVTT validator instance."""
        return WebVTTValidator()
    
    def test_valid_basic_webvtt(self, validator):
        """Test validation of basic valid WebVTT."""
        webvtt_content = """WEBVTT

00:00:00.000 --> 00:00:03.500
Hello, world!

00:00:04.000 --> 00:00:07.500
This is a test.
"""
        
        is_valid, errors, warnings = validator.validate_webvtt_content(webvtt_content)
        
        assert is_valid
        assert len(errors) == 0
        assert len(warnings) == 0
    
    def test_webvtt_with_identifiers(self, validator):
        """Test WebVTT with cue identifiers."""
        webvtt_content = """WEBVTT

cue1
00:00:00.000 --> 00:00:03.500
First cue with identifier.

cue2
00:00:04.000 --> 00:00:07.500
Second cue with identifier.
"""
        
        is_valid, errors, warnings = validator.validate_webvtt_content(webvtt_content)
        
        assert is_valid
        assert len(errors) == 0
    
    def test_webvtt_with_notes(self, validator):
        """Test WebVTT with NOTE blocks."""
        webvtt_content = """WEBVTT

NOTE This is a comment

00:00:00.000 --> 00:00:03.500
Content with note.

NOTE Another comment
spanning multiple lines

00:00:04.000 --> 00:00:07.500
More content.
"""
        
        is_valid, errors, warnings = validator.validate_webvtt_content(webvtt_content)
        
        assert is_valid
        assert len(errors) == 0
    
    def test_invalid_webvtt_header(self, validator):
        """Test invalid WebVTT header."""
        webvtt_content = """INVALID_HEADER

00:00:00.000 --> 00:00:03.500
Content with bad header.
"""
        
        is_valid, errors, warnings = validator.validate_webvtt_content(webvtt_content)
        
        assert not is_valid
        assert any("header" in error.lower() for error in errors)
    
    def test_duplicate_cue_identifiers(self, validator):
        """Test detection of duplicate cue identifiers."""
        webvtt_content = """WEBVTT

cue1
00:00:00.000 --> 00:00:03.500
First cue.

cue1
00:00:04.000 --> 00:00:07.500
Duplicate identifier.
"""
        
        is_valid, errors, warnings = validator.validate_webvtt_content(webvtt_content)
        
        assert not is_valid
        assert any("duplicate" in error.lower() for error in errors)
    
    def test_empty_webvtt_file(self, validator):
        """Test handling of empty WebVTT file."""
        webvtt_content = ""
        
        is_valid, errors, warnings = validator.validate_webvtt_content(webvtt_content)
        
        assert not is_valid
        assert any("empty" in error.lower() for error in errors)


class TestWebVTTTimingValidation:
    """Test WebVTT timing validation."""
    
    @pytest.fixture
    def validator(self):
        """Provide WebVTT validator instance."""
        return WebVTTValidator()
    
    def test_valid_timing_sequence(self, validator):
        """Test valid timing sequence."""
        webvtt_content = """WEBVTT

00:00:00.000 --> 00:00:02.500
First cue.

00:00:03.000 --> 00:00:05.500
Second cue with proper gap.

00:00:06.000 --> 00:00:08.500
Third cue.
"""
        
        is_valid, errors, warnings = validator.validate_webvtt_content(webvtt_content)
        
        assert is_valid
        assert len(errors) == 0
    
    def test_negative_timing(self, validator):
        """Test detection of negative timing."""
        webvtt_content = """WEBVTT

00:00:05.000 --> 00:00:02.500
Invalid timing: start > end.
"""
        
        is_valid, errors, warnings = validator.validate_webvtt_content(webvtt_content)
        
        assert not is_valid
        assert any("start time" in error.lower() for error in errors)
    
    def test_overlapping_cues(self, validator):
        """Test detection of overlapping cues."""
        webvtt_content = """WEBVTT

00:00:00.000 --> 00:00:05.000
First cue.

00:00:03.000 --> 00:00:07.000
Overlapping cue.
"""
        
        is_valid, errors, warnings = validator.validate_webvtt_content(webvtt_content)
        
        # Should be valid but with warnings
        assert is_valid
        assert any("overlap" in warning.lower() for warning in warnings)
    
    def test_very_short_cues(self, validator):
        """Test detection of very short cues."""
        webvtt_content = """WEBVTT

00:00:00.000 --> 00:00:00.050
Very short cue.
"""
        
        is_valid, errors, warnings = validator.validate_webvtt_content(webvtt_content)
        
        assert is_valid
        assert any("short duration" in warning.lower() for warning in warnings)
    
    def test_very_long_cues(self, validator):
        """Test detection of very long cues."""
        webvtt_content = """WEBVTT

00:00:00.000 --> 00:00:15.000
This is a very long cue that might impact readability.
"""
        
        is_valid, errors, warnings = validator.validate_webvtt_content(webvtt_content)
        
        assert is_valid
        assert any("long duration" in warning.lower() for warning in warnings)
    
    def test_large_timing_gaps(self, validator):
        """Test detection of large gaps between cues."""
        webvtt_content = """WEBVTT

00:00:00.000 --> 00:00:02.000
First cue.

00:00:10.000 --> 00:00:12.000
Cue after large gap.
"""
        
        is_valid, errors, warnings = validator.validate_webvtt_content(webvtt_content)
        
        assert is_valid
        assert any("large gap" in warning.lower() for warning in warnings)


class TestWebVTTTextValidation:
    """Test WebVTT text content validation."""
    
    @pytest.fixture
    def validator(self):
        """Provide WebVTT validator instance."""
        return WebVTTValidator()
    
    def test_valid_text_formatting(self, validator):
        """Test valid text formatting."""
        webvtt_content = """WEBVTT

00:00:00.000 --> 00:00:03.500
<b>Bold text</b> and <i>italic text</i>.

00:00:04.000 --> 00:00:07.500
<v Speaker>Text with speaker tag</v>.
"""
        
        is_valid, errors, warnings = validator.validate_webvtt_content(webvtt_content)
        
        assert is_valid
        assert len(errors) == 0
    
    def test_invalid_html_tags(self, validator):
        """Test detection of invalid HTML tags."""
        webvtt_content = """WEBVTT

00:00:00.000 --> 00:00:03.500
<div>Invalid div tag</div> in WebVTT.
"""
        
        is_valid, errors, warnings = validator.validate_webvtt_content(webvtt_content)
        
        assert is_valid  # Still valid, but with warnings
        assert any("non-standard" in warning.lower() for warning in warnings)
    
    def test_unmatched_html_tags(self, validator):
        """Test detection of unmatched HTML tags."""
        webvtt_content = """WEBVTT

00:00:00.000 --> 00:00:03.500
<b>Unmatched bold tag.

00:00:04.000 --> 00:00:07.500
Text with </i>unmatched closing tag.
"""
        
        is_valid, errors, warnings = validator.validate_webvtt_content(webvtt_content)
        
        assert not is_valid
        assert any("unmatched" in error.lower() or "unclosed" in error.lower() for error in errors)
    
    def test_empty_cue_text(self, validator):
        """Test detection of empty cue text."""
        webvtt_content = """WEBVTT

00:00:00.000 --> 00:00:03.500

"""
        
        is_valid, errors, warnings = validator.validate_webvtt_content(webvtt_content)
        
        assert is_valid
        assert any("empty" in warning.lower() for warning in warnings)
    
    def test_very_long_text(self, validator):
        """Test detection of very long text."""
        long_text = "Very long text content. " * 20  # > 200 characters
        webvtt_content = f"""WEBVTT

00:00:00.000 --> 00:00:03.500
{long_text}
"""
        
        is_valid, errors, warnings = validator.validate_webvtt_content(webvtt_content)
        
        assert is_valid
        assert any("long text" in warning.lower() for warning in warnings)
    
    def test_many_text_lines(self, validator):
        """Test detection of too many text lines."""
        webvtt_content = """WEBVTT

00:00:00.000 --> 00:00:03.500
Line 1
Line 2  
Line 3
Line 4
Line 5
"""
        
        is_valid, errors, warnings = validator.validate_webvtt_content(webvtt_content)
        
        assert is_valid
        assert any("lines" in warning.lower() for warning in warnings)


class TestWebVTTSettingsValidation:
    """Test WebVTT cue settings validation."""
    
    @pytest.fixture
    def validator(self):
        """Provide WebVTT validator instance."""
        return WebVTTValidator()
    
    def test_valid_cue_settings(self, validator):
        """Test valid cue settings."""
        webvtt_content = """WEBVTT

00:00:00.000 --> 00:00:03.500 align:center
Centered text.

00:00:04.000 --> 00:00:07.500 position:25% line:75%
Positioned text.
"""
        
        is_valid, errors, warnings = validator.validate_webvtt_content(webvtt_content)
        
        assert is_valid
        assert len(errors) == 0
    
    def test_invalid_cue_settings(self, validator):
        """Test invalid cue settings."""
        webvtt_content = """WEBVTT

00:00:00.000 --> 00:00:03.500 invalid:setting
Text with invalid setting.
"""
        
        is_valid, errors, warnings = validator.validate_webvtt_content(webvtt_content)
        
        assert not is_valid
        assert any("invalid" in error.lower() for error in errors)
    
    def test_invalid_setting_values(self, validator):
        """Test invalid setting values."""
        webvtt_content = """WEBVTT

00:00:00.000 --> 00:00:03.500 align:invalid_value
Text with invalid align value.
"""
        
        is_valid, errors, warnings = validator.validate_webvtt_content(webvtt_content)
        
        assert not is_valid
        assert any("invalid value" in error.lower() for error in errors)


class TestWebVTTTimingAccuracy:
    """Test WebVTT timing accuracy assertions."""
    
    def test_timing_precision(self):
        """Test timing precision to milliseconds."""
        # Test that timing can be represented accurately
        test_times = [0.001, 0.999, 1.234, 59.999, 3661.500]
        
        for time_seconds in test_times:
            formatted_time = self._format_webvtt_time(time_seconds)
            parsed_time = self._parse_webvtt_time(formatted_time)
            
            # Should be accurate to milliseconds
            assert abs(parsed_time - time_seconds) < 0.001
    
    def test_timing_boundaries(self):
        """Test timing boundary conditions."""
        boundary_times = [
            0.0,      # Zero time
            0.001,    # Minimum positive time
            59.999,   # Just under minute boundary  
            60.0,     # Minute boundary
            3599.999, # Just under hour boundary
            3600.0,   # Hour boundary
            7200.5    # Multi-hour with fraction
        ]
        
        for time_seconds in boundary_times:
            formatted_time = self._format_webvtt_time(time_seconds)
            
            # Should format correctly
            assert re.match(r'^\d{2}:\d{2}:\d{2}\.\d{3}$', formatted_time)
            
            # Should parse back correctly
            parsed_time = self._parse_webvtt_time(formatted_time)
            assert abs(parsed_time - time_seconds) < 0.001
    
    def test_timing_consistency_with_transcription_result(self):
        """Test timing consistency with TranscriptionResult."""
        segments = [
            TranscriptSegment(start=0.0, end=3.5, text="First", speaker=None, confidence=None),
            TranscriptSegment(start=4.0, end=8.2, text="Second", speaker=None, confidence=None),
            TranscriptSegment(start=9.0, end=12.5, text="Third", speaker=None, confidence=None)
        ]
        
        result = TranscriptionResult(
            text="First Second Third",
            segments=segments,
            language="en",
            confidence=0.9
        )
        
        # Generate WebVTT
        webvtt_content = self._generate_webvtt_from_result(result)
        
        # Validate timing consistency
        validator = WebVTTValidator()
        webvtt_file = validator.parse_webvtt(webvtt_content)
        
        assert len(webvtt_file.cues) == len(segments)
        
        for i, (cue, segment) in enumerate(zip(webvtt_file.cues, segments)):
            assert abs(cue.start_time - segment.start) < 0.001, f"Start time mismatch in cue {i}"
            assert abs(cue.end_time - segment.end) < 0.001, f"End time mismatch in cue {i}"
            assert cue.text.strip() == segment.text, f"Text mismatch in cue {i}"
    
    def _format_webvtt_time(self, seconds: float) -> str:
        """Format time for WebVTT."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    
    def _parse_webvtt_time(self, time_str: str) -> float:
        """Parse WebVTT time string to seconds."""
        pattern = re.compile(r'^(\d{2}):(\d{2}):(\d{2})\.(\d{3})$')
        match = pattern.match(time_str)
        
        if not match:
            raise ValueError(f"Invalid time format: {time_str}")
        
        hours, minutes, seconds, milliseconds = match.groups()
        return (
            int(hours) * 3600 +
            int(minutes) * 60 +
            int(seconds) +
            int(milliseconds) / 1000.0
        )
    
    def _generate_webvtt_from_result(self, result: TranscriptionResult) -> str:
        """Generate WebVTT from transcription result."""
        lines = ["WEBVTT"]
        
        if result.language:
            lines[0] += f" - {result.language}"
        
        lines.append("")
        
        for i, segment in enumerate(result.segments, 1):
            lines.append(f"{i}")
            
            start_time = self._format_webvtt_time(segment.start)
            end_time = self._format_webvtt_time(segment.end)
            lines.append(f"{start_time} --> {end_time}")
            
            lines.append(segment.text)
            lines.append("")
        
        return "\n".join(lines)


class TestWebVTTSpecCompliance:
    """Test WebVTT specification compliance."""
    
    def test_webvtt_spec_examples(self):
        """Test examples from WebVTT specification."""
        # Example from WebVTT spec
        spec_example = """WEBVTT

STYLE
::cue {
  background-image: linear-gradient(to bottom, dimgray, lightgray);
  color: papayawhip;
}

NOTE comment blocks can be used between style and cue blocks.

introduction
00:00:20.000 --> 00:00:24.400
[background music]

00:00:24.600 --> 00:00:27.800
Good day everyone, my name is John Smith

00:00:30.200 --> 00:00:32.600
and I would like to welcome you all."""
        
        validator = WebVTTValidator()
        
        # Should parse without critical errors
        # (may have warnings for unsupported features like STYLE)
        try:
            webvtt_file = validator.parse_webvtt(spec_example)
            assert len(webvtt_file.cues) >= 3
        except Exception as e:
            pytest.fail(f"Failed to parse spec example: {e}")
    
    def test_unicode_support(self):
        """Test Unicode character support in WebVTT."""
        unicode_webvtt = """WEBVTT

00:00:00.000 --> 00:00:03.500
English: Hello, world!

00:00:04.000 --> 00:00:07.500
Français: Bonjour le monde!

00:00:08.000 --> 00:00:11.500
中文: 你好世界!

00:00:12.000 --> 00:00:15.500
العربية: مرحبا بالعالم!

00:00:16.000 --> 00:00:19.500
Русский: Привет, мир!
"""
        
        validator = WebVTTValidator()
        is_valid, errors, warnings = validator.validate_webvtt_content(unicode_webvtt)
        
        assert is_valid
        assert len(errors) == 0
    
    def test_speaker_identification_tags(self):
        """Test speaker identification using voice tags."""
        speaker_webvtt = """WEBVTT

00:00:00.000 --> 00:00:03.500
<v Dr. Smith>Good morning, everyone.

00:00:04.000 --> 00:00:07.500
<v Prof. Johnson>Thank you for having me.

00:00:08.000 --> 00:00:11.500
<v Dr. Smith>Let's begin with the first topic.
"""
        
        validator = WebVTTValidator()
        is_valid, errors, warnings = validator.validate_webvtt_content(speaker_webvtt)
        
        assert is_valid
        assert len(errors) == 0
    
    def test_class_annotations(self):
        """Test class-based styling annotations."""
        class_webvtt = """WEBVTT

00:00:00.000 --> 00:00:03.500
<c.highlight>Important announcement!</c>

00:00:04.000 --> 00:00:07.500
<c.whisper>Quietly spoken text</c>

00:00:08.000 --> 00:00:11.500
Normal text with <c.emphasis>emphasized portion</c>.
"""
        
        validator = WebVTTValidator()
        is_valid, errors, warnings = validator.validate_webvtt_content(class_webvtt)
        
        assert is_valid
        assert len(errors) == 0