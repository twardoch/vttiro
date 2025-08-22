# this_file: src/vttiro/tests/test_golden_files.py
"""Golden file tests for VTTiro output consistency.

This module implements golden file (snapshot) testing to ensure output formats
remain consistent across code changes. Golden files contain expected outputs
that are compared against actual outputs during testing.
"""

import pytest
import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import asdict

from vttiro.core.types import TranscriptionResult, TranscriptSegment
from vttiro.core.config import VttiroConfig
from vttiro.core.registry import get_registry


class GoldenFileManager:
    """Manager for golden file operations."""
    
    def __init__(self, test_fixtures_dir: Path):
        """Initialize golden file manager.
        
        Args:
            test_fixtures_dir: Directory containing test fixtures
        """
        self.fixtures_dir = test_fixtures_dir
        self.golden_dir = test_fixtures_dir / "golden"
        self.golden_dir.mkdir(parents=True, exist_ok=True)
    
    def get_golden_file_path(self, test_name: str, file_type: str) -> Path:
        """Get path to golden file.
        
        Args:
            test_name: Name of the test
            file_type: Type of file (webvtt, json, etc.)
            
        Returns:
            Path to golden file
        """
        return self.golden_dir / f"{test_name}.{file_type}"
    
    def save_golden_file(self, test_name: str, content: str, file_type: str) -> None:
        """Save content as golden file.
        
        Args:
            test_name: Name of the test
            content: Content to save
            file_type: Type of file
        """
        golden_file = self.get_golden_file_path(test_name, file_type)
        golden_file.write_text(content, encoding='utf-8')
    
    def load_golden_file(self, test_name: str, file_type: str) -> str:
        """Load golden file content.
        
        Args:
            test_name: Name of the test
            file_type: Type of file
            
        Returns:
            Golden file content
            
        Raises:
            FileNotFoundError: If golden file doesn't exist
        """
        golden_file = self.get_golden_file_path(test_name, file_type)
        if not golden_file.exists():
            raise FileNotFoundError(f"Golden file not found: {golden_file}")
        return golden_file.read_text(encoding='utf-8')
    
    def compare_with_golden(
        self, 
        test_name: str, 
        actual_content: str, 
        file_type: str,
        update_golden: bool = False
    ) -> bool:
        """Compare actual content with golden file.
        
        Args:
            test_name: Name of the test
            actual_content: Actual content to compare
            file_type: Type of file
            update_golden: Whether to update golden file if different
            
        Returns:
            True if content matches golden file
        """
        try:
            expected_content = self.load_golden_file(test_name, file_type)
        except FileNotFoundError:
            if update_golden:
                self.save_golden_file(test_name, actual_content, file_type)
                return True
            else:
                raise AssertionError(f"Golden file not found: {test_name}.{file_type}")
        
        if actual_content.strip() == expected_content.strip():
            return True
        
        if update_golden:
            self.save_golden_file(test_name, actual_content, file_type)
            return True
        
        # Generate diff for debugging
        self._save_diff_file(test_name, expected_content, actual_content, file_type)
        return False
    
    def _save_diff_file(
        self, 
        test_name: str, 
        expected: str, 
        actual: str, 
        file_type: str
    ) -> None:
        """Save diff file for debugging.
        
        Args:
            test_name: Name of the test
            expected: Expected content
            actual: Actual content
            file_type: Type of file
        """
        diff_content = f"""=== EXPECTED ===
{expected}

=== ACTUAL ===
{actual}

=== DIFF INFO ===
Expected length: {len(expected)}
Actual length: {len(actual)}
Expected hash: {hashlib.md5(expected.encode()).hexdigest()}
Actual hash: {hashlib.md5(actual.encode()).hexdigest()}
"""
        
        diff_file = self.golden_dir / f"{test_name}.{file_type}.diff"
        diff_file.write_text(diff_content, encoding='utf-8')


@pytest.fixture
def golden_manager(tmp_path):
    """Fixture providing golden file manager."""
    fixtures_dir = tmp_path / "test_fixtures"
    fixtures_dir.mkdir()
    return GoldenFileManager(fixtures_dir)


@pytest.fixture
def sample_transcription_result():
    """Fixture providing sample transcription result."""
    segments = [
        TranscriptSegment(
            start=0.0,
            end=3.5,
            text="Hello, welcome to our presentation.",
            speaker="Speaker 1",
            confidence=0.95
        ),
        TranscriptSegment(
            start=4.0,
            end=8.2,
            text="Today we'll discuss the new features.",
            speaker="Speaker 1", 
            confidence=0.92
        ),
        TranscriptSegment(
            start=9.0,
            end=12.5,
            text="Thank you for joining us.",
            speaker="Speaker 2",
            confidence=0.88
        )
    ]
    
    return TranscriptionResult(
        text="Hello, welcome to our presentation. Today we'll discuss the new features. Thank you for joining us.",
        segments=segments,
        language="en",
        confidence=0.92
    )


class TestWebVTTGoldenFiles:
    """Test WebVTT output format golden files."""
    
    def test_basic_webvtt_output(self, golden_manager, sample_transcription_result):
        """Test basic WebVTT output format."""
        webvtt_content = self._generate_webvtt(sample_transcription_result)
        
        assert golden_manager.compare_with_golden(
            "basic_webvtt", webvtt_content, "vtt", update_golden=True
        )
    
    def test_webvtt_with_speaker_labels(self, golden_manager, sample_transcription_result):
        """Test WebVTT output with speaker labels."""
        webvtt_content = self._generate_webvtt(sample_transcription_result, include_speakers=True)
        
        assert golden_manager.compare_with_golden(
            "webvtt_with_speakers", webvtt_content, "vtt", update_golden=True
        )
    
    def test_webvtt_with_confidence_notes(self, golden_manager, sample_transcription_result):
        """Test WebVTT output with confidence annotations."""
        webvtt_content = self._generate_webvtt(sample_transcription_result, include_confidence=True)
        
        assert golden_manager.compare_with_golden(
            "webvtt_with_confidence", webvtt_content, "vtt", update_golden=True
        )
    
    def test_webvtt_long_text_wrapping(self, golden_manager):
        """Test WebVTT output with long text that needs wrapping."""
        long_segment = TranscriptSegment(
            start=0.0,
            end=10.0,
            text="This is a very long segment of text that should be wrapped appropriately in the WebVTT output to ensure proper display on various devices and players while maintaining readability and following WebVTT specifications.",
            speaker="Speaker 1",
            confidence=0.90
        )
        
        result = TranscriptionResult(
            text=long_segment.text,
            segments=[long_segment],
            language="en",
            confidence=0.90
        )
        
        webvtt_content = self._generate_webvtt(result, max_line_length=60)
        
        assert golden_manager.compare_with_golden(
            "webvtt_long_text", webvtt_content, "vtt", update_golden=True
        )
    
    def test_webvtt_special_characters(self, golden_manager):
        """Test WebVTT output with special characters and formatting."""
        special_segments = [
            TranscriptSegment(
                start=0.0,
                end=3.0,
                text="Text with <special> &characters; \"quotes\" and 'apostrophes'",
                speaker="Speaker 1",
                confidence=0.85
            ),
            TranscriptSegment(
                start=4.0,
                end=7.0,
                text="Unicode test: café, naïve, résumé, 中文, العربية",
                speaker="Speaker 2",
                confidence=0.80
            )
        ]
        
        result = TranscriptionResult(
            text=" ".join(seg.text for seg in special_segments),
            segments=special_segments,
            language="en",
            confidence=0.83
        )
        
        webvtt_content = self._generate_webvtt(result)
        
        assert golden_manager.compare_with_golden(
            "webvtt_special_chars", webvtt_content, "vtt", update_golden=True
        )
    
    def _generate_webvtt(
        self, 
        result: TranscriptionResult, 
        include_speakers: bool = False,
        include_confidence: bool = False,
        max_line_length: Optional[int] = None
    ) -> str:
        """Generate WebVTT content from transcription result.
        
        Args:
            result: Transcription result
            include_speakers: Whether to include speaker labels
            include_confidence: Whether to include confidence notes
            max_line_length: Maximum line length for text wrapping
            
        Returns:
            WebVTT formatted string
        """
        lines = ["WEBVTT", ""]
        
        if result.language:
            lines[0] += f" - {result.language}"
        
        # Add metadata
        if include_confidence and result.confidence:
            lines.extend([
                "NOTE",
                f"Overall confidence: {result.confidence:.2f}",
                ""
            ])
        
        for i, segment in enumerate(result.segments, 1):
            # Add cue identifier
            lines.append(f"{i}")
            
            # Add timing
            start_time = self._format_webvtt_time(segment.start)
            end_time = self._format_webvtt_time(segment.end)
            lines.append(f"{start_time} --> {end_time}")
            
            # Prepare text
            text = segment.text
            if include_speakers and segment.speaker:
                text = f"<v {segment.speaker}>{text}</v>"
            
            if include_confidence and segment.confidence:
                text += f" <c.confidence>(confidence: {segment.confidence:.2f})</c>"
            
            # Handle text wrapping
            if max_line_length and len(text) > max_line_length:
                wrapped_lines = self._wrap_text(text, max_line_length)
                lines.extend(wrapped_lines)
            else:
                lines.append(text)
            
            lines.append("")  # Empty line between cues
        
        return "\n".join(lines)
    
    def _format_webvtt_time(self, seconds: float) -> str:
        """Format time for WebVTT.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            WebVTT formatted time string
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    
    def _wrap_text(self, text: str, max_length: int) -> List[str]:
        """Wrap text to specified length.
        
        Args:
            text: Text to wrap
            max_length: Maximum line length
            
        Returns:
            List of wrapped lines
        """
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            if len(current_line + " " + word) <= max_length:
                if current_line:
                    current_line += " " + word
                else:
                    current_line = word
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        return lines


class TestJSONGoldenFiles:
    """Test JSON output format golden files."""
    
    def test_transcription_result_json(self, golden_manager, sample_transcription_result):
        """Test transcription result JSON output."""
        json_content = self._transcription_result_to_json(sample_transcription_result)
        
        assert golden_manager.compare_with_golden(
            "transcription_result", json_content, "json", update_golden=True
        )
    
    def test_minimal_transcription_result_json(self, golden_manager):
        """Test minimal transcription result JSON."""
        minimal_result = TranscriptionResult(
            text="Simple transcription.",
            segments=[],
            language=None,
            confidence=None
        )
        
        json_content = self._transcription_result_to_json(minimal_result)
        
        assert golden_manager.compare_with_golden(
            "minimal_transcription", json_content, "json", update_golden=True
        )
    
    def test_detailed_transcription_result_json(self, golden_manager):
        """Test detailed transcription result with all fields."""
        detailed_segments = [
            TranscriptSegment(
                start=0.0,
                end=2.5,
                text="Detailed segment with all metadata.",
                speaker="Dr. Smith",
                confidence=0.97
            ),
            TranscriptSegment(
                start=3.0,
                end=5.8,
                text="Another segment with different speaker.",
                speaker="Prof. Johnson", 
                confidence=0.94
            )
        ]
        
        detailed_result = TranscriptionResult(
            text="Detailed segment with all metadata. Another segment with different speaker.",
            segments=detailed_segments,
            language="en-US",
            confidence=0.955
        )
        
        json_content = self._transcription_result_to_json(detailed_result)
        
        assert golden_manager.compare_with_golden(
            "detailed_transcription", json_content, "json", update_golden=True
        )
    
    def test_config_json_output(self, golden_manager):
        """Test configuration JSON output."""
        config = VttiroConfig(
            provider="gemini",
            language="en",
            output_format="webvtt",
            enable_speaker_diarization=True,
            enable_emotion_detection=False,
            max_retries=3,
            verbose=True
        )
        
        json_content = self._config_to_json(config)
        
        assert golden_manager.compare_with_golden(
            "config_output", json_content, "json", update_golden=True
        )
    
    def test_provider_comparison_json(self, golden_manager):
        """Test provider comparison JSON output."""
        registry = get_registry()
        comparison = registry.compare_providers()
        
        json_content = json.dumps(comparison, indent=2, sort_keys=True)
        
        assert golden_manager.compare_with_golden(
            "provider_comparison", json_content, "json", update_golden=True
        )
    
    def _transcription_result_to_json(self, result: TranscriptionResult) -> str:
        """Convert transcription result to JSON.
        
        Args:
            result: Transcription result
            
        Returns:
            JSON formatted string
        """
        # Convert to dict, handling dataclass serialization
        result_dict = {
            "text": result.text,
            "segments": [
                {
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                    "speaker": seg.speaker,
                    "confidence": seg.confidence
                }
                for seg in result.segments
            ],
            "language": result.language,
            "confidence": result.confidence
        }
        
        return json.dumps(result_dict, indent=2, sort_keys=True)
    
    def _config_to_json(self, config: VttiroConfig) -> str:
        """Convert config to JSON.
        
        Args:
            config: VTTiro configuration
            
        Returns:
            JSON formatted string
        """
        return json.dumps(config.to_dict(), indent=2, sort_keys=True)


class TestErrorMessageGoldenFiles:
    """Test error message format golden files."""
    
    def test_validation_error_messages(self, golden_manager):
        """Test validation error message formats."""
        from vttiro.core.errors import ValidationError
        
        error_scenarios = [
            ("invalid_language", "Invalid language code format: xyz123"),
            ("missing_file", "Audio file not found: /path/to/missing.wav"),
            ("invalid_format", "Unsupported file format: .xyz"),
            ("file_too_large", "File size exceeds limit: 1.2GB > 100MB"),
            ("corrupted_file", "Corrupted audio file: invalid WAV header")
        ]
        
        for scenario_name, error_message in error_scenarios:
            error_dict = {
                "error_type": "ValidationError",
                "message": error_message,
                "timestamp": "2023-01-01T12:00:00Z",  # Fixed for golden file
                "context": {
                    "scenario": scenario_name,
                    "severity": "error"
                }
            }
            
            json_content = json.dumps(error_dict, indent=2, sort_keys=True)
            
            assert golden_manager.compare_with_golden(
                f"error_{scenario_name}", json_content, "json", update_golden=True
            )
    
    def test_processing_error_messages(self, golden_manager):
        """Test processing error message formats."""
        from vttiro.core.errors import ProcessingError
        
        error_scenarios = [
            ("api_timeout", "Request timeout after 300 seconds"),
            ("api_quota", "API quota exceeded: 1000 requests/hour limit reached"),
            ("auth_failed", "Authentication failed: invalid API key"),
            ("service_unavailable", "Provider service temporarily unavailable")
        ]
        
        for scenario_name, error_message in error_scenarios:
            error_dict = {
                "error_type": "ProcessingError",
                "message": error_message,
                "timestamp": "2023-01-01T12:00:00Z",
                "context": {
                    "scenario": scenario_name,
                    "provider": "gemini",
                    "retry_count": 2
                }
            }
            
            json_content = json.dumps(error_dict, indent=2, sort_keys=True)
            
            assert golden_manager.compare_with_golden(
                f"processing_error_{scenario_name}", json_content, "json", update_golden=True
            )


class TestProviderResponseGoldenFiles:
    """Test provider response format golden files."""
    
    def test_gemini_response_format(self, golden_manager):
        """Test Gemini provider response format."""
        mock_response = {
            "model": "gemini-2.0-flash",
            "response": {
                "text": "This is a sample transcription from Gemini.",
                "segments": [
                    {
                        "start_time": "0.0s",
                        "end_time": "3.5s", 
                        "text": "This is a sample transcription from Gemini.",
                        "confidence": 0.95
                    }
                ],
                "language": "en"
            },
            "metadata": {
                "processing_time": 2.34,
                "model_version": "2.0-flash",
                "features_used": ["transcription", "diarization"]
            }
        }
        
        json_content = json.dumps(mock_response, indent=2, sort_keys=True)
        
        assert golden_manager.compare_with_golden(
            "gemini_response", json_content, "json", update_golden=True
        )
    
    def test_openai_response_format(self, golden_manager):
        """Test OpenAI provider response format."""
        mock_response = {
            "model": "whisper-1",
            "text": "This is a sample transcription from OpenAI Whisper.",
            "segments": [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 3.5,
                    "text": "This is a sample transcription from OpenAI Whisper.",
                    "tokens": [123, 456, 789],
                    "temperature": 0.0,
                    "avg_logprob": -0.123,
                    "compression_ratio": 1.456,
                    "no_speech_prob": 0.001
                }
            ],
            "language": "en"
        }
        
        json_content = json.dumps(mock_response, indent=2, sort_keys=True)
        
        assert golden_manager.compare_with_golden(
            "openai_response", json_content, "json", update_golden=True
        )
    
    def test_assemblyai_response_format(self, golden_manager):
        """Test AssemblyAI provider response format."""
        mock_response = {
            "id": "abc123def456",
            "status": "completed",
            "acoustic_model": "assemblyai_default",
            "language_model": "assemblyai_default",
            "language_code": "en_us",
            "audio_url": "https://example.com/audio.wav",
            "text": "This is a sample transcription from AssemblyAI.",
            "words": [
                {
                    "text": "This",
                    "start": 0,
                    "end": 240,
                    "confidence": 0.97,
                    "speaker": "A"
                },
                {
                    "text": "is",
                    "start": 240,
                    "end": 420,
                    "confidence": 0.99,
                    "speaker": "A"
                }
            ],
            "utterances": [
                {
                    "text": "This is a sample transcription from AssemblyAI.",
                    "start": 0,
                    "end": 3500,
                    "confidence": 0.98,
                    "speaker": "A"
                }
            ],
            "confidence": 0.98,
            "audio_duration": 3.5
        }
        
        json_content = json.dumps(mock_response, indent=2, sort_keys=True)
        
        assert golden_manager.compare_with_golden(
            "assemblyai_response", json_content, "json", update_golden=True
        )


class TestGoldenFileUtilities:
    """Test golden file utility functions."""
    
    def test_golden_file_creation(self, golden_manager):
        """Test golden file creation and management."""
        test_content = "This is test content for golden file management."
        
        # Save golden file
        golden_manager.save_golden_file("test_creation", test_content, "txt")
        
        # Verify file exists
        golden_file = golden_manager.get_golden_file_path("test_creation", "txt")
        assert golden_file.exists()
        
        # Load and verify content
        loaded_content = golden_manager.load_golden_file("test_creation", "txt")
        assert loaded_content == test_content
    
    def test_golden_file_comparison_success(self, golden_manager):
        """Test successful golden file comparison."""
        test_content = "Consistent content for comparison."
        
        # Save golden file
        golden_manager.save_golden_file("test_comparison", test_content, "txt")
        
        # Compare with same content
        assert golden_manager.compare_with_golden(
            "test_comparison", test_content, "txt"
        )
    
    def test_golden_file_comparison_failure(self, golden_manager):
        """Test failed golden file comparison."""
        original_content = "Original content."
        modified_content = "Modified content."
        
        # Save golden file
        golden_manager.save_golden_file("test_failure", original_content, "txt")
        
        # Compare with different content
        assert not golden_manager.compare_with_golden(
            "test_failure", modified_content, "txt"
        )
        
        # Verify diff file was created
        diff_file = golden_manager.golden_dir / "test_failure.txt.diff"
        assert diff_file.exists()
    
    def test_golden_file_update_mode(self, golden_manager):
        """Test golden file update mode."""
        original_content = "Original content."
        updated_content = "Updated content."
        
        # Save original golden file
        golden_manager.save_golden_file("test_update", original_content, "txt")
        
        # Update with new content
        assert golden_manager.compare_with_golden(
            "test_update", updated_content, "txt", update_golden=True
        )
        
        # Verify content was updated
        loaded_content = golden_manager.load_golden_file("test_update", "txt")
        assert loaded_content == updated_content
    
    def test_missing_golden_file_handling(self, golden_manager):
        """Test handling of missing golden files."""
        test_content = "Content for missing golden file test."
        
        # Try to compare with non-existent golden file
        with pytest.raises(AssertionError, match="Golden file not found"):
            golden_manager.compare_with_golden("missing_file", test_content, "txt")
        
        # With update mode, should create the file
        assert golden_manager.compare_with_golden(
            "missing_file", test_content, "txt", update_golden=True
        )
        
        # Verify file was created
        golden_file = golden_manager.get_golden_file_path("missing_file", "txt")
        assert golden_file.exists()