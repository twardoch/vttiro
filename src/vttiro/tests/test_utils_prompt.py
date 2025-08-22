# this_file: src/vttiro/tests/test_utils_prompt.py
"""Tests for prompt building utilities.

This module tests the prompt building functions used by transcription
providers to generate effective AI prompts for various output formats.
"""

import pytest

from vttiro.utils.prompt import (
    build_webvtt_prompt,
    build_plain_text_prompt,
    optimize_prompt_for_provider,
    validate_prompt_length,
    extract_context_from_metadata
)


class TestBuildWebvttPrompt:
    """Test WebVTT prompt building functionality."""
    
    def test_basic_webvtt_prompt(self):
        """Test basic WebVTT prompt generation."""
        prompt = build_webvtt_prompt()
        
        assert "WEBVTT" in prompt
        assert "TRANSCRIPTION TASK" in prompt
        assert "STRICT OUTPUT FORMAT REQUIREMENTS" in prompt
        assert "TIMING REQUIREMENTS" in prompt
        assert "TEXT QUALITY REQUIREMENTS" in prompt
        assert "SPEAKER IDENTIFICATION" in prompt
        assert "EXAMPLE OUTPUT FORMAT" in prompt
        
    def test_webvtt_with_language(self):
        """Test WebVTT prompt with specific language."""
        prompt = build_webvtt_prompt(language="en")
        
        assert "Target language: EN" in prompt
        assert "LANGUAGE REQUIREMENTS" in prompt
        assert "Auto-detect" not in prompt
        
    def test_webvtt_without_language(self):
        """Test WebVTT prompt with auto-detection."""
        prompt = build_webvtt_prompt(language=None)
        
        assert "Auto-detect the spoken language" in prompt
        assert "Target language:" not in prompt
        
    def test_webvtt_with_context(self):
        """Test WebVTT prompt with context information."""
        context = {
            "topic": "Meeting recording",
            "participants": ["Alice", "Bob"],
            "domain": "business"
        }
        prompt = build_webvtt_prompt(context=context)
        
        assert "CONTEXT INFORMATION" in prompt
        assert "Topic/subject: Meeting recording" in prompt
        assert "Participants: Alice, Bob" in prompt
        assert "Domain: business" in prompt
        
    def test_webvtt_with_speaker_diarization(self):
        """Test WebVTT prompt with speaker diarization enabled."""
        prompt = build_webvtt_prompt(include_speaker_diarization=True)
        
        assert "SPEAKER IDENTIFICATION" in prompt
        assert "<v Speaker>" in prompt
        assert "speaker identification" in prompt.lower()
        
    def test_webvtt_without_speaker_diarization(self):
        """Test WebVTT prompt without speaker diarization."""
        prompt = build_webvtt_prompt(include_speaker_diarization=False)
        
        assert "SPEAKER IDENTIFICATION" not in prompt
        assert "<v Speaker>" not in prompt
        
    def test_webvtt_with_emotions(self):
        """Test WebVTT prompt with emotion detection."""
        prompt = build_webvtt_prompt(include_emotions=True)
        
        assert "EMOTION DETECTION" in prompt
        assert "emotional tone" in prompt.lower()
        
    def test_webvtt_without_emotions(self):
        """Test WebVTT prompt without emotion detection."""
        prompt = build_webvtt_prompt(include_emotions=False)
        
        assert "EMOTION DETECTION" not in prompt
        
    def test_webvtt_custom_segment_duration(self):
        """Test WebVTT prompt with custom segment duration."""
        prompt = build_webvtt_prompt(max_segment_duration=45.0)
        
        assert "Maximum segment duration: 45.0 seconds" in prompt
        
    def test_webvtt_all_options(self):
        """Test WebVTT prompt with all options enabled."""
        context = {"topic": "Technical presentation", "domain": "technology"}
        prompt = build_webvtt_prompt(
            language="en",
            context=context,
            include_speaker_diarization=True,
            include_emotions=True,
            max_segment_duration=20.0
        )
        
        assert "Target language: EN" in prompt
        assert "Technical presentation" in prompt
        assert "SPEAKER IDENTIFICATION" in prompt
        assert "EMOTION DETECTION" in prompt
        assert "Maximum segment duration: 20.0 seconds" in prompt


class TestBuildPlainTextPrompt:
    """Test plain text prompt building functionality."""
    
    def test_basic_plain_text_prompt(self):
        """Test basic plain text prompt generation."""
        prompt = build_plain_text_prompt()
        
        assert "TRANSCRIPTION TASK" in prompt
        assert "Convert the provided audio to clean, readable text" in prompt
        assert "TEXT QUALITY REQUIREMENTS" in prompt
        assert "OUTPUT FORMAT" in prompt
        assert "Plain text transcription" in prompt
        
    def test_plain_text_with_language(self):
        """Test plain text prompt with specific language."""
        prompt = build_plain_text_prompt(language="es")
        
        assert "Target language: ES" in prompt
        assert "LANGUAGE REQUIREMENTS" in prompt
        
    def test_plain_text_with_context(self):
        """Test plain text prompt with context."""
        context = {"topic": "Interview", "style": "formal"}
        prompt = build_plain_text_prompt(context=context)
        
        assert "CONTEXT INFORMATION" in prompt
        assert "Topic/subject: Interview" in prompt
        assert "Style: formal" in prompt
        
    def test_plain_text_with_paragraphs(self):
        """Test plain text prompt with paragraph formatting."""
        prompt = build_plain_text_prompt(include_paragraphs=True)
        
        assert "paragraph breaks" in prompt.lower()
        assert "natural speech boundaries" in prompt.lower()
        
    def test_plain_text_without_paragraphs(self):
        """Test plain text prompt without paragraph formatting."""
        prompt = build_plain_text_prompt(include_paragraphs=False)
        
        assert "paragraph" not in prompt.lower()


class TestOptimizePromptForProvider:
    """Test provider-specific prompt optimization."""
    
    def test_gemini_optimization(self):
        """Test prompt optimization for Gemini."""
        base_prompt = "Transcribe this audio to WebVTT format."
        optimized = optimize_prompt_for_provider(base_prompt, "gemini")
        
        assert base_prompt in optimized
        assert "GEMINI-SPECIFIC OPTIMIZATIONS" in optimized
        assert "multimodal understanding" in optimized.lower()
        assert "audio understanding capabilities" in optimized.lower()
        
    def test_openai_optimization(self):
        """Test prompt optimization for OpenAI."""
        base_prompt = "Transcribe this audio to WebVTT format."
        optimized = optimize_prompt_for_provider(base_prompt, "openai")
        
        assert base_prompt in optimized
        assert "OPENAI-SPECIFIC OPTIMIZATIONS" in optimized
        assert "Whisper model" in optimized.lower()
        
    def test_assemblyai_optimization(self):
        """Test prompt optimization for AssemblyAI."""
        base_prompt = "Transcribe this audio to WebVTT format."
        optimized = optimize_prompt_for_provider(base_prompt, "assemblyai")
        
        assert base_prompt in optimized
        assert "ASSEMBLYAI-SPECIFIC OPTIMIZATIONS" in optimized
        assert "Universal-2 model" in optimized.lower()
        
    def test_deepgram_optimization(self):
        """Test prompt optimization for Deepgram."""
        base_prompt = "Transcribe this audio to WebVTT format."
        optimized = optimize_prompt_for_provider(base_prompt, "deepgram")
        
        assert base_prompt in optimized
        assert "DEEPGRAM-SPECIFIC OPTIMIZATIONS" in optimized
        assert "Nova-3 model" in optimized.lower()
        
    def test_unknown_provider_passthrough(self):
        """Test that unknown providers return base prompt unchanged."""
        base_prompt = "Transcribe this audio to WebVTT format."
        optimized = optimize_prompt_for_provider(base_prompt, "unknown_provider")
        
        assert optimized == base_prompt
        
    def test_empty_provider_passthrough(self):
        """Test that empty provider name returns base prompt unchanged."""
        base_prompt = "Transcribe this audio to WebVTT format."
        optimized = optimize_prompt_for_provider(base_prompt, "")
        
        assert optimized == base_prompt


class TestValidatePromptLength:
    """Test prompt length validation functionality."""
    
    def test_short_prompt_valid(self):
        """Test that short prompts are valid."""
        prompt = "Short prompt"
        is_valid, message = validate_prompt_length(prompt, max_tokens=100)
        
        assert is_valid is True
        assert "valid" in message.lower()
        
    def test_long_prompt_invalid(self):
        """Test that overly long prompts are invalid."""
        prompt = "word " * 5000  # Very long prompt
        is_valid, message = validate_prompt_length(prompt, max_tokens=100)
        
        assert is_valid is False
        assert "exceeds" in message.lower()
        assert "100" in message
        
    def test_exact_limit_valid(self):
        """Test prompt at exact token limit."""
        prompt = "word " * 50  # Approximately 100 tokens
        is_valid, message = validate_prompt_length(prompt, max_tokens=100)
        
        # Should be valid since it's close to but not over the limit
        assert is_valid is True
        
    def test_custom_token_limit(self):
        """Test custom token limit."""
        prompt = "word " * 100
        is_valid, message = validate_prompt_length(prompt, max_tokens=50)
        
        assert is_valid is False
        assert "50" in message
        
    def test_empty_prompt_valid(self):
        """Test that empty prompt is valid."""
        is_valid, message = validate_prompt_length("", max_tokens=100)
        
        assert is_valid is True
        
    def test_default_token_limit(self):
        """Test default token limit (4000)."""
        prompt = "Normal length prompt for testing"
        is_valid, message = validate_prompt_length(prompt)
        
        assert is_valid is True


class TestExtractContextFromMetadata:
    """Test context extraction from metadata."""
    
    def test_extract_basic_context(self):
        """Test extraction of basic context fields."""
        metadata = {
            "topic": "Meeting recording",
            "domain": "business", 
            "participants": ["Alice", "Bob"],
            "language": "en",
            "other_field": "ignored"
        }
        
        context = extract_context_from_metadata(metadata)
        
        assert context["topic"] == "Meeting recording"
        assert context["domain"] == "business"
        assert context["participants"] == ["Alice", "Bob"]
        assert context["language"] == "en"
        assert "other_field" not in context
        
    def test_extract_empty_metadata(self):
        """Test extraction from empty metadata."""
        context = extract_context_from_metadata({})
        assert context == {}
        
    def test_extract_partial_metadata(self):
        """Test extraction with only some relevant fields."""
        metadata = {
            "topic": "Interview",
            "unrelated_field": "value",
            "another_field": 123
        }
        
        context = extract_context_from_metadata(metadata)
        
        assert context["topic"] == "Interview"
        assert len(context) == 1
        assert "unrelated_field" not in context
        
    def test_extract_all_supported_fields(self):
        """Test extraction of all supported context fields."""
        metadata = {
            "topic": "Technical presentation",
            "domain": "technology",
            "participants": ["Speaker", "Moderator"],
            "language": "en",
            "style": "formal",
            "purpose": "education",
            "duration": "30 minutes",
            "setting": "conference room"
        }
        
        context = extract_context_from_metadata(metadata)
        
        # Should extract all known context fields
        expected_fields = ["topic", "domain", "participants", "language", "style", "purpose", "duration", "setting"]
        for field in expected_fields:
            if field in metadata:
                assert field in context
                assert context[field] == metadata[field]
                
    def test_extract_none_metadata(self):
        """Test extraction from None metadata."""
        context = extract_context_from_metadata(None)
        assert context == {}
        
    def test_extract_with_list_values(self):
        """Test extraction with list values."""
        metadata = {
            "participants": ["Alice", "Bob", "Charlie"],
            "topics": ["intro", "main", "conclusion"]
        }
        
        context = extract_context_from_metadata(metadata)
        
        assert context["participants"] == ["Alice", "Bob", "Charlie"]
        # topics might not be a supported field, depending on implementation
        
    def test_extract_with_nested_values(self):
        """Test extraction handles nested dictionaries appropriately."""
        metadata = {
            "topic": "Meeting",
            "nested": {"inner": "value"},
            "participants": ["Alice"]
        }
        
        context = extract_context_from_metadata(metadata)
        
        assert context["topic"] == "Meeting"
        assert context["participants"] == ["Alice"]
        # Nested values should be handled appropriately (likely ignored or flattened)