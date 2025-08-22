# this_file: src/vttiro/utils/prompt.py
"""Prompt building utilities for transcription providers.

This module provides functions for building effective prompts for AI
transcription models, with support for various output formats and
context-aware optimizations.

Used by:
- Provider implementations for generating API prompts  
- Core orchestration for prompt customization
- Testing infrastructure for prompt validation
"""

from typing import Any


def build_webvtt_prompt(
    language: str | None = None,
    context: dict[str, Any] | None = None,
    include_speaker_diarization: bool = True,
    include_emotions: bool = False,
    max_segment_duration: float = 30.0,
) -> str:
    """Build a comprehensive prompt for WebVTT format transcription.

    Creates a detailed prompt that instructs AI models to generate properly
    formatted WebVTT subtitles with accurate timing, speaker identification,
    and other advanced features.

    Args:
        language: Target language code (ISO 639-1) or None for auto-detection
        context: Additional context information (domain, speakers, etc.)
        include_speaker_diarization: Whether to request speaker identification
        include_emotions: Whether to request emotion detection
        max_segment_duration: Maximum duration for each subtitle segment

    Returns:
        Comprehensive prompt string for WebVTT transcription
    """
    # Base prompt with clear WebVTT format requirements
    prompt_parts = [
        "TRANSCRIPTION TASK: Perform “clean verbatim” (“intelligent verbatim”) transcription of the provided audio into WebVTT subtitle format with intelligent timestamps.",
        "",
        "STRICT OUTPUT FORMAT REQUIREMENTS:",
        "1. Start with 'WEBVTT' header",
        "2. In the WebVTT header, always indicate the language code, based on the supplied language info or on your own recognition.",
        "3. Use precise timestamps in HH:MM:SS.mmm format",
        "4. Each subtitle segment should have:",
        "   - Timestamp line: start_time --> end_time",
        "   - Text content on following lines",
        "   - Empty line to separate segments",
        "",
        "TIMING REQUIREMENTS:",
        f"- Maximum segment duration: {max_segment_duration} seconds",
        "- Minimum segment duration: 1 second",
        "- Align segments with natural speech boundaries",
        "- Try to keep semantic units of speech together",
        "- Use precise timing based on actual audio content",
        "- Prefer slight timing shifts over splitting a logical utterance across multiple segments.",
        "- If the text contains a suspensful moment or a punchline, carry it over to the next segment"
        "- Do NOT estimate or guess timestamps",
        "- If you split a longer utterance into lines or cues, ensure that the split is logical and semantic",
        "- If you split an utterance mid-sentence, start the 2nd cue in lowercase unless the word is always capitalized"
        "",
        "TEXT QUALITY REQUIREMENTS:",
        "- Transcribe what is spoken",
        "- Use your knowledge of the domain to transcribe personal, brand & other proper names, as well as jargon, consistently and according to the domain standards",
        "- Use all of Unicode. In English, use extended Latin Unicode characters for known entity names like `Lech Wałęsa`, even if the speakers says them in an americanized fashion",
        "- Omit from the transcript hesitation markers and filler words, like `uh`, `um`, `er`, `ah` etc.",
        "- Omit from the transcript discourse markers at the beginning and in the middle of sentences, like `Well`, `So`, `Now`, `I mean`, `Basically`, `kind of`, `you know`, `you see`, `you know what I mean`, etc.",
        "- Omit from the transcript false starts, self-corrections, stutters, and repetitions",
        "- Perform intelligent minimal cleanup as you emit the trans^cript, so that the text is written in a clean, orthographically correct manner with proper punctuation",
        "- Use professional typographic quote marks and punctuation appropriate for the language",
        "- Use journalistic and broadcast guidelines when creating the transcript",
    ]

    # Add language-specific instructions
    if language:
        prompt_parts.extend(
            [
                "LANGUAGE REQUIREMENTS:",
                f"- Target language: {language.upper()}",
            ]
        )
    else:
        prompt_parts.extend(
            [
                "LANGUAGE REQUIREMENTS:",
                "- Auto-detect the spoken language",
                "",
            ]
        )

    # Add speaker diarization if requested
    if include_speaker_diarization:
        prompt_parts.extend(
            [
                "SPEAKER IDENTIFICATION:",
                "- Use <v Speaker> tags for speaker identification",
                "- Format: <v Speaker 1>Text content here",
                "- Maintain consistent speaker labels throughout",
                "- Only identify speakers when clearly distinguishable",
                "",
            ]
        )

    # Add emotion detection if requested
    if include_emotions:
        prompt_parts.extend(
            [
                "EMOTION DETECTION:",
                "- Prefix cue with `[emotion]` indicators only if the emotion is clearly evident",
                "- Do not clutter the transript with emotion indicators",
                "",
            ]
        )

    # Add context-specific instructions
    if context:
        context_instructions = []

        if "domain" in context:
            context_instructions.append(f"- Content domain: {context['domain']}")

        if "speakers" in context:
            speaker_list = ", ".join(context["speakers"])
            context_instructions.append(f"- Known speakers: {speaker_list}")

        if "technical_terms" in context:
            terms_list = ", ".join(context["technical_terms"])
            context_instructions.append(f"- Technical terms to recognize: {terms_list}")

        if "topic" in context:
            context_instructions.append(f"- Topic/subject: {context['topic']}")

        if context_instructions:
            prompt_parts.extend(["CONTEXT INFORMATION:", *context_instructions, ""])

    # Add examples for clarity
    prompt_parts.extend(
        [
            "GOOD EXAMPLE:",
            "WEBVTT",
            "Language: en",
            "",
            "00:00:01.000 --> 00:00:05.000",
            "Hello and welcome to today's presentation.",
            "",
            "00:00:05.500 --> 00:00:10.250", 
            "<v Speaker 2>Thank you for having me here.",
            "",
            "00:00:10.500 --> 00:00:15.750",
            "Today we'll discuss the Wałęsa doctrine",
            "",
            "00:00:16.000 --> 00:00:18.500",
            "and its impact on modern policy.",
            "",
            "BAD EXAMPLE (DO NOT FOLLOW):",
            "WEBVTT",
            "",
            "00:00:01.000 --> 00:00:15.000",
            "Um, hello and welcome to, uh, today's presentation. Thank you for having me here. Today we'll discuss, you know, the Walesa doctrine and its impact on, uh, modern policy.",
            "",
            "WHY THE BAD EXAMPLE IS WRONG:",
            "- Too long segment duration (14 seconds vs. max 30s guideline)",
            "- Contains filler words (um, uh, you know)",
            "- Poor name transcription (Walesa vs. Wałęsa)",
            "- No speaker identification",
            "- Poor semantic segmentation",
            "",
            "IMPORTANT NOTES:",
            "- Generate ONLY the WebVTT content, no additional text",
            "- Ensure all timestamps are based on actual audio timing",
            "- Maintain consistent formatting throughout",
            "- Use precise timing for optimal subtitle synchronization",
        ]
    )

    return "\n".join(prompt_parts)


def build_plain_text_prompt(
    language: str | None = None,
    context: dict[str, Any] | None = None,
    include_speaker_labels: bool = True,
    clean_text: bool = True,
) -> str:
    """Build a prompt for plain text transcription without timing.

    Adapted from build_webvtt_prompt specifications to maintain consistency
    in transcription quality while outputting plain text format.

    Args:
        language: Target language code or None for auto-detection
        context: Additional context information
        include_speaker_labels: Whether to include speaker identification
        clean_text: Whether to clean up filler words and speech errors

    Returns:
        Prompt string for plain text transcription
    """
    prompt_parts = [
        "TRANSCRIPTION TASK: Perform "clean verbatim" ("intelligent verbatim") transcription of the provided audio into plain text format.",
        "",
        "TEXT QUALITY REQUIREMENTS:",
        "- Transcribe what is spoken",
        "- Use your knowledge of the domain to transcribe personal, brand & other proper names, as well as jargon, consistently and according to the domain standards",
        "- Use all of Unicode. In English, use extended Latin Unicode characters for known entity names like `Lech Wałęsa`, even if the speakers says them in an americanized fashion",
    ]

    if clean_text:
        prompt_parts.extend(
            [
                "- Omit from the transcript hesitation markers and filler words, like `uh`, `um`, `er`, `ah` etc.",
                "- Omit from the transcript discourse markers at the beginning and in the middle of sentences, like `Well`, `So`, `Now`, `I mean`, `Basically`, `kind of`, `you know`, `you see`, `you know what I mean`, etc.",
                "- Omit from the transcript false starts, self-corrections, stutters, and repetitions",
                "- Perform intelligent minimal cleanup as you emit the transcript, so that the text is written in a clean, orthographically correct manner with proper punctuation",
            ]
        )
    else:
        prompt_parts.extend(
            [
                "- Include all spoken words, including filler words and hesitations",
                "- Preserve speech patterns and natural flow",
                "- Maintain verbatim accuracy while ensuring readability",
            ]
        )

    prompt_parts.extend(
        [
            "- Use professional typographic quote marks and punctuation appropriate for the language",
            "- Use journalistic and broadcast guidelines when creating the transcript",
            "",
        ]
    )

    # Add language-specific instructions
    if language:
        prompt_parts.extend(
            [
                "LANGUAGE REQUIREMENTS:",
                f"- Target language: {language.upper()}",
                "",
            ]
        )
    else:
        prompt_parts.extend(
            [
                "LANGUAGE REQUIREMENTS:",
                "- Auto-detect the spoken language",
                "",
            ]
        )

    # Add speaker identification if requested
    if include_speaker_labels:
        prompt_parts.extend(
            [
                "SPEAKER IDENTIFICATION:",
                "- Use clear speaker labels: Speaker 1:, Speaker 2:, etc.",
                "- Maintain consistent speaker labels throughout",
                "- Only identify speakers when clearly distinguishable",
                "",
            ]
        )

    # Add context-specific instructions
    if context:
        context_instructions = []

        if "domain" in context:
            context_instructions.append(f"- Content domain: {context['domain']}")

        if "speakers" in context:
            speaker_list = ", ".join(context["speakers"])
            context_instructions.append(f"- Known speakers: {speaker_list}")

        if "technical_terms" in context:
            terms_list = ", ".join(context["technical_terms"])
            context_instructions.append(f"- Technical terms to recognize: {terms_list}")

        if "topic" in context:
            context_instructions.append(f"- Topic/subject: {context['topic']}")

        if context_instructions:
            prompt_parts.extend(["CONTEXT INFORMATION:", *context_instructions, ""])

    prompt_parts.extend(
        [
            "OUTPUT FORMAT:",
            "- Generate ONLY the transcribed text",
            "- Use proper paragraph breaks for logical speech segments",
            "- No additional metadata or formatting",
        ]
    )

    return "\n".join(prompt_parts)


def optimize_prompt_for_provider(base_prompt: str, provider: str) -> str:
    """Add simple provider-specific optimizations to a base prompt.

    Args:
        base_prompt: Base prompt string
        provider: Provider name (gemini, openai, assemblyai, deepgram)

    Returns:
        Prompt with minimal provider-specific guidance
    """
    if provider.lower() == "gemini":
        optimization = "\n\nFocus on precise timing and natural speech boundaries."
    elif provider.lower() == "openai":
        optimization = "\n\nOptimize for readability while maintaining accuracy."
    else:
        optimization = "\n\nEnsure consistent formatting and timing precision."

    return base_prompt + optimization


def validate_prompt_length(prompt: str, max_tokens: int = 4000) -> tuple[bool, str]:
    """Validate prompt length for token limits.

    Args:
        prompt: Prompt string to validate
        max_tokens: Maximum allowed tokens (rough estimate)

    Returns:
        Tuple of (is_valid, message)
    """
    # Rough estimation: 1 token ≈ 4 characters for English text
    estimated_tokens = len(prompt) // 4

    if estimated_tokens <= max_tokens:
        return True, f"Prompt length OK ({estimated_tokens} estimated tokens)"
    else:
        excess = estimated_tokens - max_tokens
        return (
            False,
            f"Prompt too long ({estimated_tokens} tokens, {excess} over limit)",
        )


def extract_context_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Extract relevant context from file or session metadata.

    Args:
        metadata: Metadata dictionary with potential context information

    Returns:
        Cleaned context dictionary for prompt building
    """
    context = {}

    # Extract common context fields
    context_fields = {
        "title": ["title", "name", "filename"],
        "topic": ["topic", "subject", "description"],
        "domain": ["domain", "category", "type"],
        "speakers": ["speakers", "participants", "attendees"],
        "technical_terms": ["technical_terms", "keywords", "terminology"],
    }

    for context_key, possible_keys in context_fields.items():
        for key in possible_keys:
            if key in metadata and metadata[key]:
                context[context_key] = metadata[key]
                break

    return context
