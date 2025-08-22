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
    max_segment_duration: float = 30.0
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
        "TRANSCRIPTION TASK: Convert the provided audio to WebVTT subtitle format.",
        "",
        "STRICT OUTPUT FORMAT REQUIREMENTS:",
        "1. Start with 'WEBVTT' header",
        "2. Use precise timestamps in HH:MM:SS.mmm format",
        "3. Each subtitle segment should have:",
        "   - Timestamp line: start_time --> end_time",
        "   - Text content on following lines",
        "   - Empty line to separate segments",
        "",
        "TIMING REQUIREMENTS:",
        f"- Maximum segment duration: {max_segment_duration} seconds",
        "- Minimum segment duration: 1 second", 
        "- Align segments with natural speech boundaries",
        "- Use precise timing based on actual audio content",
        "- Do NOT estimate or guess timestamps",
        "",
        "TEXT QUALITY REQUIREMENTS:",
        "- Transcribe exactly what is spoken",
        "- Maintain proper punctuation and capitalization", 
        "- Fix obvious speech errors and filler words (um, uh, etc.)",
        "- Use proper names and technical terms when identifiable",
        ""
    ]
    
    # Add language-specific instructions
    if language:
        prompt_parts.extend([
            "LANGUAGE REQUIREMENTS:",
            f"- Target language: {language.upper()}",
            "- Maintain proper grammar and spelling for this language",
            "- Use appropriate character encoding",
            ""
        ])
    else:
        prompt_parts.extend([
            "LANGUAGE REQUIREMENTS:",
            "- Auto-detect the spoken language",
            "- Maintain proper grammar and spelling",
            "- Use appropriate character encoding for detected language",
            ""
        ])
    
    # Add speaker diarization if requested
    if include_speaker_diarization:
        prompt_parts.extend([
            "SPEAKER IDENTIFICATION:",
            "- Use <v Speaker> tags for speaker identification",
            "- Format: <v Speaker 1>Text content here",
            "- Maintain consistent speaker labels throughout",
            "- Only identify speakers when clearly distinguishable",
            ""
        ])
    
    # Add emotion detection if requested
    if include_emotions:
        prompt_parts.extend([
            "EMOTION DETECTION:",
            "- Include emotional context when clearly evident",
            "- Use subtle indicators rather than explicit emotion labels",
            "- Focus on tone and delivery style",
            ""
        ])
    
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
            prompt_parts.extend([
                "CONTEXT INFORMATION:",
                *context_instructions,
                ""
            ])
    
    # Add examples for clarity
    prompt_parts.extend([
        "EXAMPLE OUTPUT FORMAT:",
        "WEBVTT",
        "",
        "00:00:01.000 --> 00:00:05.000",
        "Hello and welcome to today's presentation.",
        "",
        "00:00:05.500 --> 00:00:10.250",
        "<v Speaker 2>Thank you for having me here.",
        "",
        "IMPORTANT NOTES:",
        "- Generate ONLY the WebVTT content, no additional text",
        "- Ensure all timestamps are based on actual audio timing",
        "- Maintain consistent formatting throughout",
        "- Use precise timing for optimal subtitle synchronization"
    ])
    
    return "\n".join(prompt_parts)


def build_plain_text_prompt(
    language: str | None = None,
    context: dict[str, Any] | None = None,
    include_speaker_labels: bool = True,
    clean_text: bool = True
) -> str:
    """Build a prompt for plain text transcription without timing.
    
    Args:
        language: Target language code or None for auto-detection
        context: Additional context information
        include_speaker_labels: Whether to include speaker identification
        clean_text: Whether to clean up filler words and speech errors
        
    Returns:
        Prompt string for plain text transcription
    """
    prompt_parts = [
        "TRANSCRIPTION TASK: Convert the provided audio to clean, readable text.",
        "",
        "OUTPUT REQUIREMENTS:",
        "- Transcribe exactly what is spoken",
        "- Use proper punctuation and paragraph breaks",
        "- Maintain proper capitalization",
    ]
    
    if clean_text:
        prompt_parts.extend([
            "- Remove filler words (um, uh, like, you know)",
            "- Fix obvious speech errors and false starts",
            "- Create readable, flowing text"
        ])
    else:
        prompt_parts.extend([
            "- Include all spoken words, including filler words",
            "- Preserve hesitations and speech patterns",
            "- Maintain verbatim accuracy"
        ])
    
    prompt_parts.append("")
    
    # Add language-specific instructions
    if language:
        prompt_parts.extend([
            f"LANGUAGE: Transcribe in {language.upper()}",
            ""
        ])
    
    # Add speaker identification if requested
    if include_speaker_labels:
        prompt_parts.extend([
            "SPEAKER IDENTIFICATION:",
            "- Clearly identify different speakers",
            "- Use format: Speaker 1: [text]",
            "- Maintain consistent speaker labels",
            ""
        ])
    
    # Add context if provided
    if context:
        if "domain" in context:
            prompt_parts.append(f"CONTEXT: This is {context['domain']} content.")
        if "topic" in context:
            prompt_parts.append(f"TOPIC: {context['topic']}")
        if context:
            prompt_parts.append("")
    
    prompt_parts.extend([
        "Generate only the transcribed text with no additional formatting or metadata."
    ])
    
    return "\n".join(prompt_parts)


def optimize_prompt_for_provider(base_prompt: str, provider: str) -> str:
    """Add provider-specific optimizations to a base prompt.
    
    Args:
        base_prompt: Base prompt string
        provider: Provider name (gemini, openai, assemblyai, deepgram)
        
    Returns:
        Optimized prompt with provider-specific instructions
    """
    if provider.lower() == "gemini":
        optimization = """

GEMINI-SPECIFIC OPTIMIZATIONS:
- Leverage your audio understanding capabilities for precise timing
- Use natural speech boundary detection for segment splitting
- Apply contextual knowledge for proper noun recognition
- Maintain consistent speaker identification throughout
- Ensure WebVTT timestamps reflect actual speech timing, not estimation
- Use your multimodal understanding for context-aware transcription"""
        
    elif provider.lower() == "openai":
        optimization = """

OPENAI-SPECIFIC OPTIMIZATIONS:
- Focus on natural language understanding and context
- Use your training on diverse speech patterns
- Apply robust error correction and normalization
- Maintain consistency in technical terminology
- Optimize for readability while preserving accuracy"""
        
    elif provider.lower() in ["assemblyai", "deepgram"]:
        optimization = """

API-SPECIFIC OPTIMIZATIONS:
- Ensure proper formatting for downstream processing
- Maintain consistent structure throughout response
- Focus on accuracy and timing precision
- Use clear speaker delineation when applicable"""
        
    else:
        # Generic optimization for unknown providers
        optimization = """

GENERAL OPTIMIZATIONS:
- Focus on accuracy and consistency
- Maintain proper formatting standards
- Use clear, unambiguous output structure
- Ensure timing precision where applicable"""
    
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
        return False, f"Prompt too long ({estimated_tokens} tokens, {excess} over limit)"


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
        "technical_terms": ["technical_terms", "keywords", "terminology"]
    }
    
    for context_key, possible_keys in context_fields.items():
        for key in possible_keys:
            if key in metadata and metadata[key]:
                context[context_key] = metadata[key]
                break
    
    return context