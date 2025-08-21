#!/usr/bin/env python3
# this_file: src/vttiro/core/prompts.py
"""WebVTT Prompt Generation for vttiro.

This module provides comprehensive prompt generation capabilities for requesting
WebVTT format directly from AI engines, replacing the previous plain text approach
that required artificial timestamp estimation.
"""

from typing import Optional, Dict, Any, List
from enum import Enum

try:
    from loguru import logger
except ImportError:
    import logging as logger


class PromptTemplate(Enum):
    """Predefined prompt templates for different use cases."""

    BASIC_WEBVTT = "basic_webvtt"
    SPEAKER_DIARIZATION = "speaker_diarization"
    EMOTION_DETECTION = "emotion_detection"
    TECHNICAL_CONTENT = "technical_content"
    MULTILINGUAL = "multilingual"


class WebVTTPromptGenerator:
    """Advanced WebVTT prompt generator for AI transcription engines.

    Generates context-aware prompts that request WebVTT format directly from
    AI engines, including proper examples, speaker diarization instructions,
    and format compliance requirements.
    """

    def __init__(
        self,
        include_examples: bool = True,
        include_diarization: bool = True,
        include_emotions: bool = True,
        template: PromptTemplate = PromptTemplate.BASIC_WEBVTT,
    ):
        """Initialize the WebVTT prompt generator.

        Args:
            include_examples: Include properly formatted WebVTT examples
            include_diarization: Include speaker diarization instructions
            include_emotions: Include emotion detection instructions
            template: Base template to use for prompt generation
        """
        self.include_examples = include_examples
        self.include_diarization = include_diarization
        self.include_emotions = include_emotions
        self.template = template

        logger.debug(
            f"WebVTTPromptGenerator initialized with template: {template.value}"
        )

    def generate_webvtt_prompt(
        self,
        language: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        custom_instructions: Optional[str] = None,
    ) -> str:
        """Generate comprehensive WebVTT format prompt for AI engines.

        Args:
            language: Target language (e.g., 'en', 'es', 'fr')
            context: Video/audio context for improved accuracy
            custom_instructions: Additional custom instructions to append

        Returns:
            Complete WebVTT generation prompt string
        """
        logger.debug(f"Generating WebVTT prompt for language: {language}")

        # Start with base prompt based on template
        prompt_parts = [self._get_base_prompt()]

        # Add WebVTT format requirements
        prompt_parts.append(self._get_format_requirements())

        # Add language specification if provided
        if language and language != "auto":
            prompt_parts.append(self._get_language_instructions(language))

        # Add context-aware instructions
        if context:
            prompt_parts.append(self._get_context_instructions(context))

        # Add examples if requested
        if self.include_examples:
            prompt_parts.append(self._get_webvtt_examples())

        # Add speaker diarization instructions
        if self.include_diarization:
            prompt_parts.append(self._get_diarization_instructions())

        # Add emotion detection instructions
        if self.include_emotions:
            prompt_parts.append(self._get_emotion_instructions())

        # Add custom instructions if provided
        if custom_instructions:
            prompt_parts.append(f"\nAdditional Instructions:\n{custom_instructions}")

        # Add final formatting reminder
        prompt_parts.append(self._get_final_instructions())

        return "\n\n".join(prompt_parts)

    def _get_base_prompt(self) -> str:
        """Get base prompt based on selected template."""
        prompt_intro = """
You are a transcription expert. You are given an audio file and you are to transcribe it into a WebVTT subtitle file with precise timestamps. You must understand the domain of the audio, and you must appropriately and contextually recognize proper names and professional terminology. 

Whenever personal names, brand names and other proper names are mentioned, you must transcribe them so accordingly to the standards of the domain, and you must transcribe them consistently throughout the entire transcript.

Use Unicode characters for non-English names and words. Especially, use Latin-extended Unicode characters for names like Lech Wałęsa, even if the speakers says them in an americanized fashion. 

Do not transcribe involuntary utterances and repetitions like uhm, err, ahhh. Perform a minimal cleanup so that the text is written in a clean, orthographically correct manner with proper punctuation. Use professional typographic quote marks and punctuation appropriate for the language. 

Whenever you split a longer utterance into cues, ensure that the split is logical and semantic. If you split an utterance mid-sentence, don’t start the 2nd cue with a capitalized word (unless it’s always capitalized). 

If in doubt, use guidelines of the video streaming and broadcasting profession. 

Now:

"""

        base_prompts = {
            PromptTemplate.BASIC_WEBVTT: f"""{prompt_intro}
Please transcribe this audio and provide the output as a properly formatted WebVTT subtitle file.

CRITICAL: Your response must be valid WebVTT format starting with "WEBVTT" and including precise timestamps.""",
            PromptTemplate.SPEAKER_DIARIZATION: f"""{prompt_intro}
Please transcribe this audio with speaker identification and provide the output as a properly formatted WebVTT subtitle file with speaker tags, e.g., `<v Speaker1>`, `<v Speaker2>`.

CRITICAL: Your response must be valid WebVTT format starting with "WEBVTT" and including precise timestamps and speaker identification.""",
            PromptTemplate.EMOTION_DETECTION: f"""{prompt_intro}
Please transcribe this audio with emotion detection and provide the output as a properly formatted WebVTT subtitle file with emotional context, with non-verbal indicators prepending the spoken lines, `[laughs]`, `[sighs]`, `[applause]`, `[music]`).

CRITICAL: Your response must be valid WebVTT format starting with "WEBVTT" and including precise timestamps and emotional indicators.""",
            PromptTemplate.TECHNICAL_CONTENT: f"""{prompt_intro}
Please transcribe this technical audio content with careful attention to specialized terminology and provide the output as a properly formatted WebVTT subtitle file.

CRITICAL: Your response must be valid WebVTT format starting with "WEBVTT" and including precise timestamps.""",
            PromptTemplate.MULTILINGUAL: f"""{prompt_intro}
Please transcribe this multilingual audio content and provide the output as a properly formatted WebVTT subtitle file with language identification where applicable.

CRITICAL: Your response must be valid WebVTT format starting with "WEBVTT" and including precise timestamps.""",
        }

        return base_prompts.get(
            self.template, base_prompts[PromptTemplate.BASIC_WEBVTT]
        )

    def _get_format_requirements(self) -> str:
        """Get WebVTT format requirements."""
        return """
WebVTT Format Requirements:
1. Start with "WEBVTT" as the first line
2. Use precise timestamps in HH:MM:SS.mmm format (e.g., 00:00:01.500)
3. Format: start_time --> end_time (with space-arrow-space)
4. Ensure end_time is ALWAYS greater than start_time
5. Keep timestamps sequential (later cues must have later timestamps)
6. Maximum 7-second duration per cue for readability
7. Split long sentences into multiple cues with appropriate timing
8. Include proper line breaks for readability (max 50 characters per line)
9. Include non-verbal indicators for emotion (e.g., [laughs], [sighs], [applause], [music])
10. Include speaker diarization (e.g., <v Speaker1>, <v Speaker2>)
"""

    def _get_language_instructions(self, language: str) -> str:
        """Get language-specific instructions."""
        language_names = {
            "en": "English",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "ru": "Russian",
            "ja": "Japanese",
            "ko": "Korean",
            "zh": "Chinese",
            "ar": "Arabic",
            "hi": "Hindi",
            "bn": "Bengali",
            "te": "Telugu",
            "mr": "Marathi",
            "ta": "Tamil",
            "ur": "Urdu",
            "gu": "Gujarati",
            "kn": "Kannada",
            "ml": "Malayalam",
            "pa": "Punjabi",
            "ne": "Nepali",
            "si": "Sinhala",
            "my": "Myanmar",
            "th": "Thai",
            "vi": "Vietnamese",
            "id": "Indonesian",
            "ms": "Malay",
            "tl": "Filipino",
            "sw": "Swahili",
        }

        lang_name = language_names.get(language, language.upper())

        return f"""
Language Instructions:
- The audio is primarily in {lang_name}
- Add "Language: {language}" header after "WEBVTT"
- Pay special attention to {lang_name}-specific pronunciation and terminology
- Maintain proper capitalization and punctuation for {lang_name}"""

    def _get_context_instructions(self, context: Dict[str, Any]) -> str:
        """Generate context-aware instructions from video/audio metadata."""
        instructions = ["Context-Aware Processing:"]

        # Video metadata context
        if context.get("video_title"):
            instructions.append(f"- Video title: '{context['video_title']}'")
            instructions.append("- Use title context for proper noun recognition")

        if context.get("video_description"):
            desc = context["video_description"][:200]  # Limit description length
            instructions.append(f"- Video description: '{desc}...'")
            instructions.append("- Use description for terminology and topic context")

        if context.get("video_uploader"):
            instructions.append(f"- Content creator: {context['video_uploader']}")

        # Segment context for multi-part processing
        if context.get("segment_index") is not None:
            total_segments = context.get("total_segments", "N/A")
            instructions.append(
                f"- This is segment {context['segment_index'] + 1} of {total_segments}"
            )

        if context.get("start_time") is not None:
            start_min = int(context["start_time"] // 60)
            start_sec = int(context["start_time"] % 60)
            instructions.append(f"- Segment starts at {start_min}:{start_sec:02d}")
            instructions.append(
                "- Adjust WebVTT timestamps relative to this start time"
            )

        # Content type hints
        if context.get("content_type"):
            content_type = context["content_type"]
            instructions.append(f"- Content type: {content_type}")
            if content_type in ["podcast", "interview"]:
                instructions.append(
                    "- Expect conversational speech with multiple speakers"
                )
            elif content_type in ["lecture", "presentation"]:
                instructions.append("- Expect formal speech with technical terminology")
            elif content_type in ["news", "documentary"]:
                instructions.append(
                    "- Expect clear narration with proper nouns and facts"
                )

        # Quality hints
        instructions.extend(
            [
                "- Pay special attention to proper nouns, brand names, and technical terms",
                "- Use context clues for accurate transcription of names and specialized vocabulary",
                "- Ensure WebVTT timestamps reflect natural speech boundaries",
            ]
        )

        return "\n".join(instructions)

    def _get_webvtt_examples(self) -> str:
        """Get properly formatted WebVTT examples."""
        return """WebVTT Format Examples:

Example 1 - Basic Format:
WEBVTT
Language: en

00:00:00.000 --> 00:00:03.240
Welcome to this presentation on
modern web development.

00:00:03.240 --> 00:00:06.180
Today we'll explore the latest
techniques and best practices.

Example 2 - Speaker Identification:
WEBVTT
Language: en

00:00:00.000 --> 00:00:02.500
<v Host>Welcome everyone to our podcast.

00:00:02.500 --> 00:00:05.200
<v Host>Today we have a special guest
joining us.

00:00:05.200 --> 00:00:07.800
<v Guest>Thanks for having me! 
I'm excited to be here.

Example 3 - Emotional Context:
WEBVTT
Language: en

00:00:00.000 --> 00:00:02.000
<v Speaker>[enthusiastic] This is amazing!

00:00:02.000 --> 00:00:04.500
I can't believe we actually
made it work.

00:00:04.500 --> 00:00:06.000
[laughs] It took us months!"""

    def _get_diarization_instructions(self) -> str:
        """Get speaker diarization instructions."""
        return """
Speaker Diarization Instructions:
- Use <v SpeakerName> tags for speaker identification
- Assign consistent speaker names throughout (Speaker1, Speaker2, or actual names if known)
- Use generic names like "Host", "Guest", "Interviewer" if roles are clear
- Maintain speaker consistency across all WebVTT cues
- If multiple speakers in one cue, separate with line breaks:
  <v Speaker1>First speaker's text
  <v Speaker2>Second speaker's text"""

    def _get_emotion_instructions(self) -> str:
        """Get emotion detection instructions."""
        return """
Emotion Detection Instructions:
- Include emotional context in square brackets: [happy], [sad], [excited], [confused]
- Add non-verbal sounds: [laughs], [sighs], [applause], [music]
- Include relevant background sounds: [phone ringing], [door closing], [typing]
- Place emotional indicators at the beginning of cues when possible
- Use sparingly - only when emotion significantly affects meaning or tone
- Common emotions: [cheerful], [serious], [concerned], [surprised], [frustrated]"""

    def _get_final_instructions(self) -> str:
        """Get final formatting instructions."""
        return """
FINAL REQUIREMENTS:
- Your entire response must be valid WebVTT format
- Start immediately with "WEBVTT" - no explanation or introduction
- Ensure all timestamps are sequential and valid (end > start)
- Keep cue durations reasonable (1.5 to 7 seconds typically, but never shorter than the actual spoken audio that the cue represents)
- Use proper line wrapping for readability
- Include empty lines between cues for proper WebVTT structure
- Double-check timestamp format: HH:MM:SS.mmm --> HH:MM:SS.mmm

Remember: The AI engine's response will be used directly as a WebVTT file, so format precision is critical."""

    def get_webvtt_example(
        self, include_speakers: bool = True, include_emotions: bool = True
    ) -> str:
        """Get a standalone WebVTT example for reference.

        Args:
            include_speakers: Include speaker diarization examples
            include_emotions: Include emotional indicators

        Returns:
            Properly formatted WebVTT example
        """
        if include_speakers and include_emotions:
            return """WEBVTT
Language: auto

00:00:00.000 --> 00:00:03.240
<v Host>Welcome to this presentation on
font design and typography.

00:00:03.240 --> 00:00:06.180
<v Host>[enthusiastic] I'm excited to share
my workflow with you today.

00:00:06.180 --> 00:00:08.920
<v Guest>Thanks for having me!
Let's dive right in.

00:00:08.920 --> 00:00:11.500
<v Guest>Typography is the art and technique
of arranging type.

00:00:11.500 --> 00:00:14.200
[applause] And it's something I'm
truly passionate about."""

        elif include_speakers:
            return """WEBVTT
Language: auto

00:00:00.000 --> 00:00:03.240
<v Speaker1>Welcome to this presentation on
modern web development.

00:00:03.240 --> 00:00:06.180
<v Speaker1>Today we'll explore the latest
techniques and best practices.

00:00:06.180 --> 00:00:08.920
<v Speaker2>That sounds great!
I'm looking forward to learning."""

        else:
            return """WEBVTT
Language: auto

00:00:00.000 --> 00:00:03.240
Welcome to this presentation on
modern web development.

00:00:03.240 --> 00:00:06.180
Today we'll explore the latest
techniques and best practices.

00:00:06.180 --> 00:00:08.920
Let's start with the fundamentals
of responsive design."""

    def create_custom_prompt(
        self,
        base_instructions: str,
        format_requirements: Optional[str] = None,
        examples: Optional[str] = None,
        additional_context: Optional[str] = None,
    ) -> str:
        """Create a custom WebVTT prompt from provided components.

        Args:
            base_instructions: Core transcription instructions
            format_requirements: Custom format requirements (optional)
            examples: Custom WebVTT examples (optional)
            additional_context: Additional context or instructions

        Returns:
            Complete custom WebVTT prompt
        """
        prompt_parts = [base_instructions]

        if format_requirements:
            prompt_parts.append(format_requirements)
        else:
            prompt_parts.append(self._get_format_requirements())

        if examples:
            prompt_parts.append(f"Examples:\n{examples}")
        elif self.include_examples:
            prompt_parts.append(self._get_webvtt_examples())

        if additional_context:
            prompt_parts.append(additional_context)

        prompt_parts.append(self._get_final_instructions())

        return "\n\n".join(prompt_parts)

    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes for multilingual prompts."""
        return [
            "en",
            "es",
            "fr",
            "de",
            "it",
            "pt",
            "ru",
            "ja",
            "ko",
            "zh",
            "ar",
            "hi",
            "bn",
            "te",
            "mr",
            "ta",
            "ur",
            "gu",
            "kn",
            "ml",
            "pa",
            "ne",
            "si",
            "my",
            "th",
            "vi",
            "id",
            "ms",
            "tl",
            "sw",
        ]

    def validate_prompt(self, prompt: str) -> bool:
        """Validate that a prompt contains necessary WebVTT instructions.

        Args:
            prompt: Prompt string to validate

        Returns:
            True if prompt contains essential WebVTT requirements
        """
        required_elements = ["WEBVTT", "timestamps", "HH:MM:SS", "format"]

        prompt_lower = prompt.lower()
        missing_elements = [
            elem for elem in required_elements if elem.lower() not in prompt_lower
        ]

        if missing_elements:
            logger.warning(f"Prompt missing required elements: {missing_elements}")
            return False

        return True

    def __repr__(self) -> str:
        """String representation of the prompt generator."""
        return (
            f"WebVTTPromptGenerator(template={self.template.value}, "
            f"examples={self.include_examples}, "
            f"diarization={self.include_diarization}, "
            f"emotions={self.include_emotions})"
        )
