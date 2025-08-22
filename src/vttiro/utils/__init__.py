# this_file: src/vttiro/utils/__init__.py
"""Essential utility functions for VTTiro transcription.

This module provides core utilities:
- Timestamp utilities for WebVTT processing
- Prompt utilities for AI providers  
- Basic input validation
"""

from .timestamp import (
    parse_timestamp, 
    format_timestamp, 
    parse_webvtt_timestamp_line,
    distribute_words_over_duration
)
from .prompt import (
    build_webvtt_prompt,
    build_plain_text_prompt,
    optimize_prompt_for_provider,
    validate_prompt_length,
    extract_context_from_metadata
)
from .input_validation import InputValidator

__all__ = [
    # Timestamp utilities
    'parse_timestamp',
    'format_timestamp', 
    'parse_webvtt_timestamp_line',
    'distribute_words_over_duration',
    # Prompt utilities
    'build_webvtt_prompt',
    'build_plain_text_prompt',
    'optimize_prompt_for_provider',
    'validate_prompt_length',
    'extract_context_from_metadata',
    # Validation utilities
    'InputValidator',
]