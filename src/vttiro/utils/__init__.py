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
from .api_keys import get_api_key_with_fallbacks, get_all_available_api_keys

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
    # API key utilities
    'get_api_key_with_fallbacks',
    'get_all_available_api_keys',
]