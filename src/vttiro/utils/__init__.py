# this_file: src/vttiro/utils/__init__.py
"""Essential utility functions for VTTiro transcription.

This module provides core utilities:
- Timestamp utilities for WebVTT processing
- Prompt utilities for AI providers
- Basic input validation
"""

from vttiro.utils.api_keys import get_all_available_api_keys, get_api_key_with_fallbacks
from vttiro.utils.input_validation import validate_file_path
from vttiro.utils.logging import get_debug_context, log_milestone, log_performance, log_provider_debug, log_system_info, log_timing, setup_logging
from vttiro.utils.prompt import build_plain_text_prompt, build_webvtt_prompt, extract_context_from_metadata, optimize_prompt_for_provider, validate_prompt_length
from vttiro.utils.timestamp import distribute_words_over_duration, format_timestamp, parse_timestamp, parse_webvtt_timestamp_line

__all__ = [
    # Timestamp utilities
    "parse_timestamp",
    "format_timestamp",
    "parse_webvtt_timestamp_line",
    "distribute_words_over_duration",
    # Prompt utilities
    "build_webvtt_prompt",
    "build_plain_text_prompt",
    "optimize_prompt_for_provider",
    "validate_prompt_length",
    "extract_context_from_metadata",
    # Validation utilities
    "validate_file_path",
    # API key utilities
    "get_api_key_with_fallbacks",
    "get_all_available_api_keys",
    # Logging utilities
    "setup_logging",
    "log_milestone",
    "log_performance",
    "log_provider_debug",
    "log_timing",
    "get_debug_context",
    "log_system_info",
]
