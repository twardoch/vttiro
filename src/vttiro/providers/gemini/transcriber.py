# this_file: src/vttiro/providers/gemini/transcriber.py
"""Google Gemini 2.0 Flash transcription provider.

This module implements the Gemini transcription provider using the new
VTTiro 2.0 architecture. Provides context-aware transcription with
WebVTT timing, speaker diarization, and safety filter handling.

Used by:
- Core orchestration for Gemini-based transcription
- Provider selection logic
- Testing infrastructure for Gemini functionality
"""

import asyncio
import os
import time
from pathlib import Path
from typing import Any

from loguru import logger

from vttiro.core.errors import APIError, AuthenticationError, ContentFilterError, ProcessingError, create_debug_context, handle_provider_exception
from vttiro.core.types import TranscriptionResult, TranscriptSegment
from vttiro.utils.api_keys import get_api_key_with_fallbacks
from vttiro.utils.logging import log_performance, log_provider_debug
from vttiro.utils.prompt import build_webvtt_prompt, optimize_prompt_for_provider
from vttiro.utils.timestamp import distribute_words_over_duration, parse_webvtt_timestamp_line

# Removed complex type validation - using simple validation in base class
from vttiro.providers.base import TranscriberABC

# Optional dependency handling
try:
    import google.generativeai as genai
    from google.generativeai.types import HarmBlockThreshold, HarmCategory

    GEMINI_AVAILABLE = True
except ImportError:
    genai = None
    HarmCategory = None
    HarmBlockThreshold = None
    GEMINI_AVAILABLE = False
