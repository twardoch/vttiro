# this_file: src/vttiro/providers/deepgram/transcriber.py
"""Deepgram Nova-3 transcription provider.

This module implements the Deepgram transcription provider using the new
VTTiro 2.0 architecture. Provides fast, accurate transcription with
Nova-3 model, speaker diarization, and advanced features.

Used by:
- Core orchestration for Deepgram-based transcription
- Provider selection logic
- Testing infrastructure for Deepgram functionality
"""

import asyncio
import os
import time
from pathlib import Path
from typing import Any

from loguru import logger

from vttiro.core.errors import APIError, AuthenticationError, ProcessingError, handle_provider_exception
from vttiro.core.types import TranscriptionResult, TranscriptSegment
from vttiro.utils.api_keys import get_api_key_with_fallbacks
from vttiro.utils.prompt import build_webvtt_prompt, optimize_prompt_for_provider
from vttiro.utils.timestamp import distribute_words_over_duration

# Removed complex type validation
from vttiro.providers.base import TranscriberABC

# Optional dependency handling
try:
    from deepgram import DeepgramClient, PrerecordedOptions

    DEEPGRAM_AVAILABLE = True
except ImportError:
    DeepgramClient = None
    PrerecordedOptions = None
    DEEPGRAM_AVAILABLE = False
