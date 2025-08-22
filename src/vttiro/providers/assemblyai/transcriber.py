# this_file: src/vttiro/providers/assemblyai/transcriber.py
"""AssemblyAI Universal-2 transcription provider.

This module implements the AssemblyAI transcription provider using the new
VTTiro 2.0 architecture. Provides high-accuracy transcription with
Universal-2 model, speaker diarization, and advanced features.

Used by:
- Core orchestration for AssemblyAI-based transcription
- Provider selection logic
- Testing infrastructure for AssemblyAI functionality
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
    import assemblyai as aai

    ASSEMBLYAI_AVAILABLE = True
except ImportError:
    aai = None
    ASSEMBLYAI_AVAILABLE = False
