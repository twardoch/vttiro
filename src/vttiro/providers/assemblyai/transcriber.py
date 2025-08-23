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

from vttiro.core.errors import APIError

# Optional dependency handling
try:
    import assemblyai as aai

    ASSEMBLYAI_AVAILABLE = True
except ImportError:
    aai = None
    ASSEMBLYAI_AVAILABLE = False
