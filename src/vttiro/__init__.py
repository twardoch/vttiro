#!/usr/bin/env python3
# this_file: src/vttiro/__init__.py
"""vttiro: Advanced video transcription to WebVTT with multi-model AI.

This package provides comprehensive video transcription capabilities that surpass 
OpenAI Whisper through multi-model AI integration, featuring precise timestamps,
speaker diarization, emotion detection, and seamless YouTube integration.

Example:
    Basic usage:
    >>> import vttiro
    >>> transcriber = vttiro.Transcriber()
    >>> result = await transcriber.transcribe("https://youtube.com/watch?v=example")
    
    CLI usage:
    $ vttiro transcribe "https://youtube.com/watch?v=example" --output subtitles.vtt
"""

from vttiro.__version__ import __version__

# Core imports
from vttiro.core.transcription import TranscriptionResult
from vttiro.core.config import VttiroConfig

# Main classes for public API
try:
    from vttiro.core.transcriber import Transcriber
except ImportError:
    # Handle missing optional dependencies gracefully
    pass

__all__ = [
    "__version__",
    "TranscriptionResult", 
    "VttiroConfig",
    "Transcriber",
]

# Package metadata
__author__ = "Adam Twardoch"
__email__ = "adam+github@twardoch.com"
__license__ = "MIT"
__description__ = "Advanced video transcription to WebVTT with multi-model AI, speaker diarization, and emotion detection"