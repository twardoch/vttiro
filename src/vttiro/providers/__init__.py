# this_file: src/vttiro/providers/__init__.py
"""Transcription provider implementations."""

from typing import TYPE_CHECKING, Optional, Type

from vttiro.providers.base import TranscriberABC

if TYPE_CHECKING:
    from vttiro.providers.assemblyai.transcriber import AssemblyAITranscriber
    from vttiro.providers.deepgram.transcriber import DeepgramTranscriber
    from vttiro.providers.gemini.transcriber import GeminiTranscriber
    from vttiro.providers.openai.transcriber import OpenAITranscriber

# Optional provider imports (fail gracefully if dependencies missing)
GeminiTranscriber: type[TranscriberABC] | None = None
GEMINI_AVAILABLE = False

try:
    from vttiro.providers.gemini.transcriber import GeminiTranscriber as _GeminiTranscriber

    GeminiTranscriber = _GeminiTranscriber
    GEMINI_AVAILABLE = True
except ImportError:
    pass

OpenAITranscriber: type[TranscriberABC] | None = None
OPENAI_AVAILABLE = False

try:
    from vttiro.providers.openai.transcriber import OpenAITranscriber as _OpenAITranscriber

    OpenAITranscriber = _OpenAITranscriber
    OPENAI_AVAILABLE = True
except ImportError:
    pass

AssemblyAITranscriber: type[TranscriberABC] | None = None
ASSEMBLYAI_AVAILABLE = False

try:
    from vttiro.providers.assemblyai.transcriber import AssemblyAITranscriber as _AssemblyAITranscriber

    AssemblyAITranscriber = _AssemblyAITranscriber
    ASSEMBLYAI_AVAILABLE = True
except ImportError:
    pass

DeepgramTranscriber: type[TranscriberABC] | None = None
DEEPGRAM_AVAILABLE = False

try:
    from vttiro.providers.deepgram.transcriber import DeepgramTranscriber as _DeepgramTranscriber

    DeepgramTranscriber = _DeepgramTranscriber
    DEEPGRAM_AVAILABLE = True
except ImportError:
    pass

__all__ = [
    "ASSEMBLYAI_AVAILABLE",
    "DEEPGRAM_AVAILABLE",
    "GEMINI_AVAILABLE",
    "OPENAI_AVAILABLE",
    "AssemblyAITranscriber",
    "DeepgramTranscriber",
    "GeminiTranscriber",
    "OpenAITranscriber",
    "TranscriberABC",
]
