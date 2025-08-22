# this_file: src/vttiro/providers/__init__.py
"""Transcription provider implementations."""

from typing import Optional, Type, TYPE_CHECKING

from .base import TranscriberABC

if TYPE_CHECKING:
    from .gemini.transcriber import GeminiTranscriber
    from .openai.transcriber import OpenAITranscriber
    from .assemblyai.transcriber import AssemblyAITranscriber
    from .deepgram.transcriber import DeepgramTranscriber

# Optional provider imports (fail gracefully if dependencies missing)
GeminiTranscriber: Optional[Type[TranscriberABC]] = None
GEMINI_AVAILABLE = False

try:
    from .gemini.transcriber import GeminiTranscriber as _GeminiTranscriber
    GeminiTranscriber = _GeminiTranscriber
    GEMINI_AVAILABLE = True
except ImportError:
    pass

OpenAITranscriber: Optional[Type[TranscriberABC]] = None
OPENAI_AVAILABLE = False

try:
    from .openai.transcriber import OpenAITranscriber as _OpenAITranscriber
    OpenAITranscriber = _OpenAITranscriber
    OPENAI_AVAILABLE = True
except ImportError:
    pass

AssemblyAITranscriber: Optional[Type[TranscriberABC]] = None
ASSEMBLYAI_AVAILABLE = False

try:
    from .assemblyai.transcriber import AssemblyAITranscriber as _AssemblyAITranscriber
    AssemblyAITranscriber = _AssemblyAITranscriber
    ASSEMBLYAI_AVAILABLE = True
except ImportError:
    pass

DeepgramTranscriber: Optional[Type[TranscriberABC]] = None
DEEPGRAM_AVAILABLE = False

try:
    from .deepgram.transcriber import DeepgramTranscriber as _DeepgramTranscriber
    DeepgramTranscriber = _DeepgramTranscriber
    DEEPGRAM_AVAILABLE = True
except ImportError:
    pass

__all__ = ["TranscriberABC"]

# Add available providers to exports
if GEMINI_AVAILABLE:
    __all__.append("GeminiTranscriber")
if OPENAI_AVAILABLE:
    __all__.append("OpenAITranscriber")
if ASSEMBLYAI_AVAILABLE:
    __all__.append("AssemblyAITranscriber")
if DEEPGRAM_AVAILABLE:
    __all__.append("DeepgramTranscriber")