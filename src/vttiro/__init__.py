# this_file: src/vttiro/__init__.py
"""
VTTiro 2.0 - Advanced video transcription with AI models.

Modern, refactored transcription package with clean provider abstractions,
optimized processing, and comprehensive format support.
"""

__version__ = "2.0.0-dev"

# Core exports for backwards compatibility and public API
from .core.types import TranscriptionResult, TranscriptSegment
from .core.transcriber import Transcriber

__all__ = [
    "TranscriptionResult", 
    "TranscriptSegment", 
    "Transcriber",
    "__version__"
]