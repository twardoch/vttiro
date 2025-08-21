#!/usr/bin/env python3
# this_file: src/vttiro/models/__init__.py
"""AI model implementations and wrappers."""

# Engine and model enums
from .base import (
    TranscriptionEngine,
    GeminiModel,
    AssemblyAIModel,
    DeepgramModel,
    OpenAIModel,
    ModelCapability,
    get_default_model,
    get_available_models,
    validate_engine_model_combination,
    get_model_enum_class,
    get_model_capabilities,
    get_models_by_capability,
    estimate_transcription_cost
)

# Import will be available once models are implemented
try:
    from .gemini import GeminiTranscriber
except ImportError:
    GeminiTranscriber = None

try:
    from .assemblyai import AssemblyAITranscriber  
except ImportError:
    AssemblyAITranscriber = None

try:
    from .deepgram import DeepgramTranscriber
except ImportError:
    DeepgramTranscriber = None

try:
    from .openai import OpenAITranscriber
except ImportError:
    OpenAITranscriber = None

__all__ = [
    # Enums and utilities
    "TranscriptionEngine",
    "GeminiModel", 
    "AssemblyAIModel",
    "DeepgramModel",
    "OpenAIModel",
    "ModelCapability",
    "get_default_model",
    "get_available_models", 
    "validate_engine_model_combination",
    "get_model_enum_class",
    "get_model_capabilities",
    "get_models_by_capability",
    "estimate_transcription_cost",
    # Transcriber classes
    "GeminiTranscriber",
    "AssemblyAITranscriber", 
    "DeepgramTranscriber",
    "OpenAITranscriber"
]