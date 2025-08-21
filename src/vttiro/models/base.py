#!/usr/bin/env python3
# this_file: src/vttiro/models/base.py
"""Base classes and enums for AI transcription engines and models."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set


@dataclass
class ModelCapability:
    """Represents the capabilities and limitations of a transcription model."""
    
    # Performance characteristics
    max_duration_seconds: Optional[int] = None  # Max audio length in seconds, None = unlimited
    cost_per_minute: Optional[float] = None     # USD cost per minute of audio
    speed_factor: float = 1.0                   # Relative speed compared to baseline (1.0 = normal)
    
    # Quality and accuracy
    accuracy_score: float = 0.8                 # Estimated accuracy score (0.0 - 1.0)
    language_support: Set[str] = None           # Supported language codes
    
    # Features
    supports_diarization: bool = False          # Speaker separation
    supports_timestamps: bool = True            # Word-level timestamps
    supports_confidence: bool = False           # Confidence scores
    supports_emotions: bool = False             # Emotion detection
    
    # Use case recommendations
    recommended_for: Set[str] = None            # Use cases: 'podcast', 'meeting', 'lecture', 'music', 'noisy'
    
    # Limitations and warnings
    limitations: List[str] = None               # Human-readable limitations
    warning_threshold_minutes: Optional[float] = None  # Warn user if file exceeds this duration
    
    def __post_init__(self):
        """Initialize default collections if None."""
        if self.language_support is None:
            self.language_support = {"en"}  # Default to English only
        if self.recommended_for is None:
            self.recommended_for = set()
        if self.limitations is None:
            self.limitations = []


class TranscriptionEngine(str, Enum):
    """AI transcription engines (providers)."""
    GEMINI = "gemini"
    ASSEMBLYAI = "assemblyai"
    DEEPGRAM = "deepgram"
    OPENAI = "openai"


class GeminiModel(str, Enum):
    """Google Gemini model variants."""
    GEMINI_2_0_FLASH = "gemini-2.0-flash"
    GEMINI_2_0_FLASH_EXP = "gemini-2.0-flash-exp"
    GEMINI_2_0_FLASH_LITE = "gemini-2.0-flash-lite"
    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    GEMINI_2_5_FLASH_LITE = "gemini-2.5-flash-lite"
    GEMINI_2_5_PRO = "gemini-2.5-pro"


class AssemblyAIModel(str, Enum):
    """AssemblyAI model variants."""
    UNIVERSAL_1 = "universal-1"
    UNIVERSAL_2 = "universal-2"
    NANO = "nano"
    BEST = "best"


class DeepgramModel(str, Enum):
    """Deepgram model variants."""
    NOVA_2 = "nova-2"
    NOVA_3 = "nova-3"
    ENHANCED = "enhanced"
    BASE = "base"
    WHISPER_CLOUD = "whisper-cloud"


class OpenAIModel(str, Enum):
    """OpenAI transcription model variants."""
    WHISPER_1 = "whisper-1"
    GPT_4O_TRANSCRIBE = "gpt-4o-transcribe"
    GPT_4O_MINI_TRANSCRIBE = "gpt-4o-mini-transcribe"


# Engine to model mapping for defaults and validation
ENGINE_DEFAULT_MODELS: Dict[TranscriptionEngine, str] = {
    TranscriptionEngine.GEMINI: GeminiModel.GEMINI_2_0_FLASH.value,
    TranscriptionEngine.ASSEMBLYAI: AssemblyAIModel.UNIVERSAL_2.value,
    TranscriptionEngine.DEEPGRAM: DeepgramModel.NOVA_3.value,
    TranscriptionEngine.OPENAI: OpenAIModel.GPT_4O_TRANSCRIBE.value,
}

ENGINE_AVAILABLE_MODELS: Dict[TranscriptionEngine, List[str]] = {
    TranscriptionEngine.GEMINI: [model.value for model in GeminiModel],
    TranscriptionEngine.ASSEMBLYAI: [model.value for model in AssemblyAIModel],
    TranscriptionEngine.DEEPGRAM: [model.value for model in DeepgramModel],
    TranscriptionEngine.OPENAI: [model.value for model in OpenAIModel],
}


def get_default_model(engine: TranscriptionEngine) -> str:
    """Get the default model for an engine."""
    return ENGINE_DEFAULT_MODELS[engine]


def get_available_models(engine: TranscriptionEngine) -> List[str]:
    """Get available models for an engine."""
    return ENGINE_AVAILABLE_MODELS[engine].copy()


def validate_engine_model_combination(engine: str, model: str) -> bool:
    """Validate that an engine and model combination is valid."""
    try:
        engine_enum = TranscriptionEngine(engine)
        available_models = get_available_models(engine_enum)
        return model in available_models
    except (ValueError, KeyError):
        return False


def get_model_enum_class(engine: TranscriptionEngine):
    """Get the model enum class for an engine."""
    if engine == TranscriptionEngine.GEMINI:
        return GeminiModel
    elif engine == TranscriptionEngine.ASSEMBLYAI:
        return AssemblyAIModel
    elif engine == TranscriptionEngine.DEEPGRAM:
        return DeepgramModel
    elif engine == TranscriptionEngine.OPENAI:
        return OpenAIModel
    else:
        raise ValueError(f"Unknown engine: {engine}")


# Model capability definitions with real-world data
MODEL_CAPABILITIES: Dict[str, ModelCapability] = {
    # Gemini Models
    "gemini-2.0-flash": ModelCapability(
        max_duration_seconds=3600,  # 1 hour limit
        cost_per_minute=0.001,      # $0.001 per minute
        speed_factor=2.0,           # Very fast
        accuracy_score=0.92,
        language_support={"en", "es", "fr", "de", "it", "pt", "ja", "ko", "zh", "hi", "ar", "ru"},
        supports_diarization=True,
        supports_timestamps=True,
        supports_confidence=True,
        recommended_for={"meeting", "lecture", "podcast", "general"},
        limitations=["May struggle with very noisy audio", "Limited to 1 hour per request"],
        warning_threshold_minutes=55.0
    ),
    
    "gemini-2.0-flash-exp": ModelCapability(
        max_duration_seconds=3600,
        cost_per_minute=0.001,
        speed_factor=2.0,
        accuracy_score=0.94,        # Higher accuracy (experimental)
        language_support={"en", "es", "fr", "de", "it", "pt", "ja", "ko", "zh", "hi", "ar", "ru"},
        supports_diarization=True,
        supports_timestamps=True,
        supports_confidence=True,
        recommended_for={"meeting", "lecture", "podcast", "critical"},
        limitations=["Experimental model - may have unexpected behavior", "Limited to 1 hour per request"],
        warning_threshold_minutes=55.0
    ),
    
    "gemini-2.5-pro": ModelCapability(
        max_duration_seconds=7200,  # 2 hours
        cost_per_minute=0.015,      # More expensive but highest quality
        speed_factor=0.7,           # Slower but more accurate
        accuracy_score=0.96,
        language_support={"en", "es", "fr", "de", "it", "pt", "ja", "ko", "zh", "hi", "ar", "ru", "pl", "nl", "tr"},
        supports_diarization=True,
        supports_timestamps=True,
        supports_confidence=True,
        supports_emotions=True,
        recommended_for={"critical", "legal", "medical", "research"},
        limitations=["Higher cost", "Slower processing"],
        warning_threshold_minutes=10.0  # Warn earlier due to cost
    ),
    
    "gemini-2.5-flash": ModelCapability(
        max_duration_seconds=3600,
        cost_per_minute=0.003,
        speed_factor=1.5,
        accuracy_score=0.90,
        language_support={"en", "es", "fr", "de", "it", "pt", "ja", "ko", "zh"},
        supports_diarization=True,
        supports_timestamps=True,
        supports_confidence=True,
        recommended_for={"general", "podcast", "meeting"},
        limitations=["Limited to 1 hour per request"],
        warning_threshold_minutes=55.0
    ),
    
    # AssemblyAI Models
    "universal-2": ModelCapability(
        max_duration_seconds=None,  # No strict limit
        cost_per_minute=0.0065,
        speed_factor=1.0,
        accuracy_score=0.94,
        language_support={"en", "es", "fr", "de", "it", "pt", "pl", "nl", "tr", "ru", "ar", "zh", "ja", "ko", "hi"},
        supports_diarization=True,
        supports_timestamps=True,
        supports_confidence=True,
        recommended_for={"professional", "podcast", "meeting", "legal"},
        limitations=["Requires paid AssemblyAI account"],
        warning_threshold_minutes=120.0
    ),
    
    "universal-1": ModelCapability(
        max_duration_seconds=None,
        cost_per_minute=0.0065,
        speed_factor=1.1,
        accuracy_score=0.91,
        language_support={"en", "es", "fr", "de", "it", "pt"},
        supports_diarization=True,
        supports_timestamps=True,
        supports_confidence=True,
        recommended_for={"general", "legacy"},
        limitations=["Previous generation model", "Fewer language options"],
        warning_threshold_minutes=120.0
    ),
    
    "nano": ModelCapability(
        max_duration_seconds=None,
        cost_per_minute=0.0025,     # Cheaper
        speed_factor=3.0,           # Very fast
        accuracy_score=0.85,        # Lower accuracy
        language_support={"en"},
        supports_diarization=False,
        supports_timestamps=True,
        supports_confidence=True,
        recommended_for={"draft", "quick", "budget"},
        limitations=["English only", "Lower accuracy", "No speaker diarization"],
        warning_threshold_minutes=240.0
    ),
    
    # Deepgram Models
    "nova-3": ModelCapability(
        max_duration_seconds=None,
        cost_per_minute=0.0043,
        speed_factor=1.2,
        accuracy_score=0.93,
        language_support={"en", "es", "fr", "de", "it", "pt", "pl", "nl", "tr", "ru", "ar", "zh", "ja", "ko", "hi", "th", "vi"},
        supports_diarization=True,
        supports_timestamps=True,
        supports_confidence=True,
        recommended_for={"real-time", "streaming", "meeting"},
        limitations=["Requires paid Deepgram account"],
        warning_threshold_minutes=180.0
    ),
    
    "nova-2": ModelCapability(
        max_duration_seconds=None,
        cost_per_minute=0.0043,
        speed_factor=1.3,
        accuracy_score=0.90,
        language_support={"en", "es", "fr", "de", "it", "pt"},
        supports_diarization=True,
        supports_timestamps=True,
        supports_confidence=True,
        recommended_for={"general", "legacy"},
        limitations=["Previous generation model"],
        warning_threshold_minutes=180.0
    ),
    
    "whisper-cloud": ModelCapability(
        max_duration_seconds=None,
        cost_per_minute=0.0043,
        speed_factor=0.8,           # Slower but very accurate
        accuracy_score=0.95,
        language_support={"en", "es", "fr", "de", "it", "pt", "pl", "nl", "tr", "ru", "ar", "zh", "ja", "ko", "hi", "th", "vi", "uk", "cs"},
        supports_diarization=False, # Whisper doesn't do native diarization
        supports_timestamps=True,
        supports_confidence=True,
        recommended_for={"multilingual", "research", "accuracy"},
        limitations=["No speaker diarization", "Slower processing"],
        warning_threshold_minutes=180.0
    ),
    
    # OpenAI Models
    "whisper-1": ModelCapability(
        max_duration_seconds=1500,  # 25MB file limit roughly 25 minutes
        cost_per_minute=0.006,      # $0.006 per minute
        speed_factor=1.0,           # Standard processing speed
        accuracy_score=0.89,        # Good baseline accuracy
        language_support={"en", "es", "fr", "de", "it", "pt", "ja", "ko", "zh", "ar", "ru", "hi", "pl", "nl", "tr", "uk", "cs", "sv", "da", "no", "fi", "et", "lv", "lt", "sl", "sk", "bg", "ro", "hu", "hr", "sr", "bs", "mk", "sq", "mt", "eu", "ca", "gl", "cy", "ga", "is", "fo", "lb", "rm", "sc", "co", "ast", "oc", "br", "kw", "gv", "gd", "mi", "haw", "zu", "xh", "af", "sw", "am", "yo", "ig", "ha", "so", "mg", "sm", "to", "fj", "ty", "chm", "kv", "kk", "ky", "uz", "tg", "mn", "my", "lo", "km", "ne", "si", "bn", "ta", "te", "kn", "ml", "th", "vi", "id", "ms", "tl", "jv", "su", "ceb", "hil", "war", "bcl", "pag", "ilo", "pam", "bik", "mag", "mdh", "mrw", "tsg"},
        supports_diarization=False, # Whisper doesn't do native diarization
        supports_timestamps=True,
        supports_confidence=True,
        recommended_for={"general", "multilingual", "batch"},
        limitations=["25MB file limit", "No native speaker diarization", "No streaming"],
        warning_threshold_minutes=20.0
    ),
    
    "gpt-4o-transcribe": ModelCapability(
        max_duration_seconds=1500,  # Similar file limit
        cost_per_minute=0.012,      # Estimated higher cost
        speed_factor=1.5,           # Faster due to GPT-4o architecture
        accuracy_score=0.94,        # Higher accuracy expected
        language_support={"en", "es", "fr", "de", "it", "pt", "ja", "ko", "zh", "ar", "ru", "hi"},
        supports_diarization=True,  # GPT-4o can do contextual speaker identification
        supports_timestamps=True,
        supports_confidence=True,
        supports_emotions=True,     # GPT-4o can detect emotions
        recommended_for={"high-quality", "english", "streaming", "real-time"},
        limitations=["Limited parameter support", "Higher cost", "Fewer languages than Whisper"],
        warning_threshold_minutes=10.0  # Warn earlier due to cost
    ),
    
    "gpt-4o-mini-transcribe": ModelCapability(
        max_duration_seconds=1500,
        cost_per_minute=0.003,      # Lower cost option
        speed_factor=2.0,           # Faster processing
        accuracy_score=0.90,        # Good accuracy, slightly lower than full GPT-4o
        language_support={"en", "es", "fr", "de", "it", "pt", "ja", "ko", "zh"},
        supports_diarization=True,
        supports_timestamps=True,
        supports_confidence=True,
        recommended_for={"cost-effective", "fast", "general"},
        limitations=["Limited parameter support", "Fewer languages"],
        warning_threshold_minutes=15.0
    ),
}


def get_model_capabilities(model_id: str) -> ModelCapability:
    """Get capabilities for a specific model.
    
    Args:
        model_id: The model identifier (e.g., 'gemini-2.0-flash')
        
    Returns:
        ModelCapability object with model information
        
    Raises:
        ValueError: If model is not found
    """
    if model_id not in MODEL_CAPABILITIES:
        raise ValueError(f"No capabilities defined for model: {model_id}")
    return MODEL_CAPABILITIES[model_id]


def get_models_by_capability(requirement: str, engine: Optional[TranscriptionEngine] = None) -> List[str]:
    """Find models that match specific capability requirements.
    
    Args:
        requirement: Capability requirement ('fast', 'accurate', 'cheap', 'multilingual', 'diarization')
        engine: Optional engine filter
        
    Returns:
        List of model IDs that meet the requirement
    """
    matching_models = []
    models_to_check = MODEL_CAPABILITIES.keys()
    
    # Filter by engine if specified
    if engine:
        engine_models = get_available_models(engine)
        models_to_check = [m for m in models_to_check if m in engine_models]
    
    for model_id in models_to_check:
        capability = MODEL_CAPABILITIES[model_id]
        
        if requirement == "fast" and capability.speed_factor >= 1.5:
            matching_models.append(model_id)
        elif requirement == "accurate" and capability.accuracy_score >= 0.93:
            matching_models.append(model_id)
        elif requirement == "cheap" and capability.cost_per_minute and capability.cost_per_minute <= 0.005:
            matching_models.append(model_id)
        elif requirement == "multilingual" and len(capability.language_support) >= 10:
            matching_models.append(model_id)
        elif requirement == "diarization" and capability.supports_diarization:
            matching_models.append(model_id)
    
    return matching_models


def estimate_transcription_cost(model_id: str, duration_minutes: float) -> Optional[float]:
    """Estimate the cost of transcribing with a specific model.
    
    Args:
        model_id: The model identifier
        duration_minutes: Duration of audio in minutes
        
    Returns:
        Estimated cost in USD, or None if cost data unavailable
    """
    try:
        capability = get_model_capabilities(model_id)
        if capability.cost_per_minute:
            return capability.cost_per_minute * duration_minutes
    except ValueError:
        pass
    return None