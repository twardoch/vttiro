#!/usr/bin/env python3
# this_file: src/vttiro/core/config.py
"""Configuration management for vttiro using Pydantic."""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass

try:
    from pydantic import BaseModel, Field, field_validator
    import yaml
except ImportError:
    raise ImportError("Required dependencies missing. Install with: uv pip install --system vttiro")


class TranscriptionConfig(BaseModel):
    """Configuration for transcription engines."""
    
    default_model: str = Field(default="auto", description="Default transcription model")
    preferred_model: str = Field(default="auto", description="Preferred transcription model")
    ensemble_enabled: bool = Field(default=False, description="Enable multi-model ensemble")
    confidence_threshold: float = Field(default=0.8, description="Minimum confidence threshold")
    language: Optional[str] = Field(default=None, description="Default language code")
    
    # API configurations
    gemini_api_key: Optional[str] = Field(default=None, description="Google Gemini API key")
    assemblyai_api_key: Optional[str] = Field(default=None, description="AssemblyAI API key")
    deepgram_api_key: Optional[str] = Field(default=None, description="Deepgram API key")
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    
    @field_validator('confidence_threshold')
    def validate_confidence(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('confidence_threshold must be between 0.0 and 1.0')
        return v


class ProcessingConfig(BaseModel):
    """Configuration for audio/video processing."""
    
    chunk_duration: int = Field(default=600, description="Audio chunk duration in seconds")
    overlap_duration: int = Field(default=30, description="Chunk overlap duration in seconds")
    max_duration: int = Field(default=36000, description="Maximum video duration in seconds")
    sample_rate: int = Field(default=16000, description="Audio sample rate in Hz")
    prefer_integer_seconds: bool = Field(default=True, description="Prefer integer second boundaries")
    
    # Energy-based segmentation
    energy_threshold_percentile: int = Field(default=20, description="Energy threshold percentile")
    min_energy_window: float = Field(default=2.0, description="Minimum energy window in seconds")
    
    @field_validator('chunk_duration')
    def validate_chunk_duration(cls, v):
        if v <= 0:
            raise ValueError('chunk_duration must be positive')
        return v


class DiarizationConfig(BaseModel):
    """Configuration for speaker diarization."""
    
    enabled: bool = Field(default=True, description="Enable speaker diarization")
    min_speakers: Optional[int] = Field(default=None, description="Minimum number of speakers")
    max_speakers: Optional[int] = Field(default=10, description="Maximum number of speakers")
    threshold: float = Field(default=0.7, description="Diarization threshold")
    huggingface_token: Optional[str] = Field(default=None, description="HuggingFace token for pyannote")


class EmotionConfig(BaseModel):
    """Configuration for emotion detection."""
    
    enabled: bool = Field(default=False, description="Enable emotion detection")
    audio_weight: float = Field(default=0.6, description="Weight for audio-based emotion detection")
    text_weight: float = Field(default=0.4, description="Weight for text-based emotion detection")
    confidence_threshold: float = Field(default=0.5, description="Minimum emotion confidence")
    cultural_adaptation: bool = Field(default=True, description="Enable cultural adaptation")


class OutputConfig(BaseModel):
    """Configuration for subtitle output generation."""
    
    default_format: str = Field(default="webvtt", description="Default output format")
    max_chars_per_line: int = Field(default=42, description="Maximum characters per subtitle line")
    max_lines_per_cue: int = Field(default=2, description="Maximum lines per subtitle cue")
    max_cue_duration: float = Field(default=7.0, description="Maximum cue duration in seconds")
    reading_speed_wpm: int = Field(default=160, description="Reading speed in words per minute")
    
    # Accessibility
    wcag_compliance: str = Field(default="AA", description="WCAG compliance level")
    include_sound_descriptions: bool = Field(default=True, description="Include sound effect descriptions")


class YouTubeConfig(BaseModel):
    """Configuration for YouTube integration."""
    
    enabled: bool = Field(default=False, description="Enable YouTube integration")
    client_secrets_file: Optional[str] = Field(default=None, description="OAuth client secrets file")
    quota_limit: int = Field(default=10000, description="Daily quota limit")
    auto_upload: bool = Field(default=False, description="Automatically upload subtitles")


class VttiroConfig(BaseModel):
    """Main configuration class for vttiro."""
    
    transcription: TranscriptionConfig = Field(default_factory=TranscriptionConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)  
    diarization: DiarizationConfig = Field(default_factory=DiarizationConfig)
    emotion: EmotionConfig = Field(default_factory=EmotionConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    youtube: YouTubeConfig = Field(default_factory=YouTubeConfig)
    
    # Global settings
    verbose: bool = Field(default=False, description="Enable verbose logging")
    temp_dir: Optional[str] = Field(default=None, description="Temporary directory path")
    
    class Config:
        """Pydantic configuration."""
        env_prefix = "VTTIRO_"
        case_sensitive = False
        
    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> 'VttiroConfig':
        """Load configuration from YAML file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
            
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
            
        return cls(**data)
        
    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            yaml.dump(self.dict(), f, default_flow_style=False, indent=2)
            
    @classmethod
    def get_default_config_path(cls) -> Path:
        """Get the default configuration file path."""
        config_dir = Path.home() / ".vttiro"
        config_dir.mkdir(exist_ok=True)
        return config_dir / "config.yaml"
        
    @classmethod
    def load_default(cls) -> 'VttiroConfig':
        """Load configuration from default location or create new one."""
        default_path = cls.get_default_config_path()
        
        if default_path.exists():
            return cls.load_from_file(default_path)
        else:
            # Create default configuration
            config = cls()
            config.save_to_file(default_path)
            return config
            
    def update_from_env(self) -> None:
        """Update configuration from environment variables with VTTIRO_ prefix."""
        # Update API keys from environment variables
        if os.getenv("VTTIRO_GEMINI_API_KEY"):
            self.transcription.gemini_api_key = os.getenv("VTTIRO_GEMINI_API_KEY")
        if os.getenv("VTTIRO_ASSEMBLYAI_API_KEY"):
            self.transcription.assemblyai_api_key = os.getenv("VTTIRO_ASSEMBLYAI_API_KEY")
        if os.getenv("VTTIRO_DEEPGRAM_API_KEY"):
            self.transcription.deepgram_api_key = os.getenv("VTTIRO_DEEPGRAM_API_KEY")
        if os.getenv("VTTIRO_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"):
            # Support both VTTIRO_ prefix and standard OPENAI_API_KEY
            self.transcription.openai_api_key = os.getenv("VTTIRO_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        
        # Update model preference
        if os.getenv("VTTIRO_MODEL"):
            self.transcription.preferred_model = os.getenv("VTTIRO_MODEL")
        
        # Update processing settings
        if os.getenv("VTTIRO_CHUNK_DURATION"):
            try:
                self.processing.chunk_duration = int(os.getenv("VTTIRO_CHUNK_DURATION"))
            except ValueError:
                pass  # Keep default if invalid
        
        # Legacy support for non-prefixed variables (for backwards compatibility)
        if os.getenv("GEMINI_API_KEY") and not self.transcription.gemini_api_key:
            self.transcription.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if os.getenv("ASSEMBLYAI_API_KEY") and not self.transcription.assemblyai_api_key:
            self.transcription.assemblyai_api_key = os.getenv("ASSEMBLYAI_API_KEY")
        if os.getenv("DEEPGRAM_API_KEY") and not self.transcription.deepgram_api_key:
            self.transcription.deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")


@dataclass
class TranscriptionResult:
    """Result of transcription processing."""
    
    text: str
    confidence: float
    word_timestamps: List[Dict[str, Any]]
    processing_time: float
    model_name: str
    start_time: float = 0.0  # Start time in seconds for the segment
    end_time: float = 0.0    # End time in seconds for the segment
    language: Optional[str] = None
    speaker_segments: Optional[List[Dict[str, Any]]] = None
    emotions: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None
    raw_response: Optional[Any] = None  # Complete raw AI model output for --raw mode