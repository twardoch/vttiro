# this_file: src/vttiro/core/config.py
"""Configuration management for VTTiro 2.0.

This module provides typed configuration using Pydantic models, replacing
the complex enhanced configuration system from v1. Focuses on simplicity,
type safety, and clear validation rules.

Key improvements over v1:
- Pydantic-based validation instead of custom validation logic
- Simplified flat structure instead of nested provider configs
- Environment variable support with clear precedence
- Typed fields with sensible defaults

Used by:
- Core orchestration for provider selection and parameters
- CLI for argument parsing and validation
- Provider implementations for accessing settings
- Tests for configuration validation
"""

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, ConfigDict

# Default transcription prompts
DEFAULT_TRANSCRIPTION_PROMPT = """Please provide an accurate transcription of the audio content.
Focus on clarity, correct spelling, and proper punctuation.
If you hear unclear speech, use [unclear] to mark uncertain sections.
Maintain natural speech patterns and include appropriate punctuation."""

# Prompt composition patterns
PROMPT_PATTERNS = {
    'append': '{default}\n\n{user}',
    'prepend': '{user}\n\n{default}',
    'template': '{user}'
}


class VttiroConfig(BaseModel):
    """Main configuration for VTTiro transcription.
    
    Simplified configuration model that replaces the complex enhanced config
    system from v1. Uses Pydantic for validation and type safety.
    
    Configuration Precedence:
    1. Explicit parameters passed to methods
    2. Environment variables (VTTIRO_*)
    3. Configuration file values
    4. Default values defined here
    """
    
    # Engine Selection (replaces provider)
    engine: Literal["gemini", "openai", "assemblyai", "deepgram"] = Field(
        default="gemini",
        description="AI transcription engine to use"
    )
    
    # Legacy provider field for backward compatibility
    provider: Literal["gemini", "openai", "assemblyai", "deepgram"] | None = Field(
        default=None,
        description="[DEPRECATED] Use 'engine' instead. AI transcription provider to use"
    )
    
    # Model Selection
    model: str | None = Field(
        default=None,
        description="Specific model to use for the selected engine"
    )
    
    # Language and Content
    language: str | None = Field(
        default=None,
        description="Language code (ISO 639-1) or None for auto-detection"
    )
    
    # Prompt Configuration (replaces context)
    full_prompt: str | None = Field(
        default=None,
        description="Complete replacement for the default built-in prompt"
    )
    
    prompt: str | None = Field(
        default=None,
        description="Additional prompt content to append to default prompt"
    )
    
    # Legacy context field for backward compatibility
    context: str | None = Field(
        default=None,
        description="[DEPRECATED] Use 'full_prompt' or 'prompt' instead. Additional context to improve transcription accuracy"
    )
    
    # Output Configuration  
    output_format: Literal["webvtt", "srt", "json"] = Field(
        default="webvtt",
        description="Output format for transcription results"
    )
    
    output_path: Path | None = Field(
        default=None,
        description="Output file path, None for auto-generation"
    )
    
    # Processing Options
    enable_speaker_diarization: bool = Field(
        default=False,
        description="Enable speaker identification when supported"
    )
    
    enable_emotion_detection: bool = Field(
        default=False,
        description="Enable emotion detection when supported"
    )
    
    # Audio Processing
    audio_preprocessing: bool = Field(
        default=True,
        description="Enable audio preprocessing (normalization, format conversion)"
    )
    
    max_segment_duration: float = Field(
        default=30.0,
        gt=0,
        le=120.0,
        description="Maximum duration for audio segments in seconds"
    )
    
    # Provider-Specific Settings
    gemini_model: str = Field(
        default="gemini-2.0-flash",
        description="Gemini model to use for transcription"
    )
    
    openai_model: str = Field(
        default="whisper-1",
        description="OpenAI model to use for transcription"
    )
    
    # Performance and Reliability
    timeout_seconds: float = Field(
        default=300.0,
        gt=0,
        description="Request timeout in seconds"
    )
    
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum number of retry attempts"
    )
    
    # Development and Debugging
    verbose: bool = Field(
        default=False,
        description="Enable verbose logging and debug output"
    )
    
    debug: bool = Field(
        default=False,
        description="Enable debug mode (preserve working directories)"
    )
    
    dry_run: bool = Field(
        default=False,
        description="Validate configuration and estimate costs without transcribing"
    )
    
    # Feature Flags for Risk Mitigation
    use_legacy_providers: bool = Field(
        default=False,
        description="Use legacy src_old providers instead of new src providers (EMERGENCY FALLBACK)"
    )
    
    legacy_fallback_enabled: bool = Field(
        default=True,
        description="Automatically fallback to src_old providers if new providers fail"
    )
    
    legacy_fallback_threshold: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Number of failures before triggering legacy fallback"
    )
    
    model_config = ConfigDict(
        env_prefix="VTTIRO_",
        env_file=".env",
        validate_assignment=True,
        extra="forbid"  # Prevent unknown fields
    )
    
    @field_validator('model')
    @classmethod
    def validate_model(cls, v, info):
        """Validate model name for the selected engine."""
        if v is not None:
            # Get engine from context - info.data contains the parsed values
            engine = info.data.get('engine', 'gemini')
            if not engine and info.data.get('provider'):
                engine = info.data.get('provider')
            
            # Define valid models per engine
            valid_models = {
                'gemini': [
                    'gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-2.5-flash-lite',
                    'gemini-2.0-flash', 'gemini-2.0-flash-lite'
                ],
                'openai': ['whisper-1', 'gpt-4o-transcribe', 'gpt-4o-mini-transcribe'],
                'assemblyai': ['universal-2'],
                'deepgram': ['nova-3']
            }
            
            if engine in valid_models and v not in valid_models[engine]:
                valid_list = ', '.join(valid_models[engine])
                raise ValueError(f"Invalid model '{v}' for engine '{engine}'. Valid models: {valid_list}")
        
        return v
    
    @field_validator('full_prompt', 'prompt')
    @classmethod
    def validate_prompt(cls, v):
        """Validate prompt content."""
        if v is not None:
            # Basic validation - length limits and content sanitization
            if len(v.strip()) == 0:
                raise ValueError("Prompt cannot be empty or whitespace only")
            if len(v) > 10000:  # Reasonable limit
                raise ValueError("Prompt is too long (maximum 10,000 characters)")
        return v
    
    @field_validator('language')
    @classmethod
    def validate_language_code(cls, v):
        """Validate language code format."""
        if v is not None and len(v) not in [2, 5]:  # ISO 639-1 or locale format
            raise ValueError(f"Invalid language code format: {v}")
        return v
    
    @field_validator('output_path')
    @classmethod
    def validate_output_path(cls, v):
        """Validate output path."""
        if v is not None:
            v = Path(v)
            # Ensure parent directory exists or can be created
            v.parent.mkdir(parents=True, exist_ok=True)
        return v
    
    def get_provider_config(self) -> dict[str, Any]:
        """Get provider-specific configuration.
        
        Returns:
            Dictionary with provider-specific settings
        """
        # Use engine, fallback to provider for backward compatibility
        current_engine = self.engine or self.provider or "gemini"
        
        config = {}
        
        # Use explicit model if provided, otherwise use engine-specific defaults
        if self.model:
            config["model"] = self.model
        elif current_engine == "gemini":
            config["model"] = self.gemini_model
        elif current_engine == "openai":
            config["model"] = self.openai_model
        
        return config
    
    def validate_effective_prompt(self, pattern: str = 'append') -> tuple[bool, str | None]:
        """Validate the effective prompt length and content.
        
        Args:
            pattern: How to combine prompts ('append', 'prepend', 'template')
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            effective_prompt = self.get_effective_prompt(pattern)
            
            # Check length limits
            if len(effective_prompt) > 15000:  # More generous limit for combined prompts
                return False, f"Combined prompt is too long ({len(effective_prompt)} chars, max 15,000)"
            
            # Check for obviously problematic content
            if not effective_prompt.strip():
                return False, "Effective prompt is empty after processing"
            
            # Check for reasonable content (basic sanity check)
            if len(effective_prompt.strip()) < 10:
                return False, "Effective prompt is too short to be meaningful"
            
            return True, None
            
        except Exception as e:
            return False, f"Error validating prompt: {str(e)}"
    
    def get_effective_prompt(self, pattern: str = 'append') -> str:
        """Get the effective prompt based on full_prompt, prompt, and context.
        
        Args:
            pattern: How to combine prompts ('append', 'prepend', 'template')
            
        Returns:
            The resolved prompt to use
        """
        # Priority: full_prompt > prompt + default > context (legacy) > default
        if self.full_prompt:
            # Full prompt completely replaces default
            return self.full_prompt.strip()
        
        elif self.prompt:
            # Combine user prompt with default prompt using specified pattern
            user_prompt = self.prompt.strip()
            
            if pattern in PROMPT_PATTERNS:
                template = PROMPT_PATTERNS[pattern]
                return template.format(
                    default=DEFAULT_TRANSCRIPTION_PROMPT.strip(),
                    user=user_prompt
                ).strip()
            else:
                # Fallback to append pattern
                return f"{DEFAULT_TRANSCRIPTION_PROMPT.strip()}\n\n{user_prompt}"
        
        elif self.context:
            # Legacy context field - treat as appended prompt for backward compatibility
            context_prompt = self.context.strip()
            return f"{DEFAULT_TRANSCRIPTION_PROMPT.strip()}\n\n{context_prompt}"
        
        else:
            # No custom prompt provided, use default
            return DEFAULT_TRANSCRIPTION_PROMPT.strip()
    
    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Configuration as dictionary with resolved paths
        """
        data = self.model_dump()
        # Convert Path objects to strings for serialization
        if data.get("output_path"):
            data["output_path"] = str(data["output_path"])
        return data
    
    @classmethod
    def from_file(cls, config_path: Path) -> "VttiroConfig":
        """Load configuration from file.
        
        Args:
            config_path: Path to configuration file (JSON or YAML)
            
        Returns:
            VttiroConfig instance
            
        Raises:
            FileNotFoundError: Configuration file not found
            ValueError: Invalid configuration format
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        import json
        
        if config_path.suffix.lower() == ".json":
            with open(config_path) as f:
                data = json.load(f)
        else:
            # Try YAML if available
            try:
                import yaml
                with open(config_path) as f:
                    data = yaml.safe_load(f)
            except ImportError:
                raise ValueError(
                    f"YAML support not available. Install with: pip install pyyaml"
                )
        
        return cls(**data)