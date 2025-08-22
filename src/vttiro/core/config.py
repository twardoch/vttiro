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

import json
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field

# Default transcription prompts
DEFAULT_TRANSCRIPTION_PROMPT = """Please provide an accurate transcription of the audio content.
Focus on clarity, correct spelling, and proper punctuation.
If you hear unclear speech, use [unclear] to mark uncertain sections.
Maintain natural speech patterns and include appropriate punctuation."""

# Prompt composition patterns
PROMPT_PATTERNS = {"append": "{default}\n\n{user}", "prepend": "{user}\n\n{default}", "template": "{user}"}


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
        default="gemini", description="AI transcription engine to use"
    )

    # Legacy provider field for backward compatibility
    provider: Literal["gemini", "openai", "assemblyai", "deepgram"] | None = Field(
        default=None, description="[DEPRECATED] Use 'engine' instead. AI transcription provider to use"
    )

    # Model Selection
    model: str | None = Field(default=None, description="Specific model to use for the selected engine")

    # Language and Content
    language: str | None = Field(default=None, description="Language code (ISO 639-1) or None for auto-detection")

    # Prompt Configuration (replaces context)
    full_prompt: str | None = Field(default=None, description="Complete replacement for the default built-in prompt")

    prompt: str | None = Field(default=None, description="Additional prompt content to append to default prompt")

    # Legacy context field for backward compatibility
    context: str | None = Field(
        default=None,
        description="[DEPRECATED] Use 'full_prompt' or 'prompt' instead. Additional context to improve transcription accuracy",
    )

    # Output Configuration
    output_format: Literal["webvtt", "srt", "json"] = Field(
        default="webvtt", description="Output format for transcription results"
    )

    output_path: Path | None = Field(default=None, description="Output file path, None for auto-generation")

    # Processing Options
    enable_speaker_diarization: bool = Field(default=False, description="Enable speaker identification when supported")

    enable_emotion_detection: bool = Field(default=False, description="Enable emotion detection when supported")

    # Audio Processing
    audio_preprocessing: bool = Field(
        default=True, description="Enable audio preprocessing (normalization, format conversion)"
    )

    max_segment_duration: float = Field(
        default=30.0, gt=0, le=120.0, description="Maximum duration for audio segments in seconds"
    )

    # Provider-Specific Settings
    gemini_model: str = Field(default="gemini-2.0-flash", description="Gemini model to use for transcription")

    openai_model: str = Field(default="whisper-1", description="OpenAI model to use for transcription")

    # Performance and Reliability
    timeout_seconds: float = Field(default=300.0, gt=0, description="Request timeout in seconds")

    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum number of retry attempts")

    # Development and Debugging
    verbose: bool = Field(default=False, description="Enable verbose logging and debug output")

    debug: bool = Field(default=False, description="Enable debug mode (preserve working directories)")

    dry_run: bool = Field(default=False, description="Validate configuration and estimate costs without transcribing")

    # Feature Flags for Risk Mitigation
    use_legacy_providers: bool = Field(
        default=False, description="Use legacy src_old providers instead of new src providers (EMERGENCY FALLBACK)"
    )

    legacy_fallback_enabled: bool = Field(
        default=True, description="Automatically fallback to src_old providers if new providers fail"
    )

    legacy_fallback_threshold: int = Field(
        default=2, ge=1, le=10, description="Number of failures before triggering legacy fallback"
    )

    model_config = ConfigDict(env_prefix="VTTIRO_", env_file=".env", validate_assignment=True, extra="forbid")

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

    def get_effective_prompt(self, pattern: str = "append") -> str:
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

        if self.prompt:
            # Combine user prompt with default prompt using specified pattern
            user_prompt = self.prompt.strip()

            if pattern in PROMPT_PATTERNS:
                template = PROMPT_PATTERNS[pattern]
                return template.format(default=DEFAULT_TRANSCRIPTION_PROMPT.strip(), user=user_prompt).strip()
            # Fallback to append pattern
            return f"{DEFAULT_TRANSCRIPTION_PROMPT.strip()}\n\n{user_prompt}"

        if self.context:
            # Legacy context field - treat as appended prompt for backward compatibility
            context_prompt = self.context.strip()
            return f"{DEFAULT_TRANSCRIPTION_PROMPT.strip()}\n\n{context_prompt}"

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
                raise ValueError("YAML support not available. Install with: pip install pyyaml")

        return cls(**data)

    def to_file(self, config_path: Path, format: Literal["json", "yaml"] = "yaml") -> None:
        """Save configuration to file.

        Args:
            config_path: Path to save configuration file
            format: Format to save ("json" or "yaml")

        Raises:
            ValueError: Invalid format or missing YAML support
        """
        # Create parent directory if it doesn't exist
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Get configuration data
        data = self.to_dict()

        # Remove None values and empty fields for cleaner output
        clean_data = {k: v for k, v in data.items() if v is not None and v != ""}

        if format.lower() == "json":
            with open(config_path, "w") as f:
                json.dump(clean_data, f, indent=2)
        elif format.lower() == "yaml":
            try:
                with open(config_path, "w") as f:
                    yaml.dump(clean_data, f, default_flow_style=False, indent=2)
            except NameError:
                raise ValueError("YAML support not available. Install with: pip install pyyaml")
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'yaml'")
