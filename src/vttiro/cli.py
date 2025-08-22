# this_file: src/vttiro/cli.py
"""Command-line interface for VTTiro 2.0.

This module provides the main CLI entry point for the vttiro command.
During the migration period, this serves as a thin wrapper that will
eventually coordinate with the core application logic.

Used by:
- Package entry point (vttiro command)
- Direct execution (python -m vttiro)
- Integration testing of CLI interface
"""

import sys
from pathlib import Path
from typing import Any

import fire
from rich.console import Console
from rich.panel import Panel

from vttiro import __version__
from vttiro.core.config import VttiroConfig
from vttiro.core.transcriber import Transcriber
from vttiro.utils.input_validation import InputValidator
from vttiro.utils.api_keys import get_all_available_api_keys

console = Console()


def main() -> None:
    """Main CLI entry point.

    Uses Fire to automatically generate CLI from the VttiroCLI class.
    Provides a clean interface for all transcription operations.
    """
    try:
        fire.Fire(VttiroCLI)
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


class VttiroCLI:
    """VTTiro 2.0 command-line interface.

    Provides transcription capabilities through a clean CLI interface.
    This is a placeholder implementation during the migration period.
    """

    def __init__(self):
        """Initialize CLI with default configuration."""
        self.config = VttiroConfig()

    def transcribe(
        self,
        input_path: str,
        output_path: str | None = None,
        engine: str = "gemini",
        model: str | None = None,
        language: str | None = None,
        full_prompt: str | None = None,
        prompt: str | None = None,
        verbose: bool = False,
        dry_run: bool = False,
        **kwargs: Any
    ) -> None:
        """Transcribe audio or video file to WebVTT subtitles.

        Args:
            input_path: Path to input audio/video file
            output_path: Path for output file (auto-generated if not provided)
            engine: Transcription engine (gemini, openai, assemblyai, deepgram)
            model: Specific model for the engine (e.g., 'gemini-2.5-flash')
            language: Language code (e.g., 'en', 'es') or None for auto-detection
            full_prompt: Complete replacement for the default built-in prompt
            prompt: Additional prompt content to append to default prompt
            verbose: Enable verbose logging and debug output
            dry_run: Validate configuration and estimate costs without transcribing
            **kwargs: Additional engine-specific parameters
        """
        # Handle prompt configuration
        effective_prompt_config = self._resolve_prompt_parameters(full_prompt, prompt)

        # Simple validation of CLI inputs
        validator = InputValidator()
        input_path_obj = Path(input_path)

        # Validate file path
        if not validator.validate_file_path(input_path_obj, engine):
            console.print(f"[red]❌ File validation failed for: {input_path_obj}[/red]")
            return

        # Validate engine
        if not validator.validate_provider_name(engine):
            console.print(f"[red]❌ Unsupported engine: {engine}[/red]")
            console.print(f"[yellow]Supported: {', '.join(['gemini', 'openai', 'assemblyai', 'deepgram'])}[/yellow]")
            return

        # Validate language if specified
        if language and not validator.validate_language_code(language):
            console.print(f"[red]❌ Unsupported language: {language}[/red]")
            return

        # Validate output path if specified
        if output_path:
            output_path_obj = Path(output_path)
            if not validator.validate_output_path(output_path_obj):
                console.print(f"[red]❌ Output path validation failed: {output_path_obj}[/red]")
                return

        # All validation passed, proceed with transcription

        # Update configuration with validated CLI parameters
        self.config.engine = engine
        self.config.model = model
        self.config.language = language
        self.config.full_prompt = effective_prompt_config.get('full_prompt')
        self.config.prompt = effective_prompt_config.get('prompt')
        self.config.verbose = verbose
        self.config.dry_run = dry_run
        self.config.debug = verbose  # Use verbose as debug flag

        if output_path:
            self.config.output_path = Path(output_path)

        # Display banner with validation success
        if verbose:
            banner_info = f"[bold blue]VTTiro 2.0[/bold blue] - AI Video Transcription\nEngine: [cyan]{engine}[/cyan]"
            if model:
                banner_info += f"\nModel: [cyan]{model}[/cyan]"
            banner_info += f"\nInput: [green]{input_path}[/green]\n✅ [dim]Input validation passed[/dim]"

            console.print(Panel(banner_info, title="🎥 Starting Transcription"))

        # Placeholder implementation
        console.print("[yellow]⚠️  VTTiro 2.0 is under development[/yellow]")
        console.print(f"Would transcribe: [green]{input_path}[/green]")
        console.print(f"Using engine: [cyan]{engine}[/cyan]")
        if model:
            console.print(f"Using model: [cyan]{model}[/cyan]")

        if dry_run:
            console.print("[blue]Dry run mode - no actual transcription performed[/blue]")
            return

        # Perform transcription using simplified transcriber
        try:
            transcriber = Transcriber(self.config)
            console.print(f"Starting transcription of: [cyan]{input_path}[/cyan]")
            
            import asyncio
            result = asyncio.run(transcriber.transcribe(
                media_path=input_path_obj,
                output_path=Path(output_path) if output_path else None,
                **kwargs
            ))
            
            console.print(f"[green]✓ Transcription completed![/green]")
            console.print(f"Segments: {len(result.segments)}")
            if output_path:
                console.print(f"Output: [cyan]{output_path}[/cyan]")
                
        except Exception as e:
            console.print(f"[red]❌ Transcription failed: {e}[/red]")
            if verbose:
                import traceback
                console.print(traceback.format_exc())

    def version(self) -> None:
        """Display version information."""
        console.print(f"VTTiro {__version__}")

    def config(self) -> None:
        """Display current configuration."""
        config_info = f"Engine: [cyan]{self.config.engine}[/cyan]\n"
        if self.config.model:
            config_info += f"Model: [cyan]{self.config.model}[/cyan]\n"
        config_info += (
            f"Language: [cyan]{self.config.language or 'auto-detect'}[/cyan]\n"
            f"Output format: [cyan]{self.config.output_format}[/cyan]\n"
            f"Max segment duration: [cyan]{self.config.max_segment_duration}s[/cyan]\n"
            f"Timeout: [cyan]{self.config.timeout_seconds}s[/cyan]\n"
            f"Max retries: [cyan]{self.config.max_retries}[/cyan]"
        )

        console.print(Panel(config_info, title="⚙️ Current Configuration"))

    def providers(self) -> None:
        """List available transcription engines and their models."""
        transcriber = Transcriber(self.config)
        providers = transcriber.get_supported_providers()

        # Define models per engine
        engine_models = {
            'gemini': [
                'gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-2.5-flash-lite',
                'gemini-2.0-flash', 'gemini-2.0-flash-lite'
            ],
            'openai': ['whisper-1', 'gpt-4o-transcribe', 'gpt-4o-mini-transcribe'],
            'assemblyai': ['universal-2'],
            'deepgram': ['nova-3']
        }

        provider_info = []
        for provider in providers:
            models = engine_models.get(provider, [])
            if models:
                models_str = ', '.join(models)
                provider_info.append(f"• [cyan]{provider}[/cyan] - Models: {models_str}")
            else:
                provider_info.append(f"• [cyan]{provider}[/cyan]")

        console.print(Panel(
            "\n".join(provider_info),
            title="🤖 Available Engines and Models"
        ))

    def apikeys(self) -> None:
        """Show available API keys for debugging configuration issues."""
        try:
            api_keys = get_all_available_api_keys()
            key_info = []
            for provider, status in api_keys.items():
                if status == "Not found":
                    key_info.append(f"❌ {provider.upper()}: {status}")
                else:
                    key_info.append(f"✅ {provider.upper()}: {status}")
            
            console.print(Panel(
                "\n".join(key_info),
                title="🔑 API Key Status",
                subtitle="Use environment variables like VTTIRO_GEMINI_API_KEY, GEMINI_API_KEY, etc.",
                border_style="green"
            ))
        except Exception as e:
            console.print(f"[red]Error checking API keys: {e}[/red]")

    def _resolve_prompt_parameters(
        self,
        full_prompt: str | None,
        prompt: str | None
    ) -> dict[str, str | None]:
        """Resolve prompt parameters.

        Args:
            full_prompt: Complete replacement for the default built-in prompt
            prompt: Additional prompt content to append to default prompt

        Returns:
            Dictionary with resolved prompt configuration
        """
        return {
            'full_prompt': full_prompt,
            'prompt': prompt
        }


if __name__ == "__main__":
    main()
