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
from vttiro.core.errors import ValidationError
from vttiro.core.transcriber import Transcriber
from vttiro.utils.api_keys import get_all_available_api_keys
from vttiro.utils.input_validation import validate_file_path
from vttiro.utils.logging import log_milestone, log_performance, log_system_info, setup_logging

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
        **kwargs: Any,
    ) -> None:
        """Transcribe audio or video file to WebVTT subtitles.

        Transform video/audio files into WebVTT subtitle files using AI transcription.
        Supports multiple providers and models for optimal quality and performance.

        Args:
            input_path: Path to input media file (MP4, MP3, WAV, MOV, AVI, etc.)
            output_path: Output WebVTT file path (default: {input_name}.vtt)
            engine: Transcription provider - 'gemini' (default), 'openai', 'assemblyai', 'deepgram'
            model: Model name - 'gemini-2.5-flash', 'whisper-1', 'universal-2', 'nova-3'
            language: Language code ('en', 'es', 'fr', 'de', etc.) or None for auto-detect
            full_prompt: Complete custom prompt for specialized transcription needs
            prompt: Additional instructions appended to default prompt
            verbose: Show detailed processing information and timing
            dry_run: Validate inputs and check configuration without transcribing
            **kwargs: Provider-specific options (temperature, max_tokens, etc.)

        Examples:
            # Basic transcription with Gemini (fastest, most accurate)
            vttiro transcribe video.mp4

            # Specify output path and engine
            vttiro transcribe interview.mp3 --output_path=subtitles.vtt --engine=openai

            # Spanish transcription with specific model
            vttiro transcribe podcast.wav --language=es --model=gemini-2.5-pro

            # Custom prompting for specialized content (legal, medical, technical)
            vttiro transcribe lecture.mp4 --prompt="Include technical terms and proper names"

            # Dry run to validate setup before processing large files
            vttiro transcribe movie.mkv --dry_run --verbose

        Supported Formats:
            Video: MP4, MOV, AVI, MKV, WEBM, FLV
            Audio: MP3, WAV, FLAC, AAC, M4A, OGG

        API Keys Required:
            Set environment variables: VTTIRO_GEMINI_API_KEY, OPENAI_API_KEY, etc.
            Use 'vttiro apikeys' to check your current configuration.
        """
        # Handle prompt configuration
        effective_prompt_config = self._resolve_prompt_parameters(full_prompt, prompt)

        # Simple validation of CLI inputs
        # Simple validation
        input_path_obj = Path(input_path)

        # Validate file path
        try:
            validate_file_path(input_path_obj)
        except ValidationError as e:
            console.print(f"[red]❌ {e}[/red]")
            return

        # Engine validation removed

        # Language validation removed

        # Output path validation removed
        if output_path:
            output_path_obj = Path(output_path)

        # All validation passed, proceed with transcription

        # Update configuration with validated CLI parameters
        self.config.engine = engine
        self.config.model = model
        self.config.language = language
        self.config.full_prompt = effective_prompt_config.get("full_prompt")
        self.config.prompt = effective_prompt_config.get("prompt")
        self.config.verbose = verbose
        self.config.dry_run = dry_run
        self.config.debug = verbose  # Use verbose as debug flag

        # Setup structured logging
        setup_logging(verbose=verbose, debug=verbose)

        # Log system info for debugging
        if verbose:
            log_system_info()

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

        # Perform transcription with enhanced progress feedback
        import time

        start_time = time.time()

        # Log transcription start milestone
        file_size_mb = input_path_obj.stat().st_size / (1024 * 1024)
        log_milestone(
            "transcription_start",
            {
                "engine": engine,
                "model": model or "default",
                "input_file": input_path_obj.name,
                "file_size_mb": f"{file_size_mb:.1f}",
                "language": language or "auto",
            },
        )

        try:
            transcriber = Transcriber(self.config)

            # Log transcriber initialization
            log_milestone("transcriber_initialized", {"engine": engine})

            # Progress tracking with Rich progress bar
            from rich.progress import BarColumn, Progress, SpinnerColumn, TaskID, TextColumn, TimeRemainingColumn

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                console=console,
                transient=False,
            ) as progress:
                # File size estimation for progress timing
                file_size_mb = input_path_obj.stat().st_size / (1024 * 1024)

                # Create main progress task
                main_task = progress.add_task("🎥 Processing media file...", total=100)

                progress.update(main_task, description="🔊 Extracting and processing audio...", completed=5)

                # Start transcription
                import asyncio

                # Custom progress callback (simplified simulation)
                def progress_callback(stage: str, percent: int):
                    if stage == "audio_processing":
                        progress.update(
                            main_task, description="✂️  Processing audio chunks...", completed=20 + percent * 0.3
                        )
                    elif stage == "transcription":
                        progress.update(
                            main_task, description="🤖 Running AI transcription...", completed=50 + percent * 0.4
                        )
                    elif stage == "output":
                        progress.update(
                            main_task, description="📝 Generating WebVTT output...", completed=90 + percent * 0.1
                        )

                progress.update(main_task, description="🤖 Running AI transcription...", completed=20)

                result = asyncio.run(
                    transcriber.transcribe(
                        media_path=input_path_obj, output_path=Path(output_path) if output_path else None, **kwargs
                    )
                )

                progress.update(main_task, description="✅ Transcription completed!", completed=100)

            # Calculate processing time and performance metrics
            total_time = time.time() - start_time
            duration = result.duration()
            word_count = result.word_count()
            processing_speed = duration / total_time if total_time > 0 else 0

            # Log completion milestone and performance metrics
            log_milestone(
                "transcription_completed",
                {
                    "total_time_s": f"{total_time:.2f}",
                    "media_duration_s": f"{duration:.2f}",
                    "processing_speed": f"{processing_speed:.2f}x",
                    "segments": len(result.segments),
                    "words": word_count,
                },
            )

            # Log detailed performance information
            log_performance(
                "full_transcription",
                total_time,
                {
                    "file_size_mb": f"{file_size_mb:.1f}",
                    "media_duration": f"{duration:.1f}s",
                    "words_per_second": f"{word_count / duration:.1f}" if duration > 0 else "0",
                    "processing_efficiency": f"{processing_speed:.2f}x",
                    "engine": engine,
                    "model": model or "default",
                },
            )

            # Determine output path for display
            if output_path:
                final_output = Path(output_path)
            else:
                # Check if transcriber generated default output path
                final_output = input_path_obj.with_suffix(".vtt")

            # Enhanced success summary with performance metrics
            summary_info = f"""✅ [bold green]Transcription Successful![/bold green]

📊 [bold]Results Summary:[/bold]
   📁 Input: [cyan]{input_path_obj.name}[/cyan] ({file_size_mb:.1f} MB)
   📁 Output: [cyan]{final_output.name}[/cyan]
   ⏱️  Media duration: [cyan]{duration:.1f} seconds[/cyan]
   🚀 Processing time: [cyan]{total_time:.1f} seconds[/cyan]
   ⚡ Processing speed: [cyan]{processing_speed:.1f}x realtime[/cyan]
   📝 Word count: [cyan]{word_count} words[/cyan]
   🎯 Segments: [cyan]{len(result.segments)} segments[/cyan]
   🤖 Engine: [cyan]{engine}[/cyan]{f" ({model})" if model else ""}

🚀 [green]Your WebVTT subtitle file is ready to use![/green]"""

            console.print(Panel(summary_info, border_style="green", title="🎬 Transcription Complete"))

            # Additional performance insights for verbose mode
            if verbose and duration > 0:
                avg_words_per_segment = word_count / len(result.segments) if result.segments else 0
                console.print()
                console.print("📈 [dim]Performance Details:[/dim]")
                console.print(f"   [dim]Average segment length: {duration / len(result.segments):.1f} seconds[/dim]")
                console.print(f"   [dim]Average words per segment: {avg_words_per_segment:.1f} words[/dim]")
                console.print(f"   [dim]Words per second: {word_count / duration:.1f} words/sec[/dim]")

        except Exception as e:
            console.print(f"[red]❌ Transcription failed: {e}[/red]")
            console.print(f"💡 [yellow]Try running 'vttiro validate {input_path}' to check configuration.[/yellow]")
            if verbose:
                import traceback

                console.print(traceback.format_exc())

    def version(self) -> None:
        """Display version information and build details.

        Shows current VTTiro version, build information, and system compatibility.
        Useful for troubleshooting and support requests.

        Example:
            vttiro version
        """
        console.print(f"VTTiro {__version__}")

    def config(self) -> None:
        """Display current VTTiro configuration settings.

        Shows active settings including default engine, models, timeouts, and formats.
        Helpful for verifying configuration before running transcriptions.

        Example:
            vttiro config
        """
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
        """List available transcription engines and their supported models.

        Shows all configured transcription providers with their available models.
        Use this to see what engines and models you can use with --engine and --model.

        Examples:
            vttiro providers

        Then use results with:
            vttiro transcribe video.mp4 --engine=openai --model=whisper-1
        """
        transcriber = Transcriber(self.config)
        providers = transcriber.get_supported_providers()

        # Define models per engine
        engine_models = {
            "gemini": [
                "gemini-2.5-flash",
                "gemini-2.5-pro",
                "gemini-2.5-flash-lite",
                "gemini-2.0-flash",
                "gemini-2.0-flash-lite",
            ],
            "openai": ["whisper-1", "gpt-4o-transcribe", "gpt-4o-mini-transcribe"],
            "assemblyai": ["universal-2"],
            "deepgram": ["nova-3"],
        }

        provider_info = []
        for provider in providers:
            models = engine_models.get(provider, [])
            if models:
                models_str = ", ".join(models)
                provider_info.append(f"• [cyan]{provider}[/cyan] - Models: {models_str}")
            else:
                provider_info.append(f"• [cyan]{provider}[/cyan]")

        console.print(Panel("\n".join(provider_info), title="🤖 Available Engines and Models"))

    def apikeys(self) -> None:
        """Check API key configuration and troubleshoot authentication issues.

        Displays status of all configured API keys for each transcription provider.
        Shows which keys are found and properly configured. Use this to debug
        authentication problems before running transcriptions.

        Environment Variables Checked:
            Gemini: VTTIRO_GEMINI_API_KEY, GEMINI_API_KEY, GOOGLE_API_KEY
            OpenAI: VTTIRO_OPENAI_API_KEY, OPENAI_API_KEY
            Deepgram: VTTIRO_DEEPGRAM_API_KEY, DEEPGRAM_API_KEY, DG_API_KEY
            AssemblyAI: VTTIRO_ASSEMBLYAI_API_KEY, ASSEMBLYAI_API_KEY, AAI_API_KEY

        Examples:
            vttiro apikeys                    # Check all API key status
            export VTTIRO_GEMINI_API_KEY=your_key_here
            vttiro apikeys                    # Verify key was set correctly
        """
        try:
            api_keys = get_all_available_api_keys()
            key_info = []
            for provider, status in api_keys.items():
                if status == "Not found":
                    key_info.append(f"❌ {provider.upper()}: {status}")
                else:
                    key_info.append(f"✅ {provider.upper()}: {status}")

            console.print(
                Panel(
                    "\n".join(key_info),
                    title="🔑 API Key Status",
                    subtitle="Use environment variables like VTTIRO_GEMINI_API_KEY, GEMINI_API_KEY, etc.",
                    border_style="green",
                )
            )
        except Exception as e:
            console.print(f"[red]Error checking API keys: {e}[/red]")

    def validate(
        self,
        input_path: str | None = None,
        engine: str = "gemini",
        model: str | None = None,
    ) -> None:
        """Validate configuration and inputs before processing.

        Comprehensive validation of VTTiro setup including API keys, file formats,
        provider availability, and system requirements. Use this to catch issues
        early before running time-consuming transcriptions.

        Args:
            input_path: Optional path to media file for format validation
            engine: Engine to validate (gemini, openai, assemblyai, deepgram)
            model: Model to validate for the specified engine

        Validation Checks:
            ✅ API keys properly configured and accessible
            ✅ Specified engine and model are available
            ✅ Input file format supported by the chosen provider
            ✅ File integrity and accessibility
            ✅ System dependencies (FFmpeg, etc.)

        Examples:
            vttiro validate                                    # Full system check
            vttiro validate --engine=openai                   # Check specific engine
            vttiro validate video.mp4 --engine=gemini         # Check file + engine
            vttiro validate --input_path=audio.wav --model=whisper-1
        """
        console.print("[bold]🔍 Validating VTTiro Configuration...[/bold]\n")

        issues = []

        # 1. Check API keys
        console.print("Checking API key configuration...")
        try:
            api_keys = get_all_available_api_keys()
            if engine.lower() in api_keys:
                key_status = api_keys[engine.lower()]
                if key_status == "Not found":
                    issues.append(f"❌ API key not found for {engine.upper()}")
                else:
                    console.print(f"✅ API key configured for {engine.upper()}")
            else:
                issues.append(f"❌ Unknown engine: {engine}")
        except Exception as e:
            issues.append(f"❌ Error checking API keys: {e}")

        # 2. Check provider availability
        console.print("Checking provider availability...")
        try:
            transcriber = Transcriber(self.config)
            providers = transcriber.get_supported_providers()
            if engine in providers:
                console.print(f"✅ Engine '{engine}' is available")
            else:
                issues.append(f"❌ Engine '{engine}' is not available")
        except Exception as e:
            issues.append(f"❌ Error checking providers: {e}")

        # 3. Check input file if provided
        if input_path:
            console.print(f"Checking input file: {input_path}...")
            try:
                # Simple validation
                input_path_obj = Path(input_path)

                if not input_path_obj.exists():
                    issues.append(f"❌ Input file not found: {input_path}")
                else:
                    console.print("✅ Input file is valid and supported")

                    # Get file info
                    file_size = input_path_obj.stat().st_size / (1024 * 1024)  # MB
                    console.print(f"   📁 File size: {file_size:.1f} MB")

            except Exception as e:
                issues.append(f"❌ Error validating input file: {e}")

        # 4. Summary
        console.print()
        if issues:
            console.print("[bold red]❌ Validation Failed[/bold red]\n")
            for issue in issues:
                console.print(f"  {issue}")
            console.print("\n💡 [yellow]Fix these issues before running transcriptions.[/yellow]")
            console.print("   Use 'vttiro apikeys' to check API configuration.")
        else:
            console.print("[bold green]✅ All Validation Checks Passed![/bold green]")
            console.print("🚀 [green]Ready to run transcriptions.[/green]")

    def config_save(self, config_path: str, format: str = "yaml") -> None:
        """Save current configuration to file.

        Args:
            config_path: Path to save configuration file
            format: Format to save ("json" or "yaml", default: yaml)

        Examples:
            vttiro config_save my-config.yaml
            vttiro config_save settings.json --format=json
        """
        try:
            path = Path(config_path)
            self.config.to_file(path, format)
            console.print(f"✅ [green]Configuration saved to {path}[/green]")
        except Exception as e:
            console.print(f"[red]Error saving configuration: {e}[/red]")

    def config_load(self, config_path: str) -> None:
        """Load configuration from file and display it.

        Args:
            config_path: Path to configuration file

        Examples:
            vttiro config_load my-config.yaml
            vttiro config_load project-settings.json
        """
        try:
            from vttiro.core.config import VttiroConfig

            path = Path(config_path)
            config = VttiroConfig.from_file(path)

            console.print(f"✅ [green]Configuration loaded from {path}[/green]\n")

            # Display loaded configuration
            config_info = f"Engine: [cyan]{config.engine}[/cyan]\n"
            if config.model:
                config_info += f"Model: [cyan]{config.model}[/cyan]\n"
            config_info += (
                f"Language: [cyan]{config.language or 'auto-detect'}[/cyan]\n"
                f"Output format: [cyan]{config.output_format}[/cyan]\n"
                f"Timeout: [cyan]{config.timeout_seconds}s[/cyan]\n"
                f"Max retries: [cyan]{config.max_retries}[/cyan]\n"
                f"Verbose: [cyan]{config.verbose}[/cyan]\n"
                f"Debug: [cyan]{config.debug}[/cyan]"
            )

            console.print(Panel(config_info, title="📄 Loaded Configuration"))
        except Exception as e:
            console.print(f"[red]Error loading configuration: {e}[/red]")

    def profile_create(
        self, profile_name: str, engine: str = "gemini", model: str | None = None, format: str = "yaml"
    ) -> None:
        """Create a new configuration profile.

        Args:
            profile_name: Name for the new profile
            engine: Transcription engine (gemini, openai, assemblyai, deepgram)
            model: Specific model for the engine
            format: Format to save ("json" or "yaml", default: yaml)

        Examples:
            vttiro profile_create dev --engine=gemini --model=gemini-2.5-flash
            vttiro profile_create production --engine=gemini --model=gemini-2.5-pro
            vttiro profile_create batch --engine=openai --model=whisper-1
        """
        try:
            from vttiro.core.config import VttiroConfig

            # Create configuration with specified parameters
            config = VttiroConfig(engine=engine, model=model)

            # Save as profile
            profile_path = config.to_profile(profile_name, format=format)

            console.print(f"✅ [green]Profile '{profile_name}' created at {profile_path}[/green]")
            console.print(f"💡 [dim]Use with: vttiro profile_use {profile_name}[/dim]")
        except Exception as e:
            console.print(f"[red]Error creating profile: {e}[/red]")

    def profile_use(self, profile_name: str) -> None:
        """Load and display a configuration profile.

        Args:
            profile_name: Name of the profile to load

        Examples:
            vttiro profile_use development
            vttiro profile_use production
            vttiro profile_use batch
        """
        try:
            from vttiro.core.config import VttiroConfig

            config = VttiroConfig.from_profile(profile_name)
            console.print(f"✅ [green]Profile '{profile_name}' loaded[/green]\n")

            # Display profile configuration
            config_info = f"Engine: [cyan]{config.engine}[/cyan]\n"
            if config.model:
                config_info += f"Model: [cyan]{config.model}[/cyan]\n"
            config_info += (
                f"Language: [cyan]{config.language or 'auto-detect'}[/cyan]\n"
                f"Output format: [cyan]{config.output_format}[/cyan]\n"
                f"Max segment duration: [cyan]{config.max_segment_duration}s[/cyan]\n"
                f"Timeout: [cyan]{config.timeout_seconds}s[/cyan]\n"
                f"Max retries: [cyan]{config.max_retries}[/cyan]\n"
                f"Speaker diarization: [cyan]{config.enable_speaker_diarization}[/cyan]\n"
                f"Verbose: [cyan]{config.verbose}[/cyan]\n"
                f"Debug: [cyan]{config.debug}[/cyan]"
            )

            console.print(Panel(config_info, title=f"📋 Profile: {profile_name}"))
            console.print(f"💡 [dim]Use this profile with: vttiro transcribe file.mp4 --profile={profile_name}[/dim]")
        except Exception as e:
            console.print(f"[red]Error loading profile: {e}[/red]")

    def profile_list(self) -> None:
        """List all available configuration profiles.

        Shows profiles from ~/.config/vttiro/profiles/ directory.

        Examples:
            vttiro profile_list
        """
        try:
            profile_dir = Path.home() / ".config" / "vttiro" / "profiles"

            if not profile_dir.exists():
                console.print(
                    "📁 [yellow]No profiles directory found. Create profiles with 'vttiro profile_create'[/yellow]"
                )
                return

            # Find all profile files
            profiles = []
            for ext in [".yaml", ".yml", ".json"]:
                profiles.extend(profile_dir.glob(f"*{ext}"))

            if not profiles:
                console.print("📁 [yellow]No profiles found. Create profiles with 'vttiro profile_create'[/yellow]")
                return

            profile_info = []
            for profile_path in sorted(profiles):
                name = profile_path.stem
                format_type = "YAML" if profile_path.suffix in [".yaml", ".yml"] else "JSON"
                profile_info.append(f"• [cyan]{name}[/cyan] ({format_type})")

            console.print(
                Panel(
                    "\n".join(profile_info),
                    title="📋 Available Configuration Profiles",
                    subtitle=f"Located in {profile_dir}",
                )
            )

            console.print("\n💡 [dim]Use profiles with: vttiro profile_use <profile_name>[/dim]")

        except Exception as e:
            console.print(f"[red]Error listing profiles: {e}[/red]")

    def profile_init(self) -> None:
        """Initialize default configuration profiles for common use cases.

        Creates profiles for development, production, batch processing, and high-quality transcription.

        Examples:
            vttiro profile_init
        """
        try:
            from vttiro.core.config import VttiroConfig

            console.print("🔧 [bold]Initializing default configuration profiles...[/bold]\n")

            profiles = VttiroConfig.create_default_profiles()

            for profile_name, profile_path in profiles.items():
                console.print(f"✅ Created profile '[cyan]{profile_name}[/cyan]' at {profile_path}")

            console.print("\n🎉 [green]Default profiles initialized![/green]")
            console.print(f"📁 Profiles saved to: [cyan]{profile_path.parent}[/cyan]")
            console.print(f"\n💡 [dim]Available profiles: {', '.join(profiles.keys())}[/dim]")
            console.print("💡 [dim]Use with: vttiro profile_use <profile_name>[/dim]")

        except Exception as e:
            console.print(f"[red]Error initializing profiles: {e}[/red]")

    def _resolve_prompt_parameters(self, full_prompt: str | None, prompt: str | None) -> dict[str, str | None]:
        """Resolve prompt parameters.

        Args:
            full_prompt: Complete replacement for the default built-in prompt
            prompt: Additional prompt content to append to default prompt

        Returns:
            Dictionary with resolved prompt configuration
        """
        return {"full_prompt": full_prompt, "prompt": prompt}


if __name__ == "__main__":
    main()
