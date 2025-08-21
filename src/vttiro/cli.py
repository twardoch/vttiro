#!/usr/bin/env python3
# this_file: src/vttiro/cli.py
"""Simple command-line interface for vttiro transcription."""

import sys
import asyncio
from pathlib import Path
from typing import Optional

try:
    import ffmpeg
except ImportError:
    ffmpeg = None

try:
    import fire
    from rich.console import Console
    from rich.progress import track
    from loguru import logger
except ImportError as e:
    print(f"Error: Missing required dependency: {e}")
    print("Please install vttiro with: uv pip install --system vttiro")
    sys.exit(1)

from vttiro.__version__ import __version__
from vttiro.core.file_transcriber import FileTranscriber
from vttiro.models import (
    TranscriptionEngine,
    GeminiModel,
    AssemblyAIModel,
    DeepgramModel,
    ModelCapability,
    get_default_model,
    get_available_models,
    validate_engine_model_combination,
    get_model_capabilities,
    get_models_by_capability,
    estimate_transcription_cost,
)
from vttiro.utils.prompt_utils import (
    process_prompt_argument,
    get_prompt_preview,
    validate_prompt_combination,
)
from vttiro.utils.exceptions import ValidationError
from vttiro.monitoring import get_performance_monitor
from vttiro.utils.config_validation import (
    ConfigurationValidator,
    ValidationSeverity,
    validate_startup_configuration,
)
from vttiro.utils.config_migration import (
    ConfigurationMigrator,
    ConfigVersion,
    check_migration_needed,
)


class VttiroCLI:
    """Simple command-line interface for vttiro file transcription."""

    def __init__(self):
        self.console = Console()

    def version(self) -> str:
        """Show vttiro version information."""
        return f"vttiro {__version__}"

    def _analyze_file_and_recommend_model(
        self, file_path: Path, engine: str, model: str
    ) -> None:
        """Analyze input file and provide model recommendations with warnings."""
        try:
            # Get file size
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            file_size_gb = file_size_mb / 1024

            # Try to get duration using ffmpeg
            duration_minutes = None
            if ffmpeg:
                try:
                    probe = ffmpeg.probe(str(file_path))
                    duration_seconds = float(probe.get("format", {}).get("duration", 0))
                    duration_minutes = duration_seconds / 60
                except Exception:
                    pass  # Ignore ffmpeg errors, just won't show duration info

            # Get current model capabilities
            try:
                current_capability = get_model_capabilities(model)
            except ValueError:
                # Model not found in capabilities, skip analysis
                return

            # File size warnings
            if file_size_gb > 1:
                self.console.print(
                    f"[yellow]⚠️  Large file detected:[/yellow] {file_size_gb:.1f}GB - processing may take a while"
                )
            elif file_size_mb > 100:
                self.console.print(f"[dim]📊 File size:[/dim] {file_size_mb:.0f}MB")

            # Duration warnings and recommendations
            if duration_minutes:
                self.console.print(
                    f"[dim]⏱️  Estimated duration:[/dim] {duration_minutes:.1f} minutes"
                )

                # Check duration limits
                if current_capability.max_duration_seconds:
                    max_minutes = current_capability.max_duration_seconds / 60
                    if duration_minutes > max_minutes:
                        self.console.print(
                            f"[red]⚠️  Duration exceeds model limit![/red]"
                        )
                        self.console.print(
                            f"[red]   {model} supports max {max_minutes:.0f} minutes, your file is {duration_minutes:.1f} minutes[/red]"
                        )

                        # Suggest alternative models
                        alternative_models = []
                        engine_enum = TranscriptionEngine(engine)
                        available_models = get_available_models(engine_enum)

                        for alt_model in available_models:
                            try:
                                alt_cap = get_model_capabilities(alt_model)
                                if (
                                    alt_cap.max_duration_seconds is None
                                    or alt_cap.max_duration_seconds / 60
                                    > duration_minutes
                                ):
                                    alternative_models.append(alt_model)
                            except ValueError:
                                continue

                        if alternative_models:
                            self.console.print(
                                f"[cyan]💡 Try these {engine} models instead:[/cyan] {', '.join(alternative_models[:3])}"
                            )
                        else:
                            # Suggest other engines
                            self.console.print(
                                "[cyan]💡 Consider using a different engine (assemblyai, deepgram) for long files[/cyan]"
                            )
                        return

                # Cost estimation and warnings
                estimated_cost = estimate_transcription_cost(model, duration_minutes)
                if estimated_cost:
                    if estimated_cost > 1.0:  # More than $1
                        self.console.print(
                            f"[yellow]💰 Estimated cost:[/yellow] ${estimated_cost:.2f}"
                        )
                        if estimated_cost > 10.0:  # High cost warning
                            self.console.print(
                                "[yellow]⚠️  High cost alert! Consider using a cheaper model for large files[/yellow]"
                            )

                            # Suggest cheaper alternatives
                            engine_enum = TranscriptionEngine(engine)
                            cheap_models = get_models_by_capability(
                                "cheap", engine_enum
                            )
                            if cheap_models and model not in cheap_models:
                                self.console.print(
                                    f"[cyan]💡 Cheaper {engine} options:[/cyan] {', '.join(cheap_models[:2])}"
                                )
                    else:
                        self.console.print(
                            f"[dim]💰 Estimated cost:[/dim] ${estimated_cost:.3f}"
                        )

                # Performance suggestions based on file characteristics
                if duration_minutes > 60:  # Long files
                    fast_models = get_models_by_capability("fast", engine_enum)
                    if fast_models and model not in fast_models:
                        self.console.print(
                            f"[cyan]🚀 For faster processing:[/cyan] {', '.join(fast_models[:2])}"
                        )

                # Quality suggestions for important content
                if duration_minutes < 30:  # Shorter, potentially important content
                    accurate_models = get_models_by_capability("accurate", engine_enum)
                    if accurate_models and model not in accurate_models:
                        self.console.print(
                            f"[cyan]🎯 For higher accuracy:[/cyan] {', '.join(accurate_models[:2])}"
                        )

                # Warn about model thresholds
                if (
                    current_capability.warning_threshold_minutes
                    and duration_minutes > current_capability.warning_threshold_minutes
                ):
                    threshold = current_capability.warning_threshold_minutes
                    self.console.print(
                        f"[yellow]⚠️  File exceeds recommended duration for {model} ({threshold:.0f} min threshold)[/yellow]"
                    )

        except Exception as e:
            # Don't let file analysis errors break transcription
            self.console.print(
                f"[dim]Note: Could not analyze file details ({str(e)})[/dim]"
            )

    def transcribe(
        self,
        input_file: str,
        output: Optional[str] = None,
        engine: str = "gemini",
        model: Optional[str] = None,
        full_prompt: Optional[str] = None,
        xtra_prompt: Optional[str] = None,
        add_cues: bool = False,
        verbose: bool = False,
        keep_audio: bool = False,
    ) -> str:
        """Transcribe audio/video file to WebVTT subtitles.

        Args:
            input_file: Path to audio/video file (MP4, MP3, WAV, MOV, etc.)
            output: Output WebVTT file path (optional)
            engine: AI engine to use (gemini, assemblyai, deepgram, openai)
            model: Specific model within engine (optional, uses engine default)
            full_prompt: Replace default prompt entirely (file path or text)
            xtra_prompt: Append to default/custom prompt (file path or text)
            add_cues: Include cue identifiers in WebVTT output (default: False)
            verbose: Enable detailed debug logging for troubleshooting (default: False)
            keep_audio: Save audio file next to video with same basename, reuse existing (default: False)

        Returns:
            Path to generated WebVTT file

        Examples:
            vttiro transcribe video.mp4
            vttiro transcribe audio.mp3 --output subtitles.vtt
            vttiro transcribe video.mp4 --engine assemblyai
            vttiro transcribe video.mp4 --engine gemini --model gemini-2.5-pro
            vttiro transcribe video.mp4 --engine openai --model whisper-1
            vttiro transcribe video.mp4 --full_prompt "Custom transcription instructions"
            vttiro transcribe video.mp4 --xtra_prompt "Focus on technical terms"
            vttiro transcribe video.mp4 --add_cues
            vttiro transcribe video.mp4 --keep_audio
        """
        try:
            # Setup verbose logging if requested
            if verbose:
                import loguru
                from loguru import logger
                # Set DEBUG level and add more detailed output
                logger.remove()  # Remove default handler
                logger.add(
                    sys.stderr, 
                    level="DEBUG", 
                    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
                    enqueue=True
                )
                self.console.print("[dim]🔍 Verbose logging enabled[/dim]")
            
            # Validate input file exists and format
            input_path = Path(input_file)
            if not input_path.exists():
                self.console.print(f"[red]✗[/red] File not found: {input_file}")
                self.console.print(
                    f"[yellow]💡 Tip:[/yellow] Make sure the file path is correct and the file exists"
                )
                return ""

            # Validate file format
            if not self._is_supported_format(input_path):
                self.console.print(
                    f"[red]✗[/red] Unsupported file format: {input_path.suffix}"
                )
                self.console.print(
                    "[yellow]💡 Tip:[/yellow] Use `vttiro formats` to see supported formats"
                )
                return ""

            # Validate engine
            try:
                engine_enum = TranscriptionEngine(engine)
            except ValueError:
                self.console.print(f"[red]✗[/red] Invalid engine: {engine}")
                self.console.print(
                    f"[cyan]Available engines:[/cyan] {', '.join([e.value for e in TranscriptionEngine])}"
                )
                self.console.print(
                    f"[yellow]💡 Tip:[/yellow] Use `vttiro engines` to see details about each engine"
                )
                return ""

            # Set default model if not specified
            if model is None:
                model = get_default_model(engine_enum)
            else:
                # Validate engine/model combination
                if not validate_engine_model_combination(engine, model):
                    self.console.print(
                        f"[red]✗[/red] Invalid model '{model}' for engine '{engine}'"
                    )
                    available = get_available_models(engine_enum)
                    self.console.print(
                        f"[cyan]Available models for {engine}:[/cyan] {', '.join(available)}"
                    )
                    self.console.print(
                        f"[yellow]💡 Tip:[/yellow] Use `vttiro models --engine {engine}` to see all {engine} models"
                    )
                    return ""

            self.console.print(f"[green]Transcribing[/green] {input_path.name}...")
            self.console.print(f"[blue]Engine:[/blue] {engine}")
            self.console.print(f"[blue]Model:[/blue] {model}")

            # Validate and process prompt arguments
            processed_full_prompt = None
            processed_xtra_prompt = None

            if full_prompt or xtra_prompt:
                try:
                    # Process full prompt
                    if full_prompt:
                        processed_full_prompt = process_prompt_argument(
                            full_prompt, "--full_prompt", is_full_prompt=True
                        )

                    # Process extra prompt
                    if xtra_prompt:
                        processed_xtra_prompt = process_prompt_argument(
                            xtra_prompt, "--xtra_prompt", is_full_prompt=False
                        )

                    # Validate prompt combination
                    is_valid, warnings = validate_prompt_combination(
                        processed_full_prompt, processed_xtra_prompt
                    )

                    if warnings:
                        for warning in warnings:
                            self.console.print(
                                f"[yellow]⚠️ Prompt Warning:[/yellow] {warning}"
                            )

                    if not is_valid:
                        self.console.print("[red]✗[/red] Prompt validation failed")
                        self.console.print(
                            "[yellow]💡 Tip:[/yellow] Check prompt content and ensure it requests WebVTT format"
                        )
                        return ""

                except ValidationError as e:
                    self.console.print(f"[red]✗ Prompt Error:[/red] {str(e)}")
                    self.console.print(
                        "[yellow]💡 Tip:[/yellow] Check file path, permissions, and content format"
                    )
                    return ""

            # Show prompt customization info if used
            if processed_full_prompt or processed_xtra_prompt or add_cues:
                self.console.print("[blue]Prompt Customization:[/blue]")
                if processed_full_prompt:
                    prompt_preview = get_prompt_preview(processed_full_prompt, 60)
                    self.console.print(f"  [dim]Custom prompt:[/dim] {prompt_preview}")
                if processed_xtra_prompt:
                    extra_preview = get_prompt_preview(processed_xtra_prompt, 60)
                    self.console.print(f"  [dim]Extra prompt:[/dim] {extra_preview}")
                if add_cues:
                    self.console.print(f"  [dim]Include cue IDs:[/dim] Yes")

            # Analyze file and provide recommendations
            self._analyze_file_and_recommend_model(input_path, engine, model)

            # Check for required dependencies
            if not self._check_engine_dependencies(engine):
                return ""

            # Create file transcriber
            transcriber = FileTranscriber()

            # Run transcription with progress indication
            self.console.print("[cyan]🎬 Starting transcription...[/cyan]")

            # Prepare prompt customization options with validated prompts
            prompt_options = {
                "full_prompt": processed_full_prompt,
                "xtra_prompt": processed_xtra_prompt,
                "add_cues": add_cues,
                "keep_audio": keep_audio,
            }

            with self.console.status(
                "[cyan]Processing audio...[/cyan]", spinner="dots"
            ):
                result_path = asyncio.run(
                    transcriber.transcribe_file(
                        input_file, output, engine, model, **prompt_options
                    )
                )

            self.console.print(
                f"[green]✓ Success![/green] Transcription saved to [cyan]{result_path}[/cyan]"
            )
            return str(result_path)

        except FileNotFoundError as e:
            self.console.print(f"[red]✗ File Error:[/red] {str(e)}")
            self.console.print(
                "[yellow]💡 Tip:[/yellow] Check that the input file exists and is accessible"
            )
        except ValidationError as e:
            self.console.print(f"[red]✗ Validation Error:[/red] {str(e)}")
            self.console.print(
                "[yellow]💡 Tip:[/yellow] Check your input parameters and file formats"
            )
        except ValueError as e:
            self.console.print(f"[red]✗ Configuration Error:[/red] {str(e)}")
            self.console.print(
                "[yellow]💡 Tip:[/yellow] Check your engine/model combination or API keys"
            )
        except ImportError as e:
            self.console.print(f"[red]✗ Missing Dependency:[/red] {str(e)}")
            self.console.print(
                "[yellow]💡 Tip:[/yellow] Install the required package or check your installation"
            )
        except ConnectionError as e:
            self.console.print(f"[red]✗ Network Error:[/red] {str(e)}")
            self.console.print(
                "[yellow]💡 Tip:[/yellow] Check your internet connection and API key validity"
            )
        except TimeoutError as e:
            self.console.print(f"[red]✗ Timeout Error:[/red] {str(e)}")
            self.console.print(
                "[yellow]💡 Tip:[/yellow] Try again or use a different model/engine"
            )
        except Exception as e:
            self.console.print(f"[red]✗ Unexpected Error:[/red] {str(e)}")
            self.console.print(
                "[yellow]💡 Tip:[/yellow] Try running with a different engine or check the file format"
            )
            self.console.print(
                "[dim]For more help, visit: https://github.com/twardoch/vttiro/issues[/dim]"
            )

        return ""  # Return empty string on any error

    def formats(self) -> None:
        """Show supported input formats."""
        transcriber = FileTranscriber()
        formats = transcriber.get_supported_formats()

        self.console.print("[bold blue]Supported Input Formats:[/bold blue]")
        for fmt in formats:
            self.console.print(f"  {fmt}")

        self.console.print("\n[green]Output format:[/green] WebVTT (.vtt)")

    def engines(self) -> None:
        """List available AI engines."""
        self.console.print("[bold blue]Available AI Engines:[/bold blue]")
        for engine in TranscriptionEngine:
            default_model = get_default_model(engine)
            self.console.print(
                f"  [cyan]{engine.value}[/cyan] (default model: {default_model})"
            )

        self.console.print(
            "\n[green]Usage:[/green] vttiro transcribe video.mp4 --engine <engine_name>"
        )

    def models(self, engine: Optional[str] = None, detailed: bool = False) -> None:
        """List available models with capabilities, optionally filtered by engine.

        Args:
            engine: Optional engine name to filter models (gemini, assemblyai, deepgram)
            detailed: Show detailed model capabilities and pricing

        Examples:
            vttiro models                           # List all models with basic info
            vttiro models --engine gemini           # List Gemini models only
            vttiro models --detailed                # Show detailed capabilities
            vttiro models --engine gemini --detailed # Detailed info for Gemini models
        """

        def format_duration(seconds: Optional[int]) -> str:
            """Format duration limit for display."""
            if seconds is None:
                return "unlimited"
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            if hours > 0:
                return f"{hours}h {minutes}m" if minutes > 0 else f"{hours}h"
            return f"{minutes}m"

        def format_cost(cost_per_min: Optional[float]) -> str:
            """Format cost per minute for display."""
            if cost_per_min is None:
                return "varies"
            return f"${cost_per_min:.4f}/min"

        def format_features(capability: ModelCapability) -> str:
            """Format feature list for display."""
            features = []
            if capability.supports_diarization:
                features.append("diarization")
            if capability.supports_timestamps:
                features.append("timestamps")
            if capability.supports_confidence:
                features.append("confidence")
            if capability.supports_emotions:
                features.append("emotions")
            return ", ".join(features) if features else "basic"

        if engine:
            # Validate engine
            try:
                engine_enum = TranscriptionEngine(engine)
                models_list = get_available_models(engine_enum)
                default_model = get_default_model(engine_enum)

                self.console.print(
                    f"[bold blue]Models for {engine.title()}:[/bold blue]"
                )

                for model in models_list:
                    marker = (
                        " [green](default)[/green]" if model == default_model else ""
                    )

                    if detailed:
                        try:
                            cap = get_model_capabilities(model)
                            self.console.print(
                                f"\n  [cyan][bold]{model}[/bold][/cyan]{marker}"
                            )
                            self.console.print(
                                f"    [dim]Accuracy:[/dim] {cap.accuracy_score:.1%}"
                            )
                            self.console.print(
                                f"    [dim]Cost:[/dim] {format_cost(cap.cost_per_minute)}"
                            )
                            self.console.print(
                                f"    [dim]Speed:[/dim] {cap.speed_factor:.1f}x"
                            )
                            self.console.print(
                                f"    [dim]Max duration:[/dim] {format_duration(cap.max_duration_seconds)}"
                            )
                            self.console.print(
                                f"    [dim]Features:[/dim] {format_features(cap)}"
                            )
                            self.console.print(
                                f"    [dim]Languages:[/dim] {len(cap.language_support)} supported"
                            )
                            if cap.recommended_for:
                                use_cases = ", ".join(cap.recommended_for)
                                self.console.print(
                                    f"    [dim]Best for:[/dim] {use_cases}"
                                )
                        except ValueError:
                            self.console.print(
                                f"  [cyan]{model}[/cyan]{marker} [dim](no capability data)[/dim]"
                            )
                    else:
                        try:
                            cap = get_model_capabilities(model)
                            accuracy = f"{cap.accuracy_score:.0%}"
                            cost = format_cost(cap.cost_per_minute)
                            speed = f"{cap.speed_factor:.1f}x"
                            self.console.print(
                                f"  [cyan]{model}[/cyan]{marker} [dim]({accuracy}, {cost}, {speed})[/dim]"
                            )
                        except ValueError:
                            self.console.print(f"  [cyan]{model}[/cyan]{marker}")

            except ValueError:
                self.console.print(f"[red]✗[/red] Invalid engine: {engine}")
                self.console.print(
                    f"Available engines: {', '.join([e.value for e in TranscriptionEngine])}"
                )
                return
        else:
            # List all models grouped by engine
            self.console.print("[bold blue]Available Models by Engine:[/bold blue]")
            for engine_enum in TranscriptionEngine:
                models_list = get_available_models(engine_enum)
                default_model = get_default_model(engine_enum)

                self.console.print(
                    f"\n[yellow][bold]{engine_enum.value.title()}:[/bold][/yellow]"
                )

                for model in models_list:
                    marker = (
                        " [green](default)[/green]" if model == default_model else ""
                    )

                    if detailed:
                        try:
                            cap = get_model_capabilities(model)
                            self.console.print(
                                f"\n  [cyan][bold]{model}[/bold][/cyan]{marker}"
                            )
                            self.console.print(
                                f"    [dim]Accuracy:[/dim] {cap.accuracy_score:.1%}"
                            )
                            self.console.print(
                                f"    [dim]Cost:[/dim] {format_cost(cap.cost_per_minute)}"
                            )
                            self.console.print(
                                f"    [dim]Speed:[/dim] {cap.speed_factor:.1f}x"
                            )
                            self.console.print(
                                f"    [dim]Max duration:[/dim] {format_duration(cap.max_duration_seconds)}"
                            )
                            self.console.print(
                                f"    [dim]Features:[/dim] {format_features(cap)}"
                            )
                            self.console.print(
                                f"    [dim]Languages:[/dim] {len(cap.language_support)} supported"
                            )
                            if cap.recommended_for:
                                use_cases = ", ".join(cap.recommended_for)
                                self.console.print(
                                    f"    [dim]Best for:[/dim] {use_cases}"
                                )
                        except ValueError:
                            self.console.print(
                                f"  [cyan]{model}[/cyan]{marker} [dim](no capability data)[/dim]"
                            )
                    else:
                        try:
                            cap = get_model_capabilities(model)
                            accuracy = f"{cap.accuracy_score:.0%}"
                            cost = format_cost(cap.cost_per_minute)
                            speed = f"{cap.speed_factor:.1f}x"
                            self.console.print(
                                f"  [cyan]{model}[/cyan]{marker} [dim]({accuracy}, {cost}, {speed})[/dim]"
                            )
                        except ValueError:
                            self.console.print(f"  [cyan]{model}[/cyan]{marker}")

        self.console.print(
            "\n[green]Usage:[/green] vttiro transcribe video.mp4 --engine <engine> --model <model>"
        )
        if not detailed:
            self.console.print(
                "[dim]Use --detailed for more information about each model[/dim]"
            )

    def _is_supported_format(self, file_path: Path) -> bool:
        """Check if file format is supported."""
        supported_extensions = {
            # Video formats
            ".mp4",
            ".avi",
            ".mov",
            ".mkv",
            ".webm",
            ".m4v",
            ".3gp",
            ".flv",
            ".wmv",
            # Audio formats
            ".mp3",
            ".wav",
            ".m4a",
            ".aac",
            ".ogg",
            ".flac",
            ".wma",
            ".opus",
        }
        return file_path.suffix.lower() in supported_extensions

    def _check_engine_dependencies(self, engine: str) -> bool:
        """Check if engine dependencies are available."""
        try:
            if engine == "gemini":
                import google.generativeai

                return True
            elif engine == "assemblyai":
                import assemblyai

                return True
            elif engine == "deepgram":
                from deepgram import DeepgramClient

                return True
            else:
                return False
        except ImportError:
            self.console.print(
                f"[red]✗ Missing Dependency:[/red] {engine.title()} SDK not installed"
            )

            install_commands = {
                "gemini": "uv add google-generativeai",
                "assemblyai": "uv add assemblyai",
                "deepgram": "uv add deepgram-sdk",
            }

            cmd = install_commands.get(engine, f"uv add {engine}-sdk")
            self.console.print(f"[yellow]💡 Install with:[/yellow] {cmd}")
            return False

    def help(self) -> None:
        """Show help information."""
        help_text = """
[bold blue]vttiro - Simple Video Transcription Tool[/bold blue]

[bold]Basic Usage:[/bold]
  vttiro transcribe video.mp4                              # Basic transcription (gemini/gemini-2.0-flash)
  vttiro transcribe video.mp4 --output subs.vtt            # Custom output file
  vttiro transcribe video.mp4 --engine assemblyai          # Different AI engine
  vttiro transcribe video.mp4 --engine gemini --model gemini-2.5-pro  # Specific model

[bold]Advanced Options:[/bold]
  --full_prompt TEXT        Replace default prompt entirely (file path or direct text)
  --xtra_prompt TEXT       Append to default/custom prompt (file path or direct text)  
  --add_cues               Include cue identifiers in WebVTT output (default: False)

[bold]Commands:[/bold]
  transcribe          Transcribe audio/video file to WebVTT subtitles
  engines             List available AI engines
  models              List available models with capabilities (all or by engine)
  formats             Show supported input formats
  performance_report  Show comprehensive performance analytics and recommendations
  performance_stats   Show quick performance statistics summary
  config_health       Check configuration health and validation status
  config_debug        Show detailed configuration debugging information
  config_validate     Validate configuration and exit with status code
  config_migrate      Check for and perform configuration migration between versions
  version             Show version information  
  help                Show this help message

[bold]AI Engines:[/bold]
  gemini        Google Gemini (default: gemini-2.0-flash)
  assemblyai    AssemblyAI (default: universal-2)
  deepgram      Deepgram (default: nova-3)

[bold]Environment Variables:[/bold]
  VTTIRO_GEMINI_API_KEY      Google Gemini API key
  VTTIRO_ASSEMBLYAI_API_KEY  AssemblyAI API key  
  VTTIRO_DEEPGRAM_API_KEY    Deepgram API key

[green]Examples:[/green]
  vttiro engines                                   # List AI engines
  vttiro models                                    # List all models with basic capabilities
  vttiro models --detailed                         # Show detailed model capabilities
  vttiro models --engine gemini --detailed         # Detailed info for Gemini models
  vttiro transcribe meeting.mp4                    # Use default (gemini/gemini-2.0-flash)
  vttiro transcribe video.mp4 --engine assemblyai  # Use AssemblyAI
  vttiro transcribe video.mp4 --add_cues          # Include cue identifiers
  vttiro transcribe video.mp4 --xtra_prompt "Focus on technical terms"
  vttiro transcribe video.mp4 --full_prompt "Custom instructions"
  vttiro performance_report --sessions 20         # Show performance for last 20 sessions
  vttiro performance_report --detailed            # Show detailed session breakdown
  vttiro performance_stats                        # Quick performance summary
  vttiro config_health                            # Check configuration health
  vttiro config_health --connectivity             # Include API connectivity tests
  vttiro config_debug                             # Show detailed configuration info
  vttiro config_validate                          # Validate config (exit code 0/1)
  vttiro config_migrate                           # Check and perform configuration migration
        """
        self.console.print(help_text)

    def performance_report(self, sessions: int = 10, detailed: bool = False) -> None:
        """Show comprehensive performance analytics and optimization recommendations.
        
        Args:
            sessions: Number of recent sessions to include (default: 10)
            detailed: Show detailed session breakdown (default: False)
        """
        self.console.print("[bold blue]📊 VTTIRO PERFORMANCE REPORT[/bold blue]\n")
        
        try:
            performance_monitor = get_performance_monitor()
            report = performance_monitor.get_performance_report(last_n_sessions=sessions)
            
            # Summary section
            summary = report['summary']
            self.console.print("[bold]📈 SUMMARY STATISTICS[/bold]")
            self.console.print(f"Total Sessions: [green]{summary['total_sessions']}[/green]")
            self.console.print(f"Success Rate: [green]{summary['success_rate']:.1f}%[/green]")
            self.console.print(f"Average Processing Time: [cyan]{summary['average_processing_time']:.2f}s[/cyan]")
            self.console.print(f"Overall Throughput: [cyan]{summary['overall_throughput_mb_per_sec']:.2f} MB/s[/cyan]")
            self.console.print(f"Total Data Processed: [yellow]{summary['total_data_processed_gb']:.2f} GB[/yellow]\n")
            
            # Operation performance
            if report['operation_performance']:
                self.console.print("[bold]⚙️ OPERATION PERFORMANCE[/bold]")
                for op_type, stats in report['operation_performance'].items():
                    success_color = "green" if stats['success_rate'] >= 95 else "yellow" if stats['success_rate'] >= 90 else "red"
                    self.console.print(
                        f"  {op_type.replace('_', ' ').title()}: "
                        f"[cyan]{stats['average_duration']:.2f}s avg[/cyan] | "
                        f"[{success_color}]{stats['success_rate']:.1f}% success[/{success_color}] | "
                        f"[dim]{stats['count']} runs[/dim]"
                    )
                self.console.print()
            
            # Engine performance
            if report['engine_performance']:
                self.console.print("[bold]🤖 ENGINE PERFORMANCE[/bold]")
                for engine_model, stats in report['engine_performance'].items():
                    success_color = "green" if stats['success_rate'] >= 95 else "yellow" if stats['success_rate'] >= 90 else "red"
                    self.console.print(
                        f"  {engine_model}: "
                        f"[cyan]{stats['average_duration']:.2f}s avg[/cyan] | "
                        f"[{success_color}]{stats['success_rate']:.1f}% success[/{success_color}] | "
                        f"[dim]{stats['count']} runs[/dim]"
                    )
                self.console.print()
            
            # Resource analysis
            if report['resource_analysis']:
                self.console.print("[bold]💾 RESOURCE USAGE[/bold]")
                resources = report['resource_analysis']
                
                if 'memory' in resources:
                    mem = resources['memory']
                    self.console.print(
                        f"  Memory: [cyan]{mem['average_peak_mb']:.1f}MB avg[/cyan] | "
                        f"[yellow]Peak: {mem['max_peak_mb']:.1f}MB[/yellow]"
                    )
                
                if 'cpu' in resources:
                    cpu = resources['cpu']
                    self.console.print(
                        f"  CPU: [cyan]{cpu['average_usage_percent']:.1f}% avg[/cyan] | "
                        f"[yellow]Peak: {cpu['max_usage_percent']:.1f}%[/yellow]"
                    )
                
                if 'file_sizes' in resources:
                    files = resources['file_sizes']
                    self.console.print(
                        f"  Files: [cyan]{files['average_input_size_mb']:.1f}MB avg[/cyan] | "
                        f"[yellow]Largest: {files['largest_file_mb']:.1f}MB[/yellow]"
                    )
                self.console.print()
            
            # Optimization recommendations
            if report['optimization_recommendations']:
                self.console.print("[bold]💡 OPTIMIZATION RECOMMENDATIONS[/bold]")
                for i, rec in enumerate(report['optimization_recommendations'], 1):
                    self.console.print(f"  {i}. {rec}")
                self.console.print()
            
            # Detailed session breakdown
            if detailed and report['recent_sessions']:
                self.console.print(f"[bold]📋 RECENT SESSIONS (Last {len(report['recent_sessions'])})[/bold]")
                for session in report['recent_sessions']:
                    status = "✅" if session['success'] else "❌"
                    self.console.print(
                        f"  {status} {session['input_file']} -> {session['engine']}/{session['model']} "
                        f"| [cyan]{session['duration']:.1f}s[/cyan] | "
                        f"[yellow]{session['input_size_mb']:.1f}MB[/yellow]"
                    )
                    
                    if session['operations']:
                        for op in session['operations']:
                            op_status = "✓" if op['success'] else "✗"
                            self.console.print(
                                f"    {op_status} {op['type']}: [dim]{op['duration']:.2f}s[/dim]"
                            )
                self.console.print()
            
        except Exception as e:
            self.console.print(f"[red]❌ Error generating performance report: {e}[/red]")
            logger.error(f"Performance report error: {e}")

    def performance_stats(self) -> None:
        """Show quick performance statistics summary."""
        self.console.print("[bold blue]⚡ QUICK PERFORMANCE STATS[/bold blue]\n")
        
        try:
            performance_monitor = get_performance_monitor()
            
            # Just log the performance report instead of generating a detailed report
            performance_monitor.log_performance_report(last_n_sessions=5)
            
            self.console.print("📋 [dim]See detailed logs above or use 'vttiro performance_report --detailed' for full analysis[/dim]")
            
        except Exception as e:
            self.console.print(f"[red]❌ Error generating performance stats: {e}[/red]")
            logger.error(f"Performance stats error: {e}")

    def config_health(self, connectivity: bool = False, detailed: bool = False) -> None:
        """Check configuration health and validation status.
        
        Args:
            connectivity: Test API connectivity (slower, requires network)
            detailed: Show detailed validation results
        """
        self.console.print("[bold blue]🔧 CONFIGURATION HEALTH CHECK[/bold blue]\n")
        
        try:
            # Create a FileTranscriber to get initialized config
            transcriber = FileTranscriber()
            validator = ConfigurationValidator(transcriber.config)
            
            with self.console.status("Validating configuration...", spinner="dots"):
                health = validator.validate_all(check_connectivity=connectivity)
            
            # Overall status with color coding
            status_colors = {
                "healthy": "green",
                "warnings": "yellow", 
                "errors": "red",
                "critical": "red"
            }
            
            status_icons = {
                "healthy": "✅",
                "warnings": "⚠️",
                "errors": "❌",
                "critical": "🚨"
            }
            
            status_color = status_colors.get(health.overall_status, "white")
            status_icon = status_icons.get(health.overall_status, "❓")
            
            self.console.print(f"[bold]Overall Status: [{status_color}]{status_icon} {health.overall_status.upper()}[/{status_color}][/bold]")
            self.console.print(f"Success Rate: [cyan]{health.success_rate:.1f}%[/cyan] ({health.passed_checks}/{health.total_checks} checks passed)\n")
            
            # Summary by severity
            if health.critical_count > 0:
                self.console.print(f"[red]🚨 Critical Issues: {health.critical_count}[/red]")
            if health.error_count > 0:
                self.console.print(f"[red]❌ Errors: {health.error_count}[/red]")
            if health.warning_count > 0:
                self.console.print(f"[yellow]⚠️ Warnings: {health.warning_count}[/yellow]")
            
            info_count = health.total_checks - health.warning_count - health.error_count - health.critical_count
            if info_count > 0:
                self.console.print(f"[green]✅ Passed: {info_count}[/green]")
            self.console.print()
            
            # Show critical and error issues always
            critical_and_errors = [
                r for r in health.results 
                if r.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR] and not r.status
            ]
            
            if critical_and_errors:
                self.console.print("[bold red]🚨 CRITICAL ISSUES & ERRORS[/bold red]")
                for result in critical_and_errors:
                    self.console.print(f"  ❌ {result.message}")
                    if result.details:
                        self.console.print(f"     [dim]{result.details}[/dim]")
                    if result.recommendation:
                        self.console.print(f"     💡 [yellow]{result.recommendation}[/yellow]")
                self.console.print()
            
            # Show warnings if detailed or if there are no critical/error issues
            warnings = [r for r in health.results if r.severity == ValidationSeverity.WARNING and not r.status]
            if warnings and (detailed or not critical_and_errors):
                self.console.print("[bold yellow]⚠️ WARNINGS[/bold yellow]")
                for result in warnings:
                    self.console.print(f"  ⚠️ {result.message}")
                    if result.recommendation and detailed:
                        self.console.print(f"     💡 [dim]{result.recommendation}[/dim]")
                self.console.print()
            
            # Show successful checks if detailed
            if detailed:
                passed_checks = [r for r in health.results if r.status]
                if passed_checks:
                    self.console.print("[bold green]✅ PASSED CHECKS[/bold green]")
                    for result in passed_checks:
                        self.console.print(f"  ✅ {result.message}")
                    self.console.print()
            
            # Recommendations
            if health.recommendations:
                self.console.print("[bold]💡 RECOMMENDATIONS[/bold]")
                for i, rec in enumerate(health.recommendations[:5], 1):  # Top 5
                    self.console.print(f"  {i}. {rec}")
                self.console.print()
            
            # Quick action suggestions
            if health.overall_status == "critical":
                self.console.print("[bold red]🚨 IMMEDIATE ACTION REQUIRED[/bold red]")
                self.console.print("Critical issues detected. Please resolve these before using vttiro.")
            elif health.overall_status == "errors":
                self.console.print("[bold red]⚠️ ACTION RECOMMENDED[/bold red]")
                self.console.print("Errors detected that may cause transcription failures.")
            elif health.overall_status == "warnings":
                self.console.print("[bold yellow]💡 OPTIMIZATION AVAILABLE[/bold yellow]")
                self.console.print("Minor issues detected. System functional but could be improved.")
            else:
                self.console.print("[bold green]🎉 EXCELLENT CONFIGURATION[/bold green]")
                self.console.print("Your configuration looks great! Ready for transcription.")
                
        except Exception as e:
            self.console.print(f"[red]❌ Error during configuration health check: {e}[/red]")
            logger.error(f"Config health check error: {e}")

    def config_debug(self) -> None:
        """Show detailed configuration debugging information."""
        self.console.print("[bold blue]🔍 CONFIGURATION DEBUG INFO[/bold blue]\n")
        
        try:
            # Create a FileTranscriber to get initialized config
            transcriber = FileTranscriber()
            validator = ConfigurationValidator(transcriber.config)
            
            # Run validation to populate results
            validator.validate_all(check_connectivity=False)
            debug_info = validator.get_debug_info()
            
            # Configuration object info
            self.console.print("[bold]📋 CONFIGURATION OBJECT[/bold]")
            config_obj = debug_info["config_object"]
            self.console.print(f"Type: [cyan]{config_obj['type']}[/cyan]")
            self.console.print("Attributes:")
            for key, value in config_obj["attributes"].items():
                if "api_key" in key.lower():
                    display_value = "***REDACTED***" if value else "[dim]None[/dim]"
                else:
                    display_value = value
                self.console.print(f"  {key}: [yellow]{display_value}[/yellow]")
            self.console.print()
            
            # Environment variables
            env_vars = debug_info["environment_variables"]
            self.console.print("[bold]🌍 ENVIRONMENT VARIABLES[/bold]")
            if env_vars:
                for key, value in env_vars.items():
                    self.console.print(f"  {key}: [yellow]{value}[/yellow]")
            else:
                self.console.print("  [dim]No VTTIRO_* environment variables set[/dim]")
            self.console.print()
            
            # System info
            system = debug_info["system_info"]
            self.console.print("[bold]💻 SYSTEM INFORMATION[/bold]")
            self.console.print(f"Python: [cyan]{system['python_version'].split()[0]}[/cyan]")
            self.console.print(f"Platform: [cyan]{system['platform']}[/cyan]")
            self.console.print(f"Working Directory: [cyan]{system['working_directory']}[/cyan]")
            self.console.print(f"User: [cyan]{system['user']}[/cyan]")
            self.console.print()
            
            # Validation summary
            validation = debug_info["validation_summary"]
            self.console.print("[bold]📊 VALIDATION SUMMARY[/bold]")
            self.console.print(f"Total Checks: [cyan]{validation['total_results']}[/cyan]")
            for severity, count in validation["by_severity"].items():
                if count > 0:
                    color = {
                        "info": "green",
                        "warning": "yellow", 
                        "error": "red",
                        "critical": "red"
                    }.get(severity, "white")
                    self.console.print(f"  {severity.title()}: [{color}]{count}[/{color}]")
            self.console.print()
            
            self.console.print("[dim]💡 Use 'vttiro config_health --detailed' for complete validation results[/dim]")
            
        except Exception as e:
            self.console.print(f"[red]❌ Error generating configuration debug info: {e}[/red]")
            logger.error(f"Config debug error: {e}")

    def config_validate(self) -> None:
        """Validate configuration and exit with appropriate code."""
        self.console.print("[bold blue]🔧 CONFIGURATION VALIDATION[/bold blue]\n")
        
        try:
            # Create a FileTranscriber to get initialized config
            transcriber = FileTranscriber()
            
            with self.console.status("Validating configuration...", spinner="dots"):
                is_valid, error_messages = validate_startup_configuration(transcriber.config)
            
            if is_valid:
                self.console.print("[green]✅ Configuration is valid and ready for use[/green]")
                self.console.print("All critical checks passed. Transcription should work correctly.")
                sys.exit(0)
            else:
                self.console.print("[red]❌ Configuration validation failed[/red]")
                self.console.print("\n[bold red]Critical Issues:[/bold red]")
                for i, message in enumerate(error_messages, 1):
                    self.console.print(f"  {i}. {message}")
                
                self.console.print(f"\n[dim]💡 Use 'vttiro config_health --detailed' for complete analysis and recommendations[/dim]")
                sys.exit(1)
                
        except Exception as e:
            self.console.print(f"[red]❌ Error during configuration validation: {e}[/red]")
            logger.error(f"Config validation error: {e}")
            sys.exit(1)

    def config_migrate(self, check_only: bool = False) -> None:
        """Check for and perform configuration migration between vttiro versions.
        
        Args:
            check_only: Only check if migration is needed, don't perform it
        """
        self.console.print("[bold blue]🔄 CONFIGURATION MIGRATION[/bold blue]\n")
        
        try:
            migrator = ConfigurationMigrator()
            
            with self.console.status("Checking migration status...", spinner="dots"):
                migration_report = migrator.create_migration_report()
            
            # Display current status
            self.console.print(f"[bold]Current Version: [cyan]{migration_report['current_version']}[/cyan][/bold]")
            
            # Check if any migration is needed
            needs_migration = migration_report.get('migration_available', False) or migration_report.get('environment_migration_needed', False)
            
            if not needs_migration:
                self.console.print("[green]✅ No migration needed - configuration is up to date[/green]")
                self.console.print("Your configuration is already using the latest version and standards.")
                return
            
            # Display what needs migration
            self.console.print("[yellow]🔄 Migration Available[/yellow]\n")
            
            if migration_report.get('environment_migration_needed', False):
                self.console.print("[bold]🌍 ENVIRONMENT VARIABLES[/bold]")
                for action in migration_report.get('environment_actions', []):
                    self.console.print(f"  • {action}")
                self.console.print()
            
            if migration_report.get('migration_available', False):
                self.console.print("[bold]📋 CONFIGURATION FILE[/bold]")
                self.console.print(f"Current version: [yellow]{migration_report.get('config_current_version', 'unknown')}[/yellow]")
                self.console.print(f"Target version: [green]{migration_report['current_version']}[/green]")
                
                if 'migration_path' in migration_report:
                    self.console.print("\nMigration steps:")
                    for step in migration_report['migration_path']:
                        required = "Required" if step['required'] else "Optional"
                        self.console.print(f"  • {step['from']} → {step['to']}: {step['description']} ({required})")
                self.console.print()
            
            # Display recommendations
            if migration_report.get('recommendations'):
                self.console.print("[bold]💡 RECOMMENDATIONS[/bold]")
                for i, rec in enumerate(migration_report['recommendations'], 1):
                    self.console.print(f"  {i}. {rec}")
                self.console.print()
            
            if check_only:
                self.console.print("[dim]💡 Use 'vttiro config_migrate' (without --check_only) to perform migration[/dim]")
            else:
                self.console.print("[bold yellow]⚠️ MIGRATION INSTRUCTIONS[/bold yellow]")
                self.console.print("Migration is available but must be performed manually:")
                self.console.print()
                
                if migration_report.get('environment_migration_needed', False):
                    self.console.print("1. [bold]Update Environment Variables[/bold]")
                    self.console.print("   Update your shell profile (.bashrc, .zshrc, etc.) with:")
                    for action in migration_report.get('environment_actions', []):
                        if "recommend renaming" in action:
                            parts = action.split()
                            old_var = parts[4]  # Extract old variable name
                            new_var = parts[8]  # Extract new variable name
                            self.console.print(f"   [dim]# Replace {old_var} with {new_var}[/dim]")
                            self.console.print(f"   export {new_var}=\"${{old_var}}\"")
                            self.console.print(f"   unset {old_var}")
                    self.console.print()
                
                self.console.print("2. [bold]Restart Terminal[/bold]")
                self.console.print("   Reload your shell environment after making changes")
                self.console.print()
                
                self.console.print("3. [bold]Verify Configuration[/bold]")
                self.console.print("   Run: [cyan]vttiro config_health[/cyan]")
                
                self.console.print(f"\n[green]ℹ️ Configuration migration guidance completed[/green]")
            
        except Exception as e:
            self.console.print(f"[red]❌ Error during configuration migration: {e}[/red]")
            logger.error(f"Config migration error: {e}")


def main() -> None:
    """Main entry point for vttiro CLI."""
    try:
        fire.Fire(VttiroCLI)
    except KeyboardInterrupt:
        print("\n🔸 Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
