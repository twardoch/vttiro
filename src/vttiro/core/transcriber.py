# this_file: src/vttiro/core/transcriber.py
"""Main transcriber facade for VTTiro 2.0.

This module provides the primary user-facing interface for transcription,
implementing a simple facade pattern that coordinates between providers,
processing modules, and output generation.

Key responsibilities:
- Provider selection and instantiation
- Audio preprocessing coordination  
- Transcription orchestration
- Output format generation
- Error handling and fallbacks

Used by:
- CLI interface for command-line transcription
- API consumers for programmatic access
- Testing infrastructure for integration tests
"""

import asyncio
from pathlib import Path
from typing import Any

from .config import VttiroConfig
from .types import TranscriptionResult
from .resilience import (
    get_resilience_manager, with_retry, timeout_for,
    RetryConfig, CircuitBreakerConfig, TimeoutConfig
)
from .errors import (
    VttiroError, ProviderError, TranscriptionError,
    TimeoutError, handle_provider_exception
)
from ..utils.input_validation import InputValidator, ProviderInputSanitizer


class Transcriber:
    """Main transcriber facade for VTTiro 2.0.
    
    Provides a simplified interface for transcription that coordinates
    between multiple subsystems. Handles provider selection, audio processing,
    and output generation automatically based on configuration.
    
    This is the primary entry point for both CLI and programmatic usage,
    offering a clean abstraction over the underlying complexity.
    
    Example usage:
        config = VttiroConfig(provider="gemini", language="en")
        transcriber = Transcriber(config)
        result = await transcriber.transcribe(Path("audio.mp3"))
    """
    
    def __init__(self, config: VttiroConfig | None = None):
        """Initialize transcriber with configuration.
        
        Args:
            config: VttiroConfig instance, uses defaults if None
        """
        self.config = config or VttiroConfig()
        
        # Initialize resilience manager with config-based settings
        self.resilience = get_resilience_manager()
        
        # Configure retry behavior based on config
        retry_config = RetryConfig(
            max_attempts=self.config.max_retries,
            base_delay=1.0,
            max_delay=min(60.0, self.config.timeout_seconds / 4),  # Don't wait longer than 1/4 of total timeout
        )
        
        # Configure timeout behavior
        timeout_config = TimeoutConfig(
            total_timeout=self.config.timeout_seconds,
            transcription_timeout=self.config.timeout_seconds,
            provider_timeouts={
                'gemini': min(300.0, self.config.timeout_seconds),
                'openai': min(180.0, self.config.timeout_seconds),
                'assemblyai': min(600.0, self.config.timeout_seconds),
                'deepgram': min(300.0, self.config.timeout_seconds),
            }
        )
        
        # Configure circuit breaker for each provider
        circuit_config = CircuitBreakerConfig(
            failure_threshold=3,  # Open circuit after 3 failures
            timeout_duration=60.0,  # Wait 1 minute before trying again
            success_threshold=2,  # Need 2 successes to close circuit
        )
        
        # Update resilience manager configuration
        from .resilience import configure_resilience
        configure_resilience(retry_config, circuit_config, timeout_config)
    
    async def transcribe(
        self, 
        audio_path: Path,
        output_path: Path | None = None,
        **kwargs: Any
    ) -> TranscriptionResult:
        """Transcribe audio file to text with timing using resilient patterns.
        
        Main transcription method that orchestrates the complete pipeline:
        1. Validate input file with comprehensive error handling
        2. Preprocess audio if needed with timeout management
        3. Select and configure provider with circuit breaker protection
        4. Perform transcription with retry logic and exponential backoff
        5. Generate output file if requested with proper cleanup
        
        Args:
            audio_path: Path to input audio/video file
            output_path: Optional output path, overrides config setting
            **kwargs: Override configuration parameters for this transcription
            
        Returns:
            TranscriptionResult with segments and metadata
            
        Raises:
            VttiroError: Comprehensive error with context and suggestions
            TimeoutError: Operation exceeded configured timeout limits
            ProviderError: Provider-specific errors with fallback suggestions
        """
        operation_start_time = asyncio.get_event_loop().time()
        
        try:
            # Use provider from config (engine takes precedence over provider)
            current_provider = self.config.engine or self.config.provider or "gemini"
            
            # Create debug context for error reporting
            from .errors import create_debug_context
            debug_context = create_debug_context(
                operation="transcribe",
                provider=current_provider,
                file_path=str(audio_path),
                config=self.config.to_dict()
            )
            
            # Step 1: Comprehensive input validation with resilience
            validation_result = await self._validate_inputs_with_resilience(
                audio_path, output_path, current_provider, **kwargs
            )
            sanitized_kwargs = validation_result['sanitized_kwargs']
            
            # Step 2: Audio preprocessing with timeout management
            processed_audio_path = await self._preprocess_audio_with_resilience(
                audio_path, current_provider
            )
            
            # Step 3: Core transcription with retry logic and circuit breaker
            transcription_result = await self._perform_transcription_with_resilience(
                processed_audio_path, current_provider, sanitized_kwargs, debug_context
            )
            
            # Step 4: Output generation with proper error handling
            if output_path:
                await self._generate_output_with_resilience(
                    transcription_result, output_path
                )
            
            # Step 5: Add resilience metrics to metadata
            operation_duration = asyncio.get_event_loop().time() - operation_start_time
            health_status = self.resilience.get_health_status()
            
            transcription_result.metadata.update({
                "operation_duration": operation_duration,
                "resilience_status": health_status['status'],
                "retry_attempts": 1,  # Will be updated if retries occurred
                "circuit_breaker_state": health_status['circuit_breakers'].get(current_provider, {}).get('state', 'closed')
            })
            
            return transcription_result
            
        except Exception as e:
            # Convert exceptions to typed VTTiro errors with comprehensive context
            operation_duration = asyncio.get_event_loop().time() - operation_start_time
            
            if isinstance(e, VttiroError):
                # Already a VTTiro error, add duration context
                e.details = e.details or {}
                e.details.update({
                    "operation_duration": operation_duration,
                    "audio_path": str(audio_path)
                })
                raise
            else:
                # Convert generic exception to typed error
                provider = getattr(self.config, 'engine', None) or getattr(self.config, 'provider', 'unknown')
                vttiro_error = handle_provider_exception(e, provider)
                vttiro_error.details = vttiro_error.details or {}
                vttiro_error.details.update({
                    "operation_duration": operation_duration,
                    "audio_path": str(audio_path),
                    "original_exception_type": type(e).__name__
                })
                raise vttiro_error
    
    async def _validate_inputs_with_resilience(
        self, 
        audio_path: Path, 
        output_path: Path | None, 
        provider: str,
        **kwargs
    ) -> dict[str, Any]:
        """Validate inputs with resilience patterns."""
        
        @with_retry("input_validation")
        async def validate_inputs():
            with timeout_for("file_read"):
                validator = InputValidator()
                provider_sanitizer = ProviderInputSanitizer(validator)
                
                # Validate file path
                file_result = validator.validate_file_path(audio_path, provider)
                if not file_result.is_valid:
                    raise ValueError(f"File validation failed: {file_result.error_message}")
                
                # Validate provider
                provider_result = validator.validate_provider_name(provider)
                if not provider_result.is_valid:
                    raise ValueError(f"Provider validation failed: {provider_result.error_message}")
                
                # Validate language if specified
                if self.config.language:
                    lang_result = validator.validate_language_code(self.config.language)
                    if not lang_result.is_valid:
                        raise ValueError(f"Language validation failed: {lang_result.error_message}")
                
                # Validate output path if specified
                if output_path:
                    output_result = validator.validate_output_path(output_path)
                    if not output_result.is_valid:
                        raise ValueError(f"Output path validation failed: {output_result.error_message}")
                
                # Provider-specific sanitization
                inputs_dict = {"file_path": audio_path, "language": self.config.language, **kwargs}
                is_valid, sanitized_inputs, warnings = provider_sanitizer.sanitize_for_provider(provider, inputs_dict)
                
                if not is_valid:
                    raise ValueError("Provider-specific validation failed")
                
                return {
                    "sanitized_kwargs": {k: v for k, v in sanitized_inputs.items() if k not in ["file_path", "language"]},
                    "warnings": warnings
                }
        
        return await validate_inputs()
    
    async def _preprocess_audio_with_resilience(self, audio_path: Path, provider: str) -> Path:
        """Preprocess audio with timeout and error handling."""
        
        @with_retry("audio_preprocessing", f"{provider}_preprocessing")
        async def preprocess_audio():
            with timeout_for("audio_processing", provider):
                # For now, return the original path (preprocessing will be implemented later)
                # This is where we would add audio format conversion, normalization, etc.
                if not audio_path.exists():
                    raise FileNotFoundError(f"Audio file not found: {audio_path}")
                
                # Simulate async processing
                await asyncio.sleep(0.1)
                
                return audio_path
        
        return await preprocess_audio()
    
    async def _perform_transcription_with_resilience(
        self, 
        audio_path: Path, 
        provider: str, 
        kwargs: dict[str, Any],
        debug_context: dict[str, Any]
    ) -> TranscriptionResult:
        """Perform transcription with comprehensive resilience patterns."""
        
        @with_retry("transcription", f"{provider}_transcription")
        async def transcribe_with_provider():
            with timeout_for("transcription", provider):
                # This is where the actual provider implementation would be called
                # For now, create a mock result with realistic error simulation
                
                # Simulate potential network/provider issues for testing resilience
                import random
                if random.random() < 0.1:  # 10% chance of simulated failure
                    if random.random() < 0.5:
                        raise TimeoutError("Simulated provider timeout", timeout_duration=30.0)
                    else:
                        from .errors import APIError
                        raise APIError("Simulated API error", provider, status_code=503)
                
                # Simulate processing delay
                await asyncio.sleep(0.2)
                
                # Create mock transcription result
                from .types import TranscriptSegment
                
                segments = [
                    TranscriptSegment(
                        start=0.0,
                        end=5.0,
                        text="This is a resilient transcription result with comprehensive error handling.",
                        confidence=0.95
                    )
                ]
                
                result = TranscriptionResult(
                    segments=segments,
                    metadata={
                        "provider": provider,
                        "audio_path": str(audio_path),
                        "config": self.config.to_dict(),
                        "debug_context": debug_context,
                        "resilience_enabled": True
                    },
                    provider=provider,
                    language=self.config.language,
                    confidence=0.95
                )
                
                return result
        
        return await transcribe_with_provider()
    
    async def _generate_output_with_resilience(
        self, 
        transcription_result: TranscriptionResult, 
        output_path: Path
    ) -> None:
        """Generate output file with resilience patterns."""
        
        @with_retry("output_generation")
        async def generate_output():
            with timeout_for("file_write"):
                # This is where output generation would be implemented
                # For now, simulate writing
                await asyncio.sleep(0.1)
                
                # Ensure parent directory exists
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Mock output generation
                output_path.write_text(f"Mock VTT output for {len(transcription_result.segments)} segments")
        
        await generate_output()
    
    def estimate_cost(self, audio_path: Path) -> float:
        """Estimate transcription cost for audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Estimated cost in USD
            
        Raises:
            FileNotFoundError: Audio file not found
            ValueError: Cannot determine audio duration
        """
        # Placeholder implementation
        # Will be replaced with actual duration calculation and provider cost estimation
        return 0.05  # Mock cost
    
    def get_supported_providers(self) -> list[str]:
        """Get list of available transcription providers.
        
        Returns:
            List of provider names
        """
        return ["gemini", "openai", "assemblyai", "deepgram"]
    
    def validate_config(self) -> dict[str, Any]:
        """Validate current configuration using comprehensive validation.
        
        Returns:
            Validation report with any issues found and resilience status
        """
        issues = []
        warnings = []
        validator = InputValidator()
        
        # Use engine, fallback to provider for backward compatibility
        current_engine = self.config.engine or self.config.provider or "gemini"
        
        # Comprehensive provider validation
        provider_result = validator.validate_provider_name(current_engine)
        if not provider_result.is_valid:
            issues.append(provider_result.error_message)
        
        # Validate language if specified
        if self.config.language:
            lang_result = validator.validate_language_code(self.config.language)
            if not lang_result.is_valid:
                issues.append(lang_result.error_message)
        
        # Validate output path if specified
        if self.config.output_path:
            output_result = validator.validate_output_path(self.config.output_path)
            if not output_result.is_valid:
                issues.append(output_result.error_message)
        
        # Validate numeric parameters
        for param_name, param_value in [
            ("max_segment_duration", getattr(self.config, "max_segment_duration", None)),
            ("timeout_seconds", self.config.timeout_seconds),
            ("max_retries", self.config.max_retries),
        ]:
            if param_value is not None:
                numeric_result = validator.validate_numeric_parameter(param_value, param_name, min_value=0.0)
                if not numeric_result.is_valid:
                    issues.append(f"{param_name}: {numeric_result.error_message}")
        
        # Validate resilience configuration
        if self.config.max_retries > 10:
            warnings.append("Very high retry count may cause long delays on failures")
        
        if self.config.timeout_seconds < 30:
            warnings.append("Short timeout may cause premature failures for large files")
        elif self.config.timeout_seconds > 1800:  # 30 minutes
            warnings.append("Very long timeout may mask underlying issues")
        
        # Get resilience health status
        resilience_health = self.resilience.get_health_status()
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "config": self.config.to_dict(),
            "resilience_status": {
                "health": resilience_health['status'],
                "circuit_breakers": resilience_health['circuit_breakers'],
                "retry_configuration": {
                    "max_attempts": self.config.max_retries,
                    "timeout_seconds": self.config.timeout_seconds
                }
            }
        }
    
    def get_health_status(self) -> dict[str, Any]:
        """Get comprehensive health status including resilience metrics.
        
        Returns:
            Detailed health report for monitoring and debugging
        """
        # Get resilience health
        resilience_health = self.resilience.get_health_status()
        
        # Current provider status
        current_provider = self.config.engine or self.config.provider or "gemini"
        provider_circuit = resilience_health['circuit_breakers'].get(current_provider, {})
        
        # Configuration health
        config_validation = self.validate_config()
        
        return {
            "overall_status": "healthy" if config_validation['valid'] and resilience_health['status'] == 'healthy' else "degraded",
            "configuration": {
                "valid": config_validation['valid'],
                "issues": config_validation.get('issues', []),
                "warnings": config_validation.get('warnings', [])
            },
            "resilience": resilience_health,
            "current_provider": {
                "name": current_provider,
                "circuit_state": provider_circuit.get('state', 'closed'),
                "failure_count": provider_circuit.get('failure_count', 0),
                "failure_rate": provider_circuit.get('failure_rate', 0.0)
            },
            "performance_metrics": resilience_health.get('metrics_summary', {}),
            "recommendations": self._generate_health_recommendations(config_validation, resilience_health)
        }
    
    def _generate_health_recommendations(self, config_validation: dict, resilience_health: dict) -> list[str]:
        """Generate health recommendations based on current status."""
        recommendations = []
        
        # Configuration recommendations
        if config_validation.get('issues'):
            recommendations.append("Fix configuration issues before production use")
        
        if config_validation.get('warnings'):
            recommendations.append("Review configuration warnings for optimal performance")
        
        # Resilience recommendations
        if resilience_health['status'] == 'degraded':
            recommendations.append("Some providers are experiencing issues - consider fallback configuration")
        
        # Circuit breaker recommendations
        open_circuits = [name for name, status in resilience_health['circuit_breakers'].items() 
                        if status.get('state') == 'open']
        if open_circuits:
            recommendations.append(f"Circuit breakers open for: {', '.join(open_circuits)} - check provider status")
        
        # Performance recommendations
        metrics = resilience_health.get('metrics_summary', {})
        if metrics.get('retry_operations', 0) > 0:
            recommendations.append("High retry activity detected - investigate underlying issues")
        
        if not recommendations:
            recommendations.append("System is healthy - no action required")
        
        return recommendations