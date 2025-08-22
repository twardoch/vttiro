# this_file: src/vttiro/core/transcriber.py
"""Simplified transcriber for VTTiro - focused on core transcription functionality.

This module provides a clean, simple interface for transcription without
over-engineered resilience patterns, circuit breakers, or excessive validation.

Key responsibilities:
- Audio file transcription using AI providers
- Basic error handling and retries
- VTT output generation

Used by:
- CLI interface for command-line transcription
- Direct API usage for programmatic access
"""

import asyncio
from pathlib import Path
from typing import Any, Optional
import time

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from .config import VttiroConfig
from .types import TranscriptionResult, TranscriptSegment
from .errors import VttiroError, ProviderError, TranscriptionError


class Transcriber:
    """Simple transcriber for VTTiro.
    
    Provides a clean interface for transcription without excessive complexity.
    Focuses on the core function: converting audio to WebVTT subtitles.
    """
    
    def __init__(self, config: Optional[VttiroConfig] = None):
        """Initialize transcriber with configuration.
        
        Args:
            config: VttiroConfig instance, uses defaults if None
        """
        self.config = config or VttiroConfig()
    
    async def transcribe(
        self, 
        audio_path: Path,
        output_path: Optional[Path] = None,
        **kwargs: Any
    ) -> TranscriptionResult:
        """Transcribe audio file to text with timing.
        
        Simple transcription method:
        1. Validate input file exists
        2. Select provider 
        3. Transcribe with basic retry
        4. Generate VTT output if requested
        
        Args:
            audio_path: Path to input audio/video file
            output_path: Optional output path for VTT file
            **kwargs: Additional parameters for transcription
            
        Returns:
            TranscriptionResult with segments and metadata
            
        Raises:
            VttiroError: Transcription failed
            FileNotFoundError: Input file not found
        """
        start_time = time.time()
        
        # Basic validation
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        provider = self.config.engine or self.config.provider or "gemini"
        logger.info(f"Transcribing {audio_path} using {provider}")
        
        try:
            # Perform transcription with simple retry
            result = await self._transcribe_with_retry(audio_path, provider, **kwargs)
            
            # Generate VTT output if requested
            if output_path:
                await self._generate_vtt(result, output_path)
            
            # Add basic metadata
            duration = time.time() - start_time
            result.metadata.update({
                "transcription_duration": duration,
                "provider": provider,
                "audio_path": str(audio_path)
            })
            
            return result
            
        except Exception as e:
            if isinstance(e, VttiroError):
                raise
            
            # Convert to VttiroError with context
            error = TranscriptionError(
                f"Transcription failed with {provider}: {str(e)}",
                provider=provider,
                details={"audio_path": str(audio_path), "error": str(e)}
            )
            raise error
    
    async def _transcribe_with_retry(
        self, 
        audio_path: Path, 
        provider: str, 
        max_retries: int = 3,
        **kwargs
    ) -> TranscriptionResult:
        """Transcribe with simple exponential backoff retry."""
        
        last_error = None
        
        for attempt in range(max_retries):
            try:
                return await self._call_provider(audio_path, provider, **kwargs)
                
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    delay = 2 ** attempt  # Simple exponential backoff
                    logger.warning(f"Transcription attempt {attempt + 1} failed, retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {max_retries} transcription attempts failed")
        
        raise last_error or TranscriptionError(f"Transcription failed after {max_retries} attempts")
    
    async def _call_provider(self, audio_path: Path, provider: str, **kwargs) -> TranscriptionResult:
        """Call the specified transcription provider."""
        
        # Import provider dynamically to avoid circular imports
        if provider == "gemini":
            from ..providers.gemini.transcriber import GeminiTranscriber
            provider_instance = GeminiTranscriber(self.config)
        elif provider == "openai":
            from ..providers.openai.transcriber import OpenAITranscriber
            provider_instance = OpenAITranscriber(self.config)
        elif provider == "assemblyai":
            from ..providers.assemblyai.transcriber import AssemblyAITranscriber
            provider_instance = AssemblyAITranscriber(self.config)
        elif provider == "deepgram":
            from ..providers.deepgram.transcriber import DeepgramTranscriber
            provider_instance = DeepgramTranscriber(self.config)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        # Call the provider's transcribe method
        return await provider_instance.transcribe(audio_path, **kwargs)
    
    async def _generate_vtt(self, result: TranscriptionResult, output_path: Path) -> None:
        """Generate VTT file from transcription result."""
        
        try:
            from ..output.enhanced_webvtt import EnhancedWebVTTFormatter
            formatter = EnhancedWebVTTFormatter()
            vtt_content = formatter.format(result)
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write VTT file
            output_path.write_text(vtt_content, encoding='utf-8')
            logger.info(f"VTT file written to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate VTT file: {e}")
            raise
    
    def get_supported_providers(self) -> list[str]:
        """Get list of available transcription providers."""
        return ["gemini", "openai", "assemblyai", "deepgram"]
    
    def validate_config(self) -> dict[str, Any]:
        """Basic configuration validation."""
        issues = []
        warnings = []
        
        # Check provider
        provider = self.config.engine or self.config.provider
        if provider and provider not in self.get_supported_providers():
            issues.append(f"Unsupported provider: {provider}")
        
        # Check timeout
        if self.config.timeout_seconds < 10:
            warnings.append("Very short timeout may cause failures")
        elif self.config.timeout_seconds > 1800:
            warnings.append("Very long timeout - consider reducing")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "config": self.config.to_dict()
        }