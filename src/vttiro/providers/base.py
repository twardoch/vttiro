# this_file: src/vttiro/providers/base.py
"""Abstract base class for transcription providers.

This module defines the contract that all transcription providers must implement.
It ensures consistent interfaces across different AI services (Gemini, OpenAI, etc.)
while allowing for provider-specific optimizations and features.

The abstract base class enforces:
- Consistent transcription method signatures
- Cost estimation capabilities  
- Error handling patterns
- Metadata requirements

Used by:
- All provider implementations (gemini/, openai/, etc.)
- Core orchestration for provider selection and switching
- Testing infrastructure for contract validation
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..core.types import TranscriptionResult
from ..utils.input_validation import InputValidator


class TranscriberABC(ABC):
    """Abstract base class for all transcription providers.
    
    Defines the minimal contract that every transcription provider must implement.
    Providers may extend this with additional methods, but must implement all
    abstract methods to ensure compatibility with the core orchestration system.
    
    Lifecycle:
    1. Provider instantiated with credentials/config
    2. estimate_cost() called for budget planning (optional)
    3. transcribe() called with audio file and parameters
    4. Provider returns TranscriptionResult with segments and metadata
    
    Error Handling:
    - Providers should raise appropriate exceptions for API failures
    - Network errors, authentication failures, and quota limits should be handled gracefully
    - Providers may implement retry logic but should respect timeout constraints
    """
    
    @abstractmethod
    async def transcribe(
        self,
        audio_path: Path,
        language: str | None = None,
        context: str | None = None,
        **kwargs: Any
    ) -> TranscriptionResult:
        """Transcribe audio file to text with timing information.
        
        Core transcription method that all providers must implement.
        Takes an audio file and returns structured transcription results
        with precise timing, optional speaker identification, and confidence scores.
        
        Args:
            audio_path: Path to audio file (WAV, MP3, M4A, etc.)
            language: Language code (ISO 639-1, e.g., 'en'), None for auto-detect
            context: Additional context to improve transcription accuracy
                    (e.g., technical terms, speaker names, domain knowledge)
            **kwargs: Provider-specific parameters (model selection, quality settings, etc.)
            
        Returns:
            TranscriptionResult with segments, metadata, and provider information
            
        Raises:
            FileNotFoundError: Audio file does not exist or is not accessible
            ValueError: Invalid audio format or unsupported language
            RuntimeError: API errors, network failures, or service unavailable
            PermissionError: Authentication failures or quota exceeded
            
        Provider Implementation Notes:
        - Should validate audio file format and accessibility
        - May perform audio preprocessing (format conversion, normalization)
        - Should populate metadata with provider-specific information
        - Must return segments in chronological order
        - Should include confidence scores when available
        """
        pass
    
    @abstractmethod
    def estimate_cost(self, duration_seconds: float) -> float:
        """Estimate transcription cost in USD.
        
        Provides cost estimation before transcription to help with budget planning
        and provider selection. Should be based on current provider pricing
        and any applicable discounts or rate limits.
        
        Args:
            duration_seconds: Audio duration in seconds
            
        Returns:
            Estimated cost in USD (may be 0.0 for free tiers)
            
        Raises:
            ValueError: Invalid duration (negative or zero)
            
        Provider Implementation Notes:
        - Should account for current pricing tiers and rate limits
        - May include processing overhead in calculations
        - Should return 0.0 for free services or free tier usage
        - Estimates may vary from actual costs due to pricing changes
        """
        pass
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name for identification and logging.
        
        Returns:
            Provider name (e.g., "gemini", "openai", "assemblyai", "deepgram")
        """
        pass
    
    @property
    def supports_speaker_diarization(self) -> bool:
        """Whether this provider supports speaker diarization.
        
        Returns:
            True if provider can identify and separate speakers
        """
        return False
    
    @property 
    def supports_streaming(self) -> bool:
        """Whether this provider supports real-time streaming transcription.
        
        Returns:
            True if provider can transcribe audio streams in real-time
        """
        return False
    
    @property
    def supported_languages(self) -> list[str]:
        """List of supported language codes.
        
        Returns:
            List of ISO 639-1 language codes (e.g., ['en', 'es', 'fr'])
            Empty list indicates no specific language restrictions
        """
        return []
    
    def validate_audio_file(self, audio_path: Path) -> None:
        """Validate audio file accessibility and format using comprehensive validation.
        
        Enhanced validation logic that uses the InputValidator for comprehensive
        file validation including existence, format, size limits, and provider-specific
        requirements.
        
        Args:
            audio_path: Path to audio file
            
        Raises:
            FileNotFoundError: Audio file does not exist
            PermissionError: Audio file is not readable  
            ValueError: Audio file format is not supported or exceeds size limits
        """
        validator = InputValidator()
        
        # Simple file validation
        if not validator.validate_file_path(audio_path, self.provider_name):
            raise ValueError(f"File validation failed for: {audio_path}")