# this_file: src/vttiro/utils/input_validation.py
"""Simple input validation for VTTiro - focused on essential checks only.

This module provides basic file validation without over-engineered security theater.
Focus: file existence, format, and basic size checks.
"""

import mimetypes
import os
from pathlib import Path
from typing import Optional, Union

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class InputValidator:
    """Simple input validator focused on essential checks."""
    
    # Supported file extensions
    AUDIO_EXTENSIONS = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac'}
    VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}
    
    # Basic file size limits (in MB)
    MAX_FILE_SIZE_MB = 500  # 500MB reasonable limit
    
    def __init__(self):
        """Initialize validator."""
        self.mime_types = mimetypes.MimeTypes()
    
    def validate_file_path(self, file_path: Union[str, Path], provider: Optional[str] = None) -> bool:
        """Basic file path validation.
        
        Args:
            file_path: File path to validate
            provider: Provider name (for compatibility, not used in simple version)
            
        Returns:
            True if file is valid, False otherwise
        """
        try:
            path = Path(file_path)
            
            # Check if file exists
            if not path.exists():
                logger.error(f"File does not exist: {path}")
                return False
            
            # Check if it's a file (not directory)
            if not path.is_file():
                logger.error(f"Path is not a file: {path}")
                return False
            
            # Check file extension
            ext = path.suffix.lower()
            if ext not in (self.AUDIO_EXTENSIONS | self.VIDEO_EXTENSIONS):
                logger.error(f"Unsupported file format: {ext}")
                return False
            
            # Basic size check
            size_mb = path.stat().st_size / (1024 * 1024)
            if size_mb > self.MAX_FILE_SIZE_MB:
                logger.error(f"File too large: {size_mb:.1f}MB > {self.MAX_FILE_SIZE_MB}MB")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"File validation error: {e}")
            return False
    
    def validate_provider_name(self, provider: str) -> bool:
        """Basic provider name validation.
        
        Args:
            provider: Provider name to validate
            
        Returns:
            True if provider is supported
        """
        supported_providers = {"gemini", "openai", "assemblyai", "deepgram"}
        return provider.lower() in supported_providers
    
    def validate_language_code(self, language: str) -> bool:
        """Basic language code validation.
        
        Args:
            language: Language code to validate
            
        Returns:
            True if language code looks valid
        """
        # Simple check for common language codes
        common_languages = {
            'auto', 'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'zh', 'ko',
            'ar', 'hi', 'th', 'vi', 'tr', 'pl', 'nl', 'sv', 'da', 'no', 'fi'
        }
        return language.lower() in common_languages
    
    def validate_output_path(self, output_path: Union[str, Path]) -> bool:
        """Basic output path validation.
        
        Args:
            output_path: Output path to validate
            
        Returns:
            True if output path is valid
        """
        try:
            path = Path(output_path)
            
            # Check if parent directory exists or can be created
            parent = path.parent
            if not parent.exists():
                try:
                    parent.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    logger.error(f"Cannot create output directory: {e}")
                    return False
            
            # Check if we can write to the location
            if path.exists() and not os.access(path, os.W_OK):
                logger.error(f"Cannot write to output path: {path}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Output path validation error: {e}")
            return False