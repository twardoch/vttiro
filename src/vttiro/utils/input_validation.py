# this_file: src/vttiro/utils/input_validation.py
"""Comprehensive input validation and sanitization system for VTTiro.

This module provides enterprise-grade input validation and sanitization to prevent
malformed data from reaching AI providers, improving reliability and user experience.

Used for:
- File path validation and sanitization
- Audio/video format validation 
- Configuration parameter validation
- Provider input sanitization
- User input cleaning and safety checks
"""

import os
import re
import mimetypes
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from urllib.parse import urlparse

try:
    from loguru import logger
except ImportError:
    import logging as logger

# Try to import additional metadata libraries
try:
    import mutagen
    from mutagen import File as MutagenFile
    HAS_MUTAGEN = True
except ImportError:
    HAS_MUTAGEN = False


@dataclass
class ValidationResult:
    """Result of input validation with details and suggestions."""
    
    is_valid: bool
    sanitized_value: Any
    error_message: Optional[str] = None
    warning_message: Optional[str] = None
    suggestions: List[str] = None
    
    def __post_init__(self):
        if self.suggestions is None:
            self.suggestions = []


class InputValidator:
    """Comprehensive input validation and sanitization system."""
    
    # File extension patterns
    AUDIO_EXTENSIONS = {
        '.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac', '.wma'
    }
    
    VIDEO_EXTENSIONS = {
        '.mp4', '.avi', '.mov', '.mkv', '.webm', '.wmv', '.flv', '.m4v'
    }
    
    # Maximum file sizes per provider (in bytes)
    PROVIDER_FILE_LIMITS = {
        'openai': 25 * 1024 * 1024,      # 25MB
        'gemini': 100 * 1024 * 1024,     # 100MB  
        'assemblyai': 500 * 1024 * 1024, # 500MB
        'deepgram': 2 * 1024 * 1024 * 1024  # 2GB
    }
    
    # Dangerous characters for path sanitization
    DANGEROUS_PATH_CHARS = r'[<>:"|?*\x00-\x1f]'
    
    # Valid language codes (ISO 639-1 and some common variants)
    VALID_LANGUAGE_CODES = {
        'auto', 'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'zh', 'ko',
        'ar', 'hi', 'th', 'vi', 'tr', 'pl', 'nl', 'sv', 'da', 'no', 'fi',
        'cs', 'sk', 'hu', 'ro', 'bg', 'hr', 'sr', 'sl', 'et', 'lv', 'lt',
        'en-US', 'en-GB', 'es-ES', 'es-MX', 'fr-FR', 'fr-CA', 'pt-BR', 'pt-PT',
        'zh-CN', 'zh-TW', 'ar-SA', 'ar-EG'
    }
    
    def __init__(self):
        """Initialize input validator."""
        self.mime_types = mimetypes.MimeTypes()
    
    def validate_file_path(self, file_path: Union[str, Path], provider: Optional[str] = None) -> ValidationResult:
        """Validate and sanitize file path input.
        
        Args:
            file_path: File path to validate
            provider: Optional provider name for size limit checking
            
        Returns:
            ValidationResult with validation status and sanitized path
        """
        try:
            # Convert to Path object for consistent handling
            if isinstance(file_path, str):
                path = Path(file_path)
            else:
                path = file_path
            
            # Basic existence check
            if not path.exists():
                return ValidationResult(
                    is_valid=False,
                    sanitized_value=str(path),
                    error_message=f"File not found: {path}",
                    suggestions=[
                        "Check if the file path is correct",
                        "Ensure the file exists and is accessible",
                        "Try using an absolute path"
                    ]
                )
            
            # Check if it's actually a file
            if not path.is_file():
                return ValidationResult(
                    is_valid=False,
                    sanitized_value=str(path),
                    error_message=f"Path is not a file: {path}",
                    suggestions=[
                        "Ensure the path points to a file, not a directory",
                        "Check file permissions"
                    ]
                )
            
            # Sanitize the path
            sanitized_path = self._sanitize_file_path(path)
            
            # Validate file format
            format_result = self._validate_file_format(sanitized_path)
            if not format_result.is_valid:
                return format_result
            
            # Check file size
            size_result = self._validate_file_size(sanitized_path, provider)
            if not size_result.is_valid:
                return size_result
                
            # Enhanced file content validation
            content_result = self._validate_file_content(sanitized_path)
            if not content_result.is_valid:
                return content_result
                
            # Additional security checks
            security_result = self._validate_file_security(sanitized_path)
            if not security_result.is_valid:
                return security_result
            
            warnings = []
            suggestions = []
            
            # Check for potential issues
            if path.suffix.lower() in {'.wav', '.flac'}:
                warnings.append("Uncompressed format may result in large file sizes")
                suggestions.append("Consider using MP3 or M4A for better performance")
            
            return ValidationResult(
                is_valid=True,
                sanitized_value=str(sanitized_path),
                warning_message="; ".join(warnings) if warnings else None,
                suggestions=suggestions
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                sanitized_value=str(file_path),
                error_message=f"Path validation error: {e}",
                suggestions=["Check file path format and permissions"]
            )
    
    def validate_url(self, url: str) -> ValidationResult:
        """Validate and sanitize URL input.
        
        Args:
            url: URL to validate
            
        Returns:
            ValidationResult with validation status and sanitized URL
        """
        try:
            # Basic URL parsing
            parsed = urlparse(url)
            
            if not parsed.scheme:
                return ValidationResult(
                    is_valid=False,
                    sanitized_value=url,
                    error_message="Invalid URL: missing scheme (http:// or https://)",
                    suggestions=[
                        "Add http:// or https:// to the beginning",
                        "Ensure the URL is properly formatted"
                    ]
                )
            
            if parsed.scheme not in ['http', 'https']:
                return ValidationResult(
                    is_valid=False,
                    sanitized_value=url,
                    error_message=f"Unsupported URL scheme: {parsed.scheme}",
                    suggestions=[
                        "Use http:// or https:// URLs only",
                        "Local file paths should be specified without URL schemes"
                    ]
                )
            
            if not parsed.netloc:
                return ValidationResult(
                    is_valid=False,
                    sanitized_value=url,
                    error_message="Invalid URL: missing domain",
                    suggestions=[
                        "Ensure the URL includes a valid domain name",
                        "Check URL format: https://example.com/video"
                    ]
                )
            
            # Sanitize URL
            sanitized_url = self._sanitize_url(url)
            
            # Check for known video/audio platforms
            suggestions = []
            warnings = []
            
            if 'youtube.com' in parsed.netloc or 'youtu.be' in parsed.netloc:
                suggestions.append("YouTube URLs are supported - ensure video is publicly accessible")
            elif 'vimeo.com' in parsed.netloc:
                suggestions.append("Vimeo URLs are supported - check privacy settings")
            else:
                warnings.append("URL from unknown platform - compatibility not guaranteed")
                suggestions.append("Test with a small file first to verify compatibility")
            
            return ValidationResult(
                is_valid=True,
                sanitized_value=sanitized_url,
                warning_message="; ".join(warnings) if warnings else None,
                suggestions=suggestions
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                sanitized_value=url,
                error_message=f"URL validation error: {e}",
                suggestions=["Check URL format and ensure it's properly formed"]
            )
    
    def validate_provider_name(self, provider: str) -> ValidationResult:
        """Validate provider name input.
        
        Args:
            provider: Provider name to validate
            
        Returns:
            ValidationResult with validation status
        """
        valid_providers = {'gemini', 'openai', 'assemblyai', 'deepgram'}
        
        if not isinstance(provider, str):
            return ValidationResult(
                is_valid=False,
                sanitized_value=str(provider),
                error_message="Provider must be a string",
                suggestions=[f"Valid providers: {', '.join(sorted(valid_providers))}"]
            )
        
        # Sanitize provider name
        sanitized_provider = provider.lower().strip()
        
        if not sanitized_provider:
            return ValidationResult(
                is_valid=False,
                sanitized_value=sanitized_provider,
                error_message="Provider name cannot be empty",
                suggestions=[f"Valid providers: {', '.join(sorted(valid_providers))}"]
            )
        
        if sanitized_provider not in valid_providers:
            return ValidationResult(
                is_valid=False,
                sanitized_value=sanitized_provider,
                error_message=f"Invalid provider: {sanitized_provider}",
                suggestions=[
                    f"Valid providers: {', '.join(sorted(valid_providers))}",
                    "Use 'vttiro engines' command to see available providers"
                ]
            )
        
        return ValidationResult(
            is_valid=True,
            sanitized_value=sanitized_provider
        )
    
    def validate_language_code(self, language: str) -> ValidationResult:
        """Validate language code input.
        
        Args:
            language: Language code to validate
            
        Returns:
            ValidationResult with validation status
        """
        if not isinstance(language, str):
            return ValidationResult(
                is_valid=False,
                sanitized_value=str(language),
                error_message="Language code must be a string",
                suggestions=["Use ISO 639-1 codes like 'en', 'es', 'fr' or 'auto'"]
            )
        
        # Sanitize language code
        sanitized_language = language.lower().strip()
        
        if not sanitized_language:
            return ValidationResult(
                is_valid=False,
                sanitized_value=sanitized_language,
                error_message="Language code cannot be empty",
                suggestions=["Use 'auto' for automatic detection or specific codes like 'en', 'es'"]
            )
        
        if sanitized_language not in self.VALID_LANGUAGE_CODES:
            return ValidationResult(
                is_valid=False,
                sanitized_value=sanitized_language,
                error_message=f"Invalid language code: {sanitized_language}",
                suggestions=[
                    "Use 'auto' for automatic language detection",
                    "Common codes: en, es, fr, de, it, pt, ru, ja, zh",
                    "Full list: ISO 639-1 language codes"
                ]
            )
        
        return ValidationResult(
            is_valid=True,
            sanitized_value=sanitized_language
        )
    
    def validate_output_path(self, output_path: Union[str, Path]) -> ValidationResult:
        """Validate and sanitize output path.
        
        Args:
            output_path: Output path to validate
            
        Returns:
            ValidationResult with validation status and sanitized path
        """
        try:
            if isinstance(output_path, str):
                path = Path(output_path)
            else:
                path = output_path
            
            # Sanitize the path
            sanitized_path = self._sanitize_file_path(path)
            
            # Check parent directory
            parent_dir = sanitized_path.parent
            if not parent_dir.exists():
                return ValidationResult(
                    is_valid=False,
                    sanitized_value=str(sanitized_path),
                    error_message=f"Output directory does not exist: {parent_dir}",
                    suggestions=[
                        f"Create directory: mkdir -p {parent_dir}",
                        "Ensure parent directory exists and is writable"
                    ]
                )
            
            # Check write permissions on parent directory
            if not os.access(parent_dir, os.W_OK):
                return ValidationResult(
                    is_valid=False,
                    sanitized_value=str(sanitized_path),
                    error_message=f"No write permission for directory: {parent_dir}",
                    suggestions=[
                        "Check directory permissions",
                        "Choose a different output location"
                    ]
                )
            
            # Check if file already exists
            warnings = []
            suggestions = []
            
            if sanitized_path.exists():
                warnings.append(f"Output file already exists: {sanitized_path}")
                suggestions.append("File will be overwritten")
            
            # Validate output format
            if sanitized_path.suffix.lower() not in {'.vtt', '.srt', '.txt'}:
                warnings.append("Unusual output file extension")
                suggestions.append("Consider using .vtt for WebVTT format")
            
            return ValidationResult(
                is_valid=True,
                sanitized_value=str(sanitized_path),
                warning_message="; ".join(warnings) if warnings else None,
                suggestions=suggestions
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                sanitized_value=str(output_path),
                error_message=f"Output path validation error: {e}",
                suggestions=["Check output path format and permissions"]
            )
    
    def validate_numeric_parameter(
        self, 
        value: Any, 
        param_name: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        allow_none: bool = False
    ) -> ValidationResult:
        """Validate numeric parameter input.
        
        Args:
            value: Value to validate
            param_name: Name of parameter for error messages
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            allow_none: Whether None is allowed
            
        Returns:
            ValidationResult with validation status
        """
        if value is None:
            if allow_none:
                return ValidationResult(is_valid=True, sanitized_value=None)
            else:
                return ValidationResult(
                    is_valid=False,
                    sanitized_value=None,
                    error_message=f"{param_name} cannot be None",
                    suggestions=["Provide a numeric value"]
                )
        
        # Try to convert to float
        try:
            if isinstance(value, str):
                # Sanitize string input
                sanitized_str = value.strip()
                if not sanitized_str:
                    return ValidationResult(
                        is_valid=False,
                        sanitized_value=value,
                        error_message=f"{param_name} cannot be empty",
                        suggestions=["Provide a numeric value"]
                    )
                numeric_value = float(sanitized_str)
            else:
                numeric_value = float(value)
        except (ValueError, TypeError):
            return ValidationResult(
                is_valid=False,
                sanitized_value=value,
                error_message=f"{param_name} must be a number, got: {type(value).__name__}",
                suggestions=["Provide a valid numeric value"]
            )
        
        # Check range constraints
        suggestions = []
        
        if min_value is not None and numeric_value < min_value:
            return ValidationResult(
                is_valid=False,
                sanitized_value=numeric_value,
                error_message=f"{param_name} must be >= {min_value}, got: {numeric_value}",
                suggestions=[f"Use a value >= {min_value}"]
            )
        
        if max_value is not None and numeric_value > max_value:
            return ValidationResult(
                is_valid=False,
                sanitized_value=numeric_value,
                error_message=f"{param_name} must be <= {max_value}, got: {numeric_value}",
                suggestions=[f"Use a value <= {max_value}"]
            )
        
        # Add helpful suggestions for common parameters
        if param_name.lower() in ['confidence_threshold', 'confidence']:
            if not (0.0 <= numeric_value <= 1.0):
                suggestions.append("Confidence values should be between 0.0 and 1.0")
        
        return ValidationResult(
            is_valid=True,
            sanitized_value=numeric_value,
            suggestions=suggestions
        )
    
    def sanitize_text_input(self, text: str, max_length: Optional[int] = None) -> ValidationResult:
        """Sanitize text input for safe processing.
        
        Args:
            text: Text to sanitize
            max_length: Maximum allowed length
            
        Returns:
            ValidationResult with sanitized text
        """
        if not isinstance(text, str):
            return ValidationResult(
                is_valid=False,
                sanitized_value=str(text),
                error_message="Input must be a string",
                suggestions=["Provide text input"]
            )
        
        # Basic sanitization
        sanitized = text.strip()
        
        # Remove null bytes and control characters
        sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', sanitized)
        
        # Normalize whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized)
        
        warnings = []
        suggestions = []
        
        if len(sanitized) != len(text):
            warnings.append("Text was sanitized (removed control characters)")
        
        if max_length and len(sanitized) > max_length:
            return ValidationResult(
                is_valid=False,
                sanitized_value=sanitized,
                error_message=f"Text too long: {len(sanitized)} characters (max: {max_length})",
                suggestions=[f"Shorten text to {max_length} characters or less"]
            )
        
        if not sanitized:
            return ValidationResult(
                is_valid=False,
                sanitized_value=sanitized,
                error_message="Text cannot be empty after sanitization",
                suggestions=["Provide non-empty text input"]
            )
        
        return ValidationResult(
            is_valid=True,
            sanitized_value=sanitized,
            warning_message="; ".join(warnings) if warnings else None,
            suggestions=suggestions
        )
    
    def _sanitize_file_path(self, path: Path) -> Path:
        """Sanitize file path by removing dangerous characters.
        
        Args:
            path: Path to sanitize
            
        Returns:
            Sanitized path
        """
        # Convert to string for processing
        path_str = str(path)
        
        # Remove dangerous characters
        sanitized_str = re.sub(self.DANGEROUS_PATH_CHARS, '', path_str)
        
        # Resolve to absolute path for security
        return Path(sanitized_str).resolve()
    
    def _sanitize_url(self, url: str) -> str:
        """Sanitize URL by removing dangerous components.
        
        Args:
            url: URL to sanitize
            
        Returns:
            Sanitized URL
        """
        # Basic URL sanitization
        # Remove leading/trailing whitespace
        sanitized = url.strip()
        
        # Remove null bytes and control characters
        sanitized = re.sub(r'[\x00-\x1f\x7f]', '', sanitized)
        
        return sanitized
    
    def _validate_file_format(self, path: Path) -> ValidationResult:
        """Validate file format based on extension and MIME type.
        
        Args:
            path: File path to validate
            
        Returns:
            ValidationResult for file format
        """
        extension = path.suffix.lower()
        
        if extension not in (self.AUDIO_EXTENSIONS | self.VIDEO_EXTENSIONS):
            return ValidationResult(
                is_valid=False,
                sanitized_value=str(path),
                error_message=f"Unsupported file format: {extension}",
                suggestions=[
                    f"Supported audio: {', '.join(sorted(self.AUDIO_EXTENSIONS))}",
                    f"Supported video: {', '.join(sorted(self.VIDEO_EXTENSIONS))}",
                    "Use 'vttiro formats' command to see all supported formats"
                ]
            )
        
        # Additional MIME type validation if available
        mime_type, _ = self.mime_types.guess_type(str(path))
        warnings = []
        
        if mime_type:
            if not (mime_type.startswith('audio/') or mime_type.startswith('video/')):
                warnings.append(f"Unexpected MIME type: {mime_type}")
        
        return ValidationResult(
            is_valid=True,
            sanitized_value=str(path),
            warning_message="; ".join(warnings) if warnings else None
        )
    
    def _validate_file_size(self, path: Path, provider: Optional[str] = None) -> ValidationResult:
        """Validate file size constraints.
        
        Args:
            path: File path to check
            provider: Provider name for size limits
            
        Returns:
            ValidationResult for file size
        """
        try:
            file_size = path.stat().st_size
        except OSError as e:
            return ValidationResult(
                is_valid=False,
                sanitized_value=str(path),
                error_message=f"Cannot read file size: {e}",
                suggestions=["Check file permissions and accessibility"]
            )
        
        warnings = []
        suggestions = []
        
        # Check general size warnings
        if file_size > 100 * 1024 * 1024:  # 100MB
            warnings.append(f"Large file: {self._format_file_size(file_size)}")
            suggestions.append("Large files may take longer to process")
        
        # Check provider-specific limits
        if provider:
            provider_limit = self.PROVIDER_FILE_LIMITS.get(provider.lower())
            if provider_limit and file_size > provider_limit:
                return ValidationResult(
                    is_valid=False,
                    sanitized_value=str(path),
                    error_message=(
                        f"File too large for {provider}: {self._format_file_size(file_size)} "
                        f"(limit: {self._format_file_size(provider_limit)})"
                    ),
                    suggestions=[
                        f"Use a provider with higher limits (deepgram: 2GB, assemblyai: 500MB)",
                        "Compress or split the file",
                        "Use a smaller input file"
                    ]
                )
        
        return ValidationResult(
            is_valid=True,
            sanitized_value=str(path),
            warning_message="; ".join(warnings) if warnings else None,
            suggestions=suggestions
        )
    
    def _validate_file_security(self, path: Path) -> ValidationResult:
        """Validate file for security concerns.
        
        Args:
            path: File path to validate
            
        Returns:
            ValidationResult for security validation
        """
        warnings = []
        suggestions = []
        
        # Check for suspicious file patterns
        if '..' in str(path):
            warnings.append("Path contains parent directory references")
            suggestions.append("Use absolute paths for security")
        
        # Check for hidden files
        if path.name.startswith('.'):
            warnings.append("Processing hidden file")
        
        # Check for executable extensions (shouldn't be processed)
        dangerous_extensions = {'.exe', '.bat', '.cmd', '.sh', '.ps1', '.scr'}
        if path.suffix.lower() in dangerous_extensions:
            return ValidationResult(
                is_valid=False,
                sanitized_value=str(path),
                error_message=f"Executable file not allowed: {path.suffix}",
                suggestions=["Only audio and video files are supported"]
            )
        
        return ValidationResult(
            is_valid=True,
            sanitized_value=str(path),
            warning_message="; ".join(warnings) if warnings else None,
            suggestions=suggestions
        )
    
    def _validate_file_content(self, path: Path) -> ValidationResult:
        """Enhanced file content validation including empty/corrupted file detection.
        
        Args:
            path: File path to validate
            
        Returns:
            ValidationResult for content validation
        """
        warnings = []
        suggestions = []
        
        try:
            # Check if file is empty
            file_size = path.stat().st_size
            if file_size == 0:
                return ValidationResult(
                    is_valid=False,
                    sanitized_value=str(path),
                    error_message="File is empty",
                    suggestions=[
                        "Ensure the file contains audio or video data",
                        "Check if the file was uploaded or copied completely",
                        "Try re-recording or re-downloading the file"
                    ]
                )
            
            # Check for very small files (likely corrupt or incomplete)
            if file_size < 1024:  # Less than 1KB
                warnings.append(f"File is very small ({self._format_file_size(file_size)})")
                suggestions.append("Verify file contains valid audio/video data")
            
            # Try to extract metadata for deeper validation
            metadata_info = self._extract_file_metadata(path)
            
            if metadata_info['has_metadata']:
                # File has metadata, likely valid
                if metadata_info['duration'] is not None:
                    if metadata_info['duration'] < 0.1:  # Less than 100ms
                        warnings.append("Audio/video duration is very short")
                        suggestions.append("Ensure the file contains sufficient content for transcription")
                    elif metadata_info['duration'] > 7200:  # More than 2 hours
                        warnings.append("Very long audio/video file")
                        suggestions.append("Consider splitting large files for better performance")
                
                if metadata_info['bitrate'] is not None and metadata_info['bitrate'] < 32000:
                    warnings.append("Low bitrate audio may affect transcription quality")
                    suggestions.append("Higher bitrate audio (64kbps+) provides better results")
                    
            else:
                # No metadata found - could be corrupted
                if file_size > 1024 * 1024:  # Larger than 1MB but no metadata
                    warnings.append("Unable to read file metadata - file may be corrupted")
                    suggestions.extend([
                        "Try converting the file to a standard format (MP3, WAV, MP4)",
                        "Check if the file plays correctly in a media player",
                        "Re-encode the file if it appears corrupted"
                    ])
            
            # Additional format-specific checks
            extension = path.suffix.lower()
            if extension in {'.wav', '.flac'} and file_size > 100 * 1024 * 1024:  # > 100MB
                warnings.append("Large uncompressed audio file")
                suggestions.append("Consider using compressed format (MP3, M4A) for better performance")
            
            return ValidationResult(
                is_valid=True,
                sanitized_value=str(path),
                warning_message="; ".join(warnings) if warnings else None,
                suggestions=suggestions
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                sanitized_value=str(path),
                error_message=f"Error reading file content: {str(e)}",
                suggestions=[
                    "Check file permissions and accessibility",
                    "Ensure the file is not corrupted or locked",
                    "Try copying the file to a different location"
                ]
            )
    
    def _extract_file_metadata(self, path: Path) -> dict:
        """Extract file metadata for validation.
        
        Args:
            path: File path to analyze
            
        Returns:
            Dictionary with metadata information
        """
        metadata = {
            'has_metadata': False,
            'duration': None,
            'bitrate': None,
            'channels': None,
            'sample_rate': None
        }
        
        if HAS_MUTAGEN:
            try:
                audio_file = MutagenFile(str(path))
                if audio_file is not None:
                    metadata['has_metadata'] = True
                    
                    # Extract duration
                    if hasattr(audio_file, 'info') and hasattr(audio_file.info, 'length'):
                        metadata['duration'] = audio_file.info.length
                    
                    # Extract bitrate
                    if hasattr(audio_file, 'info') and hasattr(audio_file.info, 'bitrate'):
                        metadata['bitrate'] = audio_file.info.bitrate
                    
                    # Extract channels
                    if hasattr(audio_file, 'info') and hasattr(audio_file.info, 'channels'):
                        metadata['channels'] = audio_file.info.channels
                        
                    # Extract sample rate
                    if hasattr(audio_file, 'info') and hasattr(audio_file.info, 'sample_rate'):
                        metadata['sample_rate'] = audio_file.info.sample_rate
                        
            except Exception:
                # If metadata extraction fails, continue with basic validation
                pass
        
        return metadata
    
    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format.
        
        Args:
            size_bytes: File size in bytes
            
        Returns:
            Formatted file size string
        """
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f}{unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f}TB"


class ProviderInputSanitizer:
    """Specialized sanitizer for provider-specific input requirements."""
    
    def __init__(self, validator: InputValidator):
        """Initialize with input validator.
        
        Args:
            validator: InputValidator instance
        """
        self.validator = validator
    
    def sanitize_for_provider(
        self, 
        provider: str, 
        inputs: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any], List[str]]:
        """Sanitize inputs for specific provider requirements.
        
        Args:
            provider: Provider name
            inputs: Dictionary of input parameters
            
        Returns:
            Tuple of (is_valid, sanitized_inputs, error_messages)
        """
        sanitized = {}
        errors = []
        
        # Validate provider name first
        provider_result = self.validator.validate_provider_name(provider)
        if not provider_result.is_valid:
            return False, inputs, [provider_result.error_message]
        
        provider = provider_result.sanitized_value
        
        # Provider-specific sanitization
        if provider == 'openai':
            return self._sanitize_openai_inputs(inputs)
        elif provider == 'gemini':
            return self._sanitize_gemini_inputs(inputs)
        elif provider == 'assemblyai':
            return self._sanitize_assemblyai_inputs(inputs)
        elif provider == 'deepgram':
            return self._sanitize_deepgram_inputs(inputs)
        else:
            return True, inputs, []
    
    def _sanitize_openai_inputs(self, inputs: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], List[str]]:
        """Sanitize inputs for OpenAI provider.
        
        Args:
            inputs: Input parameters
            
        Returns:
            Tuple of (is_valid, sanitized_inputs, errors)
        """
        sanitized = inputs.copy()
        errors = []
        
        # OpenAI-specific validations
        if 'temperature' in sanitized:
            temp_result = self.validator.validate_numeric_parameter(
                sanitized['temperature'], 'temperature', 0.0, 2.0
            )
            if not temp_result.is_valid:
                errors.append(temp_result.error_message)
            else:
                sanitized['temperature'] = temp_result.sanitized_value
        
        if 'prompt' in sanitized:
            # OpenAI has token limits for prompts
            prompt_result = self.validator.sanitize_text_input(
                sanitized['prompt'], max_length=1000
            )
            if not prompt_result.is_valid:
                errors.append(prompt_result.error_message)
            else:
                sanitized['prompt'] = prompt_result.sanitized_value
        
        return len(errors) == 0, sanitized, errors
    
    def _sanitize_gemini_inputs(self, inputs: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], List[str]]:
        """Sanitize inputs for Gemini provider.
        
        Args:
            inputs: Input parameters
            
        Returns:
            Tuple of (is_valid, sanitized_inputs, errors)
        """
        sanitized = inputs.copy()
        errors = []
        
        # Gemini-specific validations
        if 'safety_settings' in sanitized:
            # Validate safety settings structure
            if not isinstance(sanitized['safety_settings'], (dict, list)):
                errors.append("Gemini safety_settings must be dict or list")
        
        if 'context' in sanitized:
            # Gemini has generous context limits
            context_result = self.validator.sanitize_text_input(
                sanitized['context'], max_length=10000
            )
            if not context_result.is_valid:
                errors.append(context_result.error_message)
            else:
                sanitized['context'] = context_result.sanitized_value
        
        return len(errors) == 0, sanitized, errors
    
    def _sanitize_assemblyai_inputs(self, inputs: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], List[str]]:
        """Sanitize inputs for AssemblyAI provider.
        
        Args:
            inputs: Input parameters
            
        Returns:
            Tuple of (is_valid, sanitized_inputs, errors)
        """
        sanitized = inputs.copy()
        errors = []
        
        # AssemblyAI-specific validations
        if 'speaker_labels' in sanitized:
            if not isinstance(sanitized['speaker_labels'], bool):
                try:
                    sanitized['speaker_labels'] = bool(sanitized['speaker_labels'])
                except (ValueError, TypeError):
                    errors.append("AssemblyAI speaker_labels must be boolean")
        
        if 'word_boost' in sanitized:
            if not isinstance(sanitized['word_boost'], list):
                errors.append("AssemblyAI word_boost must be a list of strings")
            else:
                # Validate each word in boost list
                valid_words = []
                for word in sanitized['word_boost']:
                    word_result = self.validator.sanitize_text_input(str(word), max_length=50)
                    if word_result.is_valid:
                        valid_words.append(word_result.sanitized_value)
                sanitized['word_boost'] = valid_words
        
        return len(errors) == 0, sanitized, errors
    
    def _sanitize_deepgram_inputs(self, inputs: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], List[str]]:
        """Sanitize inputs for Deepgram provider.
        
        Args:
            inputs: Input parameters
            
        Returns:
            Tuple of (is_valid, sanitized_inputs, errors)
        """
        sanitized = inputs.copy()
        errors = []
        
        # Deepgram-specific validations
        if 'model' in sanitized:
            valid_models = {'nova-2', 'nova', 'enhanced', 'base', 'whisper-cloud'}
            if sanitized['model'] not in valid_models:
                errors.append(f"Invalid Deepgram model: {sanitized['model']}")
        
        if 'keywords' in sanitized:
            if not isinstance(sanitized['keywords'], list):
                errors.append("Deepgram keywords must be a list")
            else:
                # Validate keyword format
                valid_keywords = []
                for keyword in sanitized['keywords']:
                    if isinstance(keyword, str):
                        kw_result = self.validator.sanitize_text_input(keyword, max_length=30)
                        if kw_result.is_valid:
                            valid_keywords.append(kw_result.sanitized_value)
                sanitized['keywords'] = valid_keywords
        
        return len(errors) == 0, sanitized, errors


# Global validator instance
_validator_instance: Optional[InputValidator] = None


def get_validator() -> InputValidator:
    """Get global input validator instance.
    
    Returns:
        InputValidator instance
    """
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = InputValidator()
    return _validator_instance


def get_provider_sanitizer() -> ProviderInputSanitizer:
    """Get provider input sanitizer instance.
    
    Returns:
        ProviderInputSanitizer instance
    """
    return ProviderInputSanitizer(get_validator())