# this_file: src/vttiro/core/constants.py
"""Constants for VTTiro configuration and processing.

This module defines commonly used constants throughout the VTTiro system
to improve maintainability and avoid magic numbers in the codebase.

Used by:
- Processing modules for thresholds and limits
- Validation logic for boundaries and ranges
- Timeout and retry configurations
"""

# Memory thresholds (percentages)
MEMORY_HIGH_USAGE_THRESHOLD = 85  # System memory usage considered high
MEMORY_CRITICAL_THRESHOLD = 70  # Memory usage threshold for optimizations

# Size limits
LARGE_FILE_SIZE_MB = 1000  # Files above this size get special handling
MIN_AUDIO_DURATION = 5.0  # Minimum audio duration in seconds

# Timeout configurations (seconds)
MIN_TIMEOUT_SECONDS = 10  # Minimum reasonable timeout
MAX_TIMEOUT_SECONDS = 1800  # Maximum timeout (30 minutes)

# Timestamp processing
MIN_TIMESTAMP_PARTS = 2  # Minimum parts in timestamp (MM:SS)
MAX_TIMESTAMP_PARTS = 3  # Maximum parts in timestamp (HH:MM:SS)
MILLISECOND_PRECISION = 100  # For millisecond validation

# Processing constants
DEFAULT_MAX_RETRIES = 3  # Default retry attempts
DEFAULT_CHUNK_SIZE_MB = 20  # Default audio chunk size in MB

# Provider-specific limits
GEMINI_MAX_FILE_SIZE_MB = 20
OPENAI_MAX_FILE_SIZE_MB = 25
DEEPGRAM_MAX_FILE_SIZE_MB = 2000
ASSEMBLYAI_MAX_FILE_SIZE_MB = 500

# WebVTT format
WEBVTT_MAX_LINE_LENGTH = 80  # Maximum characters per line
WEBVTT_DEFAULT_CONFIDENCE = 0.9  # Default confidence when not provided

# Audio processing
AUDIO_SAMPLE_RATE = 16000  # Target sample rate for audio processing
AUDIO_CHANNELS = 1  # Mono audio processing
