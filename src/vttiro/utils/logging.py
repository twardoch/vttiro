# this_file: src/vttiro/utils/logging.py
"""Centralized logging configuration for VTTiro.

This module provides consistent logging setup and structured logging utilities
for debugging, performance monitoring, and diagnostic capabilities.

Used by:
- All core modules for structured logging
- CLI for diagnostic output and debug mode
- Provider modules for debugging transcription issues
- Processing modules for performance monitoring
"""

import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger


def setup_logging(verbose: bool = False, debug: bool = False, log_file: Path | None = None) -> None:
    """Configure loguru logging with consistent format and levels.

    Args:
        verbose: Enable INFO level logging
        debug: Enable DEBUG level logging (implies verbose)
        log_file: Optional file path for logging output
    """
    # Remove default logger
    logger.remove()

    # Determine log level
    if debug:
        level = "DEBUG"
    elif verbose:
        level = "INFO"
    else:
        level = "WARNING"

    # Console format with colors and structure
    console_format = (
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    # Add console handler
    logger.add(sys.stderr, format=console_format, level=level, colorize=True, backtrace=debug, diagnose=debug)

    # Add file handler if requested
    if log_file:
        file_format = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}"

        logger.add(
            log_file,
            format=file_format,
            level="DEBUG",  # Always debug level for files
            rotation="10 MB",
            retention="7 days",
            backtrace=True,
            diagnose=True,
        )

    logger.info(f"Logging initialized - Level: {level}")


def log_milestone(milestone: str, details: dict[str, Any] | None = None, timing: float | None = None) -> None:
    """Log structured milestone information for key processing steps.

    Args:
        milestone: Name of the milestone (e.g., "audio_extraction", "transcription_start")
        details: Optional dictionary of additional details
        timing: Optional elapsed time in seconds
    """
    message_parts = [f"MILESTONE: {milestone}"]

    if timing is not None:
        message_parts.append(f"elapsed={timing:.2f}s")

    if details:
        detail_strs = [f"{k}={v}" for k, v in details.items()]
        message_parts.extend(detail_strs)

    logger.info(" | ".join(message_parts))


def log_performance(operation: str, duration: float, metrics: dict[str, Any] | None = None) -> None:
    """Log performance information for monitoring and optimization.

    Args:
        operation: Name of the operation (e.g., "chunk_transcription", "audio_processing")
        duration: Operation duration in seconds
        metrics: Optional performance metrics (size, speed, etc.)
    """
    message_parts = [f"PERF: {operation}", f"duration={duration:.3f}s"]

    if metrics:
        metric_strs = [f"{k}={v}" for k, v in metrics.items()]
        message_parts.extend(metric_strs)

    logger.info(" | ".join(message_parts))


def log_provider_debug(provider: str, operation: str, details: dict[str, Any], success: bool = True) -> None:
    """Log provider-specific debugging information.

    Args:
        provider: Provider name (gemini, openai, etc.)
        operation: Operation being performed
        details: Dictionary of relevant details
        success: Whether the operation succeeded
    """
    status = "SUCCESS" if success else "FAILURE"
    message_parts = [f"PROVIDER: {provider.upper()}", f"operation={operation}", f"status={status}"]

    detail_strs = [f"{k}={v}" for k, v in details.items()]
    message_parts.extend(detail_strs)

    log_level = logger.info if success else logger.error
    log_level(" | ".join(message_parts))


@contextmanager
def log_timing(operation: str, details: dict[str, Any] | None = None):
    """Context manager for timing operations with automatic logging.

    Args:
        operation: Name of the operation being timed
        details: Optional additional details to log

    Example:
        with log_timing("audio_extraction", {"file_size": "15.2MB"}):
            # ... perform audio extraction ...
            pass
    """
    start_time = time.time()
    logger.debug(f"Starting {operation}")

    try:
        yield
        success = True
    except Exception as e:
        success = False
        logger.error(f"Operation failed: {operation} - {e}")
        raise
    finally:
        duration = time.time() - start_time
        status = "completed" if success else "failed"

        message_parts = [f"TIMING: {operation}", f"status={status}", f"duration={duration:.3f}s"]

        if details:
            detail_strs = [f"{k}={v}" for k, v in details.items()]
            message_parts.extend(detail_strs)

        logger.info(" | ".join(message_parts))


def get_debug_context() -> dict[str, Any]:
    """Get current system context for debugging purposes.

    Returns:
        Dictionary with system information for debugging
    """
    import platform

    import psutil

    from vttiro import __version__

    return {
        "vttiro_version": __version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "memory_usage_mb": psutil.Process().memory_info().rss / (1024 * 1024),
        "available_memory_mb": psutil.virtual_memory().available / (1024 * 1024),
        "cpu_count": psutil.cpu_count(),
    }


def log_system_info() -> None:
    """Log system information for diagnostic purposes."""
    try:
        context = get_debug_context()
        logger.info(f"SYSTEM: {' | '.join(f'{k}={v}' for k, v in context.items())}")
    except Exception as e:
        logger.warning(f"Could not collect system info: {e}")
