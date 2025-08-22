# this_file: src/vttiro/output/__init__.py
"""Simple output format generation focused on WebVTT.

This module provides WebVTT subtitle generation without over-engineered
quality analysis or multi-format complexity.
"""

from .enhanced_webvtt import EnhancedWebVTTFormatter, WebVTTConfig

__all__ = [
    "EnhancedWebVTTFormatter",
    "WebVTTConfig",
]