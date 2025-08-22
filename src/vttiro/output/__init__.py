# this_file: src/vttiro/output/__init__.py
"""Enhanced output format generation with quality optimization and accessibility compliance.

This module provides comprehensive subtitle generation capabilities including:
- Enhanced WebVTT formatter with accessibility features
- Multi-format export (SRT, TTML, ASS, transcript)
- Comprehensive quality analysis and optimization
- WCAG compliance validation and recommendations
"""

from .enhanced_webvtt import EnhancedWebVTTFormatter, WebVTTConfig, QualityMetrics
from .multi_format_exporter import (
    MultiFormatExporter, 
    SubtitleFormat, 
    FormatConfig, 
    ExportResult
)
from .quality_analyzer import (
    OutputQualityAnalyzer,
    QualityMetrics as AnalysisQualityMetrics,
    QualityDimension,
    QualityIssue,
    OptimizationRecommendation,
    SeverityLevel
)

__all__ = [
    # Enhanced WebVTT formatter
    "EnhancedWebVTTFormatter",
    "WebVTTConfig",
    "QualityMetrics",
    
    # Multi-format exporter
    "MultiFormatExporter",
    "SubtitleFormat",
    "FormatConfig",
    "ExportResult",
    
    # Quality analyzer
    "OutputQualityAnalyzer",
    "AnalysisQualityMetrics",
    "QualityDimension",
    "QualityIssue",
    "OptimizationRecommendation",
    "SeverityLevel",
]