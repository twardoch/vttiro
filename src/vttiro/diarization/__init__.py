#!/usr/bin/env python3
# this_file: src/vttiro/diarization/__init__.py
"""Advanced speaker diarization with pyannote.audio 3.1 and ensemble methods."""

from .core import DiarizationEngine, SpeakerSegment, DiarizationResult, DiarizationConfig
from .embeddings import SpeakerEmbeddingManager, EmbeddingExtractor
from .clustering import AdaptiveClusteringEngine, ClusteringMethod
from .quality import DiarizationQualityAssessment, QualityMetrics

__all__ = [
    "DiarizationEngine",
    "SpeakerSegment", 
    "DiarizationResult",
    "DiarizationConfig",
    "SpeakerEmbeddingManager",
    "EmbeddingExtractor",
    "AdaptiveClusteringEngine",
    "ClusteringMethod",
    "DiarizationQualityAssessment",
    "QualityMetrics"
]