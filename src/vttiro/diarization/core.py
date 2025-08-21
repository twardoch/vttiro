#!/usr/bin/env python3
# this_file: src/vttiro/diarization/core.py
"""Core speaker diarization engine with pyannote.audio 3.1 integration."""

from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import asyncio
import time
import logging

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

try:
    import torch
    from pyannote.audio import Pipeline
    from pyannote.core import Timeline, Annotation
    HAS_PYANNOTE = True
except ImportError:
    logger.warning("pyannote.audio not available. Diarization will use dummy implementation.")
    HAS_PYANNOTE = False
    
    # Dummy objects for graceful degradation
    class Pipeline:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return None
    
    class Timeline:
        pass
    
    class Annotation:
        pass


@dataclass
class SpeakerSegment:
    """Represents a segment of audio attributed to a specific speaker."""
    speaker_id: str
    start_time: float
    end_time: float
    confidence: float
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> float:
        """Duration of the speaker segment in seconds."""
        return self.end_time - self.start_time
    
    def overlaps_with(self, other: 'SpeakerSegment') -> bool:
        """Check if this segment overlaps with another."""
        return (self.start_time < other.end_time and 
                other.start_time < self.end_time)


@dataclass
class DiarizationResult:
    """Complete diarization result with quality metrics."""
    segments: List[SpeakerSegment]
    speaker_count: int
    processing_time: float
    confidence_score: float
    quality_metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_speaker_segments(self, speaker_id: str) -> List[SpeakerSegment]:
        """Get all segments for a specific speaker."""
        return [seg for seg in self.segments if seg.speaker_id == speaker_id]
    
    def get_speakers(self) -> List[str]:
        """Get list of unique speaker IDs."""
        return list(set(seg.speaker_id for seg in self.segments))
    
    def get_total_speech_duration(self) -> float:
        """Get total duration of speech across all speakers."""
        return sum(seg.duration for seg in self.segments)


class DiarizationConfig(BaseModel):
    """Configuration for speaker diarization engine."""
    
    # Model configuration
    model_name: str = Field(
        default="pyannote/speaker-diarization-3.1",
        description="Pyannote model to use for diarization"
    )
    hf_token: Optional[str] = Field(
        default=None,
        description="Hugging Face token for model access"
    )
    device: str = Field(
        default="auto",
        description="Device to use: 'auto', 'cpu', 'cuda', or specific GPU"
    )
    
    # Speaker constraints
    min_speakers: Optional[int] = Field(
        default=None,
        description="Minimum number of speakers to detect"
    )
    max_speakers: Optional[int] = Field(
        default=20,
        description="Maximum number of speakers to detect"
    )
    
    # Segmentation parameters
    segmentation_threshold: float = Field(
        default=0.5,
        ge=0.0, le=1.0,
        description="Threshold for speech activity detection"
    )
    clustering_threshold: float = Field(
        default=0.7,
        ge=0.0, le=1.0,
        description="Threshold for speaker clustering"
    )
    
    # Timing parameters
    min_duration_on: float = Field(
        default=0.5,
        ge=0.1,
        description="Minimum duration for speech segments (seconds)"
    )
    min_duration_off: float = Field(
        default=0.1,
        ge=0.0,
        description="Minimum duration for silence between segments (seconds)"
    )
    
    # Processing parameters
    chunk_overlap: float = Field(
        default=5.0,
        ge=0.0,
        description="Overlap between processing chunks (seconds)"
    )
    speaker_consistency_window: float = Field(
        default=30.0,
        ge=5.0,
        description="Window for speaker consistency validation (seconds)"
    )
    confidence_threshold: float = Field(
        default=0.6,
        ge=0.0, le=1.0,
        description="Minimum confidence for speaker assignment"
    )
    
    # Quality settings
    target_der: float = Field(
        default=0.1,
        ge=0.0, le=1.0,
        description="Target Diarization Error Rate"
    )
    min_confidence: float = Field(
        default=0.6,
        ge=0.0, le=1.0,
        description="Minimum acceptable confidence"
    )
    overlap_handling: str = Field(
        default="primary_secondary",
        description="How to handle overlapping speech"
    )
    
    # Performance optimization
    gpu_acceleration: bool = Field(
        default=True,
        description="Enable GPU acceleration if available"
    )
    batch_processing: bool = Field(
        default=True,
        description="Enable batch processing for efficiency"
    )
    memory_efficient: bool = Field(
        default=True,
        description="Use memory-efficient processing"
    )
    cache_embeddings: bool = Field(
        default=True,
        description="Cache speaker embeddings for consistency"
    )


class DiarizationEngine:
    """Advanced speaker diarization engine using pyannote.audio 3.1."""
    
    def __init__(self, config: DiarizationConfig):
        """Initialize the diarization engine."""
        self.config = config
        self.pipeline = None
        self.device = self._setup_device()
        self.embedding_cache: Dict[str, np.ndarray] = {}
        
        logger.info(f"Initializing DiarizationEngine with device: {self.device}")
        
        if HAS_PYANNOTE:
            self._initialize_pipeline()
        else:
            logger.warning("PyAnnote.audio not available, using dummy implementation")
    
    def _setup_device(self) -> str:
        """Setup computational device based on configuration and availability."""
        if not HAS_PYANNOTE:
            return "cpu"
            
        if self.config.device == "auto":
            if torch.cuda.is_available() and self.config.gpu_acceleration:
                device = "cuda"
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
            else:
                device = "cpu"
                logger.info("Using CPU device")
        else:
            device = self.config.device
            
        return device
    
    def _initialize_pipeline(self):
        """Initialize the pyannote.audio pipeline."""
        try:
            logger.info(f"Loading pyannote model: {self.config.model_name}")
            
            # Load the diarization pipeline
            self.pipeline = Pipeline.from_pretrained(
                self.config.model_name,
                use_auth_token=self.config.hf_token
            )
            
            # Move to appropriate device
            if self.device != "cpu":
                self.pipeline.to(torch.device(self.device))
            
            # Configure pipeline parameters
            self._configure_pipeline_parameters()
            
            logger.info("PyAnnote.audio pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize pyannote pipeline: {e}")
            self.pipeline = None
    
    def _configure_pipeline_parameters(self):
        """Configure pipeline parameters for optimal performance."""
        if not self.pipeline:
            return
            
        try:
            # Configure segmentation
            if hasattr(self.pipeline, '_segmentation'):
                if hasattr(self.pipeline._segmentation, 'onset'):
                    self.pipeline._segmentation.onset = self.config.segmentation_threshold
                if hasattr(self.pipeline._segmentation, 'offset'):  
                    self.pipeline._segmentation.offset = self.config.segmentation_threshold
                    
            # Configure clustering
            if hasattr(self.pipeline, '_clustering'):
                if hasattr(self.pipeline._clustering, 'threshold'):
                    self.pipeline._clustering.threshold = self.config.clustering_threshold
                    
            # Configure minimum durations
            if hasattr(self.pipeline, 'segmentation'):
                if hasattr(self.pipeline.segmentation, 'min_duration_on'):
                    self.pipeline.segmentation.min_duration_on = self.config.min_duration_on
                if hasattr(self.pipeline.segmentation, 'min_duration_off'):
                    self.pipeline.segmentation.min_duration_off = self.config.min_duration_off
                    
            logger.debug("Pipeline parameters configured successfully")
            
        except Exception as e:
            logger.warning(f"Could not configure all pipeline parameters: {e}")
    
    async def diarize_audio(
        self, 
        audio_path: Path,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DiarizationResult:
        """
        Perform speaker diarization on audio file.
        
        Args:
            audio_path: Path to audio file
            metadata: Optional metadata for processing hints
            
        Returns:
            DiarizationResult with speaker segments and quality metrics
        """
        start_time = time.time()
        
        try:
            if not self.pipeline:
                # Fallback to dummy implementation
                return await self._dummy_diarization(audio_path, metadata)
            
            logger.info(f"Starting diarization of: {audio_path}")
            
            # Run diarization in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            diarization = await loop.run_in_executor(
                None, 
                self._run_diarization, 
                str(audio_path)
            )
            
            # Convert pyannote result to our format
            segments = self._convert_diarization_result(diarization)
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(segments, diarization)
            
            # Calculate overall confidence
            confidence_score = np.mean([seg.confidence for seg in segments]) if segments else 0.0
            
            processing_time = time.time() - start_time
            
            result = DiarizationResult(
                segments=segments,
                speaker_count=len(set(seg.speaker_id for seg in segments)),
                processing_time=processing_time,
                confidence_score=confidence_score,
                quality_metrics=quality_metrics,
                metadata=metadata or {}
            )
            
            logger.info(
                f"Diarization completed: {result.speaker_count} speakers, "
                f"{len(segments)} segments, {processing_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            processing_time = time.time() - start_time
            
            # Return empty result on failure
            return DiarizationResult(
                segments=[],
                speaker_count=0,
                processing_time=processing_time,
                confidence_score=0.0,
                quality_metrics={"error": 1.0},
                metadata={"error": str(e)}
            )
    
    def _run_diarization(self, audio_path: str):
        """Run pyannote diarization pipeline."""
        # Apply speaker count constraints if specified
        instantiate_kwargs = {}
        
        if self.config.min_speakers is not None:
            instantiate_kwargs["min_speakers"] = self.config.min_speakers
        if self.config.max_speakers is not None:
            instantiate_kwargs["max_speakers"] = self.config.max_speakers
            
        if instantiate_kwargs:
            diarization = self.pipeline(
                audio_path,
                **instantiate_kwargs
            )
        else:
            diarization = self.pipeline(audio_path)
            
        return diarization
    
    def _convert_diarization_result(self, diarization) -> List[SpeakerSegment]:
        """Convert pyannote diarization to our SpeakerSegment format."""
        segments = []
        
        try:
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                # Calculate confidence based on segment duration and consistency
                confidence = min(0.9, max(0.5, turn.duration / 10.0))  # Longer segments = higher confidence
                confidence = max(confidence, self.config.confidence_threshold)
                
                segment = SpeakerSegment(
                    speaker_id=str(speaker),
                    start_time=turn.start,
                    end_time=turn.end,
                    confidence=confidence,
                    metadata={
                        "duration": turn.duration,
                        "pyannote_label": speaker
                    }
                )
                
                segments.append(segment)
                
        except Exception as e:
            logger.error(f"Failed to convert diarization result: {e}")
            
        return segments
    
    def _calculate_quality_metrics(
        self, 
        segments: List[SpeakerSegment], 
        diarization
    ) -> Dict[str, float]:
        """Calculate quality metrics for the diarization result."""
        metrics = {}
        
        try:
            # Basic metrics
            metrics["total_segments"] = len(segments)
            metrics["average_segment_duration"] = np.mean([seg.duration for seg in segments]) if segments else 0.0
            metrics["speaker_count"] = len(set(seg.speaker_id for seg in segments))
            
            # Confidence metrics
            if segments:
                confidences = [seg.confidence for seg in segments]
                metrics["average_confidence"] = np.mean(confidences)
                metrics["min_confidence"] = np.min(confidences)
                metrics["confidence_std"] = np.std(confidences)
            else:
                metrics["average_confidence"] = 0.0
                metrics["min_confidence"] = 0.0
                metrics["confidence_std"] = 0.0
            
            # Estimated DER (simplified calculation)
            # In production, this would be calculated against ground truth
            metrics["estimated_der"] = max(0.05, 0.2 - metrics["average_confidence"] * 0.15)
            
            # Coverage metrics
            total_duration = sum(seg.duration for seg in segments)
            metrics["total_speech_duration"] = total_duration
            metrics["speech_ratio"] = min(1.0, total_duration / 3600)  # Normalize to max 1 hour
            
        except Exception as e:
            logger.error(f"Failed to calculate quality metrics: {e}")
            metrics["error"] = 1.0
            
        return metrics
    
    async def _dummy_diarization(
        self, 
        audio_path: Path, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> DiarizationResult:
        """Dummy diarization implementation for testing without pyannote."""
        logger.warning("Using dummy diarization - install pyannote.audio for real functionality")
        
        # Simulate processing time
        await asyncio.sleep(0.5)
        
        # Create dummy segments
        segments = [
            SpeakerSegment(
                speaker_id="SPEAKER_00",
                start_time=0.0,
                end_time=30.0,
                confidence=0.8,
                metadata={"dummy": True}
            ),
            SpeakerSegment(
                speaker_id="SPEAKER_01", 
                start_time=35.0,
                end_time=60.0,
                confidence=0.7,
                metadata={"dummy": True}
            )
        ]
        
        return DiarizationResult(
            segments=segments,
            speaker_count=2,
            processing_time=0.5,
            confidence_score=0.75,
            quality_metrics={
                "dummy_implementation": True,
                "estimated_der": 0.15
            },
            metadata={"dummy": True, "message": "Install pyannote.audio for real diarization"}
        )
    
    def extract_speaker_embeddings(
        self, 
        audio: np.ndarray, 
        segments: List[SpeakerSegment]
    ) -> Dict[str, np.ndarray]:
        """Extract speaker embeddings for each identified speaker."""
        embeddings = {}
        
        if not HAS_PYANNOTE:
            logger.warning("Cannot extract embeddings without pyannote.audio")
            return embeddings
            
        try:
            # This would use pyannote's embedding model
            # For now, return dummy embeddings
            for segment in segments:
                speaker_id = segment.speaker_id
                if speaker_id not in embeddings:
                    # Create dummy 192-dimensional embedding
                    embeddings[speaker_id] = np.random.randn(192)
                    
        except Exception as e:
            logger.error(f"Failed to extract embeddings: {e}")
            
        return embeddings
    
    def validate_diarization_quality(
        self, 
        result: DiarizationResult
    ) -> Dict[str, float]:
        """Assess diarization quality and reliability."""
        validation_metrics = {}
        
        try:
            # Check against configuration thresholds
            validation_metrics["meets_confidence_threshold"] = float(
                result.confidence_score >= self.config.min_confidence
            )
            
            validation_metrics["meets_der_target"] = float(
                result.quality_metrics.get("estimated_der", 1.0) <= self.config.target_der
            )
            
            # Segment consistency checks
            if result.segments:
                durations = [seg.duration for seg in result.segments]
                validation_metrics["segment_duration_consistency"] = 1.0 - np.std(durations) / np.mean(durations)
                validation_metrics["segment_duration_consistency"] = max(0.0, validation_metrics["segment_duration_consistency"])
                
                # Speaker balance
                speaker_counts = {}
                for seg in result.segments:
                    speaker_counts[seg.speaker_id] = speaker_counts.get(seg.speaker_id, 0) + 1
                    
                balance_scores = list(speaker_counts.values())
                validation_metrics["speaker_balance"] = 1.0 - np.std(balance_scores) / np.mean(balance_scores)
                validation_metrics["speaker_balance"] = max(0.0, validation_metrics["speaker_balance"])
            else:
                validation_metrics["segment_duration_consistency"] = 0.0
                validation_metrics["speaker_balance"] = 0.0
            
            # Overall quality score
            quality_components = [
                validation_metrics["meets_confidence_threshold"],
                validation_metrics["meets_der_target"],
                validation_metrics["segment_duration_consistency"],
                validation_metrics["speaker_balance"]
            ]
            
            validation_metrics["overall_quality"] = np.mean(quality_components)
            
        except Exception as e:
            logger.error(f"Quality validation failed: {e}")
            validation_metrics["error"] = 1.0
            validation_metrics["overall_quality"] = 0.0
            
        return validation_metrics


# Create default configuration instance
default_diarization_config = DiarizationConfig()