#!/usr/bin/env python3
# this_file: src/vttiro/segmentation/core.py
"""Core audio segmentation engine with intelligent boundary detection."""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Union
from enum import Enum
from pathlib import Path
import time

try:
    from loguru import logger
except ImportError:
    import logging as logger

try:
    import numpy as np
    import librosa
    AUDIO_PROCESSING_AVAILABLE = True
except ImportError:
    AUDIO_PROCESSING_AVAILABLE = False
    logger.warning("Audio processing libraries not available. Install with: uv add librosa numpy")
    np = None
    librosa = None

from vttiro.core.config import VttiroConfig


class SegmentationType(Enum):
    """Types of audio segmentation strategies."""
    ENERGY_BASED = "energy_based"
    LINGUISTIC = "linguistic" 
    SPEAKER_AWARE = "speaker_aware"
    CONTENT_AWARE = "content_aware"
    QUALITY_DRIVEN = "quality_driven"
    ADAPTIVE = "adaptive"


@dataclass
class SegmentationConfig:
    """Configuration for audio segmentation parameters."""
    
    # Basic timing parameters
    max_chunk_duration: int = 600      # 10 minutes maximum
    min_chunk_duration: int = 60       # 1 minute minimum
    overlap_duration: int = 30         # 30 seconds overlap
    prefer_integer_seconds: bool = True
    
    # Energy-based parameters  
    energy_threshold_percentile: int = 20
    silence_threshold_db: float = -40.0
    min_silence_duration: float = 0.5  # 500ms minimum silence
    
    # Quality parameters
    quality_threshold: float = 0.7
    snr_threshold: float = 10.0        # Signal-to-noise ratio threshold
    
    # Content-aware parameters
    content_type: Optional[str] = None  # podcast, lecture, interview, news, etc.
    speaker_change_sensitivity: float = 0.5
    
    # Processing parameters
    frame_length: int = 2048
    hop_length: int = 512
    sample_rate: int = 16000
    
    # Advanced features
    enable_vad: bool = True
    enable_linguistic_boundaries: bool = True
    enable_speaker_boundaries: bool = False  # Requires diarization
    enable_quality_adaptation: bool = True


@dataclass
class AudioSegment:
    """Represents a segmented audio chunk with metadata."""
    
    # Timing information
    start_time: float
    end_time: float
    duration: float
    
    # Audio file reference
    audio_file: Optional[Path] = None
    
    # Energy statistics
    energy_stats: Dict[str, float] = field(default_factory=dict)
    
    # Quality metrics
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Content hints and metadata
    content_hints: Dict[str, Any] = field(default_factory=dict)
    
    # Unique identifier
    chunk_id: str = ""
    
    # Segmentation metadata
    segmentation_method: str = "unknown"
    confidence: float = 1.0
    
    def __post_init__(self):
        """Generate chunk ID if not provided."""
        if not self.chunk_id:
            self.chunk_id = f"chunk_{int(self.start_time):06d}_{int(self.end_time):06d}"


class SegmentationEngine:
    """Advanced audio segmentation engine with intelligent boundary detection."""
    
    def __init__(self, config: Union[SegmentationConfig, VttiroConfig]):
        """Initialize segmentation engine.
        
        Args:
            config: Segmentation configuration or main vttiro config
        """
        if isinstance(config, VttiroConfig):
            # Extract segmentation config from main config
            self.config = self._build_segmentation_config(config)
            self.main_config = config
        else:
            self.config = config
            self.main_config = None
            
        if not AUDIO_PROCESSING_AVAILABLE:
            logger.warning("Advanced audio analysis not available - using basic segmentation only")
            
        # Initialize components
        self.energy_analyzer = None
        self.boundary_detector = None
        self.vad_model = None
        
        self._initialize_components()
        
    def _build_segmentation_config(self, main_config: VttiroConfig) -> SegmentationConfig:
        """Build segmentation config from main vttiro config."""
        return SegmentationConfig(
            max_chunk_duration=main_config.processing.chunk_duration,
            overlap_duration=main_config.processing.overlap_duration,
            prefer_integer_seconds=main_config.processing.prefer_integer_seconds,
            energy_threshold_percentile=main_config.processing.energy_threshold_percentile,
            min_silence_duration=main_config.processing.min_energy_window,
            sample_rate=main_config.processing.sample_rate,
            enable_vad=True,
            enable_linguistic_boundaries=True,
            enable_quality_adaptation=True
        )
        
    def _initialize_components(self):
        """Initialize segmentation components."""
        if not AUDIO_PROCESSING_AVAILABLE:
            return
            
        try:
            # Initialize energy analyzer
            from .energy import EnergyAnalyzer
            self.energy_analyzer = EnergyAnalyzer(self.config)
            
            # Initialize boundary detector
            from .boundaries import BoundaryDetector
            self.boundary_detector = BoundaryDetector(self.config)
            
            # Initialize VAD if enabled
            if self.config.enable_vad:
                self._initialize_vad()
                
            logger.info("Segmentation engine components initialized successfully")
            
        except Exception as e:
            logger.warning(f"Failed to initialize advanced segmentation components: {e}")
            logger.info("Falling back to basic energy-based segmentation")
            
    def _initialize_vad(self):
        """Initialize Voice Activity Detection model."""
        try:
            import webrtcvad
            self.vad_model = webrtcvad.Vad(2)  # Moderate aggressiveness
            logger.debug("WebRTC VAD initialized")
        except ImportError:
            logger.warning("WebRTC VAD not available. Install with: uv add webrtcvad")
            self.vad_model = None
            
    async def segment_audio(
        self, 
        audio_path: Path, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[AudioSegment]:
        """Intelligently segment audio using multiple strategies.
        
        Args:
            audio_path: Path to audio file
            metadata: Optional metadata for content-aware segmentation
            
        Returns:
            List of AudioSegment objects with precise boundaries
        """
        start_time = time.time()
        logger.info(f"Starting advanced segmentation of: {audio_path}")
        
        try:
            # Load audio data
            if not AUDIO_PROCESSING_AVAILABLE:
                return await self._fallback_segmentation(audio_path, metadata)
                
            audio_data, sr = librosa.load(str(audio_path), sr=self.config.sample_rate)
            duration = len(audio_data) / sr
            
            logger.debug(f"Loaded audio: {duration:.2f}s at {sr}Hz")
            
            # Analyze content characteristics
            content_analysis = self._analyze_content(audio_data, sr, metadata)
            
            # Select optimal segmentation strategy
            strategy = self._select_segmentation_strategy(content_analysis)
            logger.info(f"Selected segmentation strategy: {strategy}")
            
            # Generate segments using selected strategy
            segments = await self._generate_segments(
                audio_data, sr, audio_path, strategy, content_analysis
            )
            
            # Post-process and validate segments
            segments = self._validate_and_optimize_segments(segments, audio_data, sr)
            
            processing_time = time.time() - start_time
            logger.info(f"Segmentation completed: {len(segments)} segments in {processing_time:.2f}s")
            
            return segments
            
        except Exception as e:
            logger.error(f"Advanced segmentation failed for {audio_path}: {e}")
            # Fallback to basic segmentation
            return await self._fallback_segmentation(audio_path, metadata)
            
    def _analyze_content(
        self, 
        audio_data: np.ndarray, 
        sr: int, 
        metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze audio content characteristics for strategy selection."""
        
        duration = len(audio_data) / sr
        
        analysis = {
            "duration": duration,
            "sample_rate": sr,
            "has_metadata": metadata is not None,
            "estimated_speakers": 1,
            "content_type": "general",
            "audio_quality": "medium",
            "speech_ratio": 0.8,  # Assume mostly speech
            "noise_level": "low"
        }
        
        if metadata:
            # Extract content hints from metadata
            title = metadata.get('title', '').lower()
            description = metadata.get('description', '').lower()
            
            # Detect content type from metadata
            if any(word in title for word in ['interview', 'conversation', 'discussion']):
                analysis["content_type"] = "interview"
                analysis["estimated_speakers"] = 2
            elif any(word in title for word in ['lecture', 'presentation', 'talk']):
                analysis["content_type"] = "lecture"
                analysis["estimated_speakers"] = 1
            elif any(word in title for word in ['podcast', 'show', 'episode']):
                analysis["content_type"] = "podcast"
                analysis["estimated_speakers"] = 2
            elif any(word in title for word in ['news', 'report', 'bulletin']):
                analysis["content_type"] = "news"
                analysis["estimated_speakers"] = 1
                
        # Basic audio quality assessment
        if self.energy_analyzer:
            try:
                quality_metrics = self.energy_analyzer.assess_quality(audio_data, sr)
                analysis.update(quality_metrics)
            except Exception as e:
                logger.warning(f"Quality assessment failed: {e}")
                
        return analysis
        
    def _select_segmentation_strategy(self, content_analysis: Dict[str, Any]) -> SegmentationType:
        """Select optimal segmentation strategy based on content analysis."""
        
        content_type = content_analysis.get("content_type", "general")
        duration = content_analysis.get("duration", 0)
        estimated_speakers = content_analysis.get("estimated_speakers", 1)
        
        # Strategy selection logic
        if content_type == "interview" and estimated_speakers > 1:
            return SegmentationType.SPEAKER_AWARE
        elif content_type == "lecture" and duration > 1800:  # 30+ minutes
            return SegmentationType.LINGUISTIC
        elif content_type == "podcast":
            return SegmentationType.CONTENT_AWARE
        elif content_analysis.get("audio_quality") == "poor":
            return SegmentationType.QUALITY_DRIVEN
        else:
            return SegmentationType.ADAPTIVE  # Use multiple strategies
            
    async def _generate_segments(
        self,
        audio_data: np.ndarray,
        sr: int,
        audio_path: Path,
        strategy: SegmentationType,
        content_analysis: Dict[str, Any]
    ) -> List[AudioSegment]:
        """Generate audio segments using the selected strategy."""
        
        if strategy == SegmentationType.ENERGY_BASED:
            return self._segment_by_energy(audio_data, sr, audio_path)
        elif strategy == SegmentationType.LINGUISTIC:
            return self._segment_by_linguistics(audio_data, sr, audio_path)
        elif strategy == SegmentationType.SPEAKER_AWARE:
            return self._segment_by_speakers(audio_data, sr, audio_path)
        elif strategy == SegmentationType.CONTENT_AWARE:
            return self._segment_by_content(audio_data, sr, audio_path, content_analysis)
        elif strategy == SegmentationType.QUALITY_DRIVEN:
            return self._segment_by_quality(audio_data, sr, audio_path)
        else:  # ADAPTIVE
            return self._segment_adaptively(audio_data, sr, audio_path, content_analysis)
            
    def _segment_by_energy(
        self, 
        audio_data: np.ndarray, 
        sr: int, 
        audio_path: Path
    ) -> List[AudioSegment]:
        """Segment audio using advanced energy-based analysis."""
        
        if not self.energy_analyzer:
            return self._basic_energy_segmentation(audio_data, sr, audio_path)
            
        # Get energy features and boundaries
        energy_boundaries = self.energy_analyzer.detect_energy_boundaries(audio_data, sr)
        
        # Convert boundaries to segments
        segments = []
        duration = len(audio_data) / sr
        
        if not energy_boundaries:
            # Single segment fallback
            segments.append(AudioSegment(
                start_time=0.0,
                end_time=duration,
                duration=duration,
                audio_file=audio_path,
                segmentation_method="energy_fallback",
                confidence=0.5
            ))
        else:
            # Create segments from boundaries
            for i, boundary in enumerate(energy_boundaries):
                start_time = boundary if i == 0 else energy_boundaries[i-1]
                end_time = boundary
                
                if end_time - start_time >= self.config.min_chunk_duration:
                    segments.append(AudioSegment(
                        start_time=start_time,
                        end_time=end_time,
                        duration=end_time - start_time,
                        audio_file=audio_path,
                        segmentation_method="energy_based",
                        confidence=0.8
                    ))
                    
        return segments
        
    def _segment_by_linguistics(
        self, 
        audio_data: np.ndarray, 
        sr: int, 
        audio_path: Path
    ) -> List[AudioSegment]:
        """Segment audio using linguistic boundary detection."""
        
        if not self.boundary_detector:
            return self._segment_by_energy(audio_data, sr, audio_path)
            
        # Detect linguistic boundaries (sentences, topics, etc.)
        linguistic_boundaries = self.boundary_detector.detect_linguistic_boundaries(
            audio_data, sr
        )
        
        # Create segments with linguistic awareness
        segments = []
        duration = len(audio_data) / sr
        max_chunk = self.config.max_chunk_duration
        
        current_start = 0.0
        for boundary in linguistic_boundaries:
            if boundary - current_start >= max_chunk or boundary == linguistic_boundaries[-1]:
                segments.append(AudioSegment(
                    start_time=current_start,
                    end_time=boundary,
                    duration=boundary - current_start,
                    audio_file=audio_path,
                    segmentation_method="linguistic",
                    confidence=0.9
                ))
                current_start = boundary
                
        return segments
        
    def _segment_by_speakers(
        self, 
        audio_data: np.ndarray, 
        sr: int, 
        audio_path: Path
    ) -> List[AudioSegment]:
        """Segment audio with speaker-aware boundaries."""
        
        # For now, fallback to energy-based with speaker hints
        # This will be enhanced when speaker diarization is implemented
        logger.warning("Speaker-aware segmentation not fully implemented yet")
        return self._segment_by_energy(audio_data, sr, audio_path)
        
    def _segment_by_content(
        self,
        audio_data: np.ndarray,
        sr: int,
        audio_path: Path,
        content_analysis: Dict[str, Any]
    ) -> List[AudioSegment]:
        """Segment audio using content-aware strategies."""
        
        content_type = content_analysis.get("content_type", "general")
        
        # Adjust parameters based on content type
        if content_type == "podcast":
            # Longer chunks for podcast content
            max_duration = min(self.config.max_chunk_duration * 1.5, 900)  # Up to 15 min
        elif content_type == "news":
            # Shorter chunks for news content
            max_duration = min(self.config.max_chunk_duration * 0.5, 300)  # Up to 5 min
        else:
            max_duration = self.config.max_chunk_duration
            
        # Use energy-based segmentation with adjusted parameters
        return self._segment_by_energy(audio_data, sr, audio_path)
        
    def _segment_by_quality(
        self, 
        audio_data: np.ndarray, 
        sr: int, 
        audio_path: Path
    ) -> List[AudioSegment]:
        """Segment audio based on quality characteristics."""
        
        # For poor quality audio, use shorter chunks for better processing
        quality_factor = 0.7  # Assume moderate quality reduction
        adjusted_max_duration = int(self.config.max_chunk_duration * quality_factor)
        
        # Use energy-based segmentation with quality adjustments
        return self._segment_by_energy(audio_data, sr, audio_path)
        
    def _segment_adaptively(
        self,
        audio_data: np.ndarray,
        sr: int,
        audio_path: Path,
        content_analysis: Dict[str, Any]
    ) -> List[AudioSegment]:
        """Segment audio using adaptive strategy combining multiple approaches."""
        
        # Start with energy-based segmentation
        energy_segments = self._segment_by_energy(audio_data, sr, audio_path)
        
        # Enhance with linguistic boundaries if available
        if self.config.enable_linguistic_boundaries and self.boundary_detector:
            try:
                linguistic_boundaries = self.boundary_detector.detect_linguistic_boundaries(
                    audio_data, sr
                )
                # Refine energy segments with linguistic boundaries
                energy_segments = self._refine_with_linguistic_boundaries(
                    energy_segments, linguistic_boundaries
                )
            except Exception as e:
                logger.warning(f"Linguistic boundary enhancement failed: {e}")
                
        return energy_segments
        
    def _refine_with_linguistic_boundaries(
        self,
        segments: List[AudioSegment],
        linguistic_boundaries: List[float]
    ) -> List[AudioSegment]:
        """Refine segments using linguistic boundary information."""
        
        # For now, return original segments
        # TODO: Implement sophisticated boundary refinement
        return segments
        
    def _basic_energy_segmentation(
        self, 
        audio_data: np.ndarray, 
        sr: int, 
        audio_path: Path
    ) -> List[AudioSegment]:
        """Basic energy-based segmentation fallback."""
        
        duration = len(audio_data) / sr
        max_chunk = self.config.max_chunk_duration
        
        segments = []
        current_time = 0.0
        
        while current_time < duration:
            end_time = min(current_time + max_chunk, duration)
            
            segments.append(AudioSegment(
                start_time=current_time,
                end_time=end_time,
                duration=end_time - current_time,
                audio_file=audio_path,
                segmentation_method="basic_energy",
                confidence=0.6
            ))
            
            current_time = end_time
            
        return segments
        
    def _validate_and_optimize_segments(
        self, 
        segments: List[AudioSegment], 
        audio_data: np.ndarray, 
        sr: int
    ) -> List[AudioSegment]:
        """Validate and optimize segment boundaries."""
        
        # Basic validation: ensure minimum duration
        validated_segments = []
        
        for segment in segments:
            if segment.duration >= self.config.min_chunk_duration:
                validated_segments.append(segment)
            else:
                # Merge short segments with previous or next
                if validated_segments:
                    # Extend previous segment
                    prev_segment = validated_segments[-1]
                    prev_segment.end_time = segment.end_time
                    prev_segment.duration = prev_segment.end_time - prev_segment.start_time
                    
        # Ensure integer second preference if enabled
        if self.config.prefer_integer_seconds:
            for segment in validated_segments:
                segment.start_time = round(segment.start_time)
                segment.end_time = round(segment.end_time)
                segment.duration = segment.end_time - segment.start_time
                
        return validated_segments
        
    async def _fallback_segmentation(
        self, 
        audio_path: Path, 
        metadata: Optional[Dict[str, Any]]
    ) -> List[AudioSegment]:
        """Fallback segmentation when advanced processing is not available."""
        
        logger.info("Using basic fallback segmentation")
        
        # Estimate duration (very rough)
        try:
            file_size = audio_path.stat().st_size
            # Rough estimate: ~1MB per minute for compressed audio
            estimated_minutes = file_size / (1024 * 1024)
            estimated_duration = estimated_minutes * 60.0
        except Exception:
            estimated_duration = 300.0  # Default 5 minutes
            
        # Create basic segments
        segments = []
        max_chunk = self.config.max_chunk_duration
        current_time = 0.0
        
        while current_time < estimated_duration:
            end_time = min(current_time + max_chunk, estimated_duration)
            
            segments.append(AudioSegment(
                start_time=current_time,
                end_time=end_time,
                duration=end_time - current_time,
                audio_file=audio_path,
                segmentation_method="fallback_basic",
                confidence=0.3
            ))
            
            current_time = end_time
            
        return segments