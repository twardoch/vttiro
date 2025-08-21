#!/usr/bin/env python3
# this_file: src/vttiro/segmentation/energy.py
"""Advanced energy analysis for audio segmentation boundary detection."""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import time

try:
    from loguru import logger
except ImportError:
    import logging as logger

try:
    import numpy as np
    import librosa
    from scipy import signal, stats
    from scipy.ndimage import gaussian_filter1d
    AUDIO_PROCESSING_AVAILABLE = True
except ImportError:
    AUDIO_PROCESSING_AVAILABLE = False
    logger.warning("Advanced audio processing not available")
    np = None
    librosa = None
    signal = None
    stats = None
    gaussian_filter1d = None

from .core import SegmentationConfig


@dataclass
class EnergyFeatures:
    """Container for multi-scale energy features."""
    
    # Basic energy features
    rms_energy: np.ndarray = field(default_factory=lambda: np.array([]))
    spectral_centroid: np.ndarray = field(default_factory=lambda: np.array([]))
    zero_crossing_rate: np.ndarray = field(default_factory=lambda: np.array([]))
    spectral_rolloff: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Advanced features
    spectral_flux: np.ndarray = field(default_factory=lambda: np.array([]))
    mfcc: np.ndarray = field(default_factory=lambda: np.array([]))
    chroma: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Temporal features
    energy_envelope: np.ndarray = field(default_factory=lambda: np.array([]))
    energy_derivative: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Quality metrics
    snr_estimate: float = 0.0
    dynamic_range: float = 0.0
    noise_floor: float = 0.0
    
    # Time axis
    time_frames: np.ndarray = field(default_factory=lambda: np.array([]))


class EnergyAnalyzer:
    """Advanced energy analyzer for intelligent boundary detection."""
    
    def __init__(self, config: SegmentationConfig):
        """Initialize energy analyzer.
        
        Args:
            config: Segmentation configuration
        """
        self.config = config
        
        if not AUDIO_PROCESSING_AVAILABLE:
            raise ImportError("Audio processing libraries required for EnergyAnalyzer")
            
    def compute_energy_features(
        self, 
        audio_data: np.ndarray, 
        sr: int
    ) -> EnergyFeatures:
        """Compute comprehensive energy features for boundary detection.
        
        Args:
            audio_data: Audio signal
            sr: Sample rate
            
        Returns:
            EnergyFeatures object with computed features
        """
        logger.debug("Computing multi-scale energy features")
        start_time = time.time()
        
        frame_length = self.config.frame_length
        hop_length = self.config.hop_length
        
        features = EnergyFeatures()
        
        try:
            # Basic energy features
            features.rms_energy = librosa.feature.rms(
                y=audio_data, 
                frame_length=frame_length, 
                hop_length=hop_length
            )[0]
            
            features.spectral_centroid = librosa.feature.spectral_centroid(
                y=audio_data, 
                sr=sr, 
                hop_length=hop_length
            )[0]
            
            features.zero_crossing_rate = librosa.feature.zero_crossing_rate(
                audio_data, 
                frame_length=frame_length, 
                hop_length=hop_length
            )[0]
            
            features.spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio_data, 
                sr=sr, 
                hop_length=hop_length,
                roll_percent=0.85
            )[0]
            
            # Advanced spectral features
            stft = librosa.stft(audio_data, hop_length=hop_length)
            magnitude = np.abs(stft)
            
            # Spectral flux (change in spectrum over time)
            features.spectral_flux = self._compute_spectral_flux(magnitude)
            
            # MFCC features
            features.mfcc = librosa.feature.mfcc(
                y=audio_data, 
                sr=sr, 
                n_mfcc=13, 
                hop_length=hop_length
            )
            
            # Chroma features for harmonic content
            features.chroma = librosa.feature.chroma_stft(
                S=magnitude, 
                sr=sr, 
                hop_length=hop_length
            )
            
            # Temporal envelope features
            features.energy_envelope = self._compute_energy_envelope(features.rms_energy)
            features.energy_derivative = self._compute_energy_derivative(features.rms_energy)
            
            # Quality metrics
            features.snr_estimate = self._estimate_snr(audio_data)
            features.dynamic_range = self._compute_dynamic_range(features.rms_energy)
            features.noise_floor = self._estimate_noise_floor(audio_data)
            
            # Time axis
            features.time_frames = librosa.frames_to_time(
                np.arange(len(features.rms_energy)), 
                sr=sr, 
                hop_length=hop_length
            )
            
            processing_time = time.time() - start_time
            logger.debug(f"Energy features computed in {processing_time:.3f}s")
            
            return features
            
        except Exception as e:
            logger.error(f"Failed to compute energy features: {e}")
            raise
            
    def _compute_spectral_flux(self, magnitude: np.ndarray) -> np.ndarray:
        """Compute spectral flux (rate of change in spectrum)."""
        if magnitude.shape[1] < 2:
            return np.zeros(1)
            
        # Compute differences between consecutive frames
        diff = np.diff(magnitude, axis=1)
        
        # Sum positive differences (increasing energy)
        flux = np.sum(np.maximum(0, diff), axis=0)
        
        # Pad to match frame count
        flux = np.pad(flux, (1, 0), mode='constant', constant_values=0)
        
        return flux
        
    def _compute_energy_envelope(self, rms_energy: np.ndarray) -> np.ndarray:
        """Compute smoothed energy envelope."""
        if len(rms_energy) == 0:
            return np.array([])
            
        # Apply Gaussian smoothing to create envelope
        sigma = max(1.0, len(rms_energy) * 0.01)  # 1% of signal length
        envelope = gaussian_filter1d(rms_energy, sigma=sigma)
        
        return envelope
        
    def _compute_energy_derivative(self, rms_energy: np.ndarray) -> np.ndarray:
        """Compute energy derivative for detecting rapid changes."""
        if len(rms_energy) < 2:
            return np.array([])
            
        # Compute first derivative
        derivative = np.gradient(rms_energy)
        
        return derivative
        
    def _estimate_snr(self, audio_data: np.ndarray) -> float:
        """Estimate signal-to-noise ratio."""
        try:
            # Simple SNR estimation using percentiles
            signal_power = np.percentile(np.abs(audio_data), 95) ** 2
            noise_power = np.percentile(np.abs(audio_data), 5) ** 2
            
            if noise_power > 0:
                snr_ratio = signal_power / noise_power
                snr_db = 10 * np.log10(snr_ratio)
                return float(snr_db)
            else:
                return 60.0  # Very high SNR if no detected noise
                
        except Exception:
            return 20.0  # Default reasonable SNR
            
    def _compute_dynamic_range(self, rms_energy: np.ndarray) -> float:
        """Compute dynamic range of the signal."""
        if len(rms_energy) == 0:
            return 0.0
            
        # Convert to dB
        rms_db = 20 * np.log10(rms_energy + 1e-10)  # Small epsilon to avoid log(0)
        
        # Dynamic range as difference between max and min
        dynamic_range = float(np.max(rms_db) - np.min(rms_db))
        
        return dynamic_range
        
    def _estimate_noise_floor(self, audio_data: np.ndarray) -> float:
        """Estimate noise floor level."""
        try:
            # Use the quietest 10% of the signal as noise estimate
            sorted_abs = np.sort(np.abs(audio_data))
            noise_samples = sorted_abs[:int(len(sorted_abs) * 0.1)]
            noise_floor = float(np.mean(noise_samples))
            
            # Convert to dB
            noise_floor_db = 20 * np.log10(noise_floor + 1e-10)
            
            return noise_floor_db
            
        except Exception:
            return -40.0  # Default noise floor
            
    def detect_energy_boundaries(
        self, 
        audio_data: np.ndarray, 
        sr: int
    ) -> List[float]:
        """Detect optimal boundaries using energy-based analysis.
        
        Args:
            audio_data: Audio signal
            sr: Sample rate
            
        Returns:
            List of boundary timestamps in seconds
        """
        logger.debug("Detecting energy-based boundaries")
        
        try:
            # Compute energy features
            features = self.compute_energy_features(audio_data, sr)
            
            # Find candidate boundaries using multiple criteria
            energy_boundaries = self._find_energy_minima(features)
            spectral_boundaries = self._find_spectral_changes(features)
            silence_boundaries = self._find_silence_boundaries(features)
            
            # Combine and score boundaries
            all_boundaries = self._combine_boundary_candidates(
                energy_boundaries, spectral_boundaries, silence_boundaries
            )
            
            # Filter boundaries by quality and constraints
            final_boundaries = self._filter_boundaries(
                all_boundaries, audio_data, sr
            )
            
            logger.debug(f"Found {len(final_boundaries)} energy-based boundaries")
            return final_boundaries
            
        except Exception as e:
            logger.error(f"Energy boundary detection failed: {e}")
            return []
            
    def _find_energy_minima(self, features: EnergyFeatures) -> List[Tuple[float, float]]:
        """Find local minima in energy signal as boundary candidates."""
        boundaries = []
        
        if len(features.rms_energy) < 3:
            return boundaries
            
        # Find local minima in RMS energy
        minima_indices = signal.argrelmin(features.rms_energy, order=5)[0]
        
        # Filter by energy threshold
        energy_threshold = np.percentile(
            features.rms_energy, 
            self.config.energy_threshold_percentile
        )
        
        for idx in minima_indices:
            if features.rms_energy[idx] <= energy_threshold:
                timestamp = float(features.time_frames[idx])
                confidence = 1.0 - (features.rms_energy[idx] / np.max(features.rms_energy))
                boundaries.append((timestamp, confidence))
                
        return boundaries
        
    def _find_spectral_changes(self, features: EnergyFeatures) -> List[Tuple[float, float]]:
        """Find boundaries based on spectral changes."""
        boundaries = []
        
        if len(features.spectral_flux) < 3:
            return boundaries
            
        # Find peaks in spectral flux (significant spectral changes)
        flux_threshold = np.percentile(features.spectral_flux, 80)
        peaks, _ = signal.find_peaks(
            features.spectral_flux, 
            height=flux_threshold,
            distance=int(sr * 2 / self.config.hop_length)  # Minimum 2 seconds apart
        )
        
        for peak in peaks:
            if peak < len(features.time_frames):
                timestamp = float(features.time_frames[peak])
                confidence = features.spectral_flux[peak] / np.max(features.spectral_flux)
                boundaries.append((timestamp, confidence * 0.7))  # Lower weight than energy
                
        return boundaries
        
    def _find_silence_boundaries(self, features: EnergyFeatures) -> List[Tuple[float, float]]:
        """Find boundaries in silence regions."""
        boundaries = []
        
        if len(features.rms_energy) == 0:
            return boundaries
            
        # Convert to dB
        rms_db = 20 * np.log10(features.rms_energy + 1e-10)
        
        # Find silence regions
        silence_mask = rms_db < self.config.silence_threshold_db
        
        # Find silence boundaries (transitions from/to silence)
        silence_changes = np.diff(silence_mask.astype(int))
        silence_starts = np.where(silence_changes == 1)[0]
        silence_ends = np.where(silence_changes == -1)[0]
        
        # Process silence regions
        for start, end in zip(silence_starts, silence_ends):
            if end < len(features.time_frames):
                silence_duration = features.time_frames[end] - features.time_frames[start]
                
                # Only consider significant silence periods
                if silence_duration >= self.config.min_silence_duration:
                    # Use middle of silence as boundary
                    mid_idx = (start + end) // 2
                    if mid_idx < len(features.time_frames):
                        timestamp = float(features.time_frames[mid_idx])
                        confidence = min(1.0, silence_duration / 2.0)  # Higher confidence for longer silence
                        boundaries.append((timestamp, confidence))
                        
        return boundaries
        
    def _combine_boundary_candidates(
        self, 
        energy_boundaries: List[Tuple[float, float]],
        spectral_boundaries: List[Tuple[float, float]],
        silence_boundaries: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """Combine boundary candidates from different methods."""
        
        # Combine all boundaries
        all_boundaries = energy_boundaries + spectral_boundaries + silence_boundaries
        
        if not all_boundaries:
            return []
            
        # Sort by timestamp
        all_boundaries.sort(key=lambda x: x[0])
        
        # Merge nearby boundaries (within 2 seconds)
        merged_boundaries = []
        merge_window = 2.0  # seconds
        
        current_group = [all_boundaries[0]]
        
        for boundary in all_boundaries[1:]:
            timestamp, confidence = boundary
            
            # Check if close to current group
            if timestamp - current_group[-1][0] <= merge_window:
                current_group.append(boundary)
            else:
                # Finalize current group and start new one
                merged_boundaries.append(self._merge_boundary_group(current_group))
                current_group = [boundary]
                
        # Add final group
        if current_group:
            merged_boundaries.append(self._merge_boundary_group(current_group))
            
        return merged_boundaries
        
    def _merge_boundary_group(self, boundaries: List[Tuple[float, float]]) -> Tuple[float, float]:
        """Merge a group of nearby boundaries."""
        if len(boundaries) == 1:
            return boundaries[0]
            
        # Weight by confidence
        total_weight = sum(conf for _, conf in boundaries)
        if total_weight == 0:
            # Simple average if no confidence
            avg_time = sum(time for time, _ in boundaries) / len(boundaries)
            return (avg_time, 0.5)
        else:
            # Weighted average
            weighted_time = sum(time * conf for time, conf in boundaries) / total_weight
            max_confidence = max(conf for _, conf in boundaries)
            return (weighted_time, max_confidence)
            
    def _filter_boundaries(
        self, 
        boundaries: List[Tuple[float, float]], 
        audio_data: np.ndarray, 
        sr: int
    ) -> List[float]:
        """Filter boundaries based on constraints and quality."""
        
        if not boundaries:
            return []
            
        duration = len(audio_data) / sr
        max_chunk = self.config.max_chunk_duration
        min_chunk = self.config.min_chunk_duration
        
        # Sort by timestamp
        boundaries.sort(key=lambda x: x[0])
        
        # Filter boundaries
        filtered = []
        last_boundary = 0.0
        
        for timestamp, confidence in boundaries:
            # Skip if too close to start
            if timestamp < min_chunk:
                continue
                
            # Skip if too close to end
            if timestamp > duration - min_chunk:
                continue
                
            # Skip if would create too short segment
            if timestamp - last_boundary < min_chunk:
                continue
                
            # Skip if low confidence
            if confidence < 0.3:
                continue
                
            # Add boundary if it creates reasonable chunk size
            next_boundary_time = duration
            for future_time, _ in boundaries:
                if future_time > timestamp:
                    next_boundary_time = future_time
                    break
                    
            if next_boundary_time - timestamp <= max_chunk:
                filtered.append(timestamp)
                last_boundary = timestamp
                
        # Ensure integer seconds if preferred
        if self.config.prefer_integer_seconds:
            filtered = [round(t) for t in filtered]
            
        return filtered
        
    def assess_quality(self, audio_data: np.ndarray, sr: int) -> Dict[str, float]:
        """Assess overall audio quality metrics.
        
        Args:
            audio_data: Audio signal
            sr: Sample rate
            
        Returns:
            Dictionary of quality metrics
        """
        try:
            features = self.compute_energy_features(audio_data, sr)
            
            quality_metrics = {
                "snr_db": features.snr_estimate,
                "dynamic_range_db": features.dynamic_range,
                "noise_floor_db": features.noise_floor,
                "overall_quality": self._compute_overall_quality_score(features)
            }
            
            # Classify quality levels
            if quality_metrics["snr_db"] > 25 and quality_metrics["dynamic_range_db"] > 40:
                quality_metrics["quality_level"] = "excellent"
            elif quality_metrics["snr_db"] > 15 and quality_metrics["dynamic_range_db"] > 25:
                quality_metrics["quality_level"] = "good"
            elif quality_metrics["snr_db"] > 10 and quality_metrics["dynamic_range_db"] > 15:
                quality_metrics["quality_level"] = "fair"
            else:
                quality_metrics["quality_level"] = "poor"
                
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return {
                "snr_db": 15.0,
                "dynamic_range_db": 25.0,
                "noise_floor_db": -30.0,
                "quality_level": "unknown",
                "overall_quality": 0.5
            }
            
    def _compute_overall_quality_score(self, features: EnergyFeatures) -> float:
        """Compute overall quality score (0-1)."""
        
        # Normalize metrics to 0-1 scale
        snr_score = min(1.0, max(0.0, (features.snr_estimate - 5) / 35))  # 5-40 dB range
        dr_score = min(1.0, max(0.0, features.dynamic_range / 60))  # 0-60 dB range
        noise_score = min(1.0, max(0.0, (features.noise_floor + 60) / 40))  # -60 to -20 dB range
        
        # Weighted combination
        overall_score = (0.4 * snr_score + 0.3 * dr_score + 0.3 * noise_score)
        
        return float(overall_score)