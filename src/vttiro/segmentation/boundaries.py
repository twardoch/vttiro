#!/usr/bin/env python3
# this_file: src/vttiro/segmentation/boundaries.py
"""Advanced boundary detection for linguistic and content-aware segmentation."""

from enum import Enum
from typing import List, Dict, Tuple, Optional
import time

try:
    from loguru import logger
except ImportError:
    import logging as logger

try:
    import numpy as np
    import librosa
    from scipy import signal
    AUDIO_PROCESSING_AVAILABLE = True
except ImportError:
    AUDIO_PROCESSING_AVAILABLE = False
    logger.warning("Advanced audio processing not available")
    np = None
    librosa = None
    signal = None

# Optional advanced libraries
try:
    import webrtcvad
    VAD_AVAILABLE = True
except ImportError:
    VAD_AVAILABLE = False
    webrtcvad = None

from .core import SegmentationConfig


class BoundaryType(Enum):
    """Types of detected boundaries."""
    PAUSE = "pause"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    TOPIC_CHANGE = "topic_change"
    SPEAKER_CHANGE = "speaker_change"
    BREATH = "breath"
    MUSIC_TRANSITION = "music_transition"


class BoundaryDetector:
    """Advanced boundary detector using multiple detection strategies."""
    
    def __init__(self, config: SegmentationConfig):
        """Initialize boundary detector.
        
        Args:
            config: Segmentation configuration
        """
        self.config = config
        
        if not AUDIO_PROCESSING_AVAILABLE:
            logger.warning("Advanced boundary detection requires audio processing libraries")
            
        # Initialize VAD if available
        self.vad_model = None
        if VAD_AVAILABLE and config.enable_vad:
            try:
                self.vad_model = webrtcvad.Vad(2)  # Moderate aggressiveness
                logger.debug("WebRTC VAD initialized for boundary detection")
            except Exception as e:
                logger.warning(f"Failed to initialize VAD: {e}")
                
    def detect_linguistic_boundaries(
        self, 
        audio_data: np.ndarray, 
        sr: int
    ) -> List[float]:
        """Detect linguistic boundaries (sentences, phrases, topics).
        
        Args:
            audio_data: Audio signal
            sr: Sample rate
            
        Returns:
            List of boundary timestamps in seconds
        """
        logger.debug("Detecting linguistic boundaries")
        
        try:
            boundaries = []
            
            # 1. Pause-based boundaries (most reliable)
            pause_boundaries = self._detect_pause_boundaries(audio_data, sr)
            boundaries.extend(pause_boundaries)
            
            # 2. Prosodic boundaries (pitch and rhythm changes)
            prosodic_boundaries = self._detect_prosodic_boundaries(audio_data, sr)
            boundaries.extend(prosodic_boundaries)
            
            # 3. VAD-based boundaries (speech activity changes)
            if self.vad_model:
                vad_boundaries = self._detect_vad_boundaries(audio_data, sr)
                boundaries.extend(vad_boundaries)
                
            # 4. Spectral change boundaries (acoustic transitions)
            spectral_boundaries = self._detect_spectral_change_boundaries(audio_data, sr)
            boundaries.extend(spectral_boundaries)
            
            # Combine and filter boundaries
            combined_boundaries = self._combine_linguistic_boundaries(boundaries)
            
            logger.debug(f"Found {len(combined_boundaries)} linguistic boundaries")
            return combined_boundaries
            
        except Exception as e:
            logger.error(f"Linguistic boundary detection failed: {e}")
            return []
            
    def _detect_pause_boundaries(
        self, 
        audio_data: np.ndarray, 
        sr: int
    ) -> List[Tuple[float, str, float]]:
        """Detect boundaries based on pauses in speech."""
        boundaries = []
        
        if not AUDIO_PROCESSING_AVAILABLE:
            return boundaries
            
        try:
            # Compute RMS energy with shorter frames for better pause detection
            frame_length = 1024
            hop_length = 256
            
            rms_energy = librosa.feature.rms(
                y=audio_data, 
                frame_length=frame_length, 
                hop_length=hop_length
            )[0]
            
            # Convert to dB
            rms_db = 20 * np.log10(rms_energy + 1e-10)
            
            # Detect silence regions
            silence_threshold = np.percentile(rms_db, 15)  # More sensitive than energy-based
            silence_mask = rms_db < silence_threshold
            
            # Find silence regions
            silence_changes = np.diff(silence_mask.astype(int))
            silence_starts = np.where(silence_changes == 1)[0]
            silence_ends = np.where(silence_changes == -1)[0]
            
            # Convert frame indices to time
            time_frames = librosa.frames_to_time(
                np.arange(len(rms_energy)), 
                sr=sr, 
                hop_length=hop_length
            )
            
            # Process significant pauses
            for start, end in zip(silence_starts, silence_ends):
                if end < len(time_frames):
                    pause_duration = time_frames[end] - time_frames[start]
                    
                    # Classify pause types by duration
                    if pause_duration >= 2.0:  # Long pause - likely sentence/paragraph boundary
                        mid_time = (time_frames[start] + time_frames[end]) / 2
                        confidence = min(1.0, pause_duration / 5.0)
                        boundaries.append((mid_time, BoundaryType.SENTENCE.value, confidence))
                    elif pause_duration >= 0.8:  # Medium pause - likely phrase boundary
                        mid_time = (time_frames[start] + time_frames[end]) / 2
                        confidence = min(0.8, pause_duration / 3.0)
                        boundaries.append((mid_time, BoundaryType.PAUSE.value, confidence))
                    elif pause_duration >= 0.3:  # Short pause - breathing or hesitation
                        mid_time = (time_frames[start] + time_frames[end]) / 2
                        confidence = min(0.5, pause_duration / 1.0)
                        boundaries.append((mid_time, BoundaryType.BREATH.value, confidence))
                        
        except Exception as e:
            logger.warning(f"Pause boundary detection failed: {e}")
            
        return boundaries
        
    def _detect_prosodic_boundaries(
        self, 
        audio_data: np.ndarray, 
        sr: int
    ) -> List[Tuple[float, str, float]]:
        """Detect boundaries based on prosodic features (pitch, rhythm)."""
        boundaries = []
        
        if not AUDIO_PROCESSING_AVAILABLE:
            return boundaries
            
        try:
            # Extract pitch contour
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio_data, 
                fmin=librosa.note_to_hz('C2'), 
                fmax=librosa.note_to_hz('C7'),
                sr=sr
            )
            
            # Smooth pitch contour
            f0_smooth = self._smooth_pitch_contour(f0, voiced_flag)
            
            # Detect pitch boundary markers
            pitch_boundaries = self._detect_pitch_boundaries(f0_smooth, sr)
            boundaries.extend(pitch_boundaries)
            
            # Detect rhythm/tempo changes
            tempo_boundaries = self._detect_tempo_boundaries(audio_data, sr)
            boundaries.extend(tempo_boundaries)
            
        except Exception as e:
            logger.warning(f"Prosodic boundary detection failed: {e}")
            
        return boundaries
        
    def _smooth_pitch_contour(
        self, 
        f0: np.ndarray, 
        voiced_flag: np.ndarray
    ) -> np.ndarray:
        """Smooth pitch contour for boundary detection."""
        
        # Replace unvoiced regions with interpolated values
        f0_smooth = f0.copy()
        f0_smooth[~voiced_flag] = np.nan
        
        # Interpolate missing values
        valid_indices = np.where(~np.isnan(f0_smooth))[0]
        if len(valid_indices) > 1:
            f0_smooth = np.interp(
                np.arange(len(f0_smooth)), 
                valid_indices, 
                f0_smooth[valid_indices]
            )
        else:
            f0_smooth = np.full_like(f0_smooth, 200.0)  # Default pitch
            
        return f0_smooth
        
    def _detect_pitch_boundaries(
        self, 
        f0_smooth: np.ndarray, 
        sr: int
    ) -> List[Tuple[float, str, float]]:
        """Detect boundaries based on pitch changes."""
        boundaries = []
        
        if len(f0_smooth) < 10:
            return boundaries
            
        # Compute pitch derivative
        pitch_derivative = np.gradient(f0_smooth)
        
        # Find significant pitch changes
        derivative_threshold = np.std(pitch_derivative) * 2
        significant_changes = np.where(np.abs(pitch_derivative) > derivative_threshold)[0]
        
        # Convert to time and filter
        hop_length = self.config.hop_length
        for idx in significant_changes:
            if idx > 5 and idx < len(f0_smooth) - 5:  # Avoid edges
                timestamp = librosa.frames_to_time(idx, sr=sr, hop_length=hop_length)
                confidence = min(1.0, np.abs(pitch_derivative[idx]) / (derivative_threshold * 3))
                boundaries.append((timestamp, BoundaryType.SENTENCE.value, confidence * 0.6))
                
        return boundaries
        
    def _detect_tempo_boundaries(
        self, 
        audio_data: np.ndarray, 
        sr: int
    ) -> List[Tuple[float, str, float]]:
        """Detect boundaries based on tempo/rhythm changes."""
        boundaries = []
        
        try:
            # Compute onset strength
            onset_envelope = librosa.onset.onset_strength(
                y=audio_data, 
                sr=sr,
                hop_length=self.config.hop_length
            )
            
            # Compute tempo over sliding windows
            window_size = int(10 * sr / self.config.hop_length)  # 10-second windows
            step_size = int(2 * sr / self.config.hop_length)     # 2-second steps
            
            tempos = []
            window_centers = []
            
            for start in range(0, len(onset_envelope) - window_size, step_size):
                end = start + window_size
                window_envelope = onset_envelope[start:end]
                
                # Estimate tempo for this window
                tempo, _ = librosa.beat.beat_track(
                    onset_envelope=window_envelope,
                    sr=sr,
                    hop_length=self.config.hop_length
                )
                
                tempos.append(tempo)
                window_centers.append((start + end) // 2)
                
            # Find significant tempo changes
            if len(tempos) > 2:
                tempo_changes = np.diff(tempos)
                change_threshold = np.std(tempos) * 0.5
                
                significant_tempo_changes = np.where(
                    np.abs(tempo_changes) > change_threshold
                )[0]
                
                for change_idx in significant_tempo_changes:
                    if change_idx < len(window_centers):
                        frame_idx = window_centers[change_idx]
                        timestamp = librosa.frames_to_time(
                            frame_idx, 
                            sr=sr, 
                            hop_length=self.config.hop_length
                        )
                        confidence = min(1.0, np.abs(tempo_changes[change_idx]) / (change_threshold * 2))
                        boundaries.append((timestamp, BoundaryType.TOPIC_CHANGE.value, confidence * 0.4))
                        
        except Exception as e:
            logger.warning(f"Tempo boundary detection failed: {e}")
            
        return boundaries
        
    def _detect_vad_boundaries(
        self, 
        audio_data: np.ndarray, 
        sr: int
    ) -> List[Tuple[float, str, float]]:
        """Detect boundaries using Voice Activity Detection."""
        boundaries = []
        
        if not self.vad_model:
            return boundaries
            
        try:
            # Resample to 16kHz for WebRTC VAD if needed
            if sr != 16000:
                audio_16k = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
            else:
                audio_16k = audio_data
                
            # Convert to 16-bit PCM
            audio_pcm = (audio_16k * 32768).astype(np.int16)
            
            # Process in 30ms frames
            frame_duration_ms = 30
            frame_length = int(16000 * frame_duration_ms / 1000)  # 480 samples
            
            vad_results = []
            
            for i in range(0, len(audio_pcm) - frame_length, frame_length):
                frame = audio_pcm[i:i + frame_length]
                is_speech = self.vad_model.is_speech(frame.tobytes(), 16000)
                vad_results.append(is_speech)
                
            # Find speech/non-speech boundaries
            vad_changes = np.diff(vad_results)
            speech_starts = np.where(vad_changes == 1)[0]  # Non-speech to speech
            speech_ends = np.where(vad_changes == -1)[0]   # Speech to non-speech
            
            # Convert frame indices to timestamps
            for boundary_idx in np.concatenate([speech_starts, speech_ends]):
                timestamp = boundary_idx * frame_duration_ms / 1000.0
                if timestamp < len(audio_data) / sr:
                    boundaries.append((timestamp, BoundaryType.PAUSE.value, 0.7))
                    
        except Exception as e:
            logger.warning(f"VAD boundary detection failed: {e}")
            
        return boundaries
        
    def _detect_spectral_change_boundaries(
        self, 
        audio_data: np.ndarray, 
        sr: int
    ) -> List[Tuple[float, str, float]]:
        """Detect boundaries based on spectral characteristics changes."""
        boundaries = []
        
        if not AUDIO_PROCESSING_AVAILABLE:
            return boundaries
            
        try:
            # Compute spectral features
            spectral_centroid = librosa.feature.spectral_centroid(
                y=audio_data, 
                sr=sr,
                hop_length=self.config.hop_length
            )[0]
            
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio_data, 
                sr=sr,
                hop_length=self.config.hop_length
            )[0]
            
            # Compute spectral contrast
            spectral_contrast = librosa.feature.spectral_contrast(
                y=audio_data, 
                sr=sr,
                hop_length=self.config.hop_length
            )
            
            # Find rapid changes in spectral characteristics
            centroid_changes = np.abs(np.gradient(spectral_centroid))
            rolloff_changes = np.abs(np.gradient(spectral_rolloff))
            contrast_changes = np.abs(np.gradient(np.mean(spectral_contrast, axis=0)))
            
            # Combine spectral change indicators
            combined_changes = (
                centroid_changes / np.std(centroid_changes) +
                rolloff_changes / np.std(rolloff_changes) +
                contrast_changes / np.std(contrast_changes)
            )
            
            # Find peaks in combined changes
            change_threshold = np.percentile(combined_changes, 85)
            peaks, _ = signal.find_peaks(
                combined_changes, 
                height=change_threshold,
                distance=int(2 * sr / self.config.hop_length)  # Minimum 2 seconds apart
            )
            
            # Convert to timestamps
            time_frames = librosa.frames_to_time(
                peaks, 
                sr=sr, 
                hop_length=self.config.hop_length
            )
            
            for timestamp in time_frames:
                confidence = min(1.0, np.max(combined_changes) / (change_threshold * 2))
                boundaries.append((timestamp, BoundaryType.TOPIC_CHANGE.value, confidence * 0.5))
                
        except Exception as e:
            logger.warning(f"Spectral change boundary detection failed: {e}")
            
        return boundaries
        
    def _combine_linguistic_boundaries(
        self, 
        boundaries: List[Tuple[float, str, float]]
    ) -> List[float]:
        """Combine and filter linguistic boundaries from different methods."""
        
        if not boundaries:
            return []
            
        # Sort boundaries by timestamp
        boundaries.sort(key=lambda x: x[0])
        
        # Group nearby boundaries and select best
        merged_boundaries = []
        merge_window = 1.0  # 1 second window
        
        current_group = [boundaries[0]]
        
        for boundary in boundaries[1:]:
            timestamp, boundary_type, confidence = boundary
            
            # Check if close to current group
            if timestamp - current_group[-1][0] <= merge_window:
                current_group.append(boundary)
            else:
                # Process current group
                best_boundary = self._select_best_boundary(current_group)
                merged_boundaries.append(best_boundary)
                current_group = [boundary]
                
        # Process final group
        if current_group:
            best_boundary = self._select_best_boundary(current_group)
            merged_boundaries.append(best_boundary)
            
        return merged_boundaries
        
    def _select_best_boundary(
        self, 
        boundary_group: List[Tuple[float, str, float]]
    ) -> float:
        """Select the best boundary from a group of nearby boundaries."""
        
        if len(boundary_group) == 1:
            return boundary_group[0][0]
            
        # Prioritize boundary types (sentences > pauses > others)
        type_priority = {
            BoundaryType.SENTENCE.value: 3,
            BoundaryType.PARAGRAPH.value: 2.5,
            BoundaryType.PAUSE.value: 2,
            BoundaryType.BREATH.value: 1.5,
            BoundaryType.TOPIC_CHANGE.value: 1,
            BoundaryType.SPEAKER_CHANGE.value: 0.5
        }
        
        # Score each boundary
        scored_boundaries = []
        for timestamp, boundary_type, confidence in boundary_group:
            type_score = type_priority.get(boundary_type, 1.0)
            total_score = confidence * type_score
            scored_boundaries.append((timestamp, total_score))
            
        # Return timestamp of highest scored boundary
        best_boundary = max(scored_boundaries, key=lambda x: x[1])
        return best_boundary[0]