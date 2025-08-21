#!/usr/bin/env python3
# this_file: src/vttiro/core/transcription.py
"""Core transcription functionality and base classes."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import asyncio

try:
    from loguru import logger
except ImportError:
    import logging as logger

from vttiro.core.config import VttiroConfig, TranscriptionResult


class TranscriptionEngine(ABC):
    """Abstract base class for transcription engines."""
    
    def __init__(self, config: VttiroConfig):
        self.config = config
        
    @abstractmethod
    async def transcribe(
        self, 
        audio_path: Path, 
        language: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> TranscriptionResult:
        """Transcribe audio file.
        
        Args:
            audio_path: Path to audio file
            language: Language code (optional)
            context: Additional context for transcription
            
        Returns:
            TranscriptionResult with transcribed text and metadata
        """
        pass
        
    @abstractmethod
    def estimate_cost(self, duration_seconds: float) -> float:
        """Estimate processing cost in USD."""
        pass
        
    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """Return list of supported language codes."""
        pass
        
    @property
    @abstractmethod
    def name(self) -> str:
        """Return engine name."""
        pass


class MockTranscriptionEngine(TranscriptionEngine):
    """Mock transcription engine for testing and development."""
    
    @property
    def name(self) -> str:
        return "mock"
        
    async def transcribe(
        self, 
        audio_path: Path, 
        language: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> TranscriptionResult:
        """Mock transcription implementation."""
        logger.info(f"Mock transcribing: {audio_path}")
        
        # Simulate processing time
        await asyncio.sleep(1)
        
        return TranscriptionResult(
            text=f"Mock transcription of {audio_path.name}",
            confidence=0.95,
            word_timestamps=[
                {"word": "Mock", "start": 0.0, "end": 0.5, "confidence": 0.98},
                {"word": "transcription", "start": 0.5, "end": 1.5, "confidence": 0.92},
            ],
            processing_time=1.0,
            model_name="mock-v1",
            language=language or "en",
            metadata={"source": str(audio_path)}
        )
        
    def estimate_cost(self, duration_seconds: float) -> float:
        """Mock cost estimation (free)."""
        return 0.0
        
    def get_supported_languages(self) -> List[str]:
        """Return mock supported languages."""
        return ["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"]


class TranscriptionEnsemble:
    """Ensemble of multiple transcription engines with intelligent routing and result fusion."""
    
    def __init__(self, engines: List[TranscriptionEngine], config: VttiroConfig):
        self.engines = engines
        self.config = config
        
    async def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> TranscriptionResult:
        """Transcribe using intelligent engine selection and ensemble methods."""
        
        if not self.engines:
            raise ValueError("No transcription engines available")
            
        # Select optimal engine based on context and requirements
        selected_engine = self._select_optimal_engine(audio_path, language, context)
        
        logger.info(f"Selected {selected_engine.name} engine for transcription")
        
        try:
            # Primary transcription with selected engine
            result = await selected_engine.transcribe(audio_path, language, context)
            
            # Add ensemble metadata
            result.metadata = result.metadata or {}
            result.metadata["ensemble"] = True
            result.metadata["engines_used"] = [selected_engine.name]
            result.metadata["selection_strategy"] = "intelligent_routing"
            
            # TODO: Implement multi-engine ensemble for critical content
            # For now, use single best engine to maximize accuracy and minimize cost
            
            return result
            
        except Exception as e:
            logger.error(f"Primary engine {selected_engine.name} failed: {e}")
            
            # Try fallback engines
            for engine in self.engines:
                if engine != selected_engine:
                    try:
                        logger.info(f"Trying fallback engine: {engine.name}")
                        result = await engine.transcribe(audio_path, language, context)
                        
                        # Mark as fallback
                        result.metadata = result.metadata or {}
                        result.metadata["ensemble"] = True
                        result.metadata["engines_used"] = [engine.name]
                        result.metadata["selection_strategy"] = "fallback"
                        result.metadata["primary_failure"] = selected_engine.name
                        
                        return result
                        
                    except Exception as fallback_error:
                        logger.warning(f"Fallback engine {engine.name} also failed: {fallback_error}")
                        continue
                        
            # All engines failed
            raise RuntimeError("All transcription engines failed")
            
    def _select_optimal_engine(
        self, 
        audio_path: Path, 
        language: Optional[str], 
        context: Optional[Dict[str, Any]]
    ) -> TranscriptionEngine:
        """Select optimal transcription engine based on content analysis."""
        
        if len(self.engines) == 1:
            return self.engines[0]
            
        # Analyze context for engine selection hints
        content_analysis = self._analyze_content(audio_path, language, context)
        
        # Score engines based on content characteristics
        engine_scores = []
        
        for engine in self.engines:
            score = self._score_engine_for_content(engine, content_analysis)
            engine_scores.append((engine, score))
            
        # Sort by score (highest first)
        engine_scores.sort(key=lambda x: x[1], reverse=True)
        
        selected_engine = engine_scores[0][0]
        
        logger.debug(f"Engine scores: {[(e.name, s) for e, s in engine_scores]}")
        
        return selected_engine
        
    def _analyze_content(
        self, 
        audio_path: Path, 
        language: Optional[str], 
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze content characteristics for optimal engine selection."""
        
        analysis = {
            "estimated_duration": self._estimate_audio_duration(audio_path),
            "language": language or "en",
            "has_context": bool(context),
            "content_type": "general",
            "complexity": "medium",
            "requires_speed": False,
            "requires_accuracy": True,
            "technical_content": False,
            "multi_speaker": False
        }
        
        if context:
            # Analyze video metadata for content hints
            title = context.get('video_title', '').lower()
            description = context.get('video_description', '').lower()
            
            # Detect content type
            if any(word in title for word in ['interview', 'podcast', 'conversation']):
                analysis["content_type"] = "interview"
                analysis["multi_speaker"] = True
            elif any(word in title for word in ['lecture', 'presentation', 'talk']):
                analysis["content_type"] = "lecture"
                analysis["multi_speaker"] = False
            elif any(word in title for word in ['meeting', 'call', 'discussion']):
                analysis["content_type"] = "meeting"
                analysis["multi_speaker"] = True
                
            # Detect technical content
            tech_indicators = [
                'programming', 'coding', 'tech', 'software', 'api', 'algorithm',
                'machine learning', 'ai', 'data science', 'engineering'
            ]
            if any(indicator in title + ' ' + description for indicator in tech_indicators):
                analysis["technical_content"] = True
                analysis["complexity"] = "high"
                
            # Detect if speed is important (short content, real-time needs)
            if analysis["estimated_duration"] < 120:  # Less than 2 minutes
                analysis["requires_speed"] = True
                
        return analysis
        
    def _estimate_audio_duration(self, audio_path: Path) -> float:
        """Estimate audio duration in seconds."""
        try:
            # Get file size as rough duration estimate (very rough!)
            file_size = audio_path.stat().st_size
            # Assume ~1MB per minute for compressed audio (very rough estimate)
            estimated_minutes = file_size / (1024 * 1024)
            return estimated_minutes * 60.0
        except Exception:
            return 300.0  # Default 5 minutes
            
    def _score_engine_for_content(
        self, 
        engine: TranscriptionEngine, 
        content_analysis: Dict[str, Any]
    ) -> float:
        """Score an engine's suitability for the given content."""
        
        base_score = 50.0  # Base score for all engines
        
        # Engine-specific scoring
        engine_name = engine.name
        
        # Gemini 2.0 Flash - Best for context understanding and technical content
        if "gemini" in engine_name:
            base_score = 90.0
            if content_analysis["has_context"]:
                base_score += 15.0  # Excellent context utilization
            if content_analysis["technical_content"]:
                base_score += 10.0  # Great for technical terms
            if content_analysis["complexity"] == "high":
                base_score += 5.0
                
        # AssemblyAI Universal-2 - Best raw accuracy
        elif "assemblyai" in engine_name:
            base_score = 85.0
            if content_analysis["requires_accuracy"]:
                base_score += 10.0  # Highest accuracy
            if content_analysis["multi_speaker"]:
                base_score += 8.0   # Excellent speaker diarization
            if content_analysis["content_type"] == "interview":
                base_score += 5.0
                
        # Deepgram Nova-3 - Best for speed
        elif "deepgram" in engine_name:
            base_score = 75.0
            if content_analysis["requires_speed"]:
                base_score += 15.0  # Fastest processing
            if content_analysis["estimated_duration"] < 300:  # Short audio
                base_score += 8.0
            if content_analysis["language"] != "en":
                base_score += 5.0   # Good multilingual support
                
        # Mock engine - Lowest priority
        elif "mock" in engine_name:
            base_score = 10.0   # Only use if nothing else available
            
        # Adjust for language support
        supported_languages = engine.get_supported_languages()
        if content_analysis["language"] not in supported_languages:
            base_score -= 20.0  # Penalize if language not supported
            
        return max(0.0, base_score)
        
    def estimate_cost(self, duration_seconds: float) -> float:
        """Estimate ensemble processing cost based on likely engine selection."""
        if not self.engines:
            return 0.0
            
        # For cost estimation, assume we'll use the most cost-effective engine
        costs = [engine.estimate_cost(duration_seconds) for engine in self.engines]
        
        # Return median cost as reasonable estimate
        costs.sort()
        if len(costs) == 1:
            return costs[0]
        elif len(costs) % 2 == 0:
            mid = len(costs) // 2
            return (costs[mid-1] + costs[mid]) / 2.0
        else:
            return costs[len(costs) // 2]