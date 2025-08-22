# this_file: src/vttiro/output/quality_analyzer.py
"""Comprehensive output quality analysis and optimization recommendations.

This module provides advanced quality assessment for subtitle outputs including:
- Multi-dimensional quality scoring (readability, timing, accessibility)
- WCAG compliance validation and scoring
- Cross-format quality comparison
- Automated optimization recommendations
- Detailed quality reports with actionable insights

Used by:
- Output validation pipeline for quality assurance
- Quality monitoring and alerting systems
- Optimization recommendation engines
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from ..core.types import TranscriptionResult, TranscriptSegment


class QualityDimension(Enum):
    """Quality assessment dimensions."""
    READABILITY = "readability"
    TIMING = "timing"
    ACCESSIBILITY = "accessibility"
    CONSISTENCY = "consistency"
    TECHNICAL = "technical"


class SeverityLevel(Enum):
    """Issue severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class QualityIssue:
    """Represents a quality issue found during analysis."""
    
    dimension: QualityDimension
    severity: SeverityLevel
    message: str
    segment_index: Optional[int] = None
    recommendation: str = ""
    auto_fixable: bool = False
    impact_score: float = 0.0  # 0.0-1.0


@dataclass
class QualityMetrics:
    """Comprehensive quality metrics for subtitle output."""
    
    # Overall scores (0.0-1.0)
    overall_score: float = 0.0
    readability_score: float = 0.0
    timing_score: float = 0.0
    accessibility_score: float = 0.0
    consistency_score: float = 0.0
    technical_score: float = 0.0
    
    # Detailed metrics
    total_segments: int = 0
    total_duration: float = 0.0
    total_words: int = 0
    average_reading_speed: float = 0.0
    
    # Timing metrics
    segments_too_short: int = 0
    segments_too_long: int = 0
    timing_gaps: int = 0
    timing_overlaps: int = 0
    
    # Readability metrics
    max_line_length: int = 0
    lines_too_long: int = 0
    avg_words_per_segment: float = 0.0
    
    # Accessibility metrics
    wcag_compliance_level: str = "Unknown"
    has_speaker_identification: bool = False
    readable_at_target_speed: bool = True
    
    # Issues found
    issues: List[QualityIssue] = field(default_factory=list)


@dataclass
class OptimizationRecommendation:
    """Automated optimization recommendation."""
    
    category: str
    priority: SeverityLevel
    description: str
    expected_improvement: float  # Expected score improvement
    implementation_effort: str   # "low", "medium", "high"
    auto_applicable: bool = False


class OutputQualityAnalyzer:
    """Comprehensive quality analyzer for subtitle outputs."""
    
    def __init__(self) -> None:
        """Initialize analyzer with configurable thresholds."""
        self.thresholds = {
            # Timing thresholds
            "min_segment_duration": 1.0,
            "max_segment_duration": 6.0,
            "max_gap_duration": 2.0,
            "min_gap_duration": 0.1,
            
            # Reading speed thresholds (words per minute)
            "min_reading_speed": 120,
            "max_reading_speed": 200,
            "optimal_reading_speed": 160,
            
            # Readability thresholds
            "max_line_length": 42,
            "max_lines_per_segment": 2,
            "max_words_per_segment": 12,
            
            # Quality score thresholds
            "excellent_threshold": 0.9,
            "good_threshold": 0.8,
            "acceptable_threshold": 0.7,
        }
    
    def analyze_quality(self, result: TranscriptionResult) -> QualityMetrics:
        """Perform comprehensive quality analysis on transcription result.
        
        Args:
            result: Transcription result to analyze
            
        Returns:
            Comprehensive quality metrics with issues and recommendations
        """
        segments = result.segments
        
        if not segments:
            return QualityMetrics()
        
        # Initialize metrics
        metrics = QualityMetrics(
            total_segments=len(segments),
            total_duration=sum(seg.duration() for seg in segments),
            total_words=sum(len(seg.text.split()) for seg in segments)
        )
        
        # Calculate derived metrics
        if metrics.total_duration > 0:
            metrics.average_reading_speed = (metrics.total_words / (metrics.total_duration / 60))
        
        if metrics.total_segments > 0:
            metrics.avg_words_per_segment = metrics.total_words / metrics.total_segments
        
        # Analyze each quality dimension
        self._analyze_timing_quality(segments, metrics)
        self._analyze_readability_quality(segments, metrics)
        self._analyze_accessibility_quality(segments, metrics)
        self._analyze_consistency_quality(segments, metrics)
        self._analyze_technical_quality(segments, metrics)
        
        # Calculate overall scores
        self._calculate_quality_scores(metrics)
        
        return metrics
    
    def generate_recommendations(self, metrics: QualityMetrics) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations based on quality analysis.
        
        Args:
            metrics: Quality metrics from analysis
            
        Returns:
            List of prioritized optimization recommendations
        """
        recommendations = []
        
        # Timing recommendations
        if metrics.timing_score < self.thresholds["good_threshold"]:
            if metrics.segments_too_short > 0:
                recommendations.append(OptimizationRecommendation(
                    category="timing",
                    priority=SeverityLevel.HIGH,
                    description=f"Merge {metrics.segments_too_short} segments that are too short for readability",
                    expected_improvement=0.1,
                    implementation_effort="low",
                    auto_applicable=True
                ))
            
            if metrics.segments_too_long > 0:
                recommendations.append(OptimizationRecommendation(
                    category="timing",
                    priority=SeverityLevel.MEDIUM,
                    description=f"Split {metrics.segments_too_long} segments that are too long",
                    expected_improvement=0.15,
                    implementation_effort="medium",
                    auto_applicable=True
                ))
        
        # Readability recommendations
        if metrics.readability_score < self.thresholds["good_threshold"]:
            if metrics.lines_too_long > 0:
                recommendations.append(OptimizationRecommendation(
                    category="readability",
                    priority=SeverityLevel.HIGH,
                    description=f"Break {metrics.lines_too_long} lines that exceed recommended length",
                    expected_improvement=0.2,
                    implementation_effort="low",
                    auto_applicable=True
                ))
        
        # Accessibility recommendations
        if metrics.accessibility_score < self.thresholds["good_threshold"]:
            if not metrics.has_speaker_identification:
                recommendations.append(OptimizationRecommendation(
                    category="accessibility",
                    priority=SeverityLevel.MEDIUM,
                    description="Add speaker identification for better accessibility",
                    expected_improvement=0.15,
                    implementation_effort="low",
                    auto_applicable=False
                ))
            
            if not metrics.readable_at_target_speed:
                recommendations.append(OptimizationRecommendation(
                    category="accessibility",
                    priority=SeverityLevel.HIGH,
                    description="Adjust segment timing to match recommended reading speed",
                    expected_improvement=0.25,
                    implementation_effort="medium",
                    auto_applicable=True
                ))
        
        # Sort by priority and expected improvement
        recommendations.sort(key=lambda r: (r.priority.value, -r.expected_improvement))
        
        return recommendations
    
    def compare_formats(self, format_results: Dict[str, QualityMetrics]) -> Dict[str, Any]:
        """Compare quality metrics across different subtitle formats.
        
        Args:
            format_results: Dictionary mapping format names to their quality metrics
            
        Returns:
            Comparison analysis with rankings and recommendations
        """
        if not format_results:
            return {}
        
        # Rank formats by overall score
        ranked_formats = sorted(
            format_results.items(),
            key=lambda x: x[1].overall_score,
            reverse=True
        )
        
        best_format = ranked_formats[0][0]
        best_score = ranked_formats[0][1].overall_score
        
        # Find best format for each dimension
        dimension_leaders = {}
        for dimension in ["readability_score", "timing_score", "accessibility_score", 
                         "consistency_score", "technical_score"]:
            leader = max(format_results.items(), key=lambda x: getattr(x[1], dimension))
            dimension_leaders[dimension] = leader[0]
        
        # Generate cross-format insights
        insights = []
        
        # Check if any format significantly outperforms others
        score_gaps = [best_score - metrics.overall_score for _, metrics in ranked_formats[1:]]
        if score_gaps and max(score_gaps) > 0.15:
            insights.append(f"{best_format} significantly outperforms other formats")
        
        # Check for dimension-specific leaders
        for dimension, leader in dimension_leaders.items():
            if leader != best_format:
                insights.append(f"{leader} excels in {dimension.replace('_score', '')}")
        
        return {
            "ranking": [(name, metrics.overall_score) for name, metrics in ranked_formats],
            "best_overall": best_format,
            "dimension_leaders": dimension_leaders,
            "insights": insights,
            "average_score": sum(m.overall_score for m in format_results.values()) / len(format_results)
        }
    
    def _analyze_timing_quality(self, segments: List[TranscriptSegment], metrics: QualityMetrics) -> None:
        """Analyze timing-related quality aspects."""
        issues = []
        
        for i, segment in enumerate(segments):
            duration = segment.duration()
            
            # Check segment duration
            if duration < self.thresholds["min_segment_duration"]:
                metrics.segments_too_short += 1
                issues.append(QualityIssue(
                    dimension=QualityDimension.TIMING,
                    severity=SeverityLevel.MEDIUM,
                    message=f"Segment {i+1} duration ({duration:.1f}s) is too short",
                    segment_index=i,
                    recommendation="Merge with adjacent segment",
                    auto_fixable=True,
                    impact_score=0.3
                ))
            
            elif duration > self.thresholds["max_segment_duration"]:
                metrics.segments_too_long += 1
                issues.append(QualityIssue(
                    dimension=QualityDimension.TIMING,
                    severity=SeverityLevel.MEDIUM,
                    message=f"Segment {i+1} duration ({duration:.1f}s) is too long",
                    segment_index=i,
                    recommendation="Split segment at natural break",
                    auto_fixable=True,
                    impact_score=0.4
                ))
            
            # Check gaps between segments
            if i > 0:
                gap = segment.start - segments[i-1].end
                
                if gap > self.thresholds["max_gap_duration"]:
                    metrics.timing_gaps += 1
                    issues.append(QualityIssue(
                        dimension=QualityDimension.TIMING,
                        severity=SeverityLevel.LOW,
                        message=f"Large gap ({gap:.1f}s) between segments {i} and {i+1}",
                        segment_index=i,
                        recommendation="Review audio for missing content",
                        auto_fixable=False,
                        impact_score=0.1
                    ))
                
                elif gap < 0:
                    metrics.timing_overlaps += 1
                    issues.append(QualityIssue(
                        dimension=QualityDimension.TIMING,
                        severity=SeverityLevel.HIGH,
                        message=f"Timing overlap between segments {i} and {i+1}",
                        segment_index=i,
                        recommendation="Adjust segment timing to prevent overlap",
                        auto_fixable=True,
                        impact_score=0.7
                    ))
        
        metrics.issues.extend(issues)
        
        # Calculate timing score
        total_timing_issues = metrics.segments_too_short + metrics.segments_too_long + \
                            metrics.timing_gaps + metrics.timing_overlaps
        
        if metrics.total_segments > 0:
            timing_error_rate = total_timing_issues / metrics.total_segments
            metrics.timing_score = max(0.0, 1.0 - timing_error_rate)
        else:
            metrics.timing_score = 1.0
    
    def _analyze_readability_quality(self, segments: List[TranscriptSegment], metrics: QualityMetrics) -> None:
        """Analyze readability-related quality aspects."""
        issues = []
        max_line_length = 0
        
        for i, segment in enumerate(segments):
            lines = segment.text.split('\n')
            
            # Check line lengths
            for line_num, line in enumerate(lines):
                line_length = len(line)
                max_line_length = max(max_line_length, line_length)
                
                if line_length > self.thresholds["max_line_length"]:
                    metrics.lines_too_long += 1
                    issues.append(QualityIssue(
                        dimension=QualityDimension.READABILITY,
                        severity=SeverityLevel.MEDIUM,
                        message=f"Line {line_num+1} in segment {i+1} exceeds recommended length ({line_length} chars)",
                        segment_index=i,
                        recommendation="Break line at natural word boundary",
                        auto_fixable=True,
                        impact_score=0.3
                    ))
            
            # Check number of lines
            if len(lines) > self.thresholds["max_lines_per_segment"]:
                issues.append(QualityIssue(
                    dimension=QualityDimension.READABILITY,
                    severity=SeverityLevel.MEDIUM,
                    message=f"Segment {i+1} has too many lines ({len(lines)})",
                    segment_index=i,
                    recommendation="Reduce to maximum 2 lines per segment",
                    auto_fixable=True,
                    impact_score=0.4
                ))
            
            # Check words per segment
            word_count = len(segment.text.split())
            if word_count > self.thresholds["max_words_per_segment"]:
                issues.append(QualityIssue(
                    dimension=QualityDimension.READABILITY,
                    severity=SeverityLevel.LOW,
                    message=f"Segment {i+1} has many words ({word_count})",
                    segment_index=i,
                    recommendation="Consider splitting segment",
                    auto_fixable=True,
                    impact_score=0.2
                ))
        
        metrics.max_line_length = max_line_length
        metrics.issues.extend(issues)
        
        # Calculate readability score
        readability_issues = metrics.lines_too_long + len([i for i in issues if i.dimension == QualityDimension.READABILITY])
        
        if metrics.total_segments > 0:
            readability_error_rate = readability_issues / (metrics.total_segments * 2)  # Assume 2 lines per segment
            metrics.readability_score = max(0.0, 1.0 - readability_error_rate)
        else:
            metrics.readability_score = 1.0
    
    def _analyze_accessibility_quality(self, segments: List[TranscriptSegment], metrics: QualityMetrics) -> None:
        """Analyze accessibility-related quality aspects."""
        issues = []
        
        # Check speaker identification
        has_speakers = any(seg.speaker for seg in segments)
        metrics.has_speaker_identification = has_speakers
        
        if not has_speakers:
            issues.append(QualityIssue(
                dimension=QualityDimension.ACCESSIBILITY,
                severity=SeverityLevel.MEDIUM,
                message="No speaker identification found",
                recommendation="Add speaker labels for better accessibility",
                auto_fixable=False,
                impact_score=0.3
            ))
        
        # Check reading speed
        readable_at_speed = True
        if metrics.average_reading_speed > 0:
            if (metrics.average_reading_speed < self.thresholds["min_reading_speed"] or
                metrics.average_reading_speed > self.thresholds["max_reading_speed"]):
                readable_at_speed = False
                issues.append(QualityIssue(
                    dimension=QualityDimension.ACCESSIBILITY,
                    severity=SeverityLevel.HIGH,
                    message=f"Reading speed ({metrics.average_reading_speed:.0f} WPM) outside recommended range",
                    recommendation="Adjust segment timing to optimize reading speed",
                    auto_fixable=True,
                    impact_score=0.5
                ))
        
        metrics.readable_at_target_speed = readable_at_speed
        
        # Determine WCAG compliance level
        accessibility_issues = len([i for i in issues if i.dimension == QualityDimension.ACCESSIBILITY])
        critical_issues = len([i for i in issues if i.severity == SeverityLevel.CRITICAL])
        
        if critical_issues == 0 and accessibility_issues == 0:
            metrics.wcag_compliance_level = "AAA"
        elif critical_issues == 0 and accessibility_issues <= 1:
            metrics.wcag_compliance_level = "AA"
        elif critical_issues == 0:
            metrics.wcag_compliance_level = "A"
        else:
            metrics.wcag_compliance_level = "Non-compliant"
        
        metrics.issues.extend(issues)
        
        # Calculate accessibility score
        speaker_score = 1.0 if has_speakers else 0.7
        speed_score = 1.0 if readable_at_speed else 0.5
        
        metrics.accessibility_score = (speaker_score * 0.4 + speed_score * 0.6)
    
    def _analyze_consistency_quality(self, segments: List[TranscriptSegment], metrics: QualityMetrics) -> None:
        """Analyze consistency-related quality aspects."""
        issues = []
        
        if not segments:
            metrics.consistency_score = 1.0
            return
        
        # Check formatting consistency
        formatting_patterns = set()
        for segment in segments:
            if segment.speaker:
                # Extract speaker label format
                if '[' in segment.text and ']' in segment.text:
                    pattern = "bracketed"
                elif segment.text.startswith(segment.speaker):
                    pattern = "prefixed"
                else:
                    pattern = "metadata"
                
                formatting_patterns.add(pattern)
        
        if len(formatting_patterns) > 1:
            issues.append(QualityIssue(
                dimension=QualityDimension.CONSISTENCY,
                severity=SeverityLevel.MEDIUM,
                message="Inconsistent speaker label formatting",
                recommendation="Standardize speaker label format across all segments",
                auto_fixable=True,
                impact_score=0.3
            ))
        
        # Check timing consistency
        durations = [seg.duration() for seg in segments]
        if durations:
            avg_duration = sum(durations) / len(durations)
            very_short = sum(1 for d in durations if d < avg_duration * 0.3)
            very_long = sum(1 for d in durations if d > avg_duration * 3.0)
            
            total_outliers = very_short + very_long
            if total_outliers > len(segments) * 0.2:  # More than 20% outliers
                issues.append(QualityIssue(
                    dimension=QualityDimension.CONSISTENCY,
                    severity=SeverityLevel.LOW,
                    message=f"Inconsistent segment durations ({total_outliers} outliers)",
                    recommendation="Review timing consistency across segments",
                    auto_fixable=False,
                    impact_score=0.2
                ))
        
        metrics.issues.extend(issues)
        
        # Calculate consistency score
        consistency_issues = len([i for i in issues if i.dimension == QualityDimension.CONSISTENCY])
        metrics.consistency_score = max(0.0, 1.0 - (consistency_issues * 0.2))
    
    def _analyze_technical_quality(self, segments: List[TranscriptSegment], metrics: QualityMetrics) -> None:
        """Analyze technical quality aspects."""
        issues = []
        
        for i, segment in enumerate(segments):
            # Check for empty or whitespace-only text
            if not segment.text.strip():
                issues.append(QualityIssue(
                    dimension=QualityDimension.TECHNICAL,
                    severity=SeverityLevel.CRITICAL,
                    message=f"Segment {i+1} has empty text",
                    segment_index=i,
                    recommendation="Remove empty segment or add content",
                    auto_fixable=True,
                    impact_score=0.8
                ))
            
            # Check for invalid timing
            if segment.start < 0 or segment.end <= segment.start:
                issues.append(QualityIssue(
                    dimension=QualityDimension.TECHNICAL,
                    severity=SeverityLevel.CRITICAL,
                    message=f"Segment {i+1} has invalid timing",
                    segment_index=i,
                    recommendation="Fix timing values",
                    auto_fixable=True,
                    impact_score=0.9
                ))
            
            # Check for unusual characters
            unusual_chars = re.findall(r'[^\w\s\.,!?\-\'\"]', segment.text)
            if unusual_chars:
                issues.append(QualityIssue(
                    dimension=QualityDimension.TECHNICAL,
                    severity=SeverityLevel.LOW,
                    message=f"Segment {i+1} contains unusual characters: {set(unusual_chars)}",
                    segment_index=i,
                    recommendation="Review and clean unusual characters",
                    auto_fixable=False,
                    impact_score=0.1
                ))
        
        metrics.issues.extend(issues)
        
        # Calculate technical score
        critical_issues = len([i for i in issues if i.severity == SeverityLevel.CRITICAL])
        total_technical_issues = len([i for i in issues if i.dimension == QualityDimension.TECHNICAL])
        
        if metrics.total_segments > 0:
            critical_error_rate = critical_issues / metrics.total_segments
            technical_error_rate = total_technical_issues / metrics.total_segments
            
            # Critical issues have major impact
            metrics.technical_score = max(0.0, (1.0 - critical_error_rate * 0.8 - technical_error_rate * 0.2))
        else:
            metrics.technical_score = 1.0
    
    def _calculate_quality_scores(self, metrics: QualityMetrics) -> None:
        """Calculate overall quality score from individual dimension scores."""
        # Weighted average of dimension scores
        dimension_weights = {
            'technical_score': 0.3,      # Technical issues are blockers
            'accessibility_score': 0.25, # Accessibility is crucial
            'readability_score': 0.2,    # Readability affects user experience
            'timing_score': 0.15,        # Timing affects comprehension
            'consistency_score': 0.1     # Consistency is nice-to-have
        }
        
        weighted_sum = sum(
            getattr(metrics, dimension) * weight
            for dimension, weight in dimension_weights.items()
        )
        
        metrics.overall_score = min(1.0, max(0.0, weighted_sum))