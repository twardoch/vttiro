#!/usr/bin/env python3
# this_file: src/vttiro/utils/timestamp_utils.py
"""WebVTT timestamp validation and repair utilities."""

from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, replace

try:
    from loguru import logger
except ImportError:
    import logging as logger

from vttiro.utils.types import SimpleTranscriptSegment


@dataclass
class TimestampIssue:
    """Represents a timestamp validation issue."""
    segment_index: int
    issue_type: str  # 'invalid_range', 'overlap', 'out_of_order', 'too_short', 'too_long'
    description: str
    severity: str  # 'error', 'warning'
    suggested_fix: Optional[str] = None


class TimestampValidator:
    """Comprehensive WebVTT timestamp validation and repair system."""
    
    def __init__(
        self,
        min_gap_seconds: float = 0.1,
        min_duration_seconds: float = 0.5,
        max_duration_seconds: float = 10.0,
        auto_repair: bool = True
    ):
        """Initialize the timestamp validator.
        
        Args:
            min_gap_seconds: Minimum gap between consecutive segments
            min_duration_seconds: Minimum duration for a segment
            max_duration_seconds: Maximum duration for a segment
            auto_repair: Whether to automatically repair issues when possible
        """
        self.min_gap = min_gap_seconds
        self.min_duration = min_duration_seconds
        self.max_duration = max_duration_seconds
        self.auto_repair = auto_repair
        
        logger.debug(f"TimestampValidator initialized: gap={min_gap_seconds}s, "
                    f"duration={min_duration_seconds}-{max_duration_seconds}s, "
                    f"auto_repair={auto_repair}")
    
    def validate_segments(
        self, 
        segments: List[SimpleTranscriptSegment]
    ) -> Tuple[bool, List[TimestampIssue], List[SimpleTranscriptSegment]]:
        """Validate and optionally repair timestamp issues in segments.
        
        Args:
            segments: List of transcript segments to validate
            
        Returns:
            Tuple of (is_valid, issues_found, repaired_segments)
        """
        if not segments:
            return True, [], []
        
        issues = []
        repaired_segments = segments.copy()
        
        # Phase 1: Individual segment validation
        issues.extend(self._validate_individual_segments(repaired_segments))
        
        # Phase 2: Sequential validation (order, overlaps, gaps)
        issues.extend(self._validate_segment_sequence(repaired_segments))
        
        # Phase 3: Apply repairs if enabled
        if self.auto_repair and issues:
            repaired_segments = self._repair_timestamp_issues(repaired_segments, issues)
            
            # Re-validate after repairs
            logger.debug("Re-validating after automatic repairs")
            post_repair_issues = []
            post_repair_issues.extend(self._validate_individual_segments(repaired_segments))
            post_repair_issues.extend(self._validate_segment_sequence(repaired_segments))
            
            # Update issues list to only include unresolved problems
            issues = post_repair_issues
        
        # Determine overall validity
        error_count = sum(1 for issue in issues if issue.severity == 'error')
        is_valid = error_count == 0
        
        if issues:
            logger.info(f"Timestamp validation: {len(issues)} issues found ({error_count} errors)")
            for issue in issues:
                level = logger.error if issue.severity == 'error' else logger.warning
                level(f"Segment {issue.segment_index}: {issue.description}")
        else:
            logger.debug(f"Timestamp validation: All {len(segments)} segments valid")
        
        return is_valid, issues, repaired_segments
    
    def _validate_individual_segments(
        self, 
        segments: List[SimpleTranscriptSegment]
    ) -> List[TimestampIssue]:
        """Validate individual segment timestamp ranges."""
        issues = []
        
        for i, segment in enumerate(segments):
            # Check for invalid time ranges (end <= start)
            if segment.end_time <= segment.start_time:
                issues.append(TimestampIssue(
                    segment_index=i,
                    issue_type='invalid_range',
                    description=f"End time ({segment.end_time:.3f}s) <= start time ({segment.start_time:.3f}s)",
                    severity='error',
                    suggested_fix=f"Set end time to {segment.start_time + self.min_duration:.3f}s"
                ))
            
            # Check for negative timestamps
            if segment.start_time < 0:
                issues.append(TimestampIssue(
                    segment_index=i,
                    issue_type='negative_time',
                    description=f"Negative start time: {segment.start_time:.3f}s",
                    severity='error',
                    suggested_fix="Set start time to 0.0s"
                ))
            
            if segment.end_time < 0:
                issues.append(TimestampIssue(
                    segment_index=i,
                    issue_type='negative_time',
                    description=f"Negative end time: {segment.end_time:.3f}s",
                    severity='error',
                    suggested_fix="Set end time to start time + minimum duration"
                ))
            
            # Check duration limits (only if times are valid)
            if segment.end_time > segment.start_time:
                duration = segment.end_time - segment.start_time
                
                if duration < self.min_duration:
                    issues.append(TimestampIssue(
                        segment_index=i,
                        issue_type='too_short',
                        description=f"Duration too short: {duration:.3f}s (min: {self.min_duration:.3f}s)",
                        severity='warning',
                        suggested_fix=f"Extend to {self.min_duration:.3f}s duration"
                    ))
                
                if duration > self.max_duration:
                    issues.append(TimestampIssue(
                        segment_index=i,
                        issue_type='too_long',
                        description=f"Duration too long: {duration:.3f}s (max: {self.max_duration:.3f}s)",
                        severity='warning',
                        suggested_fix=f"Split into multiple segments or cap at {self.max_duration:.3f}s"
                    ))
        
        return issues
    
    def _validate_segment_sequence(
        self, 
        segments: List[SimpleTranscriptSegment]
    ) -> List[TimestampIssue]:
        """Validate timestamp sequence across segments."""
        issues = []
        
        for i in range(1, len(segments)):
            current = segments[i]
            previous = segments[i - 1]
            
            # Check for out-of-order segments
            if current.start_time < previous.start_time:
                issues.append(TimestampIssue(
                    segment_index=i,
                    issue_type='out_of_order',
                    description=f"Start time ({current.start_time:.3f}s) before previous segment ({previous.start_time:.3f}s)",
                    severity='error',
                    suggested_fix=f"Reorder segments or set start time to {previous.end_time + self.min_gap:.3f}s"
                ))
            
            # Check for overlapping segments
            if current.start_time < previous.end_time:
                overlap = previous.end_time - current.start_time
                issues.append(TimestampIssue(
                    segment_index=i,
                    issue_type='overlap',
                    description=f"Overlaps with previous segment by {overlap:.3f}s",
                    severity='error',
                    suggested_fix=f"Set start time to {previous.end_time + self.min_gap:.3f}s"
                ))
            
            # Check for insufficient gaps
            elif current.start_time - previous.end_time < self.min_gap:
                gap = current.start_time - previous.end_time
                issues.append(TimestampIssue(
                    segment_index=i,
                    issue_type='small_gap',
                    description=f"Gap too small: {gap:.3f}s (min: {self.min_gap:.3f}s)",
                    severity='warning',
                    suggested_fix=f"Increase gap to {self.min_gap:.3f}s"
                ))
        
        return issues
    
    def _repair_timestamp_issues(
        self, 
        segments: List[SimpleTranscriptSegment], 
        issues: List[TimestampIssue]
    ) -> List[SimpleTranscriptSegment]:
        """Attempt to repair timestamp issues automatically."""
        repaired = segments.copy()
        
        # Sort issues by segment index to process in order
        sorted_issues = sorted(issues, key=lambda x: x.segment_index)
        
        for issue in sorted_issues:
            idx = issue.segment_index
            if idx >= len(repaired):
                continue
                
            segment = repaired[idx]
            
            if issue.issue_type == 'invalid_range':
                # Fix invalid ranges by adjusting end time
                new_end = segment.start_time + self.min_duration
                repaired[idx] = replace(segment, end_time=new_end)
                logger.debug(f"Repaired invalid range in segment {idx}: end_time = {new_end:.3f}s")
            
            elif issue.issue_type == 'negative_time':
                # Fix negative timestamps
                if segment.start_time < 0:
                    repaired[idx] = replace(segment, start_time=0.0)
                    logger.debug(f"Repaired negative start time in segment {idx}")
                
                if segment.end_time < 0:
                    new_end = max(segment.start_time + self.min_duration, 0.1)
                    repaired[idx] = replace(segment, end_time=new_end)
                    logger.debug(f"Repaired negative end time in segment {idx}: end_time = {new_end:.3f}s")
            
            elif issue.issue_type == 'too_short':
                # Extend short segments
                new_end = segment.start_time + self.min_duration
                repaired[idx] = replace(segment, end_time=new_end)
                logger.debug(f"Extended short segment {idx}: duration = {self.min_duration:.3f}s")
            
            elif issue.issue_type == 'overlap':
                # Fix overlaps by adjusting start time
                if idx > 0:
                    prev_end = repaired[idx - 1].end_time
                    new_start = prev_end + self.min_gap
                    
                    # Ensure end time is still valid
                    if new_start >= segment.end_time:
                        new_end = new_start + self.min_duration
                        repaired[idx] = replace(segment, start_time=new_start, end_time=new_end)
                    else:
                        repaired[idx] = replace(segment, start_time=new_start)
                    
                    logger.debug(f"Fixed overlap in segment {idx}: start_time = {new_start:.3f}s")
        
        # Final pass: ensure sequential order
        for i in range(1, len(repaired)):
            current = repaired[i]
            previous = repaired[i - 1]
            
            if current.start_time < previous.end_time + self.min_gap:
                new_start = previous.end_time + self.min_gap
                new_end = max(new_start + self.min_duration, current.end_time)
                repaired[i] = replace(current, start_time=new_start, end_time=new_end)
                logger.debug(f"Adjusted sequential order for segment {i}: {new_start:.3f}s - {new_end:.3f}s")
        
        return repaired
    
    def get_validation_summary(self, issues: List[TimestampIssue]) -> Dict[str, Any]:
        """Get a summary of validation results.
        
        Args:
            issues: List of timestamp issues found
            
        Returns:
            Dictionary with validation summary statistics
        """
        if not issues:
            return {
                'valid': True,
                'total_issues': 0,
                'errors': 0,
                'warnings': 0,
                'issue_types': {},
                'message': 'All timestamps are valid'
            }
        
        error_count = sum(1 for issue in issues if issue.severity == 'error')
        warning_count = len(issues) - error_count
        
        # Count issue types
        issue_types = {}
        for issue in issues:
            issue_types[issue.issue_type] = issue_types.get(issue.issue_type, 0) + 1
        
        return {
            'valid': error_count == 0,
            'total_issues': len(issues),
            'errors': error_count,
            'warnings': warning_count,
            'issue_types': issue_types,
            'message': f"{error_count} errors, {warning_count} warnings found"
        }
    
    def format_issue_report(self, issues: List[TimestampIssue]) -> str:
        """Format timestamp issues into a readable report.
        
        Args:
            issues: List of timestamp issues
            
        Returns:
            Formatted report string
        """
        if not issues:
            return "✓ All timestamps are valid"
        
        lines = ["Timestamp Validation Report:", "=" * 30]
        
        # Group issues by severity
        errors = [i for i in issues if i.severity == 'error']
        warnings = [i for i in issues if i.severity == 'warning']
        
        if errors:
            lines.append(f"\n❌ ERRORS ({len(errors)}):")
            for issue in errors:
                lines.append(f"  Segment {issue.segment_index}: {issue.description}")
                if issue.suggested_fix:
                    lines.append(f"    Fix: {issue.suggested_fix}")
        
        if warnings:
            lines.append(f"\n⚠️  WARNINGS ({len(warnings)}):")
            for issue in warnings:
                lines.append(f"  Segment {issue.segment_index}: {issue.description}")
                if issue.suggested_fix:
                    lines.append(f"    Fix: {issue.suggested_fix}")
        
        return "\n".join(lines)


def validate_webvtt_timestamps(
    segments: List[SimpleTranscriptSegment],
    auto_repair: bool = True,
    min_gap: float = 0.1,
    min_duration: float = 0.5
) -> Tuple[bool, List[SimpleTranscriptSegment], str]:
    """Convenience function for WebVTT timestamp validation.
    
    Args:
        segments: List of transcript segments to validate
        auto_repair: Whether to automatically repair issues
        min_gap: Minimum gap between segments
        min_duration: Minimum segment duration
        
    Returns:
        Tuple of (is_valid, repaired_segments, report_text)
    """
    validator = TimestampValidator(
        min_gap_seconds=min_gap,
        min_duration_seconds=min_duration,
        auto_repair=auto_repair
    )
    
    is_valid, issues, repaired_segments = validator.validate_segments(segments)
    report = validator.format_issue_report(issues)
    
    return is_valid, repaired_segments, report