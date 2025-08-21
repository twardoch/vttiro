#!/usr/bin/env python3
# this_file: src/vttiro/monitoring/performance.py
"""Performance monitoring and metrics collection for vttiro transcription operations.

This module provides comprehensive performance tracking, timing metrics, and
optimization analysis for transcription operations.
"""

import time
import psutil
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from collections import defaultdict
from datetime import datetime, timedelta

try:
    from loguru import logger
except ImportError:
    import logging as logger


@dataclass
class OperationMetrics:
    """Metrics for a single operation within a transcription session."""
    
    operation_type: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def finish(self, success: bool = True, error_message: Optional[str] = None):
        """Mark operation as finished and calculate duration."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.success = success
        self.error_message = error_message


@dataclass
class TranscriptionMetrics:
    """Complete metrics for a transcription session."""
    
    correlation_id: str
    input_file: str
    output_file: str
    engine: str
    model: str
    start_time: float
    end_time: Optional[float] = None
    total_duration: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    
    # File metrics
    input_file_size: int = 0
    output_file_size: int = 0
    audio_file_size: int = 0
    
    # Processing metrics
    operations: List[OperationMetrics] = field(default_factory=list)
    
    # Quality metrics
    transcription_quality: Dict[str, Any] = field(default_factory=dict)
    
    # Resource metrics
    peak_memory_usage: int = 0
    average_cpu_usage: float = 0.0
    memory_samples: List[int] = field(default_factory=list)
    cpu_samples: List[float] = field(default_factory=list)
    
    def finish(self, success: bool = True, error_message: Optional[str] = None):
        """Mark transcription as finished and calculate total duration."""
        self.end_time = time.time()
        self.total_duration = self.end_time - self.start_time
        self.success = success
        self.error_message = error_message
        
        # Calculate resource usage averages
        if self.memory_samples:
            self.peak_memory_usage = max(self.memory_samples)
        if self.cpu_samples:
            self.average_cpu_usage = sum(self.cpu_samples) / len(self.cpu_samples)


class PerformanceMonitor:
    """Comprehensive performance monitoring system for transcription operations.
    
    Tracks timing, resource usage, success/failure rates, and provides
    optimization recommendations.
    """
    
    def __init__(self):
        """Initialize the performance monitor."""
        self.active_sessions: Dict[str, TranscriptionMetrics] = {}
        self.completed_sessions: List[TranscriptionMetrics] = []
        self._resource_monitor_active: bool = False
        self._resource_monitor_thread: Optional[threading.Thread] = None
        
        # Global statistics
        self.global_stats = {
            'total_sessions': 0,
            'successful_sessions': 0,
            'failed_sessions': 0,
            'total_processing_time': 0.0,
            'total_files_processed': 0,
            'total_data_processed': 0,  # bytes
            'operation_stats': defaultdict(lambda: {'count': 0, 'total_time': 0.0, 'success_count': 0}),
            'engine_stats': defaultdict(lambda: {'count': 0, 'total_time': 0.0, 'success_count': 0}),
        }
        
        logger.info("PerformanceMonitor initialized")
    
    def start_transcription(
        self, 
        correlation_id: str,
        input_file: str,
        output_file: str,
        engine: str,
        model: str,
    ) -> TranscriptionMetrics:
        """Start monitoring a new transcription session.
        
        Args:
            correlation_id: Unique identifier for this transcription
            input_file: Path to input file
            output_file: Path to output file
            engine: AI engine being used
            model: Specific model being used
            
        Returns:
            TranscriptionMetrics object for this session
        """
        start_time = time.time()
        
        # Get file size
        input_file_size = 0
        try:
            input_path = Path(input_file)
            if input_path.exists():
                input_file_size = input_path.stat().st_size
        except Exception as e:
            logger.warning(f"Could not get input file size: {e}")
        
        metrics = TranscriptionMetrics(
            correlation_id=correlation_id,
            input_file=input_file,
            output_file=output_file,
            engine=engine,
            model=model,
            start_time=start_time,
            input_file_size=input_file_size,
        )
        
        self.active_sessions[correlation_id] = metrics
        self._start_resource_monitoring(correlation_id)
        
        logger.info(f"[{correlation_id}] Performance monitoring started for {input_file} ({input_file_size / 1024 / 1024:.1f}MB) -> {engine}/{model}")
        return metrics
    
    def start_operation(
        self, 
        correlation_id: str,
        operation_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> OperationMetrics:
        """Start monitoring a specific operation within a transcription.
        
        Args:
            correlation_id: Transcription session identifier
            operation_type: Type of operation (e.g., 'audio_extraction', 'ai_processing', 'webvtt_generation')
            metadata: Optional metadata about the operation
            
        Returns:
            OperationMetrics object for this operation
        """
        if correlation_id not in self.active_sessions:
            logger.warning(f"[{correlation_id}] Cannot start operation {operation_type}: session not found")
            return None
        
        operation = OperationMetrics(
            operation_type=operation_type,
            start_time=time.time(),
            metadata=metadata or {}
        )
        
        self.active_sessions[correlation_id].operations.append(operation)
        
        logger.debug(f"[{correlation_id}] Started operation: {operation_type}")
        return operation
    
    def finish_operation(
        self, 
        operation: OperationMetrics,
        success: bool = True,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Finish monitoring an operation.
        
        Args:
            operation: The operation metrics object to finish
            success: Whether the operation succeeded
            error_message: Error message if operation failed
            metadata: Additional metadata to record
        """
        if operation is None:
            return
            
        operation.finish(success=success, error_message=error_message)
        
        if metadata:
            operation.metadata.update(metadata)
        
        # Update global operation statistics
        op_stats = self.global_stats['operation_stats'][operation.operation_type]
        op_stats['count'] += 1
        op_stats['total_time'] += operation.duration
        if success:
            op_stats['success_count'] += 1
        
        status = "✓" if success else "✗"
        logger.info(f"{status} Operation {operation.operation_type} completed in {operation.duration:.2f}s")
        
        if not success and error_message:
            logger.error(f"Operation {operation.operation_type} failed: {error_message}")
    
    def finish_transcription(
        self,
        correlation_id: str,
        success: bool = True,
        error_message: Optional[str] = None,
        transcription_quality: Optional[Dict[str, Any]] = None
    ) -> Optional[TranscriptionMetrics]:
        """Finish monitoring a transcription session.
        
        Args:
            correlation_id: Transcription session identifier
            success: Whether the transcription succeeded
            error_message: Error message if transcription failed
            transcription_quality: Quality metrics (word count, confidence, etc.)
            
        Returns:
            Completed TranscriptionMetrics object
        """
        if correlation_id not in self.active_sessions:
            logger.warning(f"[{correlation_id}] Cannot finish transcription: session not found")
            return None
        
        metrics = self.active_sessions.pop(correlation_id)
        metrics.finish(success=success, error_message=error_message)
        
        if transcription_quality:
            metrics.transcription_quality = transcription_quality
        
        # Get output file size
        try:
            output_path = Path(metrics.output_file)
            if output_path.exists():
                metrics.output_file_size = output_path.stat().st_size
        except Exception as e:
            logger.warning(f"[{correlation_id}] Could not get output file size: {e}")
        
        self._stop_resource_monitoring(correlation_id)
        self.completed_sessions.append(metrics)
        
        # Update global statistics
        self.global_stats['total_sessions'] += 1
        self.global_stats['total_processing_time'] += metrics.total_duration
        self.global_stats['total_files_processed'] += 1
        self.global_stats['total_data_processed'] += metrics.input_file_size
        
        if success:
            self.global_stats['successful_sessions'] += 1
        else:
            self.global_stats['failed_sessions'] += 1
        
        # Update engine statistics
        engine_key = f"{metrics.engine}/{metrics.model}"
        engine_stats = self.global_stats['engine_stats'][engine_key]
        engine_stats['count'] += 1
        engine_stats['total_time'] += metrics.total_duration
        if success:
            engine_stats['success_count'] += 1
        
        # Log completion with performance summary
        self._log_session_summary(metrics)
        
        return metrics
    
    def _start_resource_monitoring(self, correlation_id: str):
        """Start monitoring system resources for a session."""
        if correlation_id not in self.active_sessions:
            return
            
        def monitor_resources():
            """Monitor CPU and memory usage during transcription."""
            process = psutil.Process()
            metrics = self.active_sessions.get(correlation_id)
            
            if metrics is None:
                return
            
            while correlation_id in self.active_sessions:
                try:
                    # Get memory usage (RSS - Resident Set Size)
                    memory_info = process.memory_info()
                    memory_usage = memory_info.rss  # bytes
                    
                    # Get CPU usage percentage
                    cpu_percent = process.cpu_percent(interval=0.1)
                    
                    metrics.memory_samples.append(memory_usage)
                    metrics.cpu_samples.append(cpu_percent)
                    
                    time.sleep(1.0)  # Sample every second
                    
                except (psutil.NoSuchProcess, KeyError):
                    # Process ended or session removed
                    break
                except Exception as e:
                    logger.warning(f"[{correlation_id}] Resource monitoring error: {e}")
                    break
        
        self._resource_monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
        self._resource_monitor_thread.start()
    
    def _stop_resource_monitoring(self, correlation_id: str):
        """Stop resource monitoring for a session."""
        # Resource monitoring will stop automatically when session is removed
        pass
    
    def _log_session_summary(self, metrics: TranscriptionMetrics):
        """Log a comprehensive summary of the transcription session."""
        correlation_id = metrics.correlation_id
        
        # Calculate processing speed
        if metrics.total_duration and metrics.input_file_size:
            mb_processed = metrics.input_file_size / (1024 * 1024)
            processing_speed = mb_processed / metrics.total_duration
            speed_ratio = self._calculate_speed_ratio(metrics)
        else:
            processing_speed = 0.0
            speed_ratio = 0.0
        
        # Operation breakdown
        operation_times = {}
        for op in metrics.operations:
            if op.duration:
                operation_times[op.operation_type] = op.duration
        
        # Resource usage
        peak_memory_mb = metrics.peak_memory_usage / (1024 * 1024) if metrics.peak_memory_usage else 0
        
        status = "✅ SUCCESS" if metrics.success else "❌ FAILED"
        
        logger.info(f"[{correlation_id}] === TRANSCRIPTION SUMMARY ===")
        logger.info(f"[{correlation_id}] Status: {status}")
        logger.info(f"[{correlation_id}] Total Duration: {metrics.total_duration:.2f}s")
        logger.info(f"[{correlation_id}] Engine/Model: {metrics.engine}/{metrics.model}")
        logger.info(f"[{correlation_id}] Input Size: {metrics.input_file_size / (1024*1024):.1f}MB")
        logger.info(f"[{correlation_id}] Processing Speed: {processing_speed:.2f} MB/s")
        logger.info(f"[{correlation_id}] Speed Ratio: {speed_ratio:.1f}x real-time")
        logger.info(f"[{correlation_id}] Peak Memory: {peak_memory_mb:.1f}MB")
        logger.info(f"[{correlation_id}] Average CPU: {metrics.average_cpu_usage:.1f}%")
        
        if operation_times:
            logger.info(f"[{correlation_id}] Operation Breakdown:")
            for op_type, duration in operation_times.items():
                percentage = (duration / metrics.total_duration) * 100
                logger.info(f"[{correlation_id}]   {op_type}: {duration:.2f}s ({percentage:.1f}%)")
        
        if metrics.transcription_quality:
            quality = metrics.transcription_quality
            logger.info(f"[{correlation_id}] Quality Metrics:")
            for key, value in quality.items():
                logger.info(f"[{correlation_id}]   {key}: {value}")
        
        if not metrics.success and metrics.error_message:
            logger.error(f"[{correlation_id}] Error: {metrics.error_message}")
        
        logger.info(f"[{correlation_id}] ==============================")
    
    def _calculate_speed_ratio(self, metrics: TranscriptionMetrics) -> float:
        """Calculate processing speed ratio compared to real-time.
        
        For audio/video files, this estimates how many times faster than
        real-time the processing was.
        """
        if not metrics.total_duration or not metrics.input_file_size:
            return 0.0
        
        # Rough estimation: assume average audio bitrate and calculate duration
        # This is approximate - ideally we'd get actual media duration
        estimated_audio_duration = metrics.input_file_size / (128 * 1024 / 8)  # Assume 128kbps MP3
        
        if estimated_audio_duration <= 0:
            return 0.0
        
        return estimated_audio_duration / metrics.total_duration
    
    def get_performance_report(self, last_n_sessions: int = 10) -> Dict[str, Any]:
        """Generate a comprehensive performance report.
        
        Args:
            last_n_sessions: Number of recent sessions to include in detailed analysis
            
        Returns:
            Dictionary containing performance metrics and optimization recommendations
        """
        recent_sessions = self.completed_sessions[-last_n_sessions:] if self.completed_sessions else []
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_sessions': self.global_stats['total_sessions'],
                'success_rate': self._calculate_success_rate(),
                'average_processing_time': self._calculate_average_processing_time(),
                'total_data_processed_gb': self.global_stats['total_data_processed'] / (1024**3),
                'overall_throughput_mb_per_sec': self._calculate_overall_throughput(),
            },
            'operation_performance': self._analyze_operation_performance(),
            'engine_performance': self._analyze_engine_performance(),
            'recent_sessions': [self._session_to_dict(s) for s in recent_sessions],
            'optimization_recommendations': self._generate_optimization_recommendations(),
            'resource_analysis': self._analyze_resource_usage(recent_sessions),
        }
        
        return report
    
    def _calculate_success_rate(self) -> float:
        """Calculate overall success rate as percentage."""
        if self.global_stats['total_sessions'] == 0:
            return 100.0
        return (self.global_stats['successful_sessions'] / self.global_stats['total_sessions']) * 100
    
    def _calculate_average_processing_time(self) -> float:
        """Calculate average processing time across all sessions."""
        if self.global_stats['total_sessions'] == 0:
            return 0.0
        return self.global_stats['total_processing_time'] / self.global_stats['total_sessions']
    
    def _calculate_overall_throughput(self) -> float:
        """Calculate overall data processing throughput in MB/s."""
        if self.global_stats['total_processing_time'] == 0:
            return 0.0
        total_mb = self.global_stats['total_data_processed'] / (1024 * 1024)
        return total_mb / self.global_stats['total_processing_time']
    
    def _analyze_operation_performance(self) -> Dict[str, Any]:
        """Analyze performance of different operation types."""
        analysis = {}
        for op_type, stats in self.global_stats['operation_stats'].items():
            if stats['count'] > 0:
                avg_time = stats['total_time'] / stats['count']
                success_rate = (stats['success_count'] / stats['count']) * 100
                analysis[op_type] = {
                    'count': stats['count'],
                    'average_duration': avg_time,
                    'success_rate': success_rate,
                    'total_time': stats['total_time']
                }
        return analysis
    
    def _analyze_engine_performance(self) -> Dict[str, Any]:
        """Analyze performance of different engines and models."""
        analysis = {}
        for engine_model, stats in self.global_stats['engine_stats'].items():
            if stats['count'] > 0:
                avg_time = stats['total_time'] / stats['count']
                success_rate = (stats['success_count'] / stats['count']) * 100
                analysis[engine_model] = {
                    'count': stats['count'],
                    'average_duration': avg_time,
                    'success_rate': success_rate,
                    'total_time': stats['total_time']
                }
        return analysis
    
    def _analyze_resource_usage(self, sessions: List[TranscriptionMetrics]) -> Dict[str, Any]:
        """Analyze resource usage patterns."""
        if not sessions:
            return {}
        
        memory_usage = []
        cpu_usage = []
        file_sizes = []
        
        for session in sessions:
            if session.peak_memory_usage:
                memory_usage.append(session.peak_memory_usage / (1024*1024))  # MB
            if session.average_cpu_usage:
                cpu_usage.append(session.average_cpu_usage)
            if session.input_file_size:
                file_sizes.append(session.input_file_size / (1024*1024))  # MB
        
        analysis = {}
        
        if memory_usage:
            analysis['memory'] = {
                'average_peak_mb': sum(memory_usage) / len(memory_usage),
                'max_peak_mb': max(memory_usage),
                'min_peak_mb': min(memory_usage),
            }
        
        if cpu_usage:
            analysis['cpu'] = {
                'average_usage_percent': sum(cpu_usage) / len(cpu_usage),
                'max_usage_percent': max(cpu_usage),
                'min_usage_percent': min(cpu_usage),
            }
        
        if file_sizes:
            analysis['file_sizes'] = {
                'average_input_size_mb': sum(file_sizes) / len(file_sizes),
                'largest_file_mb': max(file_sizes),
                'smallest_file_mb': min(file_sizes),
            }
        
        return analysis
    
    def _generate_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on collected metrics."""
        recommendations = []
        
        # Check success rates
        success_rate = self._calculate_success_rate()
        if success_rate < 95.0:
            recommendations.append(f"Success rate is {success_rate:.1f}%. Consider investigating common failure patterns.")
        
        # Check operation performance
        op_analysis = self._analyze_operation_performance()
        for op_type, stats in op_analysis.items():
            if stats['success_rate'] < 90.0:
                recommendations.append(f"{op_type} has low success rate ({stats['success_rate']:.1f}%). Review error handling.")
            
            # Identify slow operations
            if op_type == 'ai_processing' and stats['average_duration'] > 60.0:
                recommendations.append(f"AI processing is slow ({stats['average_duration']:.1f}s avg). Consider using faster models or chunking.")
        
        # Check engine performance
        engine_analysis = self._analyze_engine_performance()
        if len(engine_analysis) > 1:
            # Compare engine performance
            engine_times = [(engine, stats['average_duration']) for engine, stats in engine_analysis.items()]
            engine_times.sort(key=lambda x: x[1])
            fastest = engine_times[0]
            slowest = engine_times[-1]
            
            if slowest[1] > fastest[1] * 2:
                recommendations.append(f"Consider using {fastest[0]} instead of {slowest[0]} for better performance ({fastest[1]:.1f}s vs {slowest[1]:.1f}s average).")
        
        # Memory recommendations
        recent_sessions = self.completed_sessions[-10:] if self.completed_sessions else []
        if recent_sessions:
            avg_memory = sum(s.peak_memory_usage for s in recent_sessions if s.peak_memory_usage) / len([s for s in recent_sessions if s.peak_memory_usage])
            if avg_memory > 1024 * 1024 * 1024:  # > 1GB
                recommendations.append(f"High memory usage detected ({avg_memory / (1024**3):.1f}GB avg). Consider processing files in smaller chunks.")
        
        if not recommendations:
            recommendations.append("Performance looks good! No specific optimizations recommended at this time.")
        
        return recommendations
    
    def _session_to_dict(self, session: TranscriptionMetrics) -> Dict[str, Any]:
        """Convert a session metrics object to a dictionary for reporting."""
        return {
            'correlation_id': session.correlation_id,
            'input_file': Path(session.input_file).name,  # Just filename for privacy
            'engine': session.engine,
            'model': session.model,
            'duration': session.total_duration,
            'success': session.success,
            'input_size_mb': session.input_file_size / (1024*1024) if session.input_file_size else 0,
            'peak_memory_mb': session.peak_memory_usage / (1024*1024) if session.peak_memory_usage else 0,
            'avg_cpu_percent': session.average_cpu_usage,
            'operations': [
                {
                    'type': op.operation_type,
                    'duration': op.duration,
                    'success': op.success
                }
                for op in session.operations if op.duration
            ]
        }
    
    def log_performance_report(self, last_n_sessions: int = 5):
        """Log a performance report to the console.
        
        Args:
            last_n_sessions: Number of recent sessions to include in the report
        """
        report = self.get_performance_report(last_n_sessions)
        
        logger.info("=== VTTIRO PERFORMANCE REPORT ===")
        
        summary = report['summary']
        logger.info(f"Total Sessions: {summary['total_sessions']}")
        logger.info(f"Success Rate: {summary['success_rate']:.1f}%")
        logger.info(f"Average Processing Time: {summary['average_processing_time']:.2f}s")
        logger.info(f"Overall Throughput: {summary['overall_throughput_mb_per_sec']:.2f} MB/s")
        logger.info(f"Data Processed: {summary['total_data_processed_gb']:.2f} GB")
        
        if report['optimization_recommendations']:
            logger.info("Optimization Recommendations:")
            for i, rec in enumerate(report['optimization_recommendations'], 1):
                logger.info(f"  {i}. {rec}")
        
        logger.info("==================================")


# Global performance monitor instance
_performance_monitor = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor