#!/usr/bin/env python3
# this_file: src/vttiro/processing/parallel.py
"""Parallel processing utilities for high-performance audio and video processing."""

import asyncio
import multiprocessing
import concurrent.futures
from typing import List, Any, Callable, Optional, Dict, Union
from pathlib import Path
import time
import psutil
from dataclasses import dataclass

try:
    from loguru import logger
except ImportError:
    import logging as logger

from vttiro.utils.exceptions import ProcessingError, ResourceError


@dataclass
class ProcessingTask:
    """Represents a processing task with metadata."""
    task_id: str
    input_data: Any
    task_type: str
    priority: int = 0
    estimated_duration: float = 0.0
    memory_requirement_mb: float = 0.0
    
    def __post_init__(self):
        """Post-initialization validation."""
        if self.priority < 0:
            raise ValueError("Priority must be non-negative")
        if self.estimated_duration < 0:
            raise ValueError("Estimated duration must be non-negative")
        if self.memory_requirement_mb < 0:
            raise ValueError("Memory requirement must be non-negative")


@dataclass
class ProcessingResult:
    """Represents the result of a processing task."""
    task_id: str
    result: Any
    processing_time: float
    memory_used_mb: float
    success: bool
    error: Optional[Exception] = None


class ResourceMonitor:
    """Monitors system resources during processing."""
    
    def __init__(self):
        """Initialize resource monitor."""
        self.initial_memory = psutil.virtual_memory().used
        self.initial_cpu_percent = psutil.cpu_percent()
        self.peak_memory = self.initial_memory
        self.peak_cpu_percent = self.initial_cpu_percent
    
    def update(self) -> Dict[str, float]:
        """Update resource monitoring data.
        
        Returns:
            Current resource usage statistics
        """
        current_memory = psutil.virtual_memory().used
        current_cpu = psutil.cpu_percent()
        
        self.peak_memory = max(self.peak_memory, current_memory)
        self.peak_cpu_percent = max(self.peak_cpu_percent, current_cpu)
        
        return {
            "current_memory_mb": current_memory / (1024 * 1024),
            "peak_memory_mb": self.peak_memory / (1024 * 1024),
            "memory_increase_mb": (current_memory - self.initial_memory) / (1024 * 1024),
            "current_cpu_percent": current_cpu,
            "peak_cpu_percent": self.peak_cpu_percent,
        }
    
    def get_available_memory_mb(self) -> float:
        """Get available memory in MB."""
        return psutil.virtual_memory().available / (1024 * 1024)
    
    def get_cpu_count(self) -> int:
        """Get number of CPU cores."""
        return psutil.cpu_count()


class AdaptiveWorkerPool:
    """Adaptive worker pool that adjusts based on system resources and workload."""
    
    def __init__(
        self,
        min_workers: int = 1,
        max_workers: Optional[int] = None,
        memory_limit_mb: float = 2048.0,
        cpu_threshold: float = 90.0
    ):
        """Initialize adaptive worker pool.
        
        Args:
            min_workers: Minimum number of workers
            max_workers: Maximum number of workers (defaults to CPU count)
            memory_limit_mb: Memory limit for the pool
            cpu_threshold: CPU usage threshold for worker adjustment
        """
        self.min_workers = min_workers
        self.max_workers = max_workers or psutil.cpu_count()
        self.memory_limit_mb = memory_limit_mb
        self.cpu_threshold = cpu_threshold
        
        self.resource_monitor = ResourceMonitor()
        self.current_workers = min_workers
        self.task_queue: List[ProcessingTask] = []
        self.active_tasks: Dict[str, ProcessingTask] = {}
        
        # Performance tracking
        self.completed_tasks = 0
        self.total_processing_time = 0.0
        self.average_task_time = 0.0
    
    def should_scale_up(self) -> bool:
        """Determine if pool should scale up workers.
        
        Returns:
            True if should add more workers
        """
        resources = self.resource_monitor.update()
        
        # Check if we're under resource limits
        memory_ok = resources["current_memory_mb"] < self.memory_limit_mb * 0.8
        cpu_ok = resources["current_cpu_percent"] < self.cpu_threshold
        queue_has_work = len(self.task_queue) > self.current_workers
        can_scale = self.current_workers < self.max_workers
        
        return memory_ok and cpu_ok and queue_has_work and can_scale
    
    def should_scale_down(self) -> bool:
        """Determine if pool should scale down workers.
        
        Returns:
            True if should remove workers
        """
        resources = self.resource_monitor.update()
        
        # Check if we should reduce workers
        low_queue = len(self.task_queue) < self.current_workers // 2
        high_memory = resources["current_memory_mb"] > self.memory_limit_mb * 0.9
        high_cpu = resources["current_cpu_percent"] > self.cpu_threshold
        can_scale = self.current_workers > self.min_workers
        
        return (low_queue or high_memory or high_cpu) and can_scale
    
    def adjust_worker_count(self) -> int:
        """Adjust worker count based on current conditions.
        
        Returns:
            New worker count
        """
        if self.should_scale_up():
            self.current_workers = min(self.current_workers + 1, self.max_workers)
            logger.info(f"Scaling up to {self.current_workers} workers")
        elif self.should_scale_down():
            self.current_workers = max(self.current_workers - 1, self.min_workers)
            logger.info(f"Scaling down to {self.current_workers} workers")
        
        return self.current_workers
    
    def estimate_task_duration(self, task: ProcessingTask) -> float:
        """Estimate task duration based on historical data.
        
        Args:
            task: Task to estimate duration for
            
        Returns:
            Estimated duration in seconds
        """
        if task.estimated_duration > 0:
            return task.estimated_duration
        
        # Use average task time if available
        if self.average_task_time > 0:
            return self.average_task_time
        
        # Default estimate based on task type
        type_estimates = {
            "audio_processing": 30.0,
            "video_processing": 60.0,
            "transcription": 45.0,
            "segmentation": 15.0,
        }
        
        return type_estimates.get(task.task_type, 30.0)
    
    def update_performance_metrics(self, result: ProcessingResult) -> None:
        """Update performance metrics with completed task.
        
        Args:
            result: Completed task result
        """
        self.completed_tasks += 1
        self.total_processing_time += result.processing_time
        self.average_task_time = self.total_processing_time / self.completed_tasks
        
        logger.debug(
            f"Task {result.task_id} completed in {result.processing_time:.2f}s, "
            f"avg: {self.average_task_time:.2f}s"
        )


class ParallelProcessor:
    """High-performance parallel processor for audio/video operations."""
    
    def __init__(
        self,
        max_workers: Optional[int] = None,
        memory_limit_mb: float = 2048.0,
        use_async: bool = True,
        chunk_size: int = 1000
    ):
        """Initialize parallel processor.
        
        Args:
            max_workers: Maximum number of worker processes
            memory_limit_mb: Memory limit for processing
            use_async: Whether to use async processing
            chunk_size: Size of processing chunks
        """
        self.max_workers = max_workers or min(psutil.cpu_count(), 8)
        self.memory_limit_mb = memory_limit_mb
        self.use_async = use_async
        self.chunk_size = chunk_size
        
        self.worker_pool = AdaptiveWorkerPool(
            min_workers=1,
            max_workers=self.max_workers,
            memory_limit_mb=memory_limit_mb
        )
        
        # Caching for repeated operations
        self.result_cache: Dict[str, Any] = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def get_cache_key(self, func: Callable, *args, **kwargs) -> str:
        """Generate cache key for function and arguments.
        
        Args:
            func: Function to cache
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Cache key string
        """
        import hashlib
        
        # Create a hash of function name and arguments
        func_name = func.__name__
        args_str = str(args) + str(sorted(kwargs.items()))
        
        hash_obj = hashlib.md5(args_str.encode())
        return f"{func_name}_{hash_obj.hexdigest()[:8]}"
    
    def get_cached_result(self, cache_key: str) -> Optional[Any]:
        """Get cached result if available.
        
        Args:
            cache_key: Cache key to look up
            
        Returns:
            Cached result or None
        """
        if cache_key in self.result_cache:
            self.cache_hits += 1
            logger.debug(f"Cache hit for {cache_key}")
            return self.result_cache[cache_key]
        
        self.cache_misses += 1
        return None
    
    def cache_result(self, cache_key: str, result: Any) -> None:
        """Cache a result.
        
        Args:
            cache_key: Cache key
            result: Result to cache
        """
        # Simple cache size management
        if len(self.result_cache) > 1000:
            # Remove oldest 20% of entries
            to_remove = list(self.result_cache.keys())[:200]
            for key in to_remove:
                del self.result_cache[key]
        
        self.result_cache[cache_key] = result
        logger.debug(f"Cached result for {cache_key}")
    
    async def process_batch_async(
        self,
        tasks: List[ProcessingTask],
        process_func: Callable,
        progress_callback: Optional[Callable] = None
    ) -> List[ProcessingResult]:
        """Process a batch of tasks asynchronously.
        
        Args:
            tasks: List of tasks to process
            process_func: Function to process each task
            progress_callback: Optional progress callback
            
        Returns:
            List of processing results
        """
        semaphore = asyncio.Semaphore(self.worker_pool.current_workers)
        results = []
        
        async def process_single_task(task: ProcessingTask) -> ProcessingResult:
            """Process a single task with resource management."""
            async with semaphore:
                start_time = time.perf_counter()
                
                try:
                    # Check cache first
                    cache_key = self.get_cache_key(process_func, task.input_data)
                    cached_result = self.get_cached_result(cache_key)
                    
                    if cached_result is not None:
                        processing_time = time.perf_counter() - start_time
                        return ProcessingResult(
                            task_id=task.task_id,
                            result=cached_result,
                            processing_time=processing_time,
                            memory_used_mb=0.0,  # No memory used for cached result
                            success=True
                        )
                    
                    # Monitor resources before processing
                    memory_before = psutil.virtual_memory().used
                    
                    # Process the task
                    if asyncio.iscoroutinefunction(process_func):
                        result = await process_func(task.input_data)
                    else:
                        # Run sync function in thread pool
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(None, process_func, task.input_data)
                    
                    # Monitor resources after processing
                    memory_after = psutil.virtual_memory().used
                    memory_used_mb = (memory_after - memory_before) / (1024 * 1024)
                    processing_time = time.perf_counter() - start_time
                    
                    # Cache the result
                    self.cache_result(cache_key, result)
                    
                    # Update progress
                    if progress_callback:
                        await progress_callback(task, result)
                    
                    # Update performance metrics
                    result_obj = ProcessingResult(
                        task_id=task.task_id,
                        result=result,
                        processing_time=processing_time,
                        memory_used_mb=memory_used_mb,
                        success=True
                    )
                    
                    self.worker_pool.update_performance_metrics(result_obj)
                    
                    return result_obj
                    
                except Exception as e:
                    processing_time = time.perf_counter() - start_time
                    
                    error_result = ProcessingResult(
                        task_id=task.task_id,
                        result=None,
                        processing_time=processing_time,
                        memory_used_mb=0.0,
                        success=False,
                        error=e
                    )
                    
                    logger.error(f"Task {task.task_id} failed: {e}")
                    return error_result
        
        # Process tasks with adaptive worker adjustment
        task_coroutines = [process_single_task(task) for task in tasks]
        
        # Process in chunks to allow worker adjustment
        for i in range(0, len(task_coroutines), self.chunk_size):
            chunk = task_coroutines[i:i + self.chunk_size]
            
            # Adjust worker count before processing chunk
            self.worker_pool.adjust_worker_count()
            
            chunk_results = await asyncio.gather(*chunk, return_exceptions=True)
            
            for result in chunk_results:
                if isinstance(result, Exception):
                    # Convert exception to failed result
                    results.append(ProcessingResult(
                        task_id=f"unknown_{i}",
                        result=None,
                        processing_time=0.0,
                        memory_used_mb=0.0,
                        success=False,
                        error=result
                    ))
                else:
                    results.append(result)
        
        return results
    
    def process_batch_sync(
        self,
        tasks: List[ProcessingTask],
        process_func: Callable,
        progress_callback: Optional[Callable] = None
    ) -> List[ProcessingResult]:
        """Process a batch of tasks synchronously using multiprocessing.
        
        Args:
            tasks: List of tasks to process
            process_func: Function to process each task
            progress_callback: Optional progress callback
            
        Returns:
            List of processing results
        """
        results = []
        
        def process_single_task_sync(task: ProcessingTask) -> ProcessingResult:
            """Process a single task synchronously."""
            start_time = time.perf_counter()
            
            try:
                # Check cache first
                cache_key = self.get_cache_key(process_func, task.input_data)
                cached_result = self.get_cached_result(cache_key)
                
                if cached_result is not None:
                    processing_time = time.perf_counter() - start_time
                    return ProcessingResult(
                        task_id=task.task_id,
                        result=cached_result,
                        processing_time=processing_time,
                        memory_used_mb=0.0,
                        success=True
                    )
                
                # Process the task
                result = process_func(task.input_data)
                processing_time = time.perf_counter() - start_time
                
                # Cache the result
                self.cache_result(cache_key, result)
                
                return ProcessingResult(
                    task_id=task.task_id,
                    result=result,
                    processing_time=processing_time,
                    memory_used_mb=0.0,  # Memory tracking not available in sync mode
                    success=True
                )
                
            except Exception as e:
                processing_time = time.perf_counter() - start_time
                
                return ProcessingResult(
                    task_id=task.task_id,
                    result=None,
                    processing_time=processing_time,
                    memory_used_mb=0.0,
                    success=False,
                    error=e
                )
        
        # Use ProcessPoolExecutor for CPU-bound tasks
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(process_single_task_sync, task): task for task in tasks}
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if progress_callback:
                        progress_callback(task, result)
                        
                except Exception as e:
                    results.append(ProcessingResult(
                        task_id=task.task_id,
                        result=None,
                        processing_time=0.0,
                        memory_used_mb=0.0,
                        success=False,
                        error=e
                    ))
        
        return results
    
    async def process_batch(
        self,
        tasks: List[ProcessingTask],
        process_func: Callable,
        progress_callback: Optional[Callable] = None
    ) -> List[ProcessingResult]:
        """Process a batch of tasks using optimal method.
        
        Args:
            tasks: List of tasks to process
            process_func: Function to process each task
            progress_callback: Optional progress callback
            
        Returns:
            List of processing results
        """
        if not tasks:
            return []
        
        # Log processing start
        logger.info(
            f"Starting batch processing: {len(tasks)} tasks, "
            f"max_workers={self.max_workers}, "
            f"async={'yes' if self.use_async else 'no'}"
        )
        
        start_time = time.perf_counter()
        
        try:
            if self.use_async:
                results = await self.process_batch_async(tasks, process_func, progress_callback)
            else:
                results = self.process_batch_sync(tasks, process_func, progress_callback)
            
            # Log processing completion
            total_time = time.perf_counter() - start_time
            successful_tasks = sum(1 for r in results if r.success)
            failed_tasks = len(results) - successful_tasks
            
            logger.info(
                f"Batch processing completed: {successful_tasks}/{len(tasks)} successful, "
                f"{failed_tasks} failed, total time: {total_time:.2f}s, "
                f"cache hit rate: {self.cache_hits / (self.cache_hits + self.cache_misses) * 100:.1f}%"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            raise ProcessingError(
                "Batch processing failed",
                cause=e,
                context={
                    "task_count": len(tasks),
                    "max_workers": self.max_workers,
                    "use_async": self.use_async
                }
            )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics.
        
        Returns:
            Performance statistics dictionary
        """
        total_cache_requests = self.cache_hits + self.cache_misses
        cache_hit_rate = (self.cache_hits / total_cache_requests * 100) if total_cache_requests > 0 else 0
        
        return {
            "worker_pool": {
                "current_workers": self.worker_pool.current_workers,
                "min_workers": self.worker_pool.min_workers,
                "max_workers": self.worker_pool.max_workers,
                "completed_tasks": self.worker_pool.completed_tasks,
                "average_task_time": self.worker_pool.average_task_time,
            },
            "caching": {
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_hit_rate": cache_hit_rate,
                "cache_size": len(self.result_cache),
            },
            "resources": self.worker_pool.resource_monitor.update(),
        }