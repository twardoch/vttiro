# this_file: src/vttiro/core/resilience.py
"""Resilience patterns for VTTiro: retry logic, circuit breakers, and timeout management.

This module provides enterprise-grade resilience patterns to handle failures gracefully:
- Exponential backoff with jitter for retries
- Circuit breaker pattern to prevent cascading failures
- Comprehensive timeout management for different operation types
- Rate limiting compliance and monitoring
- Health checks and failure detection

Used by:
- Provider implementations for robust API interactions
- Core transcription pipeline for reliable processing
- Network operations for timeout and retry management
- Resource management for proper cleanup on failures
"""

import asyncio
import functools
import random
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Type, TypeVar, Union

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from .errors import (
    APIError, AuthenticationError, QuotaExceededError, RateLimitError,
    TimeoutError, VttiroError, ProviderUnavailableError
)

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"          # Circuit is open, failing fast
    HALF_OPEN = "half_open" # Testing if service recovered


@dataclass
class RetryConfig:
    """Configuration for retry behavior with exponential backoff."""
    
    max_attempts: int = 3
    base_delay: float = 1.0      # Initial delay in seconds
    max_delay: float = 60.0      # Maximum delay in seconds
    exponential_base: float = 2.0 # Exponential backoff multiplier
    jitter: bool = True          # Add random jitter to prevent thundering herd
    backoff_factor: float = 0.1  # Jitter factor (10% of delay)
    
    # Which exceptions should trigger retries
    retryable_exceptions: tuple = (
        APIError,
        RateLimitError,
        TimeoutError,
        ProviderUnavailableError,
        ConnectionError,
        asyncio.TimeoutError
    )
    
    # Which exceptions should NOT trigger retries
    non_retryable_exceptions: tuple = (
        AuthenticationError,
        QuotaExceededError,
        ValueError,
        TypeError
    )


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    
    failure_threshold: int = 5          # Failures before opening circuit
    success_threshold: int = 3          # Successes to close circuit from half-open
    timeout_duration: float = 60.0     # Time to wait before trying half-open
    half_open_max_calls: int = 3        # Max calls allowed in half-open state
    
    # Metrics window for failure rate calculation
    window_duration: float = 300.0     # 5 minutes rolling window
    failure_rate_threshold: float = 0.5 # 50% failure rate triggers opening


@dataclass
class TimeoutConfig:
    """Configuration for different operation timeouts."""
    
    # Network operation timeouts
    connection_timeout: float = 10.0    # Initial connection establishment
    read_timeout: float = 300.0         # Reading response data
    total_timeout: float = 600.0        # Total operation timeout
    
    # File operation timeouts
    file_read_timeout: float = 30.0     # File I/O operations
    file_write_timeout: float = 60.0    # File writing operations
    
    # Processing timeouts
    audio_processing_timeout: float = 120.0  # Audio preprocessing
    transcription_timeout: float = 600.0     # AI transcription
    
    # Provider-specific timeouts
    provider_timeouts: Dict[str, float] = field(default_factory=lambda: {
        'gemini': 300.0,
        'openai': 180.0,
        'assemblyai': 600.0,
        'deepgram': 300.0
    })


class ResilienceMetrics:
    """Tracks metrics for resilience patterns."""
    
    def __init__(self):
        """Initialize metrics tracking."""
        self.retry_stats: Dict[str, Dict[str, int]] = {}
        self.circuit_stats: Dict[str, Dict[str, Any]] = {}
        self.timeout_stats: Dict[str, Dict[str, int]] = {}
        self.start_time = time.time()
    
    def record_retry(self, operation: str, attempt: int, success: bool, duration: float) -> None:
        """Record retry attempt metrics."""
        if operation not in self.retry_stats:
            self.retry_stats[operation] = {
                'total_attempts': 0,
                'successful_retries': 0,
                'failed_retries': 0,
                'total_duration': 0.0
            }
        
        stats = self.retry_stats[operation]
        stats['total_attempts'] += 1
        stats['total_duration'] += duration
        
        if success:
            if attempt > 1:  # Only count as retry if not first attempt
                stats['successful_retries'] += 1
        else:
            stats['failed_retries'] += 1
    
    def record_circuit_state_change(self, circuit_name: str, old_state: CircuitState, new_state: CircuitState) -> None:
        """Record circuit breaker state changes."""
        if circuit_name not in self.circuit_stats:
            self.circuit_stats[circuit_name] = {
                'state_changes': [],
                'total_opens': 0,
                'total_closes': 0,
                'current_state': CircuitState.CLOSED
            }
        
        stats = self.circuit_stats[circuit_name]
        stats['state_changes'].append({
            'timestamp': time.time(),
            'from': old_state.value,
            'to': new_state.value
        })
        stats['current_state'] = new_state
        
        if new_state == CircuitState.OPEN:
            stats['total_opens'] += 1
        elif new_state == CircuitState.CLOSED:
            stats['total_closes'] += 1
    
    def record_timeout(self, operation: str, timeout_type: str, duration: float, timed_out: bool) -> None:
        """Record timeout events."""
        if operation not in self.timeout_stats:
            self.timeout_stats[operation] = {
                'total_operations': 0,
                'timeouts': 0,
                'average_duration': 0.0,
                'max_duration': 0.0
            }
        
        stats = self.timeout_stats[operation]
        stats['total_operations'] += 1
        
        if timed_out:
            stats['timeouts'] += 1
        
        # Update duration stats
        old_avg = stats['average_duration']
        stats['average_duration'] = (old_avg * (stats['total_operations'] - 1) + duration) / stats['total_operations']
        stats['max_duration'] = max(stats['max_duration'], duration)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all resilience metrics."""
        uptime = time.time() - self.start_time
        
        return {
            'uptime_seconds': uptime,
            'retry_operations': len(self.retry_stats),
            'circuit_breakers': len(self.circuit_stats),
            'timeout_operations': len(self.timeout_stats),
            'retry_stats': self.retry_stats,
            'circuit_stats': self.circuit_stats,
            'timeout_stats': self.timeout_stats
        }


class CircuitBreaker:
    """Circuit breaker implementation for preventing cascading failures."""
    
    def __init__(self, name: str, config: CircuitBreakerConfig, metrics: ResilienceMetrics):
        """Initialize circuit breaker.
        
        Args:
            name: Unique name for this circuit breaker
            config: Circuit breaker configuration
            metrics: Metrics tracker for recording events
        """
        self.name = name
        self.config = config
        self.metrics = metrics
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.half_open_calls = 0
        
        # Rolling window for failure rate calculation
        self.call_history: List[Dict[str, Any]] = []
    
    def _record_call(self, success: bool, duration: float):
        """Record a call in the rolling window."""
        now = time.time()
        
        # Add new call
        self.call_history.append({
            'timestamp': now,
            'success': success,
            'duration': duration
        })
        
        # Remove old calls outside the window
        cutoff_time = now - self.config.window_duration
        self.call_history = [call for call in self.call_history if call['timestamp'] >= cutoff_time]
    
    def _get_failure_rate(self) -> float:
        """Calculate current failure rate in the rolling window."""
        if not self.call_history:
            return 0.0
        
        total_calls = len(self.call_history)
        failed_calls = sum(1 for call in self.call_history if not call['success'])
        
        return failed_calls / total_calls
    
    def _should_open_circuit(self) -> bool:
        """Check if circuit should be opened."""
        # Check failure count threshold
        if self.failure_count >= self.config.failure_threshold:
            return True
        
        # Check failure rate threshold
        if len(self.call_history) >= 10:  # Minimum sample size
            failure_rate = self._get_failure_rate()
            if failure_rate >= self.config.failure_rate_threshold:
                return True
        
        return False
    
    def _change_state(self, new_state: CircuitState) -> None:
        """Change circuit breaker state and record metrics."""
        old_state = self.state
        self.state = new_state
        self.metrics.record_circuit_state_change(self.name, old_state, new_state)
        
        logger.info(f"Circuit breaker '{self.name}' state changed: {old_state.value} -> {new_state.value}")
        
        if new_state == CircuitState.OPEN:
            self.last_failure_time = time.time()
            self.half_open_calls = 0
        elif new_state == CircuitState.CLOSED:
            self.failure_count = 0
            self.success_count = 0
            self.half_open_calls = 0
        elif new_state == CircuitState.HALF_OPEN:
            self.half_open_calls = 0
    
    def can_execute(self) -> bool:
        """Check if execution is allowed based on circuit state."""
        if self.state == CircuitState.CLOSED:
            return True
        
        if self.state == CircuitState.OPEN:
            # Check if timeout has elapsed
            if (self.last_failure_time and 
                time.time() - self.last_failure_time >= self.config.timeout_duration):
                self._change_state(CircuitState.HALF_OPEN)
                return True
            return False
        
        if self.state == CircuitState.HALF_OPEN:
            return self.half_open_calls < self.config.half_open_max_calls
        
        return False
    
    def record_success(self, duration: float) -> None:
        """Record a successful operation."""
        self._record_call(success=True, duration=duration)
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self._change_state(CircuitState.CLOSED)
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_count = max(0, self.failure_count - 1)
    
    def record_failure(self, duration: float) -> None:
        """Record a failed operation."""
        self._record_call(success=False, duration=duration)
        
        if self.state == CircuitState.CLOSED:
            self.failure_count += 1
            if self._should_open_circuit():
                self._change_state(CircuitState.OPEN)
        elif self.state == CircuitState.HALF_OPEN:
            self._change_state(CircuitState.OPEN)
    
    def record_call_attempt(self) -> None:
        """Record that a call attempt was made (for half-open state tracking)."""
        if self.state == CircuitState.HALF_OPEN:
            self.half_open_calls += 1


class ResilienceManager:
    """Central manager for all resilience patterns."""
    
    def __init__(
        self,
        retry_config: Optional[RetryConfig] = None,
        circuit_config: Optional[CircuitBreakerConfig] = None,
        timeout_config: Optional[TimeoutConfig] = None
    ):
        """Initialize resilience manager.
        
        Args:
            retry_config: Retry behavior configuration
            circuit_config: Circuit breaker configuration  
            timeout_config: Timeout management configuration
        """
        self.retry_config = retry_config or RetryConfig()
        self.circuit_config = circuit_config or CircuitBreakerConfig()
        self.timeout_config = timeout_config or TimeoutConfig()
        
        self.metrics = ResilienceMetrics()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
    
    def get_circuit_breaker(self, name: str) -> CircuitBreaker:
        """Get or create a circuit breaker by name."""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(name, self.circuit_config, self.metrics)
        return self.circuit_breakers[name]
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate retry delay with exponential backoff and jitter."""
        if attempt <= 1:
            return 0.0
        
        # Exponential backoff: base_delay * (exponential_base ^ (attempt - 2))
        delay = self.retry_config.base_delay * (self.retry_config.exponential_base ** (attempt - 2))
        
        # Cap at maximum delay
        delay = min(delay, self.retry_config.max_delay)
        
        # Add jitter to prevent thundering herd
        if self.retry_config.jitter:
            jitter_amount = delay * self.retry_config.backoff_factor
            jitter = random.uniform(-jitter_amount, jitter_amount)
            delay = max(0.1, delay + jitter)  # Ensure minimum delay
        
        return delay
    
    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if an exception should trigger a retry."""
        if attempt >= self.retry_config.max_attempts:
            return False
        
        # Check if it's a non-retryable exception
        if isinstance(exception, self.retry_config.non_retryable_exceptions):
            return False
        
        # Check if it's a retryable exception
        if isinstance(exception, self.retry_config.retryable_exceptions):
            return True
        
        # For rate limit errors, check if we have retry-after information
        if isinstance(exception, (RateLimitError, QuotaExceededError)):
            retry_after = getattr(exception, 'retry_after', None)
            if retry_after and retry_after > self.retry_config.max_delay:
                return False  # Don't retry if wait time is too long
            return True
        
        # Default to not retrying unknown exceptions
        return False
    
    def get_timeout_for_operation(self, operation: str, provider: Optional[str] = None) -> float:
        """Get appropriate timeout for an operation type."""
        # Provider-specific timeouts
        if provider and provider in self.timeout_config.provider_timeouts:
            return self.timeout_config.provider_timeouts[provider]
        
        # Operation-specific timeouts
        timeout_map = {
            'connection': self.timeout_config.connection_timeout,
            'read': self.timeout_config.read_timeout,
            'total': self.timeout_config.total_timeout,
            'file_read': self.timeout_config.file_read_timeout,
            'file_write': self.timeout_config.file_write_timeout,
            'audio_processing': self.timeout_config.audio_processing_timeout,
            'transcription': self.timeout_config.transcription_timeout,
        }
        
        return timeout_map.get(operation, self.timeout_config.total_timeout)
    
    @contextmanager
    def timeout_context(self, operation: str, provider: Optional[str] = None, custom_timeout: Optional[float] = None):
        """Context manager for timeout management."""
        timeout_duration = custom_timeout or self.get_timeout_for_operation(operation, provider)
        start_time = time.time()
        timed_out = False
        
        try:
            # Set up timeout (this is a basic implementation - in practice you'd use proper timeout mechanisms)
            yield timeout_duration
        except Exception as e:
            if "timeout" in str(e).lower() or isinstance(e, (asyncio.TimeoutError, TimeoutError)):
                timed_out = True
            raise
        finally:
            duration = time.time() - start_time
            self.metrics.record_timeout(operation, "context", duration, timed_out)
    
    async def with_retry(
        self,
        operation: Callable[..., Awaitable[T]],
        operation_name: str,
        circuit_breaker_name: Optional[str] = None,
        *args,
        **kwargs
    ) -> T:
        """Execute an async operation with retry logic and optional circuit breaker."""
        circuit = self.get_circuit_breaker(circuit_breaker_name) if circuit_breaker_name else None
        
        last_exception = None
        start_time = time.time()
        
        for attempt in range(1, self.retry_config.max_attempts + 1):
            # Check circuit breaker
            if circuit and not circuit.can_execute():
                raise ProviderUnavailableError(
                    f"Circuit breaker '{circuit_breaker_name}' is open",
                    provider=circuit_breaker_name or "unknown"
                )
            
            if circuit:
                circuit.record_call_attempt()
            
            try:
                # Execute the operation
                result = await operation(*args, **kwargs)
                
                # Record success
                duration = time.time() - start_time
                self.metrics.record_retry(operation_name, attempt, True, duration)
                
                if circuit:
                    circuit.record_success(duration)
                
                if attempt > 1:
                    logger.info(f"Operation '{operation_name}' succeeded on attempt {attempt}")
                
                return result
                
            except Exception as e:
                last_exception = e
                operation_duration = time.time() - start_time
                
                if circuit:
                    circuit.record_failure(operation_duration)
                
                # Check if we should retry
                if not self.should_retry(e, attempt):
                    logger.error(f"Operation '{operation_name}' failed (non-retryable): {e}")
                    self.metrics.record_retry(operation_name, attempt, False, operation_duration)
                    raise
                
                if attempt == self.retry_config.max_attempts:
                    logger.error(f"Operation '{operation_name}' failed after {attempt} attempts: {e}")
                    self.metrics.record_retry(operation_name, attempt, False, operation_duration)
                    raise
                
                # Calculate delay and wait
                delay = self.calculate_delay(attempt)
                
                # For rate limiting, use the suggested retry-after if available
                if isinstance(e, (RateLimitError, QuotaExceededError)):
                    retry_after = getattr(e, 'retry_after', None)
                    if retry_after:
                        delay = min(retry_after, self.retry_config.max_delay)
                
                logger.warning(f"Operation '{operation_name}' failed on attempt {attempt}/{self.retry_config.max_attempts}: {e}. Retrying in {delay:.1f}s")
                
                if delay > 0:
                    await asyncio.sleep(delay)
        
        # This should never be reached, but just in case
        raise last_exception
    
    def with_retry_sync(
        self,
        operation: Callable[..., T],
        operation_name: str,
        circuit_breaker_name: Optional[str] = None,
        *args,
        **kwargs
    ) -> T:
        """Execute a synchronous operation with retry logic and optional circuit breaker."""
        circuit = self.get_circuit_breaker(circuit_breaker_name) if circuit_breaker_name else None
        
        last_exception = None
        start_time = time.time()
        
        for attempt in range(1, self.retry_config.max_attempts + 1):
            # Check circuit breaker
            if circuit and not circuit.can_execute():
                raise ProviderUnavailableError(
                    f"Circuit breaker '{circuit_breaker_name}' is open",
                    provider=circuit_breaker_name or "unknown"
                )
            
            if circuit:
                circuit.record_call_attempt()
            
            try:
                # Execute the operation
                result = operation(*args, **kwargs)
                
                # Record success
                duration = time.time() - start_time
                self.metrics.record_retry(operation_name, attempt, True, duration)
                
                if circuit:
                    circuit.record_success(duration)
                
                if attempt > 1:
                    logger.info(f"Operation '{operation_name}' succeeded on attempt {attempt}")
                
                return result
                
            except Exception as e:
                last_exception = e
                operation_duration = time.time() - start_time
                
                if circuit:
                    circuit.record_failure(operation_duration)
                
                # Check if we should retry
                if not self.should_retry(e, attempt):
                    logger.error(f"Operation '{operation_name}' failed (non-retryable): {e}")
                    self.metrics.record_retry(operation_name, attempt, False, operation_duration)
                    raise
                
                if attempt == self.retry_config.max_attempts:
                    logger.error(f"Operation '{operation_name}' failed after {attempt} attempts: {e}")
                    self.metrics.record_retry(operation_name, attempt, False, operation_duration)
                    raise
                
                # Calculate delay and wait
                delay = self.calculate_delay(attempt)
                
                # For rate limiting, use the suggested retry-after if available
                if isinstance(e, (RateLimitError, QuotaExceededError)):
                    retry_after = getattr(e, 'retry_after', None)
                    if retry_after:
                        delay = min(retry_after, self.retry_config.max_delay)
                
                logger.warning(f"Operation '{operation_name}' failed on attempt {attempt}/{self.retry_config.max_attempts}: {e}. Retrying in {delay:.1f}s")
                
                if delay > 0:
                    time.sleep(delay)
        
        # This should never be reached, but just in case
        raise last_exception
    
    def retry_decorator(self, operation_name: str, circuit_breaker_name: Optional[str] = None):
        """Decorator for automatic retry logic."""
        def decorator(func):
            if asyncio.iscoroutinefunction(func):
                @functools.wraps(func)
                async def async_wrapper(*args, **kwargs):
                    return await self.with_retry(func, operation_name, circuit_breaker_name, *args, **kwargs)
                return async_wrapper
            else:
                @functools.wraps(func)
                def sync_wrapper(*args, **kwargs):
                    return self.with_retry_sync(func, operation_name, circuit_breaker_name, *args, **kwargs)
                return sync_wrapper
        return decorator
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status of all resilience components."""
        circuit_status = {}
        for name, circuit in self.circuit_breakers.items():
            circuit_status[name] = {
                'state': circuit.state.value,
                'failure_count': circuit.failure_count,
                'success_count': circuit.success_count,
                'failure_rate': circuit._get_failure_rate(),
                'calls_in_window': len(circuit.call_history)
            }
        
        return {
            'status': 'healthy' if all(c.state != CircuitState.OPEN for c in self.circuit_breakers.values()) else 'degraded',
            'circuit_breakers': circuit_status,
            'metrics_summary': self.metrics.get_summary(),
            'configuration': {
                'max_retries': self.retry_config.max_attempts,
                'max_delay': self.retry_config.max_delay,
                'circuit_failure_threshold': self.circuit_config.failure_threshold,
                'default_timeout': self.timeout_config.total_timeout
            }
        }


# Global resilience manager instance
_resilience_manager: Optional[ResilienceManager] = None


def get_resilience_manager() -> ResilienceManager:
    """Get global resilience manager instance."""
    global _resilience_manager
    if _resilience_manager is None:
        _resilience_manager = ResilienceManager()
    return _resilience_manager


def configure_resilience(
    retry_config: Optional[RetryConfig] = None,
    circuit_config: Optional[CircuitBreakerConfig] = None,
    timeout_config: Optional[TimeoutConfig] = None
):
    """Configure global resilience settings."""
    global _resilience_manager
    _resilience_manager = ResilienceManager(retry_config, circuit_config, timeout_config)


# Convenience decorators using global manager
def with_retry(operation_name: str, circuit_breaker_name: Optional[str] = None):
    """Decorator for automatic retry with global resilience manager."""
    return get_resilience_manager().retry_decorator(operation_name, circuit_breaker_name)


def timeout_for(operation: str, provider: Optional[str] = None, custom_timeout: Optional[float] = None):
    """Context manager for timeout management using global resilience manager."""
    return get_resilience_manager().timeout_context(operation, provider, custom_timeout)