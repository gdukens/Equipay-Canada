"""
EquiPay Canada - Memory Monitor
===============================

Monitors memory usage and provides backpressure for streaming operations.

Features:
- Real-time memory monitoring
- Automatic chunk size adjustment
- Memory limit enforcement
- Garbage collection triggers
"""

import gc
import logging
import psutil
import os
from typing import Optional, Callable
from dataclasses import dataclass
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """Current memory statistics."""
    used_mb: float
    available_mb: float
    percent: float
    peak_mb: float
    
    def __repr__(self):
        return (
            f"MemoryStats(used={self.used_mb:.1f}MB, "
            f"available={self.available_mb:.1f}MB, "
            f"percent={self.percent:.1f}%, "
            f"peak={self.peak_mb:.1f}MB)"
        )


class MemoryMonitor:
    """
    Monitors memory usage and provides backpressure control.
    
    Usage:
        monitor = MemoryMonitor(limit_mb=2048)
        
        while processing:
            if monitor.should_pause():
                monitor.collect_garbage()
            # ... process data ...
            
        print(monitor.stats())
    """
    
    def __init__(
        self,
        limit_mb: Optional[float] = None,
        warning_threshold: float = 0.75,
        critical_threshold: float = 0.90,
        auto_gc: bool = True
    ):
        """
        Initialize memory monitor.
        
        Args:
            limit_mb: Hard memory limit in MB (None = system available)
            warning_threshold: Fraction of limit to trigger warning
            critical_threshold: Fraction of limit to trigger backpressure
            auto_gc: Automatically run garbage collection when needed
        """
        self.limit_mb = limit_mb or self._get_available_memory()
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.auto_gc = auto_gc
        
        self._peak_mb = 0.0
        self._process = psutil.Process(os.getpid())
        
    def _get_available_memory(self) -> float:
        """Get available system memory in MB."""
        mem = psutil.virtual_memory()
        return mem.available / (1024 * 1024)
    
    def _get_process_memory(self) -> float:
        """Get current process memory usage in MB."""
        return self._process.memory_info().rss / (1024 * 1024)
    
    def current(self) -> float:
        """Get current memory usage in MB."""
        used = self._get_process_memory()
        self._peak_mb = max(self._peak_mb, used)
        return used
    
    def peak(self) -> float:
        """Get peak memory usage in MB."""
        return self._peak_mb
    
    def reset_peak(self):
        """Reset peak memory tracking."""
        self._peak_mb = self._get_process_memory()
    
    def stats(self) -> MemoryStats:
        """Get current memory statistics."""
        used = self.current()
        available = self._get_available_memory()
        percent = (used / self.limit_mb) * 100 if self.limit_mb > 0 else 0
        
        return MemoryStats(
            used_mb=used,
            available_mb=available,
            percent=percent,
            peak_mb=self._peak_mb
        )
    
    def should_pause(self) -> bool:
        """Check if processing should pause due to memory pressure."""
        usage_ratio = self.current() / self.limit_mb
        return usage_ratio >= self.critical_threshold
    
    def should_warn(self) -> bool:
        """Check if memory warning should be issued."""
        usage_ratio = self.current() / self.limit_mb
        return usage_ratio >= self.warning_threshold
    
    def collect_garbage(self, full: bool = True) -> float:
        """
        Run garbage collection.
        
        Args:
            full: If True, run full collection (generations 0-2)
            
        Returns:
            Memory freed in MB
        """
        before = self.current()
        
        if full:
            gc.collect(0)
            gc.collect(1)
            gc.collect(2)
        else:
            gc.collect(0)
        
        after = self.current()
        freed = before - after
        
        if freed > 10:  # Only log if significant
            logger.info(f"Garbage collection freed {freed:.1f}MB")
        
        return freed
    
    def check_and_collect(self) -> bool:
        """
        Check memory and collect garbage if needed.
        
        Returns:
            True if garbage collection was triggered
        """
        if self.should_pause() and self.auto_gc:
            self.collect_garbage()
            return True
        return False
    
    def get_optimal_chunk_size(
        self,
        row_size_bytes: int = 500,
        target_chunk_mb: float = 200,
        min_rows: int = 10_000,
        max_rows: int = 1_000_000
    ) -> int:
        """
        Calculate optimal chunk size based on available memory.
        
        Args:
            row_size_bytes: Estimated bytes per row
            target_chunk_mb: Target chunk size in MB
            min_rows: Minimum chunk size in rows
            max_rows: Maximum chunk size in rows
            
        Returns:
            Optimal chunk size in rows
        """
        # Check available headroom
        stats = self.stats()
        available = self.limit_mb - stats.used_mb
        
        # Use smaller chunks if memory is tight
        if stats.percent > 70:
            target_chunk_mb = target_chunk_mb * 0.5
        elif stats.percent > 50:
            target_chunk_mb = target_chunk_mb * 0.75
        
        # Calculate rows per chunk
        rows = int((target_chunk_mb * 1024 * 1024) / row_size_bytes)
        
        # Clamp to bounds
        return max(min_rows, min(rows, max_rows))
    
    @contextmanager
    def track_operation(self, name: str = "operation"):
        """
        Context manager to track memory for an operation.
        
        Usage:
            with monitor.track_operation("training"):
                model.fit(X, y)
        """
        before = self.current()
        self.reset_peak()
        
        try:
            yield
        finally:
            after = self.current()
            peak = self.peak()
            delta = after - before
            
            logger.info(
                f"Memory [{name}]: "
                f"before={before:.1f}MB, after={after:.1f}MB, "
                f"delta={delta:+.1f}MB, peak={peak:.1f}MB"
            )
    
    def enforce_limit(self, hard: bool = False):
        """
        Enforce memory limit.
        
        Args:
            hard: If True, raise exception when limit exceeded
        """
        if self.should_pause():
            self.collect_garbage()
            
            if self.should_pause() and hard:
                stats = self.stats()
                raise MemoryError(
                    f"Memory limit exceeded: {stats.used_mb:.1f}MB / {self.limit_mb:.1f}MB"
                )
    
    def set_limit(self, limit_mb: float):
        """Set new memory limit."""
        self.limit_mb = limit_mb
        logger.info(f"Memory limit set to {limit_mb:.1f}MB")
    
    def __repr__(self):
        stats = self.stats()
        return f"MemoryMonitor({stats.used_mb:.1f}/{self.limit_mb:.1f}MB, {stats.percent:.1f}%)"


# Global monitor instance
_global_monitor: Optional[MemoryMonitor] = None


def get_memory_monitor(limit_mb: Optional[float] = None) -> MemoryMonitor:
    """Get or create global memory monitor."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = MemoryMonitor(limit_mb=limit_mb)
    return _global_monitor
