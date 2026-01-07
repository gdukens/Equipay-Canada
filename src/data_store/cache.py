"""
EquiPay Canada - Query Cache
============================

Intelligent caching layer for SQL query results.

Features:
- LRU cache with TTL expiration
- Query fingerprinting for cache keys
- Persistent cache (optional)
- Cache warming
- Statistics tracking

Usage:
    cache = QueryCache(max_size_mb=500, ttl_hours=24)
    
    # Check cache
    if query.fingerprint() in cache:
        return cache.get(query.fingerprint())
    
    # Execute and cache
    result = query.execute()
    cache.set(query.fingerprint(), result)
"""

import time
import pickle
import logging
import hashlib
from typing import Optional, Dict, Any, List, Union, TYPE_CHECKING
from dataclasses import dataclass, field
from pathlib import Path
from collections import OrderedDict
import sys

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A single cache entry with metadata."""
    key: str
    value: Any
    created_at: float
    accessed_at: float
    size_bytes: int
    ttl_seconds: float
    hits: int = 0
    
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl_seconds <= 0:
            return False  # No expiration
        return (time.time() - self.created_at) > self.ttl_seconds
    
    def touch(self):
        """Update access time and hit count."""
        self.accessed_at = time.time()
        self.hits += 1


@dataclass
class CacheStats:
    """Cache statistics."""
    entries: int
    size_mb: float
    max_size_mb: float
    hits: int
    misses: int
    hit_rate: float
    evictions: int
    
    def __repr__(self):
        return (
            f"CacheStats(entries={self.entries}, "
            f"size={self.size_mb:.1f}/{self.max_size_mb:.1f}MB, "
            f"hit_rate={self.hit_rate:.1%})"
        )


class QueryCache:
    """
    LRU cache for SQL query results with TTL support.
    
    Features:
    - Maximum size limit (evicts LRU entries)
    - TTL expiration for entries
    - Persistent storage (optional)
    - Query fingerprinting
    - Statistics tracking
    """
    
    def __init__(
        self,
        max_size_mb: float = 500,
        ttl_hours: float = 24,
        persist_path: Optional[Union[str, Path]] = None
    ):
        """
        Initialize cache.
        
        Args:
            max_size_mb: Maximum cache size in MB
            ttl_hours: Default TTL for entries in hours (0 = no expiration)
            persist_path: Optional path for persistent cache storage
        """
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.default_ttl_seconds = ttl_hours * 3600
        self.persist_path = Path(persist_path) if persist_path else None
        
        # LRU ordered dict
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._current_size_bytes = 0
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        
        # Load persistent cache
        if self.persist_path and self.persist_path.exists():
            self._load_persistent()
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate size of value in bytes."""
        try:
            if hasattr(value, 'memory_usage'):
                # pandas DataFrame
                return int(value.memory_usage(deep=True).sum())
            else:
                return sys.getsizeof(value)
        except Exception:
            return 1024  # Default estimate
    
    def _make_key(self, query_or_key: Union[str, Any]) -> str:
        """Create cache key from query or string."""
        if hasattr(query_or_key, 'fingerprint'):
            return query_or_key.fingerprint()
        elif hasattr(query_or_key, 'sql'):
            sql = query_or_key.sql()
            return hashlib.md5(sql.encode()).hexdigest()
        else:
            return hashlib.md5(str(query_or_key).encode()).hexdigest()
    
    def _evict_lru(self, needed_bytes: int):
        """Evict least recently used entries to free space."""
        while self._current_size_bytes + needed_bytes > self.max_size_bytes and self._cache:
            # Pop oldest (least recently used)
            key, entry = self._cache.popitem(last=False)
            self._current_size_bytes -= entry.size_bytes
            self._evictions += 1
            logger.debug(f"Evicted cache entry: {key[:16]}...")
    
    def _evict_expired(self):
        """Remove all expired entries."""
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.is_expired()
        ]
        for key in expired_keys:
            entry = self._cache.pop(key)
            self._current_size_bytes -= entry.size_bytes
            self._evictions += 1
    
    def get(self, key: Union[str, Any]) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key or query object
            
        Returns:
            Cached value or None if not found/expired
        """
        cache_key = self._make_key(key)
        
        if cache_key not in self._cache:
            self._misses += 1
            return None
        
        entry = self._cache[cache_key]
        
        # Check expiration
        if entry.is_expired():
            self._cache.pop(cache_key)
            self._current_size_bytes -= entry.size_bytes
            self._misses += 1
            return None
        
        # Move to end (most recently used)
        self._cache.move_to_end(cache_key)
        entry.touch()
        self._hits += 1
        
        return entry.value
    
    def set(
        self,
        key: Union[str, Any],
        value: Any,
        ttl_hours: Optional[float] = None
    ):
        """
        Store value in cache.
        
        Args:
            key: Cache key or query object
            value: Value to cache
            ttl_hours: Optional TTL override
        """
        cache_key = self._make_key(key)
        size = self._estimate_size(value)
        ttl = (ttl_hours * 3600) if ttl_hours is not None else self.default_ttl_seconds
        
        # Remove old entry if exists
        if cache_key in self._cache:
            old_entry = self._cache.pop(cache_key)
            self._current_size_bytes -= old_entry.size_bytes
        
        # Evict if needed
        if size > self.max_size_bytes:
            logger.warning(f"Value too large for cache: {size / 1024 / 1024:.1f}MB")
            return
        
        self._evict_lru(size)
        
        # Add new entry
        entry = CacheEntry(
            key=cache_key,
            value=value,
            created_at=time.time(),
            accessed_at=time.time(),
            size_bytes=size,
            ttl_seconds=ttl
        )
        
        self._cache[cache_key] = entry
        self._current_size_bytes += size
        
        logger.debug(f"Cached: {cache_key[:16]}... ({size / 1024:.1f}KB)")
    
    def __contains__(self, key: Union[str, Any]) -> bool:
        """Check if key is in cache (and not expired)."""
        cache_key = self._make_key(key)
        if cache_key not in self._cache:
            return False
        if self._cache[cache_key].is_expired():
            return False
        return True
    
    def delete(self, key: Union[str, Any]):
        """Remove entry from cache."""
        cache_key = self._make_key(key)
        if cache_key in self._cache:
            entry = self._cache.pop(cache_key)
            self._current_size_bytes -= entry.size_bytes
    
    def clear(self):
        """Clear all cache entries."""
        self._cache.clear()
        self._current_size_bytes = 0
        logger.info("Cache cleared")
    
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        self._evict_expired()
        
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
        
        return CacheStats(
            entries=len(self._cache),
            size_mb=self._current_size_bytes / (1024 * 1024),
            max_size_mb=self.max_size_bytes / (1024 * 1024),
            hits=self._hits,
            misses=self._misses,
            hit_rate=hit_rate,
            evictions=self._evictions
        )
    
    def warm(self, queries: List[Any], executor: callable):
        """
        Warm cache with pre-computed queries.
        
        Args:
            queries: List of queries to cache
            executor: Function to execute queries
        """
        logger.info(f"Warming cache with {len(queries)} queries...")
        
        for query in queries:
            key = self._make_key(query)
            if key not in self._cache:
                try:
                    result = executor(query)
                    self.set(key, result)
                except Exception as e:
                    logger.warning(f"Failed to warm cache for query: {e}")
        
        stats = self.stats()
        logger.info(f"Cache warmed: {stats.entries} entries, {stats.size_mb:.1f}MB")
    
    def persist(self, path: Optional[Union[str, Path]] = None):
        """
        Save cache to disk.
        
        Args:
            path: Optional path override
        """
        save_path = Path(path) if path else self.persist_path
        if not save_path:
            logger.warning("No persist path specified")
            return
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Serialize cache entries
        data = {
            'entries': dict(self._cache),
            'stats': {
                'hits': self._hits,
                'misses': self._misses,
                'evictions': self._evictions
            }
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Cache persisted to {save_path}")
    
    def _load_persistent(self):
        """Load cache from disk."""
        try:
            with open(self.persist_path, 'rb') as f:
                data = pickle.load(f)
            
            self._cache = OrderedDict(data.get('entries', {}))
            self._current_size_bytes = sum(e.size_bytes for e in self._cache.values())
            
            stats = data.get('stats', {})
            self._hits = stats.get('hits', 0)
            self._misses = stats.get('misses', 0)
            self._evictions = stats.get('evictions', 0)
            
            # Remove expired entries
            self._evict_expired()
            
            logger.info(f"Loaded {len(self._cache)} cache entries from {self.persist_path}")
            
        except Exception as e:
            logger.warning(f"Failed to load persistent cache: {e}")
    
    def __repr__(self):
        stats = self.stats()
        return f"QueryCache({stats})"


class CachedQueryExecutor:
    """
    Query executor with automatic caching.
    
    Usage:
        executor = CachedQueryExecutor(connection, cache)
        
        # Automatically cached
        result = executor.query("SELECT * FROM lfs WHERE SURVYEAR = 2024")
        
        # Second call is instant (from cache)
        result = executor.query("SELECT * FROM lfs WHERE SURVYEAR = 2024")
    """
    
    def __init__(
        self,
        connection: Any,
        cache: Optional[QueryCache] = None,
        cache_by_default: bool = True
    ):
        """
        Initialize cached executor.
        
        Args:
            connection: DuckDB connection
            cache: Optional cache instance (created if not provided)
            cache_by_default: Whether to cache all queries by default
        """
        self.connection = connection
        self.cache = cache or QueryCache()
        self.cache_by_default = cache_by_default
    
    def query(
        self,
        sql: str,
        use_cache: Optional[bool] = None,
        ttl_hours: Optional[float] = None
    ) -> 'pd.DataFrame':
        """
        Execute query with caching.
        
        Args:
            sql: SQL query string
            use_cache: Override default caching behavior
            ttl_hours: Optional TTL for this query
            
        Returns:
            Query result as DataFrame
        """
        should_cache = use_cache if use_cache is not None else self.cache_by_default
        
        if should_cache:
            # Check cache
            cached = self.cache.get(sql)
            if cached is not None:
                return cached
        
        # Execute query
        result = self.connection.execute(sql).fetchdf()
        
        if should_cache:
            self.cache.set(sql, result, ttl_hours)
        
        return result
    
    def execute_no_cache(self, sql: str) -> 'pd.DataFrame':
        """Execute query without caching."""
        return self.query(sql, use_cache=False)
    
    def invalidate(self, pattern: Optional[str] = None):
        """
        Invalidate cache entries.
        
        Args:
            pattern: Optional SQL pattern to match (None = clear all)
        """
        if pattern is None:
            self.cache.clear()
        else:
            # Find matching keys
            to_delete = []
            for key, entry in self.cache._cache.items():
                if hasattr(entry.value, '_sql') and pattern in entry.value._sql:
                    to_delete.append(key)
            for key in to_delete:
                self.cache.delete(key)
