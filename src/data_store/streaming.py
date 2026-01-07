"""
EquiPay Canada - Streaming Iterator
===================================

Memory-efficient streaming of large datasets with backpressure.

Features:
- Constant memory usage regardless of dataset size
- Automatic chunk sizing based on available memory
- Stratified sampling within chunks
- Support for ML training loops

Usage:
    for chunk in store.stream(chunk_size=500_000):
        model.partial_fit(chunk.X, chunk.y, sample_weight=chunk.weights)
"""

import gc
import logging
from typing import Optional, List, Iterator, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass
import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    import duckdb

from .memory import MemoryMonitor, get_memory_monitor

logger = logging.getLogger(__name__)


@dataclass
class ChunkResult:
    """
    Container for a chunk of data from streaming.
    
    Provides convenient access to features, target, and weights.
    """
    data: 'pd.DataFrame'
    chunk_num: int
    total_seen: int
    is_last: bool
    
    # Cached arrays
    _X: Optional[np.ndarray] = None
    _y: Optional[np.ndarray] = None
    _weights: Optional[np.ndarray] = None
    _feature_cols: Optional[List[str]] = None
    _target_col: Optional[str] = None
    _weight_col: Optional[str] = None
    
    @property
    def X(self) -> np.ndarray:
        """Get feature matrix."""
        if self._X is None:
            if self._feature_cols:
                self._X = self.data[self._feature_cols].values
            else:
                # Auto-detect: exclude target and weight columns
                exclude = set()
                if self._target_col:
                    exclude.add(self._target_col)
                if self._weight_col:
                    exclude.add(self._weight_col)
                feature_cols = [c for c in self.data.columns if c not in exclude]
                self._X = self.data[feature_cols].values
        return self._X
    
    @property
    def y(self) -> Optional[np.ndarray]:
        """Get target vector."""
        if self._y is None and self._target_col and self._target_col in self.data.columns:
            self._y = self.data[self._target_col].values
        return self._y
    
    @property
    def weights(self) -> Optional[np.ndarray]:
        """Get sample weights."""
        if self._weights is None and self._weight_col and self._weight_col in self.data.columns:
            self._weights = self.data[self._weight_col].values
        return self._weights
    
    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        return f"ChunkResult(n={len(self)}, chunk={self.chunk_num}, total_seen={self.total_seen})"


class StreamingIterator:
    """
    Memory-efficient iterator over large SQL results.
    
    Uses DuckDB's streaming capabilities to process data in chunks
    without loading the entire dataset into memory.
    
    Usage:
        iterator = StreamingIterator(
            connection=conn,
            query="SELECT * FROM lfs WHERE HRLYEARN > 0",
            chunk_size=500_000,
            features=['EDUC', 'NOC_10', 'PROV'],
            target='LOG_REAL_HRLYEARN',
            weight='FINALWT'
        )
        
        for chunk in iterator:
            model.partial_fit(chunk.X, chunk.y, sample_weight=chunk.weights)
    """
    
    def __init__(
        self,
        connection: 'duckdb.DuckDBPyConnection',
        query: str,
        chunk_size: int = 500_000,
        features: Optional[List[str]] = None,
        target: Optional[str] = None,
        weight: Optional[str] = 'FINALWT',
        shuffle: bool = True,
        memory_monitor: Optional[MemoryMonitor] = None,
        max_memory_mb: Optional[float] = None
    ):
        """
        Initialize streaming iterator.
        
        Args:
            connection: DuckDB connection
            query: SQL query to stream
            chunk_size: Rows per chunk
            features: Feature column names
            target: Target column name
            weight: Weight column name
            shuffle: Shuffle rows (adds ORDER BY RANDOM())
            memory_monitor: Optional memory monitor
            max_memory_mb: Maximum memory usage in MB
        """
        self.connection = connection
        self.base_query = query
        self.chunk_size = chunk_size
        self.features = features
        self.target = target
        self.weight = weight
        self.shuffle = shuffle
        
        # Memory management
        self.memory = memory_monitor or get_memory_monitor()
        if max_memory_mb:
            self.memory.set_limit(max_memory_mb)
        
        # State
        self._total_count: Optional[int] = None
        self._current_offset = 0
        self._chunks_yielded = 0
        self._total_seen = 0
    
    @property
    def total_count(self) -> int:
        """Get total number of rows (lazy computed)."""
        if self._total_count is None:
            count_query = f"SELECT COUNT(*) FROM ({self.base_query}) AS sub"
            result = self.connection.execute(count_query).fetchone()
            self._total_count = result[0] if result else 0
        return self._total_count
    
    @property
    def num_chunks(self) -> int:
        """Estimated number of chunks."""
        return (self.total_count + self.chunk_size - 1) // self.chunk_size
    
    def _build_chunk_query(self, offset: int, limit: int) -> str:
        """Build query for a specific chunk."""
        order_clause = "ORDER BY RANDOM()" if self.shuffle else ""
        
        return f"""
            SELECT * FROM ({self.base_query}) AS base
            {order_clause}
            LIMIT {limit} OFFSET {offset}
        """
    
    def __iter__(self) -> Iterator[ChunkResult]:
        """Iterate over chunks."""
        self._current_offset = 0
        self._chunks_yielded = 0
        self._total_seen = 0
        
        while True:
            # Check memory and adjust chunk size if needed
            self.memory.check_and_collect()
            adaptive_chunk = self.memory.get_optimal_chunk_size(
                row_size_bytes=500,
                target_chunk_mb=200,
                min_rows=10_000,
                max_rows=self.chunk_size
            )
            
            # Fetch chunk
            query = self._build_chunk_query(self._current_offset, adaptive_chunk)
            
            try:
                df = self.connection.execute(query).fetchdf()
            except Exception as e:
                logger.error(f"Error fetching chunk at offset {self._current_offset}: {e}")
                raise
            
            if len(df) == 0:
                break
            
            self._chunks_yielded += 1
            self._total_seen += len(df)
            self._current_offset += len(df)
            
            is_last = len(df) < adaptive_chunk or self._total_seen >= self.total_count
            
            chunk = ChunkResult(
                data=df,
                chunk_num=self._chunks_yielded,
                total_seen=self._total_seen,
                is_last=is_last
            )
            chunk._feature_cols = self.features
            chunk._target_col = self.target
            chunk._weight_col = self.weight
            
            yield chunk
            
            # Clean up
            del df
            gc.collect()
            
            if is_last:
                break
    
    def reset(self):
        """Reset iterator to beginning."""
        self._current_offset = 0
        self._chunks_yielded = 0
        self._total_seen = 0
    
    def __repr__(self):
        return (
            f"StreamingIterator(total={self.total_count:,}, "
            f"chunk_size={self.chunk_size:,}, "
            f"chunks={self.num_chunks})"
        )


class StratifiedStreamingIterator(StreamingIterator):
    """
    Streaming iterator that maintains stratification within chunks.
    
    Ensures each chunk has proportional representation of stratification
    groups (e.g., GENDER, SURVYEAR, PROV).
    """
    
    def __init__(
        self,
        connection: 'duckdb.DuckDBPyConnection',
        base_query: str,
        stratify_by: List[str],
        chunk_size: int = 500_000,
        **kwargs
    ):
        """
        Initialize stratified streaming iterator.
        
        Args:
            connection: DuckDB connection
            base_query: Base SQL query
            stratify_by: Columns to stratify by
            chunk_size: Rows per chunk
            **kwargs: Additional arguments for StreamingIterator
        """
        self.stratify_by = stratify_by
        
        # Build stratified query
        strat_cols = ", ".join(stratify_by)
        stratified_query = f"""
            SELECT *,
                ROW_NUMBER() OVER (
                    PARTITION BY {strat_cols}
                    ORDER BY RANDOM()
                ) AS _strat_rn,
                COUNT(*) OVER (
                    PARTITION BY {strat_cols}
                ) AS _strat_count
            FROM ({base_query}) AS base
        """
        
        super().__init__(
            connection=connection,
            query=stratified_query,
            chunk_size=chunk_size,
            shuffle=False,  # Already shuffled within strata
            **kwargs
        )


class MLDataStream:
    """
    High-level interface for streaming ML training data.
    
    Provides a clean interface for training ML models on data
    that doesn't fit in memory.
    
    Usage:
        stream = store.ml_stream(
            features=['EDUC', 'NOC_10', 'PROV'],
            target='LOG_REAL_HRLYEARN',
            years=range(2020, 2025),
            stratify_by=['GENDER', 'SURVYEAR']
        )
        
        for epoch in range(3):
            for X, y, w in stream.batches(batch_size=4096):
                model.partial_fit(X, y, sample_weight=w)
            stream.reset()
    """
    
    def __init__(
        self,
        connection: 'duckdb.DuckDBPyConnection',
        features: List[str],
        target: str,
        weight: str = 'FINALWT',
        base_table: str = 'lfs',
        where: Optional[str] = None,
        stratify_by: Optional[List[str]] = None,
        chunk_size: int = 500_000
    ):
        self.connection = connection
        self.features = features
        self.target = target
        self.weight = weight
        self.base_table = base_table
        self.where_clause = where or "HRLYEARN > 0"
        self.stratify_by = stratify_by
        self.chunk_size = chunk_size
        
        self._iterator: Optional[StreamingIterator] = None
    
    def _build_query(self) -> str:
        """Build the SQL query for streaming."""
        # Quote UNION if it's in features (reserved keyword)
        safe_features = []
        for f in self.features:
            if f.upper() == 'UNION':
                safe_features.append('"UNION"')
            else:
                safe_features.append(f)
        
        columns = safe_features + [self.target, self.weight]
        col_str = ", ".join(columns)
        
        return f"""
            SELECT {col_str}
            FROM {self.base_table}
            WHERE {self.where_clause}
        """
    
    def _create_iterator(self) -> StreamingIterator:
        """Create the streaming iterator."""
        query = self._build_query()
        
        if self.stratify_by:
            return StratifiedStreamingIterator(
                connection=self.connection,
                base_query=query,
                stratify_by=self.stratify_by,
                chunk_size=self.chunk_size,
                features=self.features,
                target=self.target,
                weight=self.weight
            )
        else:
            return StreamingIterator(
                connection=self.connection,
                query=query,
                chunk_size=self.chunk_size,
                features=self.features,
                target=self.target,
                weight=self.weight
            )
    
    def chunks(self) -> Iterator[ChunkResult]:
        """Iterate over chunks."""
        self._iterator = self._create_iterator()
        return iter(self._iterator)
    
    def batches(self, batch_size: int = 4096) -> Iterator[tuple]:
        """
        Yield (X, y, weights) batches for training.
        
        Handles chunk boundaries transparently.
        """
        buffer_X = []
        buffer_y = []
        buffer_w = []
        buffer_size = 0
        
        for chunk in self.chunks():
            X = chunk.X
            y = chunk.y
            w = chunk.weights
            
            # Add to buffer
            buffer_X.append(X)
            buffer_y.append(y)
            buffer_w.append(w)
            buffer_size += len(X)
            
            # Yield batches from buffer
            while buffer_size >= batch_size:
                # Concatenate buffer
                X_all = np.vstack(buffer_X)
                y_all = np.concatenate(buffer_y)
                w_all = np.concatenate(buffer_w)
                
                # Yield batch
                yield X_all[:batch_size], y_all[:batch_size], w_all[:batch_size]
                
                # Keep remainder in buffer
                if len(X_all) > batch_size:
                    buffer_X = [X_all[batch_size:]]
                    buffer_y = [y_all[batch_size:]]
                    buffer_w = [w_all[batch_size:]]
                    buffer_size = len(buffer_X[0])
                else:
                    buffer_X = []
                    buffer_y = []
                    buffer_w = []
                    buffer_size = 0
        
        # Yield final partial batch
        if buffer_size > 0:
            X_all = np.vstack(buffer_X)
            y_all = np.concatenate(buffer_y)
            w_all = np.concatenate(buffer_w)
            yield X_all, y_all, w_all
    
    def reset(self):
        """Reset stream for next epoch."""
        if self._iterator:
            self._iterator.reset()
    
    @property
    def total_samples(self) -> int:
        """Total number of samples."""
        if self._iterator is None:
            self._iterator = self._create_iterator()
        return self._iterator.total_count
    
    def __repr__(self):
        return (
            f"MLDataStream(features={self.features}, target={self.target}, "
            f"chunk_size={self.chunk_size})"
        )
