"""
EquiPay Canada - Advanced Data Store
====================================

Production-grade SQL + Python architecture for memory-efficient
analysis of 19.5M+ row Labour Force Survey datasets.

Architecture:
    Layer 1: LazyQuery - Fluent query builder
    Layer 2: Streaming - Memory-efficient iteration
    Layer 3: Cache - Query result caching with TTL
    Layer 4: Views - Materialized views for dashboards
    Layer 5: Features - Feature store with versioning
    Layer 6: Analytics - SQL-accelerated econometrics

Usage:
    from src.data_store import EquiPayDataStore
    
    store = EquiPayDataStore()
    
    # Fluent API (Layer 1)
    result = (
        store.query()
        .select('PROV', 'GENDER', 'HRLYEARN')
        .where("SURVYEAR = 2024 AND HRLYEARN > 0")
        .aggregate(Agg.weighted_mean('HRLYEARN', 'FINALWT'))
        .group_by('PROV', 'GENDER')
        .to_pandas()
    )
    
    # Streaming (Layer 2)
    for chunk in store.stream(chunk_size=100000):
        process(chunk)
    
    # ML Streaming
    for X, y, w in store.ml_stream(['EDUC', 'NOC_10']):
        model.partial_fit(X, y, sample_weight=w)
    
    # Analytics (Layer 6)
    decomp = store.analytics.decomposition.decompose(
        features=['EDUC', 'NOC_10', 'PROV'],
        year=2024
    )

Author: EquiPay Canada Research Team
Version: 2.0.0
"""

# Core data store
from .core import EquiPayDataStore

# Query building
from .query import LazyQuery, Agg, Func

# Streaming
from .streaming import (
    StreamingIterator,
    StratifiedStreamingIterator,
    ChunkResult,
    MLDataStream
)

# Caching
from .cache import QueryCache, CacheEntry, CacheStats, CachedQueryExecutor

# Memory management
from .memory import MemoryMonitor, MemoryStats

# Feature store
from .features import FeatureStore, FeatureDefinition, FeatureSet

# Materialized views
from .views import MaterializedViewManager, ViewDefinition

# Analytics engine
from .analytics import (
    AnalyticsEngine,
    OaxacaBlinder,
    RIFDecomposition,
    DecompositionResult,
    QuantileGapAnalyzer,
    GlassCeilingAnalyzer,
    QuantileGapResult,
    GlassCeilingResult,
    PoissonBootstrap,
    BootstrapResults,
    DifferenceInDifferences,
    EventStudy,
    SyntheticControl,
    PolicyEvaluator,
    DiDResult,
    EventStudyResult
)

__all__ = [
    # Main entry point
    'EquiPayDataStore',
    
    # Query building
    'LazyQuery',
    'Agg',
    'Func',
    
    # Streaming
    'StreamingIterator',
    'StratifiedStreamingIterator',
    'ChunkResult',
    'MLDataStream',
    
    # Caching
    'QueryCache',
    'CacheEntry',
    'CacheStats',
    'CachedQueryExecutor',
    
    # Memory
    'MemoryMonitor',
    'MemoryStats',
    
    # Features
    'FeatureStore',
    'FeatureDefinition',
    'FeatureSet',
    
    # Views
    'MaterializedViewManager',
    'ViewDefinition',
    
    # Analytics - Engine
    'AnalyticsEngine',
    
    # Analytics - Decomposition
    'OaxacaBlinder',
    'RIFDecomposition',
    'DecompositionResult',
    
    # Analytics - Quantile
    'QuantileGapAnalyzer',
    'GlassCeilingAnalyzer',
    'QuantileGapResult',
    'GlassCeilingResult',
    
    # Analytics - Bootstrap
    'PoissonBootstrap',
    'BootstrapResults',
    
    # Analytics - Causal
    'DifferenceInDifferences',
    'EventStudy',
    'SyntheticControl',
    'PolicyEvaluator',
    'DiDResult',
    'EventStudyResult',
]

__version__ = '2.0.0'


def connect(parquet_path: str = 'data/parquet', **kwargs) -> EquiPayDataStore:
    """
    Quick connect to EquiPay data store.
    
    Args:
        parquet_path: Path to Parquet data
        **kwargs: Additional arguments for EquiPayDataStore
        
    Returns:
        Configured EquiPayDataStore instance
    """
    return EquiPayDataStore(parquet_path=parquet_path, **kwargs)
