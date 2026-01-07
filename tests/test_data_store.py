"""
EquiPay Canada - Data Store Integration Tests
=============================================

Tests for the new SQL+Python hybrid data store architecture.

Run with:
    pytest tests/test_data_store.py -v
    
Or for specific test:
    pytest tests/test_data_store.py::test_lazy_query -v
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestMemoryMonitor:
    """Test memory monitoring functionality."""
    
    def test_import(self):
        """Test import works."""
        from src.data_store import MemoryMonitor
        monitor = MemoryMonitor()
        assert monitor is not None
    
    def test_current_stats(self):
        """Test getting current memory stats."""
        from src.data_store import MemoryMonitor
        
        monitor = MemoryMonitor()
        stats = monitor.current()
        
        assert stats.rss_mb > 0
        assert stats.available_mb > 0
        assert 0 <= stats.percent <= 100
    
    def test_should_pause(self):
        """Test backpressure detection."""
        from src.data_store import MemoryMonitor
        
        monitor = MemoryMonitor(warning_threshold_mb=100000)  # Very high threshold
        assert not monitor.should_pause()
        
        monitor2 = MemoryMonitor(warning_threshold_mb=0.001)  # Very low threshold
        assert monitor2.should_pause()
    
    def test_context_manager(self):
        """Test operation tracking."""
        from src.data_store import MemoryMonitor
        
        monitor = MemoryMonitor()
        
        with monitor.track_operation("test_operation"):
            data = list(range(10000))
        
        # Should complete without error


class TestLazyQuery:
    """Test lazy query building."""
    
    def test_import(self):
        """Test import works."""
        from src.data_store import LazyQuery, Agg, Func
        assert LazyQuery is not None
        assert Agg is not None
        assert Func is not None
    
    def test_basic_query_building(self):
        """Test building a basic query."""
        import duckdb
        from src.data_store import LazyQuery
        
        conn = duckdb.connect()
        conn.execute("CREATE TABLE test (a INT, b VARCHAR)")
        
        query = LazyQuery(conn, 'test')
        sql = query.select('a', 'b').where("a > 0").sql()
        
        assert 'SELECT' in sql
        assert 'test' in sql
        assert 'a > 0' in sql
    
    def test_aggregation_helpers(self):
        """Test Agg helper functions."""
        from src.data_store import Agg
        
        assert 'AVG' in Agg.mean('x')
        assert 'COUNT' in Agg.count()
        assert Agg.weighted_mean('x', 'w') is not None
        assert Agg.gender_gap('x', 'gender') is not None
    
    def test_chaining(self):
        """Test query chaining returns LazyQuery."""
        import duckdb
        from src.data_store import LazyQuery
        
        conn = duckdb.connect()
        conn.execute("CREATE TABLE test (a INT)")
        
        query = (
            LazyQuery(conn, 'test')
            .select('a')
            .where("a > 0")
            .order_by('a')
            .limit(10)
        )
        
        assert isinstance(query, LazyQuery)


class TestStreamingIterator:
    """Test streaming functionality."""
    
    def test_import(self):
        """Test import works."""
        from src.data_store import StreamingIterator, ChunkResult
        assert StreamingIterator is not None
        assert ChunkResult is not None
    
    def test_chunk_result(self):
        """Test ChunkResult dataclass."""
        import pandas as pd
        from src.data_store import ChunkResult
        
        df = pd.DataFrame({'a': [1, 2, 3]})
        chunk = ChunkResult(
            data=df,
            chunk_number=0,
            rows=3,
            total_rows=100
        )
        
        assert chunk.rows == 3
        assert chunk.is_first
        assert not chunk.is_last
    
    def test_iterator_with_small_data(self):
        """Test iterating over small dataset."""
        import duckdb
        from src.data_store import StreamingIterator
        
        conn = duckdb.connect()
        conn.execute("CREATE TABLE test AS SELECT * FROM range(100) t(id)")
        
        iterator = StreamingIterator(
            connection=conn,
            query="SELECT * FROM test",
            chunk_size=30
        )
        
        chunks = list(iterator)
        total_rows = sum(c.rows for c in chunks)
        
        assert total_rows == 100
        assert len(chunks) == 4  # 30, 30, 30, 10


class TestQueryCache:
    """Test query caching."""
    
    def test_import(self):
        """Test import works."""
        from src.data_store import QueryCache
        assert QueryCache is not None
    
    def test_cache_hit(self):
        """Test cache stores and retrieves."""
        import duckdb
        from src.data_store import QueryCache
        
        conn = duckdb.connect()
        conn.execute("CREATE TABLE test AS SELECT * FROM range(10) t(id)")
        
        cache = QueryCache(conn)
        
        # First query - should miss
        result1 = cache.execute("SELECT COUNT(*) FROM test")
        
        # Second query - should hit
        result2 = cache.execute("SELECT COUNT(*) FROM test")
        
        assert result1.iloc[0, 0] == result2.iloc[0, 0]


class TestFeatureStore:
    """Test feature store functionality."""
    
    def test_import(self):
        """Test import works."""
        from src.data_store import FeatureStore
        assert FeatureStore is not None
    
    def test_register_sql_feature(self):
        """Test registering SQL features."""
        import duckdb
        from src.data_store import FeatureStore
        
        conn = duckdb.connect()
        store = FeatureStore(conn)
        
        # Register custom feature
        store.register_sql(
            name='TEST_FEATURE',
            sql_expr='x * 2',
            description='Test feature',
            required_columns=['x']
        )
        
        assert 'TEST_FEATURE' in store.list()
    
    def test_builtin_features(self):
        """Test built-in features are registered."""
        from src.data_store import FeatureStore
        
        store = FeatureStore()
        features = store.list()
        
        assert 'IS_FEMALE' in features
        assert 'IS_FULLTIME' in features
        assert 'EXPERIENCE_PROXY' in features


class TestMaterializedViews:
    """Test materialized view functionality."""
    
    def test_import(self):
        """Test import works."""
        from src.data_store import MaterializedViewManager
        assert MaterializedViewManager is not None
    
    def test_create_virtual_view(self):
        """Test creating non-materialized view."""
        import duckdb
        import tempfile
        from src.data_store import MaterializedViewManager
        
        conn = duckdb.connect()
        conn.execute("CREATE TABLE test AS SELECT * FROM range(10) t(id)")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = MaterializedViewManager(conn, storage_dir=tmpdir)
            
            manager.create(
                name='test_view',
                sql='SELECT id, id * 2 as doubled FROM test',
                materialize=False
            )
            
            assert 'test_view' in manager.list()


class TestAnalytics:
    """Test analytics engine."""
    
    def test_import(self):
        """Test import works."""
        from src.data_store import (
            AnalyticsEngine,
            OaxacaBlinder,
            PoissonBootstrap,
            QuantileGapAnalyzer
        )
        assert AnalyticsEngine is not None
        assert OaxacaBlinder is not None
        assert PoissonBootstrap is not None
        assert QuantileGapAnalyzer is not None
    
    def test_decomposition_result(self):
        """Test DecompositionResult dataclass."""
        from src.data_store.analytics.decomposition import DecompositionResult
        
        result = DecompositionResult(
            total_gap=0.15,
            gap_pct=16.2,
            explained=0.10,
            unexplained=0.05
        )
        
        assert result.explained_pct == pytest.approx(66.67, rel=0.01)
        assert result.unexplained_pct == pytest.approx(33.33, rel=0.01)
    
    def test_bootstrap_result(self):
        """Test BootstrapResults dataclass."""
        import numpy as np
        from src.data_store import BootstrapResults
        
        estimates = np.array([0.10, 0.12, 0.11, 0.09, 0.13])
        
        result = BootstrapResults(
            estimate=0.11,
            n_bootstraps=5,
            bootstrap_estimates=estimates
        )
        
        assert result.std_error > 0
        assert result.ci_lower < result.ci_upper


class TestEquiPayDataStore:
    """Test main data store class."""
    
    def test_import(self):
        """Test import works."""
        from src.data_store import EquiPayDataStore
        assert EquiPayDataStore is not None
    
    def test_connect_function(self):
        """Test connect convenience function."""
        from src.data_store import connect
        assert connect is not None
    
    def test_initialization_without_data(self):
        """Test initialization when no data exists."""
        import tempfile
        from src.data_store import EquiPayDataStore
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Should not crash even without data
            store = EquiPayDataStore(
                parquet_path=f"{tmpdir}/nonexistent",
                cache_dir=f"{tmpdir}/cache",
                views_dir=f"{tmpdir}/views"
            )
            
            assert store.connection is not None
            store.close()
    
    def test_health_check(self):
        """Test health check returns expected keys."""
        import tempfile
        from src.data_store import EquiPayDataStore
        
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EquiPayDataStore(
                parquet_path=f"{tmpdir}/nonexistent",
                cache_dir=f"{tmpdir}/cache",
                views_dir=f"{tmpdir}/views"
            )
            
            health = store.health_check()
            
            assert 'connection' in health
            assert 'parquet_exists' in health
            assert 'memory' in health
            
            store.close()


class TestIntegration:
    """Integration tests with synthetic data."""
    
    @pytest.fixture
    def synthetic_store(self, tmp_path):
        """Create a store with synthetic data for testing."""
        import duckdb
        import pandas as pd
        import numpy as np
        from src.data_store import EquiPayDataStore
        
        # Create synthetic parquet data
        np.random.seed(42)
        n = 1000
        
        data = pd.DataFrame({
            'SURVYEAR': np.random.choice([2020, 2021, 2022, 2023], n),
            'GENDER': np.random.choice([1, 2], n),  # 1=M, 2=F
            'PROV': np.random.choice([35, 24, 48, 59], n),
            'EDUC': np.random.choice([1, 2, 3, 4, 5], n),
            'NOC_10': np.random.choice([0, 1, 2, 3, 4], n),
            'HRLYEARN': np.random.lognormal(3, 0.5, n),
            'FINALWT': np.random.uniform(100, 5000, n),
            'FTPTMAIN': np.random.choice([1, 2], n),
            'AGE_12': np.random.choice(range(1, 13), n),
            'UNION': np.random.choice([1, 2, 3], n)
        })
        
        # Create computed columns
        data['IS_FEMALE'] = (data['GENDER'] == 2).astype(int)
        data['LOG_REAL_HRLYEARN'] = np.log(data['HRLYEARN'])
        
        # Save as parquet
        parquet_dir = tmp_path / 'parquet' / 'year=2023'
        parquet_dir.mkdir(parents=True)
        data.to_parquet(parquet_dir / 'data.parquet', index=False)
        
        # Create store
        store = EquiPayDataStore(
            parquet_path=str(tmp_path / 'parquet'),
            cache_dir=str(tmp_path / 'cache'),
            views_dir=str(tmp_path / 'views'),
            enable_cache=True
        )
        
        yield store
        
        store.close()
    
    def test_query_execution(self, synthetic_store):
        """Test basic query execution."""
        df = synthetic_store.query().limit(10).to_pandas()
        assert len(df) == 10
    
    def test_count(self, synthetic_store):
        """Test row counting."""
        count = synthetic_store.count()
        assert count == 1000
    
    def test_gender_gap(self, synthetic_store):
        """Test gender gap calculation."""
        gap = synthetic_store.gender_gap()
        
        assert 'gap' in gap.columns
        assert 'male_mean' in gap.columns
        assert 'female_mean' in gap.columns
    
    def test_gender_gap_by_province(self, synthetic_store):
        """Test gender gap by province."""
        gap = synthetic_store.gender_gap(by=['PROV'])
        
        assert 'PROV' in gap.columns
        assert len(gap) > 1  # Multiple provinces
    
    def test_streaming(self, synthetic_store):
        """Test streaming iteration."""
        total_rows = 0
        for chunk in synthetic_store.stream(chunk_size=200):
            total_rows += chunk.rows
        
        assert total_rows == 1000
    
    def test_ml_stream(self, synthetic_store):
        """Test ML data streaming."""
        batches = 0
        total_samples = 0
        
        for X, y, w in synthetic_store.ml_stream(
            features=['EDUC', 'NOC_10', 'PROV'],
            batch_size=100
        ):
            batches += 1
            total_samples += len(X)
            
            assert X.shape[1] == 3  # 3 features
            assert len(y) == len(X)
            assert len(w) == len(X)
            
            if batches >= 5:  # Don't process all
                break
        
        assert batches >= 1
    
    def test_sample(self, synthetic_store):
        """Test random sampling."""
        sample = synthetic_store.sample(n=50)
        assert len(sample) <= 50  # May be slightly less due to sampling
    
    def test_years(self, synthetic_store):
        """Test getting available years."""
        years = synthetic_store.years()
        assert 2023 in years or len(years) > 0


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_query_result(self):
        """Test handling empty query results."""
        import duckdb
        from src.data_store import LazyQuery
        
        conn = duckdb.connect()
        conn.execute("CREATE TABLE test (a INT)")
        
        query = LazyQuery(conn, 'test')
        result = query.to_pandas()
        
        assert len(result) == 0
    
    def test_invalid_column_handling(self):
        """Test handling of invalid columns."""
        import duckdb
        from src.data_store import LazyQuery
        
        conn = duckdb.connect()
        conn.execute("CREATE TABLE test (a INT)")
        
        query = LazyQuery(conn, 'test')
        
        with pytest.raises(Exception):
            query.select('nonexistent_column').to_pandas()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
