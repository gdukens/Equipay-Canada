"""
EquiPay Canada - Core Data Store
================================

Main EquiPayDataStore class integrating all layers:
- LazyQuery (fluent SQL builder)
- StreamingIterator (chunked processing)
- QueryCache (result caching)
- FeatureStore (ML features)
- MaterializedViews (dashboard acceleration)
- AnalyticsEngine (econometrics)

This is the primary interface for all data access in the project.

Usage:
    from src.data_store import EquiPayDataStore
    
    store = EquiPayDataStore('data/parquet')
    
    # Fluent queries
    gap = store.query().wage_gap(by=['PROV', 'SURVYEAR']).to_pandas()
    
    # Streaming for large operations
    for chunk in store.stream(chunk_size=100000):
        process(chunk)
    
    # Analytics
    decomp = store.analytics.decomposition.decompose(features=['EDUC', 'NOC_10'])
    
    # Features for ML
    X, y, w = store.features.get(['EDUC', 'EXPERIENCE_PROXY'], target='LOG_REAL_HRLYEARN')
"""

import logging
import duckdb
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Iterator, TYPE_CHECKING
from contextlib import contextmanager

from .query import LazyQuery, Agg, Func
from .streaming import StreamingIterator, StratifiedStreamingIterator, MLDataStream
from .cache import QueryCache, CachedQueryExecutor
from .memory import MemoryMonitor
from .features import FeatureStore
from .views import MaterializedViewManager
from .analytics import AnalyticsEngine

import pandas as pd
from ..macro_data import get_macro_dataframe, get_deflator, BASE_YEAR
from ..constants import PROVINCE_ABBREV, AGE_6_MIDPOINTS, AGE_12_MIDPOINTS

if TYPE_CHECKING:
    import pandas as pd
    import numpy as np

logger = logging.getLogger(__name__)


class EquiPayDataStore:
    """
    Production-grade data store for EquiPay Canada.
    
    Provides a unified interface for:
    - Efficient SQL-first data access
    - Memory-safe streaming for large operations
    - Query caching for repeated access
    - Feature engineering for ML
    - Materialized views for dashboards
    - Advanced analytics (decomposition, bootstrap, causal)
    
    Architecture:
        Layer 1: LazyQuery - Fluent SQL builder with lazy evaluation
        Layer 2: Streaming - Chunked iteration with backpressure
        Layer 3: Cache - Query result caching with TTL
        Layer 4: Views - Materialized aggregations
        Layer 5: Features - ML feature store
        Layer 6: Analytics - Econometric methods
    
    Example:
        store = EquiPayDataStore('data/parquet')
        
        # Simple query
        df = store.query().where("SURVYEAR = 2023").to_pandas()
        
        # Aggregation
        gaps = store.query().wage_gap(by=['PROV']).to_pandas()
        
        # Streaming
        for chunk in store.stream():
            model.partial_fit(chunk)
        
        # Analytics
        result = store.analytics.decomposition.decompose(
            features=['EDUC', 'NOC_10', 'PROV']
        )
    """
    
    # Default table name
    TABLE_NAME = 'lfs'
    
    def __init__(
        self,
        parquet_path: str = 'data/parquet',
        raw_csv_path: str = None,
        cache_dir: str = 'data/cache',
        views_dir: str = 'data/views',
        memory_limit_mb: int = 1000,
        enable_cache: bool = True,
        auto_create_views: bool = False,
        use_sql_transforms: bool = True
    ):
        """
        Initialize the data store.
        
        Args:
            parquet_path: Path to Parquet data directory
            cache_dir: Path for query cache storage
            views_dir: Path for materialized view storage
            memory_limit_mb: Memory limit for operations
            enable_cache: Whether to enable query caching
            auto_create_views: Whether to create dashboard views on init
            use_sql_transforms: If True, create a SQL-derived enriched view `lfs_enriched` and
                                register macro table for SQL transformations
        """
        self.parquet_path = Path(parquet_path)
        # Backwards compatibility: accept raw_csv_path but it's optional for core store
        self.raw_csv_path = Path(raw_csv_path) if raw_csv_path is not None else None
        self.cache_dir = Path(cache_dir)
        self.views_dir = Path(views_dir)
        
        # Ensure directories exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.views_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize DuckDB connection
        self._connection = duckdb.connect()
        
        # Configure DuckDB for efficiency
        self._configure_duckdb(memory_limit_mb)
        
        # Register Parquet data
        self._register_parquet()
        
        # Create computed columns
        self._create_computed_columns()

        # Register macro & optionally create SQL-derived materialized view
        self._use_sql_transforms = use_sql_transforms
        self._register_macro()
        if self._use_sql_transforms:
            try:
                created = self.create_materialized_derived_view(refresh=True)
                if created:
                    # Use enriched view as primary table for queries
                    self.TABLE_NAME = 'lfs_enriched'
                    logger.info("SQL transforms enabled: using 'lfs_enriched' as primary table.")
                else:
                    logger.warning("lfs_enriched view creation failed; defaulting to base 'lfs' table.")
            except Exception as e:
                logger.warning(f"Failed to create materialized derived view: {e}")
        
        # Initialize components
        self._memory = MemoryMonitor(
            limit_mb=memory_limit_mb,
            warning_threshold=0.8
        )
        
        if enable_cache:
            self._cache = QueryCache(
                max_size_mb=500,
                ttl_hours=24,
                persist_path=self.cache_dir / 'query_cache.pkl'
            )
        else:
            self._cache = None
        
        self._features = FeatureStore(self._connection)
        self._views = MaterializedViewManager(self._connection, str(self.views_dir))
        self._analytics = AnalyticsEngine(self._connection)
        
        # Create dashboard views if requested
        if auto_create_views:
            self._views.create_dashboard_views(refresh=True)
        
        try:
            cnt = self.count()
            logger.info(f"EquiPayDataStore initialized with {cnt:,} records")
        except Exception:
            logger.warning("EquiPayDataStore initialized but base table not available (no parquet registered).")
    
    def _configure_duckdb(self, memory_limit_mb: int):
        """Configure DuckDB for optimal performance."""
        # Set memory limit
        self._connection.execute(f"SET memory_limit = '{memory_limit_mb}MB'")
        
        # Enable parallel execution
        self._connection.execute("SET threads TO 4")
        
        # Optimize for analytics
        self._connection.execute("SET enable_progress_bar = false")
        
        logger.debug(f"DuckDB configured with {memory_limit_mb}MB memory limit")
    
    def _register_parquet(self):
        """Register Parquet files as a table."""
        parquet_pattern = str(self.parquet_path / '**' / '*.parquet')
        
        # Check if data path exists and contains parquet files
        if not self.parquet_path.exists():
            logger.warning(f"Parquet path does not exist: {self.parquet_path}")
            return

        parquet_files = list(self.parquet_path.rglob('*.parquet'))
        if not parquet_files:
            logger.warning(f"No parquet files found in: {self.parquet_path}; skipping registration")
            return
        
        # Create view for the LFS data
        self._connection.execute(f"""
            CREATE OR REPLACE VIEW {self.TABLE_NAME} AS
            SELECT * FROM read_parquet('{parquet_pattern}', hive_partitioning=true)
        """)
        
        logger.debug(f"Registered Parquet data from {parquet_pattern}")
    
    def _create_computed_columns(self):
        """Create view with computed columns."""
        # Check if base view exists
        try:
            self._connection.execute(f"SELECT 1 FROM {self.TABLE_NAME} LIMIT 1")
        except Exception:
            logger.warning("Base table not available, skipping computed columns")
            return
        
        # Get existing columns
        columns = self._connection.execute(
            f"SELECT column_name FROM information_schema.columns WHERE table_name = '{self.TABLE_NAME}'"
        ).fetchdf()['column_name'].tolist()
        
        # Build computed column expressions
        computed = []
        
        # IS_FEMALE if not exists
        if 'IS_FEMALE' not in columns:
            computed.append("CASE WHEN GENDER = 2 THEN 1 ELSE 0 END AS IS_FEMALE")
        
        # LOG_REAL_HRLYEARN if not exists
        if 'LOG_REAL_HRLYEARN' not in columns and 'HRLYEARN' in columns:
            # Apply CPI adjustment (simplified - use actual CPI in production)
            computed.append("LN(GREATEST(HRLYEARN, 0.01)) AS LOG_REAL_HRLYEARN")
        
        # If we need to add computed columns, recreate the view
        if computed:
            computed_expr = ', '.join(computed)
            parquet_pattern = str(self.parquet_path / '**' / '*.parquet')
            
            self._connection.execute(f"""
                CREATE OR REPLACE VIEW {self.TABLE_NAME} AS
                SELECT *, {computed_expr}
                FROM read_parquet('{parquet_pattern}', hive_partitioning=true)
            """)
            
            logger.debug(f"Added computed columns: {computed}")

    def _register_macro(self):
        """Register macro data as a DuckDB table called `macro` including a deflator."""
        try:
            df_macro = get_macro_dataframe()
            df_macro['deflator'] = df_macro['year'].map(lambda y: get_deflator(y))
            # Register temporary dataframe and persist as table
            self._connection.register('tmp_macro_df', df_macro)
            self._connection.execute("CREATE OR REPLACE TABLE macro AS SELECT * FROM tmp_macro_df")
            logger.debug("Registered 'macro' table in DuckDB with deflators.")
        except Exception as e:
            logger.exception(f"Failed to register macro table: {e}")

    def create_materialized_derived_view(self, refresh: bool = True):
        """Create or refresh the `lfs_enriched` view with commonly used derived columns.

        The view includes: IS_FEMALE, IS_FULLTIME, IS_PERMANENT, IS_UNION, HAS_DEGREE,
        AGE_APPROX, EDU_COMPLETE_AGE, EXPERIENCE_PROXY, EXPERIENCE_SQ,
        LOG_HRLYEARN, REAL_HRLYEARN, LOG_REAL_HRLYEARN, and macro columns.
        """
        # Determine available base columns to avoid referencing missing columns
        try:
            cols_df = self._connection.execute(
                f"SELECT column_name FROM information_schema.columns WHERE table_name = '{self.TABLE_NAME}'"
            ).fetchdf()
            cols = cols_df['column_name'].tolist()
        except Exception:
            cols = []

        parquet_pattern = str(self.parquet_path / '**' / '*.parquet')

        # Build province abbrev CASE
        prov_when = " ".join([f"WHEN PROV = {k} THEN '{v}'" for k, v in PROVINCE_ABBREV.items()])
        prov_case = f"CASE {prov_when} ELSE NULL END AS PROV_ABBREV"

        # AGE approximation expressions (generated from constants)
        age6_when = " ".join([f"WHEN AGE_6 = {k} THEN {v}" for k, v in AGE_6_MIDPOINTS.items()])
        age12_when = " ".join([f"WHEN AGE_12 = {k} THEN {v}" for k, v in AGE_12_MIDPOINTS.items()])

        # Build AGE_APPROX only using columns that exist to avoid binding errors
        age_parts = []
        if 'AGE_6' in cols:
            age_parts.append(f"CASE {age6_when} ELSE NULL END")
        if 'AGE_12' in cols:
            age_parts.append(f"CASE {age12_when} ELSE NULL END")

        if age_parts:
            if 'AGE_6' in cols and 'AGE_12' in cols:
                age_approx_expr = (
                    f"CASE WHEN AGE_6 IS NOT NULL THEN CASE {age6_when} ELSE NULL END "
                    f"WHEN AGE_12 IS NOT NULL THEN CASE {age12_when} ELSE NULL END ELSE NULL END"
                )
            elif 'AGE_6' in cols:
                age_approx_expr = f"CASE WHEN AGE_6 IS NOT NULL THEN CASE {age6_when} ELSE NULL END ELSE NULL END"
            else:
                age_approx_expr = f"CASE WHEN AGE_12 IS NOT NULL THEN CASE {age12_when} ELSE NULL END ELSE NULL END"
        else:
            age_approx_expr = "NULL"

        # Education completion age mapping
        educ_map = {0: 16, 1: 18, 2: 19, 3: 20, 4: 22, 5: 25}
        educ_when = " ".join([f"WHEN EDUC = {k} THEN {v}" for k, v in educ_map.items()])
        edu_complete_expr = f"CASE {educ_when} ELSE 18 END"

        # Build inner SELECT: compute AGE_APPROX, EDU_COMPLETE_AGE and basic flags
        flag_exprs = []
        flag_exprs.append("CASE WHEN b.GENDER = 2 THEN 1 ELSE 0 END AS IS_FEMALE") if 'GENDER' in cols or True else None
        if 'FTPTMAIN' in cols:
            flag_exprs.append("CASE WHEN b.FTPTMAIN = 1 THEN 1 ELSE 0 END AS IS_FULLTIME")
        if 'PERMTEMP' in cols:
            flag_exprs.append("CASE WHEN b.PERMTEMP = 1 THEN 1 ELSE 0 END AS IS_PERMANENT")
        if 'UNION' in cols:
            flag_exprs.append("CASE WHEN b.UNION = 1 THEN 1 ELSE 0 END AS IS_UNION")
        flag_exprs.append("CASE WHEN b.EDUC >= 4 THEN 1 ELSE 0 END AS HAS_DEGREE")

        inner_flags = ",\n    ".join(flag_exprs)

        inner_sql = f"""
            SELECT b.*, {age_approx_expr} AS AGE_APPROX,
                   {edu_complete_expr} AS EDU_COMPLETE_AGE,
                   {inner_flags},
                   m.cpi, m.gdp_growth, m.unemployment, m.deflator
            FROM read_parquet('{parquet_pattern}', hive_partitioning=true) b
            LEFT JOIN macro m ON b.SURVYEAR = m.year
        """

        # Outer SQL: compute experience, logs, real wages, and province abbrev
        sql = f"""
            CREATE OR REPLACE VIEW lfs_enriched AS
            SELECT inner_tbl.*,
                   GREATEST(inner_tbl.AGE_APPROX - inner_tbl.EDU_COMPLETE_AGE, 0) AS EXPERIENCE_PROXY,
                   POWER(GREATEST(inner_tbl.AGE_APPROX - inner_tbl.EDU_COMPLETE_AGE, 0), 2) AS EXPERIENCE_SQ,
                   LN(GREATEST(inner_tbl.HRLYEARN, 0.01)) AS LOG_HRLYEARN,
                   inner_tbl.HRLYEARN * inner_tbl.deflator AS REAL_HRLYEARN,
                   LN(GREATEST(inner_tbl.HRLYEARN * inner_tbl.deflator, 0.01)) AS LOG_REAL_HRLYEARN,
                   {prov_case}
            FROM (
                {inner_sql}
            ) inner_tbl
        """

        try:
            self._connection.execute(sql)
            logger.info("Created/Refreshed view: lfs_enriched")
            return True
        except Exception as e:
            logger.exception(f"Failed creating lfs_enriched view: {e}")
            return False
    # =========================================================================
    # Core Properties
    # =========================================================================
    
    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        """Get the DuckDB connection."""
        return self._connection
    
    @property
    def memory(self) -> MemoryMonitor:
        """Get memory monitor."""
        return self._memory
    
    @property
    def cache(self) -> Optional[QueryCache]:
        """Get query cache."""
        return self._cache
    
    @property
    def features(self) -> FeatureStore:
        """Get feature store."""
        return self._features
    
    @property
    def views(self) -> MaterializedViewManager:
        """Get materialized view manager."""
        return self._views
    
    @property
    def analytics(self) -> AnalyticsEngine:
        """Get analytics engine."""
        return self._analytics
    
    # =========================================================================
    # Query Interface
    # =========================================================================
    
    def query(self, table: str = None) -> LazyQuery:
        """
        Start a fluent query.
        
        Returns a LazyQuery that builds SQL and executes lazily.
        
        Usage:
            # Simple filter
            df = store.query().where("SURVYEAR = 2023").to_pandas()
            
            # Aggregation
            df = store.query().aggregate(
                Agg.weighted_mean('HRLYEARN', 'FINALWT'),
                group_by=['PROV', 'SURVYEAR']
            ).to_pandas()
            
            # Wage gap
            df = store.query().wage_gap(by=['PROV']).to_pandas()
        """
        return LazyQuery(self._connection, table or self.TABLE_NAME)
    
    def sql(self, query: str) -> 'pd.DataFrame':
        """
        Execute raw SQL and return DataFrame.
        
        For complex queries that don't fit the fluent API.
        """
        return self._connection.execute(query).fetchdf()
    
    def execute(self, query: str):
        """Execute SQL without returning results."""
        self._connection.execute(query)
    
    # =========================================================================
    # Streaming Interface
    # =========================================================================
    
    def stream(
        self,
        chunk_size: int = 100000,
        columns: List[str] = None,
        where: str = None,
        order_by: str = None
    ) -> StreamingIterator:
        """
        Create a streaming iterator for memory-efficient processing.
        
        Args:
            chunk_size: Rows per chunk
            columns: Columns to select (default: all)
            where: Filter condition
            order_by: Sort order
            
        Returns:
            StreamingIterator that yields DataFrames
            
        Usage:
            for chunk in store.stream(chunk_size=50000):
                process(chunk.data)
        """
        # Build query
        col_str = ', '.join(columns) if columns else '*'
        sql = f"SELECT {col_str} FROM {self.TABLE_NAME}"
        
        if where:
            sql = f"{sql} WHERE {where}"
        if order_by:
            sql = f"{sql} ORDER BY {order_by}"
        
        return StreamingIterator(
            connection=self._connection,
            query=sql,
            chunk_size=chunk_size,
            memory_monitor=self._memory
        )
    
    def stream_stratified(
        self,
        stratify_col: str,
        chunk_size: int = 100000,
        columns: List[str] = None,
        where: str = None
    ) -> StratifiedStreamingIterator:
        """
        Create stratified streaming iterator.
        
        Ensures each chunk has similar distribution of stratify_col.
        """
        col_str = ', '.join(columns) if columns else '*'
        sql = f"SELECT {col_str} FROM {self.TABLE_NAME}"
        
        if where:
            sql = f"{sql} WHERE {where}"
        
        return StratifiedStreamingIterator(
            connection=self._connection,
            query=sql,
            stratify_col=stratify_col,
            chunk_size=chunk_size,
            memory_monitor=self._memory
        )
    
    def ml_stream(
        self,
        features: List[str],
        target: str = 'LOG_REAL_HRLYEARN',
        weight: str = 'FINALWT',
        batch_size: int = 10000,
        sample_frac: float = 1.0,
        shuffle: bool = True
    ) -> MLDataStream:
        """
        Create ML training data stream.
        
        Returns (X, y, weights) batches for training.
        
        Usage:
            for X, y, w in store.ml_stream(['EDUC', 'NOC_10'], batch_size=5000):
                model.partial_fit(X, y, sample_weight=w)
        """
        return MLDataStream(
            connection=self._connection,
            features=features,
            target=target,
            weight=weight,
            table=self.TABLE_NAME,
            batch_size=batch_size,
            sample_frac=sample_frac,
            shuffle=shuffle
        )
    
    # =========================================================================
    # Convenience Methods
    # =========================================================================
    
    def count(self, where: str = None) -> int:
        """Get row count. Returns 0 if the base table is not available (safe for FAST-mode smoke tests)."""
        sql = f"SELECT COUNT(*) FROM {self.TABLE_NAME}"
        if where:
            sql = f"{sql} WHERE {where}"
        try:
            return self._connection.execute(sql).fetchone()[0]
        except Exception:
            # Table may not exist in lightweight environments; return 0 to allow smoke tests to proceed
            return 0
    
    def years(self) -> List[int]:
        """Get available years. Returns an empty list if the base table is not available."""
        try:
            result = self._connection.execute(
                f"SELECT DISTINCT SURVYEAR FROM {self.TABLE_NAME} ORDER BY SURVYEAR"
            ).fetchdf()
            return result['SURVYEAR'].tolist()
        except Exception:
            return []
    
    def columns(self) -> List[str]:
        """Get available columns."""
        result = self._connection.execute(
            f"SELECT * FROM {self.TABLE_NAME} LIMIT 0"
        ).fetchdf()
        return list(result.columns)
    
    def describe(self, column: str = 'HRLYEARN', where: str = "HRLYEARN > 0") -> Dict[str, float]:
        """Get descriptive statistics for a column."""
        sql = f"""
            SELECT 
                COUNT(*) as count,
                AVG({column}) as mean,
                STDDEV({column}) as std,
                MIN({column}) as min,
                QUANTILE_CONT({column}, 0.25) as q25,
                QUANTILE_CONT({column}, 0.50) as median,
                QUANTILE_CONT({column}, 0.75) as q75,
                MAX({column}) as max
            FROM {self.TABLE_NAME}
            WHERE {where}
        """
        
        result = self._connection.execute(sql).fetchdf().iloc[0]
        return result.to_dict()
    
    def gender_gap(
        self,
        by: List[str] = None,
        year: int = None,
        weighted: bool = True
    ) -> 'pd.DataFrame':
        """
        Compute gender wage gap.
        
        This is a convenience wrapper around wage_gap() for backward compatibility.
        For new code, prefer using wage_gap(group_column='IS_FEMALE', ...).
        
        Args:
            by: Grouping columns (PROV, NOC_10, EDUC, etc.)
            year: Year filter
            weighted: Use survey weights
            
        Returns:
            DataFrame with gap statistics
        """
        return self.wage_gap(
            group_column='IS_FEMALE',
            reference_value=0,   # Male
            comparison_value=1,  # Female
            by=by,
            year=year,
            weighted=weighted,
            reference_label='Male',
            comparison_label='Female'
        )
    
    def wage_gap(
        self,
        group_column: str = 'IS_FEMALE',
        reference_value: Any = 0,
        comparison_value: Any = 1,
        by: List[str] = None,
        year: int = None,
        weighted: bool = True,
        reference_label: str = None,
        comparison_label: str = None
    ) -> 'pd.DataFrame':
        """
        Compute wage gap for any protected attribute.
        
        This is a generalized gap calculation that works for gender, immigration
        status, age, union status, or any binary/categorical attribute.
        
        Args:
            group_column: Column identifying groups (e.g., 'IS_FEMALE', 'IMMIG', 'AGE_6')
            reference_value: Value for reference group (typically higher-paid)
            comparison_value: Value for comparison group (typically lower-paid)
            by: Grouping columns (PROV, NOC_10, EDUC, etc.)
            year: Year filter
            weighted: Use survey weights
            reference_label: Label for reference group
            comparison_label: Label for comparison group
            
        Returns:
            DataFrame with gap statistics
            
        Examples:
            # Gender gap (traditional)
            store.wage_gap(group_column='IS_FEMALE', reference_value=0, comparison_value=1)
            
            # Immigration gap
            store.wage_gap(group_column='IMMIG', reference_value=1, comparison_value=2,
                          reference_label='Canadian-born', comparison_label='Immigrant')
        """
        weight = "FINALWT" if weighted else "1"
        
        # Build group by clause
        group_cols = by or []
        if year:
            group_cols_str = ', '.join(group_cols) if group_cols else ''
            where = f"HRLYEARN > 0 AND SURVYEAR = {year}"
        else:
            group_cols_str = ', '.join(group_cols) if group_cols else ''
            where = "HRLYEARN > 0"
        
        # Reference and comparison labels
        ref_label = reference_label or str(reference_value)
        comp_label = comparison_label or str(comparison_value)
        
        if group_cols:
            sql = f"""
                SELECT 
                    {group_cols_str},
                    SUM(CASE WHEN {group_column} = {reference_value} THEN {weight} END) as reference_weight,
                    SUM(CASE WHEN {group_column} = {comparison_value} THEN {weight} END) as comparison_weight,
                    SUM(CASE WHEN {group_column} = {reference_value} THEN LOG_REAL_HRLYEARN * {weight} END) / 
                        NULLIF(SUM(CASE WHEN {group_column} = {reference_value} THEN {weight} END), 0) as reference_mean,
                    SUM(CASE WHEN {group_column} = {comparison_value} THEN LOG_REAL_HRLYEARN * {weight} END) / 
                        NULLIF(SUM(CASE WHEN {group_column} = {comparison_value} THEN {weight} END), 0) as comparison_mean,
                    (SUM(CASE WHEN {group_column} = {reference_value} THEN LOG_REAL_HRLYEARN * {weight} END) / 
                        NULLIF(SUM(CASE WHEN {group_column} = {reference_value} THEN {weight} END), 0)) -
                    (SUM(CASE WHEN {group_column} = {comparison_value} THEN LOG_REAL_HRLYEARN * {weight} END) / 
                        NULLIF(SUM(CASE WHEN {group_column} = {comparison_value} THEN {weight} END), 0)) as gap,
                    COUNT(*) as n
                FROM {self.TABLE_NAME}
                WHERE {where}
                GROUP BY {group_cols_str}
                ORDER BY {group_cols_str}
            """
        else:
            sql = f"""
                SELECT 
                    SUM(CASE WHEN {group_column} = {reference_value} THEN {weight} END) as reference_weight,
                    SUM(CASE WHEN {group_column} = {comparison_value} THEN {weight} END) as comparison_weight,
                    SUM(CASE WHEN {group_column} = {reference_value} THEN LOG_REAL_HRLYEARN * {weight} END) / 
                        NULLIF(SUM(CASE WHEN {group_column} = {reference_value} THEN {weight} END), 0) as reference_mean,
                    SUM(CASE WHEN {group_column} = {comparison_value} THEN LOG_REAL_HRLYEARN * {weight} END) / 
                        NULLIF(SUM(CASE WHEN {group_column} = {comparison_value} THEN {weight} END), 0) as comparison_mean,
                    (SUM(CASE WHEN {group_column} = {reference_value} THEN LOG_REAL_HRLYEARN * {weight} END) / 
                        NULLIF(SUM(CASE WHEN {group_column} = {reference_value} THEN {weight} END), 0)) -
                    (SUM(CASE WHEN {group_column} = {comparison_value} THEN LOG_REAL_HRLYEARN * {weight} END) / 
                        NULLIF(SUM(CASE WHEN {group_column} = {comparison_value} THEN {weight} END), 0)) as gap,
                    COUNT(*) as n
                FROM {self.TABLE_NAME}
                WHERE {where}
            """
        
        result = self._connection.execute(sql).fetchdf()
        
        # Add percentage gap and metadata
        import numpy as np
        result['gap_pct'] = (np.exp(result['gap']) - 1) * 100
        result['group_column'] = group_column
        result['reference_label'] = ref_label
        result['comparison_label'] = comp_label
        
        # Add backward-compatible column names for gender gap calls
        if group_column == 'IS_FEMALE' and reference_value == 0:
            result['male_weight'] = result['reference_weight']
            result['female_weight'] = result['comparison_weight']
            result['male_mean'] = result['reference_mean']
            result['female_mean'] = result['comparison_mean']
        
        return result
    
    def immigration_gap(
        self,
        by: List[str] = None,
        year: int = None,
        weighted: bool = True
    ) -> 'pd.DataFrame':
        """
        Compute immigrant vs. Canadian-born wage gap.
        
        Analyzes pay disparities based on immigration status (IMMIG variable).
        
        Args:
            by: Grouping columns (e.g., ['PROV', 'EDUC'])
            year: Year filter
            weighted: Use survey weights
            
        Returns:
            DataFrame with immigration wage gap statistics
        """
        return self.wage_gap(
            group_column='IMMIG',
            reference_value=1,   # Canadian-born
            comparison_value=2,  # Immigrant
            by=by,
            year=year,
            weighted=weighted,
            reference_label='Canadian-born',
            comparison_label='Immigrant'
        )
    
    def union_gap(
        self,
        by: List[str] = None,
        year: int = None,
        weighted: bool = True
    ) -> 'pd.DataFrame':
        """
        Compute union vs. non-union wage gap (union premium).
        
        Args:
            by: Grouping columns
            year: Year filter
            weighted: Use survey weights
            
        Returns:
            DataFrame with union wage premium statistics
        """
        return self.wage_gap(
            group_column='UNION',
            reference_value=1,   # Union
            comparison_value=2,  # Non-union
            by=by,
            year=year,
            weighted=weighted,
            reference_label='Union',
            comparison_label='Non-union'
        )
    
    def sample(self, n: int = 1000, where: str = None, random_state: int = None) -> 'pd.DataFrame':
        """
        Get a random sample.
        
        Args:
            n: Number of rows
            where: Filter condition
            random_state: Random seed
            
        Returns:
            Sampled DataFrame
        """
        if random_state:
            self._connection.execute(f"SELECT setseed({random_state / 1000000.0})")
        
        sql = f"SELECT * FROM {self.TABLE_NAME}"
        if where:
            sql = f"{sql} WHERE {where}"
        
        # DuckDB uses TABLESAMPLE for random sampling
        sql = f"{sql} USING SAMPLE {n} ROWS"
        
        return self._connection.execute(sql).fetchdf()

    def register_df(self, df: 'pd.DataFrame', name: str = 'df', replace: bool = True) -> None:
        """Register a pandas DataFrame as a DuckDB table accessible via SQL.

        Args:
            df: DataFrame to register
            name: Table name to create (e.g., 'df')
            replace: Whether to replace an existing table
        """
        try:
            tmp_name = f"tmp_{name}"
            # Register temporary dataframe and create/replace a table for robust SQL usage
            self._connection.register(tmp_name, df)
            if replace:
                self._connection.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM {tmp_name}")
            else:
                self._connection.execute(f"CREATE TABLE IF NOT EXISTS {name} AS SELECT * FROM {tmp_name}")
            logger.info(f"Registered DataFrame as table '{name}' in DuckDB (rows={len(df)})")
        except Exception as e:
            logger.exception(f"Failed to register DataFrame as table '{name}': {e}")

    # =========================================================================
    # Legacy Compatibility
    # =========================================================================
    
    def get_wages(
        self,
        year: int = None,
        province: int = None,
        sample_size: int = None
    ) -> 'pd.DataFrame':
        """
        Legacy method for compatibility with existing notebooks.
        
        Prefer using query() or stream() for new code.
        """
        logger.warning("get_wages() is deprecated. Use query() or stream() instead.")
        
        conditions = ["HRLYEARN > 0"]
        
        if year:
            conditions.append(f"SURVYEAR = {year}")
        if province:
            conditions.append(f"PROV = {province}")
        
        where = ' AND '.join(conditions)
        
        if sample_size:
            return self.sample(n=sample_size, where=where)
        else:
            # Warning: may use lots of memory for full dataset
            return self.sql(f"SELECT * FROM {self.TABLE_NAME} WHERE {where}")
    
    # =========================================================================
    # Context Manager
    # =========================================================================
    
    def __enter__(self):
        """Enter context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and cleanup."""
        self.close()
    
    def close(self):
        """Close the data store and release resources."""
        try:
            self._connection.close()
            logger.info("EquiPayDataStore closed")
        except Exception as e:
            logger.warning(f"Error closing connection: {e}")
    
    # =========================================================================
    # Diagnostics
    # =========================================================================
    
    def health_check(self) -> Dict[str, Any]:
        """
        Run health check on the data store.
        
        Returns status of all components.
        """
        status = {
            'connection': False,
            'parquet_path': str(self.parquet_path),
            'parquet_exists': self.parquet_path.exists(),
            'row_count': 0,
            'years': [],
            'columns': 0,
            'memory': {},
            'cache': None,
            'views': 0
        }
        
        try:
            # Test connection
            self._connection.execute("SELECT 1")
            status['connection'] = True
            
            # Row count
            status['row_count'] = self.count()
            
            # Years
            status['years'] = self.years()
            
            # Columns
            status['columns'] = len(self.columns())
            
            # Memory
            status['memory'] = self._memory.current().to_dict()
            
            # Cache
            if self._cache:
                status['cache'] = self._cache.stats().to_dict()
            
            # Views
            status['views'] = len(self._views.list())
            
        except Exception as e:
            status['error'] = str(e)
        
        return status
    
    def __repr__(self):
        try:
            count = self.count()
            years = self.years()
            year_range = f"{min(years)}-{max(years)}" if years else "no data"
            return f"EquiPayDataStore({count:,} rows, years={year_range})"
        except Exception:
            return "EquiPayDataStore(not initialized)"
