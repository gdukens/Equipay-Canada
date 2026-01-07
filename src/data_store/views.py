"""
EquiPay Canada - Materialized Views
===================================

Persistent and virtual views for dashboard/API performance.

Provides:
1. Materialized views for expensive aggregations
2. View registry with dependency tracking
3. Incremental refresh for new data
4. Memory-efficient access patterns

Materialized views pre-compute common aggregations:
- Gender gap by province/year
- Gap by occupation/education
- Time series for dashboards
- Summary statistics

Usage:
    views = MaterializedViewManager(connection)
    
    # Create view
    views.create('gap_by_province_year', '''
        SELECT PROV, SURVYEAR, 
               AVG(CASE WHEN IS_FEMALE=0 THEN LOG_REAL_HRLYEARN END) -
               AVG(CASE WHEN IS_FEMALE=1 THEN LOG_REAL_HRLYEARN END) as gap
        FROM lfs WHERE HRLYEARN > 0
        GROUP BY PROV, SURVYEAR
    ''')
    
    # Query view
    df = views.query('gap_by_province_year', where="PROV = 35")
"""

import logging
import hashlib
import json
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

if TYPE_CHECKING:
    import pandas as pd
    import duckdb

logger = logging.getLogger(__name__)


@dataclass
class ViewDefinition:
    """Definition of a materialized or virtual view."""
    name: str
    sql: str
    
    # Type
    is_materialized: bool = True
    
    # Metadata
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    last_refresh: Optional[datetime] = None
    
    # Dependencies
    depends_on: List[str] = field(default_factory=list)
    
    # Refresh policy
    refresh_interval_hours: int = 24
    
    # Storage
    storage_path: Optional[str] = None
    
    @property
    def is_stale(self) -> bool:
        """Check if view needs refresh."""
        if self.last_refresh is None:
            return True
        hours_since = (datetime.now() - self.last_refresh).total_seconds() / 3600
        return hours_since > self.refresh_interval_hours
    
    def fingerprint(self) -> str:
        """Generate fingerprint for cache invalidation."""
        content = f"{self.name}:{self.sql}"
        return hashlib.md5(content.encode()).hexdigest()[:12]


class MaterializedViewManager:
    """
    Manages materialized views for expensive aggregations.
    
    Views are stored as Parquet files for persistence across sessions.
    Virtual views (not materialized) are created as DuckDB views.
    """
    
    # Standard dashboard views
    DASHBOARD_VIEWS = {
        'gap_by_province_year': """
            SELECT 
                PROV,
                SURVYEAR,
                SUM(CASE WHEN IS_FEMALE = 0 THEN FINALWT END) as male_weight,
                SUM(CASE WHEN IS_FEMALE = 1 THEN FINALWT END) as female_weight,
                SUM(CASE WHEN IS_FEMALE = 0 THEN LOG_REAL_HRLYEARN * FINALWT END) / 
                    NULLIF(SUM(CASE WHEN IS_FEMALE = 0 THEN FINALWT END), 0) as male_mean,
                SUM(CASE WHEN IS_FEMALE = 1 THEN LOG_REAL_HRLYEARN * FINALWT END) / 
                    NULLIF(SUM(CASE WHEN IS_FEMALE = 1 THEN FINALWT END), 0) as female_mean,
                (SUM(CASE WHEN IS_FEMALE = 0 THEN LOG_REAL_HRLYEARN * FINALWT END) / 
                    NULLIF(SUM(CASE WHEN IS_FEMALE = 0 THEN FINALWT END), 0)) -
                (SUM(CASE WHEN IS_FEMALE = 1 THEN LOG_REAL_HRLYEARN * FINALWT END) / 
                    NULLIF(SUM(CASE WHEN IS_FEMALE = 1 THEN FINALWT END), 0)) as gap,
                COUNT(*) as n_obs
            FROM lfs
            WHERE HRLYEARN > 0
            GROUP BY PROV, SURVYEAR
            ORDER BY PROV, SURVYEAR
        """,
        
        'gap_by_occupation_year': """
            SELECT 
                NOC_10,
                SURVYEAR,
                SUM(FINALWT) as total_weight,
                SUM(CASE WHEN IS_FEMALE = 1 THEN FINALWT END) / SUM(FINALWT) as female_share,
                (SUM(CASE WHEN IS_FEMALE = 0 THEN LOG_REAL_HRLYEARN * FINALWT END) / 
                    NULLIF(SUM(CASE WHEN IS_FEMALE = 0 THEN FINALWT END), 0)) -
                (SUM(CASE WHEN IS_FEMALE = 1 THEN LOG_REAL_HRLYEARN * FINALWT END) / 
                    NULLIF(SUM(CASE WHEN IS_FEMALE = 1 THEN FINALWT END), 0)) as gap,
                COUNT(*) as n_obs
            FROM lfs
            WHERE HRLYEARN > 0
            GROUP BY NOC_10, SURVYEAR
            ORDER BY NOC_10, SURVYEAR
        """,
        
        'gap_by_education_year': """
            SELECT 
                EDUC,
                SURVYEAR,
                (SUM(CASE WHEN IS_FEMALE = 0 THEN LOG_REAL_HRLYEARN * FINALWT END) / 
                    NULLIF(SUM(CASE WHEN IS_FEMALE = 0 THEN FINALWT END), 0)) -
                (SUM(CASE WHEN IS_FEMALE = 1 THEN LOG_REAL_HRLYEARN * FINALWT END) / 
                    NULLIF(SUM(CASE WHEN IS_FEMALE = 1 THEN FINALWT END), 0)) as gap,
                AVG(LOG_REAL_HRLYEARN) as mean_log_wage,
                COUNT(*) as n_obs
            FROM lfs
            WHERE HRLYEARN > 0
            GROUP BY EDUC, SURVYEAR
            ORDER BY EDUC, SURVYEAR
        """,
        
        'summary_by_year': """
            SELECT 
                SURVYEAR,
                COUNT(*) as n_obs,
                SUM(FINALWT) as sum_weight,
                AVG(HRLYEARN) as mean_wage,
                STDDEV(HRLYEARN) as std_wage,
                QUANTILE_CONT(HRLYEARN, 0.5) as median_wage,
                AVG(CASE WHEN IS_FEMALE = 1 THEN HRLYEARN END) as female_mean_wage,
                AVG(CASE WHEN IS_FEMALE = 0 THEN HRLYEARN END) as male_mean_wage,
                (AVG(CASE WHEN IS_FEMALE = 0 THEN HRLYEARN END) - 
                 AVG(CASE WHEN IS_FEMALE = 1 THEN HRLYEARN END)) / 
                    NULLIF(AVG(CASE WHEN IS_FEMALE = 0 THEN HRLYEARN END), 0) * 100 as raw_gap_pct
            FROM lfs
            WHERE HRLYEARN > 0
            GROUP BY SURVYEAR
            ORDER BY SURVYEAR
        """,
        
        'intersectional_gaps': """
            SELECT 
                SURVYEAR,
                EDUC,
                NOC_10,
                PROV,
                SUM(CASE WHEN IS_FEMALE = 0 THEN FINALWT END) as male_weight,
                SUM(CASE WHEN IS_FEMALE = 1 THEN FINALWT END) as female_weight,
                (SUM(CASE WHEN IS_FEMALE = 0 THEN LOG_REAL_HRLYEARN * FINALWT END) / 
                    NULLIF(SUM(CASE WHEN IS_FEMALE = 0 THEN FINALWT END), 0)) -
                (SUM(CASE WHEN IS_FEMALE = 1 THEN LOG_REAL_HRLYEARN * FINALWT END) / 
                    NULLIF(SUM(CASE WHEN IS_FEMALE = 1 THEN FINALWT END), 0)) as gap,
                COUNT(*) as n_obs
            FROM lfs
            WHERE HRLYEARN > 0
            GROUP BY SURVYEAR, EDUC, NOC_10, PROV
            HAVING COUNT(*) >= 30
        """
    }
    
    def __init__(
        self,
        connection: 'duckdb.DuckDBPyConnection',
        storage_dir: str = "data/views"
    ):
        """
        Initialize view manager.
        
        Args:
            connection: DuckDB connection
            storage_dir: Directory for materialized view storage
        """
        self.connection = connection
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # View registry
        self._views: Dict[str, ViewDefinition] = {}
        
        # Load metadata if exists
        self._load_registry()
    
    def _load_registry(self):
        """Load view registry from disk."""
        registry_path = self.storage_dir / "registry.json"
        if registry_path.exists():
            try:
                with open(registry_path, 'r') as f:
                    data = json.load(f)
                
                for name, view_data in data.items():
                    self._views[name] = ViewDefinition(
                        name=name,
                        sql=view_data['sql'],
                        is_materialized=view_data.get('is_materialized', True),
                        description=view_data.get('description', ''),
                        created_at=datetime.fromisoformat(view_data['created_at']),
                        last_refresh=datetime.fromisoformat(view_data['last_refresh']) if view_data.get('last_refresh') else None,
                        storage_path=view_data.get('storage_path')
                    )
                
                logger.info(f"Loaded {len(self._views)} views from registry")
            except Exception as e:
                logger.warning(f"Failed to load view registry: {e}")
    
    def _save_registry(self):
        """Save view registry to disk."""
        registry_path = self.storage_dir / "registry.json"
        
        data = {}
        for name, view in self._views.items():
            data[name] = {
                'sql': view.sql,
                'is_materialized': view.is_materialized,
                'description': view.description,
                'created_at': view.created_at.isoformat(),
                'last_refresh': view.last_refresh.isoformat() if view.last_refresh else None,
                'storage_path': view.storage_path
            }
        
        with open(registry_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def create(
        self,
        name: str,
        sql: str,
        materialize: bool = True,
        description: str = "",
        refresh: bool = True
    ):
        """
        Create a new view.
        
        Args:
            name: View name
            sql: SQL query defining the view
            materialize: If True, persist to Parquet; else create DB view
            description: Human-readable description
            refresh: If True, compute immediately
        """
        logger.info(f"Creating view '{name}' (materialized={materialize})")
        
        storage_path = str(self.storage_dir / f"{name}.parquet") if materialize else None
        
        view = ViewDefinition(
            name=name,
            sql=sql,
            is_materialized=materialize,
            description=description,
            storage_path=storage_path
        )
        
        self._views[name] = view
        
        if materialize and refresh:
            self.refresh(name)
        elif not materialize:
            # Create as DuckDB view
            self.connection.execute(f"CREATE OR REPLACE VIEW {name} AS {sql}")
        
        self._save_registry()
    
    def refresh(self, name: str):
        """
        Refresh a materialized view.
        
        Executes the SQL and saves to Parquet.
        """
        if name not in self._views:
            raise ValueError(f"View '{name}' not found")
        
        view = self._views[name]
        
        if not view.is_materialized:
            logger.info(f"View '{name}' is not materialized, skipping refresh")
            return
        
        logger.info(f"Refreshing materialized view '{name}'")
        
        # Execute query and save to Parquet
        self.connection.execute(f"""
            COPY ({view.sql}) TO '{view.storage_path}' (FORMAT PARQUET)
        """)
        
        # Update metadata
        view.last_refresh = datetime.now()
        self._save_registry()
        
        # Register as table for querying
        self.connection.execute(f"""
            CREATE OR REPLACE VIEW {name} AS 
            SELECT * FROM read_parquet('{view.storage_path}')
        """)
        
        logger.info(f"View '{name}' refreshed successfully")
    
    def refresh_all(self, force: bool = False):
        """
        Refresh all stale materialized views.
        
        Args:
            force: If True, refresh all views regardless of staleness
        """
        for name, view in self._views.items():
            if view.is_materialized and (force or view.is_stale):
                try:
                    self.refresh(name)
                except Exception as e:
                    logger.error(f"Failed to refresh view '{name}': {e}")
    
    def query(
        self,
        name: str,
        where: str = None,
        limit: int = None
    ) -> 'pd.DataFrame':
        """
        Query a view.
        
        Args:
            name: View name
            where: Optional WHERE clause
            limit: Optional row limit
            
        Returns:
            DataFrame with query results
        """
        if name not in self._views:
            raise ValueError(f"View '{name}' not found")
        
        view = self._views[name]
        
        # Ensure view is available
        if view.is_materialized:
            if view.storage_path and Path(view.storage_path).exists():
                # Register if not already
                self.connection.execute(f"""
                    CREATE OR REPLACE VIEW {name} AS 
                    SELECT * FROM read_parquet('{view.storage_path}')
                """)
            else:
                # Need to refresh first
                self.refresh(name)
        
        # Build query
        sql = f"SELECT * FROM {name}"
        if where:
            sql = f"{sql} WHERE {where}"
        if limit:
            sql = f"{sql} LIMIT {limit}"
        
        return self.connection.execute(sql).fetchdf()
    
    def create_dashboard_views(self, refresh: bool = True):
        """
        Create all standard dashboard views.
        
        Call this once during initialization to set up views
        needed by the dashboard.
        """
        for name, sql in self.DASHBOARD_VIEWS.items():
            self.create(
                name=name,
                sql=sql,
                materialize=True,
                description=f"Dashboard view: {name}",
                refresh=refresh
            )
    
    def get(self, name: str) -> 'pd.DataFrame':
        """
        Get view contents (alias for query without filters).
        """
        return self.query(name)
    
    def exists(self, name: str) -> bool:
        """Check if view exists."""
        return name in self._views
    
    def list(self) -> List[str]:
        """List all registered views."""
        return list(self._views.keys())
    
    def info(self, name: str) -> Dict[str, Any]:
        """Get information about a view."""
        if name not in self._views:
            return {'error': f"View '{name}' not found"}
        
        view = self._views[name]
        return {
            'name': view.name,
            'is_materialized': view.is_materialized,
            'description': view.description,
            'created_at': view.created_at.isoformat(),
            'last_refresh': view.last_refresh.isoformat() if view.last_refresh else None,
            'is_stale': view.is_stale,
            'storage_path': view.storage_path,
            'fingerprint': view.fingerprint()
        }
    
    def drop(self, name: str):
        """Drop a view."""
        if name not in self._views:
            return
        
        view = self._views[name]
        
        # Remove from DuckDB
        try:
            self.connection.execute(f"DROP VIEW IF EXISTS {name}")
        except Exception:
            pass
        
        # Remove storage file
        if view.storage_path and Path(view.storage_path).exists():
            Path(view.storage_path).unlink()
        
        # Remove from registry
        del self._views[name]
        self._save_registry()
        
        logger.info(f"Dropped view '{name}'")
    
    def incremental_refresh(
        self,
        name: str,
        new_data_filter: str = None,
        year: int = None
    ):
        """
        Incrementally refresh a view with new data.
        
        For append-only patterns (new years of data).
        
        Args:
            name: View name
            new_data_filter: Filter for new rows
            year: If provided, append data for this year only
        """
        if name not in self._views:
            raise ValueError(f"View '{name}' not found")
        
        view = self._views[name]
        
        if not view.is_materialized:
            logger.info("View is not materialized, no incremental refresh needed")
            return
        
        if year:
            new_data_filter = f"SURVYEAR = {year}"
        
        if not new_data_filter:
            # Full refresh
            self.refresh(name)
            return
        
        logger.info(f"Incrementally refreshing view '{name}' with filter: {new_data_filter}")
        
        # Modify SQL to filter for new data only
        original_sql = view.sql
        
        # Add filter to SQL (simplified - assumes SQL has WHERE clause)
        if 'WHERE' in original_sql.upper():
            incremental_sql = original_sql.replace(
                'WHERE', 
                f'WHERE ({new_data_filter}) AND '
            )
        else:
            incremental_sql = original_sql + f' WHERE {new_data_filter}'
        
        # Append to existing Parquet
        temp_path = str(self.storage_dir / f"{name}_temp.parquet")
        
        # Write new data
        self.connection.execute(f"""
            COPY ({incremental_sql}) TO '{temp_path}' (FORMAT PARQUET)
        """)
        
        # Combine with existing
        self.connection.execute(f"""
            COPY (
                SELECT * FROM read_parquet('{view.storage_path}')
                UNION ALL
                SELECT * FROM read_parquet('{temp_path}')
            ) TO '{view.storage_path}' (FORMAT PARQUET)
        """)
        
        # Cleanup temp
        Path(temp_path).unlink(missing_ok=True)
        
        view.last_refresh = datetime.now()
        self._save_registry()
        
        logger.info(f"Incremental refresh complete for '{name}'")
    
    def __repr__(self):
        mat_count = sum(1 for v in self._views.values() if v.is_materialized)
        return f"MaterializedViewManager({len(self._views)} views, {mat_count} materialized)"
