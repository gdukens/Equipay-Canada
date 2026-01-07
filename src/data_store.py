"""
EquiPay Canada - DuckDB Data Store
==================================

Unified data access layer using DuckDB for memory-efficient queries.

This module provides:
- Lazy loading of LFS data (only loads what's needed)
- SQL interface for complex queries
- Convenience methods for common analysis patterns
- Parquet-optimized storage with predicate pushdown

Memory Usage:
- Queries use ~50-200MB regardless of dataset size
- Full dataset (2.1GB CSV) stored as ~300MB Parquet

Usage:
    from src.data_store import EquiPayDataStore
    
    store = EquiPayDataStore()
    
    # SQL queries
    df = store.query("SELECT PROV, AVG(HRLYEARN) FROM lfs GROUP BY PROV")
    
    # Convenience methods
    df = store.get_wages(years=[2023], provinces=['ON', 'BC'])
    stats = store.get_gender_gap(by=['PROV', 'SURVYEAR'])
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import logging
import os

from .constants import (
    COLS, GENDER_CODES, PROVINCE_CODES, EDUCATION_CODES,
    NOC_10_CODES, NAICS_21_CODES, DATA_SCOPE_START, DATA_SCOPE_END
)

logger = logging.getLogger(__name__)


class EquiPayDataStore:
    """
    DuckDB-backed data store for EquiPay Canada.
    
    Provides memory-efficient access to LFS data through:
    - Parquet files (preferred, fastest)
    - Raw CSV files (fallback, still efficient via DuckDB)
    """
    
    def __init__(
        self,
        parquet_path: str = "data/parquet",
        raw_csv_path: str = "data/raw/lfs",
        memory_limit: str = "4GB",
        threads: int = None
    ):
        """
        Initialize the data store.
        
        Args:
            parquet_path: Path to consolidated Parquet file or directory
            raw_csv_path: Path to raw CSV files (fallback)
            memory_limit: DuckDB memory limit (default 4GB for 8GB RAM system)
            threads: Number of threads (None = auto)
        """
        self.parquet_path = Path(parquet_path)
        self.raw_csv_path = Path(raw_csv_path)
        self.memory_limit = memory_limit
        
        # Initialize DuckDB connection with memory limits
        self.conn = duckdb.connect(":memory:")
        self.conn.execute(f"SET memory_limit = '{memory_limit}'")
        if threads:
            self.conn.execute(f"SET threads = {threads}")
        
        # Track data source
        self._source = None
        self._initialized = False
        
    def _initialize(self):
        """Lazy initialization - register data sources."""
        if self._initialized:
            return
            
        # Try Parquet first (fastest)
        # Check if parquet directory exists and has parquet files
        parquet_files_exist = False
        if self.parquet_path.exists():
            if self.parquet_path.is_dir():
                # Check for parquet files in subdirectories (hive partitioned)
                parquet_files = list(self.parquet_path.glob("**/*.parquet"))
                parquet_files_exist = len(parquet_files) > 0
            else:
                # Single parquet file
                parquet_files_exist = self.parquet_path.suffix == '.parquet'
        
        # CPI deflators for inflation adjustment (BASE_YEAR = 2010)
        # Format: year -> deflator (multiply nominal by deflator to get real)
        cpi_deflators = {
            2010: 1.0000, 2011: 0.9716, 2012: 0.9572, 2013: 0.9487,
            2014: 0.9305, 2015: 0.9202, 2016: 0.9074, 2017: 0.8936,
            2018: 0.8733, 2019: 0.8566, 2020: 0.8504, 2021: 0.8227,
            2022: 0.7704, 2023: 0.7416, 2024: 0.7213, 2025: 0.7061
        }
        
        # Build CASE statement for CPI deflation
        deflator_cases = " ".join([
            f"WHEN SURVYEAR = {year} THEN {deflator}" 
            for year, deflator in cpi_deflators.items()
        ])
        deflator_sql = f"CASE {deflator_cases} ELSE 1.0 END"
        
        # Hours variables with implicit decimal (divide by 10)
        # Per StatsCan LFS PUMF Guide: AHRSMAIN, UHRSMAIN, ATOTHRS, UTOTHRS, HRSAWAY have 1 implicit decimal
        hours_vars = ['AHRSMAIN', 'UHRSMAIN', 'ATOTHRS', 'UTOTHRS', 'HRSAWAY']
        hours_exclude = ', '.join(hours_vars)
        hours_transform = ', '.join([f'{v} / 10.0 AS {v}' for v in hours_vars])
        
        if parquet_files_exist:
            if self.parquet_path.is_dir():
                # Partitioned Parquet
                parquet_glob = str(self.parquet_path / "**/*.parquet")
                # LFS PUMF data transformations per StatsCan Guide:
                # - HRLYEARN has 2 implicit decimals (divide by 100 → dollars)
                # - Hours variables have 1 implicit decimal (divide by 10 → hours)
                # - REAL_HRLYEARN computed using CPI deflators (2010 constant dollars)
                self.conn.execute(f"""
                    CREATE OR REPLACE VIEW lfs AS 
                    SELECT 
                        * EXCLUDE (HRLYEARN, {hours_exclude}),
                        HRLYEARN / 100.0 AS HRLYEARN,
                        (HRLYEARN / 100.0) * ({deflator_sql}) AS REAL_HRLYEARN,
                        LN(GREATEST(HRLYEARN / 100.0, 0.01)) AS LOG_HRLYEARN,
                        LN(GREATEST((HRLYEARN / 100.0) * ({deflator_sql}), 0.01)) AS LOG_REAL_HRLYEARN,
                        {hours_transform}
                    FROM read_parquet('{parquet_glob}', hive_partitioning=true)
                """)
            else:
                # Single Parquet file
                # LFS PUMF data transformations per StatsCan Guide
                self.conn.execute(f"""
                    CREATE OR REPLACE VIEW lfs AS 
                    SELECT 
                        * EXCLUDE (HRLYEARN, {hours_exclude}),
                        HRLYEARN / 100.0 AS HRLYEARN,
                        (HRLYEARN / 100.0) * ({deflator_sql}) AS REAL_HRLYEARN,
                        LN(GREATEST(HRLYEARN / 100.0, 0.01)) AS LOG_HRLYEARN,
                        LN(GREATEST((HRLYEARN / 100.0) * ({deflator_sql}), 0.01)) AS LOG_REAL_HRLYEARN,
                        {hours_transform}
                    FROM read_parquet('{self.parquet_path}')
                """)
            self._source = "parquet"
            logger.info(f"DuckDB: Registered Parquet source: {self.parquet_path}")
            
        # Fallback to CSV (raw LFS files also store HRLYEARN in cents)
        elif self.raw_csv_path.exists():
            csv_glob = str(self.raw_csv_path / "*.csv")
            # DuckDB efficiently reads CSVs with automatic schema detection
            # Apply StatsCan PUMF transformations: HRLYEARN/100, hours/10
            self.conn.execute(f"""
                CREATE OR REPLACE VIEW lfs AS 
                SELECT 
                    * EXCLUDE (HRLYEARN, {hours_exclude}),
                    HRLYEARN / 100.0 AS HRLYEARN,
                    (HRLYEARN / 100.0) * ({deflator_sql}) AS REAL_HRLYEARN,
                    LN(GREATEST(HRLYEARN / 100.0, 0.01)) AS LOG_HRLYEARN,
                    LN(GREATEST((HRLYEARN / 100.0) * ({deflator_sql}), 0.01)) AS LOG_REAL_HRLYEARN,
                    {hours_transform}
                FROM read_csv_auto('{csv_glob}', 
                    header=true,
                    ignore_errors=true,
                    parallel=true
                )
            """)
            self._source = "csv"
            logger.info(f"DuckDB: Registered CSV source: {self.raw_csv_path}")
        else:
            raise FileNotFoundError(
                f"No data found. Expected Parquet at {self.parquet_path} "
                f"or CSVs at {self.raw_csv_path}"
            )
        
        self._initialized = True
        
    @property
    def source(self) -> str:
        """Return the current data source type."""
        self._initialize()
        return self._source
    
    # =========================================================================
    # CORE QUERY METHODS
    # =========================================================================
    
    def query(self, sql: str) -> pd.DataFrame:
        """
        Execute a SQL query and return results as pandas DataFrame.
        
        Args:
            sql: SQL query string. Use 'lfs' as the table name.
            
        Returns:
            pandas DataFrame with query results
            
        Example:
            df = store.query("SELECT PROV, AVG(HRLYEARN) FROM lfs GROUP BY PROV")
        """
        self._initialize()
        return self.conn.execute(sql).fetchdf()
    
    def query_arrow(self, sql: str):
        """
        Execute a SQL query and return results as PyArrow Table.
        More memory efficient for large results.
        """
        self._initialize()
        return self.conn.execute(sql).fetch_arrow_table()
    
    def execute(self, sql: str):
        """Execute a SQL statement without returning results."""
        self._initialize()
        self.conn.execute(sql)
        
    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================
    
    def get_wages(
        self,
        years: Optional[List[int]] = None,
        provinces: Optional[List[str]] = None,
        genders: Optional[List[int]] = None,
        columns: Optional[List[str]] = None,
        limit: Optional[int] = None,
        valid_wages_only: bool = True
    ) -> pd.DataFrame:
        """
        Get wage data with optional filters.
        
        Args:
            years: List of years to include (e.g., [2020, 2021, 2022])
            provinces: List of province codes (e.g., ['ON', 'BC']) or ints
            genders: List of gender codes (1=Male, 2=Female)
            columns: Specific columns to select (None = all)
            limit: Maximum rows to return
            valid_wages_only: If True, filter HRLYEARN > 0
            
        Returns:
            pandas DataFrame with filtered wage data
        """
        # Build column list
        if columns:
            col_str = ", ".join(columns)
        else:
            col_str = "*"
            
        # Build WHERE clause
        conditions = []
        
        if valid_wages_only:
            conditions.append("HRLYEARN > 0")
            
        if years:
            years_str = ", ".join(str(y) for y in years)
            conditions.append(f"SURVYEAR IN ({years_str})")
            
        if provinces:
            # Handle both string codes and integer codes
            if isinstance(provinces[0], str):
                # Convert string codes to integers
                prov_map = {v: k for k, v in PROVINCE_CODES.items()}
                prov_ints = [prov_map.get(p, p) for p in provinces]
                prov_str = ", ".join(str(p) for p in prov_ints)
            else:
                prov_str = ", ".join(str(p) for p in provinces)
            conditions.append(f"PROV IN ({prov_str})")
            
        if genders:
            genders_str = ", ".join(str(g) for g in genders)
            conditions.append(f"GENDER IN ({genders_str})")
            
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        limit_clause = f"LIMIT {limit}" if limit else ""
        
        sql = f"SELECT {col_str} FROM lfs WHERE {where_clause} {limit_clause}"
        return self.query(sql)
    
    def get_gender_gap(
        self,
        by: Optional[List[str]] = None,
        years: Optional[List[int]] = None,
        weighted: bool = True
    ) -> pd.DataFrame:
        """
        Calculate gender wage gap statistics.
        
        Args:
            by: Group by columns (e.g., ['PROV', 'SURVYEAR'])
            years: Filter to specific years
            weighted: Use survey weights (FINALWT)
            
        Returns:
            DataFrame with columns: [grouping cols], male_wage, female_wage, 
            raw_gap, raw_gap_pct
        """
        group_cols = by or []
        group_str = ", ".join(group_cols) if group_cols else ""
        group_by = f"GROUP BY {group_str}" if group_cols else ""
        select_groups = f"{group_str}," if group_cols else ""
        
        year_filter = ""
        if years:
            years_str = ", ".join(str(y) for y in years)
            year_filter = f"AND SURVYEAR IN ({years_str})"
        
        if weighted:
            sql = f"""
                SELECT 
                    {select_groups}
                    SUM(CASE WHEN GENDER = 1 THEN HRLYEARN * FINALWT END) / 
                        NULLIF(SUM(CASE WHEN GENDER = 1 THEN FINALWT END), 0) as male_wage,
                    SUM(CASE WHEN GENDER = 2 THEN HRLYEARN * FINALWT END) / 
                        NULLIF(SUM(CASE WHEN GENDER = 2 THEN FINALWT END), 0) as female_wage,
                    SUM(CASE WHEN GENDER = 1 THEN FINALWT END) as male_n,
                    SUM(CASE WHEN GENDER = 2 THEN FINALWT END) as female_n
                FROM lfs
                WHERE HRLYEARN > 0 AND FINALWT > 0 {year_filter}
                {group_by}
            """
        else:
            sql = f"""
                SELECT 
                    {select_groups}
                    AVG(CASE WHEN GENDER = 1 THEN HRLYEARN END) as male_wage,
                    AVG(CASE WHEN GENDER = 2 THEN HRLYEARN END) as female_wage,
                    COUNT(CASE WHEN GENDER = 1 THEN 1 END) as male_n,
                    COUNT(CASE WHEN GENDER = 2 THEN 1 END) as female_n
                FROM lfs
                WHERE HRLYEARN > 0 {year_filter}
                {group_by}
            """
        
        df = self.query(sql)
        
        # View already converts HRLYEARN from cents to dollars
        # Just calculate gap metrics
        df['raw_gap'] = df['male_wage'] - df['female_wage']
        df['raw_gap_pct'] = (df['raw_gap'] / df['male_wage']) * 100
        
        return df
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics about the dataset."""
        self._initialize()
        
        stats = {}
        
        # Row count
        result = self.query("SELECT COUNT(*) as n FROM lfs")
        stats['total_records'] = int(result['n'].iloc[0])
        
        # Valid wage records
        result = self.query("SELECT COUNT(*) as n FROM lfs WHERE HRLYEARN > 0")
        stats['valid_wage_records'] = int(result['n'].iloc[0])
        
        # Year range
        result = self.query("SELECT MIN(SURVYEAR) as min_yr, MAX(SURVYEAR) as max_yr FROM lfs")
        stats['year_range'] = (int(result['min_yr'].iloc[0]), int(result['max_yr'].iloc[0]))
        
        # Records by year
        result = self.query("""
            SELECT SURVYEAR, COUNT(*) as n 
            FROM lfs 
            GROUP BY SURVYEAR 
            ORDER BY SURVYEAR
        """)
        stats['records_by_year'] = dict(zip(result['SURVYEAR'], result['n']))
        
        # Data source
        stats['source'] = self._source
        
        return stats
    
    def get_yearly_stats(self, weighted: bool = True) -> pd.DataFrame:
        """Get yearly aggregate statistics."""
        if weighted:
            sql = """
                SELECT 
                    SURVYEAR as year,
                    SUM(FINALWT) as population,
                    SUM(CASE WHEN HRLYEARN > 0 THEN FINALWT END) as employed_with_wages,
                    SUM(CASE WHEN HRLYEARN > 0 THEN HRLYEARN * FINALWT END) / 
                        NULLIF(SUM(CASE WHEN HRLYEARN > 0 THEN FINALWT END), 0) as avg_wage,
                    SUM(CASE WHEN GENDER = 2 AND HRLYEARN > 0 THEN HRLYEARN * FINALWT END) / 
                        NULLIF(SUM(CASE WHEN GENDER = 2 AND HRLYEARN > 0 THEN FINALWT END), 0) as female_avg_wage,
                    SUM(CASE WHEN GENDER = 1 AND HRLYEARN > 0 THEN HRLYEARN * FINALWT END) / 
                        NULLIF(SUM(CASE WHEN GENDER = 1 AND HRLYEARN > 0 THEN FINALWT END), 0) as male_avg_wage
                FROM lfs
                WHERE FINALWT > 0
                GROUP BY SURVYEAR
                ORDER BY SURVYEAR
            """
        else:
            sql = """
                SELECT 
                    SURVYEAR as year,
                    COUNT(*) as records,
                    AVG(CASE WHEN HRLYEARN > 0 THEN HRLYEARN END) as avg_wage,
                    AVG(CASE WHEN GENDER = 2 AND HRLYEARN > 0 THEN HRLYEARN END) as female_avg_wage,
                    AVG(CASE WHEN GENDER = 1 AND HRLYEARN > 0 THEN HRLYEARN END) as male_avg_wage
                FROM lfs
                GROUP BY SURVYEAR
                ORDER BY SURVYEAR
            """
        
        df = self.query(sql)
        df['wage_gap_pct'] = ((df['male_avg_wage'] - df['female_avg_wage']) / df['male_avg_wage']) * 100
        return df
    
    def get_provincial_stats(self, year: Optional[int] = None, weighted: bool = True) -> pd.DataFrame:
        """Get provincial aggregate statistics."""
        year_filter = f"AND SURVYEAR = {year}" if year else ""
        
        if weighted:
            sql = f"""
                SELECT 
                    PROV,
                    SUM(FINALWT) as population,
                    SUM(CASE WHEN HRLYEARN > 0 THEN HRLYEARN * FINALWT END) / 
                        NULLIF(SUM(CASE WHEN HRLYEARN > 0 THEN FINALWT END), 0) as avg_wage,
                    SUM(CASE WHEN GENDER = 2 AND HRLYEARN > 0 THEN HRLYEARN * FINALWT END) / 
                        NULLIF(SUM(CASE WHEN GENDER = 2 AND HRLYEARN > 0 THEN FINALWT END), 0) as female_avg_wage,
                    SUM(CASE WHEN GENDER = 1 AND HRLYEARN > 0 THEN HRLYEARN * FINALWT END) / 
                        NULLIF(SUM(CASE WHEN GENDER = 1 AND HRLYEARN > 0 THEN FINALWT END), 0) as male_avg_wage
                FROM lfs
                WHERE FINALWT > 0 {year_filter}
                GROUP BY PROV
                ORDER BY PROV
            """
        else:
            sql = f"""
                SELECT 
                    PROV,
                    COUNT(*) as records,
                    AVG(CASE WHEN HRLYEARN > 0 THEN HRLYEARN END) as avg_wage,
                    AVG(CASE WHEN GENDER = 2 AND HRLYEARN > 0 THEN HRLYEARN END) as female_avg_wage,
                    AVG(CASE WHEN GENDER = 1 AND HRLYEARN > 0 THEN HRLYEARN END) as male_avg_wage
                FROM lfs
                WHERE 1=1 {year_filter}
                GROUP BY PROV
                ORDER BY PROV
            """
        
        df = self.query(sql)
        df['wage_gap_pct'] = ((df['male_avg_wage'] - df['female_avg_wage']) / df['male_avg_wage']) * 100
        
        # Add province labels
        df['province_name'] = df['PROV'].map(PROVINCE_CODES)
        
        return df
    
    # =========================================================================
    # WEIGHTED STATISTICS METHODS
    # =========================================================================
    
    def get_weighted_stats(
        self,
        column: str = 'HRLYEARN',
        by: Optional[List[str]] = None,
        years: Optional[List[int]] = None,
        gender: Optional[int] = None,
        convert_cents: bool = True  # Deprecated: view already converts
    ) -> pd.DataFrame:
        """
        Get weighted statistics for a column.
        
        Args:
            column: Column to compute statistics for
            by: Grouping columns
            years: Filter by years
            gender: Filter by gender (1=Male, 2=Female)
            convert_cents: DEPRECATED - view already converts HRLYEARN to dollars
            
        Returns:
            DataFrame with weighted mean, std, quantiles
        """
        self._initialize()
        
        group_cols = by or []
        group_str = ", ".join(group_cols) if group_cols else ""
        group_by = f"GROUP BY {group_str}" if group_cols else ""
        select_groups = f"{group_str}," if group_cols else ""
        
        # Build filters
        filters = ["FINALWT > 0", f"{column} IS NOT NULL"]
        if column == 'HRLYEARN':
            filters.append(f"{column} > 0")
        if years:
            years_str = ", ".join(str(y) for y in years)
            filters.append(f"SURVYEAR IN ({years_str})")
        if gender:
            filters.append(f"GENDER = {gender}")
        
        where_clause = " AND ".join(filters)
        
        sql = f"""
            SELECT 
                {select_groups}
                SUM({column} * FINALWT) / SUM(FINALWT) as weighted_mean,
                SQRT(
                    SUM(FINALWT * POWER({column} - (
                        SELECT SUM({column} * FINALWT) / SUM(FINALWT) FROM lfs WHERE {where_clause}
                    ), 2)) / SUM(FINALWT)
                ) as weighted_std,
                SUM(FINALWT) as total_weight,
                COUNT(*) as n_obs,
                PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY {column}) as p10,
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {column}) as p25,
                PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY {column}) as median,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {column}) as p75,
                PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY {column}) as p90
            FROM lfs
            WHERE {where_clause}
            {group_by}
        """
        
        df = self.query(sql)
        
        # View already converts HRLYEARN from cents to dollars
        # No additional conversion needed
        
        return df
    
    def get_weighted_wage_gap_by_group(
        self,
        group_by: str,
        years: Optional[List[int]] = None,
        min_obs: int = 30
    ) -> pd.DataFrame:
        """
        Get weighted gender wage gap by any grouping variable.
        
        Args:
            group_by: Column to group by (e.g., 'NOC_10', 'EDUC', 'PROV')
            years: Filter by years
            min_obs: Minimum observations per cell
            
        Returns:
            DataFrame with wage gap by group
        """
        self._initialize()
        
        year_filter = ""
        if years:
            years_str = ", ".join(str(y) for y in years)
            year_filter = f"AND SURVYEAR IN ({years_str})"
        
        sql = f"""
            WITH wage_by_group AS (
                SELECT 
                    {group_by},
                    GENDER,
                    SUM(HRLYEARN * FINALWT) / SUM(FINALWT) as avg_wage,
                    SUM(FINALWT) as population,
                    COUNT(*) as n_obs
                FROM lfs
                WHERE HRLYEARN > 0 AND FINALWT > 0 {year_filter}
                GROUP BY {group_by}, GENDER
                HAVING COUNT(*) >= {min_obs}
            )
            SELECT 
                m.{group_by},
                m.avg_wage as male_wage,
                f.avg_wage as female_wage,
                m.avg_wage - f.avg_wage as wage_gap,
                (m.avg_wage - f.avg_wage) / m.avg_wage * 100 as gap_pct,
                m.population as male_pop,
                f.population as female_pop,
                m.n_obs as male_n,
                f.n_obs as female_n
            FROM wage_by_group m
            JOIN wage_by_group f ON m.{group_by} = f.{group_by}
            WHERE m.GENDER = 1 AND f.GENDER = 2
            ORDER BY gap_pct DESC
        """
        
        return self.query(sql)
    
    def get_decomposition_data(
        self,
        years: Optional[List[int]] = None,
        sample_frac: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Get data prepared for Oaxaca-Blinder decomposition.
        
        Returns individual-level data with:
        - Log hourly wage
        - Demographic controls
        - Survey weights
        
        Args:
            years: Filter by years
            sample_frac: Optional fraction to sample (for large datasets)
            
        Returns:
            DataFrame ready for decomposition analysis
        """
        self._initialize()
        
        year_filter = ""
        if years:
            years_str = ", ".join(str(y) for y in years)
            year_filter = f"AND SURVYEAR IN ({years_str})"
        
        sample_clause = ""
        if sample_frac:
            sample_clause = f"USING SAMPLE {sample_frac * 100} PERCENT (BERNOULLI)"
        
        sql = f"""
            SELECT 
                HRLYEARN as wage_dollars,
                LOG_HRLYEARN as log_wage,
                GENDER,
                CASE WHEN GENDER = 1 THEN 1 ELSE 0 END as is_male,
                SURVYEAR,
                PROV,
                EDUC,
                AGE_12,
                NOC_10,
                NAICS_21,
                UNION,
                FTPTMAIN,
                CASE WHEN FTPTMAIN = 1 THEN 1 ELSE 0 END as is_fulltime,
                TENURE,
                FINALWT
            FROM lfs
            {sample_clause}
            WHERE HRLYEARN > 0 AND FINALWT > 0 {year_filter}
        """
        
        return self.query(sql)
    
    def get_intersectional_gap(
        self,
        dimensions: List[str] = ['GENDER', 'PROV'],
        years: Optional[List[int]] = None,
        min_obs: int = 100
    ) -> pd.DataFrame:
        """
        Get weighted wage statistics for intersectional analysis.
        
        Args:
            dimensions: List of demographic dimensions to cross
            years: Filter by years
            min_obs: Minimum observations per cell
            
        Returns:
            DataFrame with mean wage by all dimension combinations
        """
        self._initialize()
        
        dims_str = ", ".join(dimensions)
        
        year_filter = ""
        if years:
            years_str = ", ".join(str(y) for y in years)
            year_filter = f"AND SURVYEAR IN ({years_str})"
        
        sql = f"""
            SELECT 
                {dims_str},
                SUM(HRLYEARN * FINALWT) / SUM(FINALWT) as avg_wage,
                SUM(FINALWT) as population,
                COUNT(*) as n_obs
            FROM lfs
            WHERE HRLYEARN > 0 AND FINALWT > 0 {year_filter}
            GROUP BY {dims_str}
            HAVING COUNT(*) >= {min_obs}
            ORDER BY avg_wage DESC
        """
        
        return self.query(sql)
    
    # =========================================================================
    # DATA MANAGEMENT
    # =========================================================================
    
    def create_parquet_from_csv(
        self,
        output_path: str = "data/parquet/lfs.parquet",
        partition_by: Optional[List[str]] = None,
        compression: str = "zstd"
    ):
        """
        Convert CSV source to Parquet (memory-efficient streaming).
        
        This uses DuckDB's streaming COPY which processes data in chunks,
        never loading the full dataset into memory.
        
        Args:
            output_path: Output Parquet file or directory
            partition_by: Columns to partition by (e.g., ['SURVYEAR'])
            compression: Compression codec (zstd recommended)
        """
        if not self.raw_csv_path.exists():
            raise FileNotFoundError(f"No CSV source at {self.raw_csv_path}")
        
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        csv_glob = str(self.raw_csv_path / "*.csv")
        
        if partition_by:
            # Partitioned output (Hive-style)
            partition_str = ", ".join(partition_by)
            self.conn.execute(f"""
                COPY (
                    SELECT * FROM read_csv_auto('{csv_glob}', 
                        header=true, 
                        ignore_errors=true,
                        parallel=true
                    )
                ) TO '{output_path}' 
                (FORMAT PARQUET, PARTITION_BY ({partition_str}), COMPRESSION {compression})
            """)
        else:
            # Single file output
            self.conn.execute(f"""
                COPY (
                    SELECT * FROM read_csv_auto('{csv_glob}', 
                        header=true, 
                        ignore_errors=true,
                        parallel=true
                    )
                ) TO '{output_path}' 
                (FORMAT PARQUET, COMPRESSION {compression})
            """)
        
        logger.info(f"Created Parquet at {output_path}")
        
        # Update source
        self.parquet_path = output
        self._initialized = False
        self._initialize()
    
    def close(self):
        """Close the DuckDB connection."""
        self.conn.close()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# =============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# =============================================================================

_default_store = None

def get_store() -> EquiPayDataStore:
    """Get or create the default data store singleton."""
    global _default_store
    if _default_store is None:
        _default_store = EquiPayDataStore()
    return _default_store


def query(sql: str) -> pd.DataFrame:
    """Execute a SQL query using the default store."""
    return get_store().query(sql)


def get_wages(**kwargs) -> pd.DataFrame:
    """Get wage data using the default store."""
    return get_store().get_wages(**kwargs)


def get_gender_gap(**kwargs) -> pd.DataFrame:
    """Get gender gap stats using the default store."""
    return get_store().get_gender_gap(**kwargs)
