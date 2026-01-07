"""
EquiPay Canada - LazyQuery Builder
==================================

Fluent SQL query builder with lazy evaluation.

Features:
- Chainable API that doesn't execute until needed
- SQL generation and optimization
- Aggregation functions (weighted means, percentiles)
- Query explanation and profiling

Usage:
    query = (
        store.select('PROV', 'GENDER', 'HRLYEARN')
        .where(year=2024, valid_wages=True)
        .aggregate(mean=Agg.weighted_mean('HRLYEARN', 'FINALWT'))
        .group_by('PROV', 'GENDER')
    )
    
    # See the SQL
    print(query.sql())
    
    # Execute
    df = query.to_pandas()
"""

import logging
from typing import Optional, List, Dict, Any, Union, Callable, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum
import hashlib

if TYPE_CHECKING:
    import duckdb
    import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# AGGREGATION FUNCTIONS
# =============================================================================

class Agg:
    """Aggregation function builders for SQL."""
    
    @staticmethod
    def count(column: str = '*') -> str:
        """COUNT aggregation."""
        return f"COUNT({column})"
    
    @staticmethod
    def sum(column: str) -> str:
        """SUM aggregation."""
        return f"SUM({column})"
    
    @staticmethod
    def avg(column: str) -> str:
        """Simple average (unweighted)."""
        return f"AVG({column})"
    
    # Alias for avg
    mean = avg
    
    @staticmethod
    def min(column: str) -> str:
        """MIN aggregation."""
        return f"MIN({column})"
    
    @staticmethod
    def max(column: str) -> str:
        """MAX aggregation."""
        return f"MAX({column})"
    
    @staticmethod
    def std(column: str) -> str:
        """Standard deviation."""
        return f"STDDEV({column})"
    
    @staticmethod
    def var(column: str) -> str:
        """Variance."""
        return f"VARIANCE({column})"
    
    @staticmethod
    def weighted_mean(value_col: str, weight_col: str = 'FINALWT') -> str:
        """
        Weighted mean using survey weights.
        Formula: SUM(value * weight) / SUM(weight)
        """
        return f"SUM({value_col} * {weight_col}) / NULLIF(SUM({weight_col}), 0)"
    
    @staticmethod
    def weighted_sum(value_col: str, weight_col: str = 'FINALWT') -> str:
        """Weighted sum."""
        return f"SUM({value_col} * {weight_col})"
    
    @staticmethod
    def weighted_std(value_col: str, weight_col: str = 'FINALWT') -> str:
        """
        Weighted standard deviation.
        Uses the formula for weighted sample variance.
        """
        mean_expr = f"(SUM({value_col} * {weight_col}) / NULLIF(SUM({weight_col}), 0))"
        return f"""SQRT(
            SUM({weight_col} * POWER({value_col} - {mean_expr}, 2)) / 
            NULLIF(SUM({weight_col}), 0)
        )"""
    
    @staticmethod
    def percentile(column: str, p: float) -> str:
        """
        Percentile (unweighted).
        
        Args:
            column: Column name
            p: Percentile (0 to 1)
        """
        return f"PERCENTILE_CONT({p}) WITHIN GROUP (ORDER BY {column})"
    
    @staticmethod
    def median(column: str) -> str:
        """Median (50th percentile)."""
        return Agg.percentile(column, 0.5)
    
    @staticmethod
    def iqr(column: str) -> str:
        """Interquartile range (P75 - P25)."""
        return f"""(
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {column}) -
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {column})
        )"""
    
    @staticmethod
    def gender_gap(value_col: str, weight_col: str = 'FINALWT', 
                   gender_col: str = 'GENDER') -> str:
        """
        Gender wage gap as percentage.
        Formula: (male_wage - female_wage) / male_wage * 100
        """
        male_mean = f"""SUM(CASE WHEN {gender_col} = 1 THEN {value_col} * {weight_col} END) / 
                       NULLIF(SUM(CASE WHEN {gender_col} = 1 THEN {weight_col} END), 0)"""
        female_mean = f"""SUM(CASE WHEN {gender_col} = 2 THEN {value_col} * {weight_col} END) / 
                         NULLIF(SUM(CASE WHEN {gender_col} = 2 THEN {weight_col} END), 0)"""
        return f"(({male_mean}) - ({female_mean})) / NULLIF({male_mean}, 0) * 100"
    
    @staticmethod
    def male_mean(value_col: str, weight_col: str = 'FINALWT',
                  gender_col: str = 'GENDER') -> str:
        """Weighted mean for males only."""
        return f"""SUM(CASE WHEN {gender_col} = 1 THEN {value_col} * {weight_col} END) / 
                   NULLIF(SUM(CASE WHEN {gender_col} = 1 THEN {weight_col} END), 0)"""
    
    @staticmethod
    def female_mean(value_col: str, weight_col: str = 'FINALWT',
                    gender_col: str = 'GENDER') -> str:
        """Weighted mean for females only."""
        return f"""SUM(CASE WHEN {gender_col} = 2 THEN {value_col} * {weight_col} END) / 
                   NULLIF(SUM(CASE WHEN {gender_col} = 2 THEN {weight_col} END), 0)"""
    
    @staticmethod
    def cv(column: str) -> str:
        """Coefficient of variation (CV = std/mean * 100)."""
        return f"STDDEV({column}) / NULLIF(AVG({column}), 0) * 100"
    
    @staticmethod
    def skewness(column: str) -> str:
        """Skewness."""
        return f"SKEWNESS({column})"
    
    @staticmethod
    def kurtosis(column: str) -> str:
        """Kurtosis."""
        return f"KURTOSIS({column})"
    
    @staticmethod
    def corr(col1: str, col2: str) -> str:
        """Correlation coefficient."""
        return f"CORR({col1}, {col2})"
    
    @staticmethod
    def covar(col1: str, col2: str) -> str:
        """Covariance."""
        return f"COVAR_SAMP({col1}, {col2})"


class Func:
    """SQL function builders."""
    
    @staticmethod
    def log(column: str) -> str:
        """Natural logarithm."""
        return f"LN({column})"
    
    @staticmethod
    def log_safe(column: str, min_val: float = 0.01) -> str:
        """Safe log that handles zeros."""
        return f"LN(GREATEST({column}, {min_val}))"
    
    @staticmethod
    def exp(column: str) -> str:
        """Exponential."""
        return f"EXP({column})"
    
    @staticmethod
    def sqrt(column: str) -> str:
        """Square root."""
        return f"SQRT({column})"
    
    @staticmethod
    def power(column: str, n: float) -> str:
        """Power function."""
        return f"POWER({column}, {n})"
    
    @staticmethod
    def abs(column: str) -> str:
        """Absolute value."""
        return f"ABS({column})"
    
    @staticmethod
    def round(column: str, decimals: int = 2) -> str:
        """Round to decimals."""
        return f"ROUND({column}, {decimals})"
    
    @staticmethod
    def coalesce(*columns) -> str:
        """Return first non-null value."""
        return f"COALESCE({', '.join(columns)})"
    
    @staticmethod
    def case_when(conditions: Dict[str, Any], else_val: Any = None) -> str:
        """
        Build CASE WHEN expression.
        
        Args:
            conditions: Dict of {condition: result}
            else_val: Default value if no conditions match
        """
        cases = " ".join([f"WHEN {cond} THEN {val}" for cond, val in conditions.items()])
        else_clause = f" ELSE {else_val}" if else_val is not None else ""
        return f"CASE {cases}{else_clause} END"
    
    @staticmethod
    def is_female(gender_col: str = 'GENDER') -> str:
        """Binary female indicator."""
        return f"CASE WHEN {gender_col} = 2 THEN 1 ELSE 0 END"


# =============================================================================
# LAZY QUERY BUILDER
# =============================================================================

@dataclass
class QueryState:
    """Internal state of a query being built."""
    
    # SELECT clause
    select_cols: List[str] = field(default_factory=list)
    select_all: bool = False
    
    # Aggregations (alias -> expression)
    aggregations: Dict[str, str] = field(default_factory=dict)
    
    # FROM clause
    from_table: str = 'lfs'
    from_subquery: Optional[str] = None
    
    # WHERE clause conditions
    where_conditions: List[str] = field(default_factory=list)
    
    # GROUP BY
    group_by_cols: List[str] = field(default_factory=list)
    
    # HAVING
    having_conditions: List[str] = field(default_factory=list)
    
    # ORDER BY
    order_by_cols: List[tuple] = field(default_factory=list)  # (col, desc)
    
    # LIMIT / OFFSET
    limit_val: Optional[int] = None
    offset_val: Optional[int] = None
    
    # Sampling
    sample_frac: Optional[float] = None
    sample_method: str = 'BERNOULLI'
    
    # CTEs
    ctes: Dict[str, str] = field(default_factory=dict)


class LazyQuery:
    """
    Fluent SQL query builder with lazy evaluation.
    
    Queries are not executed until to_pandas(), execute(), or similar
    terminal methods are called.
    
    Usage:
        query = (
            LazyQuery(connection)
            .select('PROV', 'GENDER')
            .where(year=2024)
            .aggregate(mean_wage=Agg.weighted_mean('HRLYEARN'))
            .group_by('PROV', 'GENDER')
            .order_by('mean_wage', descending=True)
            .limit(100)
        )
        
        df = query.to_pandas()
    """
    
    def __init__(self, connection: 'duckdb.DuckDBPyConnection', 
                 state_or_table: Optional[Union[QueryState, str]] = None):
        """
        Initialize query builder.
        
        Args:
            connection: DuckDB connection
            state_or_table: Optional initial state (for cloning) or table name
        """
        self._conn = connection
        
        # Handle either QueryState or table name string
        if state_or_table is None:
            self._state = QueryState()
        elif isinstance(state_or_table, str):
            self._state = QueryState(from_table=state_or_table)
        else:
            self._state = state_or_table
            
        self._last_sql: Optional[str] = None
        self._cached_result = None
    
    def _clone(self) -> 'LazyQuery':
        """Create a copy of this query with copied state."""
        import copy
        new_state = copy.deepcopy(self._state)
        return LazyQuery(self._conn, new_state)
    
    # =========================================================================
    # SELECT METHODS
    # =========================================================================
    
    def select(self, *columns: str) -> 'LazyQuery':
        """
        Select specific columns.
        
        Args:
            columns: Column names to select
        """
        q = self._clone()
        q._state.select_cols = list(columns)
        q._state.select_all = False
        return q
    
    def select_all(self) -> 'LazyQuery':
        """Select all columns (SELECT *)."""
        q = self._clone()
        q._state.select_all = True
        q._state.select_cols = []
        return q
    
    def add_column(self, column: str, alias: Optional[str] = None) -> 'LazyQuery':
        """Add a column to selection."""
        q = self._clone()
        col_expr = f"{column} AS {alias}" if alias else column
        q._state.select_cols.append(col_expr)
        return q
    
    # =========================================================================
    # FROM METHODS
    # =========================================================================
    
    def from_table(self, table: str) -> 'LazyQuery':
        """Set the source table."""
        q = self._clone()
        q._state.from_table = table
        q._state.from_subquery = None
        return q
    
    def from_subquery(self, subquery: str, alias: str = 'sub') -> 'LazyQuery':
        """Use a subquery as source."""
        q = self._clone()
        q._state.from_subquery = f"({subquery}) AS {alias}"
        q._state.from_table = None
        return q
    
    # =========================================================================
    # WHERE METHODS
    # =========================================================================
    
    def where(self, *raw_conditions: str, **kwargs) -> 'LazyQuery':
        """
        Add WHERE conditions.
        
        Supports both raw SQL and keyword arguments:
            .where("HRLYEARN > 0")
            .where(year=2024, province=35)
            .where(valid_wages=True)  # Special: HRLYEARN > 0
        """
        q = self._clone()
        
        # Add raw conditions
        for cond in raw_conditions:
            q._state.where_conditions.append(cond)
        
        # Process keyword arguments
        for key, value in kwargs.items():
            cond = self._kwarg_to_condition(key, value)
            if cond:
                q._state.where_conditions.append(cond)
        
        return q
    
    def where_in(self, column: str, values: List[Any]) -> 'LazyQuery':
        """Add IN condition."""
        q = self._clone()
        values_str = ", ".join(repr(v) if isinstance(v, str) else str(v) for v in values)
        q._state.where_conditions.append(f"{column} IN ({values_str})")
        return q
    
    def where_between(self, column: str, low: Any, high: Any) -> 'LazyQuery':
        """Add BETWEEN condition."""
        q = self._clone()
        q._state.where_conditions.append(f"{column} BETWEEN {low} AND {high}")
        return q
    
    def where_not_null(self, column: str) -> 'LazyQuery':
        """Add IS NOT NULL condition."""
        q = self._clone()
        q._state.where_conditions.append(f"{column} IS NOT NULL")
        return q
    
    def _kwarg_to_condition(self, key: str, value: Any) -> Optional[str]:
        """Convert keyword argument to SQL condition."""
        # Special cases
        if key == 'valid_wages' and value:
            return "HRLYEARN > 0 AND HRLYEARN < 500"
        elif key == 'year':
            if isinstance(value, (list, tuple)):
                return f"SURVYEAR IN ({', '.join(map(str, value))})"
            return f"SURVYEAR = {value}"
        elif key == 'years':
            if isinstance(value, range):
                return f"SURVYEAR BETWEEN {value.start} AND {value.stop - 1}"
            return f"SURVYEAR IN ({', '.join(map(str, value))})"
        elif key == 'province':
            return f"PROV = {value}"
        elif key == 'provinces':
            return f"PROV IN ({', '.join(map(str, value))})"
        elif key == 'gender':
            return f"GENDER = {value}"
        elif key == 'wage_range':
            if isinstance(value, (list, tuple)) and len(value) == 2:
                return f"HRLYEARN BETWEEN {value[0]} AND {value[1]}"
        elif key == 'min_wage':
            return f"HRLYEARN >= {value}"
        elif key == 'max_wage':
            return f"HRLYEARN <= {value}"
        else:
            # Generic equality
            if isinstance(value, str):
                return f"{key.upper()} = '{value}'"
            elif value is None:
                return f"{key.upper()} IS NULL"
            else:
                return f"{key.upper()} = {value}"
    
    # =========================================================================
    # AGGREGATION METHODS
    # =========================================================================
    
    def aggregate(self, **kwargs) -> 'LazyQuery':
        """
        Add aggregations with aliases.
        
        Usage:
            .aggregate(
                mean_wage=Agg.weighted_mean('HRLYEARN'),
                count=Agg.count(),
                gap=Agg.gender_gap('HRLYEARN')
            )
        """
        q = self._clone()
        for alias, expr in kwargs.items():
            q._state.aggregations[alias] = expr
        return q
    
    def agg(self, **kwargs) -> 'LazyQuery':
        """Alias for aggregate()."""
        return self.aggregate(**kwargs)
    
    # =========================================================================
    # GROUP BY / HAVING
    # =========================================================================
    
    def group_by(self, *columns: str) -> 'LazyQuery':
        """Add GROUP BY columns."""
        q = self._clone()
        q._state.group_by_cols = list(columns)
        return q
    
    def having(self, *conditions: str, min_obs: Optional[int] = None) -> 'LazyQuery':
        """
        Add HAVING conditions.
        
        Args:
            conditions: Raw SQL conditions
            min_obs: Minimum observations filter (COUNT(*) >= min_obs)
        """
        q = self._clone()
        for cond in conditions:
            q._state.having_conditions.append(cond)
        if min_obs is not None:
            q._state.having_conditions.append(f"COUNT(*) >= {min_obs}")
        return q
    
    # =========================================================================
    # ORDER BY / LIMIT
    # =========================================================================
    
    def order_by(self, *columns: str, descending: bool = False) -> 'LazyQuery':
        """Add ORDER BY columns."""
        q = self._clone()
        for col in columns:
            q._state.order_by_cols.append((col, descending))
        return q
    
    def limit(self, n: int) -> 'LazyQuery':
        """Limit results."""
        q = self._clone()
        q._state.limit_val = n
        return q
    
    def offset(self, n: int) -> 'LazyQuery':
        """Offset results."""
        q = self._clone()
        q._state.offset_val = n
        return q
    
    # =========================================================================
    # SAMPLING
    # =========================================================================
    
    def sample(self, frac: float, method: str = 'BERNOULLI') -> 'LazyQuery':
        """
        Add random sampling.
        
        Args:
            frac: Fraction to sample (0 to 1)
            method: 'BERNOULLI' or 'SYSTEM'
        """
        q = self._clone()
        q._state.sample_frac = frac
        q._state.sample_method = method
        return q
    
    # =========================================================================
    # CTEs (WITH clauses)
    # =========================================================================
    
    def with_cte(self, name: str, sql: str) -> 'LazyQuery':
        """Add a Common Table Expression (CTE)."""
        q = self._clone()
        q._state.ctes[name] = sql
        return q
    
    # =========================================================================
    # SQL GENERATION
    # =========================================================================
    
    def sql(self) -> str:
        """Generate the SQL query string."""
        parts = []
        
        # CTEs
        if self._state.ctes:
            cte_parts = [f"{name} AS ({sql})" for name, sql in self._state.ctes.items()]
            parts.append(f"WITH {', '.join(cte_parts)}")
        
        # SELECT
        select_items = []
        
        # Add group by columns first (if aggregating)
        if self._state.aggregations and self._state.group_by_cols:
            select_items.extend(self._state.group_by_cols)
        
        # Add explicit select columns
        if self._state.select_all:
            select_items.append('*')
        elif self._state.select_cols:
            select_items.extend(self._state.select_cols)
        
        # Add aggregations
        for alias, expr in self._state.aggregations.items():
            select_items.append(f"{expr} AS {alias}")
        
        # Default to * if nothing specified
        if not select_items:
            select_items = ['*']
        
        parts.append(f"SELECT {', '.join(select_items)}")
        
        # FROM
        if self._state.from_subquery:
            parts.append(f"FROM {self._state.from_subquery}")
        else:
            parts.append(f"FROM {self._state.from_table}")
        
        # SAMPLE
        if self._state.sample_frac:
            pct = self._state.sample_frac * 100
            parts.append(f"USING SAMPLE {pct} PERCENT ({self._state.sample_method})")
        
        # WHERE
        if self._state.where_conditions:
            parts.append(f"WHERE {' AND '.join(self._state.where_conditions)}")
        
        # GROUP BY
        if self._state.group_by_cols:
            parts.append(f"GROUP BY {', '.join(self._state.group_by_cols)}")
        
        # HAVING
        if self._state.having_conditions:
            parts.append(f"HAVING {' AND '.join(self._state.having_conditions)}")
        
        # ORDER BY
        if self._state.order_by_cols:
            order_parts = []
            for col, desc in self._state.order_by_cols:
                order_parts.append(f"{col} {'DESC' if desc else 'ASC'}")
            parts.append(f"ORDER BY {', '.join(order_parts)}")
        
        # LIMIT / OFFSET
        if self._state.limit_val is not None:
            parts.append(f"LIMIT {self._state.limit_val}")
        if self._state.offset_val is not None:
            parts.append(f"OFFSET {self._state.offset_val}")
        
        sql = "\n".join(parts)
        self._last_sql = sql
        return sql
    
    def fingerprint(self) -> str:
        """Generate a hash fingerprint of this query for caching."""
        sql = self.sql()
        return hashlib.md5(sql.encode()).hexdigest()
    
    def explain(self) -> str:
        """Get query execution plan."""
        sql = self.sql()
        result = self._conn.execute(f"EXPLAIN {sql}").fetchall()
        return "\n".join([row[0] for row in result])
    
    # =========================================================================
    # EXECUTION METHODS (Terminal)
    # =========================================================================
    
    def execute(self):
        """Execute query and return DuckDB result."""
        sql = self.sql()
        logger.debug(f"Executing: {sql[:200]}...")
        return self._conn.execute(sql)
    
    def to_pandas(self) -> 'pd.DataFrame':
        """Execute query and return pandas DataFrame."""
        return self.execute().fetchdf()
    
    def to_arrow(self):
        """Execute query and return PyArrow Table."""
        return self.execute().fetch_arrow_table()
    
    def to_dict(self) -> List[Dict[str, Any]]:
        """Execute query and return list of dicts."""
        df = self.to_pandas()
        return df.to_dict('records')
    
    def to_list(self) -> List[tuple]:
        """Execute query and return list of tuples."""
        return self.execute().fetchall()
    
    def scalar(self) -> Any:
        """Execute query and return single scalar value."""
        result = self.execute().fetchone()
        return result[0] if result else None
    
    def count(self) -> int:
        """Get count of matching rows."""
        q = self._clone()
        q._state.select_cols = []
        q._state.select_all = False
        q._state.aggregations = {'n': 'COUNT(*)'}
        q._state.group_by_cols = []
        q._state.order_by_cols = []
        q._state.limit_val = None
        return q.scalar()
    
    def exists(self) -> bool:
        """Check if any rows match."""
        return self.limit(1).count() > 0
    
    def first(self) -> Optional[Dict[str, Any]]:
        """Get first matching row as dict."""
        result = self.limit(1).to_dict()
        return result[0] if result else None
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def __repr__(self):
        sql = self.sql()
        if len(sql) > 100:
            sql = sql[:100] + "..."
        return f"LazyQuery({sql})"
    
    def __str__(self):
        return self.sql()
    
    def copy(self) -> 'LazyQuery':
        """Create a copy of this query."""
        return self._clone()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def select(*columns: str) -> 'QueryBuilder':
    """Start building a query with SELECT."""
    # This is a factory that returns a partial query
    # Actual connection is bound later
    raise NotImplementedError("Use store.select() instead")
