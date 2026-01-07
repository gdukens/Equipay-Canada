"""
EquiPay Canada - Feature Store
==============================

Feature registration, computation, and versioning for ML pipelines.

Features:
- Register Python and SQL-computed features
- Feature versioning and lineage
- Dependency resolution
- Leakage prevention integration
- Caching of computed features

Usage:
    # Register a feature
    @store.features.register('occupation_female_ratio')
    def occupation_female_ratio(df):
        return df.groupby('NOC_10')['IS_FEMALE'].transform('mean')
    
    # Register SQL feature
    store.features.register_sql(
        'provincial_premium',
        "SELECT PROV, AVG(HRLYEARN) / (SELECT AVG(HRLYEARN) FROM lfs) as premium"
    )
    
    # Get features for training
    X, y, w = store.features.get(
        ['occupation_female_ratio', 'EDUC', 'NOC_10'],
        target='LOG_REAL_HRLYEARN'
    )
"""

import logging
import hashlib
from typing import Optional, List, Dict, Any, Callable, Set, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps

if TYPE_CHECKING:
    import pandas as pd
    import numpy as np
    import duckdb

logger = logging.getLogger(__name__)


@dataclass
class FeatureDefinition:
    """Definition of a registered feature."""
    name: str
    version: str
    description: str
    
    # Computation
    compute_fn: Optional[Callable] = None  # Python function
    sql_expr: Optional[str] = None          # SQL expression
    
    # Dependencies
    dependencies: List[str] = field(default_factory=list)
    required_columns: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    author: str = "EquiPay Canada"
    
    # Validation
    is_leakage_safe: bool = True
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    
    def __hash__(self):
        return hash((self.name, self.version))
    
    def fingerprint(self) -> str:
        """Generate unique fingerprint for this feature version."""
        content = f"{self.name}:{self.version}:{self.sql_expr or ''}"
        return hashlib.md5(content.encode()).hexdigest()[:12]


@dataclass
class FeatureSet:
    """A collection of features for training."""
    name: str
    version: str
    features: List[str]
    target: Optional[str] = None
    weight: str = 'FINALWT'
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)


class FeatureStore:
    """
    Central repository for feature definitions and computation.
    
    Provides:
    - Feature registration (Python or SQL)
    - Dependency resolution
    - Version tracking
    - Leakage prevention
    - Feature computation and caching
    """
    
    # Features that indicate potential leakage
    LEAKAGE_KEYWORDS = {
        'wage', 'earn', 'income', 'salary', 'pay', 'compensation',
        'hrlyearn', 'real_hrlyearn', 'log_hrlyearn'
    }
    
    def __init__(self, connection: Optional['duckdb.DuckDBPyConnection'] = None):
        """
        Initialize feature store.
        
        Args:
            connection: DuckDB connection for SQL features
        """
        self.connection = connection
        
        # Feature registry
        self._features: Dict[str, FeatureDefinition] = {}
        
        # Feature sets
        self._feature_sets: Dict[str, FeatureSet] = {}
        
        # Computed feature cache
        self._cache: Dict[str, Any] = {}
        
        # Register built-in features
        self._register_builtins()
    
    def _register_builtins(self):
        """Register built-in features."""
        
        # IS_FEMALE indicator
        self.register_sql(
            name='IS_FEMALE',
            sql_expr="CASE WHEN GENDER = 2 THEN 1 ELSE 0 END",
            description="Binary female indicator (1=Female, 0=Male)",
            required_columns=['GENDER']
        )
        
        # IS_FULLTIME indicator
        self.register_sql(
            name='IS_FULLTIME',
            sql_expr="CASE WHEN FTPTMAIN = 1 THEN 1 ELSE 0 END",
            description="Binary full-time indicator",
            required_columns=['FTPTMAIN']
        )
        
        # IS_UNION indicator
        self.register_sql(
            name='IS_UNION',
            sql_expr='CASE WHEN "UNION" IN (1, 2) THEN 1 ELSE 0 END',
            description="Union member or covered by collective agreement",
            required_columns=['UNION']
        )
        
        # Experience proxy (Mincer-style)
        self.register_sql(
            name='EXPERIENCE_PROXY',
            sql_expr="""
                GREATEST(0, 
                    CASE AGE_12
                        WHEN 1 THEN 17 WHEN 2 THEN 22 WHEN 3 THEN 27
                        WHEN 4 THEN 32 WHEN 5 THEN 37 WHEN 6 THEN 42
                        WHEN 7 THEN 47 WHEN 8 THEN 52 WHEN 9 THEN 57
                        WHEN 10 THEN 62 WHEN 11 THEN 67 WHEN 12 THEN 72
                        ELSE 35
                    END
                    - CASE EDUC
                        WHEN 0 THEN 8 WHEN 1 THEN 10 WHEN 2 THEN 12
                        WHEN 3 THEN 14 WHEN 4 THEN 16 WHEN 5 THEN 18
                        WHEN 6 THEN 20 ELSE 12
                    END
                    - 6
                )
            """,
            description="Mincer experience proxy: Age - Years of Education - 6",
            required_columns=['AGE_12', 'EDUC']
        )
        
        # Experience squared
        self.register_sql(
            name='EXPERIENCE_SQ',
            sql_expr="POWER(EXPERIENCE_PROXY, 2)",
            description="Experience squared for diminishing returns",
            dependencies=['EXPERIENCE_PROXY']
        )
        
        logger.debug(f"Registered {len(self._features)} built-in features")
    
    def register(
        self,
        name: str,
        version: str = "1.0",
        description: str = "",
        dependencies: List[str] = None,
        required_columns: List[str] = None
    ) -> Callable:
        """
        Decorator to register a Python feature function.
        
        Usage:
            @store.features.register('custom_feature')
            def custom_feature(df):
                return df['A'] * df['B']
        """
        def decorator(fn: Callable) -> Callable:
            feature = FeatureDefinition(
                name=name,
                version=version,
                description=description or fn.__doc__ or "",
                compute_fn=fn,
                dependencies=dependencies or [],
                required_columns=required_columns or [],
                is_leakage_safe=self._check_leakage_safe(name)
            )
            self._features[name] = feature
            logger.debug(f"Registered Python feature: {name} v{version}")
            
            @wraps(fn)
            def wrapper(*args, **kwargs):
                return fn(*args, **kwargs)
            return wrapper
        
        return decorator
    
    def register_sql(
        self,
        name: str,
        sql_expr: str,
        version: str = "1.0",
        description: str = "",
        dependencies: List[str] = None,
        required_columns: List[str] = None
    ):
        """
        Register a SQL-computed feature.
        
        Args:
            name: Feature name
            sql_expr: SQL expression for the feature
            version: Feature version
            description: Human-readable description
            dependencies: Other features this depends on
            required_columns: Raw columns required
        """
        feature = FeatureDefinition(
            name=name,
            version=version,
            description=description,
            sql_expr=sql_expr,
            dependencies=dependencies or [],
            required_columns=required_columns or [],
            is_leakage_safe=self._check_leakage_safe(name, sql_expr)
        )
        self._features[name] = feature
        logger.debug(f"Registered SQL feature: {name} v{version}")
    
    def _check_leakage_safe(self, name: str, sql_expr: str = "") -> bool:
        """Check if feature might leak target information."""
        text = f"{name} {sql_expr}".lower()
        for keyword in self.LEAKAGE_KEYWORDS:
            if keyword in text:
                logger.warning(f"Potential leakage in feature '{name}': contains '{keyword}'")
                return False
        return True
    
    def _resolve_dependencies(self, feature_names: List[str]) -> List[str]:
        """Resolve feature dependencies in topological order."""
        resolved = []
        seen = set()
        
        def visit(name: str):
            if name in seen:
                return
            seen.add(name)
            
            if name in self._features:
                feature = self._features[name]
                for dep in feature.dependencies:
                    visit(dep)
            
            resolved.append(name)
        
        for name in feature_names:
            visit(name)
        
        return resolved
    
    def get_sql_for_features(
        self,
        feature_names: List[str],
        target: Optional[str] = None,
        weight: str = 'FINALWT',
        base_table: str = 'lfs'
    ) -> str:
        """
        Generate SQL that computes all requested features.
        
        Returns a SELECT statement with all features computed.
        """
        # Resolve dependencies
        all_features = self._resolve_dependencies(feature_names)
        
        # Separate SQL and raw features
        sql_features = []
        raw_columns = set()
        
        for name in all_features:
            if name in self._features:
                feature = self._features[name]
                if feature.sql_expr:
                    sql_features.append(f"({feature.sql_expr}) AS {name}")
                raw_columns.update(feature.required_columns)
            else:
                # Assume it's a raw column
                raw_columns.add(name)
        
        # Add target and weight
        if target:
            raw_columns.add(target)
        raw_columns.add(weight)
        
        # Build query with CTEs for complex dependencies
        if any(f.dependencies for f in self._features.values() if f.name in all_features):
            # Need CTEs for dependency resolution
            return self._build_cte_query(all_features, target, weight, base_table)
        else:
            # Simple query
            select_parts = list(raw_columns) + sql_features
            return f"SELECT {', '.join(select_parts)} FROM {base_table}"
    
    def _build_cte_query(
        self,
        features: List[str],
        target: Optional[str],
        weight: str,
        base_table: str
    ) -> str:
        """Build query with CTEs for dependent features."""
        ctes = []
        final_cols = []
        
        # Base CTE with raw columns
        raw_cols = set()
        for name in features:
            if name in self._features:
                raw_cols.update(self._features[name].required_columns)
            else:
                raw_cols.add(name)
        
        if target:
            raw_cols.add(target)
        raw_cols.add(weight)
        
        base_cte = f"base AS (SELECT {', '.join(raw_cols)} FROM {base_table})"
        ctes.append(base_cte)
        
        # Add feature CTEs in dependency order
        prev_cte = 'base'
        for name in features:
            if name in self._features and self._features[name].sql_expr:
                feature = self._features[name]
                cte_name = f"with_{name.lower()}"
                
                # Select all from previous + new feature
                cte_sql = f"""
                    {cte_name} AS (
                        SELECT *, ({feature.sql_expr}) AS {name}
                        FROM {prev_cte}
                    )
                """
                ctes.append(cte_sql)
                prev_cte = cte_name
            
            final_cols.append(name)
        
        if target:
            final_cols.append(target)
        final_cols.append(weight)
        
        return f"""
            WITH {', '.join(ctes)}
            SELECT {', '.join(final_cols)}
            FROM {prev_cte}
        """
    
    def compute(
        self,
        feature_names: List[str],
        target: Optional[str] = None,
        weight: str = 'FINALWT',
        where: str = "HRLYEARN > 0",
        sample_frac: Optional[float] = None
    ) -> 'pd.DataFrame':
        """
        Compute features and return as DataFrame.
        
        Args:
            feature_names: List of feature names
            target: Target column (optional)
            weight: Weight column
            where: WHERE clause
            sample_frac: Optional sampling fraction
            
        Returns:
            DataFrame with computed features
        """
        if self.connection is None:
            raise ValueError("No database connection available")
        
        base_sql = self.get_sql_for_features(feature_names, target, weight)
        
        # Add WHERE clause
        sql = f"SELECT * FROM ({base_sql}) AS features WHERE {where}"
        
        # Add sampling
        if sample_frac:
            sql = f"{sql} USING SAMPLE {sample_frac * 100} PERCENT (BERNOULLI)"
        
        return self.connection.execute(sql).fetchdf()
    
    def get(
        self,
        feature_names: List[str],
        target: str = 'LOG_REAL_HRLYEARN',
        weight: str = 'FINALWT',
        sample_frac: Optional[float] = None,
        return_arrays: bool = True
    ):
        """
        Get features for ML training.
        
        Args:
            feature_names: List of feature names
            target: Target column
            weight: Weight column
            sample_frac: Optional sampling fraction
            return_arrays: If True, return (X, y, w) numpy arrays
            
        Returns:
            If return_arrays: (X, y, weights) as numpy arrays
            Else: DataFrame with all columns
        """
        df = self.compute(
            feature_names=feature_names,
            target=target,
            weight=weight,
            sample_frac=sample_frac
        )
        
        if return_arrays:
            import numpy as np
            X = df[feature_names].values
            y = df[target].values if target in df.columns else None
            w = df[weight].values if weight in df.columns else None
            return X, y, w
        
        return df
    
    def describe(self, name: str) -> Dict[str, Any]:
        """Get description of a feature."""
        if name not in self._features:
            return {'error': f"Feature '{name}' not found"}
        
        feature = self._features[name]
        return {
            'name': feature.name,
            'version': feature.version,
            'description': feature.description,
            'type': 'SQL' if feature.sql_expr else 'Python',
            'dependencies': feature.dependencies,
            'required_columns': feature.required_columns,
            'is_leakage_safe': feature.is_leakage_safe,
            'fingerprint': feature.fingerprint()
        }
    
    def list(self, include_builtins: bool = True) -> List[str]:
        """List all registered features."""
        return list(self._features.keys())
    
    def lineage(self, name: str) -> Dict[str, Any]:
        """Get dependency lineage for a feature."""
        if name not in self._features:
            return {'error': f"Feature '{name}' not found"}
        
        def get_deps(n: str, depth: int = 0) -> Dict:
            if n not in self._features:
                return {'name': n, 'type': 'column'}
            
            feature = self._features[n]
            return {
                'name': n,
                'version': feature.version,
                'type': 'SQL' if feature.sql_expr else 'Python',
                'dependencies': [
                    get_deps(dep, depth + 1) 
                    for dep in feature.dependencies
                ]
            }
        
        return get_deps(name)
    
    def validate(self, feature_names: List[str]) -> Dict[str, Any]:
        """
        Validate feature set for potential issues.
        
        Returns validation report with warnings.
        """
        issues = []
        warnings = []
        
        for name in feature_names:
            if name not in self._features:
                # Check if it's a valid raw column
                issues.append(f"Unknown feature: {name}")
                continue
            
            feature = self._features[name]
            
            if not feature.is_leakage_safe:
                issues.append(f"Potential leakage: {name}")
            
            for dep in feature.dependencies:
                if dep not in self._features and dep not in feature_names:
                    warnings.append(f"Missing dependency for {name}: {dep}")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'features': len(feature_names),
            'resolved': len(self._resolve_dependencies(feature_names))
        }
    
    # =========================================================================
    # Feature Sets (for reproducibility)
    # =========================================================================
    
    def create_feature_set(
        self,
        name: str,
        features: List[str],
        target: Optional[str] = None,
        version: str = "1.0",
        description: str = ""
    ) -> FeatureSet:
        """
        Create a named feature set for reproducibility.
        
        Feature sets capture a specific combination of features
        for training, enabling versioned ML pipelines.
        """
        feature_set = FeatureSet(
            name=name,
            version=version,
            features=features,
            target=target,
            description=description
        )
        self._feature_sets[name] = feature_set
        return feature_set
    
    def get_feature_set(self, name: str) -> Optional[FeatureSet]:
        """Get a named feature set."""
        return self._feature_sets.get(name)
    
    def __repr__(self):
        return f"FeatureStore({len(self._features)} features, {len(self._feature_sets)} sets)"
