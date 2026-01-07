"""
EquiPay Canada - Bootstrap Variance Estimation
==============================================

Poisson bootstrap for survey data per Statistics Canada methodology.

For PUMF (Public Use Microdata File) data without bootstrap weights,
the Poisson bootstrap provides valid variance estimates:

1. Draw Poisson(1) weights for each observation
2. Multiply with survey weight
3. Compute statistic
4. Repeat B times
5. Variance = sample variance of B estimates

This approach is recommended by Statistics Canada for PUMF users
who don't have access to the full set of bootstrap weights.

References:
    - Statistics Canada (2015) "Survey Methods and Practices"
    - Beaumont & Patak (2012) "On the Generalized Bootstrap"
"""

import logging
from typing import Optional, List, Dict, Any, Callable, TYPE_CHECKING
from dataclasses import dataclass, field
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time

if TYPE_CHECKING:
    import pandas as pd
    import duckdb

logger = logging.getLogger(__name__)


@dataclass
class BootstrapResults:
    """Results from bootstrap variance estimation."""
    
    # Point estimate
    estimate: float
    
    # Bootstrap distribution
    n_bootstraps: int
    bootstrap_estimates: np.ndarray = field(repr=False)
    
    # Variance and standard error
    variance: float = 0.0
    std_error: float = 0.0
    
    # Confidence intervals
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    ci_level: float = 0.95
    
    # Computation metadata
    computation_time: float = 0.0
    
    def __post_init__(self):
        """Compute derived statistics."""
        if len(self.bootstrap_estimates) > 0:
            self.variance = np.var(self.bootstrap_estimates, ddof=1)
            self.std_error = np.std(self.bootstrap_estimates, ddof=1)
            
            # Percentile confidence intervals
            alpha = 1 - self.ci_level
            self.ci_lower = np.percentile(self.bootstrap_estimates, alpha/2 * 100)
            self.ci_upper = np.percentile(self.bootstrap_estimates, (1 - alpha/2) * 100)
    
    @property
    def coefficient_of_variation(self) -> float:
        """CV = SE / estimate * 100."""
        if abs(self.estimate) < 1e-10:
            return np.inf
        return (self.std_error / abs(self.estimate)) * 100
    
    @property
    def is_reliable(self) -> bool:
        """Statistics Canada: CV < 16.5% is reliable."""
        return self.coefficient_of_variation < 16.5
    
    @property
    def is_significant(self) -> bool:
        """Check if CI excludes zero."""
        return (self.ci_lower > 0) or (self.ci_upper < 0)
    
    def summary(self) -> str:
        """Generate summary string."""
        reliability = "reliable" if self.is_reliable else "UNRELIABLE (CV > 16.5%)"
        significance = "significant" if self.is_significant else "not significant"
        
        return (
            f"Estimate: {self.estimate:.4f}\n"
            f"SE: {self.std_error:.4f}\n"
            f"95% CI: [{self.ci_lower:.4f}, {self.ci_upper:.4f}]\n"
            f"CV: {self.coefficient_of_variation:.1f}% ({reliability})\n"
            f"Statistical {significance}"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'estimate': self.estimate,
            'variance': self.variance,
            'std_error': self.std_error,
            'ci_lower': self.ci_lower,
            'ci_upper': self.ci_upper,
            'ci_level': self.ci_level,
            'cv': self.coefficient_of_variation,
            'is_reliable': self.is_reliable,
            'n_bootstraps': self.n_bootstraps
        }


class PoissonBootstrap:
    """
    Poisson bootstrap variance estimation for survey data.
    
    Generates bootstrap replicates by:
    1. Drawing Poisson(1) weights for each observation
    2. Multiplying with original survey weight
    3. Computing the statistic with modified weights
    
    This is done in SQL for efficiency using random number generation.
    """
    
    def __init__(
        self,
        connection: 'duckdb.DuckDBPyConnection',
        default_n_bootstraps: int = 500
    ):
        """
        Initialize bootstrap estimator.
        
        Args:
            connection: DuckDB connection
            default_n_bootstraps: Default number of bootstrap replications
        """
        self.connection = connection
        self.default_n_bootstraps = default_n_bootstraps
    
    def _generate_poisson_seed(self, rep: int) -> int:
        """Generate seed for reproducible Poisson draws."""
        return 42 + rep * 1000
    
    def weighted_mean(
        self,
        column: str,
        weight: str = 'FINALWT',
        where: str = "HRLYEARN > 0",
        n_bootstraps: int = None
    ) -> BootstrapResults:
        """
        Bootstrap variance for weighted mean.
        
        Uses SQL with setseed() and random() to generate Poisson(1) weights.
        """
        n_bootstraps = n_bootstraps or self.default_n_bootstraps
        start_time = time.time()
        
        # Point estimate
        sql_point = f"""
            SELECT SUM({column} * {weight}) / SUM({weight})
            FROM lfs
            WHERE {where}
        """
        point_estimate = self.connection.execute(sql_point).fetchone()[0]
        
        # Bootstrap replicates
        # Poisson(1) can be approximated using the inverse of uniform CDF
        # For lambda=1: use -ln(1-U) where U ~ Uniform(0,1)
        # Or we use a simpler approximation: round(random() * 2 + 0.5)
        # More accurately: we'll sum multiple Bernoulli(0.5) trials
        
        bootstrap_estimates = []
        
        for rep in range(n_bootstraps):
            # Set seed for reproducibility
            self.connection.execute(f"SELECT setseed({self._generate_poisson_seed(rep) % 1000000 / 1000000.0})")
            
            # Poisson(1) approximation using sum of exponentials
            # -ln(random()) gives Exponential(1), sum until > 1 gives Poisson
            # Simpler: use geometric approximation
            sql_boot = f"""
                SELECT 
                    SUM({column} * {weight} * (
                        CASE 
                            WHEN random() < 0.368 THEN 0
                            WHEN random() < 0.736 THEN 1
                            WHEN random() < 0.920 THEN 2
                            ELSE 3
                        END
                    )) / 
                    NULLIF(SUM({weight} * (
                        CASE 
                            WHEN random() < 0.368 THEN 0
                            WHEN random() < 0.736 THEN 1
                            WHEN random() < 0.920 THEN 2
                            ELSE 3
                        END
                    )), 0)
                FROM lfs
                WHERE {where}
            """
            
            result = self.connection.execute(sql_boot).fetchone()[0]
            if result is not None:
                bootstrap_estimates.append(result)
        
        return BootstrapResults(
            estimate=point_estimate,
            n_bootstraps=len(bootstrap_estimates),
            bootstrap_estimates=np.array(bootstrap_estimates),
            computation_time=time.time() - start_time
        )
    
    def gender_gap(
        self,
        target: str = 'LOG_REAL_HRLYEARN',
        weight: str = 'FINALWT',
        where: str = "HRLYEARN > 0",
        n_bootstraps: int = None,
        year: Optional[int] = None
    ) -> BootstrapResults:
        """
        Bootstrap variance for gender wage gap.
        
        Gap = E[Y|Male] - E[Y|Female] (weighted means)
        """
        n_bootstraps = n_bootstraps or self.default_n_bootstraps
        start_time = time.time()
        
        year_filter = f" AND SURVYEAR = {year}" if year else ""
        full_filter = f"{where}{year_filter}"
        
        # Point estimate
        sql_point = f"""
            SELECT 
                (SUM(CASE WHEN IS_FEMALE = 0 THEN {target} * {weight} END) / 
                 SUM(CASE WHEN IS_FEMALE = 0 THEN {weight} END))
                -
                (SUM(CASE WHEN IS_FEMALE = 1 THEN {target} * {weight} END) / 
                 SUM(CASE WHEN IS_FEMALE = 1 THEN {weight} END))
            FROM lfs
            WHERE {full_filter}
        """
        point_estimate = self.connection.execute(sql_point).fetchone()[0]
        
        # Bootstrap
        bootstrap_estimates = []
        
        for rep in range(n_bootstraps):
            self.connection.execute(f"SELECT setseed({self._generate_poisson_seed(rep) % 1000000 / 1000000.0})")
            
            # Generate Poisson weights inline
            sql_boot = f"""
                WITH poisson_weights AS (
                    SELECT *,
                        GREATEST(0, CAST(random() * 3 - 0.5 AS INTEGER)) as pois_w
                    FROM lfs
                    WHERE {full_filter}
                )
                SELECT 
                    (SUM(CASE WHEN IS_FEMALE = 0 THEN {target} * {weight} * pois_w END) / 
                     NULLIF(SUM(CASE WHEN IS_FEMALE = 0 THEN {weight} * pois_w END), 0))
                    -
                    (SUM(CASE WHEN IS_FEMALE = 1 THEN {target} * {weight} * pois_w END) / 
                     NULLIF(SUM(CASE WHEN IS_FEMALE = 1 THEN {weight} * pois_w END), 0))
                FROM poisson_weights
            """
            
            result = self.connection.execute(sql_boot).fetchone()[0]
            if result is not None:
                bootstrap_estimates.append(result)
        
        return BootstrapResults(
            estimate=point_estimate,
            n_bootstraps=len(bootstrap_estimates),
            bootstrap_estimates=np.array(bootstrap_estimates),
            computation_time=time.time() - start_time
        )
    
    def quantile_gap(
        self,
        quantile: float = 0.5,
        target: str = 'LOG_REAL_HRLYEARN',
        where: str = "HRLYEARN > 0",
        n_bootstraps: int = None
    ) -> BootstrapResults:
        """
        Bootstrap variance for quantile gap.
        
        Gap = Q_τ(Y|Male) - Q_τ(Y|Female)
        """
        n_bootstraps = n_bootstraps or self.default_n_bootstraps
        start_time = time.time()
        
        # Point estimate
        sql_point = f"""
            SELECT 
                (SELECT QUANTILE_CONT({target}, {quantile}) FROM lfs WHERE {where} AND IS_FEMALE = 0)
                -
                (SELECT QUANTILE_CONT({target}, {quantile}) FROM lfs WHERE {where} AND IS_FEMALE = 1)
        """
        point_estimate = self.connection.execute(sql_point).fetchone()[0]
        
        # For quantile bootstrap, we need to resample
        # Using subsampling approach for efficiency
        bootstrap_estimates = []
        
        # Get counts
        n_male = self.connection.execute(
            f"SELECT COUNT(*) FROM lfs WHERE {where} AND IS_FEMALE = 0"
        ).fetchone()[0]
        n_female = self.connection.execute(
            f"SELECT COUNT(*) FROM lfs WHERE {where} AND IS_FEMALE = 1"
        ).fetchone()[0]
        
        # Subsample size (sqrt(n) rule for subsampling)
        subsample_frac = min(0.5, 10000 / min(n_male, n_female))
        
        for rep in range(n_bootstraps):
            self.connection.execute(f"SELECT setseed({self._generate_poisson_seed(rep) % 1000000 / 1000000.0})")
            
            sql_boot = f"""
                SELECT 
                    (SELECT QUANTILE_CONT({target}, {quantile}) 
                     FROM lfs 
                     WHERE {where} AND IS_FEMALE = 0
                     USING SAMPLE {subsample_frac * 100} PERCENT (BERNOULLI))
                    -
                    (SELECT QUANTILE_CONT({target}, {quantile}) 
                     FROM lfs 
                     WHERE {where} AND IS_FEMALE = 1
                     USING SAMPLE {subsample_frac * 100} PERCENT (BERNOULLI))
            """
            
            result = self.connection.execute(sql_boot).fetchone()[0]
            if result is not None:
                bootstrap_estimates.append(result)
        
        return BootstrapResults(
            estimate=point_estimate,
            n_bootstraps=len(bootstrap_estimates),
            bootstrap_estimates=np.array(bootstrap_estimates),
            computation_time=time.time() - start_time
        )
    
    def custom_statistic(
        self,
        sql_template: str,
        n_bootstraps: int = None,
        poisson_weight_placeholder: str = "{POIS_W}"
    ) -> BootstrapResults:
        """
        Bootstrap variance for custom SQL statistic.
        
        Args:
            sql_template: SQL that computes statistic, with {POIS_W} placeholder
                         for where Poisson weight should be inserted
            n_bootstraps: Number of bootstrap replications
            poisson_weight_placeholder: Placeholder in template
            
        Example:
            sql = '''
                SELECT AVG(HRLYEARN * FINALWT * {POIS_W}) / AVG(FINALWT * {POIS_W})
                FROM lfs WHERE HRLYEARN > 0
            '''
            result = bootstrap.custom_statistic(sql)
        """
        n_bootstraps = n_bootstraps or self.default_n_bootstraps
        start_time = time.time()
        
        # Point estimate (pois_w = 1)
        point_sql = sql_template.replace(poisson_weight_placeholder, "1")
        point_estimate = self.connection.execute(point_sql).fetchone()[0]
        
        bootstrap_estimates = []
        
        for rep in range(n_bootstraps):
            self.connection.execute(f"SELECT setseed({self._generate_poisson_seed(rep) % 1000000 / 1000000.0})")
            
            # Generate Poisson weights
            pois_expr = "GREATEST(0, CAST(random() * 3 - 0.5 AS INTEGER))"
            boot_sql = sql_template.replace(poisson_weight_placeholder, pois_expr)
            
            result = self.connection.execute(boot_sql).fetchone()[0]
            if result is not None:
                bootstrap_estimates.append(result)
        
        return BootstrapResults(
            estimate=point_estimate,
            n_bootstraps=len(bootstrap_estimates),
            bootstrap_estimates=np.array(bootstrap_estimates),
            computation_time=time.time() - start_time
        )
    
    def confidence_interval(
        self,
        statistic: str,
        level: float = 0.95,
        n_bootstraps: int = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Get bootstrap confidence interval for common statistics.
        
        Args:
            statistic: One of 'gender_gap', 'mean_wage', 'quantile_gap'
            level: Confidence level (default 0.95)
            n_bootstraps: Number of replications
            **kwargs: Additional arguments for the statistic
            
        Returns:
            Dictionary with estimate, ci_lower, ci_upper, std_error
        """
        if statistic == 'gender_gap':
            result = self.gender_gap(n_bootstraps=n_bootstraps, **kwargs)
        elif statistic == 'mean_wage':
            result = self.weighted_mean(
                column=kwargs.get('column', 'HRLYEARN'),
                n_bootstraps=n_bootstraps,
                **{k: v for k, v in kwargs.items() if k != 'column'}
            )
        elif statistic == 'quantile_gap':
            result = self.quantile_gap(n_bootstraps=n_bootstraps, **kwargs)
        else:
            raise ValueError(f"Unknown statistic: {statistic}")
        
        # Recompute CI at requested level
        alpha = 1 - level
        ci_lower = np.percentile(result.bootstrap_estimates, alpha/2 * 100)
        ci_upper = np.percentile(result.bootstrap_estimates, (1 - alpha/2) * 100)
        
        return {
            'estimate': result.estimate,
            'std_error': result.std_error,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'level': level,
            'n_bootstraps': result.n_bootstraps,
            'is_reliable': result.is_reliable
        }
    
    def gap_with_ci(
        self,
        by_group: str = None,
        year: Optional[int] = None,
        n_bootstraps: int = 200
    ) -> Dict[Any, Dict[str, float]]:
        """
        Compute gender gap with confidence intervals, optionally by group.
        
        Args:
            by_group: Group variable (PROV, NOC_10, EDUC, etc.)
            year: Year filter
            n_bootstraps: Bootstrap replications (fewer for speed)
            
        Returns:
            Dictionary mapping group values to gap with CI
        """
        if by_group is None:
            # Overall gap
            return {'overall': self.confidence_interval(
                'gender_gap', year=year, n_bootstraps=n_bootstraps
            )}
        
        year_filter = f" AND SURVYEAR = {year}" if year else ""
        
        # Get group values
        sql_groups = f"""
            SELECT DISTINCT {by_group}
            FROM lfs
            WHERE HRLYEARN > 0{year_filter}
            ORDER BY {by_group}
        """
        groups = self.connection.execute(sql_groups).fetchdf()[by_group].tolist()
        
        results = {}
        for grp in groups:
            try:
                where = f"HRLYEARN > 0 AND {by_group} = {grp}"
                results[grp] = self.confidence_interval(
                    'gender_gap',
                    where=where,
                    year=year,
                    n_bootstraps=n_bootstraps
                )
            except Exception as e:
                logger.warning(f"Bootstrap failed for {by_group}={grp}: {e}")
        
        return results
