"""
EquiPay Canada - Analytics Engine
==================================

SQL-accelerated econometric and statistical analysis.

This module provides production-grade implementations of:
- Oaxaca-Blinder decomposition (unexplained/explained gaps)
- Quantile decomposition (glass ceiling/sticky floor)
- Bootstrap variance estimation (Poisson per Statistics Canada)
- Causal inference (DiD, event study, synthetic control)
- Fairness metrics (demographic parity, equalized odds)
- Time series analysis (structural breaks, stationarity)

All methods use SQL for aggregation where possible,
minimizing Python memory usage.
"""

from .decomposition import OaxacaBlinder, RIFDecomposition, DecompositionResult
from .quantile import QuantileGapAnalyzer, GlassCeilingAnalyzer, QuantileGapResult, GlassCeilingResult
from .bootstrap import PoissonBootstrap, BootstrapResults
from .causal import (
    DifferenceInDifferences,
    EventStudy,
    SyntheticControl,
    PolicyEvaluator,
    DiDResult,
    EventStudyResult
)

__all__ = [
    # Decomposition
    'OaxacaBlinder',
    'RIFDecomposition',
    'DecompositionResult',
    
    # Quantile
    'QuantileGapAnalyzer',
    'GlassCeilingAnalyzer',
    'QuantileGapResult',
    'GlassCeilingResult',
    
    # Bootstrap
    'PoissonBootstrap',
    'BootstrapResults',
    
    # Causal
    'DifferenceInDifferences',
    'EventStudy',
    'SyntheticControl',
    'PolicyEvaluator',
    'DiDResult',
    'EventStudyResult',
]


class AnalyticsEngine:
    """
    Unified interface for all analytics capabilities.
    
    Provides lazy initialization and caching of analyzers.
    
    Usage:
        engine = AnalyticsEngine(connection)
        
        # Oaxaca-Blinder decomposition
        results = engine.decomposition.oaxaca_blinder(
            features=['EDUC', 'EXPERIENCE', 'NOC_10'],
            target='LOG_REAL_HRLYEARN',
            group_col='GENDER'
        )
        
        # Quantile gaps
        gaps = engine.quantile.gap_by_quantile(
            quantiles=[0.1, 0.25, 0.5, 0.75, 0.9]
        )
        
        # Bootstrap confidence intervals
        ci = engine.bootstrap.confidence_interval(
            statistic='gender_gap',
            n_bootstraps=1000
        )
    """
    
    def __init__(self, connection):
        """
        Initialize analytics engine.
        
        Args:
            connection: DuckDB connection
        """
        self.connection = connection
        
        # Lazy-initialized analyzers
        self._decomposition = None
        self._quantile = None
        self._bootstrap = None
        self._causal = None
    
    @property
    def decomposition(self) -> OaxacaBlinder:
        """Get decomposition analyzer."""
        if self._decomposition is None:
            self._decomposition = OaxacaBlinder(self.connection)
        return self._decomposition
    
    @property
    def quantile(self) -> QuantileGapAnalyzer:
        """Get quantile gap analyzer."""
        if self._quantile is None:
            self._quantile = QuantileGapAnalyzer(self.connection)
        return self._quantile
    
    @property
    def bootstrap(self) -> PoissonBootstrap:
        """Get bootstrap variance estimator."""
        if self._bootstrap is None:
            self._bootstrap = PoissonBootstrap(self.connection)
        return self._bootstrap
    
    @property
    def causal(self) -> PolicyEvaluator:
        """Get causal inference tools."""
        if self._causal is None:
            self._causal = PolicyEvaluator(self.connection)
        return self._causal
    
    def rif_decomposition(
        self,
        features: list,
        quantile: float = 0.5,
        group_col: str = 'IS_FEMALE',
        target: str = 'LOG_REAL_HRLYEARN',
        weight: str = 'FINALWT'
    ) -> dict:
        """
        Perform RIF (Recentered Influence Function) decomposition.
        
        Extends Oaxaca-Blinder to quantiles, enabling analysis of
        glass ceiling and sticky floor effects.
        
        Args:
            features: Control variables
            quantile: Quantile to analyze (0-1)
            group_col: Binary group indicator
            target: Outcome variable
            weight: Survey weight
            
        Returns:
            Dictionary with explained/unexplained components
        """
        rif = RIFDecomposition(self.connection)
        return rif.decompose(
            features=features,
            quantile=quantile,
            group_col=group_col,
            target=target,
            weight=weight
        )
    
    def gap_across_quantiles(
        self,
        quantiles: list = None,
        by_group: str = None,
        year: int = None
    ) -> dict:
        """
        Compute gender gap at multiple quantiles.
        
        Args:
            quantiles: List of quantiles (default: deciles)
            by_group: Optional grouping (PROV, NOC_10, EDUC)
            year: Optional year filter
            
        Returns:
            Dictionary mapping quantiles to gaps
        """
        if quantiles is None:
            quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        
        return self.quantile.gap_by_quantile(
            quantiles=quantiles,
            by_group=by_group,
            year=year
        )
    
    def event_study(
        self,
        treatment_var: str,
        time_var: str = 'SURVYEAR',
        event_time: int = 0,
        pre_periods: int = 3,
        post_periods: int = 3,
        controls: list = None
    ) -> dict:
        """
        Run event study analysis for policy evaluation.
        
        Args:
            treatment_var: Binary treatment indicator
            time_var: Time variable
            event_time: Period of treatment (0 = treatment year)
            pre_periods: Number of pre-treatment periods
            post_periods: Number of post-treatment periods
            controls: Control variables
            
        Returns:
            Event study coefficients and CIs
        """
        es = EventStudy(self.connection)
        return es.estimate(
            treatment_var=treatment_var,
            time_var=time_var,
            event_time=event_time,
            pre_periods=pre_periods,
            post_periods=post_periods,
            controls=controls
        )
    
    def parallel_trends_test(
        self,
        treatment_var: str,
        time_var: str = 'SURVYEAR',
        pre_periods: int = 3
    ) -> dict:
        """
        Test parallel trends assumption for DiD.
        
        Returns:
            Test statistics and p-values
        """
        did = DifferenceInDifferences(self.connection)
        return did.test_parallel_trends(
            treatment_var=treatment_var,
            time_var=time_var,
            pre_periods=pre_periods
        )
    
    def __repr__(self):
        return f"AnalyticsEngine(connection={self.connection is not None})"
