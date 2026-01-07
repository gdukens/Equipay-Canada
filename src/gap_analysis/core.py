"""
EquiPay Canada - Core Gap Analysis
===================================

Fundamental gap calculation methods with proper survey weighting,
confidence intervals, and statistical inference.

This module provides the building blocks for all gap analyses,
ensuring consistent methodology throughout the project.

Key Features:
-------------
1. Survey-weighted means and standard errors
2. Bootstrap confidence intervals
3. Multiple gap definitions (raw, adjusted, percentage)
4. Proper handling of log wages
5. Integration with leakage prevention

References:
-----------
- Statistics Canada (2020) - Labour Force Survey methodology
- Blau & Kahn (2017) - The Gender Wage Gap: Extent, Trends, & Explanations
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm, t as t_dist

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class GapResult:
    """Result of a gap calculation."""
    gap_absolute: float          # Absolute difference (e.g., $2.50/hour)
    gap_percentage: float        # Percentage gap (e.g., -12.5%)
    gap_log: float              # Log wage gap (semi-elasticity)
    se_absolute: float          # Standard error of absolute gap
    se_percentage: float        # Standard error of percentage gap
    ci_lower: float            # 95% CI lower bound (absolute)
    ci_upper: float            # 95% CI upper bound (absolute)
    group_a_mean: float        # Mean for reference group
    group_b_mean: float        # Mean for comparison group
    group_a_n: int             # Sample size reference
    group_b_n: int             # Sample size comparison
    group_a_weighted_n: float  # Weighted population reference
    group_b_weighted_n: float  # Weighted population comparison
    p_value: float             # P-value for H0: gap = 0
    significant: bool          # Significant at alpha=0.05
    method: str                # Calculation method used
    notes: List[str] = None    # Any notes/warnings
    
    def __repr__(self):
        return (f"GapResult(gap={self.gap_percentage:.1f}%, "
                f"p={self.p_value:.4f}, "
                f"CI=[{self.ci_lower:.2f}, {self.ci_upper:.2f}])")


@dataclass
class GroupStats:
    """Statistics for a single group."""
    mean: float
    se: float
    median: float
    n: int
    weighted_n: float
    ci_lower: float
    ci_upper: float


# =============================================================================
# WEIGHTED STATISTICS
# =============================================================================

def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """
    Calculate weighted mean.
    
    Parameters
    ----------
    values : array
        Data values
    weights : array
        Survey weights (FINALWT)
        
    Returns
    -------
    float
        Weighted mean
    """
    mask = ~np.isnan(values) & ~np.isnan(weights) & (weights > 0)
    if mask.sum() == 0:
        return np.nan
    return np.average(values[mask], weights=weights[mask])


def weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    """
    Calculate weighted standard deviation.
    
    Uses Bessel's correction for sample variance.
    """
    mask = ~np.isnan(values) & ~np.isnan(weights) & (weights > 0)
    if mask.sum() < 2:
        return np.nan
    
    v = values[mask]
    w = weights[mask]
    
    mean = np.average(v, weights=w)
    variance = np.average((v - mean) ** 2, weights=w)
    
    # Bessel correction
    n = mask.sum()
    variance = variance * n / (n - 1)
    
    return np.sqrt(variance)


def weighted_se(values: np.ndarray, weights: np.ndarray) -> float:
    """
    Calculate standard error of weighted mean.
    
    Uses survey sampling formula for clustered data.
    """
    mask = ~np.isnan(values) & ~np.isnan(weights) & (weights > 0)
    n = mask.sum()
    if n < 2:
        return np.nan
    
    std = weighted_std(values, weights)
    
    # Effective sample size (Kish design effect approximation)
    w = weights[mask]
    deff = n * np.sum(w**2) / (np.sum(w)**2)
    n_eff = n / deff
    
    return std / np.sqrt(n_eff)


def weighted_quantile(
    values: np.ndarray, 
    weights: np.ndarray, 
    q: float
) -> float:
    """
    Calculate weighted quantile.
    
    Parameters
    ----------
    values : array
        Data values
    weights : array
        Survey weights
    q : float
        Quantile (0 to 1)
        
    Returns
    -------
    float
        Weighted quantile value
    """
    mask = ~np.isnan(values) & ~np.isnan(weights) & (weights > 0)
    if mask.sum() == 0:
        return np.nan
    
    v = values[mask]
    w = weights[mask]
    
    # Sort by values
    idx = np.argsort(v)
    v = v[idx]
    w = w[idx]
    
    # Cumulative weights
    cumsum = np.cumsum(w)
    cutoff = q * cumsum[-1]
    
    # Find quantile
    return v[np.searchsorted(cumsum, cutoff)]


def calculate_group_stats(
    values: pd.Series,
    weights: pd.Series,
    alpha: float = 0.05
) -> GroupStats:
    """
    Calculate comprehensive statistics for a group.
    
    Parameters
    ----------
    values : Series
        Wage values for the group
    weights : Series
        Survey weights
    alpha : float
        Significance level for CI
        
    Returns
    -------
    GroupStats
        Complete group statistics
    """
    v = values.values
    w = weights.values
    
    mean = weighted_mean(v, w)
    se = weighted_se(v, w)
    median = weighted_quantile(v, w, 0.5)
    n = (~np.isnan(v)).sum()
    weighted_n = np.nansum(w[~np.isnan(v)])
    
    # Confidence interval
    z = norm.ppf(1 - alpha/2)
    ci_lower = mean - z * se
    ci_upper = mean + z * se
    
    return GroupStats(
        mean=mean,
        se=se,
        median=median,
        n=n,
        weighted_n=weighted_n,
        ci_lower=ci_lower,
        ci_upper=ci_upper
    )


# =============================================================================
# GAP CALCULATION
# =============================================================================

def calculate_weighted_gap(
    df: pd.DataFrame,
    wage_col: str,
    group_col: str,
    weight_col: str = 'FINALWT',
    reference_group: Any = 1,
    comparison_group: Any = 2,
    log_wage: bool = False,
    alpha: float = 0.05
) -> GapResult:
    """
    Calculate the wage gap between two groups with proper survey weighting.
    
    The gap is calculated as:
        gap = mean(comparison) - mean(reference)
    
    So a negative gap means the comparison group earns less.
    
    Parameters
    ----------
    df : DataFrame
        Data containing wage, group, and weight columns
    wage_col : str
        Column name for wages (e.g., 'REAL_HRLYEARN')
    group_col : str
        Column name for grouping variable (e.g., 'GENDER')
    weight_col : str
        Column name for survey weights (default 'FINALWT')
    reference_group : Any
        Value indicating reference group (e.g., 1 for Male)
    comparison_group : Any
        Value indicating comparison group (e.g., 2 for Female)
    log_wage : bool
        If True, wage_col contains log wages
    alpha : float
        Significance level for confidence intervals
        
    Returns
    -------
    GapResult
        Complete gap analysis results
    """
    notes = []
    
    # Filter to valid observations
    mask = (
        df[wage_col].notna() & 
        df[weight_col].notna() & 
        (df[weight_col] > 0) &
        df[group_col].isin([reference_group, comparison_group])
    )
    data = df.loc[mask].copy()
    
    if len(data) == 0:
        raise ValueError("No valid observations for gap calculation")
    
    # Split by group
    ref_mask = data[group_col] == reference_group
    comp_mask = data[group_col] == comparison_group
    
    ref_wages = data.loc[ref_mask, wage_col]
    ref_weights = data.loc[ref_mask, weight_col]
    comp_wages = data.loc[comp_mask, wage_col]
    comp_weights = data.loc[comp_mask, weight_col]
    
    # Calculate group statistics
    ref_stats = calculate_group_stats(ref_wages, ref_weights, alpha)
    comp_stats = calculate_group_stats(comp_wages, comp_weights, alpha)
    
    # Calculate gap
    gap_absolute = comp_stats.mean - ref_stats.mean
    
    # Standard error of difference (independent samples)
    se_diff = np.sqrt(ref_stats.se**2 + comp_stats.se**2)
    
    # Test statistic and p-value
    if se_diff > 0:
        t_stat = gap_absolute / se_diff
        # Two-tailed test
        df_approx = ref_stats.n + comp_stats.n - 2
        p_value = 2 * (1 - t_dist.cdf(abs(t_stat), df_approx))
    else:
        t_stat = 0
        p_value = 1.0
        notes.append("SE is zero - check data")
    
    # Confidence interval
    z = norm.ppf(1 - alpha/2)
    ci_lower = gap_absolute - z * se_diff
    ci_upper = gap_absolute + z * se_diff
    
    # Percentage gap
    if log_wage:
        # For log wages, gap is already semi-elasticity
        # Convert: exp(gap) - 1 gives percentage change
        gap_percentage = (np.exp(gap_absolute) - 1) * 100
        gap_log = gap_absolute
    else:
        # For level wages, percentage relative to reference
        if ref_stats.mean != 0:
            gap_percentage = (gap_absolute / ref_stats.mean) * 100
        else:
            gap_percentage = np.nan
            notes.append("Reference mean is zero")
        # Approximate log gap
        if ref_stats.mean > 0 and comp_stats.mean > 0:
            gap_log = np.log(comp_stats.mean) - np.log(ref_stats.mean)
        else:
            gap_log = np.nan
    
    # SE of percentage gap (delta method)
    if ref_stats.mean != 0:
        se_percentage = abs(se_diff / ref_stats.mean) * 100
    else:
        se_percentage = np.nan
    
    return GapResult(
        gap_absolute=gap_absolute,
        gap_percentage=gap_percentage,
        gap_log=gap_log,
        se_absolute=se_diff,
        se_percentage=se_percentage,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        group_a_mean=ref_stats.mean,
        group_b_mean=comp_stats.mean,
        group_a_n=ref_stats.n,
        group_b_n=comp_stats.n,
        group_a_weighted_n=ref_stats.weighted_n,
        group_b_weighted_n=comp_stats.weighted_n,
        p_value=p_value,
        significant=p_value < alpha,
        method='weighted_difference',
        notes=notes
    )


def calculate_gap_with_ci(
    df: pd.DataFrame,
    wage_col: str,
    group_col: str,
    weight_col: str = 'FINALWT',
    reference_group: Any = 1,
    comparison_group: Any = 2,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42
) -> GapResult:
    """
    Calculate gap with bootstrap confidence intervals.
    
    Uses bootstrap resampling for more robust inference,
    especially with complex survey designs.
    
    Parameters
    ----------
    df : DataFrame
        Data
    wage_col, group_col, weight_col : str
        Column names
    reference_group, comparison_group : Any
        Group identifiers
    n_bootstrap : int
        Number of bootstrap iterations
    alpha : float
        Significance level
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    GapResult
        Results with bootstrap CIs
    """
    np.random.seed(random_state)
    
    # Filter data
    mask = (
        df[wage_col].notna() & 
        df[weight_col].notna() &
        df[group_col].isin([reference_group, comparison_group])
    )
    data = df.loc[mask].copy()
    
    n = len(data)
    
    def calc_gap(sample_df):
        """Calculate gap for a sample."""
        ref_mask = sample_df[group_col] == reference_group
        comp_mask = sample_df[group_col] == comparison_group
        
        if ref_mask.sum() == 0 or comp_mask.sum() == 0:
            return np.nan
        
        ref_mean = weighted_mean(
            sample_df.loc[ref_mask, wage_col].values,
            sample_df.loc[ref_mask, weight_col].values
        )
        comp_mean = weighted_mean(
            sample_df.loc[comp_mask, wage_col].values,
            sample_df.loc[comp_mask, weight_col].values
        )
        
        return comp_mean - ref_mean
    
    # Point estimate
    point_estimate = calc_gap(data)
    
    # Bootstrap
    bootstrap_gaps = []
    for _ in range(n_bootstrap):
        # Resample with replacement
        idx = np.random.choice(n, size=n, replace=True)
        sample = data.iloc[idx]
        gap = calc_gap(sample)
        if not np.isnan(gap):
            bootstrap_gaps.append(gap)
    
    bootstrap_gaps = np.array(bootstrap_gaps)
    
    # Bootstrap SE
    se = np.std(bootstrap_gaps)
    
    # Percentile CI
    ci_lower = np.percentile(bootstrap_gaps, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_gaps, 100 * (1 - alpha / 2))
    
    # P-value (proportion of bootstrap samples with opposite sign)
    if point_estimate >= 0:
        p_value = np.mean(bootstrap_gaps <= 0) * 2
    else:
        p_value = np.mean(bootstrap_gaps >= 0) * 2
    p_value = min(p_value, 1.0)
    
    # Get full stats for means
    base_result = calculate_weighted_gap(
        df, wage_col, group_col, weight_col,
        reference_group, comparison_group, 
        log_wage=False, alpha=alpha
    )
    
    return GapResult(
        gap_absolute=point_estimate,
        gap_percentage=base_result.gap_percentage,
        gap_log=base_result.gap_log,
        se_absolute=se,
        se_percentage=abs(se / base_result.group_a_mean) * 100 if base_result.group_a_mean else np.nan,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        group_a_mean=base_result.group_a_mean,
        group_b_mean=base_result.group_b_mean,
        group_a_n=base_result.group_a_n,
        group_b_n=base_result.group_b_n,
        group_a_weighted_n=base_result.group_a_weighted_n,
        group_b_weighted_n=base_result.group_b_weighted_n,
        p_value=p_value,
        significant=p_value < alpha,
        method='bootstrap',
        notes=[f'{n_bootstrap} bootstrap iterations']
    )


# =============================================================================
# GAP ANALYZER CLASS
# =============================================================================

class GapAnalyzer:
    """
    Comprehensive wage gap analyzer with survey weighting.
    
    This class provides a unified interface for calculating gaps
    across multiple dimensions with proper statistical inference.
    
    Examples
    --------
    >>> analyzer = GapAnalyzer(df, weight_col='FINALWT')
    >>> 
    >>> # Gender gap
    >>> gender_gap = analyzer.calculate_gap('REAL_HRLYEARN', 
    ...                                      group_col='GENDER',
    ...                                      reference=1, comparison=2)
    >>> 
    >>> # Gaps by province
    >>> gaps_by_prov = analyzer.calculate_gaps_by_subgroup(
    ...     'REAL_HRLYEARN', 'GENDER', 'PROV'
    ... )
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        weight_col: str = 'FINALWT',
        default_wage_col: str = 'REAL_HRLYEARN'
    ):
        """
        Initialize the analyzer.
        
        Parameters
        ----------
        df : DataFrame
            Data with wages, demographics, and weights
        weight_col : str
            Survey weight column
        default_wage_col : str
            Default wage column to use
        """
        self.df = df
        self.weight_col = weight_col
        self.default_wage_col = default_wage_col
        
        # Validate
        if weight_col not in df.columns:
            raise ValueError(f"Weight column '{weight_col}' not found")
        if default_wage_col not in df.columns:
            warnings.warn(f"Default wage column '{default_wage_col}' not found")
        
        logger.info(f"GapAnalyzer initialized with {len(df)} records")
    
    def calculate_gap(
        self,
        wage_col: str = None,
        group_col: str = 'GENDER',
        reference: Any = 1,
        comparison: Any = 2,
        method: str = 'weighted',
        **kwargs
    ) -> GapResult:
        """
        Calculate the gap between two groups.
        
        Parameters
        ----------
        wage_col : str
            Wage column (default: self.default_wage_col)
        group_col : str
            Grouping column (default: 'GENDER')
        reference : Any
            Reference group value
        comparison : Any
            Comparison group value
        method : str
            'weighted' or 'bootstrap'
        **kwargs
            Additional arguments for the calculation method
            
        Returns
        -------
        GapResult
        """
        wage_col = wage_col or self.default_wage_col
        
        if method == 'bootstrap':
            return calculate_gap_with_ci(
                self.df, wage_col, group_col, self.weight_col,
                reference, comparison, **kwargs
            )
        else:
            return calculate_weighted_gap(
                self.df, wage_col, group_col, self.weight_col,
                reference, comparison, **kwargs
            )
    
    def calculate_gaps_by_subgroup(
        self,
        wage_col: str = None,
        group_col: str = 'GENDER',
        stratify_col: str = 'PROV',
        reference: Any = 1,
        comparison: Any = 2
    ) -> pd.DataFrame:
        """
        Calculate gaps stratified by another variable.
        
        For example: gender gaps by province.
        
        Parameters
        ----------
        wage_col : str
            Wage column
        group_col : str
            Primary grouping (e.g., GENDER)
        stratify_col : str
            Stratification variable (e.g., PROV)
        reference, comparison : Any
            Group values
            
        Returns
        -------
        DataFrame
            Gaps for each stratum
        """
        wage_col = wage_col or self.default_wage_col
        
        results = []
        for stratum in self.df[stratify_col].dropna().unique():
            subset = self.df[self.df[stratify_col] == stratum]
            
            # Check sufficient data
            mask = subset[group_col].isin([reference, comparison])
            if mask.sum() < 10:
                continue
            
            try:
                gap = calculate_weighted_gap(
                    subset, wage_col, group_col, self.weight_col,
                    reference, comparison
                )
                
                results.append({
                    stratify_col: stratum,
                    'gap_absolute': gap.gap_absolute,
                    'gap_percentage': gap.gap_percentage,
                    'se': gap.se_absolute,
                    'ci_lower': gap.ci_lower,
                    'ci_upper': gap.ci_upper,
                    'p_value': gap.p_value,
                    'significant': gap.significant,
                    'n_reference': gap.group_a_n,
                    'n_comparison': gap.group_b_n,
                    'mean_reference': gap.group_a_mean,
                    'mean_comparison': gap.group_b_mean,
                })
            except Exception as e:
                logger.warning(f"Could not calculate gap for {stratify_col}={stratum}: {e}")
        
        return pd.DataFrame(results)
    
    def calculate_all_gaps(
        self,
        wage_col: str = None,
        dimensions: List[Tuple[str, Any, Any]] = None
    ) -> Dict[str, GapResult]:
        """
        Calculate gaps across multiple dimensions.
        
        Parameters
        ----------
        wage_col : str
            Wage column
        dimensions : List[Tuple[str, Any, Any]]
            List of (column, reference, comparison) tuples
            Default: gender gap
            
        Returns
        -------
        Dict[str, GapResult]
            Gaps keyed by dimension name
        """
        wage_col = wage_col or self.default_wage_col
        
        if dimensions is None:
            dimensions = [('GENDER', 1, 2)]
        
        results = {}
        for col, ref, comp in dimensions:
            try:
                gap = self.calculate_gap(wage_col, col, ref, comp)
                results[f'{col}_{comp}_vs_{ref}'] = gap
            except Exception as e:
                logger.warning(f"Could not calculate {col} gap: {e}")
        
        return results
    
    def summary_table(
        self,
        wage_col: str = None,
        group_col: str = 'GENDER',
        stratify_cols: List[str] = None
    ) -> pd.DataFrame:
        """
        Create a summary table of gaps.
        
        Parameters
        ----------
        wage_col : str
            Wage column
        group_col : str
            Primary group
        stratify_cols : List[str]
            Variables to stratify by
            
        Returns
        -------
        DataFrame
            Summary table
        """
        wage_col = wage_col or self.default_wage_col
        
        if stratify_cols is None:
            stratify_cols = ['PROV', 'NOC_10', 'EDUC']
        
        tables = {}
        for strat in stratify_cols:
            if strat in self.df.columns:
                tables[strat] = self.calculate_gaps_by_subgroup(
                    wage_col, group_col, strat
                )
        
        return tables
