"""
EquiPay Canada - Quantile Gap Analysis
======================================

Analysis of wage gaps at different points of the distribution,
following methodologies from:
- Koenker & Bassett (1978) - Quantile Regression
- Machado & Mata (2005) - Counterfactual Decomposition
- Albrecht, Björklund & Vroman (2003) - Glass Ceiling
- Arulampalam, Booth & Bryan (2007) - Sticky Floor

This module implements:
1. Quantile regression for heterogeneous effects
2. Glass ceiling tests (gaps at top of distribution)
3. Sticky floor tests (gaps at bottom of distribution)
4. Distribution-wide gap visualization

Key Concepts:
-------------
- GLASS CEILING: Gender gap widens at higher wage quantiles
- STICKY FLOOR: Gender gap is largest at lower wage quantiles
- HETEROGENEOUS EFFECTS: Gap varies across the wage distribution
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg

from src.gap_analysis.core import weighted_quantile, weighted_mean

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class QuantileGapResult:
    """Result of quantile gap analysis."""
    
    quantile: float              # Quantile analyzed (e.g., 0.9)
    gap_absolute: float          # Gap at this quantile
    gap_percentage: float        # Percentage gap
    se: float                    # Standard error
    ci_lower: float             # 95% CI lower
    ci_upper: float             # 95% CI upper
    p_value: float              # P-value
    ref_value: float            # Reference group quantile value
    comp_value: float           # Comparison group quantile value


@dataclass
class QuantileProfileResult:
    """Complete profile of gaps across quantiles."""
    
    quantiles: List[float]
    gaps: List[QuantileGapResult]
    
    # Tests
    glass_ceiling_test: Dict[str, Any] = None
    sticky_floor_test: Dict[str, Any] = None
    
    # Regression coefficients (if QR used)
    qr_coefficients: pd.DataFrame = None
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame."""
        rows = []
        for g in self.gaps:
            rows.append({
                'quantile': g.quantile,
                'gap_absolute': g.gap_absolute,
                'gap_percentage': g.gap_percentage,
                'se': g.se,
                'ci_lower': g.ci_lower,
                'ci_upper': g.ci_upper,
                'p_value': g.p_value,
                'ref_value': g.ref_value,
                'comp_value': g.comp_value,
            })
        return pd.DataFrame(rows)


# =============================================================================
# QUANTILE GAP ANALYZER
# =============================================================================

class QuantileGapAnalyzer:
    """
    Analyze wage gaps across the distribution.
    
    This class provides methods for:
    1. Computing gaps at specific quantiles
    2. Creating full quantile profiles
    3. Testing for glass ceiling / sticky floor effects
    4. Quantile regression analysis
    
    Examples
    --------
    >>> analyzer = QuantileGapAnalyzer(df)
    >>> 
    >>> # Gap at 90th percentile
    >>> gap_90 = analyzer.gap_at_quantile(0.9, 'GENDER', 1, 2)
    >>> 
    >>> # Full profile
    >>> profile = analyzer.quantile_profile('GENDER', 1, 2)
    >>> 
    >>> # Glass ceiling test
    >>> glass = analyzer.glass_ceiling_test('GENDER')
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        wage_col: str = 'REAL_HRLYEARN',
        weight_col: str = 'FINALWT'
    ):
        """
        Initialize analyzer.
        
        Parameters
        ----------
        df : DataFrame
            Data with wages and weights
        wage_col : str
            Wage column
        weight_col : str
            Survey weight column
        """
        self.df = df
        self.wage_col = wage_col
        self.weight_col = weight_col
        
        logger.info("QuantileGapAnalyzer initialized")
    
    def gap_at_quantile(
        self,
        quantile: float,
        group_col: str = 'GENDER',
        reference: Any = 1,
        comparison: Any = 2,
        n_bootstrap: int = 200
    ) -> QuantileGapResult:
        """
        Calculate the wage gap at a specific quantile.
        
        Parameters
        ----------
        quantile : float
            Quantile to analyze (0 to 1)
        group_col : str
            Grouping variable
        reference, comparison : Any
            Group values
        n_bootstrap : int
            Bootstrap iterations for SE
            
        Returns
        -------
        QuantileGapResult
        """
        df = self.df.copy()
        
        # Filter
        mask = (
            df[self.wage_col].notna() &
            df[self.weight_col].notna() &
            df[group_col].isin([reference, comparison])
        )
        df = df.loc[mask]
        
        # Split
        ref_df = df[df[group_col] == reference]
        comp_df = df[df[group_col] == comparison]
        
        # Point estimates
        ref_q = weighted_quantile(
            ref_df[self.wage_col].values,
            ref_df[self.weight_col].values,
            quantile
        )
        comp_q = weighted_quantile(
            comp_df[self.wage_col].values,
            comp_df[self.weight_col].values,
            quantile
        )
        
        gap = comp_q - ref_q
        gap_pct = (gap / ref_q * 100) if ref_q != 0 else np.nan
        
        # Bootstrap SE
        np.random.seed(42)
        bootstrap_gaps = []
        
        n_ref, n_comp = len(ref_df), len(comp_df)
        
        for _ in range(n_bootstrap):
            ref_idx = np.random.choice(n_ref, size=n_ref, replace=True)
            comp_idx = np.random.choice(n_comp, size=n_comp, replace=True)
            
            ref_sample = ref_df.iloc[ref_idx]
            comp_sample = comp_df.iloc[comp_idx]
            
            ref_q_boot = weighted_quantile(
                ref_sample[self.wage_col].values,
                ref_sample[self.weight_col].values,
                quantile
            )
            comp_q_boot = weighted_quantile(
                comp_sample[self.wage_col].values,
                comp_sample[self.weight_col].values,
                quantile
            )
            
            bootstrap_gaps.append(comp_q_boot - ref_q_boot)
        
        se = np.std(bootstrap_gaps)
        ci_lower = np.percentile(bootstrap_gaps, 2.5)
        ci_upper = np.percentile(bootstrap_gaps, 97.5)
        
        # P-value
        if gap >= 0:
            p_value = np.mean(np.array(bootstrap_gaps) <= 0) * 2
        else:
            p_value = np.mean(np.array(bootstrap_gaps) >= 0) * 2
        p_value = min(p_value, 1.0)
        
        return QuantileGapResult(
            quantile=quantile,
            gap_absolute=gap,
            gap_percentage=gap_pct,
            se=se,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            ref_value=ref_q,
            comp_value=comp_q
        )
    
    def quantile_profile(
        self,
        group_col: str = 'GENDER',
        reference: Any = 1,
        comparison: Any = 2,
        quantiles: List[float] = None,
        n_bootstrap: int = 200
    ) -> QuantileProfileResult:
        """
        Calculate gaps across multiple quantiles.
        
        Parameters
        ----------
        group_col : str
            Grouping variable
        reference, comparison : Any
            Group values
        quantiles : List[float]
            Quantiles to analyze (default: deciles)
        n_bootstrap : int
            Bootstrap iterations
            
        Returns
        -------
        QuantileProfileResult
        """
        if quantiles is None:
            quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        
        gaps = []
        for q in quantiles:
            gap_result = self.gap_at_quantile(
                q, group_col, reference, comparison, n_bootstrap
            )
            gaps.append(gap_result)
        
        result = QuantileProfileResult(
            quantiles=quantiles,
            gaps=gaps
        )
        
        # Add glass ceiling and sticky floor tests
        result.glass_ceiling_test = self._test_glass_ceiling(gaps)
        result.sticky_floor_test = self._test_sticky_floor(gaps)
        
        return result
    
    def _test_glass_ceiling(
        self,
        gaps: List[QuantileGapResult]
    ) -> Dict[str, Any]:
        """
        Test for glass ceiling effect.
        
        Glass ceiling: gap widens at higher quantiles.
        H0: gap at 90th = gap at 50th
        H1: gap at 90th > gap at 50th (in absolute terms)
        """
        # Find gaps at relevant quantiles
        gap_50 = next((g for g in gaps if g.quantile == 0.5), None)
        gap_90 = next((g for g in gaps if g.quantile == 0.9), None)
        
        if gap_50 is None or gap_90 is None:
            return {'test': 'glass_ceiling', 'result': 'insufficient_quantiles'}
        
        # Compare absolute gaps
        ceiling_effect = abs(gap_90.gap_absolute) - abs(gap_50.gap_absolute)
        
        # Combined SE (simplified)
        se_diff = np.sqrt(gap_90.se**2 + gap_50.se**2)
        
        # Z-test
        if se_diff > 0:
            z_stat = ceiling_effect / se_diff
            p_value = 1 - stats.norm.cdf(z_stat)  # One-tailed
        else:
            z_stat = 0
            p_value = 0.5
        
        return {
            'test': 'glass_ceiling',
            'gap_50_pct': gap_50.gap_percentage,
            'gap_90_pct': gap_90.gap_percentage,
            'ceiling_effect': ceiling_effect,
            'ceiling_effect_pct': gap_90.gap_percentage - gap_50.gap_percentage,
            'z_statistic': z_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'interpretation': (
                'Evidence of glass ceiling' if p_value < 0.05 and ceiling_effect > 0
                else 'No significant glass ceiling effect'
            )
        }
    
    def _test_sticky_floor(
        self,
        gaps: List[QuantileGapResult]
    ) -> Dict[str, Any]:
        """
        Test for sticky floor effect.
        
        Sticky floor: gap is largest at lower quantiles.
        H0: gap at 10th = gap at 50th
        H1: gap at 10th > gap at 50th (in absolute terms)
        """
        gap_10 = next((g for g in gaps if g.quantile == 0.1), None)
        gap_50 = next((g for g in gaps if g.quantile == 0.5), None)
        
        if gap_10 is None or gap_50 is None:
            return {'test': 'sticky_floor', 'result': 'insufficient_quantiles'}
        
        floor_effect = abs(gap_10.gap_absolute) - abs(gap_50.gap_absolute)
        
        se_diff = np.sqrt(gap_10.se**2 + gap_50.se**2)
        
        if se_diff > 0:
            z_stat = floor_effect / se_diff
            p_value = 1 - stats.norm.cdf(z_stat)
        else:
            z_stat = 0
            p_value = 0.5
        
        return {
            'test': 'sticky_floor',
            'gap_10_pct': gap_10.gap_percentage,
            'gap_50_pct': gap_50.gap_percentage,
            'floor_effect': floor_effect,
            'floor_effect_pct': gap_10.gap_percentage - gap_50.gap_percentage,
            'z_statistic': z_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'interpretation': (
                'Evidence of sticky floor' if p_value < 0.05 and floor_effect > 0
                else 'No significant sticky floor effect'
            )
        }
    
    def quantile_regression(
        self,
        X: pd.DataFrame,
        y: pd.Series = None,
        gender_col: str = 'IS_FEMALE',
        quantiles: List[float] = None
    ) -> pd.DataFrame:
        """
        Run quantile regression and extract gender coefficients.
        
        Parameters
        ----------
        X : DataFrame
            Feature matrix (including gender)
        y : Series
            Log wages (optional, uses self.wage_col if None)
        gender_col : str
            Gender indicator column in X
        quantiles : List[float]
            Quantiles to estimate
            
        Returns
        -------
        DataFrame
            Gender coefficients at each quantile
        """
        if y is None:
            # Use log wages
            y = np.log(self.df[self.wage_col].replace(0, np.nan))
        
        if quantiles is None:
            quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        
        # Align
        common_idx = X.index.intersection(y.dropna().index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        
        # Add constant
        X_const = sm.add_constant(X)
        
        results = []
        for q in quantiles:
            try:
                model = QuantReg(y, X_const)
                fitted = model.fit(q=q)
                
                # Gender coefficient
                if gender_col in fitted.params:
                    coef = fitted.params[gender_col]
                    se = fitted.bse[gender_col]
                    
                    results.append({
                        'quantile': q,
                        'gender_coef': coef,
                        'gender_se': se,
                        'gender_pvalue': fitted.pvalues[gender_col],
                        'gender_coef_pct': (np.exp(coef) - 1) * 100,  # Convert to %
                    })
            except Exception as e:
                logger.warning(f"QR failed at quantile {q}: {e}")
        
        return pd.DataFrame(results)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def glass_ceiling_test(
    df: pd.DataFrame,
    wage_col: str = 'REAL_HRLYEARN',
    group_col: str = 'GENDER',
    weight_col: str = 'FINALWT',
    reference: Any = 1,
    comparison: Any = 2
) -> Dict[str, Any]:
    """
    Quick glass ceiling test.
    
    Tests whether the wage gap is larger at the top of the distribution
    compared to the middle.
    
    Returns
    -------
    Dict with test results
    """
    analyzer = QuantileGapAnalyzer(df, wage_col, weight_col)
    profile = analyzer.quantile_profile(
        group_col, reference, comparison,
        quantiles=[0.1, 0.5, 0.9]
    )
    return profile.glass_ceiling_test


def sticky_floor_test(
    df: pd.DataFrame,
    wage_col: str = 'REAL_HRLYEARN',
    group_col: str = 'GENDER',
    weight_col: str = 'FINALWT',
    reference: Any = 1,
    comparison: Any = 2
) -> Dict[str, Any]:
    """
    Quick sticky floor test.
    
    Tests whether the wage gap is larger at the bottom of the distribution
    compared to the middle.
    
    Returns
    -------
    Dict with test results
    """
    analyzer = QuantileGapAnalyzer(df, wage_col, weight_col)
    profile = analyzer.quantile_profile(
        group_col, reference, comparison,
        quantiles=[0.1, 0.5, 0.9]
    )
    return profile.sticky_floor_test


def create_quantile_comparison_table(
    df: pd.DataFrame,
    wage_col: str = 'REAL_HRLYEARN',
    group_col: str = 'GENDER',
    weight_col: str = 'FINALWT'
) -> pd.DataFrame:
    """
    Create a table comparing quantiles between groups.
    
    Returns
    -------
    DataFrame with quantiles for each group and gaps
    """
    analyzer = QuantileGapAnalyzer(df, wage_col, weight_col)
    profile = analyzer.quantile_profile(group_col, 1, 2)
    
    return profile.to_dataframe()
