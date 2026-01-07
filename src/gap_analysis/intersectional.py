"""
EquiPay Canada - Intersectional Gap Analysis
=============================================

Analysis of wage gaps across multiple intersecting dimensions,
following the intersectionality framework from:
- Crenshaw (1989) - Demarginalizing the Intersection
- Blau & Kahn (2017) - The Gender Wage Gap
- Greenman & Xie (2008) - Double Jeopardy

This module allows analysis of compound disadvantages such as:
- Female × Immigrant gaps
- Female × Province effects
- Gender × Occupation × Industry interactions

Key Features:
-------------
1. Multi-dimensional gap calculation
2. Compound penalty estimation
3. Visualization of intersection effects
4. Statistical tests for interaction significance
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from itertools import product
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from src.gap_analysis.core import (
    GapResult, 
    weighted_mean, 
    weighted_se,
    calculate_weighted_gap
)

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class IntersectionalResult:
    """Result of intersectional analysis."""
    
    # Dimensions analyzed
    dimensions: List[str]
    
    # Number of intersections
    n_intersections: int
    
    # Reference group (e.g., White Male)
    reference_group: Dict[str, Any]
    
    # Gap for each intersection relative to reference
    gaps: pd.DataFrame
    
    # Statistical summary
    largest_gap: Tuple[Dict, float]
    smallest_gap: Tuple[Dict, float]
    
    # Compound effects (beyond additive)
    compound_penalties: pd.DataFrame = None
    
    # Model fit if regression used
    r_squared: float = None
    
    def __repr__(self):
        return (
            f"IntersectionalResult(\n"
            f"  Dimensions: {self.dimensions}\n"
            f"  N intersections: {self.n_intersections}\n"
            f"  Largest gap: {self.largest_gap[1]:.1f}%\n"
            f"  Smallest gap: {self.smallest_gap[1]:.1f}%\n"
            f")"
        )


@dataclass
class CompoundPenalty:
    """Compound penalty beyond additive effects."""
    
    dimension_a: str
    value_a: Any
    dimension_b: str
    value_b: Any
    
    main_effect_a: float      # Effect of dimension A alone
    main_effect_b: float      # Effect of dimension B alone
    additive_effect: float    # Sum of main effects
    actual_effect: float      # Observed effect for intersection
    compound_penalty: float   # Actual - Additive
    
    is_double_jeopardy: bool  # penalty > 0 (worse than sum)
    
    se: float = None
    p_value: float = None


# =============================================================================
# INTERSECTIONAL ANALYZER
# =============================================================================

class IntersectionalAnalyzer:
    """
    Analyze wage gaps across multiple intersecting dimensions.
    
    This class implements intersectionality analysis to identify
    compound disadvantages beyond simple additive effects.
    
    Examples
    --------
    >>> analyzer = IntersectionalAnalyzer(df)
    >>> 
    >>> # Two-way intersection
    >>> result = analyzer.analyze(['GENDER', 'IMMIG'])
    >>> 
    >>> # Check for compound penalties
    >>> penalties = analyzer.test_compound_effects('GENDER', 'IMMIG')
    >>> 
    >>> # Full intersection matrix
    >>> matrix = analyzer.create_intersection_matrix(
    ...     'GENDER', 'PROV', 'REAL_HRLYEARN'
    ... )
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        weight_col: str = 'FINALWT',
        wage_col: str = 'REAL_HRLYEARN'
    ):
        """
        Initialize analyzer.
        
        Parameters
        ----------
        df : DataFrame
            Data with wages, weights, and demographic variables
        weight_col : str
            Survey weight column
        wage_col : str
            Wage column for gap calculations
        """
        self.df = df
        self.weight_col = weight_col
        self.wage_col = wage_col
        
        # Validate
        if weight_col not in df.columns:
            raise ValueError(f"Weight column '{weight_col}' not found")
        if wage_col not in df.columns:
            raise ValueError(f"Wage column '{wage_col}' not found")
        
        logger.info(f"IntersectionalAnalyzer initialized")
    
    def analyze(
        self,
        dimensions: List[str],
        reference: Dict[str, Any] = None,
        min_n: int = 50
    ) -> IntersectionalResult:
        """
        Analyze gaps across intersecting dimensions.
        
        Parameters
        ----------
        dimensions : List[str]
            Columns defining the intersection (e.g., ['GENDER', 'IMMIG'])
        reference : Dict[str, Any]
            Reference group values (e.g., {'GENDER': 1, 'IMMIG': 4})
            If None, uses the group with highest mean wage
        min_n : int
            Minimum sample size for valid intersection
            
        Returns
        -------
        IntersectionalResult
        """
        df = self.df.copy()
        
        # Validate dimensions
        for dim in dimensions:
            if dim not in df.columns:
                raise ValueError(f"Dimension '{dim}' not found in data")
        
        # Create intersection groups
        df['_intersection_'] = df[dimensions].apply(
            lambda x: tuple(x.values), axis=1
        )
        
        # Filter to valid observations
        mask = (
            df[self.wage_col].notna() &
            df[self.weight_col].notna() &
            (df[self.weight_col] > 0)
        )
        df = df.loc[mask]
        
        # Calculate stats for each intersection
        results = []
        for group, group_df in df.groupby('_intersection_'):
            if len(group_df) < min_n:
                continue
            
            mean_wage = weighted_mean(
                group_df[self.wage_col].values,
                group_df[self.weight_col].values
            )
            se = weighted_se(
                group_df[self.wage_col].values,
                group_df[self.weight_col].values
            )
            
            # Create group dict
            group_dict = dict(zip(dimensions, group))
            
            results.append({
                **group_dict,
                'mean_wage': mean_wage,
                'se': se,
                'n': len(group_df),
                'weighted_n': group_df[self.weight_col].sum(),
            })
        
        results_df = pd.DataFrame(results)
        
        # Determine reference group
        if reference is None:
            # Use highest wage group as reference
            ref_idx = results_df['mean_wage'].idxmax()
            reference = {dim: results_df.loc[ref_idx, dim] for dim in dimensions}
        
        # Get reference wage
        ref_mask = pd.Series(True, index=results_df.index)
        for dim, val in reference.items():
            ref_mask &= (results_df[dim] == val)
        
        if ref_mask.sum() == 0:
            raise ValueError(f"Reference group {reference} not found in data")
        
        ref_wage = results_df.loc[ref_mask, 'mean_wage'].iloc[0]
        
        # Calculate gaps
        results_df['gap_absolute'] = results_df['mean_wage'] - ref_wage
        results_df['gap_percentage'] = (results_df['gap_absolute'] / ref_wage) * 100
        
        # Find extreme gaps
        largest_idx = results_df['gap_percentage'].idxmin()  # Most negative
        smallest_idx = results_df['gap_percentage'].idxmax()  # Least negative/most positive
        
        largest_group = {dim: results_df.loc[largest_idx, dim] for dim in dimensions}
        smallest_group = {dim: results_df.loc[smallest_idx, dim] for dim in dimensions}
        
        return IntersectionalResult(
            dimensions=dimensions,
            n_intersections=len(results_df),
            reference_group=reference,
            gaps=results_df,
            largest_gap=(largest_group, results_df.loc[largest_idx, 'gap_percentage']),
            smallest_gap=(smallest_group, results_df.loc[smallest_idx, 'gap_percentage'])
        )
    
    def test_compound_effects(
        self,
        dimension_a: str,
        dimension_b: str,
        value_a: Any = None,
        value_b: Any = None,
        ref_a: Any = None,
        ref_b: Any = None
    ) -> CompoundPenalty:
        """
        Test for compound effects beyond additive.
        
        Tests whether the intersection of two disadvantaged groups
        experiences a penalty beyond the sum of individual effects.
        
        Parameters
        ----------
        dimension_a, dimension_b : str
            The two dimensions to test (e.g., 'GENDER', 'IMMIG')
        value_a, value_b : Any
            Values to test (e.g., 2 for Female, 1 for Recent Immigrant)
        ref_a, ref_b : Any
            Reference values (e.g., 1 for Male, 4 for Non-immigrant)
            
        Returns
        -------
        CompoundPenalty
            Analysis of compound effects
        """
        df = self.df.copy()
        
        # Determine values if not specified
        if value_a is None:
            # Use minority group (lower mean wage)
            means = df.groupby(dimension_a).apply(
                lambda x: weighted_mean(x[self.wage_col].values, x[self.weight_col].values)
            )
            value_a = means.idxmin()
        
        if ref_a is None:
            ref_a = df[dimension_a].mode().iloc[0]
            if ref_a == value_a:
                ref_a = df[df[dimension_a] != value_a][dimension_a].mode().iloc[0]
        
        if value_b is None:
            means = df.groupby(dimension_b).apply(
                lambda x: weighted_mean(x[self.wage_col].values, x[self.weight_col].values)
            )
            value_b = means.idxmin()
        
        if ref_b is None:
            ref_b = df[dimension_b].mode().iloc[0]
            if ref_b == value_b:
                ref_b = df[df[dimension_b] != value_b][dimension_b].mode().iloc[0]
        
        # Calculate four group means
        def group_mean(mask):
            return weighted_mean(
                df.loc[mask, self.wage_col].values,
                df.loc[mask, self.weight_col].values
            )
        
        # Reference: ref_a, ref_b
        ref_mean = group_mean((df[dimension_a] == ref_a) & (df[dimension_b] == ref_b))
        
        # Main effect A: value_a, ref_b
        a_mean = group_mean((df[dimension_a] == value_a) & (df[dimension_b] == ref_b))
        main_effect_a = a_mean - ref_mean
        
        # Main effect B: ref_a, value_b
        b_mean = group_mean((df[dimension_a] == ref_a) & (df[dimension_b] == value_b))
        main_effect_b = b_mean - ref_mean
        
        # Intersection: value_a, value_b
        ab_mean = group_mean((df[dimension_a] == value_a) & (df[dimension_b] == value_b))
        actual_effect = ab_mean - ref_mean
        
        # Additive expectation
        additive_effect = main_effect_a + main_effect_b
        
        # Compound penalty
        compound_penalty = actual_effect - additive_effect
        
        # Is it double jeopardy (worse than sum)?
        is_double_jeopardy = compound_penalty < 0  # More negative = worse
        
        return CompoundPenalty(
            dimension_a=dimension_a,
            value_a=value_a,
            dimension_b=dimension_b,
            value_b=value_b,
            main_effect_a=main_effect_a,
            main_effect_b=main_effect_b,
            additive_effect=additive_effect,
            actual_effect=actual_effect,
            compound_penalty=compound_penalty,
            is_double_jeopardy=is_double_jeopardy
        )
    
    def create_intersection_matrix(
        self,
        row_dim: str,
        col_dim: str,
        stat: str = 'gap_percentage'
    ) -> pd.DataFrame:
        """
        Create a matrix showing gaps for each intersection.
        
        Parameters
        ----------
        row_dim : str
            Dimension for rows (e.g., 'GENDER')
        col_dim : str
            Dimension for columns (e.g., 'PROV')
        stat : str
            Statistic to show: 'gap_percentage', 'mean_wage', 'n'
            
        Returns
        -------
        DataFrame
            Matrix with row_dim as rows and col_dim as columns
        """
        result = self.analyze([row_dim, col_dim])
        
        # Pivot to matrix
        matrix = result.gaps.pivot(
            index=row_dim,
            columns=col_dim,
            values=stat
        )
        
        return matrix
    
    def full_intersection_analysis(
        self,
        dimensions: List[str],
        reference: Dict[str, Any] = None,
        min_n: int = 50
    ) -> pd.DataFrame:
        """
        Comprehensive analysis of all intersections.
        
        Returns a detailed table with all statistics for each intersection.
        
        Parameters
        ----------
        dimensions : List[str]
            Dimensions to analyze
        reference : Dict[str, Any]
            Reference group
        min_n : int
            Minimum sample size
            
        Returns
        -------
        DataFrame
            Full analysis table
        """
        result = self.analyze(dimensions, reference, min_n)
        
        df = result.gaps.copy()
        
        # Add confidence intervals
        z = 1.96
        df['ci_lower'] = df['gap_absolute'] - z * df['se']
        df['ci_upper'] = df['gap_absolute'] + z * df['se']
        
        # Add significance
        df['significant'] = ~((df['ci_lower'] <= 0) & (df['ci_upper'] >= 0))
        
        # Sort by gap magnitude
        df = df.sort_values('gap_percentage')
        
        return df


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def calculate_compound_gap(
    df: pd.DataFrame,
    dimensions: List[str],
    values: List[Any],
    references: List[Any],
    wage_col: str = 'REAL_HRLYEARN',
    weight_col: str = 'FINALWT'
) -> Dict[str, float]:
    """
    Quick calculation of compound gap.
    
    Parameters
    ----------
    df : DataFrame
        Data
    dimensions : List[str]
        Dimension columns
    values : List[Any]
        Values defining the target group
    references : List[Any]
        Values defining the reference group
    wage_col, weight_col : str
        Column names
        
    Returns
    -------
    Dict with 'gap_absolute', 'gap_percentage', 'n'
    """
    # Reference group
    ref_mask = pd.Series(True, index=df.index)
    for dim, ref in zip(dimensions, references):
        ref_mask &= (df[dim] == ref)
    
    # Target group
    target_mask = pd.Series(True, index=df.index)
    for dim, val in zip(dimensions, values):
        target_mask &= (df[dim] == val)
    
    # Add wage/weight filters
    valid_mask = df[wage_col].notna() & df[weight_col].notna()
    
    ref_df = df.loc[ref_mask & valid_mask]
    target_df = df.loc[target_mask & valid_mask]
    
    if len(ref_df) == 0 or len(target_df) == 0:
        return {'gap_absolute': np.nan, 'gap_percentage': np.nan, 'n': 0}
    
    ref_mean = weighted_mean(ref_df[wage_col].values, ref_df[weight_col].values)
    target_mean = weighted_mean(target_df[wage_col].values, target_df[weight_col].values)
    
    gap_abs = target_mean - ref_mean
    gap_pct = (gap_abs / ref_mean) * 100 if ref_mean != 0 else np.nan
    
    return {
        'gap_absolute': gap_abs,
        'gap_percentage': gap_pct,
        'n_reference': len(ref_df),
        'n_target': len(target_df),
        'mean_reference': ref_mean,
        'mean_target': target_mean,
    }
