"""
EquiPay Canada - Oaxaca-Blinder Decomposition
==============================================

Implementation of wage gap decomposition methods following:
- Oaxaca (1973) - Male Discrimination
- Blinder (1973) - Wage Differentials
- Neumark (1988) - Pooled Decomposition
- Cotton (1988) - Weighted Reference
- Fortin, Lemieux & Firpo (2011) - Review of Methods

The decomposition separates the wage gap into:
1. EXPLAINED (Endowments): Differences in characteristics
2. UNEXPLAINED (Discrimination): Differences in returns to characteristics

Threefold decomposition adds:
3. INTERACTION: Interaction between endowments and coefficients

References:
-----------
- Oaxaca, R. (1973). Male-Female Wage Differentials in Urban Labor Markets
- Blinder, A. (1973). Wage Discrimination: Reduced Form and Structural Estimates
- Jann, B. (2008). The Blinder-Oaxaca Decomposition for Linear Regression Models
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.regression.linear_model import WLS

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class DecompositionResult:
    """Results of Oaxaca-Blinder decomposition."""
    
    # Overall gap
    mean_difference: float      # Raw mean difference
    
    # Twofold decomposition
    explained: float            # Due to differences in characteristics
    unexplained: float          # Due to differences in returns (discrimination)
    
    # Threefold decomposition (optional)
    endowments: float = None    # Effect of X differences
    coefficients: float = None  # Effect of coefficient differences
    interaction: float = None   # Interaction effect
    
    # Standard errors
    se_explained: float = None
    se_unexplained: float = None
    
    # Detailed decomposition
    detailed_explained: Dict[str, float] = None
    detailed_unexplained: Dict[str, float] = None
    
    # Model statistics
    r2_group_a: float = None
    r2_group_b: float = None
    n_group_a: int = None
    n_group_b: int = None
    
    # Reference structure used
    reference: str = 'pooled'
    
    # Mean outcomes by group (for display)
    mean_a: float = None        # Mean outcome for group A (e.g., males)
    mean_b: float = None        # Mean outcome for group B (e.g., females)
    
    def get_shares(self) -> Dict[str, float]:
        """Get shares of total gap explained/unexplained."""
        total = abs(self.mean_difference) if self.mean_difference != 0 else 1
        return {
            'explained_share': self.explained / self.mean_difference * 100,
            'unexplained_share': self.unexplained / self.mean_difference * 100,
        }
    
    def __repr__(self):
        shares = self.get_shares()
        return (
            f"DecompositionResult(\n"
            f"  Total gap: {self.mean_difference:.4f}\n"
            f"  Explained: {self.explained:.4f} ({shares['explained_share']:.1f}%)\n"
            f"  Unexplained: {self.unexplained:.4f} ({shares['unexplained_share']:.1f}%)\n"
            f")"
        )


# =============================================================================
# OAXACA-BLINDER DECOMPOSITION
# =============================================================================

class OaxacaBlinderDecomposition:
    """
    Oaxaca-Blinder wage decomposition.
    
    Decomposes the wage gap between two groups into:
    - Explained component (differences in characteristics)
    - Unexplained component (differences in returns / discrimination)
    
    Parameters
    ----------
    reference : str
        Which group's coefficients to use as reference:
        - 'pooled': Pooled regression (Neumark 1988) [DEFAULT]
        - 'group_a': Reference group coefficients (Oaxaca 1973)
        - 'group_b': Comparison group coefficients (Blinder 1973)
        - 'cotton': Weighted average (Cotton 1988)
    
    Examples
    --------
    >>> decomp = OaxacaBlinderDecomposition(reference='pooled')
    >>> result = decomp.fit(X, y, group_indicator)
    >>> print(result)
    """
    
    def __init__(self, reference: str = 'pooled'):
        if reference not in ['pooled', 'group_a', 'group_b', 'cotton']:
            raise ValueError(f"Unknown reference: {reference}")
        self.reference = reference
        
        # Fitted models
        self.model_a_ = None
        self.model_b_ = None
        self.model_pooled_ = None
        
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        group_indicator: pd.Series,
        weights: pd.Series = None,
        group_a_value: Any = 0,
        group_b_value: Any = 1
    ) -> DecompositionResult:
        """
        Perform Oaxaca-Blinder decomposition.
        
        Parameters
        ----------
        X : DataFrame
            Feature matrix (should NOT include gender or wages)
        y : Series
            Log wages (or wages)
        group_indicator : Series
            Binary group indicator
        weights : Series, optional
            Survey weights
        group_a_value : Any
            Value indicating group A (reference)
        group_b_value : Any
            Value indicating group B (comparison)
            
        Returns
        -------
        DecompositionResult
        """
        # Align indices
        common_idx = X.index.intersection(y.index).intersection(group_indicator.index)
        if weights is not None:
            common_idx = common_idx.intersection(weights.index)
        
        X = X.loc[common_idx].copy()
        y = y.loc[common_idx].copy()
        group = group_indicator.loc[common_idx].copy()
        
        if weights is not None:
            weights = weights.loc[common_idx].copy()
        
        # Split by group
        mask_a = group == group_a_value
        mask_b = group == group_b_value
        
        X_a, y_a = X.loc[mask_a], y.loc[mask_a]
        X_b, y_b = X.loc[mask_b], y.loc[mask_b]
        
        w_a = weights.loc[mask_a] if weights is not None else None
        w_b = weights.loc[mask_b] if weights is not None else None
        
        # Add constant
        X_a_const = sm.add_constant(X_a, has_constant='add')
        X_b_const = sm.add_constant(X_b, has_constant='add')
        X_const = sm.add_constant(X, has_constant='add')
        
        # Fit separate models for each group
        if w_a is not None:
            model_a = WLS(y_a, X_a_const, weights=w_a).fit()
            model_b = WLS(y_b, X_b_const, weights=w_b).fit()
        else:
            model_a = sm.OLS(y_a, X_a_const).fit()
            model_b = sm.OLS(y_b, X_b_const).fit()
        
        self.model_a_ = model_a
        self.model_b_ = model_b
        
        # Fit pooled model (with group indicator)
        X_pooled = X_const.copy()
        X_pooled['_GROUP_'] = group.values
        
        if weights is not None:
            model_pooled = WLS(y, X_pooled, weights=weights).fit()
        else:
            model_pooled = sm.OLS(y, X_pooled).fit()
        
        self.model_pooled_ = model_pooled
        
        # Get coefficients
        beta_a = model_a.params
        beta_b = model_b.params
        
        # Get reference coefficients
        if self.reference == 'pooled':
            # Use pooled coefficients (excluding group dummy)
            beta_star = model_pooled.params.drop('_GROUP_', errors='ignore')
        elif self.reference == 'group_a':
            beta_star = beta_a
        elif self.reference == 'group_b':
            beta_star = beta_b
        else:  # cotton
            # Weight by group sizes
            n_a, n_b = len(y_a), len(y_b)
            omega = n_a / (n_a + n_b)
            beta_star = omega * beta_a + (1 - omega) * beta_b
        
        # Mean characteristics
        if w_a is not None:
            X_mean_a = np.average(X_a_const, weights=w_a, axis=0)
            X_mean_b = np.average(X_b_const, weights=w_b, axis=0)
            y_mean_a = np.average(y_a, weights=w_a)
            y_mean_b = np.average(y_b, weights=w_b)
        else:
            X_mean_a = X_a_const.mean().values
            X_mean_b = X_b_const.mean().values
            y_mean_a = y_a.mean()
            y_mean_b = y_b.mean()
        
        # Raw gap
        gap = y_mean_b - y_mean_a
        
        # Decomposition
        X_diff = X_mean_b - X_mean_a
        
        # Explained: (X_B - X_A) * beta*
        explained = np.dot(X_diff, beta_star)
        
        # Unexplained: gap - explained
        unexplained = gap - explained
        
        # Threefold decomposition
        endowments = np.dot(X_diff, beta_a.values)
        coefficients = np.dot(X_mean_a, (beta_b.values - beta_a.values))
        interaction = np.dot(X_diff, (beta_b.values - beta_a.values))
        
        # Detailed decomposition (contribution of each variable)
        detailed_explained = {}
        detailed_unexplained = {}
        
        for i, col in enumerate(X_a_const.columns):
            detailed_explained[col] = X_diff[i] * beta_star.iloc[i]
            detailed_unexplained[col] = X_mean_a[i] * (beta_b.iloc[i] - beta_a.iloc[i])
        
        return DecompositionResult(
            mean_difference=gap,
            explained=explained,
            unexplained=unexplained,
            endowments=endowments,
            coefficients=coefficients,
            interaction=interaction,
            detailed_explained=detailed_explained,
            detailed_unexplained=detailed_unexplained,
            r2_group_a=model_a.rsquared,
            r2_group_b=model_b.rsquared,
            n_group_a=len(y_a),
            n_group_b=len(y_b),
            reference=self.reference,
            mean_a=y_mean_a,
            mean_b=y_mean_b
        )
    
    def bootstrap_inference(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        group_indicator: pd.Series,
        weights: pd.Series = None,
        n_bootstrap: int = 500,
        random_state: int = 42
    ) -> Tuple[DecompositionResult, Dict[str, float]]:
        """
        Bootstrap standard errors for decomposition.
        
        Returns
        -------
        Tuple[DecompositionResult, Dict]
            (Point estimates, Standard errors)
        """
        np.random.seed(random_state)
        
        # Point estimate
        result = self.fit(X, y, group_indicator, weights)
        
        # Bootstrap
        explained_boots = []
        unexplained_boots = []
        
        n = len(X)
        for _ in range(n_bootstrap):
            idx = np.random.choice(n, size=n, replace=True)
            
            X_boot = X.iloc[idx]
            y_boot = y.iloc[idx]
            group_boot = group_indicator.iloc[idx]
            w_boot = weights.iloc[idx] if weights is not None else None
            
            try:
                r = self.fit(X_boot, y_boot, group_boot, w_boot)
                explained_boots.append(r.explained)
                unexplained_boots.append(r.unexplained)
            except Exception:
                continue
        
        se = {
            'se_explained': np.std(explained_boots),
            'se_unexplained': np.std(unexplained_boots),
        }
        
        result.se_explained = se['se_explained']
        result.se_unexplained = se['se_unexplained']
        
        return result, se


# =============================================================================
# THREEFOLD DECOMPOSITION
# =============================================================================

class ThreefoldDecomposition(OaxacaBlinderDecomposition):
    """
    Threefold Oaxaca-Blinder decomposition.
    
    Separates the gap into:
    1. Endowments (E): Differences in X with group A coefficients
    2. Coefficients (C): Differences in beta with group A means
    3. Interaction (I): Interaction between E and C
    
    Gap = E + C + I
    """
    
    def fit(self, *args, **kwargs) -> DecompositionResult:
        """Fit with emphasis on threefold components."""
        result = super().fit(*args, **kwargs)
        
        # Verify threefold adds up
        total = result.endowments + result.coefficients + result.interaction
        if not np.isclose(total, result.mean_difference, rtol=0.01):
            warnings.warn(
                f"Threefold components don't sum to gap: "
                f"{total:.4f} vs {result.mean_difference:.4f}"
            )
        
        return result


# =============================================================================
# DETAILED DECOMPOSITION
# =============================================================================

class DetailedDecomposition:
    """
    Detailed decomposition showing contribution of each variable.
    
    Provides variable-by-variable breakdown of the explained and
    unexplained components of the wage gap.
    """
    
    def __init__(self, reference: str = 'pooled'):
        self.decomp = OaxacaBlinderDecomposition(reference=reference)
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        group_indicator: pd.Series,
        weights: pd.Series = None
    ) -> pd.DataFrame:
        """
        Perform detailed decomposition.
        
        Returns
        -------
        DataFrame
            Variable-by-variable decomposition
        """
        result = self.decomp.fit(X, y, group_indicator, weights)
        
        # Create detailed table
        rows = []
        
        for var in result.detailed_explained.keys():
            rows.append({
                'variable': var,
                'explained': result.detailed_explained[var],
                'unexplained': result.detailed_unexplained.get(var, 0),
                'total': (result.detailed_explained[var] + 
                         result.detailed_unexplained.get(var, 0))
            })
        
        df = pd.DataFrame(rows)
        
        # Add shares
        total_explained = result.explained
        total_unexplained = result.unexplained
        
        df['explained_share'] = df['explained'] / total_explained * 100
        df['unexplained_share'] = df['unexplained'] / total_unexplained * 100
        
        # Sort by absolute explained contribution
        df = df.reindex(df['explained'].abs().sort_values(ascending=False).index)
        
        return df
    
    def summary(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        group_indicator: pd.Series,
        weights: pd.Series = None,
        top_n: int = 10
    ) -> str:
        """
        Get text summary of detailed decomposition.
        
        Parameters
        ----------
        top_n : int
            Number of top variables to show
            
        Returns
        -------
        str
            Formatted summary
        """
        result = self.decomp.fit(X, y, group_indicator, weights)
        df = self.fit(X, y, group_indicator, weights)
        
        lines = [
            "=" * 60,
            "DETAILED OAXACA-BLINDER DECOMPOSITION",
            "=" * 60,
            "",
            f"Total wage gap: {result.mean_difference:.4f}",
            f"  Explained: {result.explained:.4f} ({result.explained/result.mean_difference*100:.1f}%)",
            f"  Unexplained: {result.unexplained:.4f} ({result.unexplained/result.mean_difference*100:.1f}%)",
            "",
            f"Top {top_n} variables (by explained contribution):",
            "-" * 60,
        ]
        
        for _, row in df.head(top_n).iterrows():
            lines.append(
                f"  {row['variable']:30s} "
                f"E={row['explained']:+.4f} ({row['explained_share']:+.1f}%) "
                f"U={row['unexplained']:+.4f}"
            )
        
        return "\n".join(lines)
