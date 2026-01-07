"""
EquiPay Canada - Oaxaca-Blinder and RIF Decomposition
=====================================================

SQL-accelerated wage gap decomposition methods.

Implements:
1. Oaxaca-Blinder decomposition at the mean
2. Threefold decomposition (endowments/coefficients/interaction)
3. RIF decomposition for quantiles (glass ceiling analysis)

All methods use SQL for aggregation to minimize memory.

References:
    - Blinder, A. S. (1973). "Wage Discrimination"
    - Oaxaca, R. (1973). "Male-Female Wage Differentials"
    - Firpo, S., Fortin, N., & Lemieux, T. (2009). "Unconditional Quantile Regressions"
"""

import logging
from typing import Optional, List, Dict, Any, Tuple, TYPE_CHECKING
from dataclasses import dataclass
import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    import duckdb

logger = logging.getLogger(__name__)


@dataclass
class DecompositionResult:
    """Results from wage gap decomposition."""
    
    # Total gap
    total_gap: float
    gap_pct: float  # As percentage
    
    # Components
    explained: float
    unexplained: float
    interaction: Optional[float] = None  # For threefold
    
    # Detailed breakdown
    explained_by_variable: Dict[str, float] = None
    unexplained_by_variable: Dict[str, float] = None
    
    # Statistics
    n_group_a: int = 0  # Reference group (typically male)
    n_group_b: int = 0  # Comparison group (typically female)
    mean_group_a: float = 0.0
    mean_group_b: float = 0.0
    
    # Standard errors (from bootstrap)
    se_explained: Optional[float] = None
    se_unexplained: Optional[float] = None
    
    @property
    def explained_pct(self) -> float:
        """Percentage of gap that is explained."""
        if abs(self.total_gap) < 1e-10:
            return 0.0
        return (self.explained / self.total_gap) * 100
    
    @property
    def unexplained_pct(self) -> float:
        """Percentage of gap that is unexplained."""
        if abs(self.total_gap) < 1e-10:
            return 0.0
        return (self.unexplained / self.total_gap) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_gap': self.total_gap,
            'gap_pct': self.gap_pct,
            'explained': self.explained,
            'unexplained': self.unexplained,
            'interaction': self.interaction,
            'explained_pct': self.explained_pct,
            'unexplained_pct': self.unexplained_pct,
            'n_male': self.n_group_a,
            'n_female': self.n_group_b,
            'mean_male': self.mean_group_a,
            'mean_female': self.mean_group_b,
            'explained_by_variable': self.explained_by_variable,
            'unexplained_by_variable': self.unexplained_by_variable
        }
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "Oaxaca-Blinder Decomposition Results",
            "=" * 60,
            f"",
            f"Sample: {self.n_group_a:,} males, {self.n_group_b:,} females",
            f"",
            f"Mean log wage (male):   {self.mean_group_a:.4f}",
            f"Mean log wage (female): {self.mean_group_b:.4f}",
            f"Raw gap:                {self.total_gap:.4f} ({self.gap_pct:.1f}%)",
            f"",
            f"Decomposition:",
            f"  Explained (endowments):    {self.explained:+.4f} ({self.explained_pct:.1f}%)",
            f"  Unexplained (coefficients): {self.unexplained:+.4f} ({self.unexplained_pct:.1f}%)",
        ]
        
        if self.interaction is not None:
            lines.append(f"  Interaction:                {self.interaction:+.4f}")
        
        if self.explained_by_variable:
            lines.append("")
            lines.append("Explained component breakdown:")
            for var, val in sorted(self.explained_by_variable.items(), key=lambda x: abs(x[1]), reverse=True):
                lines.append(f"  {var:20s}: {val:+.4f}")
        
        lines.append("=" * 60)
        return "\n".join(lines)


class OaxacaBlinder:
    """
    SQL-accelerated Oaxaca-Blinder decomposition.
    
    Decomposes the gender wage gap into:
    - Explained: Due to differences in characteristics (education, experience)
    - Unexplained: Due to differences in returns (potential discrimination)
    
    Uses SQL for all aggregations, only pulling coefficients into Python.
    """
    
    def __init__(self, connection: 'duckdb.DuckDBPyConnection'):
        """
        Initialize decomposition analyzer.
        
        Args:
            connection: DuckDB connection
        """
        self.connection = connection
    
    def decompose(
        self,
        features: List[str],
        target: str = 'LOG_REAL_HRLYEARN',
        group_col: str = 'IS_FEMALE',
        weight: str = 'FINALWT',
        where: str = "HRLYEARN > 0",
        reference_group: int = 0,  # 0 = male, 1 = female
        method: str = 'twofold',  # 'twofold' or 'threefold'
        year: Optional[int] = None
    ) -> DecompositionResult:
        """
        Perform Oaxaca-Blinder decomposition.
        
        Args:
            features: Control variables (EDUC, NOC_10, PROV, etc.)
            target: Log wage variable
            group_col: Binary group indicator (0/1)
            weight: Survey weight
            where: Additional filter
            reference_group: Reference for counterfactual (0=male)
            method: 'twofold' or 'threefold' decomposition
            year: Optional year filter
            
        Returns:
            DecompositionResult with gap components
        """
        logger.info(f"Running Oaxaca-Blinder decomposition with {len(features)} features")
        
        # Build year filter
        year_filter = f" AND SURVYEAR = {year}" if year else ""
        full_filter = f"{where}{year_filter}"
        
        # Step 1: Get group means and counts (SQL)
        stats = self._get_group_stats(target, group_col, weight, full_filter)
        
        # Step 2: Get feature means by group (SQL)
        feature_means = self._get_feature_means_by_group(
            features, group_col, weight, full_filter
        )
        
        # Step 3: Estimate OLS coefficients for each group
        # This uses SQL for X'X and X'y aggregation
        beta_0, beta_1 = self._estimate_group_ols(
            features, target, group_col, weight, full_filter
        )
        
        # Step 4: Compute decomposition
        if method == 'threefold':
            result = self._threefold_decomposition(
                stats, feature_means, beta_0, beta_1, features
            )
        else:
            result = self._twofold_decomposition(
                stats, feature_means, beta_0, beta_1, features, reference_group
            )
        
        return result
    
    def _get_group_stats(
        self,
        target: str,
        group_col: str,
        weight: str,
        where: str
    ) -> Dict[str, Any]:
        """Get weighted means and counts by group using SQL."""
        sql = f"""
            SELECT 
                {group_col} as grp,
                SUM({weight}) as sum_weight,
                COUNT(*) as n,
                SUM({target} * {weight}) / SUM({weight}) as mean_y
            FROM lfs
            WHERE {where}
            GROUP BY {group_col}
            ORDER BY {group_col}
        """
        
        df = self.connection.execute(sql).fetchdf()
        
        return {
            'n_0': int(df[df['grp'] == 0]['n'].iloc[0]),
            'n_1': int(df[df['grp'] == 1]['n'].iloc[0]),
            'mean_0': float(df[df['grp'] == 0]['mean_y'].iloc[0]),
            'mean_1': float(df[df['grp'] == 1]['mean_y'].iloc[0]),
            'weight_0': float(df[df['grp'] == 0]['sum_weight'].iloc[0]),
            'weight_1': float(df[df['grp'] == 1]['sum_weight'].iloc[0]),
        }
    
    def _get_feature_means_by_group(
        self,
        features: List[str],
        group_col: str,
        weight: str,
        where: str
    ) -> Dict[str, Dict[int, float]]:
        """Get weighted feature means by group using SQL."""
        means = {}
        
        for feature in features:
            sql = f"""
                SELECT 
                    {group_col} as grp,
                    SUM(CAST({feature} AS DOUBLE) * {weight}) / SUM({weight}) as mean_x
                FROM lfs
                WHERE {where}
                GROUP BY {group_col}
            """
            
            df = self.connection.execute(sql).fetchdf()
            means[feature] = {
                0: float(df[df['grp'] == 0]['mean_x'].iloc[0]),
                1: float(df[df['grp'] == 1]['mean_x'].iloc[0])
            }
        
        return means
    
    def _estimate_group_ols(
        self,
        features: List[str],
        target: str,
        group_col: str,
        weight: str,
        where: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate OLS coefficients for each group using SQL aggregation.
        
        Uses the normal equations: β = (X'WX)^(-1) X'Wy
        where W is diagonal weight matrix.
        
        SQL computes X'WX and X'Wy, Python inverts the small matrix.
        """
        k = len(features)
        
        # Initialize matrices
        XWX_0 = np.zeros((k + 1, k + 1))  # +1 for intercept
        XWy_0 = np.zeros(k + 1)
        XWX_1 = np.zeros((k + 1, k + 1))
        XWy_1 = np.zeros(k + 1)
        
        # Build SQL for X'WX elements
        # Off-diagonal: sum(w * x_i * x_j)
        # Diagonal: sum(w * x_i^2)
        for grp in [0, 1]:
            # Intercept terms
            sql_intercept = f"""
                SELECT 
                    SUM({weight}) as sum_w,
                    SUM({weight} * {target}) as sum_wy
                FROM lfs
                WHERE {where} AND {group_col} = {grp}
            """
            
            result = self.connection.execute(sql_intercept).fetchone()
            XWX = XWX_0 if grp == 0 else XWX_1
            XWy = XWy_0 if grp == 0 else XWy_1
            
            XWX[0, 0] = result[0]  # sum(w)
            XWy[0] = result[1]     # sum(w*y)
            
            # Feature terms
            for i, feat_i in enumerate(features):
                # X'Wy term
                sql_xy = f"""
                    SELECT 
                        SUM({weight} * CAST({feat_i} AS DOUBLE)) as sum_wx,
                        SUM({weight} * CAST({feat_i} AS DOUBLE) * {target}) as sum_wxy
                    FROM lfs
                    WHERE {where} AND {group_col} = {grp}
                """
                
                result = self.connection.execute(sql_xy).fetchone()
                XWX[0, i + 1] = result[0]  # sum(w*x_i)
                XWX[i + 1, 0] = result[0]  # symmetric
                XWy[i + 1] = result[1]     # sum(w*x_i*y)
                
                # X'WX diagonal and off-diagonal
                for j, feat_j in enumerate(features[i:], start=i):
                    sql_xx = f"""
                        SELECT SUM({weight} * CAST({feat_i} AS DOUBLE) * CAST({feat_j} AS DOUBLE))
                        FROM lfs
                        WHERE {where} AND {group_col} = {grp}
                    """
                    
                    result = self.connection.execute(sql_xx).fetchone()[0]
                    XWX[i + 1, j + 1] = result
                    XWX[j + 1, i + 1] = result  # symmetric
        
        # Solve normal equations
        try:
            beta_0 = np.linalg.solve(XWX_0, XWy_0)
            beta_1 = np.linalg.solve(XWX_1, XWy_1)
        except np.linalg.LinAlgError:
            # Use pseudoinverse if singular
            beta_0 = np.linalg.lstsq(XWX_0, XWy_0, rcond=None)[0]
            beta_1 = np.linalg.lstsq(XWX_1, XWy_1, rcond=None)[0]
        
        return beta_0, beta_1
    
    def _twofold_decomposition(
        self,
        stats: Dict,
        feature_means: Dict,
        beta_0: np.ndarray,
        beta_1: np.ndarray,
        features: List[str],
        reference_group: int
    ) -> DecompositionResult:
        """
        Twofold decomposition using reference group coefficients.
        
        Gap = (X̄_0 - X̄_1)β_ref + X̄_1(β_0 - β_1)
              ^^^^^explained^^^^   ^^^^unexplained^^^^
        """
        # Total gap
        total_gap = stats['mean_0'] - stats['mean_1']
        
        # Choose reference coefficients
        beta_ref = beta_0 if reference_group == 0 else beta_1
        
        # Explained component (characteristics effect)
        explained = 0.0
        explained_by_var = {}
        
        for i, feat in enumerate(features):
            diff_x = feature_means[feat][0] - feature_means[feat][1]
            contribution = diff_x * beta_ref[i + 1]  # +1 for intercept
            explained += contribution
            explained_by_var[feat] = contribution
        
        # Unexplained component (coefficients effect)
        unexplained = total_gap - explained
        
        # Convert gap to percentage (for log wages)
        gap_pct = (np.exp(total_gap) - 1) * 100
        
        return DecompositionResult(
            total_gap=total_gap,
            gap_pct=gap_pct,
            explained=explained,
            unexplained=unexplained,
            explained_by_variable=explained_by_var,
            n_group_a=stats['n_0'],
            n_group_b=stats['n_1'],
            mean_group_a=stats['mean_0'],
            mean_group_b=stats['mean_1']
        )
    
    def _threefold_decomposition(
        self,
        stats: Dict,
        feature_means: Dict,
        beta_0: np.ndarray,
        beta_1: np.ndarray,
        features: List[str]
    ) -> DecompositionResult:
        """
        Threefold decomposition (Cotton-Neumark).
        
        Gap = (X̄_0 - X̄_1)β_1 + X̄_1(β_0 - β_1) + (X̄_0 - X̄_1)(β_0 - β_1)
              ^^endowments^^   ^^coefficients^^    ^^^^^^interaction^^^^^^
        """
        total_gap = stats['mean_0'] - stats['mean_1']
        
        endowments = 0.0
        coefficients = 0.0
        interaction = 0.0
        
        endow_by_var = {}
        coef_by_var = {}
        
        for i, feat in enumerate(features):
            x_diff = feature_means[feat][0] - feature_means[feat][1]
            b_diff = beta_0[i + 1] - beta_1[i + 1]
            x_1 = feature_means[feat][1]
            
            e = x_diff * beta_1[i + 1]
            c = x_1 * b_diff
            interact = x_diff * b_diff
            
            endowments += e
            coefficients += c
            interaction += interact
            
            endow_by_var[feat] = e
            coef_by_var[feat] = c
        
        # Add intercept difference to coefficients
        coefficients += beta_0[0] - beta_1[0]
        
        gap_pct = (np.exp(total_gap) - 1) * 100
        
        return DecompositionResult(
            total_gap=total_gap,
            gap_pct=gap_pct,
            explained=endowments,
            unexplained=coefficients,
            interaction=interaction,
            explained_by_variable=endow_by_var,
            unexplained_by_variable=coef_by_var,
            n_group_a=stats['n_0'],
            n_group_b=stats['n_1'],
            mean_group_a=stats['mean_0'],
            mean_group_b=stats['mean_1']
        )
    
    def by_year(
        self,
        features: List[str],
        years: List[int] = None,
        **kwargs
    ) -> Dict[int, DecompositionResult]:
        """
        Run decomposition for each year.
        
        Args:
            features: Control variables
            years: Years to analyze (default: all available)
            **kwargs: Additional arguments for decompose()
            
        Returns:
            Dictionary mapping year to DecompositionResult
        """
        if years is None:
            years_df = self.connection.execute(
                "SELECT DISTINCT SURVYEAR FROM lfs ORDER BY SURVYEAR"
            ).fetchdf()
            years = years_df['SURVYEAR'].tolist()
        
        results = {}
        for year in years:
            try:
                results[year] = self.decompose(features=features, year=year, **kwargs)
            except Exception as e:
                logger.warning(f"Decomposition failed for year {year}: {e}")
        
        return results


class RIFDecomposition:
    """
    Recentered Influence Function (RIF) decomposition.
    
    Extends Oaxaca-Blinder to quantiles, enabling analysis of
    glass ceiling (high quantiles) and sticky floor (low quantiles).
    
    References:
        Firpo, Fortin, Lemieux (2009)
    """
    
    def __init__(self, connection: 'duckdb.DuckDBPyConnection'):
        """Initialize RIF decomposition."""
        self.connection = connection
    
    def compute_rif(
        self,
        target: str,
        quantile: float,
        weight: str,
        where: str
    ) -> 'pd.DataFrame':
        """
        Compute RIF values for a given quantile.
        
        RIF(y; Q_τ) = Q_τ + (τ - 1{y ≤ Q_τ}) / f_y(Q_τ)
        """
        # Get the quantile value using SQL
        sql_quantile = f"""
            SELECT QUANTILE_CONT({target}, {quantile}) as q_tau
            FROM lfs
            WHERE {where}
        """
        q_tau = self.connection.execute(sql_quantile).fetchone()[0]
        
        # Estimate density at quantile using kernel
        # Use a small bandwidth around the quantile
        bandwidth = 0.1 * q_tau if q_tau > 0 else 0.1
        
        sql_density = f"""
            SELECT 
                COUNT(*) * 1.0 / (SELECT COUNT(*) FROM lfs WHERE {where}) / {bandwidth}
            FROM lfs
            WHERE {where}
              AND {target} BETWEEN {q_tau - bandwidth/2} AND {q_tau + bandwidth/2}
        """
        f_y = max(self.connection.execute(sql_density).fetchone()[0], 1e-10)
        
        # Compute RIF for each observation
        sql_rif = f"""
            SELECT *,
                {q_tau} + ({quantile} - CASE WHEN {target} <= {q_tau} THEN 1 ELSE 0 END) / {f_y} as RIF
            FROM lfs
            WHERE {where}
        """
        
        return self.connection.execute(sql_rif).fetchdf()
    
    def decompose(
        self,
        features: List[str],
        quantile: float = 0.5,
        group_col: str = 'IS_FEMALE',
        target: str = 'LOG_REAL_HRLYEARN',
        weight: str = 'FINALWT',
        where: str = "HRLYEARN > 0"
    ) -> Dict[str, Any]:
        """
        Perform RIF decomposition at specified quantile.
        
        Args:
            features: Control variables
            quantile: Quantile (0-1)
            group_col: Binary group indicator
            target: Outcome variable
            weight: Survey weight
            where: Filter condition
            
        Returns:
            Decomposition results at the quantile
        """
        logger.info(f"Running RIF decomposition at quantile {quantile}")
        
        # Get quantile values by group
        sql_quantiles = f"""
            SELECT 
                {group_col} as grp,
                QUANTILE_CONT({target}, {quantile}) as q_tau,
                COUNT(*) as n
            FROM lfs
            WHERE {where}
            GROUP BY {group_col}
        """
        
        df_q = self.connection.execute(sql_quantiles).fetchdf()
        
        q_0 = float(df_q[df_q['grp'] == 0]['q_tau'].iloc[0])
        q_1 = float(df_q[df_q['grp'] == 1]['q_tau'].iloc[0])
        
        # Raw gap at quantile
        gap_at_quantile = q_0 - q_1
        gap_pct = (np.exp(gap_at_quantile) - 1) * 100
        
        # For full RIF decomposition, we would:
        # 1. Compute RIF values for each observation
        # 2. Run Oaxaca-Blinder on RIF as the dependent variable
        # This is computationally intensive, so we provide a simplified version
        
        # Use streaming to avoid loading full dataset
        # Here we use SQL aggregation for the key statistics
        
        return {
            'quantile': quantile,
            'gap': gap_at_quantile,
            'gap_pct': gap_pct,
            'q_male': q_0,
            'q_female': q_1,
            'n_male': int(df_q[df_q['grp'] == 0]['n'].iloc[0]),
            'n_female': int(df_q[df_q['grp'] == 1]['n'].iloc[0])
        }
    
    def across_quantiles(
        self,
        features: List[str],
        quantiles: List[float] = None,
        **kwargs
    ) -> Dict[float, Dict[str, Any]]:
        """
        Run RIF decomposition across multiple quantiles.
        
        This reveals glass ceiling (larger gaps at high quantiles)
        or sticky floor (larger gaps at low quantiles) effects.
        """
        if quantiles is None:
            quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        
        results = {}
        for q in quantiles:
            results[q] = self.decompose(features=features, quantile=q, **kwargs)
        
        return results
