"""
EquiPay Canada - Quantile Gap Analysis
======================================

SQL-accelerated quantile analysis for pay equity.

Implements:
1. Gender gap at multiple quantiles
2. Glass ceiling detection (90th percentile gap)
3. Sticky floor detection (10th percentile gap)
4. Conditional quantile analysis by occupation/education

All methods use SQL quantile functions for efficiency.

References:
    - Albrecht, Björklund, Vroman (2003) "Is There a Glass Ceiling in Sweden?"
    - Arulampalam, Booth, Bryan (2007) "Is There a Glass Ceiling over Europe?"
"""

import logging
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass
import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    import duckdb

logger = logging.getLogger(__name__)


@dataclass
class QuantileGapResult:
    """Results from quantile gap analysis."""
    quantile: float
    male_value: float
    female_value: float
    gap: float
    gap_pct: float
    n_male: int
    n_female: int
    
    @property
    def is_glass_ceiling(self) -> bool:
        """Check if this indicates glass ceiling (high quantile, large gap)."""
        return self.quantile >= 0.75 and self.gap_pct > 15
    
    @property
    def is_sticky_floor(self) -> bool:
        """Check if this indicates sticky floor (low quantile, large gap)."""
        return self.quantile <= 0.25 and self.gap_pct > 15


@dataclass 
class GlassCeilingResult:
    """Comprehensive glass ceiling analysis results."""
    
    # Gap at different quantiles
    gap_p10: float
    gap_p25: float
    gap_p50: float
    gap_p75: float
    gap_p90: float
    
    # Pattern detection
    has_glass_ceiling: bool
    has_sticky_floor: bool
    ceiling_severity: float  # p90 gap / median gap
    floor_severity: float    # p10 gap / median gap
    
    # Statistical tests
    ceiling_test_stat: Optional[float] = None
    ceiling_p_value: Optional[float] = None
    
    def summary(self) -> str:
        """Generate summary of glass ceiling analysis."""
        lines = [
            "=" * 60,
            "Glass Ceiling / Sticky Floor Analysis",
            "=" * 60,
            "",
            "Gender Gap by Quantile (log wage difference):",
            f"  P10 (bottom):  {self.gap_p10:+.4f} ({(np.exp(self.gap_p10)-1)*100:+.1f}%)",
            f"  P25:           {self.gap_p25:+.4f} ({(np.exp(self.gap_p25)-1)*100:+.1f}%)",
            f"  P50 (median):  {self.gap_p50:+.4f} ({(np.exp(self.gap_p50)-1)*100:+.1f}%)",
            f"  P75:           {self.gap_p75:+.4f} ({(np.exp(self.gap_p75)-1)*100:+.1f}%)",
            f"  P90 (top):     {self.gap_p90:+.4f} ({(np.exp(self.gap_p90)-1)*100:+.1f}%)",
            "",
            "Pattern Detection:",
        ]
        
        if self.has_glass_ceiling:
            lines.append(f"  ⚠️  GLASS CEILING DETECTED (severity: {self.ceiling_severity:.2f}x median)")
        else:
            lines.append("  ✓ No glass ceiling pattern")
        
        if self.has_sticky_floor:
            lines.append(f"  ⚠️  STICKY FLOOR DETECTED (severity: {self.floor_severity:.2f}x median)")
        else:
            lines.append("  ✓ No sticky floor pattern")
        
        lines.append("=" * 60)
        return "\n".join(lines)


class QuantileGapAnalyzer:
    """
    Analyze gender pay gap across the wage distribution.
    
    Uses SQL QUANTILE_CONT for all computations.
    """
    
    # Standard quantiles to analyze
    STANDARD_QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]
    
    def __init__(self, connection: 'duckdb.DuckDBPyConnection'):
        """Initialize analyzer."""
        self.connection = connection
    
    def gap_at_quantile(
        self,
        quantile: float,
        target: str = 'LOG_REAL_HRLYEARN',
        group_col: str = 'IS_FEMALE',
        weight: str = 'FINALWT',
        where: str = "HRLYEARN > 0",
        year: Optional[int] = None
    ) -> QuantileGapResult:
        """
        Compute gender gap at a specific quantile using SQL.
        
        Args:
            quantile: Quantile (0-1)
            target: Wage variable
            group_col: Gender indicator (0=male, 1=female)
            weight: Survey weight (not used for unweighted quantile)
            where: Filter condition
            year: Optional year filter
            
        Returns:
            QuantileGapResult with gap at quantile
        """
        year_filter = f" AND SURVYEAR = {year}" if year else ""
        
        # Use SQL QUANTILE_CONT for efficient computation
        sql = f"""
            SELECT 
                {group_col} as grp,
                QUANTILE_CONT({target}, {quantile}) as q_value,
                COUNT(*) as n
            FROM lfs
            WHERE {where}{year_filter}
            GROUP BY {group_col}
            ORDER BY {group_col}
        """
        
        df = self.connection.execute(sql).fetchdf()
        
        male_val = float(df[df['grp'] == 0]['q_value'].iloc[0])
        female_val = float(df[df['grp'] == 1]['q_value'].iloc[0])
        gap = male_val - female_val
        
        return QuantileGapResult(
            quantile=quantile,
            male_value=male_val,
            female_value=female_val,
            gap=gap,
            gap_pct=(np.exp(gap) - 1) * 100,
            n_male=int(df[df['grp'] == 0]['n'].iloc[0]),
            n_female=int(df[df['grp'] == 1]['n'].iloc[0])
        )
    
    def gap_by_quantile(
        self,
        quantiles: List[float] = None,
        by_group: Optional[str] = None,
        year: Optional[int] = None,
        **kwargs
    ) -> Dict[float, Any]:
        """
        Compute gender gap at multiple quantiles.
        
        Args:
            quantiles: List of quantiles (default: deciles)
            by_group: Optional grouping (PROV, NOC_10, EDUC)
            year: Optional year filter
            
        Returns:
            Dictionary mapping quantiles to results
        """
        if quantiles is None:
            quantiles = self.STANDARD_QUANTILES
        
        if by_group:
            return self._gap_by_quantile_grouped(quantiles, by_group, year, **kwargs)
        
        results = {}
        for q in quantiles:
            results[q] = self.gap_at_quantile(quantile=q, year=year, **kwargs)
        
        return results
    
    def _gap_by_quantile_grouped(
        self,
        quantiles: List[float],
        group_by: str,
        year: Optional[int],
        target: str = 'LOG_REAL_HRLYEARN',
        where: str = "HRLYEARN > 0"
    ) -> Dict[Any, Dict[float, QuantileGapResult]]:
        """Compute gaps by quantile for each group value."""
        year_filter = f" AND SURVYEAR = {year}" if year else ""
        
        # Get distinct group values
        groups_sql = f"""
            SELECT DISTINCT {group_by}
            FROM lfs
            WHERE {where}{year_filter}
            ORDER BY {group_by}
        """
        groups = self.connection.execute(groups_sql).fetchdf()[group_by].tolist()
        
        results = {}
        for grp_val in groups:
            grp_filter = f"{where}{year_filter} AND {group_by} = {grp_val}"
            
            results[grp_val] = {}
            for q in quantiles:
                try:
                    results[grp_val][q] = self.gap_at_quantile(
                        quantile=q,
                        target=target,
                        where=grp_filter
                    )
                except Exception as e:
                    logger.warning(f"Failed for group {grp_val}, quantile {q}: {e}")
        
        return results
    
    def glass_ceiling_analysis(
        self,
        year: Optional[int] = None,
        ceiling_threshold: float = 1.2,  # Gap at p90 > 1.2x median gap
        floor_threshold: float = 1.2,
        **kwargs
    ) -> GlassCeilingResult:
        """
        Comprehensive glass ceiling and sticky floor analysis.
        
        A glass ceiling exists when the gap is larger at higher quantiles,
        suggesting women face barriers to reaching top positions/wages.
        
        A sticky floor exists when the gap is larger at lower quantiles,
        suggesting women are stuck at the bottom of the distribution.
        """
        gaps = self.gap_by_quantile(
            quantiles=[0.10, 0.25, 0.50, 0.75, 0.90],
            year=year,
            **kwargs
        )
        
        gap_p10 = gaps[0.10].gap
        gap_p25 = gaps[0.25].gap
        gap_p50 = gaps[0.50].gap
        gap_p75 = gaps[0.75].gap
        gap_p90 = gaps[0.90].gap
        
        # Avoid division by zero
        median_gap = gap_p50 if abs(gap_p50) > 0.01 else 0.01
        
        ceiling_severity = gap_p90 / median_gap if median_gap > 0 else 0
        floor_severity = gap_p10 / median_gap if median_gap > 0 else 0
        
        return GlassCeilingResult(
            gap_p10=gap_p10,
            gap_p25=gap_p25,
            gap_p50=gap_p50,
            gap_p75=gap_p75,
            gap_p90=gap_p90,
            has_glass_ceiling=ceiling_severity > ceiling_threshold,
            has_sticky_floor=floor_severity > floor_threshold,
            ceiling_severity=ceiling_severity,
            floor_severity=floor_severity
        )
    
    def quantile_gap_over_time(
        self,
        quantile: float = 0.5,
        years: List[int] = None,
        **kwargs
    ) -> Dict[int, QuantileGapResult]:
        """
        Track quantile gap over time.
        
        Args:
            quantile: Quantile to track
            years: Years to analyze (default: all)
            
        Returns:
            Dictionary mapping year to gap result
        """
        if years is None:
            years_df = self.connection.execute(
                "SELECT DISTINCT SURVYEAR FROM lfs ORDER BY SURVYEAR"
            ).fetchdf()
            years = years_df['SURVYEAR'].tolist()
        
        results = {}
        for year in years:
            try:
                results[year] = self.gap_at_quantile(quantile=quantile, year=year, **kwargs)
            except Exception as e:
                logger.warning(f"Failed for year {year}: {e}")
        
        return results


class GlassCeilingAnalyzer:
    """
    Specialized analyzer for glass ceiling effects.
    
    Provides additional methods for understanding barriers
    at the top of the wage distribution.
    """
    
    def __init__(self, connection: 'duckdb.DuckDBPyConnection'):
        """Initialize analyzer."""
        self.connection = connection
        self.quantile_analyzer = QuantileGapAnalyzer(connection)
    
    def analyze(
        self,
        years: List[int] = None,
        by_occupation: bool = True,
        by_education: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Comprehensive glass ceiling analysis.
        
        Returns:
            Dictionary with overall results and breakdowns
        """
        results = {'overall': {}}
        
        # Get years
        if years is None:
            years_df = self.connection.execute(
                "SELECT DISTINCT SURVYEAR FROM lfs ORDER BY SURVYEAR"
            ).fetchdf()
            years = years_df['SURVYEAR'].tolist()
        
        # Overall analysis by year
        for year in years:
            results['overall'][year] = self.quantile_analyzer.glass_ceiling_analysis(
                year=year, **kwargs
            )
        
        # By occupation (NOC_10 - broad categories)
        if by_occupation:
            results['by_occupation'] = {}
            for year in [years[-1]]:  # Most recent year only to save time
                results['by_occupation'][year] = self._analyze_by_group(
                    'NOC_10', year, **kwargs
                )
        
        # By education
        if by_education:
            results['by_education'] = {}
            for year in [years[-1]]:
                results['by_education'][year] = self._analyze_by_group(
                    'EDUC', year, **kwargs
                )
        
        return results
    
    def _analyze_by_group(
        self,
        group_col: str,
        year: int,
        **kwargs
    ) -> Dict[Any, GlassCeilingResult]:
        """Analyze glass ceiling by group."""
        where = kwargs.get('where', "HRLYEARN > 0")
        
        # Get distinct groups with sufficient sample
        sql = f"""
            SELECT {group_col}, COUNT(*) as n
            FROM lfs
            WHERE {where} AND SURVYEAR = {year}
            GROUP BY {group_col}
            HAVING COUNT(*) >= 500
        """
        
        groups = self.connection.execute(sql).fetchdf()
        
        results = {}
        for _, row in groups.iterrows():
            grp_val = row[group_col]
            grp_filter = f"{where} AND {group_col} = {grp_val}"
            
            try:
                results[grp_val] = self.quantile_analyzer.glass_ceiling_analysis(
                    year=year,
                    where=grp_filter,
                    **{k: v for k, v in kwargs.items() if k != 'where'}
                )
            except Exception as e:
                logger.warning(f"Glass ceiling analysis failed for {group_col}={grp_val}: {e}")
        
        return results
    
    def representation_at_top(
        self,
        top_quantile: float = 0.90,
        year: Optional[int] = None,
        by_group: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze female representation at top of wage distribution.
        
        Returns the share of women among top earners.
        """
        year_filter = f" AND SURVYEAR = {year}" if year else ""
        
        # Get threshold for top quantile
        sql_threshold = f"""
            SELECT QUANTILE_CONT(LOG_REAL_HRLYEARN, {top_quantile}) as threshold
            FROM lfs
            WHERE HRLYEARN > 0{year_filter}
        """
        threshold = self.connection.execute(sql_threshold).fetchone()[0]
        
        # Count by gender at top
        sql_top = f"""
            SELECT 
                IS_FEMALE,
                COUNT(*) as n,
                SUM(FINALWT) as weighted_n
            FROM lfs
            WHERE HRLYEARN > 0{year_filter}
              AND LOG_REAL_HRLYEARN >= {threshold}
            GROUP BY IS_FEMALE
        """
        
        df = self.connection.execute(sql_top).fetchdf()
        
        total = df['weighted_n'].sum()
        female_share = float(df[df['IS_FEMALE'] == 1]['weighted_n'].iloc[0]) / total
        
        # Compare to overall female share
        sql_overall = f"""
            SELECT 
                SUM(CASE WHEN IS_FEMALE = 1 THEN FINALWT ELSE 0 END) * 1.0 / SUM(FINALWT)
            FROM lfs
            WHERE HRLYEARN > 0{year_filter}
        """
        overall_female_share = self.connection.execute(sql_overall).fetchone()[0]
        
        return {
            'top_quantile': top_quantile,
            'threshold': threshold,
            'female_share_at_top': female_share,
            'overall_female_share': overall_female_share,
            'representation_ratio': female_share / overall_female_share if overall_female_share > 0 else 0,
            'underrepresentation': overall_female_share - female_share
        }
    
    def progression_barriers(
        self,
        year: Optional[int] = None,
        quantiles: List[float] = None
    ) -> Dict[str, Any]:
        """
        Analyze barriers to wage progression for women.
        
        Computes the probability of being in each quantile
        conditional on gender and characteristics.
        """
        if quantiles is None:
            quantiles = [0.25, 0.50, 0.75, 0.90]
        
        year_filter = f" AND SURVYEAR = {year}" if year else ""
        
        # Compute quantile thresholds
        thresholds = []
        for q in quantiles:
            sql = f"""
                SELECT QUANTILE_CONT(LOG_REAL_HRLYEARN, {q})
                FROM lfs WHERE HRLYEARN > 0{year_filter}
            """
            thresholds.append(self.connection.execute(sql).fetchone()[0])
        
        results = {'quantiles': quantiles, 'thresholds': thresholds, 'by_gender': {}}
        
        for gender in [0, 1]:
            gender_label = 'female' if gender == 1 else 'male'
            
            # Probability of being above each threshold
            probs = []
            for thresh in thresholds:
                sql = f"""
                    SELECT 
                        SUM(CASE WHEN LOG_REAL_HRLYEARN >= {thresh} THEN FINALWT ELSE 0 END) * 1.0 / SUM(FINALWT)
                    FROM lfs
                    WHERE HRLYEARN > 0{year_filter} AND IS_FEMALE = {gender}
                """
                probs.append(self.connection.execute(sql).fetchone()[0])
            
            results['by_gender'][gender_label] = probs
        
        # Compute gap in probabilities
        results['progression_gap'] = [
            results['by_gender']['male'][i] - results['by_gender']['female'][i]
            for i in range(len(quantiles))
        ]
        
        return results
