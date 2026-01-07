"""
EquiPay Canada - Gap Analysis Package
=====================================

Comprehensive tools for analyzing wage gaps across multiple dimensions
with proper survey weighting, decomposition methods, and statistical inference.

This package provides publication-quality gap analysis using best practices
from labor economics, with built-in safeguards against data leakage and
proper handling of survey weights.

Modules:
--------
- core: Core gap calculation methods with survey weights
- decomposition: Oaxaca-Blinder and related decomposition methods
- intersectional: Intersectionality analysis for compound gaps
- quantile: Quantile-based analysis (glass ceiling, sticky floor)
- temporal: Time-series analysis of gap evolution
- reporting: Publication-ready tables and figures

Key Features:
-------------
1. SURVEY WEIGHTS: All calculations use FINALWT for population inference
2. LEAKAGE PREVENTION: Integration with LeakageGuard throughout
3. PROPER INFERENCE: Bootstrap/cluster-robust standard errors
4. DECOMPOSITION: Multiple decomposition methodologies
5. INTERSECTIONALITY: Multi-dimensional gap analysis

Usage:
------
    from src.gap_analysis import GapAnalyzer, OaxacaBlinderDecomposition
    
    analyzer = GapAnalyzer(df, weight_col='FINALWT')
    
    # Simple gap
    gap = analyzer.calculate_gap('REAL_HRLYEARN', by='GENDER')
    
    # Decomposition
    decomp = OaxacaBlinderDecomposition()
    results = decomp.fit(X, y, group_indicator=df['IS_FEMALE'])
    
    # Intersectional
    from src.gap_analysis import IntersectionalAnalyzer
    intersect = IntersectionalAnalyzer(df)
    results = intersect.analyze(['GENDER', 'IMMIG', 'PROV'])

Author: EquiPay Canada Research Team
Version: 1.0.0
"""

from src.gap_analysis.core import (
    GapAnalyzer,
    calculate_weighted_gap,
    calculate_gap_with_ci,
)

from src.gap_analysis.decomposition import (
    OaxacaBlinderDecomposition,
    ThreefoldDecomposition,
    DetailedDecomposition,
)

from src.gap_analysis.intersectional import (
    IntersectionalAnalyzer,
    calculate_compound_gap,
)

from src.gap_analysis.quantile import (
    QuantileGapAnalyzer,
    glass_ceiling_test,
    sticky_floor_test,
)

from src.gap_analysis.selection import (
    HeckmanTwoStep,
    HeckmanMLE,
    HeckmanResult,
    run_heckman_gender_gap,
    prepare_lfs_selection_data,
)

from src.gap_analysis.matching import (
    PropensityScoreMatching,
    InverseProbabilityWeighting,
    DoublyRobust,
    MatchingResult,
    IPWResult,
    DoublyRobustResult,
    run_matching_gender_gap,
)

from src.gap_analysis.policy import (
    DifferenceInDifferences,
    EventStudy,
    TripleDifference,
    DiDResult,
    EventStudyResult,
    CANADIAN_POLICY_EVENTS,
    analyze_policy_impact,
)

__all__ = [
    # Core
    'GapAnalyzer',
    'calculate_weighted_gap',
    'calculate_gap_with_ci',
    
    # Decomposition
    'OaxacaBlinderDecomposition',
    'ThreefoldDecomposition',
    'DetailedDecomposition',
    
    # Intersectional
    'IntersectionalAnalyzer',
    'calculate_compound_gap',
    
    # Quantile
    'QuantileGapAnalyzer',
    'glass_ceiling_test',
    'sticky_floor_test',
    
    # Selection (Heckman)
    'HeckmanTwoStep',
    'HeckmanMLE',
    'HeckmanResult',
    'run_heckman_gender_gap',
    'prepare_lfs_selection_data',
    
    # Matching (PSM/IPW)
    'PropensityScoreMatching',
    'InverseProbabilityWeighting',
    'DoublyRobust',
    'MatchingResult',
    'IPWResult',
    'DoublyRobustResult',
    'run_matching_gender_gap',
    
    # Policy evaluation
    'DifferenceInDifferences',
    'EventStudy',
    'TripleDifference',
    'DiDResult',
    'EventStudyResult',
    'CANADIAN_POLICY_EVENTS',
    'analyze_policy_impact',
]
