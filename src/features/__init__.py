"""
EquiPay Canada - Feature Engineering Package
=============================================

This package provides comprehensive feature engineering capabilities for
the Canadian Labour Force Survey (LFS) data, with built-in safeguards
against data leakage.

Modules:
--------
- comprehensive: Full-spectrum feature engineering using all 60 LFS columns
- lfs_columns: Complete mapping and metadata for LFS PUMF columns
- interactions: Intersection features for gender gap analysis
- derived: Derived features (safe, non-wage-based)

Usage:
------
    from src.features import ComprehensiveFeatureEngineer
    
    engineer = ComprehensiveFeatureEngineer()
    df_features = engineer.create_all_features(df)
    
    # Get feature list (guaranteed leak-free)
    features = engineer.get_feature_names()
"""

from src.features.comprehensive import ComprehensiveFeatureEngineer
from src.features.lfs_columns import (
    LFS_COLUMN_METADATA,
    get_column_description,
    get_columns_by_category,
    DEMOGRAPHIC_COLS,
    HUMAN_CAPITAL_COLS,
    JOB_COLS,
    GEOGRAPHIC_COLS,
    TIME_COLS,
)

__all__ = [
    'ComprehensiveFeatureEngineer',
    'LFS_COLUMN_METADATA',
    'get_column_description',
    'get_columns_by_category',
    'DEMOGRAPHIC_COLS',
    'HUMAN_CAPITAL_COLS',
    'JOB_COLS',
    'GEOGRAPHIC_COLS',
    'TIME_COLS',
]
