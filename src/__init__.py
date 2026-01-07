"""
EquiPay Canada - Core Package
ML-Powered Compensation Analysis & Pay Equity System

DATA SCOPE:
This project uses ONLY two data sources:
1. LFS PUMF microdata (2010-2025) - Statistics Canada catalogue 71M0001X
2. Macroeconomic data (CPI, GDP, unemployment, interest rates)

SURVEY WEIGHTS:
All analyses use FINALWT for population-level inference. This is MANDATORY.
- Descriptive stats: weighted means, medians, quantiles
- ML training: sample_weight parameter
- ML evaluation: WeightedMetrics class

Key Modules:
- constants: Centralized column names and LFS code mappings
- macro_data: Macroeconomic data (CPI, GDP, unemployment) 2010-2025
- data_pipeline: Data loading, cleaning, and feature creation
- lfs_loader: LFS PUMF microdata loader with weighted statistics
- ml_utils: Weighted train/test splits and evaluation metrics
- feature_engineering: ML feature preparation
- models: Salary prediction ensemble models (with sample weights)
- analysis: Pay equity statistical analysis
- fairness: Algorithmic fairness evaluation (with survey weights)
- statistical_tests: Advanced econometric tests
- time_series: Time series analysis for wage trends
- utils: Utility functions
"""

__version__ = "1.0.0"
__author__ = "EquiPay Canada Team"

# Core constants and utilities
from .constants import (
    COLS, DATA_SCOPE_START, DATA_SCOPE_END, DATA_YEARS, DATA_SOURCES,
    GENDER_CODES, PROVINCE_CODES, EDUCATION_CODES, NOC_10_CODES,
    IMMIG_CODES, MARSTAT_CODES, COWMAIN_CODES, FIRMSIZE_CODES,
    LFSSTAT_CODES, WHYPT_CODES, EFAMTYPE_CODES, AGYOWNK_CODES, SCHOOLN_CODES,
    EXTENDED_BINARY_FEATURES, EXTENDED_CATEGORICAL_FEATURES, INTERSECTIONAL_ATTRIBUTES,
    normalize_column_names, get_wage_column, get_all_mappings
)

# Data loading
from .data_pipeline import LFSDataPipeline
from .lfs_loader import LFSDataLoader, load_lfs_data

# Macro data
from .macro_data import (
    MACRO_DATA, get_macro_dataframe, add_macro_to_dataframe,
    get_deflator, adjust_for_inflation, BASE_YEAR
)

# ML utilities - weighted train/test splits and evaluation
from .ml_utils import (
    WeightedMLSplitter, WeightedMetrics, WeightedGapAnalysis,
    prepare_weighted_training_data
)

# Analysis modules
from .feature_engineering import FeatureEngineer
from .econometric_features import (
    EconometricFeatureEngineer, FeatureConfig, FeatureComplexity,
    create_econometric_features, get_feature_groups
)
from .models import SalaryPredictor, WageGapModel
from .analysis import PayEquityAnalyzer
from .fairness import FairnessAnalyzer

# Statistical tests
from .statistical_tests import AdvancedStatisticalTests
from .time_series import WageGapTimeSeriesAnalyzer

# Variance estimation (per StatsCan PUMF methodology)
from .bootstrap_variance import (
    PoissonBootstrap, QualityIndicators, assess_quality,
    analyze_wage_gap_with_variance, combine_monthly_weights
)

# Utilities
from .utils import setup_logging, Timer, format_currency, format_percentage

# Exceptions
from .exceptions import (
    EquiPayError,
    DataLoadError,
    DataValidationError,
    MissingColumnError,
    ModelError,
    ModelNotTrainedError,
    ModelNotFoundError,
    ConfigurationError,
    AnalysisError,
    InsufficientDataError,
    FairnessError,
    APIError,
)

# Logging
from .logging_config import setup_logging, get_logger

__all__ = [
    # Version
    '__version__',
    
    # Constants
    'COLS',
    'DATA_SCOPE_START',
    'DATA_SCOPE_END',
    
    # Data
    'LFSDataPipeline',
    'LFSDataLoader',
    'load_lfs_data',
    
    # Macro
    'MACRO_DATA',
    'get_macro_dataframe',
    'add_macro_to_dataframe',
    'get_deflator',
    'adjust_for_inflation',
    'BASE_YEAR',
    
    # Analysis
    'FeatureEngineer',
    'SalaryPredictor',
    'PayEquityAnalyzer',
    'FairnessAnalyzer',
    'AdvancedStatisticalTests',
    'WageGapTimeSeriesAnalyzer',
    
    # Utilities
    'setup_logging',
    'Timer',
    'format_currency',
    'format_percentage',
    'get_logger',
    
    # Exceptions
    'EquiPayError',
    'DataLoadError',
    'DataValidationError',
    'MissingColumnError',
    'ModelError',
    'ModelNotTrainedError',
    'ModelNotFoundError',
    'ConfigurationError',
    'AnalysisError',
    'InsufficientDataError',
    'FairnessError',
    'APIError',
]
