"""
EquiPay Canada - Core Package
ML-Powered Compensation Analysis & Pay Equity System

DATA SCOPE:
This project uses ONLY two data sources:
1. LFS PUMF microdata (2010-2025) - Statistics Canada catalogue 71M0001X
2. Macroeconomic data (CPI, GDP, unemployment, interest rates)

Key Modules:
- constants: Centralized column names and LFS code mappings
- macro_data: Macroeconomic data (CPI, GDP, unemployment) 2010-2025
- data_pipeline: Data loading, cleaning, and feature creation
- lfs_loader: LFS PUMF microdata loader with weighted statistics
- feature_engineering: ML feature preparation
- models: Salary prediction ensemble models
- analysis: Pay equity statistical analysis
- fairness: Algorithmic fairness evaluation
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
    normalize_column_names, get_wage_column
)

# Data loading
from .data_pipeline import LFSDataPipeline
from .lfs_loader import LFSDataLoader, load_lfs_data

# Macro data
from .macro_data import (
    MACRO_DATA, get_macro_dataframe, add_macro_to_dataframe,
    get_deflator, adjust_for_inflation, BASE_YEAR
)

# Analysis modules
from .feature_engineering import FeatureEngineer
from .models import SalaryPredictor
from .analysis import PayEquityAnalyzer
from .fairness import FairnessAnalyzer

# Statistical tests
from .statistical_tests import AdvancedStatisticalTests
from .time_series import WageGapTimeSeriesAnalyzer

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
