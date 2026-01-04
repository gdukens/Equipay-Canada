"""
EquiPay Canada - Centralized Constants and LFS Code Mappings
=============================================================

This module provides a single source of truth for:
- LFS PUMF variable codes and labels
- Column naming conventions
- Analysis configuration
- Province, education, occupation mappings

All other modules should import from here to ensure consistency.
"""

from typing import Dict, List, Any
from dataclasses import dataclass


# =============================================================================
# PROJECT CONFIGURATION
# =============================================================================

PROJECT_NAME = "EquiPay Canada"
PROJECT_VERSION = "1.0.0"

# DATA SCOPE: This project uses ONLY LFS PUMF data (2010-2025) + macroeconomic data
# No other external data sources are required or used
DATA_SCOPE_START = 2010
DATA_SCOPE_END = 2025
DATA_YEARS = list(range(DATA_SCOPE_START, DATA_SCOPE_END + 1))  # [2010, 2011, ..., 2025]

# Base year for real wage calculations (CPI deflator)
BASE_YEAR = 2010

# Data sources documentation
DATA_SOURCES = {
    'primary': 'Statistics Canada Labour Force Survey (LFS) PUMF',
    'catalogue': '71M0001X',
    'years': f'{DATA_SCOPE_START}-{DATA_SCOPE_END}',
    'macro': 'Canadian macroeconomic indicators (CPI, GDP, unemployment, interest rates)',
}


# =============================================================================
# COLUMN NAMING CONVENTIONS
# =============================================================================

@dataclass
class Columns:
    """Standardized column names used throughout the project."""
    
    # Identifiers
    RECORD_ID = 'REC_NUM'
    
    # Time dimensions
    SURVEY_YEAR = 'SURVYEAR'
    SURVEY_MONTH = 'SURVMNTH'
    YEAR = 'YEAR'
    MONTH = 'MONTH'
    YEAR_MONTH = 'YEAR_MONTH'
    REF_DATE = 'REF_DATE'
    
    # Target variable
    HOURLY_EARNINGS = 'HRLYEARN'
    REAL_HOURLY_EARNINGS = 'REAL_HRLYEARN'
    LOG_HOURLY_EARNINGS = 'LOG_HRLYEARN'
    LOG_REAL_HOURLY_EARNINGS = 'LOG_REAL_HRLYEARN'
    
    # Demographics
    GENDER = 'GENDER'          # LFS PUMF uses GENDER (1=Male, 2=Female)
    GENDER_LABEL = 'GENDER_LABEL'
    IS_FEMALE = 'IS_FEMALE'    # Binary: 1=Female, 0=Male
    AGE_6 = 'AGE_6'            # 6 age categories
    AGE_12 = 'AGE_12'          # 12 age categories
    AGE_LABEL = 'AGE_LABEL'
    MARITAL_STATUS = 'MARSTAT'
    IMMIGRATION = 'IMMIG'
    
    # Education
    EDUCATION = 'EDUC'
    EDUCATION_LABEL = 'EDUC_LABEL'
    HAS_DEGREE = 'HAS_DEGREE'  # Binary: 1=Bachelor's or higher
    
    # Employment
    OCCUPATION_10 = 'NOC_10'   # 10 broad occupation categories
    OCCUPATION_43 = 'NOC_43'   # 43 detailed occupation categories
    OCCUPATION_LABEL = 'NOC_LABEL'
    INDUSTRY = 'NAICS_21'      # 21 industry categories
    INDUSTRY_LABEL = 'INDUSTRY_LABEL'
    PROVINCE = 'PROV'
    PROVINCE_LABEL = 'PROV_LABEL'
    CMA = 'CMA'                # Census Metropolitan Area
    
    # Job characteristics
    FULLTIME_PARTTIME = 'FTPTMAIN'
    FTPT_LABEL = 'FTPT_LABEL'
    IS_FULLTIME = 'IS_FULLTIME'
    PERMANENT_TEMP = 'PERMTEMP'
    PERMTEMP_LABEL = 'PERMTEMP_LABEL'
    IS_PERMANENT = 'IS_PERMANENT'
    UNION = 'UNION'
    UNION_LABEL = 'UNION_LABEL'
    IS_UNION = 'IS_UNION'
    TENURE = 'TENURE'
    USUAL_HOURS = 'UHRSMAIN'
    ACTUAL_HOURS = 'AHRSMAIN'
    ESTABLISHMENT_SIZE = 'ESTSIZE'
    FIRM_SIZE = 'FIRMSIZE'
    
    # Survey weight
    FINAL_WEIGHT = 'FINALWT'
    
    # Macroeconomic context
    CPI = 'cpi'
    GDP_GROWTH = 'gdp_growth'
    UNEMPLOYMENT = 'unemployment'
    INTEREST_RATE = 'interest_rate'
    INFLATION = 'inflation'
    DEFLATOR = 'deflator'
    RECESSION = 'recession'
    COVID = 'covid'
    ECONOMIC_PERIOD = 'ECONOMIC_PERIOD'
    
    # Derived/Analysis
    EXPERIENCE_PROXY = 'EXPERIENCE_PROXY'
    AGE_APPROX = 'AGE_APPROX'
    SOURCE = 'source'
    
    # =========================================================================
    # CONVENIENCE ALIASES (short names used in scripts)
    # =========================================================================
    # These provide backward compatibility with code using short column names
    EDUC = EDUCATION
    NOC_10 = OCCUPATION_10
    NOC_43 = OCCUPATION_43
    PROV = PROVINCE
    FTPTMAIN = FULLTIME_PARTTIME
    HRLYEARN = HOURLY_EARNINGS
    REAL_HRLYEARN = REAL_HOURLY_EARNINGS
    LOG_HRLYEARN = LOG_HOURLY_EARNINGS
    NAICS_21 = INDUSTRY


# Singleton instance
COLS = Columns()


# =============================================================================
# WAGE COLUMN USAGE GUIDELINES
# =============================================================================
# For CROSS-YEAR analysis (time series, trend comparisons):
#   Use COLS.REAL_HOURLY_EARNINGS (inflation-adjusted to 2010 dollars)
#
# For WITHIN-YEAR analysis (single year comparisons):
#   Use COLS.HOURLY_EARNINGS (nominal wages are fine)
#
# For REGRESSIONS spanning multiple years:
#   Use COLS.LOG_REAL_HOURLY_EARNINGS


def get_wage_column(analysis_type: str = 'cross_year') -> str:
    """
    Get the appropriate wage column based on analysis type.
    
    Args:
        analysis_type: 'cross_year' for time series/trends (uses real wages)
                      'within_year' for single-year analysis (uses nominal)
                      'regression' for log real wages (Mincer equation)
    
    Returns:
        Column name string
    """
    if analysis_type == 'cross_year':
        return COLS.REAL_HOURLY_EARNINGS
    elif analysis_type == 'within_year':
        return COLS.HOURLY_EARNINGS
    elif analysis_type == 'regression':
        return COLS.LOG_REAL_HOURLY_EARNINGS
    else:
        # Default to real wages for safety
        return COLS.REAL_HOURLY_EARNINGS


# =============================================================================
# LFS PUMF CODE MAPPINGS
# =============================================================================

# Gender codes (GENDER variable)
GENDER_CODES = {
    1: 'Male',
    2: 'Female'
}

# Reverse lookup: label -> code (for use in notebooks/scripts)
GENDER_CODES_REVERSE = {
    'Male': 1,
    'Female': 2
}

# Province codes (PROV variable)
PROVINCE_CODES = {
    10: 'Newfoundland and Labrador',
    11: 'Prince Edward Island',
    12: 'Nova Scotia',
    13: 'New Brunswick',
    24: 'Quebec',
    35: 'Ontario',
    46: 'Manitoba',
    47: 'Saskatchewan',
    48: 'Alberta',
    59: 'British Columbia',
}

# Province abbreviations
PROVINCE_ABBREV = {
    10: 'NL',
    11: 'PE',
    12: 'NS',
    13: 'NB',
    24: 'QC',
    35: 'ON',
    46: 'MB',
    47: 'SK',
    48: 'AB',
    59: 'BC',
}

# Province codes to ISO 3166-2 (for Plotly choropleth maps)
PROVINCE_TO_ISO = {
    10: 'CA-NL',  # Newfoundland and Labrador
    11: 'CA-PE',  # Prince Edward Island
    12: 'CA-NS',  # Nova Scotia
    13: 'CA-NB',  # New Brunswick
    24: 'CA-QC',  # Quebec
    35: 'CA-ON',  # Ontario
    46: 'CA-MB',  # Manitoba
    47: 'CA-SK',  # Saskatchewan
    48: 'CA-AB',  # Alberta
    59: 'CA-BC',  # British Columbia
}

# Canadian regions for aggregate analysis
REGIONS = {
    'Atlantic': [10, 11, 12, 13],
    'Central': [24, 35],
    'Prairies': [46, 47, 48],
    'West Coast': [59]
}

# Education codes (EDUC variable)
EDUCATION_CODES = {
    0: 'Less than high school',
    1: 'High school graduate',
    2: 'Some postsecondary',
    3: 'Postsecondary certificate/diploma',
    4: "Bachelor's degree",
    5: 'Above bachelor\'s degree'
}

# Education ordered levels for analysis
EDUCATION_LEVELS = [
    'Less than high school',
    'High school graduate', 
    'Some postsecondary',
    'Postsecondary certificate/diploma',
    "Bachelor's degree",
    "Above bachelor's degree"
]

# Age groups (AGE_6 variable)
AGE_6_CODES = {
    1: '15-24',
    2: '25-34',
    3: '35-44',
    4: '45-54',
    5: '55-64',
    6: '65+'
}

# Age groups (AGE_12 variable)
AGE_12_CODES = {
    1: '15-19',
    2: '20-24',
    3: '25-29',
    4: '30-34',
    5: '35-39',
    6: '40-44',
    7: '45-49',
    8: '50-54',
    9: '55-59',
    10: '60-64',
    11: '65-69',
    12: '70+'
}

# Age midpoints for numerical approximation
AGE_6_MIDPOINTS = {
    1: 20,   # 15-24
    2: 30,   # 25-34
    3: 40,   # 35-44
    4: 50,   # 45-54
    5: 60,   # 55-64
    6: 70    # 65+
}

AGE_12_MIDPOINTS = {
    1: 17, 2: 22, 3: 27, 4: 32, 5: 37, 6: 42,
    7: 47, 8: 52, 9: 57, 10: 62, 11: 67, 12: 75
}

# NOC 10 categories (broad occupational groups)
NOC_10_CODES = {
    0: 'Management',
    1: 'Business/Finance/Admin',
    2: 'Natural/Applied Sciences',
    3: 'Health',
    4: 'Education/Law/Social/Government',
    5: 'Art/Culture/Recreation/Sport',
    6: 'Sales and Service',
    7: 'Trades/Transport/Equipment',
    8: 'Natural Resources/Agriculture',
    9: 'Manufacturing/Utilities'
}

# NAICS 21 industry sectors
NAICS_21_CODES = {
    11: 'Agriculture',
    21: 'Mining/Oil/Gas',
    22: 'Utilities',
    23: 'Construction',
    31: 'Manufacturing',
    32: 'Manufacturing',
    33: 'Manufacturing',
    41: 'Wholesale Trade',
    44: 'Retail Trade',
    45: 'Retail Trade',
    48: 'Transportation/Warehousing',
    49: 'Transportation/Warehousing',
    51: 'Information/Cultural',
    52: 'Finance/Insurance',
    53: 'Real Estate',
    54: 'Professional Services',
    55: 'Management of Companies',
    56: 'Administrative Services',
    61: 'Education',
    62: 'Health Care',
    71: 'Arts/Entertainment',
    72: 'Accommodation/Food',
    81: 'Other Services',
    91: 'Public Administration'
}

# Simplified industry mapping (for cases where we get 1-10 codes)
INDUSTRY_SIMPLE_CODES = {
    1: 'Agriculture',
    2: 'Mining/Oil/Gas',
    3: 'Utilities',
    4: 'Construction',
    5: 'Manufacturing',
    6: 'Wholesale Trade',
    7: 'Retail Trade',
    8: 'Transportation',
    9: 'Finance/Insurance',
    10: 'Professional Services',
    11: 'Education',
    12: 'Health Care',
    13: 'Other Services',
    14: 'Public Administration'
}

# Full-time/Part-time (FTPTMAIN variable)
FTPT_CODES = {
    1: 'Full-time',
    2: 'Part-time'
}

# Union status (UNION variable)
UNION_CODES = {
    1: 'Union member',
    2: 'Covered by collective agreement',
    3: 'Non-unionized',
    6: 'Not applicable'
}

# Permanent/Temporary (PERMTEMP variable)
PERMTEMP_CODES = {
    1: 'Permanent',
    2: 'Temporary',
    3: 'Seasonal',
    4: 'Casual',
    6: 'Not applicable'
}

# Establishment size (ESTSIZE variable)
ESTSIZE_CODES = {
    1: '<20 employees',
    2: '20-99 employees',
    3: '100-499 employees',
    4: '500-999 employees',
    5: '1000+ employees',
    6: 'Unknown'
}

# Marital status (MARSTAT variable)
MARSTAT_CODES = {
    1: 'Married',
    2: 'Common-law',
    3: 'Widowed',
    4: 'Separated',
    5: 'Divorced',
    6: 'Single, never married'
}

# Immigration status (IMMIG variable)
IMMIG_CODES = {
    1: 'Immigrant',
    2: 'Non-immigrant',
    3: 'Non-permanent resident'
}

# =============================================================================
# ADDITIONAL LFS PUMF CODE MAPPINGS (Previously Untapped)
# =============================================================================

# Labour Force Status (LFSSTAT variable)
LFSSTAT_CODES = {
    1: 'Employed, at work',
    2: 'Employed, absent',
    3: 'Unemployed',
    4: 'Not in labour force'
}

# Class of Worker - Main Job (COWMAIN variable)
COWMAIN_CODES = {
    1: 'Public sector employee',
    2: 'Private sector employee',
    3: 'Self-employed incorporated',
    4: 'Self-employed unincorporated',
    5: 'Unpaid family worker',
    6: 'Not applicable'
}

# Multiple Job Holder (MJH variable)
MJH_CODES = {
    1: 'Single job holder',
    2: 'Multiple job holder',
    6: 'Not applicable'
}

# Ever Worked (EVERWORK variable)
EVERWORK_CODES = {
    1: 'Worked in past year',
    2: 'Worked 1-5 years ago',
    3: 'Worked 5+ years ago',
    4: 'Never worked',
    6: 'Not applicable'
}

# Reason for Part-Time Work (WHYPT variable)
WHYPT_CODES = {
    1: 'Own illness/disability',
    2: 'Caring for children',
    3: 'Other personal/family',
    4: 'Going to school',
    5: 'Personal preference',
    6: 'Business conditions',
    7: 'Could not find FT work',
    8: 'Other',
    0: 'Not applicable'
}

# Paid Overtime (PAIDOT variable)
PAIDOT_CODES = {
    1: 'Yes, paid overtime',
    2: 'No paid overtime',
    6: 'Not applicable'
}

# Unpaid Overtime (UNPAIDOT variable)
UNPAIDOT_CODES = {
    1: 'Yes, unpaid overtime',
    2: 'No unpaid overtime',
    6: 'Not applicable'
}

# Firm Size (FIRMSIZE variable) - different from establishment
FIRMSIZE_CODES = {
    1: '<20 employees',
    2: '20-99 employees',
    3: '100-499 employees',
    4: '500+ employees',
    6: 'Unknown'
}

# Reason for Absence (YABSENT variable)
YABSENT_CODES = {
    1: 'Own illness/disability',
    2: 'Caring for children',
    3: 'Maternity/parental leave',
    4: 'Other personal/family',
    5: 'Vacation',
    6: 'Labour dispute',
    7: 'Temporary layoff',
    8: 'Seasonal business',
    9: 'Casual job, no work',
    10: 'Work schedule',
    11: 'Self-employed, no work',
    12: 'Other',
    0: 'Not applicable'
}

# Economic Family Type (EFAMTYPE variable)
EFAMTYPE_CODES = {
    1: 'Married couple, children',
    2: 'Married couple, no children',
    3: 'Lone parent',
    4: 'Child in family',
    5: 'Other family member',
    6: 'Unattached individual',
    7: 'Not applicable'
}

# Age of Youngest Child (AGYOWNK variable)
AGYOWNK_CODES = {
    0: 'No children',
    1: 'Under 1 year',
    2: '1-2 years',
    3: '3-5 years',
    4: '6-12 years',
    5: '13-17 years',
    6: '18-24 years',
    7: '25+ years',
    8: 'Not applicable'
}

# School Attendance (SCHOOLN variable)
SCHOOLN_CODES = {
    1: 'Not attending school',
    2: 'Full-time student',
    3: 'Part-time student',
    6: 'Not applicable'
}

# Duration of Unemployment (DURUNEMP variable) - weeks
DURUNEMP_RANGES = {
    0: 'Not unemployed',
    1: '<5 weeks',
    2: '5-13 weeks',
    3: '14-26 weeks',
    4: '27-52 weeks',
    5: '52+ weeks'
}

# Why Left Last Job - New (WHYLEFTN variable)
WHYLEFTN_CODES = {
    1: 'Lost job/laid off',
    2: 'End of seasonal/temp',
    3: 'Illness/disability',
    4: 'Personal/family',
    5: 'Retired',
    6: 'Dissatisfied',
    7: 'Other',
    0: 'Not applicable'
}

# Census Metropolitan Area / Census Agglomeration (CMA variable)
# Note: CMA codes are complex; here are major ones
CMA_MAJOR_CODES = {
    0: 'Non-CMA/CA (Rural)',
    1: 'St. John\'s',
    2: 'Halifax',
    3: 'Moncton',
    4: 'Saint John',
    5: 'Saguenay',
    6: 'Québec',
    7: 'Sherbrooke',
    8: 'Trois-Rivières',
    9: 'Montréal',
    10: 'Ottawa-Gatineau (QC)',
    11: 'Ottawa-Gatineau (ON)',
    12: 'Kingston',
    13: 'Peterborough',
    14: 'Oshawa',
    15: 'Toronto',
    16: 'Hamilton',
    17: 'St. Catharines-Niagara',
    18: 'Kitchener-Cambridge-Waterloo',
    19: 'Brantford',
    20: 'Guelph',
    21: 'London',
    22: 'Windsor',
    23: 'Barrie',
    24: 'Greater Sudbury',
    25: 'Thunder Bay',
    26: 'Winnipeg',
    27: 'Regina',
    28: 'Saskatoon',
    29: 'Calgary',
    30: 'Edmonton',
    31: 'Kelowna',
    32: 'Abbotsford-Mission',
    33: 'Vancouver',
    34: 'Victoria',
    997: 'Unidentified CMA/CA',
    998: 'Unidentified non-CMA/CA',
    999: 'Not stated'
}

# =============================================================================
# FEATURE CONFIGURATION FOR ML MODELS
# =============================================================================

# Core features for pay equity analysis
CORE_NUMERIC_FEATURES = [
    'AGE_APPROX',
    'EXPERIENCE_PROXY',
    'TENURE',
    'UHRSMAIN',
]

# Extended numeric features (exploit more columns)
EXTENDED_NUMERIC_FEATURES = [
    'AGE_APPROX',
    'EXPERIENCE_PROXY',
    'TENURE',
    'PREVTEN',           # Previous job tenure - career stability
    'UHRSMAIN',          # Usual hours main job
    'AHRSMAIN',          # Actual hours main job
    'UTOTHRS',           # Usual total hours all jobs
    'ATOTHRS',           # Actual total hours all jobs
    'HOURS_GAP',         # Derived: difference between usual and actual
    'OVERTIME_HOURS',    # Derived: paid + unpaid overtime
]

CORE_CATEGORICAL_FEATURES = [
    'GENDER',
    'EDUC',
    'NOC_10',
    'NAICS_21',
    'PROV',
    'FTPTMAIN',
    'PERMTEMP',
    'UNION',
    'ESTSIZE',
]

# Extended categorical features (exploit more columns)
EXTENDED_CATEGORICAL_FEATURES = [
    'GENDER',
    'EDUC',
    'NOC_10',
    'NOC_43',            # More detailed occupation (43 categories)
    'NAICS_21',
    'PROV',
    'CMA_TYPE',          # Derived: Urban/Suburban/Rural
    'FTPTMAIN',
    'PERMTEMP',
    'UNION',
    'ESTSIZE',
    'FIRMSIZE',          # Firm size (different from establishment)
    'MARSTAT',           # Marital status
    'IMMIG',             # Immigration status
    'COWMAIN',           # Class of worker (public/private/self-employed)
    'MJH',               # Multiple job holder
    'WHYPT',             # Reason for part-time
    'EFAMTYPE',          # Family type
    'AGYOWNK',           # Age of youngest child
    'SCHOOLN',           # Student status
]

BINARY_FEATURES = [
    'IS_FEMALE',
    'IS_FULLTIME',
    'IS_PERMANENT',
    'IS_UNION',
    'HAS_DEGREE',
]

# Extended binary features (exploit more columns)
EXTENDED_BINARY_FEATURES = [
    'IS_FEMALE',
    'IS_FULLTIME',
    'IS_PERMANENT',
    'IS_UNION',
    'HAS_DEGREE',
    'IS_IMMIGRANT',          # Immigrant status
    'IS_URBAN',              # Lives in CMA (metropolitan area)
    'IS_PUBLIC_SECTOR',      # Works in public sector
    'IS_SELF_EMPLOYED',      # Self-employed (incorporated or not)
    'IS_MULTIPLE_JOBS',      # Has multiple jobs
    'IS_MARRIED',            # Married or common-law
    'HAS_YOUNG_CHILDREN',    # Has children under 6
    'HAS_CHILDREN',          # Has any children
    'IS_STUDENT',            # Currently attending school
    'WORKS_OVERTIME',        # Has paid or unpaid overtime
    'IS_INVOLUNTARY_PT',     # Part-time but wants full-time
    'IS_SEASONAL',           # Seasonal or temporary work
]

# Control variables for regression analysis
REGRESSION_CONTROLS = [
    'EDUC',
    'NOC_10',
    'AGE_6',
    'PROV',
    'FTPTMAIN',
    'UNION',
]

# Extended regression controls
EXTENDED_REGRESSION_CONTROLS = [
    'EDUC',
    'NOC_10',
    'AGE_6',
    'PROV',
    'FTPTMAIN',
    'UNION',
    'MARSTAT',
    'IMMIG',
    'COWMAIN',
    'ESTSIZE',
    'AGYOWNK',
    'CMA_TYPE',
]

# Protected attributes for fairness analysis
PROTECTED_ATTRIBUTES = ['GENDER', 'AGE_6', 'PROV', 'IMMIG']

# Additional intersectional analysis dimensions
INTERSECTIONAL_ATTRIBUTES = [
    ('GENDER', 'IMMIG'),       # Gender × Immigration
    ('GENDER', 'EDUC'),        # Gender × Education
    ('GENDER', 'AGYOWNK'),     # Gender × Parenthood (motherhood penalty)
    ('GENDER', 'MARSTAT'),     # Gender × Marriage
    ('GENDER', 'PROV'),        # Gender × Province
    ('GENDER', 'NOC_10'),      # Gender × Occupation
    ('IMMIG', 'EDUC'),         # Immigration × Education (credential recognition)
]


# =============================================================================
# ANALYSIS THRESHOLDS AND PARAMETERS
# =============================================================================

# Valid wage range for filtering outliers
MIN_HOURLY_WAGE = 5.0    # Below minimum wage = likely error
MAX_HOURLY_WAGE = 500.0  # Above $500/hr = likely error

# Statistical significance levels
ALPHA_STANDARD = 0.05
ALPHA_STRICT = 0.01
ALPHA_RELAXED = 0.10

# Fairness thresholds
DEMOGRAPHIC_PARITY_THRESHOLD = 0.80  # 80% rule
WAGE_GAP_THRESHOLD = 0.10            # 10% gap threshold


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_code_label(code: int, mapping: Dict[int, str], default: str = 'Unknown') -> str:
    """Get label for a code value."""
    return mapping.get(code, default)


def get_all_mappings() -> Dict[str, Dict[int, str]]:
    """Return all code mappings as a dictionary."""
    return {
        'GENDER': GENDER_CODES,
        'PROV': PROVINCE_CODES,
        'EDUC': EDUCATION_CODES,
        'AGE_6': AGE_6_CODES,
        'AGE_12': AGE_12_CODES,
        'NOC_10': NOC_10_CODES,
        'NAICS_21': NAICS_21_CODES,
        'FTPTMAIN': FTPT_CODES,
        'UNION': UNION_CODES,
        'PERMTEMP': PERMTEMP_CODES,
        'ESTSIZE': ESTSIZE_CODES,
        'MARSTAT': MARSTAT_CODES,
        'IMMIG': IMMIG_CODES,
        # New extended mappings
        'LFSSTAT': LFSSTAT_CODES,
        'COWMAIN': COWMAIN_CODES,
        'MJH': MJH_CODES,
        'EVERWORK': EVERWORK_CODES,
        'WHYPT': WHYPT_CODES,
        'PAIDOT': PAIDOT_CODES,
        'UNPAIDOT': UNPAIDOT_CODES,
        'FIRMSIZE': FIRMSIZE_CODES,
        'YABSENT': YABSENT_CODES,
        'EFAMTYPE': EFAMTYPE_CODES,
        'AGYOWNK': AGYOWNK_CODES,
        'SCHOOLN': SCHOOLN_CODES,
        'WHYLEFTN': WHYLEFTN_CODES,
        'CMA': CMA_MAJOR_CODES,
    }


def apply_labels(df, columns: List[str] = None):
    """
    Apply human-readable labels to coded columns in a DataFrame.
    
    Args:
        df: DataFrame with coded columns
        columns: List of columns to label (if None, labels all known columns)
        
    Returns:
        DataFrame with added _LABEL columns
    """
    import pandas as pd
    
    df = df.copy()
    mappings = get_all_mappings()
    
    columns_to_process = columns or list(mappings.keys())
    
    for col in columns_to_process:
        if col in df.columns and col in mappings:
            label_col = f"{col}_LABEL" if col != 'GENDER' else 'GENDER_LABEL'
            df[label_col] = df[col].map(mappings[col])
    
    return df


def get_feature_config() -> Dict[str, Any]:
    """Return standard feature configuration for ML models."""
    return {
        'numeric_features': CORE_NUMERIC_FEATURES.copy(),
        'categorical_features': CORE_CATEGORICAL_FEATURES.copy(),
        'binary_features': BINARY_FEATURES.copy(),
        'target': COLS.HOURLY_EARNINGS,
        'log_target': COLS.LOG_HOURLY_EARNINGS,
        'weight_column': COLS.FINAL_WEIGHT,
    }


# =============================================================================
# BACKWARD COMPATIBILITY - SEX column alias
# =============================================================================
# Some older code uses SEX instead of GENDER. These mappings provide compatibility.

SEX_CODES = GENDER_CODES  # Alias for backward compatibility

# Legacy column name mapping
LEGACY_COLUMN_MAP = {
    'SEX': 'GENDER',
    'NOC_40': 'NOC_43',
    'NAICS': 'NAICS_21',
}


def normalize_column_names(df):
    """
    Rename legacy column names to standard names.
    
    Args:
        df: DataFrame with potentially legacy column names
        
    Returns:
        DataFrame with standardized column names
    """
    df = df.copy()
    
    for old_name, new_name in LEGACY_COLUMN_MAP.items():
        if old_name in df.columns and new_name not in df.columns:
            df = df.rename(columns={old_name: new_name})
    
    return df


# =============================================================================
# HUMAN-READABLE COLUMN LABELS FOR EXPORTS
# =============================================================================
# Maps coded variable names to clean, publication-ready labels

COLUMN_LABELS = {
    # Demographics
    'GENDER': 'Gender',
    'SEX': 'Gender',
    'GENDER_LABEL': 'Gender',
    'IS_FEMALE': 'Female',
    'AGE_6': 'Age Group (6 cat.)',
    'AGE_12': 'Age Group (12 cat.)',
    'AGE_LABEL': 'Age Group',
    'AGE_APPROX': 'Approximate Age',
    'MARSTAT': 'Marital Status',
    'IMMIG': 'Immigration Status',
    
    # Education
    'EDUC': 'Education Level',
    'EDUCATION': 'Education Level',
    'EDUC_LABEL': 'Education Level',
    'HAS_DEGREE': 'Has University Degree',
    
    # Employment / Occupation
    'NOC_10': 'Occupation (Broad)',
    'NOC_43': 'Occupation (Detailed)',
    'OCCUPATION_10': 'Occupation (Broad)',
    'OCCUPATION_43': 'Occupation (Detailed)',
    'NOC_LABEL': 'Occupation',
    'NAICS_21': 'Industry',
    'INDUSTRY': 'Industry',
    'INDUSTRY_LABEL': 'Industry',
    
    # Geography
    'PROV': 'Province',
    'PROVINCE': 'Province',
    'PROV_LABEL': 'Province',
    'CMA': 'Metropolitan Area',
    
    # Wages
    'HRLYEARN': 'Hourly Wage ($)',
    'HOURLY_EARNINGS': 'Hourly Wage ($)',
    'REAL_HRLYEARN': 'Real Hourly Wage ($)',
    'REAL_HOURLY_EARNINGS': 'Real Hourly Wage ($)',
    'LOG_HRLYEARN': 'Log Hourly Wage',
    'LOG_HOURLY_EARNINGS': 'Log Hourly Wage',
    
    # Job characteristics
    'FTPTMAIN': 'Full-Time/Part-Time',
    'FULLTIME_PARTTIME': 'Full-Time/Part-Time',
    'FTPT_LABEL': 'Work Status',
    'IS_FULLTIME': 'Full-Time',
    'PERMTEMP': 'Permanent/Temporary',
    'PERMTEMP_LABEL': 'Job Permanence',
    'IS_PERMANENT': 'Permanent Job',
    'UNION': 'Union Status',
    'UNION_LABEL': 'Union Coverage',
    'IS_UNION': 'Unionized',
    'TENURE': 'Job Tenure',
    'UHRSMAIN': 'Usual Hours/Week',
    'AHRSMAIN': 'Actual Hours/Week',
    'ESTSIZE': 'Establishment Size',
    'FIRMSIZE': 'Firm Size',
    
    # Time
    'SURVYEAR': 'Year',
    'YEAR': 'Year',
    'SURVMNTH': 'Month',
    'MONTH': 'Month',
    'YEAR_MONTH': 'Year-Month',
    'REF_DATE': 'Reference Date',
    
    # Macroeconomic
    'cpi': 'Consumer Price Index',
    'gdp_growth': 'GDP Growth (%)',
    'unemployment': 'Unemployment Rate (%)',
    'interest_rate': 'Interest Rate (%)',
    'inflation': 'Inflation Rate (%)',
    'deflator': 'CPI Deflator',
    'recession': 'Recession Period',
    'covid': 'COVID-19 Period',
    'ECONOMIC_PERIOD': 'Economic Period',
    
    # Analysis results
    'wage_gap': 'Wage Gap (%)',
    'raw_gap': 'Raw Wage Gap (%)',
    'adjusted_gap': 'Adjusted Wage Gap (%)',
    'mean_wage': 'Mean Wage ($)',
    'median_wage': 'Median Wage ($)',
    'male_wage': 'Male Wage ($)',
    'female_wage': 'Female Wage ($)',
    'n': 'Sample Size',
    'n_obs': 'Observations',
    'coef': 'Coefficient',
    'std_err': 'Std. Error',
    'pvalue': 'P-Value',
    'ci_lower': '95% CI Lower',
    'ci_upper': '95% CI Upper',
    'r_squared': 'R²',
    'adj_r_squared': 'Adjusted R²',
    
    # Experience
    'EXPERIENCE_PROXY': 'Experience (Est.)',
    
    # Survey
    'FINALWT': 'Survey Weight',
    'REC_NUM': 'Record ID',
}

# French labels (optional - can be used for French reports)
COLUMN_LABELS_FR = {
    'GENDER': 'Genre',
    'SEX': 'Genre',
    'IS_FEMALE': 'Femme',
    'AGE_6': 'Groupe d\'âge (6 cat.)',
    'AGE_12': 'Groupe d\'âge (12 cat.)',
    'EDUC': 'Niveau d\'éducation',
    'EDUCATION': 'Niveau d\'éducation',
    'NOC_10': 'Profession (large)',
    'NOC_43': 'Profession (détaillée)',
    'PROV': 'Province',
    'PROVINCE': 'Province',
    'HRLYEARN': 'Salaire horaire ($)',
    'REAL_HRLYEARN': 'Salaire horaire réel ($)',
    'YEAR': 'Année',
    'wage_gap': 'Écart salarial (%)',
    'raw_gap': 'Écart brut (%)',
    'adjusted_gap': 'Écart ajusté (%)',
    'mean_wage': 'Salaire moyen ($)',
    'median_wage': 'Salaire médian ($)',
    'n': 'Taille d\'échantillon',
}


def humanize_columns(df, language: str = 'en', inplace: bool = False):
    """
    Replace coded column names with human-readable labels for export/display.
    
    Args:
        df: DataFrame with coded column names
        language: 'en' for English (default), 'fr' for French
        inplace: If True, modify DataFrame in place; otherwise return a copy
        
    Returns:
        DataFrame with human-readable column names
        
    Example:
        >>> df_export = humanize_columns(results_df)
        >>> df_export.to_csv('report.csv', index=False)
        
        >>> # For French reports:
        >>> df_fr = humanize_columns(results_df, language='fr')
    """
    labels = COLUMN_LABELS_FR if language == 'fr' else COLUMN_LABELS
    
    if not inplace:
        df = df.copy()
    
    # Build rename mapping for columns that exist in the DataFrame
    rename_map = {col: labels.get(col, col) for col in df.columns if col in labels}
    
    df.rename(columns=rename_map, inplace=True)
    
    return df


def get_column_label(column_name: str, language: str = 'en') -> str:
    """
    Get the human-readable label for a single column name.
    
    Args:
        column_name: The coded column name (e.g., 'NOC_10', 'HRLYEARN')
        language: 'en' for English (default), 'fr' for French
        
    Returns:
        Human-readable label, or the original name if not found
        
    Example:
        >>> get_column_label('NOC_10')
        'Occupation (Broad)'
        >>> get_column_label('EDUC', language='fr')
        "Niveau d'éducation"
    """
    labels = COLUMN_LABELS_FR if language == 'fr' else COLUMN_LABELS
    return labels.get(column_name, column_name)
