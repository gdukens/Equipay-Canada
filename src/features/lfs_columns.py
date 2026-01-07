"""
EquiPay Canada - LFS Column Metadata
====================================

Complete mapping and metadata for all 60+ columns in the Statistics Canada
Labour Force Survey (LFS) Public Use Microdata File (PUMF).

This module provides:
1. Column descriptions and categories
2. Value code mappings
3. Column groupings for analysis
4. Data type specifications

Source: Statistics Canada LFS PUMF Documentation
"""

from typing import Dict, List, Any


# =============================================================================
# COMPLETE COLUMN METADATA
# =============================================================================

LFS_COLUMN_METADATA: Dict[str, Dict[str, Any]] = {
    # =========================================================================
    # IDENTIFICATION
    # =========================================================================
    'REC_NUM': {
        'description': 'Record number (unique identifier)',
        'category': 'identification',
        'dtype': 'int64',
        'use_in_analysis': False,
    },
    'SURVYEAR': {
        'description': 'Survey year',
        'category': 'time',
        'dtype': 'int16',
        'use_in_analysis': True,
        'values': 'Year (e.g., 2010-2025)',
    },
    'SURVMNTH': {
        'description': 'Survey month',
        'category': 'time',
        'dtype': 'int8',
        'use_in_analysis': True,
        'values': {
            1: 'January', 2: 'February', 3: 'March', 4: 'April',
            5: 'May', 6: 'June', 7: 'July', 8: 'August',
            9: 'September', 10: 'October', 11: 'November', 12: 'December'
        },
    },
    'LFSSTAT': {
        'description': 'Labour force status',
        'category': 'employment',
        'dtype': 'int8',
        'use_in_analysis': True,
        'values': {
            1: 'Employed at work',
            2: 'Employed absent',
            3: 'Unemployed temporary layoff',
            4: 'Unemployed job searcher',
            5: 'Unemployed future job',
            6: 'Not in labour force'
        },
    },
    
    # =========================================================================
    # DEMOGRAPHICS
    # =========================================================================
    'GENDER': {
        'description': 'Gender of respondent',
        'category': 'demographics',
        'dtype': 'int8',
        'use_in_analysis': True,
        'is_key_variable': True,
        'values': {
            1: 'Male',
            2: 'Female'
        },
    },
    'AGE_12': {
        'description': 'Age group (12 categories)',
        'category': 'demographics',
        'dtype': 'int8',
        'use_in_analysis': True,
        'values': {
            1: '15-19', 2: '20-24', 3: '25-29', 4: '30-34',
            5: '35-39', 6: '40-44', 7: '45-49', 8: '50-54',
            9: '55-59', 10: '60-64', 11: '65-69', 12: '70+'
        },
    },
    'AGE_6': {
        'description': 'Age group (6 categories)',
        'category': 'demographics',
        'dtype': 'int8',
        'use_in_analysis': True,
        'values': {
            1: '15-24', 2: '25-34', 3: '35-44',
            4: '45-54', 5: '55-64', 6: '65+'
        },
    },
    'MARSTAT': {
        'description': 'Marital status',
        'category': 'demographics',
        'dtype': 'int8',
        'use_in_analysis': True,
        'values': {
            1: 'Married',
            2: 'Living common-law',
            3: 'Widowed',
            4: 'Separated',
            5: 'Divorced',
            6: 'Single, never married'
        },
    },
    'IMMIG': {
        'description': 'Immigrant status',
        'category': 'demographics',
        'dtype': 'int8',
        'use_in_analysis': True,
        'is_key_variable': True,
        'values': {
            1: 'Immigrant landed 0-5 years',
            2: 'Immigrant landed 5-10 years',
            3: 'Immigrant landed 10+ years',
            4: 'Non-immigrant (born in Canada)'
        },
    },
    'EFAMTYPE': {
        'description': 'Economic family type',
        'category': 'demographics',
        'dtype': 'int8',
        'use_in_analysis': True,
    },
    'AGYOWNK': {
        'description': 'Age of youngest child (if any)',
        'category': 'demographics',
        'dtype': 'int8',
        'use_in_analysis': True,
        'values': {
            0: 'No children in household',
            1: 'Child under 1 year',
            2: 'Child 1-2 years',
            3: 'Child 3-5 years',
            4: 'Child 6-12 years',
            5: 'Child 13-14 years',
            6: 'Child 15-17 years',
            7: 'Child 18-24 years',
        },
    },
    
    # =========================================================================
    # HUMAN CAPITAL
    # =========================================================================
    'EDUC': {
        'description': 'Highest education level attained',
        'category': 'human_capital',
        'dtype': 'int8',
        'use_in_analysis': True,
        'is_key_variable': True,
        'values': {
            0: 'Less than Grade 9',
            1: 'Grade 9-10',
            2: 'Grade 11-13 (no diploma)',
            3: 'High school diploma',
            4: 'Some post-secondary',
            5: 'Post-secondary certificate/diploma',
            6: "Bachelor's degree",
            7: 'Above bachelor\'s degree'
        },
    },
    'TENURE': {
        'description': 'Job tenure (years with current employer)',
        'category': 'human_capital',
        'dtype': 'int8',
        'use_in_analysis': True,
        'is_key_variable': True,
        'values': {
            0: '1-3 months',
            1: '4-6 months',
            2: '7-12 months',
            3: '1-5 years',
            4: '5-10 years',
            5: '10-20 years',
            6: '20+ years'
        },
    },
    
    # =========================================================================
    # JOB CHARACTERISTICS
    # =========================================================================
    'NOC_10': {
        'description': 'Occupation (NOC 10 major groups)',
        'category': 'job',
        'dtype': 'int8',
        'use_in_analysis': True,
        'is_key_variable': True,
        'values': {
            0: 'Management',
            1: 'Business/finance/administration',
            2: 'Natural sciences/engineering',
            3: 'Health',
            4: 'Education/law/social/community/government',
            5: 'Art/culture/recreation/sport',
            6: 'Sales and service',
            7: 'Trades/transport/equipment operators',
            8: 'Natural resources/agriculture',
            9: 'Manufacturing/utilities'
        },
    },
    'NOC_43': {
        'description': 'Occupation (NOC 43 minor groups)',
        'category': 'job',
        'dtype': 'int8',
        'use_in_analysis': True,
        'provides_detail_for': 'NOC_10',
    },
    'NAICS_21': {
        'description': 'Industry (NAICS 21 sectors)',
        'category': 'job',
        'dtype': 'int8',
        'use_in_analysis': True,
        'is_key_variable': True,
        'values': {
            1: 'Agriculture',
            2: 'Forestry/fishing/mining/oil/gas',
            3: 'Utilities',
            4: 'Construction',
            5: 'Manufacturing - durables',
            6: 'Manufacturing - non-durables',
            7: 'Wholesale trade',
            8: 'Retail trade',
            9: 'Transportation/warehousing',
            10: 'Finance/insurance',
            11: 'Real estate',
            12: 'Professional/scientific/technical',
            13: 'Business/building/other support',
            14: 'Educational services',
            15: 'Health care/social assistance',
            16: 'Information/culture/recreation',
            17: 'Accommodation/food services',
            18: 'Other services',
            19: 'Public administration',
            20: 'Management of companies',
            21: 'Other',
        },
    },
    'COWMAIN': {
        'description': 'Class of worker (employment type)',
        'category': 'job',
        'dtype': 'int8',
        'use_in_analysis': True,
        'is_key_variable': True,
        'values': {
            1: 'Public sector employee',
            2: 'Private sector employee',
            3: 'Self-employed incorporated',
            4: 'Self-employed unincorporated',
            5: 'Unpaid family worker'
        },
    },
    'PERMTEMP': {
        'description': 'Permanent/temporary job status',
        'category': 'job',
        'dtype': 'int8',
        'use_in_analysis': True,
        'values': {
            1: 'Permanent',
            2: 'Temporary - seasonal',
            3: 'Temporary - term/contract',
            4: 'Temporary - casual',
            5: 'Temporary - other'
        },
    },
    'FTPTMAIN': {
        'description': 'Full-time/part-time status',
        'category': 'job',
        'dtype': 'int8',
        'use_in_analysis': True,
        'values': {
            1: 'Full-time',
            2: 'Part-time'
        },
    },
    'UNION': {
        'description': 'Union membership/coverage',
        'category': 'job',
        'dtype': 'int8',
        'use_in_analysis': True,
        'values': {
            1: 'Union member',
            2: 'Not a member but covered',
            3: 'Not a member or covered'
        },
    },
    'ESTSIZE': {
        'description': 'Establishment size (employees)',
        'category': 'job',
        'dtype': 'int8',
        'use_in_analysis': True,
        'values': {
            1: 'Less than 20',
            2: '20-99',
            3: '100-499',
            4: '500+'
        },
    },
    'FIRMSIZE': {
        'description': 'Firm size (total employees)',
        'category': 'job',
        'dtype': 'int8',
        'use_in_analysis': True,
    },
    'MJH': {
        'description': 'Multiple job holder',
        'category': 'job',
        'dtype': 'int8',
        'use_in_analysis': True,
        'values': {
            1: 'Yes',
            2: 'No'
        },
    },
    'WHYPT': {
        'description': 'Reason for part-time work',
        'category': 'job',
        'dtype': 'int8',
        'use_in_analysis': True,
        'values': {
            0: 'Not applicable (full-time)',
            1: 'Own illness',
            2: 'Personal/family responsibilities',
            3: 'Going to school',
            4: 'Could only find part-time',
            5: 'Did not want full-time',
            6: 'Other'
        },
    },
    
    # =========================================================================
    # WORK HOURS
    # =========================================================================
    'UHRSMAIN': {
        'description': 'Usual hours worked per week (main job)',
        'category': 'hours',
        'dtype': 'float32',
        'use_in_analysis': True,
    },
    'AHRSMAIN': {
        'description': 'Actual hours worked per week (main job)',
        'category': 'hours',
        'dtype': 'float32',
        'use_in_analysis': True,
    },
    'UTOTHRS': {
        'description': 'Usual hours worked per week (all jobs)',
        'category': 'hours',
        'dtype': 'float32',
        'use_in_analysis': True,
    },
    'ATOTHRS': {
        'description': 'Actual hours worked per week (all jobs)',
        'category': 'hours',
        'dtype': 'float32',
        'use_in_analysis': True,
    },
    
    # =========================================================================
    # GEOGRAPHY
    # =========================================================================
    'PROV': {
        'description': 'Province',
        'category': 'geography',
        'dtype': 'int8',
        'use_in_analysis': True,
        'is_key_variable': True,
        'values': {
            10: 'Newfoundland and Labrador',
            11: 'Prince Edward Island',
            12: 'Nova Scotia',
            13: 'New Brunswick',
            24: 'Quebec',
            35: 'Ontario',
            46: 'Manitoba',
            47: 'Saskatchewan',
            48: 'Alberta',
            59: 'British Columbia'
        },
    },
    'CMA': {
        'description': 'Census Metropolitan Area',
        'category': 'geography',
        'dtype': 'int16',
        'use_in_analysis': True,
        'values': {
            0: 'Rural/small town',
            205: "St. John's",
            305: 'Halifax',
            408: 'Moncton',
            421: 'Saint John',
            505: 'Saguenay',
            421: 'Trois-Rivières',
            433: 'Sherbrooke',
            462: 'Montréal',
            505: 'Ottawa-Gatineau (QC)',
            505: 'Ottawa-Gatineau (ON)',
            521: 'Kingston',
            532: 'Peterborough',
            535: 'Oshawa',
            535: 'Toronto',
            539: 'Hamilton',
            541: 'St. Catharines-Niagara',
            543: 'Kitchener-Cambridge-Waterloo',
            550: 'London',
            555: 'Windsor',
            559: 'Barrie',
            568: 'Greater Sudbury',
            595: 'Thunder Bay',
            602: 'Winnipeg',
            705: 'Regina',
            725: 'Saskatoon',
            825: 'Calgary',
            835: 'Edmonton',
            915: 'Kelowna',
            932: 'Abbotsford-Mission',
            933: 'Vancouver',
            935: 'Victoria',
        },
    },
    
    # =========================================================================
    # WAGES AND WEIGHTS
    # =========================================================================
    'HRLYEARN': {
        'description': 'Hourly earnings (nominal)',
        'category': 'wages',
        'dtype': 'float32',
        'use_in_analysis': False,  # Use REAL_HRLYEARN instead
        'is_target_related': True,
    },
    'FINALWT': {
        'description': 'Survey final weight (for population inference)',
        'category': 'weights',
        'dtype': 'float64',
        'use_in_analysis': True,
        'is_mandatory': True,
        'note': 'MUST use this weight for all population-level statistics',
    },
    
    # =========================================================================
    # DERIVED COLUMNS (created by pipeline)
    # =========================================================================
    'REAL_HRLYEARN': {
        'description': 'Hourly earnings (CPI-adjusted to 2010)',
        'category': 'wages',
        'dtype': 'float32',
        'use_in_analysis': False,  # This is the TARGET
        'is_target': True,
        'is_derived': True,
    },
    'IS_FEMALE': {
        'description': 'Binary female indicator (1=Female, 0=Male)',
        'category': 'derived',
        'dtype': 'int8',
        'use_in_analysis': True,
        'is_derived': True,
        'derived_from': 'GENDER',
    },
    'EXPERIENCE_PROXY': {
        'description': 'Approximate years of experience (age - education years - 6)',
        'category': 'derived',
        'dtype': 'float32',
        'use_in_analysis': True,
        'is_derived': True,
        'note': 'Mincer-style experience proxy',
    },
}


# =============================================================================
# COLUMN GROUPINGS
# =============================================================================

# Core columns always needed
CORE_COLS = ['SURVYEAR', 'SURVMNTH', 'FINALWT', 'HRLYEARN', 'GENDER']

# Demographics
DEMOGRAPHIC_COLS = ['GENDER', 'AGE_12', 'AGE_6', 'MARSTAT', 'IMMIG', 'EFAMTYPE', 'AGYOWNK']

# Human capital (education and experience)
HUMAN_CAPITAL_COLS = ['EDUC', 'TENURE']

# Job characteristics
JOB_COLS = ['NOC_10', 'NOC_43', 'NAICS_21', 'COWMAIN', 'PERMTEMP', 
           'FTPTMAIN', 'UNION', 'ESTSIZE', 'FIRMSIZE', 'MJH', 'WHYPT']

# Work hours
HOURS_COLS = ['UHRSMAIN', 'AHRSMAIN', 'UTOTHRS', 'ATOTHRS']

# Geographic
GEOGRAPHIC_COLS = ['PROV', 'CMA']

# Time
TIME_COLS = ['SURVYEAR', 'SURVMNTH']

# Wages (TARGET - never use as features!)
WAGE_COLS = ['HRLYEARN', 'REAL_HRLYEARN']

# Weights
WEIGHT_COLS = ['FINALWT']

# Key variables for stratification
KEY_ANALYSIS_COLS = [
    'GENDER', 'PROV', 'NOC_10', 'EDUC', 'TENURE', 'IMMIG', 
    'COWMAIN', 'NAICS_21', 'UNION', 'FTPTMAIN', 'AGE_12'
]

# All safe columns (no wage leakage risk)
ALL_SAFE_COLS = (
    DEMOGRAPHIC_COLS + HUMAN_CAPITAL_COLS + JOB_COLS + 
    HOURS_COLS + GEOGRAPHIC_COLS + TIME_COLS + WEIGHT_COLS
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_column_description(col: str) -> str:
    """Get human-readable description of a column."""
    if col in LFS_COLUMN_METADATA:
        return LFS_COLUMN_METADATA[col].get('description', col)
    return col


def get_column_values(col: str) -> dict:
    """Get value mappings for a column."""
    if col in LFS_COLUMN_METADATA:
        return LFS_COLUMN_METADATA[col].get('values', {})
    return {}


def get_columns_by_category(category: str) -> List[str]:
    """Get all columns in a category."""
    return [
        col for col, meta in LFS_COLUMN_METADATA.items()
        if meta.get('category') == category
    ]


def get_key_variables() -> List[str]:
    """Get columns marked as key analysis variables."""
    return [
        col for col, meta in LFS_COLUMN_METADATA.items()
        if meta.get('is_key_variable', False)
    ]


def get_target_related_columns() -> List[str]:
    """Get columns related to the target (for leakage prevention)."""
    return [
        col for col, meta in LFS_COLUMN_METADATA.items()
        if meta.get('is_target_related', False) or meta.get('is_target', False)
    ]


def is_safe_feature(col: str) -> bool:
    """Check if a column is safe to use as a feature (no leakage risk)."""
    if col not in LFS_COLUMN_METADATA:
        return False
    meta = LFS_COLUMN_METADATA[col]
    return (
        not meta.get('is_target', False) and 
        not meta.get('is_target_related', False) and
        meta.get('use_in_analysis', False)
    )


def get_all_safe_features() -> List[str]:
    """Get all columns that are safe to use as ML features."""
    return [col for col in LFS_COLUMN_METADATA if is_safe_feature(col)]
