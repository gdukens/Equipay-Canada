"""
Centralized Macroeconomic Data for EquiPay Canada
==================================================

This module provides consistent macroeconomic data for Canada (2010-2025).

DATA SCOPE:
- This is ONE OF ONLY TWO data sources used in this project:
  1. LFS PUMF microdata (primary source)
  2. Macroeconomic data (this module)

Data sources:
- CPI: Statistics Canada Table 18-10-0005-01
- GDP Growth: Statistics Canada Table 36-10-0104-01
- Unemployment: Statistics Canada Table 14-10-0287-01
- Interest Rate: Bank of Canada Policy Rate
"""

import pandas as pd
import numpy as np

# Import BASE_YEAR from constants (single source of truth)
# Handle both relative (from package) and absolute (direct) imports
try:
    from .constants import BASE_YEAR
except ImportError:
    from constants import BASE_YEAR
from typing import Dict, Optional

# ===========================================================================
# CANADIAN MACROECONOMIC DATA (2010-2025)
# ===========================================================================
# All values are annual averages from official Statistics Canada sources
# 2025 values are estimates/projections

MACRO_DATA = {
    2010: {'cpi': 116.5, 'gdp_growth': 3.1, 'unemployment': 8.1, 'interest_rate': 0.59},
    2011: {'cpi': 119.9, 'gdp_growth': 3.1, 'unemployment': 7.5, 'interest_rate': 1.00},
    2012: {'cpi': 121.7, 'gdp_growth': 1.8, 'unemployment': 7.3, 'interest_rate': 1.00},
    2013: {'cpi': 122.8, 'gdp_growth': 2.3, 'unemployment': 7.1, 'interest_rate': 1.00},
    2014: {'cpi': 125.2, 'gdp_growth': 2.9, 'unemployment': 6.9, 'interest_rate': 1.00},
    2015: {'cpi': 126.6, 'gdp_growth': 0.7, 'unemployment': 6.9, 'interest_rate': 0.63},
    2016: {'cpi': 128.4, 'gdp_growth': 1.0, 'unemployment': 7.0, 'interest_rate': 0.50},
    2017: {'cpi': 130.4, 'gdp_growth': 3.0, 'unemployment': 6.3, 'interest_rate': 0.71},
    2018: {'cpi': 133.4, 'gdp_growth': 2.8, 'unemployment': 5.8, 'interest_rate': 1.42},
    2019: {'cpi': 136.0, 'gdp_growth': 1.9, 'unemployment': 5.7, 'interest_rate': 1.75},
    2020: {'cpi': 137.0, 'gdp_growth': -5.1, 'unemployment': 9.5, 'interest_rate': 0.50},
    2021: {'cpi': 141.6, 'gdp_growth': 5.0, 'unemployment': 7.5, 'interest_rate': 0.25},
    2022: {'cpi': 151.2, 'gdp_growth': 3.8, 'unemployment': 5.3, 'interest_rate': 2.50},
    2023: {'cpi': 157.1, 'gdp_growth': 1.2, 'unemployment': 5.4, 'interest_rate': 4.75},
    2024: {'cpi': 161.5, 'gdp_growth': 1.5, 'unemployment': 6.1, 'interest_rate': 4.25},
    2025: {'cpi': 165.0, 'gdp_growth': 1.8, 'unemployment': 6.3, 'interest_rate': 3.75},  # Estimated
}

# Use BASE_YEAR from constants (single source of truth - imported above)
BASE_CPI = MACRO_DATA[BASE_YEAR]['cpi']

# Economic period classifications
ECONOMIC_PERIODS = {
    'pre_crisis': (2010, 2014),
    'oil_shock': (2015, 2016),
    'recovery': (2017, 2019),
    'covid': (2020, 2021),
    'inflation': (2022, 2023),
    'stabilization': (2024, 2025),
}


def get_macro_dataframe() -> pd.DataFrame:
    """Return macroeconomic data as a DataFrame."""
    df = pd.DataFrame.from_dict(MACRO_DATA, orient='index')
    df.index.name = 'year'
    df = df.reset_index()
    
    # Add derived variables
    df['inflation'] = df['cpi'].pct_change() * 100
    df['real_gdp_index'] = (1 + df['gdp_growth']/100).cumprod() * 100
    df['recession'] = (df['gdp_growth'] < 0).astype(int)
    df['covid'] = df['year'].isin([2020, 2021]).astype(int)
    df['high_inflation'] = (df['inflation'] > 3).astype(int)
    
    return df


def get_cpi(year: int) -> float:
    """Get CPI for a specific year."""
    return MACRO_DATA.get(year, {}).get('cpi', np.nan)


def get_deflator(year: int, base_year: int = BASE_YEAR) -> float:
    """Get price deflator to convert nominal to real values."""
    base_cpi = MACRO_DATA.get(base_year, {}).get('cpi', 100)
    year_cpi = MACRO_DATA.get(year, {}).get('cpi', np.nan)
    return base_cpi / year_cpi if year_cpi else np.nan


def adjust_for_inflation(value: float, year: int, base_year: int = BASE_YEAR) -> float:
    """Convert nominal value to real value (inflation-adjusted)."""
    deflator = get_deflator(year, base_year)
    return value * deflator if not np.isnan(deflator) else np.nan


def add_macro_to_dataframe(df: pd.DataFrame, year_col: str = 'year') -> pd.DataFrame:
    """Add macroeconomic variables to any DataFrame with a year column."""
    df = df.copy()
    
    for var in ['cpi', 'gdp_growth', 'unemployment', 'interest_rate']:
        df[var] = df[year_col].map(lambda y: MACRO_DATA.get(y, {}).get(var, np.nan))
    
    # Add derived variables
    df['inflation'] = df['cpi'].pct_change() * 100
    df['recession'] = (df['gdp_growth'] < 0).astype(int)
    df['covid'] = df[year_col].isin([2020, 2021]).astype(int)
    df['deflator'] = df[year_col].map(lambda y: get_deflator(y))
    
    return df


def get_economic_period(year: int) -> str:
    """Classify year into economic period."""
    for period, (start, end) in ECONOMIC_PERIODS.items():
        if start <= year <= end:
            return period
    return 'unknown'


def get_macro_controls_summary() -> str:
    """Return a formatted summary of macro controls for reports."""
    df = get_macro_dataframe()
    
    summary = """
Macroeconomic Controls (2010-2024)
==================================
Source: Statistics Canada / Bank of Canada

Variable Definitions:
- CPI: Consumer Price Index (2002=100)
- GDP Growth: Real GDP annual % change
- Unemployment: Unemployment rate (%)
- Interest Rate: Bank of Canada policy rate (%)

Period Averages:
"""
    for period, (start, end) in ECONOMIC_PERIODS.items():
        period_df = df[(df['year'] >= start) & (df['year'] <= end)]
        summary += f"\n{period.replace('_', ' ').title()} ({start}-{end}):\n"
        summary += f"  Avg CPI: {period_df['cpi'].mean():.1f}\n"
        summary += f"  Avg GDP Growth: {period_df['gdp_growth'].mean():.1f}%\n"
        summary += f"  Avg Unemployment: {period_df['unemployment'].mean():.1f}%\n"
    
    return summary
