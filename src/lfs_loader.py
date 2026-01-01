"""
LFS PUMF Data Loader for EquiPay Canada
========================================

This module loads actual Labour Force Survey Public Use Microdata Files (PUMF)
and integrates macroeconomic context for comprehensive pay equity analysis.

Key Features:
- Loads real LFS microdata (2010-2025)
- Applies survey weights (FINALWT) for population inference
- Converts nominal wages to real wages using CPI
- Adds macroeconomic context (unemployment, GDP, recession flags)
- Provides properly weighted statistics

Data Sources:
- LFS PUMF: Statistics Canada Catalogue 71M0001X
- Macro Data: Statistics Canada / Bank of Canada
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
import logging
import re

# Import from centralized constants
from .constants import (
    COLS, GENDER_CODES, PROVINCE_CODES, EDUCATION_CODES,
    AGE_6_CODES, AGE_12_CODES, NOC_10_CODES, NAICS_21_CODES,
    FTPT_CODES, UNION_CODES, PERMTEMP_CODES,
    MIN_HOURLY_WAGE, MAX_HOURLY_WAGE,
    DATA_SCOPE_START, DATA_SCOPE_END
)

from .macro_data import (
    MACRO_DATA, 
    get_deflator, 
    get_economic_period,
    add_macro_to_dataframe,
    BASE_YEAR
)

logger = logging.getLogger(__name__)


# =============================================================================
# LFS CODE MAPPINGS (imported from constants for backward compatibility)
# =============================================================================

GENDER_MAP = GENDER_CODES
PROVINCE_MAP = PROVINCE_CODES
EDUCATION_MAP = EDUCATION_CODES
AGE_6_MAP = AGE_6_CODES
NOC_10_MAP = NOC_10_CODES
NAICS_21_MAP = NAICS_21_CODES
FTPT_MAP = FTPT_CODES
UNION_MAP = UNION_CODES
PERMTEMP_MAP = PERMTEMP_CODES


# =============================================================================
# LFS DATA LOADER CLASS
# =============================================================================

class LFSDataLoader:
    """
    Loader for LFS PUMF microdata with macroeconomic integration.
    
    Usage:
        loader = LFSDataLoader(data_dir='data/raw/lfs')
        df = loader.load_and_process(years=range(2010, 2026))
        
        # Get weighted statistics
        stats = loader.weighted_wage_stats(df, by=['GENDER', 'SURVYEAR'])
    """
    
    def __init__(self, data_dir: Union[str, Path] = 'data/raw/lfs'):
        """
        Initialize the LFS loader.
        
        Args:
            data_dir: Directory containing LFS PUMF files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Columns we need for pay equity analysis
        self.core_columns = [
            'REC_NUM',      # Record ID
            'SURVYEAR',     # Survey year
            'SURVMNTH',     # Survey month
            'FINALWT',      # Survey weight
            'GENDER',       # Gender
            'HRLYEARN',     # Hourly earnings (target variable)
            'PROV',         # Province
            'EDUC',         # Education level
            'AGE_6',        # Age group
            'NOC_10',       # Occupation (10 categories)
            'NAICS_21',     # Industry
            'TENURE',       # Job tenure
            'FTPTMAIN',     # Full-time/Part-time
            'UNION',        # Union status
            'PERMTEMP',     # Permanent/Temporary
            'UHRSMAIN',     # Usual hours
        ]
        
        # Optional columns (may not be in all years)
        self.optional_columns = [
            'NOC_43',       # Occupation (43 categories)
            'AGE_12',       # Age (12 categories)
            'ESTSIZE',      # Establishment size
            'FIRMSIZE',     # Firm size
            'IMMIG',        # Immigration status
            'CMA',          # Census Metropolitan Area
            'MARSTAT',      # Marital status
        ]
    
    def load_pumf_file(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """
        Load a single LFS PUMF file.
        
        Args:
            filepath: Path to the PUMF file (CSV, SAS, or fixed-width)
            
        Returns:
            DataFrame with LFS records
        """
        filepath = Path(filepath)
        logger.info(f"Loading LFS file: {filepath}")
        
        if filepath.suffix.lower() == '.csv':
            # Try UTF-8 first, fall back to latin-1 for French characters
            try:
                df = pd.read_csv(filepath, low_memory=False, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(filepath, low_memory=False, encoding='latin-1')
        elif filepath.suffix.lower() in ['.sas7bdat', '.sas']:
            df = pd.read_sas(filepath)
        else:
            # Assume fixed-width format (common for older PUMF)
            # Would need to define column positions
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        # Standardize column names to uppercase
        df.columns = df.columns.str.upper()
        
        # Check if this is a codebook/data dictionary file vs actual microdata
        # Codebook files have columns like Field_Champ, Position_Position, etc.
        codebook_indicators = ['FIELD_CHAMP', 'POSITION_POSITION', 'LENGTH_LONGUEUR', 
                               'VARIABLE_VARIABLE', 'ENGLISHLABEL_ETIQUETTEANGLAIS']
        if any(col in df.columns for col in codebook_indicators):
            logger.warning(f"File {filepath} is a codebook/data dictionary file, not microdata. "
                          f"Contains {len(df)} rows of metadata describing variable definitions.")
            # Return empty DataFrame - this file has no wage data to analyze
            return pd.DataFrame()
        
        # Standardize SEX to GENDER (older LFS files use SEX)
        if 'SEX' in df.columns and 'GENDER' not in df.columns:
            df['GENDER'] = df['SEX']
            logger.info(f"Renamed SEX to GENDER for {filepath}")
        
        # Select available columns
        available_cols = [c for c in self.core_columns if c in df.columns]
        optional_avail = [c for c in self.optional_columns if c in df.columns]
        
        # If no core columns found, this isn't a valid data file
        if not available_cols:
            logger.warning(f"File {filepath} has no recognizable LFS columns. Columns found: {list(df.columns)[:10]}")
            return pd.DataFrame()
        
        df = df[available_cols + optional_avail]
        
        logger.info(f"Loaded {len(df):,} records with columns: {list(df.columns)}")
        return df
    
    def load_all_years(self, years: range = range(2010, 2026), 
                       months: Optional[range] = None) -> pd.DataFrame:
        """
        Load LFS data for multiple years/months.
        
        Args:
            years: Range of years to load
            months: Range of months to load (1-12). If None, loads all months.
            
        Returns:
            Combined DataFrame with all years/months
        """
        all_data = []
        months = months or range(1, 13)
        files_found = 0
        
        for year in years:
            year_short = str(year)[-2:]  # e.g., "10" for 2010
            
            # Try monthly file patterns for this year
            for month in months:
                month_str = str(month).zfill(2)
                monthly_patterns = [
                    # Statistics Canada historical naming: lfs_YYYY_pubMMYY.csv
                    self.data_dir / f"lfs_{year}_pub{month_str}{year_short}.csv",
                    # 2025 monthly naming: lfs_2025_MM.csv
                    self.data_dir / f"lfs_{year}_{month_str}.csv",
                    # Other common patterns
                    self.data_dir / f"lfs_{year}{month_str}.csv",
                    self.data_dir / f"LFS_{year}_{month_str}.csv",
                    self.data_dir / f"LFS_{year}{month_str}.csv",
                    self.data_dir / f"lfs-{year}-{month_str}.csv",
                    self.data_dir / f"pumf_{year}_{month_str}.csv",
                    self.data_dir / f"pub{year}{month_str}.csv",
                    # Nested by year
                    self.data_dir / str(year) / f"lfs_{month_str}.csv",
                    self.data_dir / str(year) / f"lfs_{year}_{month_str}.csv",
                ]
                
                for pattern in monthly_patterns:
                    if pattern.exists():
                        df = self.load_pumf_file(pattern)
                        # Skip empty DataFrames (e.g., from codebook files)
                        if len(df) == 0:
                            continue
                        # Ensure SURVYEAR and SURVMNTH are set if not in file
                        if 'SURVYEAR' not in df.columns:
                            df['SURVYEAR'] = year
                        if 'SURVMNTH' not in df.columns:
                            df['SURVMNTH'] = month
                        all_data.append(df)
                        files_found += 1
                        break
            
            # Fallback: try single yearly file if no monthly files found
            if not any(d.get('SURVYEAR', pd.Series()).eq(year).any() if 'SURVYEAR' in d.columns else False for d in all_data):
                yearly_patterns = [
                    self.data_dir / f"lfs_{year}.csv",
                    self.data_dir / f"LFS_{year}.csv",
                    self.data_dir / f"lfs{year}.csv",
                    self.data_dir / f"pumf_{year}.csv",
                ]
                
                for pattern in yearly_patterns:
                    if pattern.exists():
                        df = self.load_pumf_file(pattern)
                        if len(df) > 0:
                            all_data.append(df)
                            files_found += 1
                            break
        
        if not all_data:
            raise FileNotFoundError(
                f"No LFS files found in {self.data_dir}\n"
                f"Expected patterns:\n"
                f"  - Yearly: lfs_YYYY.csv\n"
                f"  - Monthly: lfs_YYYY_MM.csv or lfs_YYYYMM.csv\n"
                f"  - Nested: YYYY/lfs_MM.csv"
            )
        
        combined = pd.concat(all_data, ignore_index=True)
        logger.info(f"Combined dataset: {len(combined):,} records from {files_found} files")
        
        return combined
    
    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process LFS data: clean, map codes, add derived variables.
        
        Args:
            df: Raw LFS DataFrame
            
        Returns:
            Processed DataFrame with labels and derived variables
        """
        df = df.copy()
        original_count = len(df)
        
        # =====================================================================
        # PRESERVE ALL RECORDS - Only flag valid wages, don't delete records
        # =====================================================================
        
        # Convert HRLYEARN to numeric, coercing errors to NaN
        df['HRLYEARN'] = pd.to_numeric(df['HRLYEARN'], errors='coerce')
        
        # Flag records with valid hourly earnings (positive and within reasonable range)
        # LFS PUMF uses various codes for missing/not applicable:
        # - 0 or blank: Not applicable (e.g., self-employed, unpaid)
        # - Very high values (>500): Likely top-coded or errors
        df['HAS_VALID_WAGE'] = (
            df['HRLYEARN'].notna() & 
            (df['HRLYEARN'] > 0) & 
            (df['HRLYEARN'] >= 5) & 
            (df['HRLYEARN'] <= 500)
        )
        
        valid_wage_count = df['HAS_VALID_WAGE'].sum()
        logger.info(f"Records with valid wages: {valid_wage_count:,} / {original_count:,} "
                   f"({100*valid_wage_count/original_count:.1f}%)")
        
        # =====================================================================
        # ADD LABELS FOR CODED VARIABLES
        # =====================================================================
        
        df['GENDER_LABEL'] = df['GENDER'].map(GENDER_MAP)
        df['PROV_LABEL'] = df['PROV'].map(PROVINCE_MAP)
        df['EDUC_LABEL'] = df['EDUC'].map(EDUCATION_MAP)
        df['AGE_LABEL'] = df['AGE_6'].map(AGE_6_MAP)
        df['NOC_LABEL'] = df['NOC_10'].map(NOC_10_MAP)
        df['FTPT_LABEL'] = df['FTPTMAIN'].map(FTPT_MAP)
        
        if 'UNION' in df.columns:
            df['UNION_LABEL'] = df['UNION'].map(UNION_MAP)
        if 'PERMTEMP' in df.columns:
            df['PERMTEMP_LABEL'] = df['PERMTEMP'].map(PERMTEMP_MAP)
        if 'NAICS_21' in df.columns:
            df['INDUSTRY_LABEL'] = df['NAICS_21'].map(NAICS_21_MAP)
        
        # =====================================================================
        # ADD TIME VARIABLES
        # =====================================================================
        
        df['YEAR'] = df['SURVYEAR'].astype(int)
        df['MONTH'] = df['SURVMNTH'].astype(int)
        df['YEAR_MONTH'] = df['YEAR'].astype(str) + '-' + df['MONTH'].astype(str).str.zfill(2)
        
        # =====================================================================
        # ADD MACROECONOMIC CONTEXT
        # =====================================================================
        
        df = add_macro_to_dataframe(df, year_col='YEAR')
        
        # Economic period classification
        df['ECONOMIC_PERIOD'] = df['YEAR'].apply(get_economic_period)
        
        # =====================================================================
        # CALCULATE REAL WAGES (INFLATION-ADJUSTED)
        # Only for records with valid wages
        # =====================================================================
        
        # Initialize with NaN
        df['REAL_HRLYEARN'] = np.nan
        df['LOG_HRLYEARN'] = np.nan
        df['LOG_REAL_HRLYEARN'] = np.nan
        
        # Calculate only for valid wages
        valid_mask = df['HAS_VALID_WAGE']
        if valid_mask.any():
            df.loc[valid_mask, 'REAL_HRLYEARN'] = df.loc[valid_mask].apply(
                lambda row: row['HRLYEARN'] * get_deflator(row['YEAR']), 
                axis=1
            )
            # Log wages for regression analysis (only for positive values)
            df.loc[valid_mask, 'LOG_HRLYEARN'] = np.log(df.loc[valid_mask, 'HRLYEARN'])
            df.loc[valid_mask, 'LOG_REAL_HRLYEARN'] = np.log(df.loc[valid_mask, 'REAL_HRLYEARN'])
        
        # =====================================================================
        # ADD ANALYSIS FLAGS
        # =====================================================================
        
        # Binary gender for regression (handle missing)
        df['IS_FEMALE'] = (df['GENDER'] == 2).astype(int)
        
        # Full-time flag (handle missing)
        df['IS_FULLTIME'] = (df['FTPTMAIN'] == 1).astype(int)
        
        # High education flag (handle missing)
        df['HAS_DEGREE'] = (df['EDUC'] >= 4).astype(int)
        
        # Summary stats
        valid_wage_pct = 100 * df['HAS_VALID_WAGE'].sum() / len(df)
        logger.info(f"Processing complete. Final dataset: {len(df):,} records "
                   f"({valid_wage_pct:.1f}% with valid wages)")
        
        return df
    
    def load_and_process(self, years: range = range(2010, 2026),
                          months: Optional[range] = None) -> pd.DataFrame:
        """
        Load and process LFS data in one step.
        
        Args:
            years: Range of years to load
            months: Range of months (1-12). If None, loads all months.
            
        Returns:
            Fully processed DataFrame ready for analysis
        """
        df = self.load_all_years(years, months)
        df = self.process_data(df)
        return df
    
    def list_available_files(self) -> Dict[int, List[int]]:
        """
        Scan directory and list available LFS files by year and month.
        
        Returns:
            Dictionary: {year: [list of available months]}
        """
        import re
        
        available = {}
        
        if not self.data_dir.exists():
            return available
        
        # Scan all CSV files
        for f in self.data_dir.rglob("*.csv"):
            filename = f.stem.lower()
            
            # Try to extract year and month from filename
            # Pattern: lfs_2020_01, lfs202001, lfs-2020-01
            patterns = [
                r'lfs[_-]?(\d{4})[_-]?(\d{2})',  # lfs_2020_01 or lfs202001
                r'pub(\d{4})(\d{2})',             # pub202001
                r'lfs[_-]?(\d{4})$',              # lfs_2020 (yearly)
            ]
            
            for pattern in patterns:
                match = re.search(pattern, filename)
                if match:
                    groups = match.groups()
                    year = int(groups[0])
                    month = int(groups[1]) if len(groups) > 1 else 0  # 0 = yearly file
                    
                    if year not in available:
                        available[year] = []
                    if month not in available[year]:
                        available[year].append(month)
                    break
        
        # Sort months within each year
        for year in available:
            available[year] = sorted(available[year])
        
        return dict(sorted(available.items()))
    
    def get_data_coverage_summary(self) -> str:
        """
        Get a summary of available data coverage.
        
        Returns:
            Formatted string showing available years and months
        """
        available = self.list_available_files()
        
        if not available:
            return f"No LFS files found in {self.data_dir}"
        
        summary = f"LFS Data Coverage in {self.data_dir}:\n"
        summary += "=" * 50 + "\n"
        
        for year, months in available.items():
            if months == [0]:
                summary += f"  {year}: Yearly file\n"
            else:
                month_str = ', '.join(str(m) for m in months if m > 0)
                n_months = len([m for m in months if m > 0])
                summary += f"  {year}: {n_months} months ({month_str})\n"
        
        total_years = len(available)
        total_months = sum(len([m for m in months if m > 0]) for months in available.values())
        
        summary += "=" * 50 + "\n"
        summary += f"Total: {total_years} years, {total_months} monthly files\n"
        
        return summary
    
    # =========================================================================
    # WEIGHTED STATISTICS METHODS
    # =========================================================================
    
    def weighted_mean(self, df: pd.DataFrame, col: str, 
                      weight_col: str = 'FINALWT') -> float:
        """Calculate weighted mean."""
        return np.average(df[col], weights=df[weight_col])
    
    def weighted_median(self, df: pd.DataFrame, col: str,
                        weight_col: str = 'FINALWT') -> float:
        """Calculate weighted median."""
        sorted_idx = np.argsort(df[col].values)
        sorted_values = df[col].values[sorted_idx]
        sorted_weights = df[weight_col].values[sorted_idx]
        cumsum = np.cumsum(sorted_weights)
        cutoff = cumsum[-1] / 2
        return sorted_values[np.searchsorted(cumsum, cutoff)]
    
    def weighted_percentile(self, df: pd.DataFrame, col: str, 
                            percentile: float, weight_col: str = 'FINALWT') -> float:
        """Calculate weighted percentile."""
        sorted_idx = np.argsort(df[col].values)
        sorted_values = df[col].values[sorted_idx]
        sorted_weights = df[weight_col].values[sorted_idx]
        cumsum = np.cumsum(sorted_weights)
        cutoff = cumsum[-1] * (percentile / 100)
        return sorted_values[np.searchsorted(cumsum, cutoff)]
    
    def weighted_std(self, df: pd.DataFrame, col: str,
                     weight_col: str = 'FINALWT') -> float:
        """Calculate weighted standard deviation."""
        mean = self.weighted_mean(df, col, weight_col)
        variance = np.average((df[col] - mean)**2, weights=df[weight_col])
        return np.sqrt(variance)
    
    def weighted_wage_stats(self, df: pd.DataFrame, 
                            by: Optional[List[str]] = None,
                            wage_col: str = 'HRLYEARN',
                            real_wage_col: str = 'REAL_HRLYEARN') -> pd.DataFrame:
        """
        Calculate comprehensive weighted wage statistics.
        
        Args:
            df: Processed LFS DataFrame
            by: Grouping columns (e.g., ['GENDER', 'YEAR'])
            wage_col: Nominal wage column
            real_wage_col: Real wage column
            
        Returns:
            DataFrame with weighted statistics
        """
        def calc_stats(group):
            n = len(group)
            pop_n = group['FINALWT'].sum()
            
            return pd.Series({
                'n_sample': n,
                'n_population': pop_n,
                'mean_nominal': self.weighted_mean(group, wage_col),
                'mean_real': self.weighted_mean(group, real_wage_col),
                'median_nominal': self.weighted_median(group, wage_col),
                'median_real': self.weighted_median(group, real_wage_col),
                'std_nominal': self.weighted_std(group, wage_col),
                'p10_nominal': self.weighted_percentile(group, wage_col, 10),
                'p25_nominal': self.weighted_percentile(group, wage_col, 25),
                'p75_nominal': self.weighted_percentile(group, wage_col, 75),
                'p90_nominal': self.weighted_percentile(group, wage_col, 90),
            })
        
        if by:
            stats = df.groupby(by).apply(calc_stats).reset_index()
        else:
            stats = calc_stats(df).to_frame().T
        
        return stats
    
    def calculate_gender_gap(self, df: pd.DataFrame,
                             by: Optional[List[str]] = None,
                             use_real_wages: bool = True) -> pd.DataFrame:
        """
        Calculate gender pay gap with proper weighting.
        
        Gap = (Male_wage - Female_wage) / Male_wage * 100
        
        Args:
            df: Processed LFS DataFrame
            by: Additional grouping columns (e.g., ['YEAR', 'NOC_10'])
            use_real_wages: Use real (inflation-adjusted) wages
            
        Returns:
            DataFrame with gap statistics
        """
        wage_col = 'REAL_HRLYEARN' if use_real_wages else 'HRLYEARN'
        
        group_cols = ['GENDER_LABEL'] + (by or [])
        
        stats = self.weighted_wage_stats(df, by=group_cols, 
                                         wage_col=wage_col,
                                         real_wage_col=wage_col)
        
        # Pivot to get male/female side by side
        if by:
            pivot = stats.pivot_table(
                index=by,
                columns='GENDER_LABEL',
                values=['mean_nominal', 'median_nominal', 'n_population']
            ).reset_index()
            pivot.columns = ['_'.join(col).strip('_') for col in pivot.columns.values]
        else:
            male_stats = stats[stats['GENDER_LABEL'] == 'Male'].iloc[0]
            female_stats = stats[stats['GENDER_LABEL'] == 'Female'].iloc[0]
            
            return pd.DataFrame([{
                'male_mean': male_stats['mean_nominal'],
                'female_mean': female_stats['mean_nominal'],
                'gap_mean': (male_stats['mean_nominal'] - female_stats['mean_nominal']) / male_stats['mean_nominal'] * 100,
                'male_median': male_stats['median_nominal'],
                'female_median': female_stats['median_nominal'],
                'gap_median': (male_stats['median_nominal'] - female_stats['median_nominal']) / male_stats['median_nominal'] * 100,
                'n_male': male_stats['n_population'],
                'n_female': female_stats['n_population'],
            }])
        
        # Calculate gaps
        pivot['gap_mean'] = (
            (pivot['mean_nominal_Male'] - pivot['mean_nominal_Female']) / 
            pivot['mean_nominal_Male'] * 100
        )
        pivot['gap_median'] = (
            (pivot['median_nominal_Male'] - pivot['median_nominal_Female']) / 
            pivot['median_nominal_Male'] * 100
        )
        
        return pivot


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def load_lfs_data(data_dir: str = 'data/raw/lfs', 
                  years: range = range(2010, 2026),
                  months: Optional[range] = None) -> pd.DataFrame:
    """
    Convenience function to load and process LFS data.
    
    Args:
        data_dir: Directory with LFS files
        years: Years to load
        months: Months to load (1-12). If None, loads all months.
        
    Returns:
        Processed DataFrame
    """
    loader = LFSDataLoader(data_dir)
    return loader.load_and_process(years, months)


def get_annual_gap_series(df: pd.DataFrame, 
                          use_real_wages: bool = True) -> pd.DataFrame:
    """
    Get annual gender pay gap time series.
    
    Args:
        df: Processed LFS DataFrame
        use_real_wages: Use inflation-adjusted wages
        
    Returns:
        DataFrame with annual gap by year
    """
    loader = LFSDataLoader()
    return loader.calculate_gender_gap(df, by=['YEAR'], use_real_wages=use_real_wages)


def get_gap_decomposition(df: pd.DataFrame,
                          by_cols: List[str] = ['NOC_10', 'EDUC', 'PROV']) -> Dict[str, pd.DataFrame]:
    """
    Decompose gender gap by multiple dimensions.
    
    Args:
        df: Processed LFS DataFrame
        by_cols: Columns to decompose by
        
    Returns:
        Dictionary of gap DataFrames by dimension
    """
    loader = LFSDataLoader()
    results = {}
    
    for col in by_cols:
        results[col] = loader.calculate_gender_gap(df, by=['YEAR', col])
    
    return results


# =============================================================================
# SUMMARY STATISTICS FOR REPORTS
# =============================================================================

def generate_lfs_summary(df: pd.DataFrame) -> str:
    """
    Generate a text summary of the LFS dataset.
    
    Args:
        df: Processed LFS DataFrame
        
    Returns:
        Formatted summary string
    """
    loader = LFSDataLoader()
    
    summary = """
LFS Data Summary
================

Dataset Overview:
-----------------
Total Records: {:,}
Years Covered: {} to {}
Total Population (Weighted): {:,.0f}

Gender Distribution:
{}

Provincial Distribution:
{}

Education Distribution:
{}

Wage Statistics (Real, 2010 Dollars):
-------------------------------------
Overall Mean: ${:.2f}/hour
Overall Median: ${:.2f}/hour

Gender Gap Summary:
{}
""".format(
        len(df),
        df['YEAR'].min(),
        df['YEAR'].max(),
        df['FINALWT'].sum(),
        df.groupby('GENDER_LABEL')['FINALWT'].sum().to_string(),
        df.groupby('PROV_LABEL')['FINALWT'].sum().to_string(),
        df.groupby('EDUC_LABEL')['FINALWT'].sum().to_string(),
        loader.weighted_mean(df, 'REAL_HRLYEARN'),
        loader.weighted_median(df, 'REAL_HRLYEARN'),
        loader.calculate_gender_gap(df).to_string(),
    )
    
    return summary


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    print("LFS Data Loader")
    print("===============")
    print("\nTo use this loader, place LFS PUMF files in data/raw/lfs/")
    print("\nSupported file naming patterns:")
    print("  Monthly files:")
    print("    - lfs_YYYY_MM.csv  (e.g., lfs_2020_01.csv)")
    print("    - lfs_YYYYMM.csv   (e.g., lfs_202001.csv)")
    print("    - lfs-YYYY-MM.csv  (e.g., lfs-2020-01.csv)")
    print("    - YYYY/lfs_MM.csv  (nested by year)")
    print("  Yearly files:")
    print("    - lfs_YYYY.csv     (e.g., lfs_2020.csv)")
    print("\nExample usage:")
    print("  from src.lfs_loader import LFSDataLoader")
    print("  loader = LFSDataLoader('data/raw/lfs')")
    print("  ")
    print("  # Check available data")
    print("  print(loader.get_data_coverage_summary())")
    print("  ")
    print("  # Load all data (2010-2025, all months)")
    print("  df = loader.load_and_process()")
    print("  ")
    print("  # Load specific period")
    print("  df = loader.load_and_process(years=range(2020, 2026), months=range(1, 7))")
    print("  ")
    print("  # Calculate gender pay gap by year")
    print("  gap = loader.calculate_gender_gap(df, by=['YEAR'])")
