"""
Data Pipeline Module for EquiPay Canada
========================================

Handles loading and processing of LFS PUMF data for pay equity analysis.

DATA SCOPE:
This project uses ONLY two data sources:
1. LFS PUMF microdata (2010-2025) - Statistics Canada catalogue 71M0001X
2. Macroeconomic data - CPI, GDP, unemployment, interest rates (see macro_data.py)

No other external data sources are required.

Supports:
1. Real LFS PUMF microdata (preferred) - actual survey responses
2. Synthetic data generation - for testing/development when PUMF unavailable
"""

import os
import logging
import requests
import zipfile
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from datetime import datetime

import pandas as pd
import numpy as np
import yaml
from tqdm import tqdm

# Import centralized constants and mappings
from .constants import (
    COLS, GENDER_CODES, PROVINCE_CODES, EDUCATION_CODES,
    AGE_6_CODES, AGE_12_CODES, AGE_6_MIDPOINTS, AGE_12_MIDPOINTS,
    NOC_10_CODES, NAICS_21_CODES, FTPT_CODES, UNION_CODES,
    PERMTEMP_CODES, ESTSIZE_CODES, MARSTAT_CODES,
    MIN_HOURLY_WAGE, MAX_HOURLY_WAGE,
    DATA_SCOPE_START, DATA_SCOPE_END,
    normalize_column_names, apply_labels
)

# Import macro data integration
from .macro_data import (
    add_macro_to_dataframe, get_deflator, get_economic_period, BASE_YEAR
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LFSDataPipeline:
    """
    Unified pipeline for processing Labour Force Survey data.
    
    Supports:
    - Real LFS PUMF microdata files
    - Statistics Canada Web Data Service API
    - Synthetic data generation for testing
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the pipeline.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config = self._load_config(config_path)
        self.raw_path = Path(self.config.get('data', {}).get('raw_path', 'data/raw'))
        self.processed_path = Path(self.config.get('data', {}).get('processed_path', 'data/processed'))
        self.lfs_path = self.raw_path / 'lfs'
        
        # Create directories
        self.raw_path.mkdir(parents=True, exist_ok=True)
        self.processed_path.mkdir(parents=True, exist_ok=True)
        self.lfs_path.mkdir(parents=True, exist_ok=True)
        
        # Data scope
        self.start_year = self.config.get('data', {}).get('start_year', DATA_SCOPE_START)
        self.end_year = self.config.get('data', {}).get('end_year', DATA_SCOPE_END)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}. Using defaults.")
            return {}
    
    # =========================================================================
    # DATA LOADING METHODS
    # =========================================================================
    
    def load_lfs_pumf(self, years: Optional[range] = None,
                      months: Optional[range] = None) -> pd.DataFrame:
        """
        Load actual LFS PUMF microdata files.
        
        This is the preferred method when real microdata is available.
        
        Args:
            years: Range of years to load (default: 2010-2025)
            months: Range of months (1-12). If None, loads all months.
            
        Returns:
            DataFrame with LFS microdata
        """
        from .lfs_loader import LFSDataLoader
        
        years = years or range(self.start_year, self.end_year + 1)
        
        loader = LFSDataLoader(self.lfs_path)
        
        # Check for available files
        available = loader.list_available_files()
        if not available:
            raise FileNotFoundError(
                f"No LFS PUMF files found in {self.lfs_path}\n"
                f"Please download LFS PUMF from Statistics Canada catalogue 71M0001X\n"
                f"and place files in: {self.lfs_path.absolute()}"
            )
        
        logger.info(loader.get_data_coverage_summary())
        
        df = loader.load_and_process(years=years, months=months)
        logger.info(f"Loaded {len(df):,} LFS PUMF records")
        
        return df
    
    def download_aggregate_data(self) -> List[pd.DataFrame]:
        """
        Download aggregate wage data from Statistics Canada Web Data Service API.
        
        This fetches aggregate statistics (not microdata) from key LFS tables.
        Use when real PUMF is not available.
        
        Returns:
            List of DataFrames with aggregate data
        """
        logger.info("=" * 60)
        logger.info("Downloading aggregate data from Statistics Canada API...")
        logger.info("=" * 60)
        
        BASE_URL = "https://www150.statcan.gc.ca/t1/wds/rest"
        
        # Key LFS-related tables (Product IDs)
        tables = {
            'wages_by_occupation': '14100064',
            'wages_by_industry': '14100340',
            'wages_by_sex_age': '14100063',
            'employment_by_sex': '14100287',
            'employment_by_occupation': '14100296',
        }
        
        all_data = []
        
        for name, pid in tables.items():
            try:
                logger.info(f"Fetching {name} (PID: {pid})...")
                
                url = f"{BASE_URL}/getFullTableDownloadCSV/{pid}/en"
                response = requests.get(url, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get('status') == 'SUCCESS':
                        csv_url = result.get('object')
                        if csv_url:
                            csv_response = requests.get(csv_url, timeout=120)
                            if csv_response.status_code == 200:
                                with zipfile.ZipFile(BytesIO(csv_response.content)) as z:
                                    csv_files = [f for f in z.namelist() if f.endswith('.csv')]
                                    if csv_files:
                                        with z.open(csv_files[0]) as f:
                                            df = pd.read_csv(f)
                                            df['source_table'] = name
                                            all_data.append(df)
                                            logger.info(f"  ✓ Downloaded {len(df):,} rows from {name}")
                                            
                                            # Save raw data
                                            raw_path = self.raw_path / f"{name}.csv"
                                            df.to_csv(raw_path, index=False)
            except Exception as e:
                logger.warning(f"  ✗ Failed to fetch {name}: {e}")
                continue
        
        if all_data:
            logger.info(f"\nSuccessfully downloaded {len(all_data)} tables!")
        else:
            logger.warning("No data downloaded from API.")
            
        return all_data
    
    def create_dataset_from_aggregate_data(self, n_samples: int = 50000) -> pd.DataFrame:
        """
        Create individual-level records from aggregate Statistics Canada data.
        
        This synthesizes microdata-like records using wage distributions
        from aggregate API data when real PUMF is not available.
        
        Args:
            n_samples: Number of records to generate
            
        Returns:
            DataFrame with synthesized individual records
        """
        logger.info("Creating dataset from aggregate Statistics Canada data...")
        
        wages_file = self.raw_path / 'wages_by_sex_age.csv'
        occupation_file = self.raw_path / 'wages_by_occupation.csv'
        
        if not wages_file.exists():
            logger.info("Aggregate data not found. Downloading from Statistics Canada...")
            self.download_aggregate_data()
        
        if not wages_file.exists():
            logger.warning("Could not load aggregate data. Falling back to synthetic.")
            return self.generate_synthetic_data(n_samples)
        
        return self._synthesize_from_aggregate(wages_file, occupation_file, n_samples)
    
    def _synthesize_from_aggregate(self, wages_file: Path, occupation_file: Path,
                                    n_samples: int = 50000) -> pd.DataFrame:
        """
        Synthesize individual records from aggregate wage data.
        
        Uses real wage distributions by gender, age, and province from Statistics Canada
        aggregate tables to generate realistic individual-level records.
        """
        logger.info("Synthesizing records from aggregate wage data...")
        
        # Load wage data with only needed columns
        needed_cols = ['REF_DATE', 'GEO', 'Wages', 'Gender', 'Age group', 'VALUE', 'Type of work']
        
        try:
            df = pd.read_csv(wages_file, low_memory=False, usecols=needed_cols)
        except ValueError:
            df = pd.read_csv(wages_file, low_memory=False, nrows=500000)
        
        logger.info(f"Loaded {len(df):,} rows from wages file")
        
        # Filter to scope (2010-2025) and relevant records
        df = df[df['REF_DATE'] >= f'{self.start_year}-01'].copy()
        df = df[df['REF_DATE'] <= f'{self.end_year}-12'].copy()
        df = df[df['Wages'].str.contains('Average hourly wage', na=False)]
        df = df[df['Gender'].isin(['Men+', 'Women+'])]
        df = df[df['GEO'] != 'Canada']  # Get provincial data
        df = df[pd.notna(df['VALUE']) & (df['VALUE'] > 0)]
        
        logger.info(f"Filtered to {len(df):,} relevant wage records")
        
        # Province mapping (label to code)
        prov_map = {v: k for k, v in PROVINCE_CODES.items()}
        
        # Age group mapping
        age_map = {
            '15 to 24 years': [1, 2],
            '25 to 54 years': [2, 3, 4],
            '55 years and over': [5, 6],
            '15 years and over': [1, 2, 3, 4, 5]
        }
        
        # Build wage distribution preserving time
        wage_data = []
        for _, row in df.iterrows():
            wage = float(row['VALUE'])
            gender = 1 if 'Men' in row['Gender'] else 2
            prov = prov_map.get(row['GEO'], 35)
            age_group = row.get('Age group', '25 to 54 years')
            ages = age_map.get(age_group, [3])
            
            # Parse REF_DATE (format: YYYY-MM)
            ref_date = str(row['REF_DATE'])
            try:
                year = int(ref_date.split('-')[0])
                month = int(ref_date.split('-')[1]) if '-' in ref_date else 6
            except (ValueError, IndexError):
                year = 2020
                month = 6
            
            for age in ages:
                wage_data.append({
                    'wage': wage,
                    'gender': gender,
                    'prov': prov,
                    'age_6': age,
                    'year': year,
                    'month': month,
                    'ref_date': ref_date
                })
        
        wage_df = pd.DataFrame(wage_data)
        logger.info(f"Created wage distribution with {len(wage_df):,} reference points")
        
        # Load occupation/industry wages if available
        occ_wages = self._load_occupation_wages(occupation_file)
        
        # Generate individual records
        np.random.seed(42)
        records = []
        
        for i in range(n_samples):
            # Sample from real wage distribution
            ref = wage_df.sample(1).iloc[0]
            
            # Add realistic variation (15%)
            base_wage = ref['wage']
            noise = np.random.normal(0, base_wage * 0.15)
            wage = max(MIN_HOURLY_WAGE, base_wage + noise)
            
            # Demographics from sampled reference
            gender = ref['gender']
            prov = ref['prov']
            age_6 = ref['age_6']
            year = ref['year']
            month = ref['month']
            
            # Generate other attributes
            educ = np.random.choice([0, 1, 2, 3, 4, 5],
                                     p=[0.08, 0.20, 0.15, 0.25, 0.20, 0.12])
            noc_10 = np.random.choice(range(10),
                                       p=[0.08, 0.12, 0.10, 0.08, 0.12, 0.05, 0.15, 0.12, 0.08, 0.10])
            naics = np.random.choice(range(11, 92),
                                      p=self._get_naics_probs())
            ftpt = np.random.choice([1, 2], p=[0.80, 0.20])
            union = np.random.choice([1, 2, 3], p=[0.25, 0.05, 0.70])
            permtemp = np.random.choice([1, 2, 3, 4], p=[0.85, 0.05, 0.05, 0.05])
            estsize = np.random.choice(range(1, 7), p=[0.20, 0.18, 0.18, 0.15, 0.15, 0.14])
            
            # Apply occupation premium
            if noc_10 in occ_wages:
                occ_adjustment = occ_wages[noc_10] / 30.0
                wage = wage * occ_adjustment
            
            # Apply education premium
            educ_premium = {0: 0.75, 1: 0.85, 2: 0.95, 3: 1.0, 4: 1.20, 5: 1.40}
            wage = wage * educ_premium.get(educ, 1.0)
            
            # Full-time premium
            if ftpt == 1:
                wage *= 1.05
            
            # Union premium
            if union == 1:
                wage *= 1.10
            
            # Cap wage to realistic range
            wage = max(MIN_HOURLY_WAGE, min(MAX_HOURLY_WAGE, wage))
            
            # Generate survey weight (approximate)
            finalwt = np.random.uniform(800, 1500)
            
            records.append({
                COLS.GENDER: gender,
                COLS.HOURLY_EARNINGS: round(wage, 2),
                COLS.AGE_6: age_6,
                'AGE_12': self._map_age6_to_age12(age_6),
                COLS.EDUCATION: educ,
                COLS.OCCUPATION_10: noc_10,
                COLS.INDUSTRY: naics,
                COLS.PROVINCE: prov,
                COLS.FULLTIME_PARTTIME: ftpt,
                COLS.PERMANENT_TEMP: permtemp,
                COLS.UNION: union,
                COLS.ESTABLISHMENT_SIZE: estsize,
                'MARSTAT': np.random.choice(range(1, 7), p=[0.35, 0.25, 0.05, 0.05, 0.10, 0.20]),
                COLS.SURVEY_YEAR: year,
                COLS.SURVEY_MONTH: month,
                COLS.YEAR: year,
                COLS.MONTH: month,
                COLS.FINAL_WEIGHT: round(finalwt, 2),
                COLS.SOURCE: 'StatCan_API_Synthesized'
            })
        
        result = pd.DataFrame(records)
        logger.info(f"Generated {len(result):,} records from aggregate data")
        
        # Report statistics
        self._report_data_summary(result)
        
        return result
    
    def _load_occupation_wages(self, occupation_file: Path) -> Dict[int, float]:
        """Load occupation wage data."""
        occ_wages = {}
        
        if not occupation_file.exists():
            return occ_wages
        
        try:
            occ_df = pd.read_csv(occupation_file, low_memory=False, nrows=100000)
            occ_df = occ_df[occ_df['Wages'].str.contains('Average hourly', na=False)]
            occ_df = occ_df[pd.notna(occ_df['VALUE']) & (occ_df['VALUE'] > 0)]
            
            # Map industries to occupation proxies
            naics_col = 'North American Industry Classification System (NAICS)'
            if naics_col in occ_df.columns:
                industries = occ_df[naics_col].dropna().unique()[:10]
                for i, ind in enumerate(industries):
                    ind_data = occ_df[occ_df[naics_col] == ind]
                    if len(ind_data) > 0:
                        occ_wages[i] = float(ind_data['VALUE'].mean())
            
            logger.info(f"Loaded occupation wage data for {len(occ_wages)} groups")
        except Exception as e:
            logger.warning(f"Could not process occupation data: {e}")
        
        return occ_wages
    
    def _get_naics_probs(self) -> List[float]:
        """Get probability distribution for NAICS codes."""
        # Simplified - maps to key NAICS codes
        codes = [11, 21, 22, 23, 31, 41, 44, 48, 51, 52, 53, 54, 56, 61, 62, 71, 72, 81, 91]
        probs = [0.02, 0.02, 0.01, 0.07, 0.10, 0.03, 0.11, 0.05, 0.02, 0.06, 0.02, 0.08, 0.04, 0.07, 0.13, 0.02, 0.06, 0.04, 0.05]
        # Normalize
        total = sum(probs)
        return [p/total for p in probs]
    
    def _map_age6_to_age12(self, age_6: int) -> int:
        """Map AGE_6 category to AGE_12."""
        mapping = {1: 2, 2: 4, 3: 6, 4: 8, 5: 10, 6: 12}
        return mapping.get(age_6, 5)
    
    def generate_synthetic_data(self, n_samples: int = 50000) -> pd.DataFrame:
        """
        Generate synthetic LFS-like data for testing and development.
        
        Creates realistic wage distributions with known gender gap for validation.
        
        Args:
            n_samples: Number of records to generate
            
        Returns:
            DataFrame with synthetic data
        """
        logger.info(f"Generating {n_samples:,} synthetic records...")
        
        np.random.seed(42)
        
        # Gender (approximately 50/50 in labor force)
        gender = np.random.choice([1, 2], size=n_samples, p=[0.52, 0.48])
        
        # Year distribution across scope
        years = np.random.choice(
            range(self.start_year, self.end_year + 1), 
            size=n_samples
        )
        months = np.random.choice(range(1, 13), size=n_samples)
        
        # Age groups (AGE_6)
        age_probs = [0.12, 0.22, 0.24, 0.22, 0.15, 0.05]
        age_6 = np.random.choice(range(1, 7), size=n_samples, p=age_probs)
        
        # Education
        educ_probs = [0.08, 0.18, 0.14, 0.28, 0.20, 0.12]
        educ = np.random.choice(range(0, 6), size=n_samples, p=educ_probs)
        
        # Occupation (NOC_10)
        noc_probs = [0.08, 0.12, 0.10, 0.08, 0.12, 0.05, 0.15, 0.12, 0.08, 0.10]
        noc_10 = np.random.choice(range(0, 10), size=n_samples, p=noc_probs)
        
        # Province
        prov_probs = [0.02, 0.01, 0.03, 0.02, 0.23, 0.39, 0.04, 0.03, 0.11, 0.12]
        prov = np.random.choice(
            [10, 11, 12, 13, 24, 35, 46, 47, 48, 59],
            size=n_samples, p=prov_probs
        )
        
        # Employment characteristics
        ftpt = np.random.choice([1, 2], size=n_samples, p=[0.80, 0.20])
        union = np.random.choice([1, 2, 3], size=n_samples, p=[0.25, 0.05, 0.70])
        permtemp = np.random.choice([1, 2, 3, 4], size=n_samples, p=[0.85, 0.05, 0.05, 0.05])
        estsize = np.random.choice(range(1, 7), size=n_samples, p=[0.20, 0.18, 0.18, 0.15, 0.15, 0.14])
        
        # Generate wages based on factors
        base_wage = 25.0
        
        # Factor premiums
        educ_premium = {0: 0.75, 1: 0.85, 2: 0.95, 3: 1.0, 4: 1.20, 5: 1.40}
        noc_premium = {0: 1.5, 1: 1.2, 2: 1.35, 3: 1.25, 4: 1.1, 5: 0.95, 6: 0.85, 7: 1.05, 8: 0.90, 9: 0.88}
        age_premium = {1: 0.70, 2: 0.95, 3: 1.10, 4: 1.15, 5: 1.05, 6: 0.95}
        prov_premium = {10: 0.88, 11: 0.82, 12: 0.86, 13: 0.85, 24: 0.95, 35: 1.05, 46: 0.92, 47: 0.94, 48: 1.12, 59: 1.06}
        
        # Gender gap (raw ~12%)
        gender_factor = np.where(gender == 2, 0.88, 1.0)
        
        # Full-time premium
        ft_factor = np.where(ftpt == 1, 1.05, 0.92)
        
        # Union premium
        union_factor = np.where(union == 1, 1.12, 1.0)
        
        # Calculate wages
        hourly_earn = (
            base_wage
            * np.array([educ_premium.get(e, 1.0) for e in educ])
            * np.array([noc_premium.get(n, 1.0) for n in noc_10])
            * np.array([age_premium.get(a, 1.0) for a in age_6])
            * np.array([prov_premium.get(p, 1.0) for p in prov])
            * gender_factor
            * ft_factor
            * union_factor
            * np.random.lognormal(0, 0.18, n_samples)
        )
        
        # Clip to realistic range
        hourly_earn = np.clip(hourly_earn, MIN_HOURLY_WAGE, MAX_HOURLY_WAGE)
        
        # Generate survey weights
        finalwt = np.random.uniform(800, 1500, n_samples)
        
        # Create DataFrame with standardized column names
        df = pd.DataFrame({
            COLS.GENDER: gender,
            COLS.AGE_6: age_6,
            'AGE_12': np.array([self._map_age6_to_age12(a) for a in age_6]),
            COLS.EDUCATION: educ,
            COLS.OCCUPATION_10: noc_10,
            COLS.INDUSTRY: np.random.choice(range(11, 92), size=n_samples),
            COLS.PROVINCE: prov,
            COLS.FULLTIME_PARTTIME: ftpt,
            COLS.PERMANENT_TEMP: permtemp,
            COLS.UNION: union,
            COLS.ESTABLISHMENT_SIZE: estsize,
            'MARSTAT': np.random.choice(range(1, 7), size=n_samples, p=[0.35, 0.25, 0.05, 0.05, 0.10, 0.20]),
            COLS.HOURLY_EARNINGS: np.round(hourly_earn, 2),
            COLS.SURVEY_YEAR: years,
            COLS.SURVEY_MONTH: months,
            COLS.YEAR: years,
            COLS.MONTH: months,
            COLS.FINAL_WEIGHT: np.round(finalwt, 2),
            COLS.SOURCE: 'Synthetic'
        })
        
        logger.info(f"Generated {len(df):,} synthetic records")
        self._report_data_summary(df)
        
        return df
    
    def _report_data_summary(self, df: pd.DataFrame) -> None:
        """Report summary statistics of the dataset."""
        gender_col = COLS.GENDER
        wage_col = COLS.HOURLY_EARNINGS
        
        # Use only records with valid wages for wage statistics
        if 'HAS_VALID_WAGE' in df.columns:
            wage_df = df[df['HAS_VALID_WAGE']]
        else:
            wage_df = df[df[wage_col] > 0] if wage_col in df.columns else df
        
        if gender_col in wage_df.columns and wage_col in wage_df.columns:
            male_avg = wage_df[wage_df[gender_col] == 1][wage_col].mean()
            female_avg = wage_df[wage_df[gender_col] == 2][wage_col].mean()
            if pd.notna(male_avg) and pd.notna(female_avg) and male_avg > 0:
                gap = (male_avg - female_avg) / male_avg * 100
                logger.info(f"  Male avg: ${male_avg:.2f}/hr, Female avg: ${female_avg:.2f}/hr")
                logger.info(f"  Raw gender wage gap: {gap:.1f}%")
        
        if COLS.YEAR in df.columns:
            year_range = f"{df[COLS.YEAR].min()}-{df[COLS.YEAR].max()}"
            logger.info(f"  Year range: {year_range}")
    
    # =========================================================================
    # DATA CLEANING AND FEATURE CREATION
    # =========================================================================
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate the data while PRESERVING ALL RECORDS.
        
        - Does NOT filter out records with missing wages
        - Normalizes column names
        - Flags records with valid wages via HAS_VALID_WAGE column
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame with all original records
        """
        logger.info("Cleaning data...")
        df = df.copy()
        initial_count = len(df)
        
        # Normalize legacy column names (SEX -> GENDER, etc.)
        df = normalize_column_names(df)
        
        # Convert wage column to numeric (coerce errors to NaN)
        wage_col = COLS.HOURLY_EARNINGS
        if wage_col in df.columns:
            df[wage_col] = pd.to_numeric(df[wage_col], errors='coerce')
            
            # Flag valid wages instead of filtering them out
            # This preserves ALL records for demographic analysis
            df['HAS_VALID_WAGE'] = (
                df[wage_col].notna() & 
                (df[wage_col] > 0) & 
                (df[wage_col] >= MIN_HOURLY_WAGE) & 
                (df[wage_col] <= MAX_HOURLY_WAGE)
            )
            
            valid_count = df['HAS_VALID_WAGE'].sum()
            logger.info(f"  Valid wages: {valid_count:,} / {initial_count:,} records "
                       f"({100*valid_count/initial_count:.1f}%)")
        
        logger.info(f"Cleaned data: {len(df):,} records (all preserved)")
        return df
    
    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features for analysis.
        
        IMPORTANT: This method applies INFLATION ADJUSTMENT EARLY in the pipeline.
        For cross-year analysis (2010-2025), always use REAL_HRLYEARN (not HRLYEARN).
        
        Wage columns created:
        - HRLYEARN: Nominal hourly earnings (as reported in survey)
        - REAL_HRLYEARN: Real wages in 2010 constant dollars (CPI-deflated)
        - LOG_HRLYEARN: Log of nominal wages
        - LOG_REAL_HRLYEARN: Log of real wages (use for regressions)
        
        Other features:
        - Binary indicators (IS_FEMALE, IS_FULLTIME, etc.)
        - Age/experience approximations
        - Human-readable labels
        - Macroeconomic context (CPI, unemployment, etc.)
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            DataFrame with derived features including inflation-adjusted wages
        """
        logger.info("Creating derived features...")
        df = df.copy()
        
        # Normalize column names first
        df = normalize_column_names(df)
        
        # Binary indicators
        gender_col = COLS.GENDER
        if gender_col in df.columns:
            df[COLS.IS_FEMALE] = (df[gender_col] == 2).astype(int)
        
        if COLS.FULLTIME_PARTTIME in df.columns:
            df[COLS.IS_FULLTIME] = (df[COLS.FULLTIME_PARTTIME] == 1).astype(int)
        
        if COLS.PERMANENT_TEMP in df.columns:
            df['IS_PERMANENT'] = (df[COLS.PERMANENT_TEMP] == 1).astype(int)
        
        if COLS.UNION in df.columns:
            df['IS_UNION'] = (df[COLS.UNION] == 1).astype(int)
        
        if COLS.EDUCATION in df.columns:
            df['HAS_DEGREE'] = (df[COLS.EDUCATION] >= 4).astype(int)
        
        # Age approximation
        if 'AGE_6' in df.columns:
            df['AGE_APPROX'] = df['AGE_6'].map(AGE_6_MIDPOINTS)
        elif 'AGE_12' in df.columns:
            df['AGE_APPROX'] = df['AGE_12'].map(AGE_12_MIDPOINTS)
        
        # Experience proxy (Mincer approach)
        if 'AGE_APPROX' in df.columns and COLS.EDUCATION in df.columns:
            educ_complete = {0: 16, 1: 18, 2: 19, 3: 20, 4: 22, 5: 25}
            df['EDUC_COMPLETE_AGE'] = df[COLS.EDUCATION].map(educ_complete).fillna(18)
            df['EXPERIENCE_PROXY'] = (df['AGE_APPROX'] - df['EDUC_COMPLETE_AGE']).clip(lower=0)
            df['EXPERIENCE_SQ'] = df['EXPERIENCE_PROXY'] ** 2
        
        # Log wages (nominal) - only for valid wages
        wage_col = COLS.HOURLY_EARNINGS
        if wage_col in df.columns:
            # Initialize with NaN
            df[COLS.LOG_HOURLY_EARNINGS] = np.nan
            # Only calculate log for valid positive wages
            valid_mask = df.get('HAS_VALID_WAGE', df[wage_col] > 0)
            if valid_mask.any():
                df.loc[valid_mask, COLS.LOG_HOURLY_EARNINGS] = np.log(df.loc[valid_mask, wage_col])
        
        # Ensure YEAR column exists
        if COLS.YEAR not in df.columns:
            if COLS.SURVEY_YEAR in df.columns:
                df[COLS.YEAR] = df[COLS.SURVEY_YEAR]
            else:
                df[COLS.YEAR] = 2020  # Default
        
        # =====================================================================
        # INFLATION ADJUSTMENT - Critical for cross-year analysis
        # =====================================================================
        # Add macroeconomic context (CPI, unemployment, GDP, etc.)
        df = add_macro_to_dataframe(df, year_col=COLS.YEAR)
        
        # Calculate REAL WAGES (inflation-adjusted to 2010 constant dollars)
        # For any analysis spanning multiple years, use REAL_HRLYEARN not HRLYEARN
        # Only calculate for records with valid wages
        if wage_col in df.columns and 'deflator' in df.columns:
            # Initialize with NaN
            df[COLS.REAL_HOURLY_EARNINGS] = np.nan
            df[COLS.LOG_REAL_HOURLY_EARNINGS] = np.nan
            
            # Only calculate for valid wages
            valid_mask = df.get('HAS_VALID_WAGE', df[wage_col] > 0)
            if valid_mask.any():
                df.loc[valid_mask, COLS.REAL_HOURLY_EARNINGS] = (
                    df.loc[valid_mask, wage_col] * df.loc[valid_mask, 'deflator']
                )
                df.loc[valid_mask, COLS.LOG_REAL_HOURLY_EARNINGS] = np.log(
                    df.loc[valid_mask, COLS.REAL_HOURLY_EARNINGS]
                )
            
            # Log summary of deflation
            base_year_cpi = df[df[COLS.YEAR] == BASE_YEAR]['cpi'].iloc[0] if len(df[df[COLS.YEAR] == BASE_YEAR]) > 0 else None
            max_year = df[COLS.YEAR].max()
            max_year_cpi = df[df[COLS.YEAR] == max_year]['cpi'].iloc[0] if len(df[df[COLS.YEAR] == max_year]) > 0 else None
            if base_year_cpi and max_year_cpi:
                inflation_total = (max_year_cpi / base_year_cpi - 1) * 100
                logger.info(f"  Inflation adjustment: {inflation_total:.1f}% cumulative ({BASE_YEAR}-{max_year})")
                logger.info(f"  Real wages expressed in {BASE_YEAR} constant dollars")
        
        # Economic period classification
        if COLS.YEAR in df.columns:
            df[COLS.ECONOMIC_PERIOD] = df[COLS.YEAR].apply(get_economic_period)
        
        # Apply human-readable labels
        df = apply_labels(df)
        
        # Province abbreviations
        if COLS.PROVINCE in df.columns:
            from .constants import PROVINCE_ABBREV
            df['PROV_ABBREV'] = df[COLS.PROVINCE].map(PROVINCE_ABBREV)
        
        logger.info(f"Created {len(df.columns)} features")
        return df
    
    # =========================================================================
    # SAVE/LOAD METHODS
    # =========================================================================
    
    def save_processed_data(self, df: pd.DataFrame, 
                            filename: str = "lfs_processed.csv") -> Path:
        """Save processed data to CSV."""
        output_path = self.processed_path / filename
        df.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")
        return output_path
    
    def load_processed_data(self, filename: str = "lfs_processed.csv") -> pd.DataFrame:
        """Load processed data from CSV."""
        input_path = self.processed_path / filename
        if not input_path.exists():
            raise FileNotFoundError(f"Processed data not found at {input_path}")
        return pd.read_csv(input_path)
    
    # =========================================================================
    # MAIN PIPELINE
    # =========================================================================
    
    def run_pipeline(self, data_source: str = 'auto', 
                     n_samples: int = 50000,
                     save: bool = True) -> pd.DataFrame:
        """
        Run the complete data pipeline.
        
        Args:
            data_source: 'pumf' (real LFS microdata), 
                        'synthetic' (generated for testing), 
                        or 'auto' (tries pumf first, then synthetic)
            n_samples: Number of samples for synthetic data
            save: Whether to save processed data
            
        Returns:
            Processed DataFrame ready for analysis
        """
        logger.info("=" * 60)
        logger.info("EQUIPAY CANADA DATA PIPELINE")
        logger.info("=" * 60)
        
        df = None
        
        if data_source == 'auto':
            # Try real PUMF first, fall back to synthetic
            try:
                df = self.load_lfs_pumf()
                logger.info("Using real LFS PUMF microdata")
            except FileNotFoundError:
                logger.info("PUMF not available, generating synthetic data...")
                df = self.generate_synthetic_data(n_samples)
                logger.info("Using synthetic data")
        
        elif data_source == 'pumf':
            df = self.load_lfs_pumf()
        
        elif data_source == 'synthetic':
            df = self.generate_synthetic_data(n_samples)
        
        else:
            raise ValueError(f"Unknown data_source: {data_source}. Use 'pumf', 'synthetic', or 'auto'.")
        
        # Clean and create features
        df = self.clean_data(df)
        df = self.create_derived_features(df)
        
        if save:
            self.save_processed_data(df)
        
        logger.info("=" * 60)
        logger.info(f"Pipeline complete: {len(df):,} records, {len(df.columns)} features")
        logger.info("=" * 60)
        
        return df


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='EquiPay Canada Data Pipeline')
    parser.add_argument('--source', choices=['auto', 'pumf', 'synthetic'],
                        default='auto', help='Data source (pumf or synthetic)')
    parser.add_argument('--samples', type=int, default=50000,
                        help='Number of samples for synthetic data')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save processed data')
    
    args = parser.parse_args()
    
    pipeline = LFSDataPipeline()
    df = pipeline.run_pipeline(
        data_source=args.source,
        n_samples=args.samples,
        save=not args.no_save
    )
    
    print(f"\nDataset ready: {len(df):,} records")
    print(df.head())


if __name__ == "__main__":
    main()
