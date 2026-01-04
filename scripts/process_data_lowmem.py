#!/usr/bin/env python3
"""
Memory-efficient LFS data processing script.
Processes data year-by-year to stay within 8GB RAM limit.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import gc
import sys

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.macro_data import get_deflator

# Configuration
RAW_DIR = Path(__file__).parent.parent / 'data' / 'raw' / 'lfs'
OUT_FILE = Path(__file__).parent.parent / 'data' / 'processed' / 'lfs_processed.csv'
YEARS = range(2010, 2026)  # 2010-2025

# Columns to keep (reduce memory footprint) - use actual LFS PUMF column names
KEEP_COLS = [
    'SURVYEAR', 'SURVMNTH', 'LFSSTAT', 'GENDER', 'AGE_12', 'AGE_6',
    'PROV', 'CMA', 'EDUC', 'TENURE', 'COWMAIN', 
    'NOC_10', 'NOC_43', 'NAICS_21', 'FTPTMAIN',
    'HRLYEARN', 'UHRSMAIN', 'AHRSMAIN', 'UTOTHRS', 'ATOTHRS',
    'UNION', 'PERMTEMP', 'WHYPT', 'DURJLESS', 'IMMIG', 'YRIMM',
    'FINALWT', 'MARSTAT'
]

def get_files_for_year(year):
    """Get all monthly files for a given year."""
    pattern = f"lfs_{year}_*.csv"
    files = sorted(RAW_DIR.glob(pattern))
    return files

def process_chunk(df):
    """Apply transformations to a chunk of data."""
    # Standardize column names to uppercase
    df.columns = df.columns.str.upper()
    
    # Handle SEX -> GENDER mapping for older files
    if 'SEX' in df.columns and 'GENDER' not in df.columns:
        df['GENDER'] = df['SEX']
    
    # Filter to employed with valid wages
    if 'HRLYEARN' not in df.columns:
        return None
    
    # Filter: employed (LFSSTAT in [1,2,3]) with valid wages
    if 'LFSSTAT' in df.columns:
        df = df[df['LFSSTAT'].isin([1, 2, 3])]
    
    # HRLYEARN is in cents (e.g., 2500 = $25.00/hour)
    # Valid wages: $1-$500/hour = 100-50000 cents
    df = df[(df['HRLYEARN'] >= 100) & (df['HRLYEARN'] <= 50000)]
    
    # Convert cents to dollars
    df['HRLYEARN'] = df['HRLYEARN'] / 100.0
    
    if len(df) == 0:
        return None
    
    # Create derived features
    if 'GENDER' in df.columns:
        df['IS_FEMALE'] = (df['GENDER'] == 2).astype(np.int8)
    
    # Full-time indicator
    if 'FTPTMAIN' in df.columns:
        df['IS_FULLTIME'] = (df['FTPTMAIN'] == 1).astype(np.int8)
    
    # Union indicator
    if 'UNION' in df.columns:
        df['IS_UNION'] = df['UNION'].isin([1, 2]).astype(np.int8)
    
    # Experience proxy (age - education - 6)
    if 'AGE_12' in df.columns and 'EDUC' in df.columns:
        # Map age groups to midpoints
        age_midpoints = {1: 17, 2: 22, 3: 27, 4: 32, 5: 37, 6: 42, 7: 47, 8: 52, 9: 57, 10: 62, 11: 67, 12: 72}
        df['AGE_MIDPOINT'] = df['AGE_12'].map(age_midpoints).fillna(40)
        
        # Estimate years of education
        educ_years = {0: 8, 1: 10, 2: 12, 3: 13, 4: 14, 5: 16, 6: 18, 7: 20}
        df['EDUC_YEARS'] = df['EDUC'].map(educ_years).fillna(12)
        
        df['EXPERIENCE'] = np.maximum(0, df['AGE_MIDPOINT'] - df['EDUC_YEARS'] - 6)
        df['EXPERIENCE_SQ'] = df['EXPERIENCE'] ** 2
    
    # Log wage
    df['LOG_HRLYEARN'] = np.log(df['HRLYEARN'].clip(lower=1))
    
    # Real wage (inflation-adjusted to 2010)
    if 'SURVYEAR' in df.columns:
        years = df['SURVYEAR'].values
        deflators = np.array([get_deflator(y) for y in years])
        df['REAL_HRLYEARN'] = df['HRLYEARN'] / deflators
        df['LOG_REAL_HRLYEARN'] = np.log(df['REAL_HRLYEARN'].clip(lower=1))
    
    return df

def process_year(year, first_year=False):
    """Process all files for a single year."""
    files = get_files_for_year(year)
    if not files:
        print(f"  No files found for {year}")
        return None
    
    year_dfs = []
    for f in files:
        try:
            # Read full file (column selection later)
            df = pd.read_csv(f, low_memory=True)
            df.columns = df.columns.str.upper()
            
            # Process chunk
            processed = process_chunk(df)
            if processed is not None and len(processed) > 0:
                year_dfs.append(processed)
            
            del df
            gc.collect()
        except Exception as e:
            print(f"    Error processing {f.name}: {e}")
            continue
    
    if not year_dfs:
        return None
    
    # Combine year data
    year_df = pd.concat(year_dfs, ignore_index=True)
    del year_dfs
    gc.collect()
    
    return year_df

def main():
    print("=" * 60)
    print("MEMORY-EFFICIENT LFS DATA PROCESSING")
    print("=" * 60)
    print(f"Processing years: {min(YEARS)}-{max(YEARS)}")
    print(f"Output: {OUT_FILE}")
    print()
    
    # Process year by year and append to file
    total_rows = 0
    first_year = True
    
    for year in YEARS:
        print(f"Processing {year}...", end=" ", flush=True)
        
        year_df = process_year(year, first_year)
        
        if year_df is not None and len(year_df) > 0:
            # Write to CSV (append mode after first year)
            mode = 'w' if first_year else 'a'
            header = first_year
            year_df.to_csv(OUT_FILE, mode=mode, header=header, index=False)
            
            rows = len(year_df)
            total_rows += rows
            print(f"✓ {rows:,} rows (total: {total_rows:,})")
            
            first_year = False
            del year_df
        else:
            print("⚠ No valid data")
        
        # Force garbage collection
        gc.collect()
    
    print()
    print("=" * 60)
    print(f"PROCESSING COMPLETE")
    print(f"Total rows: {total_rows:,}")
    print(f"Output: {OUT_FILE}")
    print("=" * 60)

if __name__ == "__main__":
    main()
