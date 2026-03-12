"""
Extract Yearly Wage Time Series from LFS PUMF Data
===================================================

Creates a clean yearly time series for 2010-2025 from processed LFS data.

Data Source: LFS PUMF only (no external CSV files)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.constants import COLS, GENDER_CODES_REVERSE, normalize_column_names, DATA_SCOPE_START, DATA_SCOPE_END
from src.macro_data import MACRO_DATA, get_macro_dataframe, BASE_YEAR


def extract_yearly_wage_timeseries(parquet_path: str = None):
    """
    Extract yearly wage data by gender from processed LFS PUMF data using SQL (DuckDB).

    Falls back to CSV-based processing if DuckDB/parquet is not available.
    """
    processed_file = Path(__file__).parent.parent / 'data' / 'processed' / 'lfs_processed.csv'
    output_file = Path(__file__).parent.parent / 'data' / 'processed' / 'yearly_wages_by_gender.csv'

    # Try SQL-first path using EquiPayDataStore (DuckDB over Parquet)
    try:
        from src.data_store import EquiPayDataStore
        store = EquiPayDataStore(parquet_path=parquet_path or 'data/parquet')

        sql = f"""
            SELECT s.SURVYEAR AS year,
                   SUM(CASE WHEN s.GENDER = 2 THEN s.HRLYEARN * s.FINALWT END)/NULLIF(SUM(CASE WHEN s.GENDER = 2 THEN s.FINALWT END), 0) AS female_wage,
                   SUM(CASE WHEN s.GENDER = 1 THEN s.HRLYEARN * s.FINALWT END)/NULLIF(SUM(CASE WHEN s.GENDER = 1 THEN s.FINALWT END), 0) AS male_wage,
                   SUM(CASE WHEN s.GENDER = 2 THEN s.REAL_HRLYEARN * s.FINALWT END)/NULLIF(SUM(CASE WHEN s.GENDER = 2 THEN s.FINALWT END), 0) AS real_female_wage,
                   SUM(CASE WHEN s.GENDER = 1 THEN s.REAL_HRLYEARN * s.FINALWT END)/NULLIF(SUM(CASE WHEN s.GENDER = 1 THEN s.FINALWT END), 0) AS real_male_wage,
                   m.cpi, m.gdp_growth, m.unemployment
            FROM {store.TABLE_NAME} s
            LEFT JOIN macro m ON s.SURVYEAR = m.year
            WHERE s.SURVYEAR BETWEEN {DATA_SCOPE_START} AND {DATA_SCOPE_END}
            GROUP BY s.SURVYEAR, m.cpi, m.gdp_growth, m.unemployment
            ORDER BY s.SURVYEAR
        """

        df = store.sql(sql)

        if df is None or df.empty:
            raise RuntimeError("SQL query returned no results; falling back to CSV path")

        # Calculate gap metrics
        df['wage_gap_pct'] = (df['male_wage'] - df['female_wage']) / df['male_wage'] * 100
        df['wage_ratio'] = df['female_wage'] / df['male_wage'] * 100

        # Save
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"\n✓ Saved yearly wage time series to: {output_file}")

        # Display summary
        print("\nYearly Wage Gap Summary:")
        print("-" * 60)
        print(df[['year', 'male_wage', 'female_wage', 'wage_gap_pct']].round(2).to_string(index=False))

        return df

    except Exception as e:
        print(f"SQL path failed, falling back to CSV implementation: {e}")

    # Fallback CSV-based path (previous implementation) ------------------------------------------------
    if not processed_file.exists():
        print(f"ERROR: {processed_file} not found")
        print("Please run the data pipeline first to generate lfs_processed.csv")
        return None

    print(f"Reading from: {processed_file}")
    print(f"Extracting wage time series for: {DATA_SCOPE_START}-{DATA_SCOPE_END}")

    # Load data
    df = pd.read_csv(processed_file)
    df = normalize_column_names(df)

    # Identify columns
    gender_col = COLS.GENDER if COLS.GENDER in df.columns else 'SEX'
    wage_col = COLS.HRLYEARN
    year_col = 'YEAR' if 'YEAR' in df.columns else 'SURVYEAR' if 'SURVYEAR' in df.columns else None

    if year_col is None:
        print("ERROR: No year column found in data")
        return None

    print(f"  Total records: {len(df):,}")
    print(f"  Year range: {df[year_col].min()}-{df[year_col].max()}")

    # Filter to data scope
    df = df[(df[year_col] >= DATA_SCOPE_START) & (df[year_col] <= DATA_SCOPE_END)]
    print(f"  Records in scope: {len(df):,}")

    # Create gender labels
    if df[gender_col].dtype in ['int64', 'float64']:
        df['gender_label'] = df[gender_col].map({
            GENDER_CODES_REVERSE['Male']: 'Male',
            GENDER_CODES_REVERSE['Female']: 'Female'
        })
    else:
        df['gender_label'] = df[gender_col]

    # Calculate yearly statistics by gender
    yearly_stats = []

    for year in sorted(df[year_col].unique()):
        year_data = df[df[year_col] == year]

        for gender in ['Male', 'Female']:
            gender_data = year_data[year_data['gender_label'] == gender]
            if len(gender_data) > 0:
                yearly_stats.append({
                    'year': int(year),
                    'gender': gender,
                    'mean_wage': gender_data[wage_col].mean(),
                    'median_wage': gender_data[wage_col].median(),
                    'n': len(gender_data)
                })

    result = pd.DataFrame(yearly_stats)

    # Pivot to wide format
    pivot = result.pivot(index='year', columns='gender', values='mean_wage')
    pivot.columns = ['female_wage', 'male_wage']
    pivot = pivot.reset_index()

    # Calculate wage gap
    pivot['wage_gap_pct'] = (pivot['male_wage'] - pivot['female_wage']) / pivot['male_wage'] * 100
    pivot['wage_ratio'] = pivot['female_wage'] / pivot['male_wage'] * 100

    # Add macro data
    macro_df = get_macro_dataframe()
    pivot = pivot.merge(macro_df[['year', 'cpi', 'gdp_growth', 'unemployment']], on='year', how='left')

    # Calculate real wages (CPI-adjusted)
    base_cpi = MACRO_DATA[BASE_YEAR]['cpi']
    pivot['real_male_wage'] = pivot['male_wage'] * (base_cpi / pivot['cpi'])
    pivot['real_female_wage'] = pivot['female_wage'] * (base_cpi / pivot['cpi'])

    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    pivot.to_csv(output_file, index=False)
    print(f"\n✓ Saved yearly wage time series to: {output_file}")

    # Display summary
    print("\nYearly Wage Gap Summary:")
    print("-" * 60)
    print(pivot[['year', 'male_wage', 'female_wage', 'wage_gap_pct']].round(2).to_string(index=False))

    return pivot


if __name__ == '__main__':
    extract_yearly_wage_timeseries()
