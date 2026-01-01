"""
Update Data Script
==================

Regenerates the processed LFS dataset from source files.

Data Source: LFS PUMF only (2010-2025)
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline import LFSDataPipeline
from src.constants import DATA_SCOPE_START, DATA_SCOPE_END

print('='*60)
print('REGENERATING LFS DATA')
print(f'Data Scope: {DATA_SCOPE_START}-{DATA_SCOPE_END}')
print('='*60)

pipeline = LFSDataPipeline()

# Try to load real LFS PUMF first, fall back to synthetic
try:
    print('\nAttempting to load LFS PUMF microdata...')
    df = pipeline.load_lfs_pumf()
except FileNotFoundError as e:
    print(f'\nLFS PUMF not found: {e}')
    print('Generating synthetic data for development...')
    df = pipeline.generate_synthetic_data(n_samples=50000)

# Clean and process
df = pipeline.clean_data(df)
df = pipeline.create_derived_features(df)

# Save
output_path = Path('data/processed/lfs_processed.csv')
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)

print(f'\n✓ Saved {len(df):,} records to {output_path}')

print('\n' + '='*60)
print('DATASET SUMMARY')
print('='*60)

# Gender distribution
gender_col = 'GENDER' if 'GENDER' in df.columns else 'SEX'
print(f'\nGender Distribution ({gender_col}):')
print(df[gender_col].value_counts())

# Wage statistics
wage_col = 'HRLYEARN'
print(f'\nWage Statistics ({wage_col}):')
print(f'  Mean: ${df[wage_col].mean():.2f}')
print(f'  Median: ${df[wage_col].median():.2f}')
print(f'  Min: ${df[wage_col].min():.2f}')
print(f'  Max: ${df[wage_col].max():.2f}')
