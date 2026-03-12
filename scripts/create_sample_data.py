#!/usr/bin/env python3
"""
Create sample data for Railway deployment (no large data files)
This generates a minimal dataset that demonstrates the application functionality.
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

def create_sample_data():
    """Create sample LFS data for Railway deployment."""
    
    # Ensure we're in the right directory
    if 'scripts' in str(Path.cwd()):
        os.chdir('..')
    
    # Create directories
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    Path('data/raw').mkdir(parents=True, exist_ok=True)
    
    print("Creating sample dataset for Railway deployment...")
    
    # Generate sample data matching LFS structure
    np.random.seed(42)
    n_samples = 10000  # Much smaller than real dataset
    
    # Sample provinces and demographics
    provinces = ['ON', 'QC', 'BC', 'AB', 'MB', 'SK', 'NS', 'NB', 'NL', 'PE']
    education_levels = [1, 2, 3, 4, 5, 6, 7, 8]  # LFS education codes
    noc_codes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]  # Sample NOC broad categories
    
    # Generate synthetic data
    sample_data = {
        'SURVYEAR': np.random.choice(range(2015, 2024), n_samples),
        'PROV': np.random.choice(provinces, n_samples),
        'SEX': np.random.choice([1, 2], n_samples),  # 1=Male, 2=Female
        'AGE_12': np.random.choice(range(15, 65), n_samples),
        'EDUC': np.random.choice(education_levels, n_samples),
        'NOC_10': np.random.choice(noc_codes, n_samples),
        'HRLYEARN': np.random.lognormal(mean=3.0, sigma=0.5, size=n_samples),  # Realistic wage distribution
        'UHRSMAIN': np.random.choice(range(20, 45), n_samples),  # Hours worked
        'IMMIG': np.random.choice([1, 2, 3, 4], n_samples),  # Immigration status
        'FINALWT': np.random.uniform(100, 2000, n_samples),  # Survey weights
    }\n    \n    # Create DataFrame\n    df = pd.DataFrame(sample_data)\n    \n    # Add some realistic wage gaps\n    # Gender gap\n    gender_gap = np.where(df['SEX'] == 2, 0.85, 1.0)  # 15% gap\n    df['HRLYEARN'] *= gender_gap\n    \n    # Education premium\n    edu_premium = 1.0 + (df['EDUC'] - 1) * 0.1\n    df['HRLYEARN'] *= edu_premium\n    \n    # Make sure wages are reasonable (CAD per hour)\n    df['HRLYEARN'] = np.clip(df['HRLYEARN'], 15, 100)\n    \n    # Save sample data\n    output_path = 'data/processed/lfs_processed.csv'\n    df.to_csv(output_path, index=False)\n    print(f"✓ Created sample dataset: {output_path} ({len(df):,} records)")\n    \n    # Create a minimal DuckDB file\n    try:\n        import duckdb\n        conn = duckdb.connect('data/processed/lfs_data.duckdb')\n        conn.execute("CREATE TABLE lfs_data AS SELECT * FROM df")\n        conn.close()\n        print("✓ Created sample DuckDB file")\n    except ImportError:\n        print("! DuckDB not available, skipping database creation")\n    \n    return True\n\nif __name__ == "__main__":\n    create_sample_data()\n    print("Sample data creation complete!")