"""
Memory-efficient data loading utilities for EquiPay Canada.
Handles the 9.8M row dataset within 8GB RAM constraints.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Union
import gc
import warnings

# =============================================================================
# CONSTANTS
# =============================================================================

# Optimized dtypes to reduce memory footprint by ~60%
# Using float32 for columns that may have NaN values
OPTIMIZED_DTYPES = {
    'SURVYEAR': 'int16',
    'SURVMNTH': 'float32',  # May have NaN
    'LFSSTAT': 'float32',
    'PROV': 'float32',
    'CMA': 'float32',
    'AGE_12': 'float32',
    'AGE_6': 'float32',
    'SEX': 'float32',
    'GENDER': 'float32',
    'MARSTAT': 'float32',
    'EDUC': 'float32',
    'COWMAIN': 'float32',
    'IMMIG': 'float32',
    'NAICS_21': 'float32',
    'NOC_10': 'float32',
    'NOC_43': 'float32',
    'FTPTMAIN': 'float32',
    'UNION': 'float32',
    'PERMTEMP': 'float32',
    'TENURE': 'float32',
    'ESTSIZE': 'float32',
    'FIRMSIZE': 'float32',
    'IS_FEMALE': 'float32',
    'IS_FULLTIME': 'float32',
    'IS_UNION': 'float32',
    'HRLYEARN': 'float32',
    'REAL_HRLYEARN': 'float32',
    'LOG_HRLYEARN': 'float32',
    'LOG_REAL_HRLYEARN': 'float32',
    'EXPERIENCE': 'float32',
    'EXPERIENCE_SQ': 'float32',
    'FINALWT': 'float32',
    'UHRSMAIN': 'float32',
    'AHRSMAIN': 'float32',
}

# Default data path
DATA_DIR = Path(__file__).parent.parent / 'data'
PROCESSED_FILE = DATA_DIR / 'processed' / 'lfs_processed.csv'
AGGREGATES_DIR = DATA_DIR / 'processed' / 'aggregates'

# =============================================================================
# MEMORY-EFFICIENT DATA LOADING
# =============================================================================

def load_data(
    sample_size: Optional[int] = None,
    sample_frac: Optional[float] = None,
    columns: Optional[List[str]] = None,
    years: Optional[List[int]] = None,
    stratify_by: Optional[str] = None,
    random_state: int = 42,
    data_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Load LFS data with memory-efficient chunked reading.
    
    Args:
        sample_size: Number of rows to sample (e.g., 500_000)
        sample_frac: Fraction of data to sample (e.g., 0.1 for 10%)
        columns: Specific columns to load (reduces memory)
        years: List of years to filter (e.g., [2020, 2021, 2022])
        stratify_by: Column to stratify sampling (e.g., 'SURVYEAR', 'IS_FEMALE')
        random_state: Random seed for reproducibility
        data_path: Override default data path
        
    Returns:
        DataFrame with optimized dtypes
        
    Example:
        # Load 500k sample with key columns
        df = load_data(sample_size=500_000, columns=['SURVYEAR', 'HRLYEARN', 'IS_FEMALE', 'PROV'])
        
        # Load 10% sample stratified by year
        df = load_data(sample_frac=0.1, stratify_by='SURVYEAR')
    """
    data_path = data_path or PROCESSED_FILE
    
    # Determine which dtypes to use for requested columns
    use_dtypes = {}
    if columns:
        use_dtypes = {k: v for k, v in OPTIMIZED_DTYPES.items() if k in columns}
    else:
        use_dtypes = OPTIMIZED_DTYPES.copy()
    
    # If we need to sample, use chunked reading to avoid loading everything
    if sample_size or sample_frac:
        # Ensure stratify column is included if specified
        read_cols = columns.copy() if columns else None
        if stratify_by and read_cols and stratify_by not in read_cols:
            read_cols.append(stratify_by)
        if read_cols and 'SURVYEAR' not in read_cols:
            read_cols.append('SURVYEAR')
        
        # Calculate sampling rate for chunks
        TOTAL_ROWS = 9_900_000  # Approximate total
        if sample_frac:
            chunk_frac = sample_frac
        else:
            chunk_frac = min(1.0, sample_size / TOTAL_ROWS * 1.5)  # Oversample slightly
        
        chunks = []
        np.random.seed(random_state)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for chunk in pd.read_csv(data_path, chunksize=500_000, 
                                     usecols=read_cols, dtype=use_dtypes):
                # Filter by years if specified
                if years and 'SURVYEAR' in chunk.columns:
                    chunk = chunk[chunk['SURVYEAR'].isin(years)]
                
                # Random sample from this chunk
                if len(chunk) > 0:
                    n_sample = max(1, int(len(chunk) * chunk_frac))
                    chunk_sample = chunk.sample(n=min(n_sample, len(chunk)))
                    chunks.append(chunk_sample)
                
                del chunk
                gc.collect()
        
        df = pd.concat(chunks, ignore_index=True)
        del chunks
        gc.collect()
        
        # Final adjustment to exact sample size
        if sample_size and len(df) > sample_size:
            if stratify_by and stratify_by in df.columns:
                df = stratified_sample(df, stratify_by, sample_size, None, random_state)
            else:
                df = df.sample(n=sample_size, random_state=random_state)
        
        return df.reset_index(drop=True)
    
    # No sampling - still use chunked reading for memory safety
    chunks = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for chunk in pd.read_csv(data_path, chunksize=500_000,
                                 usecols=columns, dtype=use_dtypes):
            if years and 'SURVYEAR' in chunk.columns:
                chunk = chunk[chunk['SURVYEAR'].isin(years)]
            chunks.append(chunk)
            gc.collect()
    
    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()
    return df


def stratified_sample(
    df: pd.DataFrame,
    stratify_col: str,
    sample_size: Optional[int] = None,
    sample_frac: Optional[float] = None,
    random_state: int = 42
) -> pd.DataFrame:
    """Stratified sampling maintaining group proportions."""
    if sample_frac:
        return df.groupby(stratify_col, group_keys=False).apply(
            lambda x: x.sample(frac=sample_frac, random_state=random_state)
        )
    elif sample_size:
        # Calculate per-group sizes maintaining proportions
        group_sizes = df.groupby(stratify_col).size()
        total = len(df)
        samples = []
        for group, size in group_sizes.items():
            n = max(1, int(sample_size * size / total))
            group_df = df[df[stratify_col] == group]
            samples.append(group_df.sample(n=min(n, len(group_df)), random_state=random_state))
        return pd.concat(samples, ignore_index=True)
    return df


def load_chunked(
    chunksize: int = 500_000,
    process_func=None,
    columns: Optional[List[str]] = None,
    data_path: Optional[Path] = None
):
    """
    Process data in chunks for memory-constrained operations.
    
    Args:
        chunksize: Number of rows per chunk
        process_func: Function to apply to each chunk (receives df, returns result)
        columns: Columns to load
        data_path: Data file path
        
    Yields:
        Processed chunk results
        
    Example:
        # Calculate mean wage by year using chunks
        def get_yearly_mean(chunk):
            return chunk.groupby('SURVYEAR')['HRLYEARN'].mean()
        
        results = list(load_chunked(chunksize=1_000_000, process_func=get_yearly_mean))
        combined = pd.concat(results).groupby(level=0).mean()
    """
    data_path = data_path or PROCESSED_FILE
    
    use_dtypes = {k: v for k, v in OPTIMIZED_DTYPES.items() if not columns or k in columns}
    
    for chunk in pd.read_csv(data_path, chunksize=chunksize, usecols=columns, dtype=use_dtypes):
        if process_func:
            yield process_func(chunk)
        else:
            yield chunk
        gc.collect()


# =============================================================================
# PRE-AGGREGATED DATA
# =============================================================================

def create_aggregates(force: bool = False):
    """
    Create pre-aggregated datasets for efficient analysis.
    Generates yearly, provincial, and demographic summaries.
    """
    AGGREGATES_DIR.mkdir(parents=True, exist_ok=True)
    
    yearly_file = AGGREGATES_DIR / 'yearly_stats.csv'
    prov_file = AGGREGATES_DIR / 'provincial_stats.csv'
    demo_file = AGGREGATES_DIR / 'demographic_stats.csv'
    
    if not force and yearly_file.exists() and prov_file.exists():
        print("Aggregates already exist. Use force=True to regenerate.")
        return
    
    print("Creating pre-aggregated datasets (this may take a moment)...")
    
    # Load with minimal columns
    cols = ['SURVYEAR', 'SURVMNTH', 'PROV', 'IS_FEMALE', 'HRLYEARN', 'REAL_HRLYEARN', 
            'EDUC', 'NOC_10', 'AGE_12', 'IS_FULLTIME', 'IS_UNION', 'FINALWT']
    
    # Process in chunks to aggregate
    yearly_results = []
    prov_results = []
    demo_results = []
    
    for chunk in load_chunked(chunksize=1_000_000, columns=cols):
        # Yearly aggregates
        yearly = chunk.groupby('SURVYEAR').agg({
            'HRLYEARN': ['mean', 'median', 'std', 'count'],
            'REAL_HRLYEARN': ['mean', 'median'],
            'IS_FEMALE': 'mean',
            'IS_FULLTIME': 'mean',
            'IS_UNION': 'mean',
            'FINALWT': 'sum'
        })
        yearly.columns = ['_'.join(col).strip() for col in yearly.columns]
        yearly_results.append(yearly)
        
        # Provincial aggregates by year and gender
        prov = chunk.groupby(['SURVYEAR', 'PROV', 'IS_FEMALE']).agg({
            'HRLYEARN': ['mean', 'median', 'count'],
            'REAL_HRLYEARN': 'mean',
            'FINALWT': 'sum'
        })
        prov.columns = ['_'.join(col).strip() for col in prov.columns]
        prov_results.append(prov.reset_index())
        
        # Demographic aggregates
        demo = chunk.groupby(['SURVYEAR', 'IS_FEMALE', 'EDUC', 'NOC_10']).agg({
            'HRLYEARN': ['mean', 'count'],
            'FINALWT': 'sum'
        })
        demo.columns = ['_'.join(col).strip() for col in demo.columns]
        demo_results.append(demo.reset_index())
    
    # Combine and save
    yearly_df = pd.concat(yearly_results).groupby(level=0).mean()
    yearly_df.to_csv(yearly_file)
    print(f"  ✓ Yearly stats: {yearly_file}")
    
    prov_df = pd.concat(prov_results)
    prov_df = prov_df.groupby(['SURVYEAR', 'PROV', 'IS_FEMALE']).sum().reset_index()
    prov_df.to_csv(prov_file, index=False)
    print(f"  ✓ Provincial stats: {prov_file}")
    
    demo_df = pd.concat(demo_results)
    demo_df = demo_df.groupby(['SURVYEAR', 'IS_FEMALE', 'EDUC', 'NOC_10']).sum().reset_index()
    demo_df.to_csv(demo_file, index=False)
    print(f"  ✓ Demographic stats: {demo_file}")
    
    print("Aggregates created successfully!")


def load_yearly_stats() -> pd.DataFrame:
    """Load pre-aggregated yearly statistics."""
    path = AGGREGATES_DIR / 'yearly_stats.csv'
    if not path.exists():
        create_aggregates()
    return pd.read_csv(path, index_col=0)


def load_provincial_stats() -> pd.DataFrame:
    """Load pre-aggregated provincial statistics."""
    path = AGGREGATES_DIR / 'provincial_stats.csv'
    if not path.exists():
        create_aggregates()
    return pd.read_csv(path)


def load_demographic_stats() -> pd.DataFrame:
    """Load pre-aggregated demographic statistics."""
    path = AGGREGATES_DIR / 'demographic_stats.csv'
    if not path.exists():
        create_aggregates()
    return pd.read_csv(path)


# =============================================================================
# MEMORY MONITORING
# =============================================================================

def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage in GB."""
    try:
        import psutil
        process = psutil.Process()
        return {
            'rss_gb': process.memory_info().rss / 1e9,
            'vms_gb': process.memory_info().vms / 1e9,
            'percent': process.memory_percent()
        }
    except ImportError:
        return {'rss_gb': 0, 'vms_gb': 0, 'percent': 0}


def print_memory_status():
    """Print current memory status."""
    mem = get_memory_usage()
    print(f"Memory: {mem['rss_gb']:.2f} GB ({mem['percent']:.1f}%)")


def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame memory by converting to efficient dtypes."""
    for col in df.columns:
        if col in OPTIMIZED_DTYPES:
            try:
                df[col] = df[col].astype(OPTIMIZED_DTYPES[col])
            except (ValueError, TypeError):
                pass
    return df


# =============================================================================
# SAMPLE SIZE RECOMMENDATIONS
# =============================================================================

SAMPLE_RECOMMENDATIONS = {
    'exploration': 100_000,      # Quick data exploration
    'visualization': 50_000,     # Charts and plots
    'regression': 500_000,       # OLS, logistic regression
    'quantile_regression': 100_000,  # Quantile regression (memory intensive)
    'ml_training': 500_000,      # Machine learning models
    'ml_gridsearch': 100_000,    # Hyperparameter tuning
    'bootstrap': 10_000,         # Bootstrap confidence intervals
    'effect_size': 10_000,       # Cohen's d, etc. (converges quickly)
    'fairness': 200_000,         # Fairness metrics
    'decomposition': 300_000,    # Oaxaca-Blinder decomposition
    'time_series': 'aggregate',  # Use pre-aggregated yearly data
    'geographic': 'aggregate',   # Use pre-aggregated provincial data
}


def get_recommended_sample(analysis_type: str) -> Union[int, str]:
    """Get recommended sample size for a given analysis type."""
    return SAMPLE_RECOMMENDATIONS.get(analysis_type, 100_000)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def load_for_ml(sample_size: int = 500_000) -> pd.DataFrame:
    """Load data optimized for machine learning."""
    cols = ['SURVYEAR', 'PROV', 'EDUC', 'AGE_12', 'NOC_10', 'NAICS_21',
            'FTPTMAIN', 'UNION', 'IS_FEMALE', 'EXPERIENCE', 'EXPERIENCE_SQ',
            'HRLYEARN', 'REAL_HRLYEARN', 'LOG_HRLYEARN', 'FINALWT']
    return load_data(sample_size=sample_size, columns=cols, stratify_by='SURVYEAR')


def load_for_equity(sample_size: int = 300_000) -> pd.DataFrame:
    """Load data optimized for pay equity analysis."""
    cols = ['SURVYEAR', 'PROV', 'EDUC', 'AGE_12', 'NOC_10',
            'IS_FEMALE', 'EXPERIENCE', 'EXPERIENCE_SQ',
            'HRLYEARN', 'REAL_HRLYEARN', 'LOG_HRLYEARN', 'FINALWT',
            'IS_FULLTIME', 'IS_UNION']
    return load_data(sample_size=sample_size, columns=cols, stratify_by='IS_FEMALE')


def load_for_regression(sample_size: int = 500_000) -> pd.DataFrame:
    """Load data optimized for econometric analysis."""
    cols = ['SURVYEAR', 'PROV', 'EDUC', 'AGE_12', 'NOC_10', 'NAICS_21',
            'IS_FEMALE', 'EXPERIENCE', 'EXPERIENCE_SQ',
            'LOG_HRLYEARN', 'LOG_REAL_HRLYEARN', 'HRLYEARN', 'REAL_HRLYEARN',
            'IS_FULLTIME', 'IS_UNION', 'FINALWT']
    return load_data(sample_size=sample_size, columns=cols, stratify_by='SURVYEAR')
