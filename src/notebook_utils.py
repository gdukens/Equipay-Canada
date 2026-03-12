"""
Notebook utility helpers for running notebooks in FAST mode in CI/smoke tests.

Provides:
- ensure_store_and_sample: ensures a store and registers a small sample table if no parquet data is available
- get_sample_from_store: safe sampling helper with synthetic fallback
- make_synthetic_sample: produces a small DataFrame with expected columns
- safe_weight_col: ensures a weight column exists on a DataFrame
"""

from typing import Optional, Tuple
import logging
import pandas as pd
from .data_store.core import EquiPayDataStore

logger = logging.getLogger(__name__)


def make_synthetic_sample(n: int = 4) -> pd.DataFrame:
    """Create a tiny synthetic sample that mimics key columns used in notebooks."""
    data = {
        'SURVYEAR': [2010 + i for i in range(n)],
        'HRLYEARN': [20.0 + 1.0 * i for i in range(n)],
        'REAL_HRLYEARN': [20.0 + 1.0 * i for i in range(n)],
        'FINALWT': [100.0 for _ in range(n)],
        'PROV': [1 + (i % 3) for i in range(n)],
        'GENDER': [1 if i % 2 == 0 else 2 for i in range(n)],
        'IS_FEMALE': [0 if i % 2 == 0 else 1 for i in range(n)],
        'NOC_10': [1000 + i for i in range(n)]
    }
    df = pd.DataFrame(data)
    return df


def get_sample_from_store(store: Optional[EquiPayDataStore] = None, query: Optional[str] = None, limit: int = 1000) -> pd.DataFrame:
    """Attempt to fetch a sample from store; fallback to synthetic sample when no data available."""
    if store is None:
        try:
            store = EquiPayDataStore()
        except Exception as e:
            logger.warning(f"Could not instantiate EquiPayDataStore: {e}")
            return make_synthetic_sample(min(4, limit))

    try:
        if store.count() > 0:
            if query:
                sql = f"SELECT * FROM {store.TABLE_NAME} WHERE {query} LIMIT {limit}"
                return store.sql(sql)
            else:
                # Prefer store.sample when available
                try:
                    return store.sample(n=limit)
                except Exception:
                    # Fallback to limit query
                    return store.sql(f"SELECT * FROM {store.TABLE_NAME} LIMIT {limit}")
    except Exception as e:
        logger.warning(f"Failed to sample from store: {e}")

    # Fallback synthetic sample
    logger.info("Using synthetic sample (no parquet data available)")
    return make_synthetic_sample(min(4, limit))


def ensure_store_and_sample(store: Optional[EquiPayDataStore] = None, df_sample: Optional[pd.DataFrame] = None, table_name: str = 'df', sample_limit: int = 1000) -> Tuple[EquiPayDataStore, pd.DataFrame]:
    """Ensure a store exists and that a small sample table named `table_name` is registered if no parquet data found.

    Returns (store, df_sample).
    """
    if store is None:
        store = EquiPayDataStore()

    if df_sample is None:
        df_sample = get_sample_from_store(store, limit=sample_limit)

    try:
        if store.count() == 0:
            # Register sample as a table needed by many notebook SQL snippets
            if isinstance(df_sample, pd.DataFrame) and len(df_sample) > 0:
                store.register_df(df_sample, name=table_name, replace=True)
                logger.info(f"Registered sample DataFrame as table '{table_name}' in DuckDB for FAST runs")
            else:
                logger.warning("No sample available to register as table; notebooks may fail if they expect SQL tables.")
    except Exception as e:
        logger.exception(f"Error while ensuring store and sample: {e}")

    return store, df_sample


def safe_weight_col(df: pd.DataFrame, weight_col: str = 'FINALWT') -> str:
    """Ensure the weight column exists in df, create it with ones if missing."""
    if weight_col not in df.columns:
        logger.warning(f"Weight column '{weight_col}' not found in DataFrame; creating default weight=1")
        df[weight_col] = 1.0
    return weight_col
