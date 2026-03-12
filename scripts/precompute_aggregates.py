"""
Precompute canonical aggregates for dashboard, API, and publication outputs.
Saves results to reports/cache/ as Parquet files.
"""
import os
import pandas as pd
from pathlib import Path
import sys
# Ensure project root is importable when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_store import EquiPayDataStore, Agg

CACHE_DIR = Path(__file__).parent.parent / 'reports' / 'cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

PARQUET_PATH = Path(__file__).parent.parent / 'data' / 'parquet'
store = EquiPayDataStore(parquet_path=str(PARQUET_PATH), memory_limit_mb=1000, enable_cache=True)

import os
EQUIPAY_MODE = os.environ.get('EQUIPAY_MODE', 'FULL')  # FAST for interactive/smoke-tests, FULL for production
# Conservative default: in FAST mode we require larger groups when computing per-province aggregates
MIN_GROUP_N = 1000 if EQUIPAY_MODE == 'FAST' else 30

# 1. Annual gender gap time series (SQL-aggregated, memory-efficient)
def compute_annual_gap():
    q = """
    SELECT YEAR,
           SUM(CASE WHEN GENDER = 1 THEN REAL_HRLYEARN * FINALWT END) / NULLIF(SUM(CASE WHEN GENDER = 1 THEN FINALWT END),0) AS mean_male,
           SUM(CASE WHEN GENDER = 2 THEN REAL_HRLYEARN * FINALWT END) / NULLIF(SUM(CASE WHEN GENDER = 2 THEN FINALWT END),0) AS mean_female
    FROM lfs_enriched
    WHERE HRLYEARN > 0
    GROUP BY YEAR
    ORDER BY YEAR
    """
    annual = store.sql(q)
    annual['gap_pct'] = (annual['mean_male'] - annual['mean_female']) / annual['mean_male'] * 100
    annual.to_parquet(CACHE_DIR / 'annual_gap.parquet', index=False)
    print('✓ annual_gap.parquet saved')

# 2. Provincial means (latest year) - computed with SQL and MIN_GROUP_N filter
def compute_provincial_means():
    # Get latest year using the store metadata (robust to column case and empty results)
    years = store.sql("SELECT DISTINCT YEAR FROM lfs_enriched ORDER BY YEAR DESC LIMIT 1")
    if years.empty:
        print('Warning: No years found in store; skipping provincial means')
        return
    col = years.columns[0]
    try:
        latest_year = int(years.iloc[0][col])
    except Exception as e:
        print(f'Warning: Could not parse latest year from store ({e}); skipping provincial means')
        return

    q = f"""
    SELECT PROV AS province,
           SUM(CASE WHEN GENDER = 1 THEN REAL_HRLYEARN * FINALWT END) / NULLIF(SUM(CASE WHEN GENDER = 1 THEN FINALWT END),0) AS mean_male,
           SUM(CASE WHEN GENDER = 2 THEN REAL_HRLYEARN * FINALWT END) / NULLIF(SUM(CASE WHEN GENDER = 2 THEN FINALWT END),0) AS mean_female,
           COUNT(*) AS N
    FROM lfs_enriched
    WHERE YEAR = {latest_year}
    GROUP BY PROV
    HAVING COUNT(*) >= {MIN_GROUP_N}
    ORDER BY province
    """
    try:
        prov = store.sql(q)
        if prov.empty:
            print(f'Warning: provincial query returned no rows (YEAR={latest_year}); skipping save')
            return
        prov['gap_pct'] = (prov['mean_male'] - prov['mean_female']) / prov['mean_male'] * 100
        prov.to_parquet(CACHE_DIR / 'provincial_means.parquet', index=False)
        print('✓ provincial_means.parquet saved')
    except Exception as e:
        print(f'Error computing provincial means: {e}')

if __name__ == '__main__':
    compute_annual_gap()
    compute_provincial_means()
