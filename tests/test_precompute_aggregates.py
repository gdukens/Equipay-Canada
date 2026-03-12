import os
import sys
import pandas as pd
from pathlib import Path
import importlib

# Ensure project root is importable when running tests
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest


def test_compute_annual_gap_and_provincial_means(tmp_path, monkeypatch):
    # Import module fresh
    import scripts.precompute_aggregates as pre

    # Use a fresh EquiPayDataStore and register a small sample
    from src.data_store import EquiPayDataStore
    store = EquiPayDataStore(parquet_path=str(tmp_path / 'parquet'), memory_limit_mb=100)

    df = pd.DataFrame({
        'YEAR': [2020, 2020, 2020, 2021],
        'SURVYEAR': [2020, 2020, 2020, 2021],
        'HRLYEARN': [10, 20, 30, 40],
        'REAL_HRLYEARN': [10, 20, 30, 40],
        'FINALWT': [1, 1, 1, 1],
        'GENDER': [1, 2, 1, 2],
        'PROV': [10, 10, 20, 20]
    })

    store.register_df(df, name='lfs_enriched', replace=True)

    monkeypatch.setattr(pre, 'store', store)
    monkeypatch.setattr(pre, 'CACHE_DIR', tmp_path)

    # Ensure MIN_GROUP_N small so provinces are included
    monkeypatch.setattr(pre, 'MIN_GROUP_N', 1)

    # Run annual gap
    pre.compute_annual_gap()
    assert (tmp_path / 'annual_gap.parquet').exists()
    annual = pd.read_parquet(tmp_path / 'annual_gap.parquet')
    assert 'gap_pct' in annual.columns
    assert not annual.empty

    # Run provincial means
    pre.compute_provincial_means()
    assert (tmp_path / 'provincial_means.parquet').exists()
    prov = pd.read_parquet(tmp_path / 'provincial_means.parquet')
    assert 'gap_pct' in prov.columns
    assert prov['province'].nunique() >= 1


def test_compute_provincial_means_no_years(monkeypatch, tmp_path, capsys):
    import scripts.precompute_aggregates as pre

    # Replace store with dummy that returns empty DataFrame for years
    class DummyStore:
        def sql(self, q):
            import pandas as pd
            return pd.DataFrame()

    monkeypatch.setattr(pre, 'store', DummyStore())
    monkeypatch.setattr(pre, 'CACHE_DIR', tmp_path)

    pre.compute_provincial_means()
    captured = capsys.readouterr()
    assert 'Warning: No years found in store' in captured.out


def test_compute_provincial_means_min_group_skips(tmp_path, monkeypatch, capsys):
    import scripts.precompute_aggregates as pre
    from src.data_store import EquiPayDataStore

    store = EquiPayDataStore(parquet_path=str(tmp_path / 'parquet'), memory_limit_mb=100)

    # Single-year data with small groups
    df = pd.DataFrame({
        'YEAR': [2022, 2022],
        'SURVYEAR': [2022, 2022],
        'HRLYEARN': [10, 20],
        'REAL_HRLYEARN': [10, 20],
        'FINALWT': [1, 1],
        'GENDER': [1, 2],
        'PROV': [10, 20]
    })

    store.register_df(df, name='lfs_enriched', replace=True)
    monkeypatch.setattr(pre, 'store', store)
    monkeypatch.setattr(pre, 'CACHE_DIR', tmp_path)

    # Set MIN_GROUP_N larger than available groups so we skip saving
    monkeypatch.setattr(pre, 'MIN_GROUP_N', 100)

    pre.compute_provincial_means()
    # No file should be created
    assert not (tmp_path / 'provincial_means.parquet').exists()
    captured = capsys.readouterr()
    assert 'returned no rows' in captured.out or 'skipping save' in captured.out
