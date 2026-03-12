import sys
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_store import EquiPayDataStore
from src.macro_data import MACRO_DATA


def test_register_macro_creates_table(tmp_path):
    # No parquet needed; registering macro should create a 'macro' table with deflator
    store = EquiPayDataStore(parquet_path=str(tmp_path), use_sql_transforms=False)
    store._register_macro()

    df = store.sql("SELECT year, cpi, deflator FROM macro WHERE year = 2010")
    assert not df.empty
    assert 'deflator' in df.columns
    assert df.iloc[0]['cpi'] == MACRO_DATA[2010]['cpi']


def test_materialized_view_created_with_sample_parquet(tmp_path):
    # Create a tiny parquet dataset that the store can register and create the enriched view
    sample = pd.DataFrame({
        'SURVYEAR': [2010, 2010],
        'GENDER': [1, 2],
        'HRLYEARN': [20.0, 18.0],
        'FINALWT': [1000, 1100],
        'EDUC': [4, 2],
        'AGE_6': [2, 3],
        'PROV': [35, 35]
    })

    parquet_dir = tmp_path / 'parquet'
    parquet_dir.mkdir()
    sample_file = parquet_dir / 'sample.parquet'
    sample.to_parquet(sample_file, index=False)

    # Initialize store - this should create macro and lfs_enriched view
    store = EquiPayDataStore(parquet_path=str(parquet_dir), use_sql_transforms=True)

    # Confirm lfs_enriched view exists
    res = store.sql("SELECT table_name FROM information_schema.tables WHERE table_name = 'lfs_enriched'")
    assert not res.empty
