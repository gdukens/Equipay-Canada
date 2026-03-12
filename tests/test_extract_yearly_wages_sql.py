import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.extract_yearly_wages import extract_yearly_wage_timeseries


def test_extract_yearly_wages_sql_path(tmp_path):
    # Create a small parquet dataset for the script to use
    sample = pd.DataFrame({
        'SURVYEAR': [2010, 2010, 2011],
        'GENDER': [1, 2, 1],
        'HRLYEARN': [20.0, 18.0, 22.0],
        'FINALWT': [1000, 1100, 1200],
        'EDUC': [4, 2, 4],
        'AGE_6': [2, 3, 2],
        'PROV': [35, 35, 35]
    })

    parquet_dir = tmp_path / 'parquet'
    parquet_dir.mkdir()
    sample_file = parquet_dir / 'sample.parquet'
    sample.to_parquet(sample_file, index=False)

    # Monkeypatch the script's expected parquet path by creating data/parquet symlink into tmp dir
    data_dir = Path('data')
    orig_exists = data_dir.exists()
    data_dir.mkdir(exist_ok=True)
    data_parquet = data_dir / 'parquet'
    data_parquet.mkdir(parents=True, exist_ok=True)

    # Copy sample parquet into repository data/parquet so the script reads it
    import shutil
    shutil.copy(sample_file, data_parquet / 'sample.parquet')

    try:
        df = extract_yearly_wage_timeseries(parquet_path=str(parquet_dir))
        assert df is not None
        assert 'female_wage' in df.columns
        assert 'male_wage' in df.columns
    finally:
        # cleanup: remove added file
        try:
            (data_parquet / 'sample.parquet').unlink()
        except Exception:
            pass
        if not orig_exists:
            # if data dir was created by test, remove it if empty
            try:
                data_parquet.rmdir()
                data_dir.rmdir()
            except Exception:
                pass
