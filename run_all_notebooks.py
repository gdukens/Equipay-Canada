#!/usr/bin/env python
"""
Run all EquiPay Canada Jupyter notebooks in sequence.
"""

import subprocess
import sys
from pathlib import Path

# Notebook execution order
NOTEBOOKS = [
    "01_data_exploration.ipynb",
    "02_model_training.ipynb",
    "03_pay_equity_analysis.ipynb",
    "04_fairness_evaluation.ipynb",
    "05_econometric_analysis.ipynb",
    "06_time_series_analysis.ipynb",
    "07_advanced_statistics.ipynb",
    "08_geographic_analysis.ipynb",
]

def run_notebook_as_script(notebook_path: Path) -> bool:
    """
    Execute a notebook by converting it to a Python script and running it.
    """
    import json
    
    print(f"\n{'='*70}")
    print(f"Running: {notebook_path.name}")
    print('='*70)
    
    try:
        # Read the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        # Extract code cells
        code_cells = []
        for cell in notebook.get('cells', []):
            if cell.get('cell_type') == 'code':
                source = cell.get('source', [])
                if isinstance(source, list):
                    code = ''.join(source)
                else:
                    code = source
                # Skip cells with magic commands that won't work
                if not code.strip().startswith('!') and not code.strip().startswith('%'):
                    code_cells.append(code)
        
        # Create a temporary Python script
        script_content = f'''# Auto-generated from {notebook_path.name}
import warnings
warnings.filterwarnings('ignore')

# Ensure `display` is defined when running outside IPython
try:
    from IPython.display import display
except Exception:
    def display(obj, **kwargs):
        print(obj)

import sys
from pathlib import Path

# Add project root to path
project_root = Path(r"{notebook_path.parent.parent}")
sys.path.insert(0, str(project_root))

# Ensure project root is current working directory and add src to sys.path so package imports work
import os
os.chdir(str(project_root))
sys.path.insert(0, str(project_root / 'src'))
# Ensure EQUIPAY_MODE=FAST by default for automated smoke tests
os.environ.setdefault('EQUIPAY_MODE', 'FAST')

# Make src subpackages available under expected top-level names to preserve relative imports
import importlib
try:
    import src
    importlib.import_module('src.data_store')
    # alias common packages if notebooks import them as top-level modules
    sys.modules['data_store'] = sys.modules.get('src.data_store')
except Exception:
    # proceed; imports in notebooks will raise meaningful errors
    pass

# Notebook helpers: ensure a small sample table and store exist for FAST-mode smoke tests
try:
    from src.notebook_utils import ensure_store_and_sample, get_sample_from_store, safe_weight_col
    store, df_sample = ensure_store_and_sample()
    # Make df_sample available under a convenient name for compatibility with notebook variables
    globals()['df_sample'] = df_sample
except Exception as e:
    print("Warning: notebook helpers not available:", e)

'''
        script_content += '\n\n'.join(code_cells)
        
        # Write temporary script
        temp_script = notebook_path.parent / f"_temp_{notebook_path.stem}.py"
        with open(temp_script, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        # Run the script
        result = subprocess.run(
            [sys.executable, str(temp_script)],
            capture_output=False,
            text=True,
            cwd=str(notebook_path.parent)
        )
        
        # Clean up
        temp_script.unlink()
        
        if result.returncode == 0:
            print(f"✓ {notebook_path.name} completed successfully")
            return True
        else:
            print(f"✗ {notebook_path.name} failed with code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"✗ Error running {notebook_path.name}: {e}")
        return False


def main():
    """Run all notebooks."""
    print("="*70)
    print("EQUIPAY CANADA - RUNNING ALL ANALYSIS NOTEBOOKS")
    print("="*70)
    
    notebooks_dir = Path(__file__).parent / "notebooks"
    
    results = {}
    for nb_name in NOTEBOOKS:
        nb_path = notebooks_dir / nb_name
        if nb_path.exists():
            success = run_notebook_as_script(nb_path)
            results[nb_name] = "✓ Success" if success else "✗ Failed"
        else:
            print(f"⚠ Notebook not found: {nb_name}")
            results[nb_name] = "⚠ Not found"
    
    # Summary
    print("\n" + "="*70)
    print("EXECUTION SUMMARY")
    print("="*70)
    for nb, status in results.items():
        print(f"  {nb}: {status}")
    
    print("\n" + "="*70)
    print("ALL NOTEBOOKS PROCESSED")
    print("="*70)


if __name__ == "__main__":
    main()
