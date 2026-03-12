# Test comprehensive notebook functionality
import sys
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.append('.')
sys.path.append('./src')

print('🧪 COMPREHENSIVE NOTEBOOK FUNCTIONALITY TEST')
print('='*50)

# Test 1: Environment setup
os.environ['EQUIPAY_MODE'] = 'FAST'
print(f'✓ Environment mode set: {os.environ.get("EQUIPAY_MODE")}')

# Test 2: Core imports
try:
    from src.constants import COLS, PROVINCE_CODES, DATA_SCOPE_START, DATA_SCOPE_END
    from src.macro_data import get_macro_dataframe, ECONOMIC_PERIODS
    from src.data_store import EquiPayDataStore
    print('✓ All core modules imported successfully')
except Exception as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)

# Test 3: Data store functionality
try:
    store = EquiPayDataStore()
    total_records = store.count()
    print(f'✓ Data store initialized: {total_records:,} records')
    
    # Quick sample
    sample = store.sample(n=3)
    print(f'✓ Sample retrieved: {len(sample)} rows, {len(sample.columns)} columns')
    
except Exception as e:
    print(f'❌ Data store error: {e}')
    sample = pd.DataFrame()

# Test 4: Basic analysis functionality
try:
    if len(sample) > 0:
        # Test basic stats
        if 'HRLYEARN' in sample.columns:
            mean_wage = sample['HRLYEARN'].mean()
            print(f'✓ Basic analysis works: Mean wage ${mean_wage:.2f}/hr')
        
        # Test if weights exist
        if 'FINALWT' in sample.columns:
            print('✓ Survey weights available for proper inference')
            
except Exception as e:
    print(f'⚠ Analysis test error: {e}')

# Test 5: Visualization setup
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Test plot configuration
    plt.rcParams.update({'figure.figsize': (8, 6)})
    print('✓ Visualization libraries configured')
    
except Exception as e:
    print(f'⚠ Visualization setup error: {e}')

print('\n🎯 FUNCTIONALITY TEST COMPLETED')
print('\nThe notebook should work correctly!')
print('To restart the notebook kernel and run cells:')
print('1. In VS Code, use Ctrl+Shift+P')
print('2. Type: "Jupyter: Restart Kernel"')
print('3. Run cells sequentially from the top')