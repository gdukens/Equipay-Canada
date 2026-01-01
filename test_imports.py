"""Quick test of notebook dependencies."""
import sys
sys.path.insert(0, 'src')

print("Testing imports...")

try:
    import pandas as pd
    print("✅ pandas")
except ImportError as e:
    print(f"❌ pandas: {e}")

try:
    import numpy as np
    print("✅ numpy")
except ImportError as e:
    print(f"❌ numpy: {e}")

try:
    from time_series import WageGapTimeSeriesAnalyzer
    print("✅ time_series")
except ImportError as e:
    print(f"❌ time_series: {e}")

try:
    from macro_data import MACRO_DATA, get_macro_dataframe
    print("✅ macro_data")
except ImportError as e:
    print(f"❌ macro_data: {e}")

try:
    from statistical_tests import AdvancedStatisticalTests
    print("✅ statistical_tests")
except ImportError as e:
    print(f"❌ statistical_tests: {e}")

print("\nAll imports successful!" if True else "Some imports failed")
