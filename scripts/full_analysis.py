"""
Full Analysis Script - Gender Pay Gap in Canada
Comprehensive econometric analysis with experience, tenure, and controls
Uses standardized column names from src/constants.py
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from pathlib import Path
import sys

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.constants import COLS, normalize_column_names, GENDER_CODES, GENDER_CODES_REVERSE

df = pd.read_csv('data/processed/lfs_processed.csv')
df = normalize_column_names(df)

# Determine gender column (support both GENDER and SEX for backward compatibility)
if COLS.GENDER in df.columns:
    gender_col = COLS.GENDER
elif 'SEX' in df.columns:
    gender_col = 'SEX'
else:
    raise ValueError("No gender column (GENDER or SEX) found in data")

print('='*70)
print('COMPREHENSIVE PAY EQUITY ANALYSIS')
print('With Experience, Tenure, and Inclusive Gender Variables')
print('='*70)

print('\n1. DATA OVERVIEW')
print('-'*70)
print(f'Total observations: {len(df):,}')
if 'source' in df.columns:
    print(f'Data source: {df["source"].iloc[0]}')
print(f'\nGender distribution:')
print(df[gender_col].value_counts())

print('\n2. EXPERIENCE & TENURE SUMMARY')
print('-'*70)

# Check for EXPERIENCE and TENURE columns
has_experience = 'EXPERIENCE' in df.columns
has_tenure = COLS.TENURE in df.columns

if has_experience:
    print(f'Experience (years): Mean={df["EXPERIENCE"].mean():.1f}, Median={df["EXPERIENCE"].median():.1f}')
if has_tenure:
    print(f'Tenure (years): Mean={df[COLS.TENURE].mean():.1f}, Median={df[COLS.TENURE].median():.1f}')

print('\nBy Gender:')
for gender in sorted(df[gender_col].unique()):
    subset = df[df[gender_col] == gender]
    print(f'  {gender}:')
    print(f'    N = {len(subset):,}')
    print(f'    Mean wage = ${subset[COLS.HRLYEARN].mean():.2f}/hr')
    if has_experience:
        print(f'    Mean experience = {subset["EXPERIENCE"].mean():.1f} years')
    if has_tenure:
        print(f'    Mean tenure = {subset[COLS.TENURE].mean():.1f} years')
    if COLS.EDUC in df.columns:
        print(f'    Mean education = {subset[COLS.EDUC].mean():.1f}')

print('\n3. MINCER WAGE EQUATION (Full Model)')
print('-'*70)
print('log(wage) = β₀ + β₁·Female + β₂·Education + β₃·Experience + β₄·Experience² + β₅·Tenure + ε')

# Prepare variables
df['LOG_WAGE'] = np.log(df[COLS.HRLYEARN].clip(lower=1))

# Create IS_FEMALE based on gender column type
if df[gender_col].dtype == 'object':
    df['IS_FEMALE'] = (df[gender_col] == 'Women+').astype(int)
else:
    df['IS_FEMALE'] = (df[gender_col] == GENDER_CODES_REVERSE['Female']).astype(int)

# Build feature list based on available columns
features = ['IS_FEMALE']
if COLS.EDUC in df.columns:
    features.append(COLS.EDUC)
if has_experience:
    features.append('EXPERIENCE')
    if 'EXPERIENCE_SQ' not in df.columns:
        df['EXPERIENCE_SQ'] = df['EXPERIENCE'] ** 2
    features.append('EXPERIENCE_SQ')
if has_tenure:
    features.append(COLS.TENURE)

X = df[features].copy()
X = sm.add_constant(X)
y = df['LOG_WAGE']

model = sm.OLS(y, X).fit()

print('\nRegression Results:')
print('-'*70)
print(f'{"Variable":<20} {"Coef":>10} {"Std Err":>10} {"t":>10} {"P>|t|":>10}')
print('-'*70)
for var in model.params.index:
    coef = model.params[var]
    se = model.bse[var]
    t = model.tvalues[var]
    p = model.pvalues[var]
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    print(f'{var:<20} {coef:>10.4f} {se:>10.4f} {t:>10.2f} {p:>10.4f} {sig}')

print('-'*70)
print(f'R-squared: {model.rsquared:.4f}')
print(f'Adj. R-squared: {model.rsquared_adj:.4f}')
print(f'N: {int(model.nobs):,}')

# Interpretation
female_coef = model.params['IS_FEMALE']
female_pct = (np.exp(female_coef) - 1) * 100

print('\n4. KEY FINDINGS')
print('-'*70)
print(f'GENDER WAGE GAP (controlling for available covariates):')
print(f'  Coefficient: {female_coef:.4f}')
print(f'  Interpretation: Women earn {abs(female_pct):.1f}% {"less" if female_pct < 0 else "more"} than Men')
print(f'  Statistical significance: p = {model.pvalues["IS_FEMALE"]:.6f}')

if has_experience and 'EXPERIENCE' in model.params:
    exp_return = model.params['EXPERIENCE'] * 100
    print(f'\nRETURNS TO EXPERIENCE:')
    print(f'  Each additional year of experience: +{exp_return:.2f}% wage')
    if 'EXPERIENCE_SQ' in model.params:
        print(f'  Experience² (diminishing returns): {model.params["EXPERIENCE_SQ"]*100:.4f}%')

if has_tenure and COLS.TENURE in model.params:
    tenure_return = model.params[COLS.TENURE] * 100
    print(f'\nRETURNS TO TENURE:')
    print(f'  Each additional year with current employer: +{tenure_return:.2f}% wage')

print('\n5. MODEL COMPARISON')
print('-'*70)

# Model without experience/tenure (if we have EDUC)
if COLS.EDUC in df.columns:
    X_basic = df[['IS_FEMALE', COLS.EDUC]].copy()
    X_basic = sm.add_constant(X_basic)
    model_basic = sm.OLS(y, X_basic).fit()
    gap_basic = (np.exp(model_basic.params['IS_FEMALE']) - 1) * 100
    
    print(f'Model 1 (Gender + Education only):')
    print(f'  Gender gap: {abs(gap_basic):.1f}%')
    print(f'  R-squared: {model_basic.rsquared:.4f}')
    
    print(f'\nModel 2 (Full model with all available controls):')
    print(f'  Gender gap: {abs(female_pct):.1f}%')
    print(f'  R-squared: {model.rsquared:.4f}')
    
    gap_change = abs(gap_basic) - abs(female_pct)
    print(f'\nAdditional controls explain: {gap_change:.1f} percentage points of the gap')
else:
    print('Skipping model comparison (EDUC not available)')

print('\n6. DATA LIMITATIONS')
print('-'*70)
print('''
IMPORTANT NOTES ON GENDER VARIABLE:

1. The Statistics Canada LFS uses binary SEX coding (Male/Female)
   
2. Following StatCan's 2021+ Gender Framework:
   - "Men+" includes cisgender men, transgender men, and some 
     non-binary persons who identify more closely with this category
   - "Women+" includes cisgender women, transgender women, and some
     non-binary persons who identify more closely with this category

3. LIMITATIONS:
   - No separate category for non-binary, Two-Spirit, or gender diverse persons
   - This underrepresents the full spectrum of gender identities
   - Statistics Canada is working on expanding gender variables

4. For complete gender-inclusive analysis, additional data sources
   such as the Canadian Community Health Survey (CCHS) or Census
   may provide more detailed gender identity categories.

Reference: https://www.statcan.gc.ca/en/concepts/definitions/gender-sex
''')
