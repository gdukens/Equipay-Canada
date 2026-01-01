"""
Check Experience Variables in LFS Data
=======================================

Analyzes how experience affects the gender wage gap using LFS PUMF data.

Data Source: LFS PUMF only (2010-2025)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.constants import COLS, GENDER_CODES_REVERSE, normalize_column_names

df = pd.read_csv('data/processed/lfs_processed.csv')
df = normalize_column_names(df)

# Identify gender column
gender_col = COLS.GENDER if COLS.GENDER in df.columns else 'SEX'

# Current experience proxy: AGE_12 * 3
df['EXPERIENCE'] = df[COLS.AGE_12] * 3 if COLS.AGE_12 in df.columns else df.get('AGE_12', 10) * 3
df['EXPERIENCE_SQ'] = df['EXPERIENCE'] ** 2

# Create IS_FEMALE based on gender column
if df[gender_col].dtype == 'object':
    df['IS_FEMALE'] = (df[gender_col] == 'Women+').astype(int)
else:
    df['IS_FEMALE'] = (df[gender_col] == GENDER_CODES_REVERSE['Female']).astype(int)

df['LOG_WAGE'] = np.log(df[COLS.HRLYEARN].clip(lower=1))

print('='*60)
print('EXPERIENCE VARIABLE IN THE MODEL')
print('='*60)
print(f'Experience proxy: AGE_12 * 3 (age category x 3 years)')
print(f'Experience range: {df.EXPERIENCE.min():.0f} - {df.EXPERIENCE.max():.0f} years')
print(f'Mean experience: {df.EXPERIENCE.mean():.1f} years')
print()

# Mincer wage equation with experience
educ_col = COLS.EDUC if COLS.EDUC in df.columns else 'EDUC'
features = ['IS_FEMALE', educ_col, 'EXPERIENCE', 'EXPERIENCE_SQ']
df_clean = df[features + ['LOG_WAGE']].dropna()

X = sm.add_constant(df_clean[features])
y = df_clean['LOG_WAGE']
model = sm.OLS(y, X).fit()

print('MINCER WAGE EQUATION RESULTS:')
print('-'*60)
for var in ['const', 'IS_FEMALE', educ_col, 'EXPERIENCE', 'EXPERIENCE_SQ']:
    coef = model.params[var]
    pval = model.pvalues[var]
    sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
    print(f'{var:15} B = {coef:8.4f}  (p = {pval:.4f}) {sig}')

print()
print('INTERPRETATION:')
gap = (np.exp(model.params['IS_FEMALE'])-1)*100
exp_effect = model.params['EXPERIENCE']*100
exp_sq = model.params['EXPERIENCE_SQ']*100
print(f'  - Gender gap (controlling for experience): {gap:.1f}%')
print(f'  - Each year of experience: +{exp_effect:.2f}% wage')
print(f'  - Experience squared (diminishing returns): {exp_sq:.4f}%')
print()

# Compare models WITH and WITHOUT experience
print('='*60)
print('COMPARISON: WITH vs WITHOUT EXPERIENCE CONTROL')
print('='*60)

# Without experience
X_no_exp = sm.add_constant(df[['IS_FEMALE', 'EDUC']])
model_no_exp = sm.OLS(y, X_no_exp).fit()

gap_no_exp = (np.exp(model_no_exp.params['IS_FEMALE'])-1)*100
gap_with_exp = (np.exp(model.params['IS_FEMALE'])-1)*100

print(f'Gender gap WITHOUT experience control: {gap_no_exp:.1f}%')
print(f'Gender gap WITH experience control:    {gap_with_exp:.1f}%')
print(f'Difference: {abs(gap_no_exp - gap_with_exp):.1f} percentage points')
print()

# Experience by gender
print('='*60)
print('EXPERIENCE DISTRIBUTION BY GENDER')
print('='*60)
male_exp = df[df[gender_col]==GENDER_CODES_REVERSE['Male']]['EXPERIENCE'].mean()
female_exp = df[df[gender_col]==GENDER_CODES_REVERSE['Female']]['EXPERIENCE'].mean()
print(f'Male mean experience:   {male_exp:.1f} years')
print(f'Female mean experience: {female_exp:.1f} years')
print(f'Difference: {male_exp - female_exp:.1f} years')
