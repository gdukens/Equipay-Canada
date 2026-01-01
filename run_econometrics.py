"""
Econometric Analysis Script - Gender Pay Gap in Canada
Runs Oaxaca-Blinder Decomposition, Quantile Regression, and Robust SE comparison

Uses standardized column names from src/constants.py
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
import os
import sys

# Import constants for consistent column naming
from src.constants import COLS, normalize_column_names, GENDER_CODES, GENDER_CODES_REVERSE, humanize_columns

# Load data
df = pd.read_csv('data/processed/lfs_processed.csv')
df = normalize_column_names(df)

# Determine gender column (support both GENDER and SEX for backward compatibility)
if COLS.GENDER in df.columns:
    gender_col = COLS.GENDER
elif 'SEX' in df.columns:
    gender_col = 'SEX'
else:
    raise ValueError("No gender column (GENDER or SEX) found in data")

# Create IS_FEMALE if not exists
if 'IS_FEMALE' not in df.columns:
    df['IS_FEMALE'] = (df[gender_col] == GENDER_CODES_REVERSE['Female']).astype(int)

# Create LOG_HRLYEARN if not exists
if 'LOG_HRLYEARN' not in df.columns and COLS.HRLYEARN in df.columns:
    df['LOG_HRLYEARN'] = np.log(df[COLS.HRLYEARN].clip(lower=1))

print(f'Loaded {len(df):,} records')
print(f'Male: {(df[gender_col]==GENDER_CODES_REVERSE["Male"]).sum():,}  Female: {(df[gender_col]==GENDER_CODES_REVERSE["Female"]).sum():,}')

# ===============================
# OAXACA-BLINDER DECOMPOSITION
# ===============================
print('\n' + '='*60)
print('OAXACA-BLINDER DECOMPOSITION')
print('='*60)

# Define control variables - use constants where available
control_vars = [COLS.EDUC, COLS.NOC_10, COLS.AGE_12, COLS.PROV, COLS.FTPTMAIN, COLS.UNION]

# Filter to only available columns
available_controls = [c for c in control_vars if c in df.columns]
print(f'Using controls: {available_controls}')

df_m = df[df[gender_col] == GENDER_CODES_REVERSE['Male']].copy()
df_f = df[df[gender_col] == GENDER_CODES_REVERSE['Female']].copy()

X_m = sm.add_constant(df_m[available_controls])
X_f = sm.add_constant(df_f[available_controls])
y_m = df_m['LOG_HRLYEARN']
y_f = df_f['LOG_HRLYEARN']

# Estimate separate regressions
beta_m = sm.OLS(y_m, X_m).fit(cov_type='HC3').params
beta_f = sm.OLS(y_f, X_f).fit(cov_type='HC3').params

# Decomposition
total_gap = y_m.mean() - y_f.mean()
explained = np.dot(X_m.mean() - X_f.mean(), beta_m)
unexplained = np.dot(X_f.mean(), beta_m - beta_f)

print(f'Male mean log wage:   {y_m.mean():.4f}')
print(f'Female mean log wage: {y_f.mean():.4f}')
print(f'Total gap:            {total_gap:.4f} ({total_gap/y_m.mean()*100:.1f}%)')
print(f'\nTWOFOLD DECOMPOSITION:')
print(f'  Explained (characteristics): {explained:.4f} ({explained/total_gap*100:.1f}%)')
print(f'  Unexplained (discrimination): {unexplained:.4f} ({unexplained/total_gap*100:.1f}%)')

# ===============================
# QUANTILE REGRESSION
# ===============================
print('\n' + '='*60)
print('QUANTILE REGRESSION - Glass Ceiling Analysis')
print('='*60)

# Build feature matrix with available columns
qr_features = ['IS_FEMALE'] + available_controls
X = sm.add_constant(df[qr_features])
y = df['LOG_HRLYEARN']

print(f'{"Quantile":>10} {"Coef":>10} {"SE":>10} {"Gap%":>10}')
print('-'*45)

qr_results = []
for q in [0.10, 0.25, 0.50, 0.75, 0.90]:
    res = QuantReg(y, X).fit(q=q)
    coef = res.params['IS_FEMALE']
    se = res.bse['IS_FEMALE']
    gap_pct = (1 - np.exp(coef)) * 100
    print(f'{q:>10.2f} {coef:>10.4f} {se:>10.4f} {gap_pct:>9.1f}%')
    qr_results.append({'quantile': q, 'coef': coef, 'se': se, 'gap_pct': gap_pct})

# Glass ceiling detection
gap_10 = qr_results[0]['gap_pct']
gap_90 = qr_results[4]['gap_pct']
if gap_90 > gap_10:
    print(f'\n** GLASS CEILING DETECTED: Gap at 90th ({gap_90:.1f}%) > 10th ({gap_10:.1f}%)')
else:
    print(f'\n** STICKY FLOOR DETECTED: Gap at 10th ({gap_10:.1f}%) > 90th ({gap_90:.1f}%)')

# ===============================
# ROBUST STANDARD ERRORS
# ===============================
print('\n' + '='*60)
print('ROBUST STANDARD ERRORS COMPARISON')
print('='*60)
print(f'{"Estimator":<15} {"Female Coef":>12} {"SE":>10} {"t-stat":>10}')
print('-'*50)

for cov in ['nonrobust', 'HC0', 'HC1', 'HC3']:
    m = sm.OLS(y, X).fit(cov_type=cov)
    print(f'{cov:<15} {m.params["IS_FEMALE"]:>12.4f} {m.bse["IS_FEMALE"]:>10.4f} {m.tvalues["IS_FEMALE"]:>10.2f}')

# ===============================
# SAVE RESULTS
# ===============================
os.makedirs('reports', exist_ok=True)

# Save quantile regression results
humanize_columns(pd.DataFrame(qr_results)).to_csv('reports/quantile_regression_results.csv', index=False)

# Save full summary
with open('reports/econometric_analysis_summary.txt', 'w') as f:
    f.write('ECONOMETRIC ANALYSIS OF GENDER PAY GAP - CANADA\n')
    f.write('='*50 + '\n\n')
    f.write('DATA SUMMARY\n')
    f.write(f'  Sample size: {len(df):,}\n')
    f.write(f'  Male: {(df[gender_col]==GENDER_CODES_REVERSE["Male"]).sum():,} ({(df[gender_col]==GENDER_CODES_REVERSE["Male"]).mean()*100:.1f}%)\n')
    f.write(f'  Female: {(df[gender_col]==GENDER_CODES_REVERSE["Female"]).sum():,} ({(df[gender_col]==GENDER_CODES_REVERSE["Female"]).mean()*100:.1f}%)\n\n')
    
    f.write('1. OAXACA-BLINDER DECOMPOSITION\n')
    f.write('-'*40 + '\n')
    f.write(f'  Total log wage gap: {total_gap:.4f}\n')
    f.write(f'  Explained (characteristics): {explained:.4f} ({explained/total_gap*100:.1f}%)\n')
    f.write(f'  Unexplained (discrimination): {unexplained:.4f} ({unexplained/total_gap*100:.1f}%)\n\n')
    
    f.write('2. QUANTILE REGRESSION\n')
    f.write('-'*40 + '\n')
    for r in qr_results:
        f.write(f'  Q{int(r["quantile"]*100):02d}: {r["gap_pct"]:.1f}% gap\n')
    f.write('\n')
    
    f.write('3. KEY FINDINGS\n')
    f.write('-'*40 + '\n')
    # Calculate raw wage gap using COLS constant
    male_wage = df[df[gender_col]==GENDER_CODES_REVERSE["Male"]][COLS.HRLYEARN].mean()
    female_wage = df[df[gender_col]==GENDER_CODES_REVERSE["Female"]][COLS.HRLYEARN].mean()
    raw_gap = (1 - female_wage / male_wage) * 100
    f.write(f'  - Raw hourly wage gap: ~{raw_gap:.1f}%\n')
    f.write(f'  - Unexplained gap after controls: {unexplained:.4f} log points\n')
    if gap_90 > gap_10:
        f.write('  - Glass ceiling effect: gap wider at top of distribution\n')
    f.write('\n')
    
    f.write('METHODOLOGY NOTES\n')
    f.write('-'*40 + '\n')
    f.write('  - Oaxaca-Blinder uses male coefficients as reference\n')
    f.write('  - HC3 robust standard errors throughout\n')
    f.write('  - Controls: education, occupation, age, province, FT/PT, union\n')

print('\nResults saved to reports/')
print('  - quantile_regression_results.csv')
print('  - econometric_analysis_summary.txt')
print('\nDone!')
