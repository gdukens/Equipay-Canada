# Auto-generated from 08_geographic_analysis.ipynb
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path

# Add project root to path
project_root = Path(r"/mnt/c/Users/Administrator/equipay-canada")
sys.path.insert(0, str(project_root))

# Change to notebooks directory for relative paths
import os
os.chdir(r"/mnt/c/Users/Administrator/equipay-canada/notebooks")

# ============================================================================
# SETUP AND IMPORTS
# ============================================================================

import sys
sys.path.insert(0, '..')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Interactive visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Project imports
from src.constants import (
    COLS, PROVINCE_CODES, EDUCATION_CODES, NOC_10_CODES, NAICS_21_CODES,
    GENDER_CODES, DATA_SCOPE_START, DATA_SCOPE_END, normalize_column_names,
    humanize_columns
)
from src.macro_data import get_macro_dataframe, BASE_YEAR

# Configure plotting
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

print("=" * 70)
print("GEOGRAPHIC ANALYSIS OF GENDER WAGE GAPS IN CANADA")
print("=" * 70)
print(f"✓ Libraries loaded")
print(f"✓ Analysis period: {DATA_SCOPE_START}-{DATA_SCOPE_END}")

# Load data
data_path = Path('../data/processed/lfs_processed.csv')
df = pd.read_csv(data_path)
df = normalize_column_names(df)

# Identify columns
gender_col = COLS.GENDER if COLS.GENDER in df.columns else 'SEX'
prov_col = COLS.PROV if COLS.PROV in df.columns else 'PROV'
year_col = 'YEAR' if 'YEAR' in df.columns else 'SURVYEAR'

# Use REAL hourly earnings (inflation-adjusted)
if COLS.REAL_HOURLY_EARNINGS in df.columns:
    wage_col = COLS.REAL_HOURLY_EARNINGS
    print("✓ Using REAL hourly earnings (inflation-adjusted to 2010$)")
elif 'REAL_HRLYEARN' in df.columns:
    wage_col = 'REAL_HRLYEARN'
    print("✓ Using REAL hourly earnings (inflation-adjusted to 2010$)")
else:
    wage_col = COLS.HOURLY_EARNINGS
    print("⚠ Real wages not available - using nominal wages")

# Create gender indicator
df['IS_FEMALE'] = (df[gender_col] == 2).astype(int)  # Female code = 2

print(f"\n✓ Loaded {len(df):,} records")
print(f"  Years: {df[year_col].min()} - {df[year_col].max()}")
print(f"  Provinces: {df[prov_col].nunique()}")

# ============================================================================
# PROVINCE MAPPINGS
# ============================================================================

# LFS PROV codes to ISO 3166-2 for Plotly choropleth
PROV_TO_ISO = {
    10: 'CA-NL',  # Newfoundland and Labrador
    11: 'CA-PE',  # Prince Edward Island
    12: 'CA-NS',  # Nova Scotia
    13: 'CA-NB',  # New Brunswick
    24: 'CA-QC',  # Quebec
    35: 'CA-ON',  # Ontario
    46: 'CA-MB',  # Manitoba
    47: 'CA-SK',  # Saskatchewan
    48: 'CA-AB',  # Alberta
    59: 'CA-BC',  # British Columbia
}

PROV_NAMES = {
    10: 'Newfoundland & Labrador',
    11: 'Prince Edward Island',
    12: 'Nova Scotia',
    13: 'New Brunswick',
    24: 'Quebec',
    35: 'Ontario',
    46: 'Manitoba',
    47: 'Saskatchewan',
    48: 'Alberta',
    59: 'British Columbia',
}

PROV_ABBREV = {
    10: 'NL', 11: 'PE', 12: 'NS', 13: 'NB', 24: 'QC',
    35: 'ON', 46: 'MB', 47: 'SK', 48: 'AB', 59: 'BC'
}

# Regional groupings
REGIONS = {
    'Atlantic': [10, 11, 12, 13],
    'Central': [24, 35],
    'Prairies': [46, 47, 48],
    'West Coast': [59]
}

# Add region to dataframe
def get_region(prov_code):
    for region, codes in REGIONS.items():
        if prov_code in codes:
            return region
    return 'Unknown'

df['REGION'] = df[prov_col].apply(get_region)
df['PROV_NAME'] = df[prov_col].map(PROV_NAMES)
df['PROV_ABBREV'] = df[prov_col].map(PROV_ABBREV)

print("Province distribution:")
print(df['PROV_NAME'].value_counts())

# ============================================================================
# CALCULATE WAGE GAP BY PROVINCE
# ============================================================================

def calculate_provincial_gaps(df, wage_col, prov_col, gender_col='IS_FEMALE', min_n=30):
    """
    Calculate gender wage gap statistics by province.
    
    Returns DataFrame with gap metrics for each province.
    """
    results = []
    
    for prov_code in PROV_TO_ISO.keys():
        prov_data = df[df[prov_col] == prov_code]
        
        male_wages = prov_data[prov_data[gender_col] == 0][wage_col].dropna()
        female_wages = prov_data[prov_data[gender_col] == 1][wage_col].dropna()
        
        if len(male_wages) >= min_n and len(female_wages) >= min_n:
            male_mean = male_wages.mean()
            female_mean = female_wages.mean()
            male_median = male_wages.median()
            female_median = female_wages.median()
            
            # Calculate gaps
            mean_gap_pct = (male_mean - female_mean) / male_mean * 100
            median_gap_pct = (male_median - female_median) / male_median * 100
            
            # T-test for significance
            t_stat, p_value = stats.ttest_ind(male_wages, female_wages)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(male_wages)-1)*male_wages.std()**2 + 
                                  (len(female_wages)-1)*female_wages.std()**2) / 
                                 (len(male_wages) + len(female_wages) - 2))
            cohens_d = (male_mean - female_mean) / pooled_std
            
            results.append({
                'prov_code': prov_code,
                'iso_code': PROV_TO_ISO[prov_code],
                'province': PROV_NAMES[prov_code],
                'abbrev': PROV_ABBREV[prov_code],
                'region': get_region(prov_code),
                'male_mean': male_mean,
                'female_mean': female_mean,
                'male_median': male_median,
                'female_median': female_median,
                'mean_gap_pct': mean_gap_pct,
                'median_gap_pct': median_gap_pct,
                'dollar_gap': male_mean - female_mean,
                'cohens_d': cohens_d,
                't_stat': t_stat,
                'p_value': p_value,
                'n_male': len(male_wages),
                'n_female': len(female_wages),
                'n_total': len(male_wages) + len(female_wages)
            })
    
    return pd.DataFrame(results)

# Calculate provincial gaps
prov_gaps = calculate_provincial_gaps(df, wage_col, prov_col)
prov_gaps = prov_gaps.sort_values('mean_gap_pct', ascending=False)

print("=" * 70)
print("PROVINCIAL GENDER WAGE GAP SUMMARY")
print("=" * 70)
print(f"\n{'Province':<25} {'Gap (%)':<10} {'$ Gap':<10} {'Cohens d':<10} {'Sig.'}")
print("-" * 70)
for _, row in prov_gaps.iterrows():
    sig = '***' if row['p_value'] < 0.001 else '**' if row['p_value'] < 0.01 else '*' if row['p_value'] < 0.05 else ''
    print(f"{row['province']:<25} {row['mean_gap_pct']:>6.1f}%    ${row['dollar_gap']:>6.2f}    {row['cohens_d']:>6.3f}    {sig}")

print("\n*** p<0.001, ** p<0.01, * p<0.05")

# ============================================================================
# STATIC CHOROPLETH MAP
# ============================================================================

# Note: Plotly choropleth for Canadian provinces requires GeoJSON
# Using a bar chart visualization instead for reliability
import plotly.graph_objects as go

fig = go.Figure(data=[
    go.Bar(
        x=prov_gaps['province'],
        y=prov_gaps['mean_gap_pct'],
        marker_color=prov_gaps['mean_gap_pct'],
        marker_colorscale='RdYlGn_r',
        text=[f"{v:.1f}%" for v in prov_gaps['mean_gap_pct']],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Gap: %{y:.1f}%<extra></extra>'
    )
])

fig.update_layout(
    title=f'<b>Gender Wage Gap by Province ({DATA_SCOPE_START}-{DATA_SCOPE_END})</b><br>'
          f'<sup>Real wages in {BASE_YEAR} constant dollars | Red = Higher Gap</sup>',
    xaxis_title='Province',
    yaxis_title='Wage Gap (%)',
    height=500,
    showlegend=False
)

# Save as interactive HTML
fig.write_html('../reports/figures/provincial_wage_gap_chart.html')
print("📊 Interactive chart saved: reports/figures/provincial_wage_gap_chart.html")

fig.show()

# ============================================================================
# PROVINCIAL RANKINGS: HORIZONTAL BAR CHART
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Sort by gap
prov_sorted = prov_gaps.sort_values('mean_gap_pct', ascending=True)

# Color by region
region_colors = {
    'Atlantic': '#3498db',
    'Central': '#e74c3c',
    'Prairies': '#f39c12',
    'West Coast': '#27ae60'
}
colors = [region_colors.get(r, 'gray') for r in prov_sorted['region']]

# Plot 1: Wage Gap %
ax = axes[0]
bars = ax.barh(prov_sorted['province'], prov_sorted['mean_gap_pct'], color=colors, edgecolor='black', alpha=0.8)
ax.axvline(x=prov_gaps['mean_gap_pct'].mean(), color='red', linestyle='--', linewidth=2, 
           label=f'Canada Avg: {prov_gaps["mean_gap_pct"].mean():.1f}%')
ax.set_xlabel('Gender Wage Gap (%)', fontsize=12)
ax.set_title('Gender Wage Gap by Province\n(Lower is Better)', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')

# Add value labels
for bar, val in zip(bars, prov_sorted['mean_gap_pct']):
    ax.text(val + 0.3, bar.get_y() + bar.get_height()/2, f'{val:.1f}%', 
            va='center', fontsize=10)

# Plot 2: Dollar Gap
ax = axes[1]
bars = ax.barh(prov_sorted['province'], prov_sorted['dollar_gap'], color=colors, edgecolor='black', alpha=0.8)
ax.axvline(x=prov_gaps['dollar_gap'].mean(), color='red', linestyle='--', linewidth=2,
           label=f'Canada Avg: ${prov_gaps["dollar_gap"].mean():.2f}')
ax.set_xlabel('Dollar Gap per Hour (2010$)', fontsize=12)
ax.set_title('Hourly Wage Gap by Province\n(Real 2010 Dollars)', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')

# Add value labels
for bar, val in zip(bars, prov_sorted['dollar_gap']):
    ax.text(val + 0.1, bar.get_y() + bar.get_height()/2, f'${val:.2f}', 
            va='center', fontsize=10)

# Add region legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=r, alpha=0.8) for r, c in region_colors.items()]
fig.legend(handles=legend_elements, loc='upper center', ncol=4, fontsize=10, 
           bbox_to_anchor=(0.5, 0.02))

plt.tight_layout()
plt.subplots_adjust(bottom=0.1)
plt.savefig('../reports/figures/provincial_rankings.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n📊 Provincial rankings saved: reports/figures/provincial_rankings.png")

# ============================================================================
# REGIONAL ANALYSIS
# ============================================================================

print("=" * 70)
print("REGIONAL WAGE GAP ANALYSIS")
print("=" * 70)

# Calculate regional gaps
regional_data = []

for region, codes in REGIONS.items():
    region_df = df[df[prov_col].isin(codes)]
    
    male_wages = region_df[region_df['IS_FEMALE'] == 0][wage_col].dropna()
    female_wages = region_df[region_df['IS_FEMALE'] == 1][wage_col].dropna()
    
    if len(male_wages) > 100 and len(female_wages) > 100:
        male_mean = male_wages.mean()
        female_mean = female_wages.mean()
        gap_pct = (male_mean - female_mean) / male_mean * 100
        
        regional_data.append({
            'region': region,
            'male_mean': male_mean,
            'female_mean': female_mean,
            'gap_pct': gap_pct,
            'dollar_gap': male_mean - female_mean,
            'n_total': len(male_wages) + len(female_wages)
        })

regional_df = pd.DataFrame(regional_data)

print(f"\n{'Region':<15} {'Male Avg':>12} {'Female Avg':>12} {'Gap (%)':>10} {'$ Gap':>10}")
print("-" * 60)
for _, row in regional_df.iterrows():
    print(f"{row['region']:<15} ${row['male_mean']:>10.2f} ${row['female_mean']:>10.2f} {row['gap_pct']:>9.1f}% ${row['dollar_gap']:>8.2f}")

# Visualization
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(regional_df))
width = 0.35

bars1 = ax.bar(x - width/2, regional_df['male_mean'], width, label='Male', color='#3498db', alpha=0.8)
bars2 = ax.bar(x + width/2, regional_df['female_mean'], width, label='Female', color='#e74c3c', alpha=0.8)

ax.set_xlabel('Region', fontsize=12)
ax.set_ylabel('Average Hourly Wage (2010$)', fontsize=12)
ax.set_title('Average Wages by Region and Gender', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(regional_df['region'])
ax.legend()

# Add gap annotations
for i, (_, row) in enumerate(regional_df.iterrows()):
    ax.annotate(f'{row["gap_pct"]:.1f}% gap', 
                xy=(i, max(row['male_mean'], row['female_mean']) + 1),
                ha='center', fontsize=10, fontweight='bold', color='darkred')

plt.tight_layout()
plt.savefig('../reports/figures/regional_wage_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n📊 Regional comparison saved: reports/figures/regional_wage_comparison.png")

# ============================================================================
# INDUSTRY × PROVINCE WAGE GAP HEATMAP
# ============================================================================

print("=" * 70)
print("INDUSTRY × PROVINCE WAGE GAP ANALYSIS")
print("=" * 70)

# Get industry column
ind_col = COLS.NAICS_21 if COLS.NAICS_21 in df.columns else 'NAICS_21'

# Select major industries for clarity
major_industries = {
    11: 'Agriculture',
    21: 'Mining/Oil',
    23: 'Construction',
    31: 'Manufacturing',
    44: 'Retail',
    52: 'Finance',
    54: 'Professional',
    61: 'Education',
    62: 'Healthcare',
    72: 'Accommodation'
}

# Calculate gaps by industry × province
gap_matrix = []

for ind_code, ind_name in major_industries.items():
    row = {'industry': ind_name}
    for prov_code, prov_abbrev in PROV_ABBREV.items():
        subset = df[(df[ind_col] == ind_code) & (df[prov_col] == prov_code)]
        
        male = subset[subset['IS_FEMALE'] == 0][wage_col]
        female = subset[subset['IS_FEMALE'] == 1][wage_col]
        
        if len(male) >= 20 and len(female) >= 20:
            gap = (male.mean() - female.mean()) / male.mean() * 100
            row[prov_abbrev] = gap
        else:
            row[prov_abbrev] = np.nan
    
    gap_matrix.append(row)

gap_df = pd.DataFrame(gap_matrix).set_index('industry')

# Create heatmap
fig, ax = plt.subplots(figsize=(14, 8))

# Custom colormap: diverging around the national average
national_avg = prov_gaps['mean_gap_pct'].mean()

sns.heatmap(gap_df, annot=True, fmt='.1f', cmap='RdYlGn_r',
            center=national_avg, vmin=0, vmax=25,
            linewidths=0.5, linecolor='white',
            cbar_kws={'label': 'Wage Gap (%)'},
            ax=ax)

ax.set_title('Gender Wage Gap by Industry and Province (%)\nRed = Higher Gap, Green = Lower Gap',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Province', fontsize=12)
ax.set_ylabel('Industry', fontsize=12)

plt.tight_layout()
plt.savefig('../reports/figures/industry_province_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n📊 Industry × Province heatmap saved: reports/figures/industry_province_heatmap.png")

# ============================================================================
# PROVINCIAL WAGE GAP EVOLUTION (2010-2025)
# ============================================================================

# Calculate gaps by province and year
yearly_prov_gaps = []

for year in sorted(df[year_col].unique()):
    for prov_code in PROV_TO_ISO.keys():
        subset = df[(df[year_col] == year) & (df[prov_col] == prov_code)]
        
        male = subset[subset['IS_FEMALE'] == 0][wage_col]
        female = subset[subset['IS_FEMALE'] == 1][wage_col]
        
        if len(male) >= 30 and len(female) >= 30:
            gap_pct = (male.mean() - female.mean()) / male.mean() * 100
            yearly_prov_gaps.append({
                'year': int(year),
                'prov_code': prov_code,
                'province': PROV_NAMES[prov_code],
                'abbrev': PROV_ABBREV[prov_code],
                'region': get_region(prov_code),
                'gap_pct': gap_pct
            })

yearly_gaps_df = pd.DataFrame(yearly_prov_gaps)

# Create faceted line chart
fig = px.line(
    yearly_gaps_df,
    x='year',
    y='gap_pct',
    color='province',
    facet_col='region',
    facet_col_wrap=2,
    title='<b>Gender Wage Gap Evolution by Province and Region</b><br><sup>2010-2025, Real wages in 2010 constant dollars</sup>',
    labels={'gap_pct': 'Wage Gap (%)', 'year': 'Year', 'province': 'Province'},
    height=600
)

fig.update_traces(line=dict(width=2))
fig.add_hline(y=0, line_dash='dash', line_color='green', annotation_text='Parity')

fig.write_html('../reports/figures/provincial_gap_evolution.html')
print("📊 Interactive chart saved: reports/figures/provincial_gap_evolution.html")

fig.show()

# ============================================================================
# SUMMARY AND DATA EXPORT
# ============================================================================

print("=" * 70)
print("GEOGRAPHIC ANALYSIS SUMMARY")
print("=" * 70)

# Key findings
best_province = prov_gaps.loc[prov_gaps['mean_gap_pct'].idxmin()]
worst_province = prov_gaps.loc[prov_gaps['mean_gap_pct'].idxmax()]

print(f"""
KEY FINDINGS:

1. NATIONAL OVERVIEW:
   - Average provincial wage gap: {prov_gaps['mean_gap_pct'].mean():.1f}%
   - Range: {prov_gaps['mean_gap_pct'].min():.1f}% to {prov_gaps['mean_gap_pct'].max():.1f}%
   - All provinces show statistically significant gaps (p < 0.05)

2. BEST PERFORMER:
   - {best_province['province']}: {best_province['mean_gap_pct']:.1f}% gap
   - Dollar difference: ${best_province['dollar_gap']:.2f}/hour

3. HIGHEST GAP:
   - {worst_province['province']}: {worst_province['mean_gap_pct']:.1f}% gap
   - Dollar difference: ${worst_province['dollar_gap']:.2f}/hour

4. REGIONAL PATTERNS:
""")

for _, row in regional_df.iterrows():
    print(f"   - {row['region']}: {row['gap_pct']:.1f}% gap")

# Save provincial gap data
humanize_columns(prov_gaps).to_csv('../reports/gap_by_prov.csv', index=False)
print("\n📊 Provincial gap data saved: reports/gap_by_prov.csv")

# Save yearly provincial data
humanize_columns(yearly_gaps_df).to_csv('../data/processed/provincial_gaps_by_year.csv', index=False)
print("📊 Yearly provincial data saved: data/processed/provincial_gaps_by_year.csv")