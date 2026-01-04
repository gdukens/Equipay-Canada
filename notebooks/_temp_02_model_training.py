# Auto-generated from 02_model_training.ipynb
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
# SETUP: Import Libraries and Configure Environment
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Statistical tests
from scipy import stats as sp_stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Add project root
sys.path.insert(0, str(Path.cwd().parent))

from src.constants import COLS, normalize_column_names, DATA_SCOPE_START, DATA_SCOPE_END

# NEW: Weighted ML utilities for survey data
from src.ml_utils import WeightedMLSplitter, WeightedMetrics, WeightedGapAnalysis

# Publication-quality figures
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white'
})
plt.style.use('seaborn-v0_8-whitegrid')

# Ensure figures directory exists
Path('../reports/figures').mkdir(parents=True, exist_ok=True)

print("✓ Libraries loaded successfully")
print("✓ Weighted ML utilities loaded (WeightedMLSplitter, WeightedMetrics)")
print(f"✓ Analysis period: {DATA_SCOPE_START}-{DATA_SCOPE_END}")

# ============================================================================
# DATA LOADING VIA EquiPayDataStore (DuckDB + Parquet)
# ============================================================================
# Memory-efficient: queries 19.5M rows using ~300MB RAM
# Uses Parquet files directly - no need to generate CSV first

from src.data_store import EquiPayDataStore
from pathlib import Path

print("🚀 Loading data via EquiPayDataStore")
print("=" * 60)

# Initialize data store
PROJECT_ROOT = Path.cwd().parent
store = EquiPayDataStore(
    parquet_path=str(PROJECT_ROOT / "data" / "parquet"),
    raw_csv_path=str(PROJECT_ROOT / "data" / "raw" / "lfs")
)

# Get summary stats
stats = store.get_summary_stats()
print(f"✓ Data source: {stats['source']}")
print(f"✓ Total records: {stats['total_records']:,}")
print(f"✓ Valid wage records: {stats['valid_wage_records']:,}")
print(f"✓ Year range: {stats['year_range'][0]} - {stats['year_range'][1]}")

# Load full dataset for ML (with valid wages only)
df = store.get_wages(valid_wages_only=True)
df = normalize_column_names(df)

print(f"\n✓ Loaded {len(df):,} records for modeling")

# Identify columns
gender_col = COLS.GENDER if COLS.GENDER in df.columns else 'SEX'
weight_col = 'FINALWT'  # MANDATORY: Survey weights for population inference

# Validate weight column exists
if weight_col not in df.columns:
    raise ValueError(f"Weight column '{weight_col}' not found! Survey weights are MANDATORY.")

# Use REAL hourly earnings for cross-year model training (inflation-adjusted)
# This ensures predictions are in 2010 constant dollars
if COLS.REAL_HOURLY_EARNINGS in df.columns:
    wage_col = COLS.REAL_HOURLY_EARNINGS
    print("✓ Using REAL hourly earnings (inflation-adjusted to 2010$)")
elif 'REAL_HRLYEARN' in df.columns:
    wage_col = 'REAL_HRLYEARN'
    print("✓ Using REAL hourly earnings (inflation-adjusted to 2010$)")
else:
    wage_col = COLS.HOURLY_EARNINGS if COLS.HOURLY_EARNINGS in df.columns else 'HRLYEARN'
    print("⚠ Real wages not available - using nominal wages")

print(f"Gender column: {gender_col}")
print(f"Wage column: {wage_col}")
print(f"Weight column: {weight_col} ✓ (MANDATORY for population inference)")

# Create IS_FEMALE (Female code = 2)
df['IS_FEMALE'] = (df[gender_col] == 2).astype(int)

# Show weight statistics
print(f"\nSurvey weight statistics:")
print(f"  Total weighted population: {df[weight_col].sum():,.0f}")
print(f"  Mean weight: {df[weight_col].mean():.1f}")
print(f"  Weight range: [{df[weight_col].min():.1f}, {df[weight_col].max():.1f}]")

# Define features
potential_features = [
    'IS_FEMALE', COLS.EDUC, COLS.NOC_10, COLS.PROV, 
    COLS.AGE_12, COLS.TENURE, COLS.FTPTMAIN, COLS.UNION,
    'EXPERIENCE', 'EXPERIENCE_SQ'
]

# Filter to available features
features = [f for f in potential_features if f in df.columns]
print(f"Available features: {features}")

# Target
target = wage_col

# Remove rows with missing target
df_model = df[features + [target]].dropna()
print(f"\nRows for modeling: {len(df_model):,}")

# Prepare X and y
X = df_model[features]
y = df_model[target]

# Log transform target for better distribution
y_log = np.log(y.clip(lower=1))

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"\nTarget statistics:")
print(f"  Original - Mean: ${y.mean():.2f}, Std: ${y.std():.2f}")
print(f"  Log - Mean: {y_log.mean():.3f}, Std: {y_log.std():.3f}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)

print(f"Training set: {len(X_train):,} samples")
print(f"Test set: {len(X_test):,} samples")

# Variance Inflation Factor (VIF) Analysis
print("=" * 60)
print("MULTICOLLINEARITY DIAGNOSTICS: VARIANCE INFLATION FACTOR")
print("=" * 60)

# Calculate VIF for each feature
X_vif = X.dropna()
vif_data = pd.DataFrame()
vif_data["Feature"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
vif_data = vif_data.sort_values("VIF", ascending=False)

print(f"\n{'Feature':<20} {'VIF':>10} {'Status':>15}")
print("-" * 50)
for _, row in vif_data.iterrows():
    if row['VIF'] > 10:
        status = "⚠️ SEVERE"
    elif row['VIF'] > 5:
        status = "⚠ MODERATE"
    else:
        status = "✓ OK"
    print(f"{row['Feature']:<20} {row['VIF']:>10.2f} {status:>15}")

# Summary
high_vif = vif_data[vif_data['VIF'] > 10]
if len(high_vif) > 0:
    print(f"\n⚠️ WARNING: {len(high_vif)} feature(s) with VIF > 10")
    print("   Consider removing or combining these features.")
else:
    print("\n✓ No severe multicollinearity detected (all VIF < 10)")

# ============================================================================
# ECONOMETRIC FEATURE ENGINEERING
# ============================================================================
# Import the theory-driven feature engineering module

from src.econometric_features import (
    EconometricFeatureEngineer, 
    FeatureConfig, 
    FeatureComplexity
)

print("=" * 70)
print("THEORY-DRIVEN ECONOMETRIC FEATURE ENGINEERING")
print("=" * 70)

# Configure feature engineering with anti-overfitting safeguards
config = FeatureConfig(
    complexity=FeatureComplexity.STANDARD,  # Balanced: ~49 features
    max_vif=10.0,                           # Remove features with VIF > 10
    min_variance=0.01,                      # Remove near-constant features
    max_pairwise_correlation=0.95,          # Remove highly correlated pairs
    min_correlation_with_target=0.01,       # Require predictive signal
    scale_features=True,
    compute_selection_features=True,
    compute_temporal_features=False,        # Disable for cross-sectional
    random_state=42
)

print(f"\nComplexity Level: {config.complexity.name}")
print(f"Anti-Overfitting Thresholds:")
print(f"  Max VIF: {config.max_vif}")
print(f"  Min Variance: {config.min_variance}")
print(f"  Max Pairwise Correlation: {config.max_pairwise_correlation}")
print(f"  Min Target Correlation: {config.min_correlation_with_target}")

# Create the feature engineer
# We use a sample for fitting to avoid memory issues with large datasets

# Prepare sample for feature engineering
n_sample = min(50_000, len(df_model))
df_sample = df_model.sample(n=n_sample, random_state=42).copy()

# Ensure required columns exist
df_sample['IS_FEMALE'] = (df_sample['IS_FEMALE'] if 'IS_FEMALE' in df_sample.columns 
                          else (df_sample[gender_col] == 2).astype(int))
df_sample['LOG_WAGE'] = y_log.loc[df_sample.index]

print(f"Using {n_sample:,} records for feature engineering")
print(f"Female ratio: {df_sample['IS_FEMALE'].mean():.1%}")

# Initialize engineer
engineer = EconometricFeatureEngineer(
    config=config,
    gender_col='IS_FEMALE',
    wage_col='LOG_WAGE',
    female_code=1  # IS_FEMALE uses 1 for female
)

# Fit and transform
print("\nFitting econometric feature engineer...")
df_enhanced = engineer.fit_transform(df_sample)

# Show created features by category
print(f"\n✓ Created {len(engineer.feature_metadata_)} theory-driven features")
print("\nFeatures by Category:")
print("-" * 50)
categories = {}
for name, meta in engineer.feature_metadata_.items():
    cat = meta.category
    if cat not in categories:
        categories[cat] = []
    categories[cat].append(name)

for cat, feats in sorted(categories.items()):
    print(f"  {cat}: {len(feats)} features")

# Apply automatic feature selection with anti-overfitting safeguards
print("=" * 70)
print("AUTOMATIC FEATURE SELECTION (ANTI-OVERFITTING)")
print("=" * 70)

# Get target variable
y_sample = df_enhanced['LOG_WAGE']

# Run feature selection
selected_features = engineer.select_features(df_enhanced, y_sample)

print(f"\n✓ Selected {len(selected_features)} features after filtering")
print(f"  Removed {len(engineer.feature_metadata_) - len(selected_features)} features")

# Show removal reasons
if engineer.removed_features_:
    print("\nRemoval Reasons Summary:")
    reasons = {}
    for feat, reason in engineer.removed_features_.items():
        if reason not in reasons:
            reasons[reason] = 0
        reasons[reason] += 1
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count} features")

# Display selected features
print("\nSelected Features:")
print("-" * 50)
for i, feat in enumerate(selected_features[:20], 1):
    if feat in engineer.feature_metadata_:
        meta = engineer.feature_metadata_[feat]
        print(f"  {i:2d}. {feat:<35} [{meta.category}]")
    else:
        print(f"  {i:2d}. {feat:<35}")
if len(selected_features) > 20:
    print(f"  ... and {len(selected_features) - 20} more")

# Generate detailed feature report
report = engineer.get_feature_report()

# Display as DataFrame for better visualization
report_df = pd.DataFrame([
    {
        'Feature': name,
        'Category': info.get('category', 'unknown'),
        'Theory': info.get('theory_source', 'N/A'),
        'Selected': info.get('selected', False),
        'Reason': info.get('removal_reason', '-') if not info.get('selected') else '-'
    }
    for name, info in report.items()
])

# Show selected features sorted by category
selected_df = report_df[report_df['Selected'] == True].sort_values('Category')
print(f"Selected Features Report ({len(selected_df)} features):")
print(selected_df[['Feature', 'Category', 'Theory']].to_string(index=False))

# Create enhanced feature matrix for model training
# Combine original features with selected econometric features

# Get numeric econometric features only
econometric_cols = [col for col in selected_features if col in df_enhanced.columns]
X_econometric = df_enhanced[econometric_cols].copy()

# Align with original data indices
df_model_enhanced = df_model.loc[df_sample.index].copy()

# Add econometric features
for col in econometric_cols:
    if col not in df_model_enhanced.columns:
        df_model_enhanced[col] = X_econometric[col].values

# Create combined feature list
all_features = features.copy()  # Original features
for feat in econometric_cols:
    if feat not in all_features and feat != 'LOG_WAGE' and feat not in ['IS_FEMALE', gender_col]:
        all_features.append(feat)

print(f"Original features: {len(features)}")
print(f"Econometric features added: {len(econometric_cols)}")
print(f"Total unique features: {len(all_features)}")

# Prepare enhanced X and y
X_enhanced = df_model_enhanced[all_features].fillna(0)
y_enhanced = df_model_enhanced[target].apply(lambda x: np.log(max(x, 1)))

print(f"\nEnhanced dataset shape: {X_enhanced.shape}")
print(f"Ready for model training with theory-driven features!")

# Compare models with basic vs enhanced features
from sklearn.model_selection import cross_val_score

print("=" * 70)
print("MODEL COMPARISON: BASIC vs ENHANCED FEATURES")
print("=" * 70)

# Prepare data for comparison
X_basic = df_model_enhanced[features].fillna(0)
X_enhanced_clean = X_enhanced.fillna(0)
y_comparison = y_enhanced

# Scale both
scaler_basic = StandardScaler()
scaler_enhanced = StandardScaler()

X_basic_scaled = scaler_basic.fit_transform(X_basic)
X_enhanced_scaled = scaler_enhanced.fit_transform(X_enhanced_clean)

# Cross-validation comparison
print("\n5-Fold Cross-Validation R² Scores:")
print("-" * 50)
print(f"{'Model':<25} {'Basic Features':>15} {'Enhanced':>15} {'Improvement':>12}")
print("-" * 50)

models_cv = {
    'Linear Regression': LinearRegression(),
    'Ridge (α=1.0)': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1),
}

results = {}
for name, model in models_cv.items():
    # Basic features
    cv_basic = cross_val_score(model, X_basic_scaled, y_comparison, cv=5, scoring='r2')
    r2_basic = cv_basic.mean()
    
    # Enhanced features
    cv_enhanced = cross_val_score(model, X_enhanced_scaled, y_comparison, cv=5, scoring='r2')
    r2_enhanced = cv_enhanced.mean()
    
    improvement = (r2_enhanced - r2_basic) / abs(r2_basic) * 100 if r2_basic != 0 else 0
    
    print(f"{name:<25} {r2_basic:>15.4f} {r2_enhanced:>15.4f} {improvement:>+11.1f}%")
    results[name] = {'basic': r2_basic, 'enhanced': r2_enhanced, 'improvement': improvement}

print("\n✓ Theory-driven features typically improve model performance")
print("  while maintaining interpretability and avoiding overfitting.")

# ============================================================================
# WEIGHTED TRAIN/TEST/VALIDATION SPLIT
# ============================================================================
# Survey weights (FINALWT) are MANDATORY for population-level inference

print("=" * 70)
print("WEIGHTED DATA SPLITTING FOR ML TRAINING")
print("=" * 70)

# Prepare data for splitting - need DataFrame with features, target, and weights
df_for_split = df_model_enhanced.copy()
df_for_split['target'] = y_enhanced
df_for_split['weights'] = df.loc[df_for_split.index, weight_col]

# Use WeightedMLSplitter for proper stratified splitting
splitter = WeightedMLSplitter(
    df=df_for_split,
    target_col='target',
    weight_col='weights',
    feature_cols=all_features,
    stratify_cols=[gender_col],  # Stratify by gender
    random_state=42
)

# Create splits (70% train, 10% val, 20% test)
splits = splitter.create_splits(test_size=0.2, val_size=0.1)

# Extract data
X_train = splits['train']['X']
y_train = splits['train']['y']
w_train = splits['train']['weights']

X_val = splits['val']['X']
y_val = splits['val']['y']
w_val = splits['val']['weights']

X_test = splits['test']['X']
y_test = splits['test']['y']
w_test = splits['test']['weights']

feature_names = splits['feature_names']

print(f"\n✓ Split complete with survey weights preserved")
print(f"\nSplit Statistics:")
print(f"{'Split':<10} {'Samples':>12} {'Weighted Pop':>15} {'% of Pop':>10}")
print("-" * 50)
total_weight = w_train.sum() + w_val.sum() + w_test.sum()
for name, w in [('Train', w_train), ('Val', w_val), ('Test', w_test)]:
    print(f"{name:<10} {len(w):>12,} {w.sum():>15,.0f} {w.sum()/total_weight*100:>9.1f}%")

# ============================================================================
# BASELINE MODEL: OLS WITH SURVEY WEIGHTS
# ============================================================================

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Linear Regression with sample weights
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train, sample_weight=w_train)

# Predictions on test set
y_pred_lr = lr_model.predict(X_test_scaled)

# WEIGHTED Metrics (population-level inference)
print("LINEAR REGRESSION RESULTS (WEIGHTED)")
print("=" * 50)
metrics_lr = WeightedMetrics.evaluate(y_test, y_pred_lr, w_test)

print(f"Weighted R² Score: {metrics_lr['weighted_r2']:.4f}")
print(f"Weighted RMSE (log): {metrics_lr['weighted_rmse']:.4f}")
print(f"Weighted MAE (log): {metrics_lr['weighted_mae']:.4f}")

# Compare to unweighted
print(f"\nUnweighted R² Score: {metrics_lr['unweighted_r2']:.4f}")
print(f"Difference: {(metrics_lr['weighted_r2'] - metrics_lr['unweighted_r2'])*100:+.2f} pp")

# Convert back to dollars for interpretation
y_test_dollars = np.exp(y_test)
y_pred_dollars = np.exp(y_pred_lr)
dollar_rmse = WeightedMetrics.weighted_rmse(y_test_dollars, y_pred_dollars, w_test)
dollar_mae = WeightedMetrics.weighted_mae(y_test_dollars, y_pred_dollars, w_test)

print(f"\nIn dollars (weighted):")
print(f"RMSE: ${dollar_rmse:.2f}")
print(f"MAE: ${dollar_mae:.2f}")

# Feature coefficients
coef_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': lr_model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

print("\nFeature Coefficients (Linear Regression):")
print("-" * 50)
for _, row in coef_df.iterrows():
    pct_effect = (np.exp(row['Coefficient']) - 1) * 100
    print(f"{row['Feature']:<20} {row['Coefficient']:>10.4f}  ({pct_effect:>+6.1f}%)")

# Gender effect
gender_coef = coef_df[coef_df['Feature'] == 'IS_FEMALE']['Coefficient'].values[0]
gender_effect = (np.exp(gender_coef) - 1) * 100
print(f"\n*** Gender Effect: {gender_effect:.1f}% (negative = women earn less) ***")

# Residual Diagnostics for OLS
residuals = y_test - y_pred_lr

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Residuals vs Fitted
ax = axes[0, 0]
ax.scatter(y_pred_lr, residuals, alpha=0.3, s=10)
ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Fitted Values (log wage)')
ax.set_ylabel('Residuals')
ax.set_title('Residuals vs Fitted', fontweight='bold')

# 2. Q-Q Plot of Residuals
ax = axes[0, 1]
sp_stats.probplot(residuals, dist="norm", plot=ax)
ax.set_title('Q-Q Plot of Residuals', fontweight='bold')

# 3. Histogram of Residuals
ax = axes[1, 0]
ax.hist(residuals, bins=50, edgecolor='white', alpha=0.7, density=True)
xmin, xmax = ax.get_xlim()
x = np.linspace(xmin, xmax, 100)
p = sp_stats.norm.pdf(x, residuals.mean(), residuals.std())
ax.plot(x, p, 'r-', linewidth=2, label='Normal Distribution')
ax.set_xlabel('Residuals')
ax.set_ylabel('Density')
ax.set_title('Distribution of Residuals', fontweight='bold')
ax.legend()

# 4. Scale-Location Plot
ax = axes[1, 1]
ax.scatter(y_pred_lr, np.sqrt(np.abs(residuals)), alpha=0.3, s=10)
ax.set_xlabel('Fitted Values')
ax.set_ylabel('√|Residuals|')
ax.set_title('Scale-Location (Homoscedasticity Check)', fontweight='bold')

plt.tight_layout()
plt.savefig('../reports/figures/ols_residual_diagnostics.png', dpi=150, 
            bbox_inches='tight', facecolor='white')
plt.show()

# Formal tests
print("\n" + "=" * 60)
print("RESIDUAL DIAGNOSTIC TESTS")
print("=" * 60)

# Jarque-Bera test for normality
jb_stat, jb_pval = sp_stats.jarque_bera(residuals)
print(f"\nJarque-Bera Test (Normality):")
print(f"  Statistic: {jb_stat:.2f}")
print(f"  p-value: {jb_pval:.4e}")
print(f"  Conclusion: {'Residuals NOT normal' if jb_pval < 0.05 else 'Residuals normal'}")

# Durbin-Watson (approximate using lag-1 autocorrelation)
from statsmodels.stats.stattools import durbin_watson
dw_stat = durbin_watson(residuals)
print(f"\nDurbin-Watson Test (Autocorrelation):")
print(f"  Statistic: {dw_stat:.4f}")
print(f"  Interpretation: ~2 = no autocorrelation, <1.5 = positive, >2.5 = negative")

print("\n📊 Figure saved: reports/figures/ols_residual_diagnostics.png")

# Ridge Regression
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_train)
y_pred_ridge = ridge_model.predict(X_test_scaled)

print("RIDGE REGRESSION RESULTS")
print("=" * 40)
print(f"R² Score: {r2_score(y_test, y_pred_ridge):.4f}")
print(f"RMSE (log): {np.sqrt(mean_squared_error(y_test, y_pred_ridge)):.4f}")

# Lasso Regression
lasso_model = Lasso(alpha=0.001)
lasso_model.fit(X_train_scaled, y_train)
y_pred_lasso = lasso_model.predict(X_test_scaled)

print("LASSO REGRESSION RESULTS")
print("=" * 40)
print(f"R² Score: {r2_score(y_test, y_pred_lasso):.4f}")
print(f"RMSE (log): {np.sqrt(mean_squared_error(y_test, y_pred_lasso)):.4f}")

# Show selected features (non-zero coefficients)
selected = [(f, c) for f, c in zip(features, lasso_model.coef_) if abs(c) > 0.001]
print(f"\nSelected features: {len(selected)}/{len(features)}")

# ============================================================================
# RANDOM FOREST WITH SURVEY WEIGHTS
# ============================================================================

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=12,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)
# Train with sample weights
rf_model.fit(X_train, y_train, sample_weight=w_train)
y_pred_rf = rf_model.predict(X_test)

print("RANDOM FOREST RESULTS (WEIGHTED)")
print("=" * 50)
metrics_rf = WeightedMetrics.evaluate(y_test, y_pred_rf, w_test)
print(f"Weighted R² Score: {metrics_rf['weighted_r2']:.4f}")
print(f"Weighted RMSE (log): {metrics_rf['weighted_rmse']:.4f}")

# Feature importance
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#e74c3c' if f == 'IS_FEMALE' else '#3498db' for f in importance_df['Feature']]
ax.barh(importance_df['Feature'], importance_df['Importance'], color=colors)
ax.set_xlabel('Feature Importance')
ax.set_title('Random Forest Feature Importance\n(Red = Gender)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# ============================================================================
# GRADIENT BOOSTING WITH SURVEY WEIGHTS
# ============================================================================

gb_model = GradientBoostingRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
# GradientBoostingRegressor supports sample_weight
gb_model.fit(X_train, y_train, sample_weight=w_train)
y_pred_gb = gb_model.predict(X_test)

print("GRADIENT BOOSTING RESULTS (WEIGHTED)")
print("=" * 50)
metrics_gb = WeightedMetrics.evaluate(y_test, y_pred_gb, w_test)
print(f"Weighted R² Score: {metrics_gb['weighted_r2']:.4f}")
print(f"Weighted RMSE (log): {metrics_gb['weighted_rmse']:.4f}")

# ============================================================================
# XGBOOST WITH SURVEY WEIGHTS
# ============================================================================

try:
    import xgboost as xgb
    
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    # XGBoost fully supports sample_weight
    xgb_model.fit(X_train, y_train, sample_weight=w_train)
    y_pred_xgb = xgb_model.predict(X_test)
    
    print("XGBOOST RESULTS (WEIGHTED)")
    print("=" * 50)
    metrics_xgb = WeightedMetrics.evaluate(y_test, y_pred_xgb, w_test)
    print(f"Weighted R² Score: {metrics_xgb['weighted_r2']:.4f}")
    print(f"Weighted RMSE (log): {metrics_xgb['weighted_rmse']:.4f}")
    HAS_XGB = True
except ImportError:
    print("XGBoost not installed. Skipping...")
    HAS_XGB = False
    y_pred_xgb = y_pred_gb  # fallback
    metrics_xgb = metrics_gb

# ============================================================================
# MODEL COMPARISON WITH WEIGHTED METRICS
# ============================================================================
# All metrics use survey weights (FINALWT) for population-level inference

# Collect all model predictions
models_preds = {
    'Linear Regression': y_pred_lr,
    'Ridge': y_pred_ridge,
    'Lasso': y_pred_lasso,
    'Random Forest': y_pred_rf,
    'Gradient Boosting': y_pred_gb,
}

if HAS_XGB:
    models_preds['XGBoost'] = y_pred_xgb

# Evaluate each model using WEIGHTED metrics
results = []
for name, preds in models_preds.items():
    # Weighted metrics in log space
    metrics = WeightedMetrics.evaluate(y_test, preds, w_test)
    
    # Weighted metrics in dollar space
    y_test_dollars = np.exp(y_test)
    pred_dollars = np.exp(preds)
    dollar_rmse = WeightedMetrics.weighted_rmse(y_test_dollars, pred_dollars, w_test)
    dollar_mae = WeightedMetrics.weighted_mae(y_test_dollars, pred_dollars, w_test)
    
    results.append({
        'Model': name,
        'Weighted R²': metrics['weighted_r2'],
        'Unweighted R²': metrics['unweighted_r2'],
        'Wtd RMSE (log)': metrics['weighted_rmse'],
        'Wtd MAE (log)': metrics['weighted_mae'],
        'Wtd RMSE ($)': dollar_rmse,
        'Wtd MAE ($)': dollar_mae
    })

results_df = pd.DataFrame(results).sort_values('Weighted R²', ascending=False)

print("=" * 90)
print("MODEL COMPARISON (WEIGHTED METRICS - Survey-Weighted for Population Inference)")
print("=" * 90)
print(f"\nTotal test observations: {len(y_test):,}")
print(f"Total survey weight: {w_test.sum():,.0f} (represents population)")
print()
display(results_df.round(4))

# Best model by weighted R²
best_model = results_df.iloc[0]
print(f"\n🏆 Best Model (by Weighted R²): {best_model['Model']}")
print(f"   Weighted R²: {best_model['Weighted R²']:.4f}")
print(f"   Population RMSE: ${best_model['Wtd RMSE ($)']:.2f}")

# Visualization of WEIGHTED model comparison
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Weighted R² comparison
ax = axes[0]
colors = ['#2ecc71' if r == results_df['Weighted R²'].max() else '#3498db' 
          for r in results_df['Weighted R²']]
bars = ax.barh(results_df['Model'], results_df['Weighted R²'], color=colors)
ax.set_xlabel('Weighted R² Score')
ax.set_title('Population-Weighted R² Comparison', fontweight='bold')
ax.set_xlim(0, 1)
for bar, val in zip(bars, results_df['Weighted R²']):
    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center')

# Weighted vs Unweighted R² comparison
ax = axes[1]
x = np.arange(len(results_df))
width = 0.35
bars1 = ax.bar(x - width/2, results_df['Weighted R²'], width, label='Weighted', color='#2ecc71')
bars2 = ax.bar(x + width/2, results_df['Unweighted R²'], width, label='Unweighted', color='#e74c3c', alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
ax.set_ylabel('R² Score')
ax.set_title('Weighted vs Unweighted R²', fontweight='bold')
ax.legend()
ax.set_ylim(0, 1)

# Weighted RMSE in dollars
ax = axes[2]
colors = ['#2ecc71' if r == results_df['Wtd RMSE ($)'].min() else '#e74c3c' 
          for r in results_df['Wtd RMSE ($)']]
bars = ax.barh(results_df['Model'], results_df['Wtd RMSE ($)'], color=colors)
ax.set_xlabel('Weighted RMSE ($)')
ax.set_title('Population-Weighted RMSE (lower = better)', fontweight='bold')
for bar, val in zip(bars, results_df['Wtd RMSE ($)']):
    ax.text(val + 0.5, bar.get_y() + bar.get_height()/2, f'${val:.2f}', va='center')

plt.tight_layout()
plt.savefig('../reports/figures/model_comparison_weighted.png', dpi=150, 
            bbox_inches='tight', facecolor='white')
plt.show()

print("\n📊 Figure saved: reports/figures/model_comparison_weighted.png")

# ============================================================================
# GENDER EFFECT ANALYSIS WITH SURVEY WEIGHTS
# ============================================================================
# For tree models: compute WEIGHTED average effect using FINALWT

print("=" * 70)
print("GENDER EFFECT BY MODEL (WEIGHTED Analysis)")
print("=" * 70)
print("(Negative coefficient = women earn less than men, controlling for other factors)")
print()

# Linear models have explicit coefficients
gender_idx = features.index('IS_FEMALE')

print(f"{'Model':<25} {'Coefficient':>12} {'Effect (%)':>12}")
print("-" * 55)

lr_gender = lr_model.coef_[gender_idx]
print(f"{'Linear Regression':<25} {lr_gender:>12.4f} {(np.exp(lr_gender)-1)*100:>11.1f}%")

ridge_gender = ridge_model.coef_[gender_idx]
print(f"{'Ridge':<25} {ridge_gender:>12.4f} {(np.exp(ridge_gender)-1)*100:>11.1f}%")

lasso_gender = lasso_model.coef_[gender_idx]
print(f"{'Lasso':<25} {lasso_gender:>12.4f} {(np.exp(lasso_gender)-1)*100:>11.1f}%")

# For tree models: compute WEIGHTED average effect using survey weights
X_male = X_test.copy()
X_male['IS_FEMALE'] = 0
X_female = X_test.copy()
X_female['IS_FEMALE'] = 1

# Random Forest - weighted average
rf_pred_male = rf_model.predict(X_male)
rf_pred_female = rf_model.predict(X_female)
rf_effect = np.average(rf_pred_female - rf_pred_male, weights=w_test)
print(f"{'Random Forest (weighted)':<25} {rf_effect:>12.4f} {(np.exp(rf_effect)-1)*100:>11.1f}%")

# Gradient Boosting - weighted average
gb_pred_male = gb_model.predict(X_male)
gb_pred_female = gb_model.predict(X_female)
gb_effect = np.average(gb_pred_female - gb_pred_male, weights=w_test)
print(f"{'Gradient Boosting (wtd)':<25} {gb_effect:>12.4f} {(np.exp(gb_effect)-1)*100:>11.1f}%")

if HAS_XGB:
    xgb_pred_male = xgb_model.predict(X_male)
    xgb_pred_female = xgb_model.predict(X_female)
    xgb_effect = np.average(xgb_pred_female - xgb_pred_male, weights=w_test)
    print(f"{'XGBoost (weighted)':<25} {xgb_effect:>12.4f} {(np.exp(xgb_effect)-1)*100:>11.1f}%")

# Summary
print("\n" + "=" * 70)
print("INTERPRETATION (Survey-Weighted Estimates)")
print("=" * 70)
print("These effects represent the unexplained gender wage gap after")
print("controlling for: education, age, province, occupation, industry,")
print("union membership, job permanence, and full/part-time status.")
print("\nA negative effect means women earn LESS than men with identical")
print("observable characteristics - the 'unexplained' gap that may reflect")
print("discrimination, negotiation differences, or omitted variables.")

# ============================================================================
# BIAS AMPLIFICATION CHECK USING WeightedGapAnalysis
# ============================================================================
# Check if models amplify or reduce existing wage gap bias

print("=" * 70)
print("BIAS AMPLIFICATION CHECK (Survey-Weighted)")
print("=" * 70)

# Prepare data for bias analysis
# Need gender indicator in test set
test_gender = df.loc[X_test.index, 'SEX'].values  # 1=Male, 2=Female

# Create analyzer
from src.ml_utils import WeightedGapAnalysis

# Analyze each model
bias_results = []

for name, preds in models_preds.items():
    analyzer = WeightedGapAnalysis(
        y_true=y_test.values,
        y_pred=preds,
        weights=w_test.values,
        group_indicator=test_gender,
        group_names={1: 'Male', 2: 'Female'}
    )
    
    actual_gap = analyzer.compute_gap(use_predictions=False)
    predicted_gap = analyzer.compute_gap(use_predictions=True)
    
    bias_check = analyzer.check_bias_amplification()
    
    bias_results.append({
        'Model': name,
        'Actual Gap (log)': actual_gap,
        'Predicted Gap (log)': predicted_gap,
        'Bias Amplification': 'YES' if bias_check['amplifies_bias'] else 'NO',
        'Gap Change (%)': (predicted_gap - actual_gap) / abs(actual_gap) * 100 if actual_gap != 0 else 0
    })
    
    print(f"\n{name}:")
    print(f"  Actual wage gap (log scale): {actual_gap:.4f}")
    print(f"  Predicted wage gap (log scale): {predicted_gap:.4f}")
    print(f"  Amplifies bias: {'⚠️ YES' if bias_check['amplifies_bias'] else '✅ NO'}")

# Summary table
bias_df = pd.DataFrame(bias_results)
print("\n" + "=" * 70)
print("BIAS AMPLIFICATION SUMMARY")
print("=" * 70)
display(bias_df.round(4))

# Warning for models that amplify bias
amplifying = bias_df[bias_df['Bias Amplification'] == 'YES']
if len(amplifying) > 0:
    print(f"\n⚠️ WARNING: {len(amplifying)} model(s) AMPLIFY the wage gap bias!")
    print("Consider bias mitigation techniques for production deployment.")
else:
    print("\n✅ All models reduce or maintain the existing wage gap bias.")

# ============================================================================
# NESTED CROSS-VALIDATION FOR UNBIASED PERFORMANCE ESTIMATION
# ============================================================================
# Implements the double CV procedure of Cawley & Talbot (2010)
# - Outer loop: 5-fold CV for unbiased performance estimation
# - Inner loop: 3-fold CV for hyperparameter tuning
# ============================================================================

from sklearn.model_selection import cross_val_score, KFold, GridSearchCV

print("=" * 70)
print("NESTED CROSS-VALIDATION (Unbiased Performance Estimation)")
print("=" * 70)

# Define models with hyperparameter grids
nested_cv_models = {
    'Ridge': {
        'estimator': Ridge(),
        'param_grid': {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
    },
    'Random Forest': {
        'estimator': RandomForestRegressor(random_state=42, n_jobs=-1),
        'param_grid': {
            'n_estimators': [50, 100],
            'max_depth': [5, 10, 15],
            'min_samples_leaf': [5, 10]
        }
    }
}

# Outer CV loop
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)

nested_results = []

for name, config in nested_cv_models.items():
    print(f"\nEvaluating {name}...")
    
    # Inner CV for hyperparameter tuning
    grid_search = GridSearchCV(
        config['estimator'],
        config['param_grid'],
        cv=inner_cv,
        scoring='r2',
        n_jobs=-1
    )
    
    # Outer CV for unbiased performance
    # Use scaled data for linear models
    if 'Ridge' in name or 'Linear' in name:
        X_scaled = scaler.fit_transform(X)
        outer_scores = cross_val_score(grid_search, X_scaled, y_log, cv=outer_cv, scoring='r2')
    else:
        outer_scores = cross_val_score(grid_search, X, y_log, cv=outer_cv, scoring='r2')
    
    nested_results.append({
        'Model': name,
        'Nested CV Mean R²': outer_scores.mean(),
        'Nested CV Std': outer_scores.std(),
        'CV Scores': outer_scores
    })
    
    print(f"  Nested CV R²: {outer_scores.mean():.4f} ± {outer_scores.std():.4f}")
    print(f"  Fold scores: {', '.join([f'{s:.4f}' for s in outer_scores])}")

# Compare nested CV results
print("\n" + "=" * 70)
print("NESTED CV RESULTS SUMMARY")
print("=" * 70)
print(f"\n{'Model':<20} {'Nested CV R²':>18} {'Std Dev':>12}")
print("-" * 50)

for nested_result in nested_results:
    model_name = nested_result['Model']
    nested_r2 = nested_result['Nested CV Mean R²']
    nested_std = nested_result['Nested CV Std']
    print(f"{model_name:<20} {nested_r2:>18.4f} {nested_std:>12.4f}")

print("\n📊 Note: Nested CV provides unbiased performance estimates")

# ============================================================================
# LEARNING CURVES FOR BIAS-VARIANCE DIAGNOSIS
# ============================================================================

from sklearn.model_selection import learning_curve

print("=" * 70)
print("LEARNING CURVE ANALYSIS (Bias-Variance Trade-off)")
print("=" * 70)

def plot_learning_curves(estimator, title, X, y, cv=5, scoring='r2'):
    """
    Generate learning curve plot for bias-variance diagnosis.
    """
    train_sizes = np.linspace(0.1, 1.0, 10)
    
    # learning_curve returns 5 values in newer sklearn versions
    result = learning_curve(
        estimator, X, y,
        train_sizes=train_sizes,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        random_state=42
    )
    
    # Unpack only the first 3 values (train_sizes, train_scores, val_scores)
    train_sizes_abs = result[0]
    train_scores = result[1]
    val_scores = result[2]
    
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    return train_sizes_abs, train_mean, train_std, val_mean, val_std

# Sample data for faster computation
sample_size = min(10000, len(X))
X_sample = X.sample(sample_size, random_state=42)
y_sample = y_log.loc[X_sample.index]
X_sample_scaled = scaler.fit_transform(X_sample)

# Learning curves for different models
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

models_for_lc = [
    ('Linear Regression', LinearRegression(), X_sample_scaled, axes[0, 0]),
    ('Ridge (α=1.0)', Ridge(alpha=1.0), X_sample_scaled, axes[0, 1]),
    ('Random Forest', RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42), X_sample, axes[1, 0]),
    ('Gradient Boosting', GradientBoostingRegressor(n_estimators=50, max_depth=5, random_state=42), X_sample, axes[1, 1])
]

for name, model, X_lc, ax in models_for_lc:
    train_sizes, train_mean, train_std, val_mean, val_std = plot_learning_curves(
        model, name, X_lc, y_sample, cv=5
    )
    
    # Plot training scores
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
    
    # Plot validation scores
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='orange')
    ax.plot(train_sizes, val_mean, 'o-', color='orange', label='Cross-Validation Score')
    
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('R² Score')
    ax.set_title(f'{name}', fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # Diagnose bias-variance
    final_gap = train_mean[-1] - val_mean[-1]
    if final_gap > 0.1:
        diagnosis = "High Variance (overfitting)"
    elif val_mean[-1] < 0.3:
        diagnosis = "High Bias (underfitting)"
    else:
        diagnosis = "Balanced"
    ax.annotate(diagnosis, xy=(0.5, 0.05), xycoords='axes fraction', 
                fontsize=9, ha='center', style='italic')

plt.tight_layout()
plt.savefig('../reports/figures/learning_curves.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n📊 Learning Curves Analysis Complete")

# ============================================================================
# SHAP VALUES FOR MODEL INTERPRETABILITY
# ============================================================================
# Lundberg & Lee (2017): "A Unified Approach to Interpreting Model Predictions"
# ============================================================================

try:
    import shap
    
    print("=" * 70)
    print("SHAP VALUE ANALYSIS (Model-Agnostic Interpretability)")
    print("=" * 70)
    
    # Use a sample for SHAP computation (SHAP can be slow for large datasets)
    shap_sample_size = min(500, len(X_test))
    X_shap = X_test.sample(shap_sample_size, random_state=42)
    
    # Use TreeExplainer for tree-based models (fast)
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_shap)
    
    # Summary plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Beeswarm plot
    plt.subplot(1, 2, 1)
    shap.summary_plot(shap_values, X_shap, feature_names=features, 
                      plot_type="violin", show=False)
    plt.title('SHAP Value Distribution by Feature', fontweight='bold')
    
    # Bar plot (mean absolute SHAP values)
    plt.subplot(1, 2, 2)
    shap.summary_plot(shap_values, X_shap, feature_names=features, 
                      plot_type="bar", show=False)
    plt.title('Mean |SHAP Value| (Feature Importance)', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('../reports/figures/shap_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # SHAP values for gender specifically
    gender_idx = features.index('IS_FEMALE')
    gender_shap = shap_values[:, gender_idx]
    
    print(f"\nGender (IS_FEMALE) SHAP Analysis:")
    print(f"  Mean SHAP value:   {gender_shap.mean():.4f}")
    print(f"  Median SHAP value: {np.median(gender_shap):.4f}")
    print(f"  Std SHAP value:    {gender_shap.std():.4f}")
    print(f"  Effect range:      [{gender_shap.min():.4f}, {gender_shap.max():.4f}]")
    
    # Interpretation
    avg_gender_effect = gender_shap.mean()
    pct_effect = (np.exp(avg_gender_effect) - 1) * 100
    print(f"\n  → Average gender effect: {pct_effect:+.2f}% on wages")
    print(f"    (negative = women earn less)")
    
    print("\n📊 SHAP plots saved to reports/figures/shap_analysis.png")
    
except ImportError:
    print("⚠️ SHAP not installed. Install with: pip install shap")
    print("   Skipping SHAP analysis...")

# ============================================================================
# PREDICTION INTERVALS VIA BOOTSTRAP
# ============================================================================
# Following Efron & Tibshirani (1993) bootstrap methodology
# ============================================================================

print("=" * 70)
print("PREDICTION INTERVALS VIA BOOTSTRAP")
print("=" * 70)

def bootstrap_prediction_intervals(model, X_train, y_train, X_test, 
                                   n_bootstraps=100, alpha=0.05):
    """
    Compute prediction intervals using bootstrap aggregation.
    
    Parameters
    ----------
    model : sklearn estimator
        Base model to bootstrap
    X_train, y_train : training data
    X_test : test features
    n_bootstraps : int
        Number of bootstrap samples
    alpha : float
        Significance level (0.05 for 95% CI)
    
    Returns
    -------
    predictions, lower_bound, upper_bound
    """
    n_train = len(X_train)
    n_test = len(X_test)
    
    # Store predictions from each bootstrap
    bootstrap_preds = np.zeros((n_bootstraps, n_test))
    
    for i in range(n_bootstraps):
        # Sample with replacement
        idx = np.random.choice(n_train, size=n_train, replace=True)
        X_boot = X_train.iloc[idx] if hasattr(X_train, 'iloc') else X_train[idx]
        y_boot = y_train.iloc[idx] if hasattr(y_train, 'iloc') else y_train[idx]
        
        # Fit model and predict
        model_clone = type(model)(**model.get_params())
        model_clone.fit(X_boot, y_boot)
        bootstrap_preds[i, :] = model_clone.predict(X_test)
    
    # Calculate prediction intervals
    predictions = bootstrap_preds.mean(axis=0)
    lower = np.percentile(bootstrap_preds, (alpha/2) * 100, axis=0)
    upper = np.percentile(bootstrap_preds, (1 - alpha/2) * 100, axis=0)
    
    return predictions, lower, upper

# Compute prediction intervals for a subset (bootstrap is expensive)
pi_sample_size = min(500, len(X_test))
X_test_pi = X_test.sample(pi_sample_size, random_state=42)
y_test_pi = y_test.loc[X_test_pi.index]

print(f"Computing 95% prediction intervals via {50} bootstrap samples...")

preds, lower, upper = bootstrap_prediction_intervals(
    RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1),
    X_train, y_train, X_test_pi, n_bootstraps=50
)

# Coverage calculation
coverage = np.mean((y_test_pi >= lower) & (y_test_pi <= upper))
avg_width = np.mean(upper - lower)

print(f"\n95% Prediction Interval Results:")
print(f"  Coverage: {coverage*100:.1f}% (target: 95%)")
print(f"  Average interval width: {avg_width:.3f} (log scale)")
print(f"  Average width in $: ${np.exp(avg_width):.2f}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Sort by predicted value for cleaner plot
sort_idx = np.argsort(preds)[:100]  # First 100 for visibility

ax = axes[0]
ax.fill_between(range(len(sort_idx)), 
                np.exp(lower[sort_idx]), np.exp(upper[sort_idx]), 
                alpha=0.3, label='95% PI')
ax.scatter(range(len(sort_idx)), np.exp(y_test_pi.iloc[sort_idx]), 
           c='red', s=20, alpha=0.7, label='Actual')
ax.plot(range(len(sort_idx)), np.exp(preds[sort_idx]), 'b-', 
        linewidth=1.5, label='Predicted')
ax.set_xlabel('Observation (sorted by prediction)')
ax.set_ylabel('Hourly Wage ($)')
ax.set_title('Prediction Intervals (95% Bootstrap)')
ax.legend()

# Coverage by prediction level
ax = axes[1]
n_bins = 5
bins = np.percentile(preds, np.linspace(0, 100, n_bins + 1))
bin_coverage = []
for i in range(n_bins):
    mask = (preds >= bins[i]) & (preds < bins[i+1])
    if mask.sum() > 0:
        bin_cov = np.mean((y_test_pi.values[mask] >= lower[mask]) & 
                          (y_test_pi.values[mask] <= upper[mask]))
        bin_coverage.append(bin_cov * 100)
    else:
        bin_coverage.append(0)

ax.bar(range(n_bins), bin_coverage, color='steelblue', edgecolor='black')
ax.axhline(95, color='red', linestyle='--', label='Target 95%')
ax.set_xlabel('Prediction Quintile')
ax.set_ylabel('Coverage (%)')
ax.set_title('Coverage by Prediction Level')
ax.set_xticks(range(n_bins))
ax.set_xticklabels([f'Q{i+1}' for i in range(n_bins)])
ax.legend()

plt.tight_layout()
plt.savefig('../reports/figures/prediction_intervals.png', dpi=150, bbox_inches='tight')
plt.show()

print("📊 Prediction intervals saved to reports/figures/prediction_intervals.png")

# Cross-validation for best model
print("5-FOLD CROSS-VALIDATION")
print("=" * 40)

cv_models = [
    ('Linear Regression', LinearRegression()),
    ('Ridge', Ridge(alpha=1.0)),
    ('Random Forest', RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)),
    ('Gradient Boosting', GradientBoostingRegressor(n_estimators=50, max_depth=5, random_state=42))
]

cv_results = []
for name, model in cv_models:
    # Use scaled data for linear models
    if 'Linear' in name or 'Ridge' in name:
        scores = cross_val_score(model, scaler.fit_transform(X), y_log, cv=5, scoring='r2')
    else:
        scores = cross_val_score(model, X, y_log, cv=5, scoring='r2')
    
    cv_results.append({
        'Model': name,
        'Mean R²': scores.mean(),
        'Std': scores.std(),
        'Min': scores.min(),
        'Max': scores.max()
    })
    print(f"{name}: R² = {scores.mean():.4f} (+/- {scores.std()*2:.4f})")

cv_df = pd.DataFrame(cv_results)
display(cv_df.round(4))

# Select best model based on R²
best_model_name = results_df.iloc[0]['Model']
print(f"Best model: {best_model_name}")

# Save the models
models_path = Path('../models')
models_path.mkdir(exist_ok=True)

# Save the best performing model
if 'Random Forest' in best_model_name:
    best_model = rf_model
elif 'XGBoost' in best_model_name and HAS_XGB:
    best_model = xgb_model
elif 'Gradient' in best_model_name:
    best_model = gb_model
else:
    best_model = lr_model

joblib.dump(best_model, models_path / 'salary_predictor.joblib')
joblib.dump(scaler, models_path / 'feature_scaler.joblib')

print(f"\n✓ Saved model to {models_path / 'salary_predictor.joblib'}")
print(f"✓ Saved scaler to {models_path / 'feature_scaler.joblib'}")

# Summary
print("\n" + "=" * 60)
print("MODEL TRAINING SUMMARY")
print("=" * 60)

print(f"\nDataset: {len(df):,} records")
print(f"Features: {len(features)}")
print(f"Best model: {best_model_name}")
print(f"Best R²: {results_df.iloc[0]['R²']:.4f}")
print(f"Best RMSE: ${results_df.iloc[0]['RMSE ($)']:.2f}")

print(f"\nKey Finding:")
print(f"Controlling for education, occupation, experience, and other factors,")
print(f"women earn approximately {abs((np.exp(lr_gender)-1)*100):.1f}% less than men.")