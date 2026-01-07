# EquiPay Canada: Methodology & Critical Analysis

## Executive Summary

This document critically evaluates Statistics Canada's existing gender wage gap methodology and proposes significant improvements for the EquiPay Canada project. Our approach addresses key limitations in current official analyses while maintaining scientific rigor.

---

## Part I: Critique of Existing Canadian Pay Equity Research

### 1. Statistics Canada's Approach (Pelletier, Patterson & Moyser, 2019)

**What They Did Right:**
- Used Blinder-Oaxaca decomposition (standard econometric method)
- Focused on core-aged workers (25-54) to avoid school/retirement transitions
- Compared hourly wages (controls for hours worked)
- Used 20-year time series (1998-2018)
- Applied survey weights (FINALWT)

**Key Findings:**
- Gender wage gap: 13.3% in 2018 (down from 18.8% in 1998)
- ~36% explained by observable characteristics
- **~64% UNEXPLAINED** - this is the critical gap in understanding

### 2. Critical Limitations of Current Methodology

#### A. The "Unexplained Portion" Problem (64% of the gap!)

The StatCan report explicitly acknowledges:
> "Similar to other studies, nearly two-thirds of the gap in 2018 was unexplained."

**Why this is problematic:**
1. **Missing Variable Bias**: LFS lacks critical wage determinants:
   - Actual work experience (only job tenure available)
   - Field of study (not just education level)
   - Negotiation behavior
   - Career interruptions history
   - Firm-level effects

2. **Aggregation Bias**: 2-digit NOC/NAICS codes are too coarse
   - A "management" category includes CEOs and first-line supervisors
   - Within-occupation variation can be larger than between-occupation

3. **Composition Effects**: Compares averages, not matched comparisons
   - A female nurse vs. male engineer comparison inflates "unexplained"

#### B. Methodological Limitations

| Issue | StatCan Approach | Problem |
|-------|------------------|---------|
| Decomposition | Pooled Blinder-Oaxaca | Assumes same returns for men/women |
| Standard Errors | Not consistently reported | Inference uncertain |
| Selection Bias | Acknowledged but not corrected | Women's labor force participation differs |
| Quantile Effects | Means only | Misses "glass ceiling" (gap larger at top) |
| Intersectionality | Minimal | Race × gender interactions ignored |
| Time Dynamics | Static snapshots | No cohort or lifecycle analysis |

#### C. The "Equal Work" Fallacy

The Pay Equity Act mandates "equal pay for work of equal value" - but StatCan analysis cannot measure "value" directly:
- Job evaluations require firm-level data
- Market wages ≠ intrinsic job value
- Comparable worth analysis impossible with LFS

### 3. What's Missing from Policy Discourse

1. **Intersectionality**: Indigenous women, racialized women, immigrant women face compounded gaps
2. **Within-firm analysis**: Most discrimination occurs at firm level, not market level  
3. **Career trajectory analysis**: Starting wage gaps compound over lifetime
4. **Part-time penalty analysis**: Is the 9% part-time "explanation" actually discrimination?
5. **Regional labor market effects**: BC gap (18.6%) vs NB gap (7.4%) - why?

---

## Part II: EquiPay Canada's Enhanced Methodology

### 1. Survey Weighting (MANDATORY)

**Why FINALWT Matters:**
- LFS is a stratified random sample (~56,000 households/month)
- Each respondent represents X Canadians (weight varies)
- Unweighted estimates are **biased** for population inference

**Implementation:**

```python
# WRONG: Unweighted mean
unweighted_mean = df['HRLYEARN'].mean()  # Treats all observations equally

# CORRECT: Weighted mean
weighted_mean = np.average(df['HRLYEARN'] / 100, weights=df['FINALWT'])

# In DuckDB:
"""
SELECT SUM(HRLYEARN * FINALWT) / SUM(FINALWT) / 100 as weighted_avg_wage
FROM lfs WHERE HRLYEARN > 0
"""
```

**Variance Estimation with Poisson Bootstrap (per StatsCan Guide, January 2025):**

The LFS PUMF uses a complex survey design. For proper variance estimation, we implement
the Poisson Bootstrap method as specified in the official StatsCan LFS PUMF User Guide:

```python
from src.bootstrap_variance import PoissonBootstrap

# Generate 1000 bootstrap replicates (per StatsCan recommendation)
bs = PoissonBootstrap(df, n_replicates=1000, seed=42, calibrate=True)

# Estimate wage gap with proper variance
gap_result = bs.estimate_wage_gap()
# Returns: male_wage, female_wage, gap, gap_cv, 95% CI, quality assessment

# Quality thresholds per StatsCan:
# - CV < 15%: Acceptable
# - CV 15-35%: Marginal (requires warning)
# - CV > 35%: Unacceptable (should not publish)
```

Reference: Beaumont, J.-F., & Patak, Z. (2012). "On the generalized bootstrap for sample
surveys with special attention to Poisson sampling." International Statistical Review.

### 1.5 Critical Data Transformations (per StatsCan PUMF Guide)

The LFS PUMF stores variables with implicit decimals for space efficiency:

| Variable | Raw Storage | Transformation | Example |
|----------|-------------|----------------|---------|
| HRLYEARN | Cents (integer) | ÷ 100 → Dollars | 2345 → $23.45 |
| AHRSMAIN | Tenths (integer) | ÷ 10 → Hours | 435 → 43.5 hrs |
| UHRSMAIN | Tenths (integer) | ÷ 10 → Hours | 400 → 40.0 hrs |
| ATOTHRS | Tenths (integer) | ÷ 10 → Hours | 450 → 45.0 hrs |
| UTOTHRS | Tenths (integer) | ÷ 10 → Hours | 440 → 44.0 hrs |
| HRSAWAY | Tenths (integer) | ÷ 10 → Hours | 80 → 8.0 hrs |

**Implementation in EquiPay Canada:**
These transformations are applied automatically in `src/data_store.py` when creating 
the DuckDB view, ensuring all downstream analyses use correctly scaled values.

### 2. Enhanced Decomposition Methods

#### A. Oaxaca-Blinder with Robustness
```python
# Threefold decomposition (Blinder 1973, Oaxaca 1973)
# Gap = Endowments + Coefficients + Interaction

# Use weighted regression for both groups
from statsmodels.api import WLS
model_male = WLS(y_male, X_male, weights=w_male).fit()
model_female = WLS(y_female, X_female, weights=w_female).fit()
```

#### B. Reweighted (DiNardo-Fortin-Lemieux) Decomposition
- Creates counterfactual distributions
- Shows what women would earn if they had men's characteristics
- Better for quantile analysis

#### C. Unconditional Quantile Regression (Firpo-Fortin-Lemieux)
- Examines gap at 10th, 25th, 50th, 75th, 90th percentiles
- Reveals "sticky floor" (bottom) vs "glass ceiling" (top) effects

### 3. Machine Learning Augmentation

**Why ML improves on OLS decomposition:**
1. Handles nonlinear interactions automatically
2. Flexible functional forms (doesn't assume log-linear)
3. Can capture complex occupation × education × industry interactions

**Our Approach:**
```python
# Train separate models for men and women
model_male = GradientBoostingRegressor()
model_female = GradientBoostingRegressor()

# Counterfactual: What would women earn with male model?
counterfactual_female = model_male.predict(X_female)

# Explained gap = avg(counterfactual) - avg(female_actual)
# Unexplained = avg(male_actual) - avg(counterfactual)
```

**SHAP Values for Interpretability:**
- Decompose predictions to feature contributions
- Identify which characteristics drive the gap
- Non-linear effects visible

### 4. Intersectional Analysis

**Framework:**
```
Total Gap = Gender Effect 
          + Race Effect 
          + Gender × Race Interaction
          + Other Demographics
          + Labor Market Characteristics
          + Unexplained
```

**Groups to Analyze:**
- Indigenous women (IMMIG + detailed responses)
- Visible minority women by group
- Immigrant women by period of landing
- Women with disabilities (if identifiable)

### 5. Regional and Temporal Dynamics

**Panel/Pseudo-Panel Methods:**
- Cohort analysis: Track birth cohorts over time
- Age-period-cohort decomposition
- Regional fixed effects with time trends

**Spatial Analysis:**
- CMA-level variation
- Urban vs rural gaps
- Cross-border comparisons (US border cities)

---

## Part III: Sampling - Is It Necessary?

### For This Project: **NO, Sampling is NOT Necessary**

**Reasons:**

1. **Dataset Size is Manageable**
   - 9.88M valid wage records
   - ~650MB in memory as DataFrame
   - DuckDB handles 19.5M rows efficiently

2. **We Have the Full Population of the Survey**
   - LFS is already a sample (56,000 households/month)
   - Sub-sampling would reduce statistical power unnecessarily
   - Rare subgroups (e.g., Indigenous women in specific occupations) already small

3. **Survey Weights Account for Sampling**
   - FINALWT already represents population extrapolation
   - Further sampling would require reweighting

4. **Computational Efficiency is Solved**
   - DuckDB queries run in seconds on full dataset
   - Parquet format enables efficient I/O
   - Aggregations don't require full DataFrame in memory

### When Sampling WOULD Be Appropriate:

1. **Computationally Intensive ML Models**
   - Deep learning on 10M rows can be slow
   - Use stratified sampling (by gender, province, year) for training
   - Validate on full dataset

2. **Interactive Exploration**
   - For notebook experimentation, 1M rows may suffice
   - Use `TABLESAMPLE` in DuckDB for quick iterations

3. **Bootstrap Variance Estimation**
   - 1000 bootstrap samples of ~100K each
   - Provides confidence intervals for complex statistics

**Recommended Approach:**
```python
# For quick exploration
df_sample = store.query("""
    SELECT * FROM lfs 
    USING SAMPLE 10 PERCENT (BERNOULLI, REPEATABLE(42))
    WHERE HRLYEARN > 0
""")

# For final analysis: ALWAYS use full data with weights
df_full = store.get_lfs_data(valid_wages_only=True)
```

---

## Part IV: Machine Learning - Train/Test/Validation Splits

### Why Splitting is Necessary for ML (Even Though Sampling is Not)

**Key Distinction:**
- **Sampling** = reducing dataset size for efficiency → NOT needed
- **Splitting** = holding out data for unbiased model evaluation → REQUIRED

Machine learning models must be evaluated on data they never saw during training. This prevents overfitting and provides realistic performance estimates.

### Survey Weight Considerations for ML

#### 1. Weights Must Travel with Data

```python
# CORRECT: Split data AND weights together
from src.ml_utils import WeightedMLSplitter

splitter = WeightedMLSplitter(
    df=df,
    target_col='HRLYEARN',
    weight_col='FINALWT',  # MANDATORY
    stratify_cols=['SEX']   # Preserve gender balance
)

splits = splitter.create_splits(test_size=0.2, val_size=0.1)
# Returns: {'train': {'X', 'y', 'weights'}, 'val': {...}, 'test': {...}}
```

#### 2. Stratification Preserves Population Structure

| Split Strategy | Problem | Solution |
|----------------|---------|----------|
| Random | May under-represent women in test set | Stratify by SEX |
| Random | May exclude rare provinces | Stratify by PROV |
| Time-based | Future test on past training | Use temporal split for forecasting |

#### 3. Weights in Training

Most scikit-learn compatible models support `sample_weight`:

```python
# XGBoost, LightGBM, CatBoost, RandomForest, Ridge, etc.
model.fit(
    X_train, y_train,
    sample_weight=splits['train']['weights']  # CRITICAL
)
```

This ensures the model learns patterns that generalize to the **population**, not just the sample.

#### 4. Weighted Evaluation Metrics

Standard metrics (MSE, MAE, R²) are biased for population inference:

```python
from src.ml_utils import WeightedMetrics

# CORRECT: Weighted metrics for population inference
metrics = WeightedMetrics.evaluate(
    y_true=splits['test']['y'],
    y_pred=model.predict(splits['test']['X']),
    weights=splits['test']['weights']
)
# Returns: weighted_rmse, weighted_mae, weighted_r2, weighted_mape
```

### Recommended Split Strategies

#### A. Stratified Random Split (Default)

Best for cross-sectional analysis:

```python
splits = splitter.create_splits(
    test_size=0.2,   # 20% for final evaluation
    val_size=0.1     # 10% for hyperparameter tuning
)
```

**Population breakdown:**
- Training: ~70% (weighted pop: ~14M Canadians)
- Validation: ~10% (weighted pop: ~2M Canadians)
- Test: ~20% (weighted pop: ~4M Canadians)

#### B. Temporal Split (For Forecasting)

Best for predicting future gaps:

```python
splits = splitter.create_splits(
    temporal_split=True,
    temporal_test_years=[2024, 2025]  # Hold out recent years
)
```

**Prevents data leakage:**
- Train on 2010-2022
- Validate on 2023
- Test on 2024-2025

#### C. Cross-Validation with Weights

For robust model comparison:

```python
folds = splitter.create_cv_folds(n_folds=5, stratify=True)

for fold_idx, (train_idx, val_idx) in enumerate(folds):
    train, val = splitter.get_fold_data((train_idx, val_idx))
    
    model.fit(train['X'], train['y'], sample_weight=train['weights'])
    
    score = WeightedMetrics.weighted_r2(
        val['y'], model.predict(val['X']), val['weights']
    )
```

### Bias Detection in Predictions

After training, check if the model amplifies or reduces the wage gap:

```python
from src.ml_utils import WeightedGapAnalysis

bias_check = WeightedGapAnalysis.check_bias_amplification(
    y_true=test_y,
    y_pred=model.predict(test_X),
    weights=test_weights,
    groups=test_gender  # 1=male, 2=female
)

print(bias_check['status'])  # 'AMPLIFYING', 'REDUCING', or 'NEUTRAL'
print(bias_check['recommendation'])
```

### Summary: ML Pipeline with Survey Weights

```
┌─────────────────────────────────────────────────────────────────┐
│                    FULL LFS DATASET                             │
│                   (19.5M records, 100%)                          │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
         ┌──────────────────────────────────┐
         │  WeightedMLSplitter               │
         │  - Stratify by SEX                │
         │  - Keep FINALWT with each split   │
         └──────────────────────────────────┘
                        │
            ┌───────────┼───────────┐
            ▼           ▼           ▼
       ┌────────┐  ┌────────┐  ┌────────┐
       │ TRAIN  │  │  VAL   │  │  TEST  │
       │  70%   │  │  10%   │  │  20%   │
       │+weights│  │+weights│  │+weights│
       └───┬────┘  └───┬────┘  └───┬────┘
           │           │           │
           ▼           ▼           ▼
    ┌─────────────────────────────────────┐
    │  Model Training with sample_weight   │
    └───────────────────────────────────────┘
           │
           ▼
    ┌─────────────────────────────────────┐
    │  Evaluation: WeightedMetrics         │
    │  Bias Check: WeightedGapAnalysis     │
    └─────────────────────────────────────┘
```

---

## Part IV: Implementation Roadmap

### Phase 1: Weighted Infrastructure (Priority: HIGH)

1. **Add weighted methods to data_store.py:**
   - `get_weighted_mean(column, weights='FINALWT')`
   - `get_weighted_gender_gap()`
   - `get_weighted_quantiles()`

2. **Update all notebooks to use weighted statistics**

3. **Implement proper variance estimation**

### Phase 2: Enhanced Decomposition

4. **Implement Oaxaca-Blinder with survey weights**
5. **Add quantile decomposition**
6. **Integrate SHAP-based ML decomposition**

### Phase 3: Advanced Analysis

7. **Intersectionality module**
8. **Regional analysis with spatial effects**
9. **Time series of gap evolution**

### Phase 4: Validation & Benchmarking

10. **Compare our estimates to StatCan official figures**
11. **Document differences and explanations**
12. **Policy recommendations**

---

## References

1. Pelletier, R., Patterson, M., & Moyser, M. (2019). The gender wage gap in Canada: 1998 to 2018. Statistics Canada Catalogue no. 75-004-M.

2. Blinder, A. S. (1973). Wage discrimination: Reduced form and structural estimates. Journal of Human Resources, 8(4), 436-455.

3. Oaxaca, R. (1973). Male-female wage differentials in urban labor markets. International Economic Review, 14(3), 693-709.

4. Firpo, S., Fortin, N. M., & Lemieux, T. (2009). Unconditional quantile regressions. Econometrica, 77(3), 953-973.

5. DiNardo, J., Fortin, N. M., & Lemieux, T. (1996). Labor market institutions and the distribution of wages, 1973-1992. Econometrica, 64(5), 1001-1044.

6. Moyser, M. (2019). Measuring and analyzing the gender pay gap: A conceptual and methodological overview. Statistics Canada Catalogue no. 45-20-0002.

---

## Key Takeaways for EquiPay Canada

| Aspect | StatCan Approach | Our Enhancement |
|--------|------------------|-----------------|
| Weighting | Applied | **Mandatory + proper variance** |
| Decomposition | Blinder-Oaxaca only | **Multiple methods + ML** |
| Unexplained portion | 64% black box | **SHAP interpretability** |
| Quantiles | Means only | **Full distribution analysis** |
| Intersectionality | Minimal | **Core feature** |
| Temporal | Snapshots | **Cohort + lifecycle** |
| Variables | LFS only | **LFS + macro integration** |

**Our Goal:** Reduce the "unexplained" portion through better methodology and provide actionable insights for policy intervention.
