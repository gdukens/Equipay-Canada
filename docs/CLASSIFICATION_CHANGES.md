# NOC and NAICS Classification Changes: Impact on EquiPay Canada Analysis

## Executive Summary

This document tracks the changes in the National Occupational Classification (NOC) and North American Industry Classification System (NAICS) from 2010-2025, and assesses their potential impact on time series analysis of the gender wage gap.

**Key Finding:** The LFS PUMF uses **aggregated classification codes** (NOC_10, NOC_43, NAICS_21) that are designed for consistency across classification revisions. However, users should be aware of potential methodological breaks.

---

## 1. Classification Revision Timeline

### National Occupational Classification (NOC)

| Version | Years Used | Key Changes |
|---------|------------|-------------|
| NOC 2011 | 2011-2015 | First major restructuring in decade; skill level embedded in codes |
| NOC 2016 v1.0-1.3 | 2016-2021 | Minor updates to emerging occupations (e.g., data scientists) |
| **NOC 2021** | 2022-present | **MAJOR RESTRUCTURING**: New 5-digit codes, TEER categories replace skill levels |

#### NOC 2021 Key Changes:
1. **New structure**: Changed from 4-digit to 5-digit codes
2. **TEER categories**: Replaced "skill level" with "Training, Education, Experience, Responsibilities"
3. **New broad categories**: 
   - Category 0: Legislative/senior management → Now 6 TEER 0 categories
   - Digital/tech occupations expanded significantly
   - Healthcare occupations restructured

### North American Industry Classification System (NAICS)

| Version | Years Used | Key Changes |
|---------|------------|-------------|
| NAICS 2007 | 2008-2011 | Baseline for our data |
| NAICS 2012 | 2012-2016 | Information sector restructured; Electronic shopping added |
| NAICS 2017 | 2017-2021 | Cannabis production added; more detail in professional services |
| **NAICS 2022** | 2022-present | Digital industries expanded; remote work impacts |

---

## 2. LFS PUMF Aggregation Strategy

Statistics Canada mitigates classification breaks through **aggregation**:

### NOC Codes in PUMF

| Variable | Levels | Description | Stability |
|----------|--------|-------------|-----------|
| `NOC_10` | 10 | Broad occupational categories | **HIGH** - Consistent 2010-2025 |
| `NOC_43` | 43 | 2-digit minor groups | **MEDIUM** - Some shifts at 2022 |
| `NOC_40` | 40 | Alternative 2-digit grouping | **MEDIUM** - Present in some years |

### NAICS Codes in PUMF

| Variable | Levels | Description | Stability |
|----------|--------|-------------|-----------|
| `NAICS_21` | 21 | 2-digit sector codes | **HIGH** - Mostly stable |

---

## 3. Empirical Analysis: Discontinuity Detection

### 3.1 Code Presence Analysis

```
NOC_10 codes by year: 10 codes present in ALL years (2010-2025) ✓
NOC_43 codes by year: 43 codes present in ALL years (2010-2025) ✓
NAICS_21 codes by year: 21 codes present in ALL years (2010-2025) ✓
```

**Result:** No codes disappeared or appeared - aggregation maintains continuity.

### 3.2 Distribution Shift Analysis

Largest share changes at NAICS revision years:

| Transition | Industry | Share Change |
|------------|----------|--------------|
| 2021→2022 | NAICS 14 (Finance/Insurance) | +0.9pp |
| 2021→2022 | NAICS 12 (Manufacturing) | +0.6pp |
| 2021→2022 | NAICS 18 (Healthcare) | +0.5pp |

**Interpretation:** The 2022 NAICS revision coincided with post-COVID labour market shifts, making it difficult to isolate classification effects from economic effects.

### 3.3 Wage Gap Discontinuity Analysis

Year-over-year changes in gender wage gap by NOC at NOC 2021 revision:

| NOC_10 | 2021→2022 Gap Change |
|--------|----------------------|
| 10 (Manufacturing/Utilities) | -4.3pp |
| 4 (Admin/Finance) | -2.6pp |
| 2 (Business/Finance) | -2.4pp |

**Caution:** These large changes may indicate:
1. Classification boundary changes moving workers between categories
2. Post-COVID wage dynamics
3. Combination of both effects

---

## 4. Impact on EquiPay Canada Analysis

### 4.1 Potential Issues

1. **Time Series Breaks at 2022:**
   - NOC 2021 restructuring may cause artificial trend breaks
   - Occupation-level analysis may show spurious changes

2. **Within-Category Heterogeneity:**
   - Aggregated codes (NOC_10, NOC_43) mask detailed changes
   - The same code may contain different underlying occupations pre/post revision

3. **Decomposition Sensitivity:**
   - Oaxaca-Blinder decomposition uses occupation as explanatory variable
   - Classification changes affect "explained" portion of gap

### 4.2 Recommended Mitigations

#### A. For Time Series Analysis
```python
# Option 1: Split analysis at structural break
pre_2022 = df[df['SURVYEAR'] < 2022]
post_2022 = df[df['SURVYEAR'] >= 2022]

# Option 2: Include structural break dummy
df['post_noc2021'] = (df['SURVYEAR'] >= 2022).astype(int)
# Include in regression: ... + post_noc2021 + post_noc2021:NOC_10 + ...
```

#### B. For Cross-Sectional Analysis
```python
# Use broad categories (NOC_10) for multi-year analysis
# Use detailed categories (NOC_43) only for single-year analysis
if analysis_years > 1:
    occupation_col = 'NOC_10'
else:
    occupation_col = 'NOC_43'
```

#### C. For Decomposition Analysis
```python
# Report sensitivity analysis with and without occupation controls
gap_with_occ = decompose(X_with_occupation)
gap_without_occ = decompose(X_without_occupation)
# Compare to assess classification sensitivity
```

### 4.3 What EquiPay Canada Does

Our implementation handles classification issues through:

1. **Conservative aggregation:** Uses NOC_10 and NAICS_21 for trend analysis
2. **Survey-weighted estimation:** FINALWT accounts for population composition
3. **Flexible time controls:** Year fixed effects capture structural breaks
4. **Robustness checks:** Reports results with/without occupation controls

---

## 5. Concordance Tables

For users requiring exact mappings:

### NOC 2016 → NOC 2021

Statistics Canada provides official concordance tables:
- [NOC 2016 to NOC 2021 Concordance](https://www.statcan.gc.ca/en/subjects/standard/noc/2021/concordancetables)

### NAICS Concordances

- [NAICS 2017 to NAICS 2022](https://www.statcan.gc.ca/en/subjects/standard/naics/2022/concordances)
- [NAICS 2012 to NAICS 2017](https://www.statcan.gc.ca/en/subjects/standard/naics/2017/concordances)

---

## 6. Recommendations for Future Analysis

1. **Document classification version:** Always report which NOC/NAICS version applies to your analysis period

2. **Use stable aggregations:** Prefer NOC_10 and NAICS_21 for multi-year analysis

3. **Test for structural breaks:** Run Chow tests or include structural break dummies at 2012, 2017, 2022

4. **Report sensitivity:** Show how results change with different occupation/industry controls

5. **Cite concordances:** When comparing pre/post revision data, reference official concordance tables

---

## References

1. Statistics Canada. (2021). National Occupational Classification (NOC) 2021 Version 1.0.
2. Statistics Canada. (2022). North American Industry Classification System (NAICS) Canada 2022.
3. Statistics Canada. (2025). Guide de l'utilisateur des microdonnées - EPA FMGD.
4. Statistics Canada. (2024). Improvements to the Labour Force Survey (71F0031X).
