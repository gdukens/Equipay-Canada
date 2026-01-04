# EquiPay-Canada: Comprehensive Scientific Framework

## Theoretical & Methodological Foundations for Gender Pay Equity Analysis

**Version 2.0** | January 2026  
**Authors**: Research Team  
**Data Source**: Statistics Canada Labour Force Survey (LFS) PUMF, 2010-2025

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Economic Theories](#2-economic-theories)
3. [Sociological Frameworks](#3-sociological-frameworks)
4. [Statistical/Econometric Methods](#4-statisticaleconometric-methods)
5. [Demographic Methods](#5-demographic-methods)
6. [Variable-to-Theory Mapping](#6-variable-to-theory-mapping)
7. [Implementation Status](#7-implementation-status)
8. [Mathematical Appendix](#8-mathematical-appendix)
9. [Bibliography](#9-bibliography)

---

## 1. Executive Summary

This document provides the scientific foundation for the EquiPay-Canada gender pay equity analysis system. It integrates theories and methods from:

- **Economics**: Human capital, discrimination, labor market structure
- **Sociology**: Devaluation, segregation, intersectionality  
- **Statistics**: Decomposition methods, causal inference
- **Demography**: Life course, cohort analysis

### Key Theories Implemented

| Category | Theory | Status |
|----------|--------|--------|
| Human Capital | Mincerian Earnings Function | ✅ Implemented |
| Discrimination | Oaxaca-Blinder Decomposition | ✅ Implemented |
| Discrimination | Statistical Discrimination | ✅ Partial |
| Labor Structure | Monopsony | ⏳ Framework Ready |
| Sociological | Devaluation Hypothesis | ✅ Implemented |
| Distributional | RIF Decomposition (Firpo-Fortin-Lemieux) | ✅ Implemented |
| Distributional | DFL Reweighting | ✅ Implemented |
| Distributional | Machado-Mata Quantile Decomposition | ✅ Implemented |
| Causal | Propensity Score Matching | ✅ Implemented |
| Causal | Doubly Robust Estimation (AIPW) | ✅ Implemented |
| Causal | Heckman Selection Correction | ✅ Implemented |
| Causal | Staggered DiD (Callaway-Sant'Anna) | ✅ Implemented |
| Causal | Synthetic Control | ✅ Implemented |
| Occupational | Brown-Moon-Zoloth Decomposition | ✅ Implemented |
| Segregation | Duncan Index | ✅ Implemented |
| Segregation | Karmel-MacLachlan Index | ✅ Implemented |
| Segregation | Glass Ceiling Index | ✅ Implemented |
| Family | Motherhood Penalty | ✅ Implemented |
| Family | Kleven Child Penalty Event Study | ✅ Implemented |
| Demographic | Age-Period-Cohort (APC) Decomposition | ✅ Implemented |
| Intersectionality | Multiplicative Model | ✅ Implemented |
| Time Series | Structural Breaks (Bai-Perron, Zivot-Andrews) | ✅ Implemented |
| Time Series | Markov-Switching Regimes | ✅ Implemented |
| Time Series | Time-Varying Parameters (Kalman Filter) | ✅ Implemented |
| Time Series | VECM/Cointegration | ✅ Implemented |
| Convergence | Beta/Sigma/Club Convergence | ✅ Implemented |

---

## 2. Economic Theories

### 2.1 Human Capital Theory

**Source**: Becker (1964), Mincer (1974)

**Core Proposition**: Wage differences reflect productivity differences arising from investments in education, training, and experience.

**Mincerian Earnings Function**:

$$\ln(W_i) = \alpha + \beta_1 S_i + \beta_2 X_i + \beta_3 X_i^2 + \epsilon_i$$

Where:
- $S_i$ = years of schooling
- $X_i$ = years of experience
- $\beta_1$ = return to schooling (~8-12%)
- $\beta_2, \beta_3$ = experience profile (concave)

**LFS Variables**:
| Variable | Mapping | Notes |
|----------|---------|-------|
| `EDUC` | Education level (0-5) | Maps to years via lookup |
| `AGE_12` | Age categories | For experience proxy |
| `TENURE` | Job tenure | Firm-specific capital |

**Implementation** (`src/analysis.py`):
```python
df['EXPERIENCE_PROXY'] = df['AGE_APPROX'] - df['EDUC_YEARS'] - 6
model = sm.OLS.from_formula(
    'LOG_WAGE ~ EDUC + EXPERIENCE_PROXY + I(EXPERIENCE_PROXY**2) + TENURE',
    data=df
)
```

---

### 2.2 Discrimination Theories

#### 2.2.1 Taste-Based Discrimination (Becker 1957)

**Core Proposition**: Employers, co-workers, or customers have preferences against certain groups and are willing to pay a cost to avoid them.

**Discrimination Coefficient**:
$$W_F = MRP_F - d$$

Where $d$ = employer's discrimination coefficient (willingness to pay to avoid hiring women).

**Testable Implications**:
- Competitive markets should eliminate discriminating firms
- Gap should be smaller in competitive industries
- Gap varies with employer/customer demographic preferences

**LFS Test**:
```python
# Proxy: Gap variation by industry concentration
model = 'ln_wage ~ female * industry_hhi + controls'
```

#### 2.2.2 Statistical Discrimination (Phelps 1972, Arrow 1973)

**Core Proposition**: Employers use group membership as a signal when individual productivity is unobservable.

**Bayesian Updating Model**:

Initial assessment:
$$\hat{y}_i = \lambda \cdot z_i + (1-\lambda) \cdot \bar{y}_g$$

Where:
- $z_i$ = observable signal (education, test score)
- $\bar{y}_g$ = group average productivity
- $\lambda$ = signal precision

**With Learning (Altonji-Pierret 2001)**:

As tenure increases, $\lambda \to 1$ and group membership becomes irrelevant:
$$\hat{y}_{it} = \lambda_t \cdot y_i + (1-\lambda_t) \cdot \bar{y}_g$$

**Testable Implication**: Gender coefficient should decrease with tenure.

**LFS Test**:
```python
model = 'ln_wage ~ female * tenure + educ * tenure + controls'
# H0: Coefficient on female × tenure > 0 (gap closes with learning)
```

#### 2.2.3 Oaxaca-Blinder Decomposition (Oaxaca 1973, Blinder 1973)

**Core Method**: Decompose wage gap into explained (characteristics) and unexplained (returns/discrimination) components.

**Two-Fold Decomposition**:

$$\bar{W}_M - \bar{W}_F = \underbrace{(\bar{X}_M - \bar{X}_F)\hat{\beta}^*}_{\text{Explained (Endowments)}} + \underbrace{\bar{X}_M(\hat{\beta}_M - \hat{\beta}^*) + \bar{X}_F(\hat{\beta}^* - \hat{\beta}_F)}_{\text{Unexplained (Coefficients)}}$$

**Three-Fold (Cotton 1988)**:

$$\Delta\bar{W} = \underbrace{(\bar{X}_M - \bar{X}_F)\hat{\beta}^*}_{\text{Endowments}} + \underbrace{\bar{X}_M(\hat{\beta}_M - \hat{\beta}^*)}_{\text{Male Advantage}} + \underbrace{\bar{X}_F(\hat{\beta}^* - \hat{\beta}_F)}_{\text{Female Disadvantage}}$$

**Implementation Status**: ✅ Implemented in `src/models.py`

---

### 2.3 Labor Market Structure Theories

#### 2.3.1 Monopsony (Robinson 1933, Manning 2003)

**Core Proposition**: Employers have wage-setting power due to labor market frictions. Women face higher mobility costs.

**Optimal Wage Setting**:

$$w_g = MRP \cdot \frac{\varepsilon_g}{\varepsilon_g + 1}$$

If $\varepsilon_F < \varepsilon_M$ (women have lower labor supply elasticity):
$$\frac{w_F}{w_M} = \frac{\varepsilon_F(\varepsilon_M + 1)}{\varepsilon_M(\varepsilon_F + 1)} < 1$$

**Testable Implications**:
1. Gap larger in concentrated labor markets
2. Gap larger for married women (geographic constraints)
3. Gap varies inversely with job mobility

**LFS Variables**:
| Indicator | LFS Proxy |
|-----------|-----------|
| Market concentration | # employers by NAICS × CMA |
| Mobility constraint | MARSTAT × PROV |
| Labor supply elasticity | Tenure variation by gender |

**Implementation Status**: ❌ Not implemented

---

#### 2.3.2 Dual Labor Market Theory (Doeringer & Piore 1971)

**Core Proposition**: Labor market is segmented into primary (good jobs) and secondary (bad jobs) sectors.

| Primary Sector | Secondary Sector |
|----------------|------------------|
| High wages | Low wages |
| Job security | High turnover |
| Advancement opportunities | Dead-end jobs |
| Union coverage | Non-union |
| Full-time | Part-time/temporary |

**LFS Segmentation Indicators**:
```python
# Primary sector indicators
df['IS_PRIMARY'] = (
    (df['PERMTEMP'] == 1) &  # Permanent
    (df['FTPTMAIN'] == 1) &  # Full-time
    (df['UNION'] == 1) &      # Union
    (df['ESTSIZE'] >= 3)      # Medium/large establishment
)

df['IS_SECONDARY'] = ~df['IS_PRIMARY']
```

**Implementation Status**: ⚠️ Partial (IS_PRECARIOUS feature exists)

---

#### 2.3.3 Tournament Theory (Lazear & Rosen 1981)

**Core Proposition**: Compensation reflects prizes in promotion tournaments, not just marginal productivity.

**Optimal Effort**:
$$e^* = h'^{-1}\left(\frac{\Delta W}{2}\right)$$

Where $\Delta W = W_{winner} - W_{loser}$.

**Glass Ceiling as Biased Tournament**:
If women face lower $P(\text{win}|e)$ due to bias:
$$E[U_F] < E[U_M] \Rightarrow \text{lower effort or participation}$$

**Testable Implications**:
1. Gender gap increases at higher wage percentiles
2. Gap larger in firms with steep hierarchies
3. Women underrepresented in "up-or-out" tracks

**LFS Test**:
```python
# Quantile regression for tournament effects
for tau in [0.10, 0.25, 0.50, 0.75, 0.90]:
    qr = smf.quantreg('ln_wage ~ female + controls', df)
    result = qr.fit(q=tau)
    print(f"τ={tau}: Female coefficient = {result.params['female']}")
```

**Implementation Status**: ❌ Not implemented

---

#### 2.3.4 Efficiency Wage Theory (Shapiro & Stiglitz 1984)

**Core Proposition**: Firms pay above market-clearing wages to prevent shirking.

**No-Shirking Condition**:
$$w \geq \bar{w} + \frac{e(r + b + q)}{q}$$

Where:
- $e$ = effort cost
- $b$ = job separation rate
- $q$ = shirking detection probability

**Gender Implication**: If women expected to have higher $b$ (maternity, care), efficiency wage premium may differ.

---

#### 2.3.5 Compensating Differentials (Rosen 1986)

**Core Proposition**: Wages compensate for job disamenities; observed gaps may reflect preference-driven sorting.

**Hedonic Wage Function**:
$$w = f(a_1, a_2, ..., a_K) = \bar{w} - \sum_{k=1}^{K} \pi_k a_k$$

Where $\pi_k$ = implicit price of amenity $k$.

**Goldin (2014) Flexibility Framework**:

Convex earnings in hours:
$$w = \alpha \cdot h^\gamma$$

- $\gamma = 1$: Linear (pharmacy, tech)
- $\gamma > 1$: Convex/superlinear (law, finance)

If $\gamma > 1$ and women work fewer hours:
$$\frac{w_F}{w_M} = \left(\frac{h_F}{h_M}\right)^\gamma < \frac{h_F}{h_M}$$

**LFS Amenity Proxies**:
| Amenity | Variable |
|---------|----------|
| Flexibility | FTPTMAIN, UHRSMAIN |
| Security | PERMTEMP |
| Schedule | AHRSMAIN vs UHRSMAIN |
| Commute | CMA indicator |

**Implementation Status**: ⚠️ Partial (hours variables used)

---

### 2.4 Behavioral Economics

#### 2.4.1 Negotiation Aversion (Babcock & Laschever 2003)

**Finding**: Women negotiate less frequently and less aggressively.

**Expected Payoff Model**:
$$E[\pi_{\text{negotiate}}] = p \cdot \Delta w - (1-p) \cdot BC - NC$$

Where:
- $p$ = success probability
- $BC$ = backlash cost (higher for women)
- $NC$ = negotiation cost

**Implication**: Gap should be smaller where negotiation is constrained:
- Union jobs (collective bargaining)
- Public sector (transparent pay scales)
- Entry-level positions (standardized offers)

**LFS Test**:
```python
model = 'ln_wage ~ female * union + female * public_sector + controls'
# H0: Interactions should be positive (gap smaller with constraints)
```

#### 2.4.2 Risk Preferences (Croson & Gneezy 2009)

**Finding**: Women exhibit greater risk aversion on average.

**Certainty Equivalent**:
$$CE = E[w] - \frac{1}{2}\rho\sigma_w^2$$

If $\rho_F > \rho_M$:
- Women sort into stable, lower-variance jobs
- Lower expected wages but lower risk

**LFS Proxy**: Self-employment, variable-pay industries

---

## 3. Sociological Frameworks

### 3.1 Devaluation Hypothesis (England 1992, Levanon et al. 2009)

**Core Proposition**: Work performed predominantly by women is culturally devalued, leading to lower pay independent of skill requirements.

**Occupational Wage Model**:
$$\ln(W_j) = \alpha + \beta_1 \cdot \text{FemShare}_j + \beta_2 \cdot \text{SkillLevel}_j + \epsilon_j$$

$\beta_1 < 0$ even after controlling for skill → evidence of devaluation.

**Dynamic Test (Levanon et al.)**:
$$\Delta\ln(W_j) = \gamma \cdot \Delta\text{FemShare}_j + \text{controls}$$

As occupations feminize, wages decline (causal direction).

**LFS Implementation**:
```python
# Calculate female share by occupation
occ_fem_share = df.groupby('NOC_10').apply(
    lambda x: (x['GENDER'] == 2).mean()
)
df['FEM_SHARE_OCC'] = df['NOC_10'].map(occ_fem_share)

# Test devaluation
model = 'ln_wage ~ FEM_SHARE_OCC + educ + tenure + controls'
```

**Implementation Status**: ❌ Not implemented

---

### 3.2 Occupational Segregation Measures

#### 3.2.1 Duncan Dissimilarity Index (Duncan & Duncan 1955)

$$D = \frac{1}{2}\sum_{j=1}^{J}\left|\frac{F_j}{F} - \frac{M_j}{M}\right|$$

**Interpretation**: Proportion of women (or men) needing to change occupations for perfect integration.

#### 3.2.2 Karmel-MacLachlan Index

$$KM = \sum_j \left|a \cdot F_j - (1-a) \cdot M_j\right| / T$$

Where $a = M/T$.

#### 3.2.3 IP Index (Size-Standardized)

$$IP = \frac{1}{2}\sum_j \left|\frac{F_j}{T_j} - \frac{F}{T}\right| \cdot \frac{T_j}{T}$$

**Implementation**:
```python
def duncan_index(df, occ_col='NOC_10', gender_col='GENDER'):
    """Calculate Duncan Dissimilarity Index"""
    crosstab = pd.crosstab(df[occ_col], df[gender_col])
    f_share = crosstab[2] / crosstab[2].sum()
    m_share = crosstab[1] / crosstab[1].sum()
    return 0.5 * np.abs(f_share - m_share).sum()
```

**Implementation Status**: ❌ Not implemented

---

### 3.3 Brown-Moon-Zoloth Decomposition (1980)

**Purpose**: Decompose wage gap into occupational access + within-occupation components.

**Stage 1 - Occupational Attainment** (Multinomial Logit):
$$P(O_i = j | X_i) = \frac{\exp(\gamma_j X_i)}{\sum_k \exp(\gamma_k X_i)}$$

**Stage 2 - Within-Occupation Wages**:
$$\ln(W_{ij}) = \alpha_j + \beta_j X_i + \epsilon_{ij}$$

**Full Decomposition**:
$$\bar{W}_M - \bar{W}_F = \underbrace{\sum_j(\bar{P}_j^M - \bar{P}_j^F)\bar{W}_j}_{\text{Occupational}} + \underbrace{\sum_j \bar{P}_j^F (\bar{W}_j^M - \bar{W}_j^F)}_{\text{Within-Occupation}}$$

**Implementation Status**: ❌ Not implemented

---

### 3.4 Status Expectations Theory (Ridgeway 2011)

**Core Proposition**: Cultural beliefs create gendered status hierarchies affecting competence assessments.

**Double Standards Model**:
$$P(\text{Competent} | \text{Performance}, \text{Gender}) = f(\text{Performance} + \beta \cdot \text{Male})$$

With $\beta > 0$: Men receive competence attributions at lower performance thresholds.

**Burden of Proof**:
$$\theta_F > \theta_M$$

Women require higher evidence to demonstrate equal competence.

---

### 3.5 Organizational Inequality Regimes (Acker 2006)

**Dimensions**:
1. **Visibility**: How transparent are disparities?
2. **Legitimacy**: Are inequalities seen as fair?
3. **Control**: Mechanisms maintaining inequality
4. **Severity**: Steepness of disparities

**Gendered Organizations**: Jobs designed around "ideal worker" assumed male and unencumbered.

---

### 3.6 Intersectionality (Crenshaw 1989)

**Core Insight**: Multiple dimensions of identity interact, creating unique experiences not captured by single-axis analysis.

**Additive Model** (Baseline):
$$\ln(W_i) = \alpha + \beta_1 \cdot \text{Female} + \beta_2 \cdot \text{Immigrant} + X\gamma + \epsilon$$

**Multiplicative Model**:
$$\ln(W_i) = \alpha + \beta_1 \cdot \text{Female} + \beta_2 \cdot \text{Immigrant} + \beta_3 \cdot (\text{Female} \times \text{Immigrant}) + X\gamma + \epsilon$$

**Interpretation of $\beta_3$**:
- $\beta_3 < 0$: Amplification (double jeopardy)
- $\beta_3 > 0$: Cushioning
- $\beta_3 = 0$: Purely additive

**LFS Implementation**:
```python
# Intersectional indicators
df['IS_IMMIGRANT_FEMALE'] = df['IS_FEMALE'] & df['IS_IMMIGRANT']
df['IS_MOTHER_YOUNG_CHILD'] = df['IS_FEMALE'] & df['HAS_YOUNG_CHILDREN']

# Full factorial model
model = '''
ln_wage ~ C(GENDER)*C(IMMIG) + C(GENDER)*C(HAS_YOUNG_CHILDREN) + controls
'''
```

**Implementation Status**: ✅ Implemented (intersectional features exist)

---

## 4. Statistical/Econometric Methods

### 4.1 Distributional Decomposition

#### 4.1.1 RIF-Regression (Firpo, Fortin, Lemieux 2009, 2011)

**Recentered Influence Function** for quantile $Q_\tau$:

$$RIF(Y; Q_\tau) = Q_\tau + \frac{\tau - \mathbf{1}(Y \leq Q_\tau)}{f_Y(Q_\tau)}$$

**Unconditional Quantile Regression**:
$$E[RIF(Y; Q_\tau)|X] = X'\beta^\tau$$

**RIF-Oaxaca Decomposition**:
$$Q_\tau^M - Q_\tau^F = \underbrace{(\bar{X}_M - \bar{X}_F)'\hat{\beta}^\tau_F}_{\text{Composition}} + \underbrace{\bar{X}_M'(\hat{\beta}^\tau_M - \hat{\beta}^\tau_F)}_{\text{Wage Structure}}$$

**Glass Ceiling Test**: Compare $\beta_{female}$ at $\tau = 0.10$ vs $\tau = 0.90$.

**Implementation**:
```python
from econtools.quantile import RIF_regression

# Estimate at deciles
for tau in np.arange(0.1, 1.0, 0.1):
    rif = compute_rif(df['ln_wage'], tau)
    model = sm.OLS(rif, df[['IS_FEMALE'] + controls])
    results[tau] = model.fit().params['IS_FEMALE']

# Glass ceiling: gap at 90th > gap at 10th
```

**Implementation Status**: ❌ Not implemented

---

#### 4.1.2 DiNardo-Fortin-Lemieux Reweighting (1996)

**Counterfactual Density**:
$$f_Y^c(y) = \int f_{Y|X}^F(y|x) \cdot dF_X^M(x)$$

**Reweighting Function**:
$$\psi(X) = \frac{P(M|X)}{P(F|X)} \cdot \frac{N_F}{N_M}$$

**Implementation**:
```python
from sklearn.linear_model import LogisticRegression

# Estimate propensity score
ps_model = LogisticRegression()
ps_model.fit(X, df['GENDER'] == 1)  # P(Male)
propensity = ps_model.predict_proba(X)[:, 1]

# Reweighting factor for females
df.loc[df['GENDER'] == 2, 'dfl_weight'] = propensity / (1 - propensity)

# Counterfactual wage distribution
cf_wages = df[df['GENDER'] == 2]['HRLYEARN'] * df['dfl_weight']
```

**Implementation Status**: ❌ Not implemented

---

#### 4.1.3 Machado-Mata Decomposition (2005)

**Procedure**:
1. Estimate conditional quantile functions: $Q_\tau(Y|X) = X'\beta(\tau)$
2. Draw $\tau_i \sim U[0,1]$ and $X_i^*$ from male distribution
3. Construct counterfactual: $Y_i^c = X_i^{*'}\hat{\beta}_F(\tau_i)$

**Implementation Status**: ❌ Not implemented

---

### 4.2 Causal Inference Methods

#### 4.2.1 Propensity Score Matching (Rosenbaum & Rubin 1983)

**Propensity Score**:
$$e(X) = P(\text{Female} = 1 | X)$$

**ATT Estimator**:
$$\hat{\tau}_{ATT} = \frac{1}{N_F}\sum_{i: F_i=1}\left[W_i - \hat{W}_i^{(M)}\right]$$

Where $\hat{W}_i^{(M)}$ = matched male wage.

**Matching Methods**:
- Nearest neighbor
- Kernel matching
- Caliper matching

**Implementation**:
```python
from causalinference import CausalModel

model = CausalModel(
    Y=df['ln_wage'].values,
    D=(df['GENDER'] == 2).astype(int).values,
    X=df[controls].values
)
model.est_propensity()
model.stratify()
model.est_via_matching()
print(f"ATT: {model.estimates['matching']['att']}")
```

**Implementation Status**: ❌ Not implemented

---

#### 4.2.2 Inverse Probability Weighting (IPW)

**ATE Estimator**:
$$\hat{\tau}_{IPW} = \frac{1}{N}\sum_i\left[\frac{D_i Y_i}{\hat{e}(X_i)} - \frac{(1-D_i)Y_i}{1-\hat{e}(X_i)}\right]$$

**Normalized/Hajek Estimator** (more stable):
$$\hat{\tau}_{NIPW} = \frac{\sum_i D_i Y_i / \hat{e}(X_i)}{\sum_i D_i / \hat{e}(X_i)} - \frac{\sum_i (1-D_i)Y_i/(1-\hat{e}(X_i))}{\sum_i (1-D_i)/(1-\hat{e}(X_i))}$$

**Implementation Status**: ❌ Not implemented

---

#### 4.2.3 Doubly Robust Estimation (AIPW)

$$\hat{\tau}_{DR} = \frac{1}{N}\sum_i\left[\hat{\mu}_1(X_i) - \hat{\mu}_0(X_i) + \frac{D_i(Y_i - \hat{\mu}_1(X_i))}{\hat{e}(X_i)} - \frac{(1-D_i)(Y_i - \hat{\mu}_0(X_i))}{1-\hat{e}(X_i)}\right]$$

**Property**: Consistent if either propensity score OR outcome model is correctly specified.

**Implementation Status**: ❌ Not implemented

---

#### 4.2.4 Heckman Selection Correction (1979)

**Selection Equation**:
$$D_i^* = Z_i'\gamma + u_i, \quad D_i = \mathbf{1}(D_i^* > 0)$$

**Outcome Equation**:
$$W_i = X_i'\beta + \epsilon_i \quad (\text{observed only if } D_i = 1)$$

**Two-Step Procedure**:
1. Probit for selection: $\hat{\gamma}$
2. OLS with Inverse Mills Ratio:
$$W_i = X_i'\beta + \sigma_\epsilon\rho \cdot \lambda(Z_i'\hat{\gamma}) + \nu_i$$

Where $\lambda(z) = \phi(z)/\Phi(z)$.

**Application**: Correct for selection into employment (women with high unobserved wages more likely to work).

**Implementation Status**: ❌ Not implemented

---

### 4.3 Panel Data Methods

#### 4.3.1 Fixed Effects with Mundlak Correction

**Standard FE**:
$$Y_{it} = \alpha_i + X_{it}'\beta + \epsilon_{it}$$

**Mundlak (1978)**:
$$c_i = \bar{X}_i'\xi + a_i$$

**Combined Model**:
$$Y_{it} = X_{it}'\beta + \bar{X}_i'\xi + a_i + \epsilon_{it}$$

**Advantage**: Can estimate effects of time-invariant variables (like gender).

---

#### 4.3.2 Arellano-Bond GMM

**Dynamic Panel**:
$$Y_{it} = \rho Y_{i,t-1} + X_{it}'\beta + c_i + \epsilon_{it}$$

**First-Difference**:
$$\Delta Y_{it} = \rho\Delta Y_{i,t-1} + \Delta X_{it}'\beta + \Delta\epsilon_{it}$$

**Instruments**: $Y_{i,t-2}, Y_{i,t-3}, ...$ (lagged levels)

**Application**: Wage dynamics, persistence of gender gaps.

---

## 5. Demographic Methods

### 5.1 Age-Period-Cohort (APC) Analysis

**Identification Problem**: Age + Birth Year = Survey Year (perfect collinearity)

**APC Model**:
$$Y_{apc} = \mu + \alpha_a + \pi_p + \gamma_c + \epsilon$$

**Solutions**:
1. **Intrinsic Estimator**: Constrained estimation
2. **HAPC** (Yang-Land): Hierarchical random effects
3. **Detrending**: Assign linear trend theoretically

**Gender Gap APC**:
$$\text{Gap}_{apc} = (\alpha_a^M - \alpha_a^F) + (\pi_p^M - \pi_p^F) + (\gamma_c^M - \gamma_c^F)$$

---

### 5.2 Child Penalty (Kleven et al. 2019)

**Event Study Design**:
$$Y_{it}^g = \sum_{j \neq -1} \alpha_j^g \cdot \mathbf{1}[t - t_i^* = j] + \text{age FE} + \text{year FE} + \epsilon_{it}^g$$

**Child Penalty**:
$$P_j = \frac{\hat{\alpha}_j^F - \hat{\alpha}_j^M}{\text{counterfactual}_j^M}$$

**Components**:
$$P = P_{\text{participation}} + P_{\text{hours}} + P_{\text{wage rate}} + P_{\text{occupation}}$$

**LFS Limitation**: No direct child birth timing; must proxy with AGYOWNK (age of youngest child).

---

### 5.3 Life Course Perspective

**Trajectory Analysis**:
$$S_t = f(S_{t-1}, X_t, E_t, \epsilon_t)$$

Where:
- $S_t$ = state (employment, occupation, wage)
- $E_t$ = life events (birth, marriage)

**Sequence Analysis**: Optimal matching distance between career paths.

---

## 6. Variable-to-Theory Mapping

### 6.1 LFS Raw Variables (27 columns)

| Variable | Theory | Mechanism | Implementation |
|----------|--------|-----------|----------------|
| **GENDER** | All discrimination theories | Direct protected attribute | ✅ `IS_FEMALE` |
| **AGE_12** | Human capital | Experience accumulation | ✅ `AGE_APPROX`, `EXPERIENCE_PROXY` |
| **EDUC** | Human capital | Schooling investment | ✅ `HAS_DEGREE` |
| **NOC_10/43** | Segregation, Devaluation | Occupational sorting | ✅ Categorical |
| **NAICS_21** | Dual labor market | Industry segmentation | ✅ Categorical |
| **PROV** | Spatial wage theory | Regional markets | ✅ Categorical |
| **TENURE** | Firm-specific capital, Learning | Tenure-wage profile | ✅ Continuous |
| **HRLYEARN** | Dependent variable | Mincerian outcome | ✅ `LOG_HRLYEARN` |
| **UHRSMAIN** | Labor supply, Flexibility | Hours preferences | ✅ Used |
| **AHRSMAIN** | Work intensity | Actual vs. desired | ✅ `HOURS_GAP` |
| **FTPTMAIN** | Motherhood, Flexibility | Part-time penalty | ✅ `IS_FULLTIME` |
| **PERMTEMP** | Dual labor market | Job security | ✅ `IS_PERMANENT` |
| **UNION** | Rent-sharing, Negotiation | Union premium | ✅ `IS_UNION` |
| **ESTSIZE** | Employer size effect | Size-wage premium | ✅ Categorical |
| **MARSTAT** | Marriage effects | Premium/penalty | ✅ `IS_MARRIED` |
| **IMMIG** | Immigration economics | Credential recognition | ✅ `IS_IMMIGRANT` |
| **COWMAIN** | Segmentation | Public/private/self | ✅ `IS_PUBLIC_SECTOR` |

### 6.2 Derived Features (50+)

| Feature | Source Variables | Theory | Formula |
|---------|-----------------|--------|---------|
| `IS_FEMALE` | GENDER | All discrimination | GENDER == 2 |
| `EXPERIENCE_PROXY` | AGE, EDUC | Human capital | AGE - EDUC_YEARS - 6 |
| `HAS_DEGREE` | EDUC | Human capital | EDUC >= 4 |
| `IS_FULLTIME` | FTPTMAIN | Labor supply | FTPTMAIN == 1 |
| `IS_PERMANENT` | PERMTEMP | Dual market | PERMTEMP == 1 |
| `IS_UNION` | UNION | Rent-sharing | UNION == 1 |
| `IS_IMMIGRANT` | IMMIG | Immigration | IMMIG != 1 |
| `IS_PUBLIC_SECTOR` | COWMAIN | Segmentation | COWMAIN == 1 |
| `HAS_YOUNG_CHILDREN` | AGYOWNK | Motherhood | AGYOWNK ∈ {1,2,3} |
| `IS_MOTHER_YOUNG_CHILD` | GENDER, AGYOWNK | Motherhood penalty | IS_FEMALE & HAS_YOUNG_CHILDREN |
| `IS_PRECARIOUS` | Multiple | Dual market | Temp & PT & low hours |
| `HOURS_GAP` | AHRSMAIN, UHRSMAIN | Flexibility | AHRSMAIN - UHRSMAIN |
| `IS_IMMIGRANT_FEMALE` | IMMIG, GENDER | Intersectionality | IS_IMMIGRANT & IS_FEMALE |

---

## 7. Implementation Status

### 7.1 Currently Implemented ✅

| Method/Feature | Module | Function |
|----------------|--------|----------|
| Mincerian wage equation | `analysis.py` | `compute_adjusted_wage_gap()` |
| Oaxaca-Blinder decomposition | `models.py` | `WageGapModel` |
| Raw wage gap | `analysis.py` | `compute_raw_wage_gap()` |
| Motherhood penalty analysis | `analysis.py` | `motherhood_penalty_analysis()` |
| Intersectional analysis | `analysis.py` | `intersectional_analysis()` |
| Fairness metrics | `fairness.py` | `FairnessAnalyzer` |
| Survey weighting | `analysis.py` | `FINALWT` integration |

### 7.2 Partially Implemented ⚠️

| Feature | Status | Gap |
|---------|--------|-----|
| Quantile regression | Basic | No RIF decomposition |
| Panel methods | Time-series only | No individual FE |
| Selection correction | None | Need Heckman |
| Dual labor market | `IS_PRECARIOUS` only | No formal segmentation model |

### 7.3 Not Implemented ❌

| Method | Priority | Complexity |
|--------|----------|------------|
| **RIF Decomposition** | 🔴 HIGH | Medium |
| **Propensity Score Matching** | 🔴 HIGH | Low |
| **Quantile Regression (distributional)** | 🔴 HIGH | Medium |
| **DFL Reweighting** | 🟡 MEDIUM | Medium |
| **Heckman Selection** | 🟡 MEDIUM | Medium |
| **Brown-Moon-Zoloth** | 🟡 MEDIUM | High |
| **Doubly Robust Estimation** | 🟡 MEDIUM | Medium |
| **Duncan Segregation Index** | 🟢 LOW | Low |
| **Devaluation Model** | 🟢 LOW | Low |
| **APC Decomposition** | 🟢 LOW | High |
| **Kleven Child Penalty** | 🟢 LOW | High |

---

## 8. Mathematical Appendix

### 8.1 Key Formulas Reference

#### Oaxaca-Blinder Two-Fold
$$\Delta\bar{W} = (\bar{X}_M - \bar{X}_F)\hat{\beta}^* + \bar{X}_F(\hat{\beta}_M - \hat{\beta}_F)$$

#### RIF for Quantile
$$RIF(Y; Q_\tau) = Q_\tau + \frac{\tau - \mathbf{1}(Y \leq Q_\tau)}{f_Y(Q_\tau)}$$

#### Propensity Score
$$e(X) = P(D=1|X) = \frac{\exp(X'\gamma)}{1 + \exp(X'\gamma)}$$

#### Inverse Mills Ratio
$$\lambda(z) = \frac{\phi(z)}{\Phi(z)}$$

#### Duncan Index
$$D = \frac{1}{2}\sum_j \left|\frac{F_j}{F} - \frac{M_j}{M}\right|$$

#### Child Penalty
$$P_j = \frac{\hat{\alpha}_j^F - \hat{\alpha}_j^M}{\hat{Y}_{j,\text{counterfactual}}^M}$$

---

## 9. Bibliography

### Economics

- Altonji, J. G., & Pierret, C. R. (2001). Employer learning and statistical discrimination. *Quarterly Journal of Economics*, 116(1), 313-350.
- Arrow, K. J. (1973). The theory of discrimination. In O. Ashenfelter & A. Rees (Eds.), *Discrimination in Labor Markets*.
- Babcock, L., & Laschever, S. (2003). *Women Don't Ask: Negotiation and the Gender Divide*. Princeton University Press.
- Becker, G. S. (1957). *The Economics of Discrimination*. University of Chicago Press.
- Becker, G. S. (1964). *Human Capital*. University of Chicago Press.
- Blinder, A. S. (1973). Wage discrimination: Reduced form and structural estimates. *Journal of Human Resources*, 8(4), 436-455.
- Card, D. (2001). The effect of unions on wage inequality. *ILR Review*, 54(2), 296-315.
- Chiswick, B. R. (1978). The effect of Americanization on the earnings of foreign-born men. *Journal of Political Economy*, 86(5), 897-921.
- Croson, R., & Gneezy, U. (2009). Gender differences in preferences. *Journal of Economic Literature*, 47(2), 448-474.
- Doeringer, P. B., & Piore, M. J. (1971). *Internal Labor Markets and Manpower Analysis*. Lexington.
- Freeman, R. B., & Medoff, J. L. (1984). *What Do Unions Do?* Basic Books.
- Goldin, C. (2014). A grand gender convergence: Its last chapter. *American Economic Review*, 104(4), 1091-1119.
- Lazear, E. P., & Rosen, S. (1981). Rank-order tournaments as optimum labor contracts. *Journal of Political Economy*, 89(5), 841-864.
- Manning, A. (2003). *Monopsony in Motion: Imperfect Competition in Labor Markets*. Princeton University Press.
- Mincer, J. (1974). *Schooling, Experience, and Earnings*. NBER.
- Oaxaca, R. (1973). Male-female wage differentials in urban labor markets. *International Economic Review*, 14(3), 693-709.
- Phelps, E. S. (1972). The statistical theory of racism and sexism. *American Economic Review*, 62(4), 659-661.
- Rosen, S. (1986). The theory of equalizing differences. In O. Ashenfelter & R. Layard (Eds.), *Handbook of Labor Economics*, Vol. 1.
- Shapiro, C., & Stiglitz, J. E. (1984). Equilibrium unemployment as a worker discipline device. *American Economic Review*, 74(3), 433-444.

### Sociology

- Acker, J. (2006). Inequality regimes: Gender, class, and race in organizations. *Gender & Society*, 20(4), 441-464.
- Bergmann, B. R. (1974). Occupational segregation, wages and profits when employers discriminate by race or sex. *Eastern Economic Journal*, 1(2), 103-110.
- Crenshaw, K. (1989). Demarginalizing the intersection of race and sex. *University of Chicago Legal Forum*, 1989(1), 139-167.
- England, P. (1992). *Comparable Worth: Theories and Evidence*. Aldine de Gruyter.
- Levanon, A., England, P., & Allison, P. (2009). Occupational feminization and pay. *Social Forces*, 88(2), 865-891.
- McCall, L. (2005). The complexity of intersectionality. *Signs*, 30(3), 1771-1800.
- Petersen, T., & Morgan, L. A. (1995). Separate and unequal: Occupation-establishment sex segregation. *American Journal of Sociology*, 101(2), 329-365.
- Ridgeway, C. L. (2011). *Framed by Gender*. Oxford University Press.
- Williams, C. L. (1992). The glass escalator: Hidden advantages for men in female professions. *Social Problems*, 39(3), 253-267.

### Econometrics

- Arellano, M., & Bond, S. (1991). Some tests of specification for panel data. *Review of Economic Studies*, 58(2), 277-297.
- Callaway, B., & Sant'Anna, P. H. (2021). Difference-in-differences with multiple time periods. *Journal of Econometrics*, 225(2), 200-230.
- Chernozhukov, V., et al. (2018). Double/debiased machine learning for treatment and structural parameters. *Econometrics Journal*, 21(1), C1-C68.
- DiNardo, J., Fortin, N. M., & Lemieux, T. (1996). Labor market institutions and the distribution of wages. *Econometrica*, 64(5), 1001-1044.
- Firpo, S., Fortin, N. M., & Lemieux, T. (2009). Unconditional quantile regressions. *Econometrica*, 77(3), 953-973.
- Fortin, N., Lemieux, T., & Firpo, S. (2011). Decomposition methods in economics. *Handbook of Labor Economics*, 4A, 1-102.
- Heckman, J. J. (1979). Sample selection bias as a specification error. *Econometrica*, 47(1), 153-161.
- Koenker, R., & Bassett, G. (1978). Regression quantiles. *Econometrica*, 46(1), 33-50.
- Machado, J. A., & Mata, J. (2005). Counterfactual decomposition of changes in wage distributions. *Journal of Applied Econometrics*, 20(4), 445-465.
- Mundlak, Y. (1978). On the pooling of time series and cross section data. *Econometrica*, 46(1), 69-85.
- Rosenbaum, P. R., & Rubin, D. B. (1983). The central role of the propensity score. *Biometrika*, 70(1), 41-55.

### Family & Demography

- Brown, R. S., Moon, M., & Zoloth, B. S. (1980). Incorporating occupational attainment in studies of male-female earnings differentials. *Journal of Human Resources*, 15(1), 3-28.
- Budig, M. J., & England, P. (2001). The wage penalty for motherhood. *American Sociological Review*, 66(2), 204-225.
- Correll, S. J., Benard, S., & Paik, I. (2007). Getting a job: Is there a motherhood penalty? *American Journal of Sociology*, 112(5), 1297-1339.
- Duncan, O. D., & Duncan, B. (1955). A methodological analysis of segregation indexes. *American Sociological Review*, 20(2), 210-217.
- Kleven, H., Landais, C., & Søgaard, J. E. (2019). Children and gender inequality: Evidence from Denmark. *American Economic Journal: Applied Economics*, 11(4), 181-209.
- Waldfogel, J. (1997). The effect of children on women's wages. *American Sociological Review*, 62(2), 209-217.
- Yang, Y., & Land, K. C. (2006). A mixed models approach to the age-period-cohort analysis. *Sociological Methods & Research*, 34(3), 374-398.

---

*Document generated: January 2026*  
*Last updated: v2.0*
