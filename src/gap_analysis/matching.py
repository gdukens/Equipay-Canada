"""
EquiPay Canada - Propensity Score Matching
==========================================

Implements propensity score methods for causal inference in wage gap analysis.

These methods address selection-on-observables by creating comparable
groups of male and female workers based on their characteristics.

Methods Implemented:
--------------------
1. Propensity Score Matching (PSM)
2. Inverse Probability Weighting (IPW)
3. Doubly Robust Estimation (AIPW)
4. Covariate Matching (Mahalanobis)

Key References:
--------------
- Rosenbaum & Rubin (1983). Central Role of Propensity Score
- Imbens (2004). Nonparametric Estimation of Average Treatment Effects
- Bang & Robins (2005). Doubly Robust Estimation

Author: EquiPay Canada Research Team
Version: 1.0.0
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class MatchingResult:
    """Results from propensity score matching."""
    
    # Treatment effect estimates
    ate: float              # Average Treatment Effect
    att: float              # Average Treatment on Treated
    atc: float              # Average Treatment on Control
    
    # Standard errors
    ate_se: float
    att_se: float
    
    # Confidence intervals
    ate_ci: Tuple[float, float]
    att_ci: Tuple[float, float]
    
    # Sample information
    n_treated: int
    n_control: int
    n_matched: int
    
    # Match quality
    mean_propensity_treated: float
    mean_propensity_control: float
    caliper_used: float
    
    # Balance diagnostics
    balance_before: pd.DataFrame
    balance_after: pd.DataFrame
    
    # Interpretation for wage gap
    gender_gap: float
    gender_gap_se: float
    
    def __repr__(self):
        return (
            f"MatchingResult(\n"
            f"  Gender Gap (ATT): {self.gender_gap:.1%} ± {1.96*self.gender_gap_se:.1%}\n"
            f"  ATE: {self.ate:.4f} ± {1.96*self.ate_se:.4f}\n"
            f"  Matched: {self.n_matched:,} pairs from {self.n_treated:,} treated\n"
            f")"
        )


@dataclass
class IPWResult:
    """Results from Inverse Probability Weighting."""
    
    # Weighted estimates
    ate: float
    ate_se: float
    ate_ci: Tuple[float, float]
    
    # Propensity model fit
    propensity_auc: float
    
    # Weight diagnostics
    max_weight: float
    mean_weight_treated: float
    mean_weight_control: float
    effective_sample_size: float
    
    # Trimming applied
    trimming_pct: float
    
    # Gender gap interpretation
    gender_gap: float
    gender_gap_se: float


@dataclass
class DoublyRobustResult:
    """Results from Augmented IPW (Doubly Robust)."""
    
    ate: float
    ate_se: float
    ate_ci: Tuple[float, float]
    
    # Component estimates
    ipw_component: float
    regression_component: float
    
    # Model diagnostics
    propensity_auc: float
    outcome_r2: float
    
    # Gender gap
    gender_gap: float
    gender_gap_se: float


# =============================================================================
# PROPENSITY SCORE ESTIMATION
# =============================================================================

def estimate_propensity_scores(
    df: pd.DataFrame,
    treatment_var: str,
    covariates: List[str],
    weight_col: str = None,
    method: str = 'logit'
) -> Tuple[np.ndarray, float]:
    """
    Estimate propensity scores P(T=1|X).
    
    Parameters
    ----------
    df : DataFrame
        Data with treatment and covariates
    treatment_var : str
        Binary treatment indicator
    covariates : list
        Control variables for propensity model
    weight_col : str
        Optional survey weights
    method : str
        'logit' or 'probit'
        
    Returns
    -------
    propensity : array
        Estimated propensity scores
    auc : float
        AUC of propensity model
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    
    X = df[covariates].values
    t = df[treatment_var].values
    
    # Handle missing values
    valid = ~(np.isnan(X).any(axis=1) | np.isnan(t))
    X = X[valid]
    t = t[valid]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit logistic regression
    weights = df[weight_col].values[valid] if weight_col else None
    
    model = LogisticRegression(
        penalty='l2',
        C=1.0,
        solver='lbfgs',
        max_iter=1000,
        random_state=42
    )
    
    model.fit(X_scaled, t, sample_weight=weights)
    
    # Predict propensity
    propensity_valid = model.predict_proba(X_scaled)[:, 1]
    
    # Full propensity array with NaN for invalid
    propensity = np.full(len(df), np.nan)
    propensity[valid] = propensity_valid
    
    # AUC
    auc = roc_auc_score(t, propensity_valid, sample_weight=weights)
    
    logger.info(f"Propensity scores estimated. AUC: {auc:.4f}")
    
    return propensity, auc


# =============================================================================
# MATCHING METHODS
# =============================================================================

class PropensityScoreMatching:
    """
    Propensity Score Matching for Gender Wage Gap.
    
    Matches female workers to similar male workers based on
    propensity scores, then compares wages within matched pairs.
    
    Parameters
    ----------
    caliper : float
        Maximum distance for matches (in std of propensity)
    n_matches : int
        Number of matches per treated unit (1 = pair matching)
    replacement : bool
        Whether to match with replacement
    weight_col : str
        Survey weight column
        
    Examples
    --------
    >>> psm = PropensityScoreMatching(caliper=0.2)
    >>> result = psm.fit(
    ...     df=data,
    ...     outcome_var='LOG_HRLYEARN',
    ...     treatment_var='IS_FEMALE',
    ...     covariates=['AGE', 'EDUC', 'EXP', 'PROV', 'NOC_10']
    ... )
    >>> print(f"Gender gap (matched): {result.gender_gap:.1%}")
    """
    
    def __init__(
        self,
        caliper: float = 0.2,
        n_matches: int = 1,
        replacement: bool = True,
        weight_col: str = 'FINALWT'
    ):
        self.caliper = caliper
        self.n_matches = n_matches
        self.replacement = replacement
        self.weight_col = weight_col
        self.result_ = None
        
    def fit(
        self,
        df: pd.DataFrame,
        outcome_var: str,
        treatment_var: str,
        covariates: List[str]
    ) -> MatchingResult:
        """
        Perform propensity score matching.
        
        Parameters
        ----------
        df : DataFrame
            Data with outcomes and covariates
        outcome_var : str
            Wage variable (log wages recommended)
        treatment_var : str
            Binary treatment (IS_FEMALE for gender gap)
        covariates : list
            Variables to match on
            
        Returns
        -------
        MatchingResult
            Matching estimates and diagnostics
        """
        logger.info("Performing propensity score matching")
        
        # Clean data
        required = [outcome_var, treatment_var] + covariates
        if self.weight_col in df.columns:
            required.append(self.weight_col)
        df_clean = df.dropna(subset=required).copy()
        
        # Estimate propensity scores
        propensity, auc = estimate_propensity_scores(
            df_clean, treatment_var, covariates, self.weight_col
        )
        df_clean['propensity'] = propensity
        
        # Calculate caliper in absolute terms
        ps_std = np.nanstd(propensity)
        caliper_abs = self.caliper * ps_std
        
        # Separate treated and control
        treated = df_clean[df_clean[treatment_var] == 1].copy()
        control = df_clean[df_clean[treatment_var] == 0].copy()
        
        n_treated = len(treated)
        n_control = len(control)
        
        logger.info(f"Matching {n_treated} treated to {n_control} controls")
        
        # Perform matching
        ps_treated = treated['propensity'].values
        ps_control = control['propensity'].values
        
        # Distance matrix
        distances = np.abs(ps_treated.reshape(-1, 1) - ps_control.reshape(1, -1))
        
        matched_pairs = []
        control_matched = np.zeros(n_control, dtype=bool)
        
        # Sort treated by propensity (random order within caliper)
        order = np.argsort(ps_treated)[::-1]  # High to low
        
        for i in order:
            dists = distances[i, :]
            
            if not self.replacement:
                dists[control_matched] = np.inf
            
            # Find matches within caliper
            candidates = np.where(dists <= caliper_abs)[0]
            
            if len(candidates) > 0:
                # Select closest match(es)
                n_select = min(self.n_matches, len(candidates))
                matches = candidates[np.argsort(dists[candidates])[:n_select]]
                
                for j in matches:
                    matched_pairs.append((i, j))
                    control_matched[j] = True
        
        n_matched = len(matched_pairs)
        logger.info(f"Created {n_matched} matched pairs")
        
        if n_matched == 0:
            raise ValueError("No matches found. Consider relaxing caliper.")
        
        # Calculate ATT
        outcomes_t = treated[outcome_var].values
        outcomes_c = control[outcome_var].values
        
        y1 = np.array([outcomes_t[i] for i, j in matched_pairs])
        y0 = np.array([outcomes_c[j] for i, j in matched_pairs])
        
        diffs = y1 - y0
        att = np.mean(diffs)
        att_se = np.std(diffs) / np.sqrt(n_matched)
        
        # ATE (approximate via weighting)
        ate = att  # With replacement, ATT ≈ ATE for matched sample
        ate_se = att_se
        
        # Confidence intervals
        ate_ci = (ate - 1.96 * ate_se, ate + 1.96 * ate_se)
        att_ci = (att - 1.96 * att_se, att + 1.96 * att_se)
        
        # Balance diagnostics
        balance_before = self._calculate_balance(
            treated[covariates], control[covariates]
        )
        
        # Matched samples
        matched_treated_idx = [i for i, j in matched_pairs]
        matched_control_idx = [j for i, j in matched_pairs]
        
        treated_matched = treated.iloc[matched_treated_idx]
        control_matched_df = control.iloc[matched_control_idx]
        
        balance_after = self._calculate_balance(
            treated_matched[covariates], control_matched_df[covariates]
        )
        
        # Gender gap interpretation
        # ATT on log scale = approximate % gap
        gender_gap = np.exp(att) - 1  # Convert log difference to %
        gender_gap_se = np.exp(att) * att_se  # Delta method approximation
        
        self.result_ = MatchingResult(
            ate=ate,
            att=att,
            atc=att,  # Approximate
            ate_se=ate_se,
            att_se=att_se,
            ate_ci=ate_ci,
            att_ci=att_ci,
            n_treated=n_treated,
            n_control=n_control,
            n_matched=n_matched,
            mean_propensity_treated=np.mean(ps_treated),
            mean_propensity_control=np.mean(ps_control),
            caliper_used=caliper_abs,
            balance_before=balance_before,
            balance_after=balance_after,
            gender_gap=att,  # Log scale = approx % for small values
            gender_gap_se=att_se
        )
        
        return self.result_
    
    def _calculate_balance(
        self,
        treated: pd.DataFrame,
        control: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate standardized mean differences."""
        balance = []
        
        for col in treated.columns:
            t_mean = treated[col].mean()
            c_mean = control[col].mean()
            
            # Pooled std
            t_std = treated[col].std()
            c_std = control[col].std()
            pooled_std = np.sqrt((t_std**2 + c_std**2) / 2)
            
            if pooled_std > 0:
                smd = (t_mean - c_mean) / pooled_std
            else:
                smd = 0
            
            balance.append({
                'Variable': col,
                'Mean Treated': t_mean,
                'Mean Control': c_mean,
                'SMD': smd,
                'Balanced': abs(smd) < 0.1
            })
        
        return pd.DataFrame(balance)


class InverseProbabilityWeighting:
    """
    Inverse Probability Weighting for causal inference.
    
    Reweights observations to create balance between treated
    and control groups.
    
    Parameters
    ----------
    trim_pct : float
        Trim propensity scores at this percentile to avoid
        extreme weights (default: 0.01 = 1st and 99th percentile)
    weight_col : str
        Survey weight column
    """
    
    def __init__(self, trim_pct: float = 0.01, weight_col: str = 'FINALWT'):
        self.trim_pct = trim_pct
        self.weight_col = weight_col
        self.result_ = None
        
    def fit(
        self,
        df: pd.DataFrame,
        outcome_var: str,
        treatment_var: str,
        covariates: List[str]
    ) -> IPWResult:
        """
        Estimate ATE using inverse probability weighting.
        """
        logger.info("Performing inverse probability weighting")
        
        # Clean data
        required = [outcome_var, treatment_var] + covariates
        df_clean = df.dropna(subset=required).copy()
        
        # Estimate propensity
        propensity, auc = estimate_propensity_scores(
            df_clean, treatment_var, covariates, self.weight_col
        )
        
        # Trim propensities
        lower = np.nanpercentile(propensity, self.trim_pct * 100)
        upper = np.nanpercentile(propensity, (1 - self.trim_pct) * 100)
        propensity_trimmed = np.clip(propensity, lower, upper)
        
        pct_trimmed = np.mean((propensity < lower) | (propensity > upper))
        
        # Calculate IPW weights
        t = df_clean[treatment_var].values
        y = df_clean[outcome_var].values
        
        # ATE weights
        w_ate = t / propensity_trimmed + (1 - t) / (1 - propensity_trimmed)
        
        # Combine with survey weights if available
        if self.weight_col in df_clean.columns:
            survey_weights = df_clean[self.weight_col].values
            w_ate = w_ate * survey_weights
        
        # Normalize weights
        w_ate = w_ate / np.nanmean(w_ate)
        
        # Weighted means
        y1_weighted = np.nansum(t * w_ate * y) / np.nansum(t * w_ate)
        y0_weighted = np.nansum((1-t) * w_ate * y) / np.nansum((1-t) * w_ate)
        
        ate = y1_weighted - y0_weighted
        
        # Variance via influence function
        # Approximate SE
        n = len(y)
        resid = t * (y - y1_weighted) / propensity_trimmed - \
                (1-t) * (y - y0_weighted) / (1 - propensity_trimmed)
        ate_se = np.std(resid) / np.sqrt(n)
        
        # Effective sample size
        ess = np.sum(w_ate)**2 / np.sum(w_ate**2)
        
        # Gender gap interpretation
        gender_gap = ate
        
        self.result_ = IPWResult(
            ate=ate,
            ate_se=ate_se,
            ate_ci=(ate - 1.96*ate_se, ate + 1.96*ate_se),
            propensity_auc=auc,
            max_weight=np.max(w_ate),
            mean_weight_treated=np.mean(w_ate[t == 1]),
            mean_weight_control=np.mean(w_ate[t == 0]),
            effective_sample_size=ess,
            trimming_pct=pct_trimmed,
            gender_gap=gender_gap,
            gender_gap_se=ate_se
        )
        
        return self.result_


class DoublyRobust:
    """
    Augmented Inverse Probability Weighting (Doubly Robust).
    
    Combines propensity weighting with outcome modeling.
    Consistent if either propensity OR outcome model is correct.
    
    Parameters
    ----------
    weight_col : str
        Survey weight column
    """
    
    def __init__(self, weight_col: str = 'FINALWT'):
        self.weight_col = weight_col
        self.result_ = None
        
    def fit(
        self,
        df: pd.DataFrame,
        outcome_var: str,
        treatment_var: str,
        covariates: List[str]
    ) -> DoublyRobustResult:
        """
        Estimate ATE using doubly robust estimation.
        """
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        
        logger.info("Performing doubly robust estimation")
        
        # Clean data
        required = [outcome_var, treatment_var] + covariates
        df_clean = df.dropna(subset=required).copy()
        
        X = df_clean[covariates].values
        t = df_clean[treatment_var].values
        y = df_clean[outcome_var].values
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Estimate propensity scores
        propensity, auc = estimate_propensity_scores(
            df_clean, treatment_var, covariates
        )
        
        # Trim propensities
        propensity = np.clip(propensity, 0.01, 0.99)
        
        # Outcome models (separate for treated/control)
        treated_mask = t == 1
        control_mask = t == 0
        
        model_1 = Ridge(alpha=1.0)
        model_0 = Ridge(alpha=1.0)
        
        model_1.fit(X_scaled[treated_mask], y[treated_mask])
        model_0.fit(X_scaled[control_mask], y[control_mask])
        
        # Predict counterfactual outcomes
        mu_1 = model_1.predict(X_scaled)
        mu_0 = model_0.predict(X_scaled)
        
        # Outcome model R²
        y_pred = np.where(t == 1, mu_1, mu_0)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - y.mean())**2)
        outcome_r2 = 1 - ss_res / ss_tot
        
        # AIPW estimator
        n = len(y)
        
        # E[Y(1)]
        aipw_1 = np.mean(
            mu_1 + t * (y - mu_1) / propensity
        )
        
        # E[Y(0)]
        aipw_0 = np.mean(
            mu_0 + (1 - t) * (y - mu_0) / (1 - propensity)
        )
        
        ate = aipw_1 - aipw_0
        
        # Influence function for variance
        psi = (
            mu_1 - mu_0 +
            t * (y - mu_1) / propensity -
            (1 - t) * (y - mu_0) / (1 - propensity) -
            ate
        )
        
        ate_se = np.std(psi) / np.sqrt(n)
        
        # Component decomposition
        ipw_only = np.mean(t * y / propensity - (1-t) * y / (1-propensity))
        reg_only = np.mean(mu_1 - mu_0)
        
        self.result_ = DoublyRobustResult(
            ate=ate,
            ate_se=ate_se,
            ate_ci=(ate - 1.96*ate_se, ate + 1.96*ate_se),
            ipw_component=ipw_only,
            regression_component=reg_only,
            propensity_auc=auc,
            outcome_r2=outcome_r2,
            gender_gap=ate,
            gender_gap_se=ate_se
        )
        
        return self.result_


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def run_matching_gender_gap(
    df: pd.DataFrame,
    outcome_var: str = 'LOG_HRLYEARN',
    method: str = 'psm',
    weight_col: str = 'FINALWT'
) -> Union[MatchingResult, IPWResult, DoublyRobustResult]:
    """
    Run propensity-based gender gap analysis with standard controls.
    
    Parameters
    ----------
    df : DataFrame
        LFS PUMF data
    outcome_var : str
        Log wage variable
    method : str
        'psm' = Propensity Score Matching
        'ipw' = Inverse Probability Weighting
        'dr' = Doubly Robust
    weight_col : str
        Survey weight column
        
    Returns
    -------
    Result object with gender gap estimate
    """
    # Standard covariates
    covariates = []
    
    # Human capital
    for col in ['AGE', 'AGE_SQ', 'EDUC_YEARS', 'EXPERIENCE_PROXY']:
        if col in df.columns:
            covariates.append(col)
    
    # Job characteristics
    for col in ['PROV', 'NOC_10', 'NAICS_21', 'UNION', 'FTPTMAIN', 'COWMAIN']:
        if col in df.columns:
            covariates.append(col)
    
    # Other demographics
    for col in ['IMMIG', 'MARSTAT']:
        if col in df.columns:
            covariates.append(col)
    
    if len(covariates) < 3:
        raise ValueError(f"Insufficient covariates. Found: {covariates}")
    
    logger.info(f"Using {len(covariates)} covariates: {covariates}")
    
    # Choose method
    if method.lower() == 'psm':
        matcher = PropensityScoreMatching(weight_col=weight_col)
    elif method.lower() == 'ipw':
        matcher = InverseProbabilityWeighting(weight_col=weight_col)
    elif method.lower() == 'dr':
        matcher = DoublyRobust(weight_col=weight_col)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'psm', 'ipw', or 'dr'")
    
    result = matcher.fit(
        df=df,
        outcome_var=outcome_var,
        treatment_var='IS_FEMALE',
        covariates=covariates
    )
    
    return result
