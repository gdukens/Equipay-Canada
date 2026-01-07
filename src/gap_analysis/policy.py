"""
EquiPay Canada - Policy Impact Evaluation
==========================================

Implements causal inference methods for evaluating the impact of
pay equity policies and legislation on wage gaps.

Methods Implemented:
--------------------
1. Difference-in-Differences (DiD)
2. Event Study Design
3. Synthetic Control Method
4. Regression Discontinuity (RD)
5. Triple Differences (DDD)

Canadian Policy Context:
-----------------------
- Quebec Pay Equity Act (1996)
- Ontario Pay Transparency Act (2018)
- Federal Pay Equity Act (2021)
- Various provincial minimum wage changes

References:
-----------
- Angrist & Pischke (2009). Mostly Harmless Econometrics
- Abadie & Gardeazabal (2003). Synthetic Control Method
- Callaway & Sant'Anna (2021). Difference-in-Differences with Multiple Time Periods

Author: EquiPay Canada Research Team
Version: 1.0.0
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import warnings

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# =============================================================================
# POLICY EVENT DEFINITIONS
# =============================================================================

CANADIAN_POLICY_EVENTS = {
    'QC_PAY_EQUITY_1996': {
        'name': 'Quebec Pay Equity Act',
        'year': 1996,
        'month': 11,
        'province': 10,  # Quebec PROV code
        'description': 'Mandatory pay equity for employers with 10+ employees',
        'treated': [10],  # Quebec
        'control': [35, 46, 47, 48],  # MB, SK, AB, BC
    },
    'ON_PAY_TRANSPARENCY_2018': {
        'name': 'Ontario Pay Transparency Act',
        'year': 2018,
        'month': 4,
        'province': 35,  # Ontario PROV code
        'description': 'Salary range disclosure requirements',
        'treated': [35],  # Ontario
        'control': [46, 47, 48],  # SK, AB, BC
    },
    'FED_PAY_EQUITY_2021': {
        'name': 'Federal Pay Equity Act',
        'year': 2021,
        'month': 8,
        'province': None,  # Federal - all provinces
        'description': 'Proactive pay equity for federally regulated employers',
        'treated': 'federal_sector',  # COWMAIN indicator
        'control': 'provincial_sector',
    },
    'BC_MIN_WAGE_2021': {
        'name': 'BC Minimum Wage Increase',
        'year': 2021,
        'month': 6,
        'province': 59,  # BC
        'description': '$15.20/hour minimum wage',
        'treated': [59],
        'control': [46, 47, 48],  # SK, AB
    },
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class DiDResult:
    """Results from Difference-in-Differences estimation."""
    
    # Main estimate
    did_estimate: float
    did_se: float
    did_ci: Tuple[float, float]
    did_pvalue: float
    
    # Components
    treated_before: float
    treated_after: float
    control_before: float
    control_after: float
    
    # Pre-treatment trend test
    parallel_trends_pvalue: float
    parallel_trends_passed: bool
    
    # Sample sizes
    n_treated_before: int
    n_treated_after: int
    n_control_before: int
    n_control_after: int
    
    # Model info
    covariates_used: List[str]
    fixed_effects: List[str]
    
    # Policy interpretation
    policy_effect_pct: float
    policy_effect_se: float
    
    def __repr__(self):
        return (
            f"DiDResult(\n"
            f"  DiD Estimate: {self.did_estimate:.4f} (SE: {self.did_se:.4f})\n"
            f"  Policy Effect: {self.policy_effect_pct:.1%}\n"
            f"  95% CI: [{self.did_ci[0]:.4f}, {self.did_ci[1]:.4f}]\n"
            f"  Parallel Trends: {'PASSED' if self.parallel_trends_passed else 'FAILED'}\n"
            f")"
        )
    
    def summary(self) -> str:
        """Generate formatted summary."""
        lines = [
            "=" * 70,
            "DIFFERENCE-IN-DIFFERENCES RESULTS",
            "=" * 70,
            "",
            "--- DiD Estimate ---",
            f"  Effect: {self.did_estimate:.4f} (SE: {self.did_se:.4f})",
            f"  95% CI: [{self.did_ci[0]:.4f}, {self.did_ci[1]:.4f}]",
            f"  p-value: {self.did_pvalue:.4f}",
            "",
            "--- Policy Impact ---",
            f"  Wage change: {self.policy_effect_pct:.1%}",
            "",
            "--- 2x2 Table ---",
            f"{'Group':<15} {'Before':>12} {'After':>12} {'Diff':>12}",
            "-" * 55,
            f"{'Treated':<15} {self.treated_before:>12.4f} {self.treated_after:>12.4f} {self.treated_after - self.treated_before:>12.4f}",
            f"{'Control':<15} {self.control_before:>12.4f} {self.control_after:>12.4f} {self.control_after - self.control_before:>12.4f}",
            f"{'Diff':<15} {self.treated_before - self.control_before:>12.4f} {self.treated_after - self.control_after:>12.4f} {self.did_estimate:>12.4f}",
            "",
            "--- Parallel Trends Test ---",
            f"  H0: Parallel pre-trends",
            f"  p-value: {self.parallel_trends_pvalue:.4f}",
            f"  Result: {'Cannot reject parallel trends ✓' if self.parallel_trends_passed else 'Parallel trends rejected ✗'}",
            "",
            "--- Sample Sizes ---",
            f"  Treated before:  {self.n_treated_before:>10,}",
            f"  Treated after:   {self.n_treated_after:>10,}",
            f"  Control before:  {self.n_control_before:>10,}",
            f"  Control after:   {self.n_control_after:>10,}",
            "",
            "=" * 70,
        ]
        return "\n".join(lines)


@dataclass
class EventStudyResult:
    """Results from event study design."""
    
    # Coefficients by relative time
    coefficients: pd.DataFrame  # time, coef, se, ci_lower, ci_upper
    
    # Pre-trend test
    pre_trend_fstat: float
    pre_trend_pvalue: float
    
    # Average post-treatment effect
    avg_post_effect: float
    avg_post_se: float
    
    # Sample info
    n_obs: int
    n_treated: int
    n_periods: int
    
    def plot_data(self) -> pd.DataFrame:
        """Return data formatted for plotting."""
        return self.coefficients.copy()


@dataclass
class SyntheticControlResult:
    """Results from synthetic control method."""
    
    # Weights for control units
    weights: pd.Series
    
    # Treated vs synthetic
    treated_series: pd.Series
    synthetic_series: pd.Series
    
    # Treatment effect
    effect_by_period: pd.Series
    avg_effect: float
    
    # Inference
    placebo_effects: pd.DataFrame  # For inference
    pvalue: float
    
    # Fit quality
    pre_treatment_rmse: float


# =============================================================================
# DIFFERENCE-IN-DIFFERENCES
# =============================================================================

class DifferenceInDifferences:
    """
    Difference-in-Differences Estimator.
    
    Compares changes over time between treated and control groups
    to estimate causal policy effects.
    
    Parameters
    ----------
    weight_col : str
        Survey weight column
    cluster_var : str
        Variable for clustered standard errors
        
    Examples
    --------
    >>> did = DifferenceInDifferences()
    >>> result = did.fit(
    ...     df=data,
    ...     outcome_var='LOG_HRLYEARN',
    ...     treated_var='TREATED_PROV',
    ...     post_var='POST_POLICY',
    ...     covariates=['AGE', 'EDUC', 'NOC_10']
    ... )
    >>> print(result.summary())
    """
    
    def __init__(
        self,
        weight_col: str = 'FINALWT',
        cluster_var: str = None
    ):
        self.weight_col = weight_col
        self.cluster_var = cluster_var
        self.result_ = None
        
    def fit(
        self,
        df: pd.DataFrame,
        outcome_var: str,
        treated_var: str,
        post_var: str,
        covariates: List[str] = None,
        fixed_effects: List[str] = None
    ) -> DiDResult:
        """
        Estimate difference-in-differences.
        
        Parameters
        ----------
        df : DataFrame
            Panel or repeated cross-section data
        outcome_var : str
            Outcome variable (log wages recommended)
        treated_var : str
            Binary indicator for treated group
        post_var : str
            Binary indicator for post-treatment period
        covariates : list
            Control variables
        fixed_effects : list
            Variables for fixed effects
            
        Returns
        -------
        DiDResult
            DiD estimation results
        """
        logger.info("Estimating Difference-in-Differences")
        
        # Create interaction term
        df = df.copy()
        df['TREAT_POST'] = df[treated_var] * df[post_var]
        
        # Filter to valid observations
        required = [outcome_var, treated_var, post_var]
        if covariates:
            required += covariates
        df_clean = df.dropna(subset=required)
        
        # Get sample sizes
        n_treated_before = ((df_clean[treated_var] == 1) & (df_clean[post_var] == 0)).sum()
        n_treated_after = ((df_clean[treated_var] == 1) & (df_clean[post_var] == 1)).sum()
        n_control_before = ((df_clean[treated_var] == 0) & (df_clean[post_var] == 0)).sum()
        n_control_after = ((df_clean[treated_var] == 0) & (df_clean[post_var] == 1)).sum()
        
        # Simple 2x2 means
        weights = df_clean[self.weight_col].values if self.weight_col in df_clean.columns else None
        
        def wmean(mask):
            y = df_clean.loc[mask, outcome_var].values
            if weights is not None:
                w = weights[mask]
                return np.average(y, weights=w)
            return np.mean(y)
        
        treated_before = wmean((df_clean[treated_var] == 1) & (df_clean[post_var] == 0))
        treated_after = wmean((df_clean[treated_var] == 1) & (df_clean[post_var] == 1))
        control_before = wmean((df_clean[treated_var] == 0) & (df_clean[post_var] == 0))
        control_after = wmean((df_clean[treated_var] == 0) & (df_clean[post_var] == 1))
        
        # Simple DiD
        did_simple = (treated_after - treated_before) - (control_after - control_before)
        
        # Regression DiD
        X_vars = [treated_var, post_var, 'TREAT_POST']
        if covariates:
            X_vars += covariates
        
        X = df_clean[X_vars].values
        X = np.column_stack([np.ones(len(X)), X])
        y = df_clean[outcome_var].values
        
        # Weighted OLS
        if weights is not None:
            W = np.diag(weights)
            XtWX = X.T @ W @ X
            XtWy = X.T @ W @ y
        else:
            XtWX = X.T @ X
            XtWy = X.T @ y
        
        beta = np.linalg.solve(XtWX, XtWy)
        
        # DiD coefficient is the TREAT_POST coefficient
        did_idx = 3  # const, treated, post, treat_post
        did_estimate = beta[did_idx]
        
        # Standard errors
        resid = y - X @ beta
        n = len(y)
        k = len(beta)
        
        if weights is not None:
            sigma_sq = np.sum(weights * resid**2) / np.sum(weights)
        else:
            sigma_sq = np.sum(resid**2) / (n - k)
        
        try:
            var_beta = sigma_sq * np.linalg.inv(XtWX)
            did_se = np.sqrt(var_beta[did_idx, did_idx])
        except:
            did_se = np.nan
        
        # Confidence interval and p-value
        did_ci = (did_estimate - 1.96 * did_se, did_estimate + 1.96 * did_se)
        did_t = did_estimate / did_se if did_se > 0 else 0
        did_pvalue = 2 * (1 - stats.t.cdf(abs(did_t), n - k))
        
        # Parallel trends test (pre-period trend interaction)
        parallel_trends_pvalue = self._test_parallel_trends(
            df_clean, outcome_var, treated_var, post_var
        )
        
        # Policy interpretation (convert log to %)
        policy_effect_pct = np.exp(did_estimate) - 1
        policy_effect_se = np.exp(did_estimate) * did_se  # Delta method
        
        self.result_ = DiDResult(
            did_estimate=did_estimate,
            did_se=did_se,
            did_ci=did_ci,
            did_pvalue=did_pvalue,
            treated_before=treated_before,
            treated_after=treated_after,
            control_before=control_before,
            control_after=control_after,
            parallel_trends_pvalue=parallel_trends_pvalue,
            parallel_trends_passed=parallel_trends_pvalue > 0.05,
            n_treated_before=n_treated_before,
            n_treated_after=n_treated_after,
            n_control_before=n_control_before,
            n_control_after=n_control_after,
            covariates_used=covariates or [],
            fixed_effects=fixed_effects or [],
            policy_effect_pct=policy_effect_pct,
            policy_effect_se=policy_effect_se
        )
        
        return self.result_
    
    def _test_parallel_trends(
        self,
        df: pd.DataFrame,
        outcome_var: str,
        treated_var: str,
        post_var: str
    ) -> float:
        """Test for parallel pre-treatment trends."""
        
        # Use only pre-period data
        pre_data = df[df[post_var] == 0].copy()
        
        if 'YEAR' not in pre_data.columns:
            # Cannot test without time variable
            return np.nan
        
        # Check if we have multiple pre-periods
        years = pre_data['YEAR'].unique()
        if len(years) < 2:
            return np.nan
        
        # Trend interaction: YEAR * TREATED
        pre_data['TREND'] = pre_data['YEAR'] - pre_data['YEAR'].min()
        pre_data['TREND_TREAT'] = pre_data['TREND'] * pre_data[treated_var]
        
        # Regression
        X = pre_data[[treated_var, 'TREND', 'TREND_TREAT']].values
        X = np.column_stack([np.ones(len(X)), X])
        y = pre_data[outcome_var].values
        
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            resid = y - X @ beta
            n, k = len(y), len(beta)
            
            sigma_sq = np.sum(resid**2) / (n - k)
            var_beta = sigma_sq * np.linalg.inv(X.T @ X)
            
            # Test TREND_TREAT = 0
            trend_treat_coef = beta[-1]
            trend_treat_se = np.sqrt(var_beta[-1, -1])
            t_stat = trend_treat_coef / trend_treat_se
            pvalue = 2 * (1 - stats.t.cdf(abs(t_stat), n - k))
            
            return pvalue
        except:
            return np.nan


class EventStudy:
    """
    Event Study Design for Dynamic Treatment Effects.
    
    Estimates treatment effects by period relative to
    treatment timing, allowing visualization of:
    1. Pre-treatment trends (should be zero)
    2. Dynamic treatment effects post-treatment
    
    Parameters
    ----------
    weight_col : str
        Survey weight column
    pre_periods : int
        Number of pre-treatment periods to include
    post_periods : int
        Number of post-treatment periods to include
    """
    
    def __init__(
        self,
        weight_col: str = 'FINALWT',
        pre_periods: int = 4,
        post_periods: int = 4
    ):
        self.weight_col = weight_col
        self.pre_periods = pre_periods
        self.post_periods = post_periods
        self.result_ = None
        
    def fit(
        self,
        df: pd.DataFrame,
        outcome_var: str,
        treated_var: str,
        time_var: str,
        event_time: int,
        covariates: List[str] = None
    ) -> EventStudyResult:
        """
        Estimate event study.
        
        Parameters
        ----------
        df : DataFrame
            Panel or repeated cross-section data
        outcome_var : str
            Outcome variable
        treated_var : str
            Binary treatment indicator
        time_var : str
            Time variable (year)
        event_time : int
            Year of treatment
        covariates : list
            Control variables
            
        Returns
        -------
        EventStudyResult
            Event study results with dynamic coefficients
        """
        logger.info(f"Estimating event study around {event_time}")
        
        df = df.copy()
        
        # Create relative time variable
        df['REL_TIME'] = df[time_var] - event_time
        
        # Limit to specified window
        df = df[
            (df['REL_TIME'] >= -self.pre_periods) & 
            (df['REL_TIME'] <= self.post_periods)
        ].copy()
        
        # Create dummies for each relative time period
        # Omit t=-1 as reference
        periods = list(range(-self.pre_periods, self.post_periods + 1))
        periods.remove(-1)  # Reference period
        
        for t in periods:
            df[f'REL_{t}'] = ((df['REL_TIME'] == t) & (df[treated_var] == 1)).astype(int)
        
        # Regression
        X_vars = [f'REL_{t}' for t in periods] + [treated_var]
        if covariates:
            X_vars += covariates
        
        df_clean = df.dropna(subset=[outcome_var] + X_vars)
        
        X = df_clean[X_vars].values
        X = np.column_stack([np.ones(len(X)), X])
        y = df_clean[outcome_var].values
        
        weights = df_clean[self.weight_col].values if self.weight_col in df_clean.columns else None
        
        # Weighted OLS
        if weights is not None:
            W = np.diag(weights)
            XtWX = X.T @ W @ X
            XtWy = X.T @ W @ y
        else:
            XtWX = X.T @ X
            XtWy = X.T @ y
        
        beta = np.linalg.solve(XtWX, XtWy)
        
        # Standard errors
        resid = y - X @ beta
        n, k = len(y), len(beta)
        
        if weights is not None:
            sigma_sq = np.sum(weights * resid**2) / np.sum(weights)
        else:
            sigma_sq = np.sum(resid**2) / (n - k)
        
        try:
            var_beta = sigma_sq * np.linalg.inv(XtWX)
            se_beta = np.sqrt(np.diag(var_beta))
        except:
            se_beta = np.full(len(beta), np.nan)
        
        # Extract coefficients (skip constant and base variables)
        results = []
        for i, t in enumerate(periods):
            idx = i + 1  # Skip constant
            coef = beta[idx]
            se = se_beta[idx]
            results.append({
                'time': t,
                'coef': coef,
                'se': se,
                'ci_lower': coef - 1.96 * se,
                'ci_upper': coef + 1.96 * se
            })
        
        # Add t=-1 (reference)
        results.append({
            'time': -1,
            'coef': 0,
            'se': 0,
            'ci_lower': 0,
            'ci_upper': 0
        })
        
        coefficients = pd.DataFrame(results).sort_values('time')
        
        # Pre-trend test (joint F-test of pre-period coefficients)
        pre_coefs = [beta[i+1] for i, t in enumerate(periods) if t < 0]
        pre_var = var_beta[1:len(pre_coefs)+1, 1:len(pre_coefs)+1]
        
        try:
            chi2 = np.array(pre_coefs).T @ np.linalg.inv(pre_var) @ np.array(pre_coefs)
            pre_trend_pvalue = 1 - stats.chi2.cdf(chi2, len(pre_coefs))
        except:
            pre_trend_pvalue = np.nan
        
        # Average post-treatment effect
        post_coefs = [beta[i+1] for i, t in enumerate(periods) if t >= 0]
        avg_post = np.mean(post_coefs)
        
        self.result_ = EventStudyResult(
            coefficients=coefficients,
            pre_trend_fstat=chi2 if 'chi2' in dir() else np.nan,
            pre_trend_pvalue=pre_trend_pvalue,
            avg_post_effect=avg_post,
            avg_post_se=np.std(post_coefs) / np.sqrt(len(post_coefs)),
            n_obs=n,
            n_treated=(df_clean[treated_var] == 1).sum(),
            n_periods=len(periods) + 1
        )
        
        return self.result_


class TripleDifference:
    """
    Triple Difference (DDD) Estimator.
    
    Extends DiD by adding a third dimension, typically
    gender for pay equity analysis:
    
    Effect = ΔΔ(Women) - ΔΔ(Men)
    
    This controls for differential trends by:
    - Province (treated vs control)
    - Time (before vs after policy)
    - Gender (women vs men)
    
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
        treated_var: str,
        post_var: str,
        gender_var: str,
        covariates: List[str] = None
    ) -> Dict:
        """
        Estimate triple difference.
        
        The DDD estimate captures the differential policy effect
        on women vs men in treated vs control areas.
        
        Returns
        -------
        dict
            DDD results with all interaction terms
        """
        logger.info("Estimating Triple Difference (DDD)")
        
        df = df.copy()
        
        # Create all interactions
        df['TREAT_POST'] = df[treated_var] * df[post_var]
        df['TREAT_GENDER'] = df[treated_var] * df[gender_var]
        df['POST_GENDER'] = df[post_var] * df[gender_var]
        df['TREAT_POST_GENDER'] = df[treated_var] * df[post_var] * df[gender_var]
        
        # Regression
        X_vars = [
            treated_var, post_var, gender_var,
            'TREAT_POST', 'TREAT_GENDER', 'POST_GENDER',
            'TREAT_POST_GENDER'
        ]
        
        if covariates:
            X_vars += covariates
        
        df_clean = df.dropna(subset=[outcome_var] + X_vars)
        
        X = df_clean[X_vars].values
        X = np.column_stack([np.ones(len(X)), X])
        y = df_clean[outcome_var].values
        
        weights = df_clean[self.weight_col].values if self.weight_col in df_clean.columns else None
        
        if weights is not None:
            W = np.diag(weights)
            beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ y)
        else:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
        
        # DDD coefficient
        ddd_idx = 7  # TREAT_POST_GENDER
        ddd_estimate = beta[ddd_idx]
        
        # Standard errors
        resid = y - X @ beta
        n, k = len(y), len(beta)
        
        if weights is not None:
            sigma_sq = np.sum(weights * resid**2) / np.sum(weights)
        else:
            sigma_sq = np.sum(resid**2) / (n - k)
        
        try:
            if weights is not None:
                var_beta = sigma_sq * np.linalg.inv(X.T @ W @ X)
            else:
                var_beta = sigma_sq * np.linalg.inv(X.T @ X)
            ddd_se = np.sqrt(var_beta[ddd_idx, ddd_idx])
        except:
            ddd_se = np.nan
        
        # Results
        results = {
            'ddd_estimate': ddd_estimate,
            'ddd_se': ddd_se,
            'ddd_ci': (ddd_estimate - 1.96*ddd_se, ddd_estimate + 1.96*ddd_se),
            'policy_effect_on_gender_gap': np.exp(ddd_estimate) - 1,
            'n_obs': n,
            'coefficients': dict(zip(['const'] + X_vars, beta))
        }
        
        self.result_ = results
        return results


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def analyze_policy_impact(
    df: pd.DataFrame,
    policy_key: str,
    outcome_var: str = 'LOG_HRLYEARN',
    method: str = 'did',
    weight_col: str = 'FINALWT'
) -> Union[DiDResult, EventStudyResult]:
    """
    Analyze impact of a specific Canadian pay equity policy.
    
    Parameters
    ----------
    df : DataFrame
        LFS PUMF data with PROV and YEAR
    policy_key : str
        Key from CANADIAN_POLICY_EVENTS
    outcome_var : str
        Outcome variable (log wages)
    method : str
        'did' or 'event_study'
    weight_col : str
        Survey weight column
        
    Returns
    -------
    Result object with policy impact estimate
    """
    if policy_key not in CANADIAN_POLICY_EVENTS:
        raise ValueError(f"Unknown policy: {policy_key}. Available: {list(CANADIAN_POLICY_EVENTS.keys())}")
    
    policy = CANADIAN_POLICY_EVENTS[policy_key]
    
    logger.info(f"Analyzing impact of {policy['name']}")
    
    df = df.copy()
    
    # Create treatment indicator
    if isinstance(policy['treated'], list):
        df['TREATED'] = df['PROV'].isin(policy['treated']).astype(int)
        df['CONTROL'] = df['PROV'].isin(policy['control']).astype(int)
    else:
        # Handle federal sector case
        raise NotImplementedError("Federal sector analysis not yet implemented")
    
    # Create post-treatment indicator
    df['POST'] = (df['YEAR'] >= policy['year']).astype(int)
    
    # Filter to treated and control only
    df = df[(df['TREATED'] == 1) | (df['CONTROL'] == 1)]
    
    # Run analysis
    if method == 'did':
        did = DifferenceInDifferences(weight_col=weight_col)
        result = did.fit(
            df=df,
            outcome_var=outcome_var,
            treated_var='TREATED',
            post_var='POST',
            covariates=['AGE', 'EDUC_YEARS', 'NOC_10'] if 'AGE' in df.columns else None
        )
    elif method == 'event_study':
        es = EventStudy(weight_col=weight_col)
        result = es.fit(
            df=df,
            outcome_var=outcome_var,
            treated_var='TREATED',
            time_var='YEAR',
            event_time=policy['year']
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return result


def create_policy_timeline_plot(
    results: Dict[str, DiDResult],
    save_path: str = None
) -> None:
    """
    Create timeline visualization of multiple policy effects.
    
    Parameters
    ----------
    results : dict
        Mapping of policy name to DiDResult
    save_path : str
        Path to save figure (optional)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available for plotting")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    policies = list(results.keys())
    effects = [r.policy_effect_pct for r in results.values()]
    errors = [1.96 * r.policy_effect_se for r in results.values()]
    
    y_pos = np.arange(len(policies))
    
    ax.barh(y_pos, effects, xerr=errors, capsize=5, color='steelblue', alpha=0.7)
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(policies)
    ax.set_xlabel('Policy Effect on Gender Wage Gap')
    ax.set_title('Impact of Pay Equity Policies')
    
    for i, (eff, err) in enumerate(zip(effects, errors)):
        sig = '***' if abs(eff/err*1.96) > 2.576 else '**' if abs(eff/err*1.96) > 1.96 else '*' if abs(eff/err*1.96) > 1.645 else ''
        ax.text(eff + err + 0.005, i, f'{eff:.1%}{sig}', va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved policy timeline to {save_path}")
    
    return fig
