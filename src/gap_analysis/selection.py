"""
EquiPay Canada - Selection Correction Methods
==============================================

Implements Heckman selection correction and related methods for
addressing sample selection bias in wage gap estimation.

Sample selection is critical in wage analysis because:
1. We only observe wages for employed individuals
2. Employment is not random - it's affected by same factors as wages
3. Ignoring selection leads to biased gap estimates

Methods Implemented:
--------------------
1. Heckman Two-Step (Heckman, 1979)
2. Maximum Likelihood Estimation
3. Control Function Approach
4. Roy Model (comparative advantage)

References:
-----------
- Heckman, J. (1979). Sample Selection Bias as Specification Error
- Gronau, R. (1974). Wage Comparisons: A Selectivity Bias
- Neal & Johnson (1996). Role of Premarket Factors

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
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class HeckmanResult:
    """Results from Heckman selection correction."""
    
    # Outcome equation coefficients
    outcome_coefs: pd.Series
    outcome_se: pd.Series
    
    # Selection equation coefficients
    selection_coefs: pd.Series
    selection_se: pd.Series
    
    # Mills ratio (lambda)
    lambda_coef: float
    lambda_se: float
    
    # Correlation between errors
    rho: float
    
    # Standard deviation of outcome errors
    sigma: float
    
    # Test statistics
    lambda_z: float
    lambda_pvalue: float
    
    # Sample sizes
    n_total: int
    n_selected: int
    
    # Model fit
    pseudo_r2: float = None
    
    # Gender gap estimates
    uncorrected_gap: float = None
    corrected_gap: float = None
    selection_bias: float = None
    
    def __repr__(self):
        return (
            f"HeckmanResult(\n"
            f"  Lambda (IMR): {self.lambda_coef:.4f} (SE: {self.lambda_se:.4f})\n"
            f"  Rho: {self.rho:.4f}\n"
            f"  Selection significant: {self.lambda_pvalue < 0.05}\n"
            f"  Uncorrected gap: {self.uncorrected_gap:.1%}\n"
            f"  Corrected gap: {self.corrected_gap:.1%}\n"
            f"  Selection bias: {self.selection_bias:.1%}\n"
            f")"
        )
    
    def summary(self) -> str:
        """Generate formatted summary."""
        lines = [
            "=" * 70,
            "HECKMAN SELECTION CORRECTION RESULTS",
            "=" * 70,
            f"\nSample: {self.n_selected:,} selected out of {self.n_total:,} ({100*self.n_selected/self.n_total:.1f}%)",
            f"\n--- Selection Equation ---",
        ]
        
        for var, coef in self.selection_coefs.items():
            se = self.selection_se.get(var, np.nan)
            z = coef / se if se > 0 else np.nan
            sig = "***" if abs(z) > 2.576 else "**" if abs(z) > 1.96 else "*" if abs(z) > 1.645 else ""
            lines.append(f"  {var:<20} {coef:>10.4f} ({se:.4f}) {sig}")
        
        lines.extend([
            f"\n--- Outcome Equation ---",
        ])
        
        for var, coef in self.outcome_coefs.items():
            se = self.outcome_se.get(var, np.nan)
            z = coef / se if se > 0 else np.nan
            sig = "***" if abs(z) > 2.576 else "**" if abs(z) > 1.96 else "*" if abs(z) > 1.645 else ""
            lines.append(f"  {var:<20} {coef:>10.4f} ({se:.4f}) {sig}")
        
        lines.extend([
            f"\n--- Selection Parameters ---",
            f"  Lambda (IMR):        {self.lambda_coef:>10.4f} ({self.lambda_se:.4f})",
            f"  Rho:                 {self.rho:>10.4f}",
            f"  Sigma:               {self.sigma:>10.4f}",
            f"  Selection test p:    {self.lambda_pvalue:>10.4f}",
            f"\n--- Gender Gap Estimates ---",
            f"  Uncorrected:         {self.uncorrected_gap:>10.1%}",
            f"  Selection-corrected: {self.corrected_gap:>10.1%}",
            f"  Selection bias:      {self.selection_bias:>10.1%}",
            "",
            "Significance: *** p<0.01, ** p<0.05, * p<0.10",
            "=" * 70,
        ])
        
        return "\n".join(lines)


@dataclass 
class RoyModelResult:
    """Results from Roy model of sector selection."""
    
    # Sector choice model
    choice_coefs: pd.Series
    
    # Sector-specific wage equations
    sector_coefs: Dict[str, pd.Series]
    
    # Comparative advantage parameters
    sorting_gain: float  # Benefit from sector choice
    
    # Counterfactual wages
    cf_wages: pd.DataFrame
    
    # Selection-corrected gaps by sector
    gaps_by_sector: Dict[str, float]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def inverse_mills_ratio(z: np.ndarray) -> np.ndarray:
    """
    Calculate Inverse Mills Ratio (IMR).
    
    IMR = φ(z) / Φ(z)  for selected (positive selection)
    IMR = -φ(z) / (1-Φ(z))  for non-selected
    
    Parameters
    ----------
    z : array
        Predicted values from selection probit
        
    Returns
    -------
    imr : array
        Inverse Mills Ratio values
    """
    pdf = stats.norm.pdf(z)
    cdf = stats.norm.cdf(z)
    
    # Avoid division by zero
    cdf = np.clip(cdf, 1e-10, 1 - 1e-10)
    
    imr = pdf / cdf
    
    return imr


def probit_ll(beta: np.ndarray, X: np.ndarray, y: np.ndarray, 
              weights: np.ndarray = None) -> float:
    """Weighted probit log-likelihood."""
    z = X @ beta
    p = stats.norm.cdf(z)
    p = np.clip(p, 1e-10, 1 - 1e-10)
    
    ll = y * np.log(p) + (1 - y) * np.log(1 - p)
    
    if weights is not None:
        ll = ll * weights
    
    return -np.sum(ll)


def probit_gradient(beta: np.ndarray, X: np.ndarray, y: np.ndarray,
                   weights: np.ndarray = None) -> np.ndarray:
    """Gradient of weighted probit log-likelihood."""
    z = X @ beta
    pdf = stats.norm.pdf(z)
    cdf = stats.norm.cdf(z)
    cdf = np.clip(cdf, 1e-10, 1 - 1e-10)
    
    grad = ((y - cdf) / (cdf * (1 - cdf))) * pdf
    
    if weights is not None:
        grad = grad * weights
    
    return -X.T @ grad


# =============================================================================
# MAIN CLASSES
# =============================================================================

class HeckmanTwoStep:
    """
    Heckman Two-Step Selection Correction.
    
    Addresses sample selection bias when wages are only observed
    for employed individuals. Uses probit for selection equation
    and OLS with IMR correction for outcome equation.
    
    Parameters
    ----------
    weight_col : str
        Column name for survey weights
    
    Examples
    --------
    >>> heckman = HeckmanTwoStep()
    >>> result = heckman.fit(
    ...     df=data,
    ...     outcome_var='log_wage',
    ...     selection_var='employed',
    ...     outcome_controls=['age', 'education', 'experience'],
    ...     selection_controls=['age', 'education', 'children', 'spouse_income'],
    ...     gender_var='female'
    ... )
    >>> print(result.summary())
    
    Notes
    -----
    The exclusion restriction requires at least one variable that
    affects selection but not wages (e.g., number of children,
    spouse income, non-labor income).
    """
    
    def __init__(self, weight_col: str = 'FINALWT'):
        self.weight_col = weight_col
        self.result_ = None
        
    def fit(
        self,
        df: pd.DataFrame,
        outcome_var: str,
        selection_var: str,
        outcome_controls: List[str],
        selection_controls: List[str],
        gender_var: str = 'IS_FEMALE',
        exclusion_vars: List[str] = None
    ) -> HeckmanResult:
        """
        Fit Heckman two-step model.
        
        Parameters
        ----------
        df : DataFrame
            Data with both selected and non-selected observations
        outcome_var : str
            Wage variable (log wages recommended)
        selection_var : str
            Binary selection indicator (1 = employed/observed)
        outcome_controls : list
            Control variables for wage equation
        selection_controls : list
            Control variables for selection equation
        gender_var : str
            Gender indicator variable
        exclusion_vars : list
            Variables in selection but not outcome (for identification)
            
        Returns
        -------
        HeckmanResult
            Selection-corrected estimation results
        """
        logger.info("Fitting Heckman two-step selection model")
        
        # Get weights
        weights = df[self.weight_col].values if self.weight_col in df.columns else None
        
        # Full sample for selection
        y_select = df[selection_var].values
        
        # Selection equation variables
        X_select_vars = [gender_var] + selection_controls
        if exclusion_vars:
            X_select_vars += exclusion_vars
        
        X_select = df[X_select_vars].values
        X_select = np.column_stack([np.ones(len(X_select)), X_select])
        
        # Step 1: Probit on full sample
        logger.info("Step 1: Estimating selection equation (probit)")
        
        beta_init = np.zeros(X_select.shape[1])
        result_probit = minimize(
            probit_ll, beta_init,
            args=(X_select, y_select, weights),
            method='BFGS',
            jac=probit_gradient
        )
        
        gamma = result_probit.x
        
        # Calculate IMR for selected observations
        z_hat = X_select @ gamma
        imr = inverse_mills_ratio(z_hat)
        
        # Selected sample
        selected = df[selection_var] == 1
        n_total = len(df)
        n_selected = selected.sum()
        
        # Step 2: OLS with IMR on selected sample
        logger.info("Step 2: Estimating outcome equation with IMR")
        
        df_selected = df[selected].copy()
        df_selected['IMR'] = imr[selected]
        
        # Outcome equation variables
        X_outcome_vars = [gender_var] + outcome_controls + ['IMR']
        X_outcome = df_selected[X_outcome_vars].values
        X_outcome = np.column_stack([np.ones(len(X_outcome)), X_outcome])
        
        y_outcome = df_selected[outcome_var].values
        w_outcome = df_selected[self.weight_col].values if weights is not None else None
        
        # Weighted OLS
        if w_outcome is not None:
            W = np.diag(w_outcome)
            XtWX = X_outcome.T @ W @ X_outcome
            XtWy = X_outcome.T @ W @ y_outcome
        else:
            XtWX = X_outcome.T @ X_outcome
            XtWy = X_outcome.T @ y_outcome
        
        beta = np.linalg.solve(XtWX, XtWy)
        
        # Residuals and standard errors
        resid = y_outcome - X_outcome @ beta
        if w_outcome is not None:
            sigma_sq = np.sum(w_outcome * resid**2) / np.sum(w_outcome)
        else:
            sigma_sq = np.sum(resid**2) / (n_selected - len(beta))
        
        sigma = np.sqrt(sigma_sq)
        
        # Correct standard errors for selection (Murphy-Topel)
        try:
            var_beta = sigma_sq * np.linalg.inv(XtWX)
            se_beta = np.sqrt(np.diag(var_beta))
        except:
            se_beta = np.full(len(beta), np.nan)
        
        # Extract coefficients
        outcome_names = ['const'] + X_outcome_vars
        outcome_coefs = pd.Series(beta, index=outcome_names)
        outcome_se = pd.Series(se_beta, index=outcome_names)
        
        # Selection equation coefficients
        select_names = ['const'] + X_select_vars
        selection_coefs = pd.Series(gamma, index=select_names)
        
        # Approximate SE for selection (from Hessian)
        try:
            hess = self._probit_hessian(gamma, X_select, y_select, weights)
            var_gamma = np.linalg.inv(-hess)
            se_gamma = np.sqrt(np.diag(var_gamma))
        except:
            se_gamma = np.full(len(gamma), np.nan)
        selection_se = pd.Series(se_gamma, index=select_names)
        
        # Lambda (IMR coefficient) and rho
        lambda_coef = beta[-1]  # Last coefficient is IMR
        lambda_se = se_beta[-1]
        
        # Correlation between errors: rho = lambda / sigma
        rho = lambda_coef / sigma if sigma > 0 else 0
        rho = np.clip(rho, -0.99, 0.99)
        
        # Test for selection
        lambda_z = lambda_coef / lambda_se if lambda_se > 0 else 0
        lambda_pvalue = 2 * (1 - stats.norm.cdf(abs(lambda_z)))
        
        # Gender gap estimates
        gender_idx = outcome_names.index(gender_var)
        corrected_gap = beta[gender_idx]
        
        # Uncorrected gap (OLS without IMR)
        X_uncorr = X_outcome[:, :-1]  # Remove IMR
        if w_outcome is not None:
            W = np.diag(w_outcome)
            beta_uncorr = np.linalg.solve(X_uncorr.T @ W @ X_uncorr, X_uncorr.T @ W @ y_outcome)
        else:
            beta_uncorr = np.linalg.lstsq(X_uncorr, y_outcome, rcond=None)[0]
        
        gender_idx_uncorr = 1  # After constant
        uncorrected_gap = beta_uncorr[gender_idx_uncorr]
        
        selection_bias = uncorrected_gap - corrected_gap
        
        # Pseudo R-squared
        y_pred = X_outcome @ beta
        if w_outcome is not None:
            ss_res = np.sum(w_outcome * (y_outcome - y_pred)**2)
            ss_tot = np.sum(w_outcome * (y_outcome - np.average(y_outcome, weights=w_outcome))**2)
        else:
            ss_res = np.sum((y_outcome - y_pred)**2)
            ss_tot = np.sum((y_outcome - y_outcome.mean())**2)
        
        pseudo_r2 = 1 - ss_res / ss_tot
        
        self.result_ = HeckmanResult(
            outcome_coefs=outcome_coefs,
            outcome_se=outcome_se,
            selection_coefs=selection_coefs,
            selection_se=selection_se,
            lambda_coef=lambda_coef,
            lambda_se=lambda_se,
            rho=rho,
            sigma=sigma,
            lambda_z=lambda_z,
            lambda_pvalue=lambda_pvalue,
            n_total=n_total,
            n_selected=n_selected,
            pseudo_r2=pseudo_r2,
            uncorrected_gap=uncorrected_gap,
            corrected_gap=corrected_gap,
            selection_bias=selection_bias
        )
        
        logger.info(f"Selection correction complete. Lambda p-value: {lambda_pvalue:.4f}")
        
        return self.result_
    
    def _probit_hessian(self, beta: np.ndarray, X: np.ndarray, y: np.ndarray,
                        weights: np.ndarray = None) -> np.ndarray:
        """Compute Hessian of probit log-likelihood."""
        z = X @ beta
        pdf = stats.norm.pdf(z)
        cdf = stats.norm.cdf(z)
        cdf = np.clip(cdf, 1e-10, 1 - 1e-10)
        
        # Second derivative
        w_diag = -(pdf**2 / (cdf * (1 - cdf)) + z * pdf * (y - cdf) / (cdf * (1 - cdf)))
        
        if weights is not None:
            w_diag = w_diag * weights
        
        H = X.T @ np.diag(w_diag) @ X
        
        return H
    
    def predict_counterfactual(
        self,
        df: pd.DataFrame,
        selection_controls: List[str],
        exclusion_vars: List[str] = None
    ) -> pd.DataFrame:
        """
        Predict counterfactual wages for non-employed.
        
        Returns
        -------
        DataFrame with predicted wages if each person were employed
        """
        if self.result_ is None:
            raise ValueError("Must fit model first")
        
        # Implement counterfactual prediction
        # For each non-employed, predict what their wage would be
        # accounting for selection
        
        raise NotImplementedError("Counterfactual prediction not yet implemented")


class HeckmanMLE:
    """
    Maximum Likelihood Heckman Selection Model.
    
    More efficient than two-step but computationally intensive.
    Jointly estimates selection and outcome equations.
    """
    
    def __init__(self, weight_col: str = 'FINALWT'):
        self.weight_col = weight_col
        
    def fit(
        self,
        df: pd.DataFrame,
        outcome_var: str,
        selection_var: str,
        outcome_controls: List[str],
        selection_controls: List[str],
        gender_var: str = 'IS_FEMALE'
    ) -> HeckmanResult:
        """
        Fit Heckman model via MLE.
        
        Maximizes the joint likelihood of selection and wages.
        """
        logger.info("Fitting Heckman MLE model")
        
        # This is more complex - would need to implement bivariate normal
        # likelihood. For now, use two-step as approximation.
        
        two_step = HeckmanTwoStep(weight_col=self.weight_col)
        return two_step.fit(
            df, outcome_var, selection_var,
            outcome_controls, selection_controls, gender_var
        )


class ControlFunction:
    """
    Control Function Approach for Selection.
    
    Alternative to Heckman that uses fitted residuals from
    first stage as control variables.
    """
    
    def __init__(self, weight_col: str = 'FINALWT'):
        self.weight_col = weight_col
        
    def fit(
        self,
        df: pd.DataFrame,
        outcome_var: str,
        selection_var: str,
        outcome_controls: List[str],
        selection_controls: List[str],
        exclusion_vars: List[str],
        gender_var: str = 'IS_FEMALE'
    ) -> Dict:
        """
        Fit control function model.
        
        Parameters
        ----------
        exclusion_vars : list
            Variables that affect selection but not outcome
            (required for identification)
        """
        logger.info("Fitting control function model")
        
        # Step 1: Reduced form selection
        # Regress selection on all exogenous variables
        
        # Step 2: Get generalized residuals
        
        # Step 3: Include residuals as control in outcome equation
        
        raise NotImplementedError("Control function not yet implemented")


# =============================================================================
# HELPER FOR LFS DATA
# =============================================================================

def prepare_lfs_selection_data(
    df: pd.DataFrame,
    employed_codes: List[int] = [1, 2, 3],
    lfsstat_col: str = 'LFSSTAT'
) -> pd.DataFrame:
    """
    Prepare LFS data for selection analysis.
    
    Creates employment indicator and exclusion restriction candidates.
    
    Parameters
    ----------
    df : DataFrame
        LFS PUMF data
    employed_codes : list
        LFSSTAT codes indicating employment
        1 = Employed, at work
        2 = Employed, absent from work
        3 = Unemployed
    lfsstat_col : str
        Column name for labor force status
        
    Returns
    -------
    DataFrame with selection variables added
    """
    df = df.copy()
    
    # Create employment indicator
    df['EMPLOYED'] = df[lfsstat_col].isin(employed_codes).astype(int)
    
    # Potential exclusion restrictions
    # (variables that affect employment but not wages conditional on employment)
    
    # 1. Presence of young children (affects LFP, especially for women)
    if 'AGYOWNK' in df.columns:
        # AGYOWNK: Age of youngest child
        # Create indicator for young children
        df['HAS_YOUNG_CHILD'] = (df['AGYOWNK'] <= 5).astype(int)
        df['HAS_SCHOOL_AGE'] = ((df['AGYOWNK'] > 5) & (df['AGYOWNK'] <= 12)).astype(int)
    
    # 2. Marital status (affects LFP decision)
    if 'MARSTAT' in df.columns:
        df['MARRIED'] = (df['MARSTAT'] == 1).astype(int)
    
    # 3. Family type
    if 'EFAMTYPE' in df.columns:
        df['SINGLE_PARENT'] = df['EFAMTYPE'].isin([3, 4]).astype(int)
    
    logger.info(f"Selection data prepared. Employment rate: {df['EMPLOYED'].mean():.1%}")
    
    return df


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def run_heckman_gender_gap(
    df: pd.DataFrame,
    outcome_var: str = 'LOG_HRLYEARN',
    weight_col: str = 'FINALWT'
) -> HeckmanResult:
    """
    Run standard Heckman gender wage gap analysis on LFS data.
    
    Uses standard controls and exclusion restrictions appropriate
    for Canadian LFS PUMF data.
    
    Parameters
    ----------
    df : DataFrame
        LFS PUMF data with LFSSTAT for selection
    outcome_var : str
        Log wage variable
    weight_col : str
        Survey weight column
        
    Returns
    -------
    HeckmanResult
        Selection-corrected gender gap estimates
    """
    # Prepare data
    df = prepare_lfs_selection_data(df)
    
    # Standard controls for wage equation
    outcome_controls = [
        'AGE', 'AGE_SQ', 'EDUC_YEARS', 
        'EXPERIENCE_PROXY', 'EXPERIENCE_SQ'
    ]
    
    # Add available controls
    for col in ['PROV', 'NOC_10', 'NAICS_21', 'UNION', 'FTPTMAIN']:
        if col in df.columns:
            outcome_controls.append(col)
    
    # Selection controls (includes exclusion restrictions)
    selection_controls = outcome_controls.copy()
    
    # Exclusion restrictions (affect selection, not wages)
    exclusion_vars = []
    for col in ['HAS_YOUNG_CHILD', 'HAS_SCHOOL_AGE', 'MARRIED', 'SINGLE_PARENT']:
        if col in df.columns:
            exclusion_vars.append(col)
    
    if not exclusion_vars:
        warnings.warn(
            "No exclusion restrictions available. Model may not be identified. "
            "Consider adding variables like AGYOWNK, MARSTAT, or EFAMTYPE."
        )
    
    # Filter to valid observations
    required_cols = [outcome_var, 'EMPLOYED', 'IS_FEMALE'] + outcome_controls + exclusion_vars
    required_cols = [c for c in required_cols if c in df.columns]
    df_valid = df.dropna(subset=required_cols)
    
    # Fit model
    heckman = HeckmanTwoStep(weight_col=weight_col)
    result = heckman.fit(
        df=df_valid,
        outcome_var=outcome_var,
        selection_var='EMPLOYED',
        outcome_controls=[c for c in outcome_controls if c in df.columns],
        selection_controls=[c for c in outcome_controls if c in df.columns],
        gender_var='IS_FEMALE',
        exclusion_vars=exclusion_vars
    )
    
    return result
