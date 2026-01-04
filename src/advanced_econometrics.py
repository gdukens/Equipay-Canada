"""
Advanced Econometric Methods for Pay Equity Analysis
=====================================================

This module implements state-of-the-art decomposition and causal inference
methods from the labor economics literature:

1. RIF (Recentered Influence Function) Decomposition
2. Unconditional Quantile Regression
3. DiNardo-Fortin-Lemieux (DFL) Reweighting
4. Propensity Score Matching
5. Doubly Robust Estimation
6. Heckman Selection Correction
7. Segregation Indices

References:
-----------
- Firpo, S., Fortin, N.M., & Lemieux, T. (2009). "Unconditional Quantile Regressions"
- Fortin, N., Lemieux, T., & Firpo, S. (2011). "Decomposition Methods in Economics"
- DiNardo, J., Fortin, N.M., & Lemieux, T. (1996). "Labor Market Institutions"
- Rosenbaum, P.R., & Rubin, D.B. (1983). "The Central Role of the Propensity Score"
- Heckman, J.J. (1979). "Sample Selection Bias as a Specification Error"
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings
import logging

logger = logging.getLogger(__name__)


@dataclass
class DecompositionResult:
    """Container for decomposition results."""
    total_gap: float
    explained: float
    unexplained: float
    detailed: Optional[Dict] = None
    standard_errors: Optional[Dict] = None
    confidence_intervals: Optional[Dict] = None


@dataclass
class QuantileDecompositionResult:
    """Container for quantile-specific decomposition."""
    quantile: float
    total_gap: float
    composition_effect: float
    wage_structure_effect: float
    detailed: Optional[Dict] = None


# =============================================================================
# 1. RIF (RECENTERED INFLUENCE FUNCTION) REGRESSION
# =============================================================================

class RIFRegression:
    """
    Unconditional Quantile Regression using RIF.
    
    Based on Firpo, Fortin & Lemieux (2009).
    
    The RIF for quantile τ is:
        RIF(Y; Q_τ) = Q_τ + (τ - 1(Y ≤ Q_τ)) / f_Y(Q_τ)
    
    Parameters
    ----------
    bandwidth : str or float
        Kernel bandwidth for density estimation. 
        'silverman' uses Silverman's rule of thumb.
    """
    
    def __init__(self, bandwidth: str = 'silverman'):
        self.bandwidth = bandwidth
        self.quantiles_ = None
        self.densities_ = None
        self.coefficients_ = {}
        self.results_ = {}
    
    def compute_rif(self, y: np.ndarray, tau: float) -> np.ndarray:
        """
        Compute RIF values for a given quantile.
        
        Parameters
        ----------
        y : array-like
            Outcome variable (e.g., log wages)
        tau : float
            Quantile (0 < tau < 1)
        
        Returns
        -------
        rif : array-like
            RIF values for each observation
        """
        y = np.asarray(y)
        q_tau = np.quantile(y, tau)
        
        # Estimate density at quantile using kernel density estimation
        if self.bandwidth == 'silverman':
            # Silverman's rule of thumb
            h = 1.06 * np.std(y) * len(y) ** (-1/5)
        else:
            h = self.bandwidth
        
        # Gaussian kernel density at quantile
        kernel_vals = norm.pdf((y - q_tau) / h) / h
        f_tau = np.mean(kernel_vals)
        
        # Avoid division by zero
        f_tau = max(f_tau, 1e-10)
        
        # RIF formula
        indicator = (y <= q_tau).astype(float)
        rif = q_tau + (tau - indicator) / f_tau
        
        return rif
    
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            quantiles: List[float] = None) -> 'RIFRegression':
        """
        Fit RIF-OLS regression at specified quantiles.
        
        Parameters
        ----------
        X : DataFrame
            Covariates including treatment variable
        y : Series
            Outcome variable
        quantiles : list
            Quantiles to estimate (default: deciles)
        """
        if quantiles is None:
            quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        
        self.quantiles_ = quantiles
        X = sm.add_constant(X)
        
        for tau in quantiles:
            rif = self.compute_rif(y.values, tau)
            model = sm.OLS(rif, X).fit(cov_type='HC1')
            self.coefficients_[tau] = model.params
            self.results_[tau] = model
        
        return self
    
    def get_coefficients(self, variable: str) -> Dict[float, float]:
        """Get coefficient for a variable across quantiles."""
        return {tau: self.coefficients_[tau].get(variable, np.nan) 
                for tau in self.quantiles_}
    
    def decomposition(self, X_1: pd.DataFrame, X_0: pd.DataFrame,
                      y_1: pd.Series, y_0: pd.Series,
                      quantiles: List[float] = None) -> Dict[float, QuantileDecompositionResult]:
        """
        RIF-based Oaxaca-Blinder decomposition at each quantile.
        
        Parameters
        ----------
        X_1, X_0 : DataFrames
            Covariates for group 1 (reference) and group 0
        y_1, y_0 : Series
            Outcomes for each group
        quantiles : list
            Quantiles to decompose
        
        Returns
        -------
        results : dict
            Decomposition results by quantile
        """
        if quantiles is None:
            quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        
        results = {}
        
        for tau in quantiles:
            # Compute RIF for each group
            rif_1 = self.compute_rif(y_1.values, tau)
            rif_0 = self.compute_rif(y_0.values, tau)
            
            # Fit RIF regressions
            X_1_const = sm.add_constant(X_1)
            X_0_const = sm.add_constant(X_0)
            
            model_1 = sm.OLS(rif_1, X_1_const).fit()
            model_0 = sm.OLS(rif_0, X_0_const).fit()
            
            # Means
            X_bar_1 = X_1_const.mean()
            X_bar_0 = X_0_const.mean()
            
            # Counterfactual: Group 0 characteristics, Group 1 returns
            counterfactual = X_bar_0 @ model_1.params
            
            # Decomposition
            total_gap = np.quantile(y_1, tau) - np.quantile(y_0, tau)
            composition = (X_bar_1 - X_bar_0) @ model_0.params
            wage_structure = X_bar_1 @ (model_1.params - model_0.params)
            
            results[tau] = QuantileDecompositionResult(
                quantile=tau,
                total_gap=total_gap,
                composition_effect=composition,
                wage_structure_effect=wage_structure,
                detailed={
                    'group_1_mean_rif': rif_1.mean(),
                    'group_0_mean_rif': rif_0.mean(),
                    'counterfactual': counterfactual
                }
            )
        
        return results


# =============================================================================
# 2. DINARDO-FORTIN-LEMIEUX REWEIGHTING
# =============================================================================

class DFLReweighting:
    """
    DiNardo-Fortin-Lemieux (1996) Reweighting Decomposition.
    
    Creates counterfactual wage distribution by reweighting
    one group to have the covariate distribution of another.
    
    Parameters
    ----------
    propensity_model : str
        'logit' or 'probit' for propensity score estimation
    """
    
    def __init__(self, propensity_model: str = 'logit'):
        self.propensity_model = propensity_model
        self.propensity_scores_ = None
        self.weights_ = None
    
    def fit(self, X: pd.DataFrame, group: pd.Series) -> 'DFLReweighting':
        """
        Estimate propensity scores and reweighting factors.
        
        Parameters
        ----------
        X : DataFrame
            Covariates
        group : Series
            Binary group indicator (1 = reference, 0 = comparison)
        """
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit propensity model
        if self.propensity_model == 'logit':
            model = LogisticRegression(max_iter=1000, solver='lbfgs')
        else:
            # Probit via statsmodels
            model = sm.Probit(group, sm.add_constant(X_scaled))
            probit_result = model.fit(disp=0)
            self.propensity_scores_ = probit_result.predict()
            self._compute_weights(group)
            return self
        
        model.fit(X_scaled, group)
        self.propensity_scores_ = model.predict_proba(X_scaled)[:, 1]
        self._compute_weights(group)
        
        return self
    
    def _compute_weights(self, group: pd.Series):
        """Compute reweighting factors."""
        ps = self.propensity_scores_
        
        # For group 0 (comparison), weight to look like group 1
        # ψ(X) = P(G=1|X) / P(G=0|X) * P(G=0) / P(G=1)
        p_1 = group.mean()
        p_0 = 1 - p_1
        
        self.weights_ = np.ones(len(group))
        mask_0 = (group == 0)
        
        # Avoid extreme weights
        ps_clipped = np.clip(ps, 0.01, 0.99)
        
        self.weights_[mask_0] = (ps_clipped[mask_0] / (1 - ps_clipped[mask_0])) * (p_0 / p_1)
        
        # Normalize weights
        self.weights_[mask_0] = self.weights_[mask_0] / self.weights_[mask_0].sum() * mask_0.sum()
    
    def counterfactual_distribution(self, y: pd.Series, group: pd.Series) -> np.ndarray:
        """
        Compute counterfactual wage distribution.
        
        Returns wages of group 0 reweighted to have group 1's covariate distribution.
        """
        mask_0 = (group == 0)
        return y[mask_0].values, self.weights_[mask_0]
    
    def decomposition(self, y: pd.Series, group: pd.Series,
                      percentiles: List[int] = None) -> Dict:
        """
        Full distributional decomposition.
        
        Parameters
        ----------
        y : Series
            Outcome (wages)
        group : Series
            Group indicator (1 = reference/male, 0 = comparison/female)
        percentiles : list
            Percentiles to decompose (default: deciles)
        
        Returns
        -------
        results : dict
            Decomposition at each percentile
        """
        if percentiles is None:
            percentiles = list(range(10, 100, 10))
        
        mask_1 = (group == 1)
        mask_0 = (group == 0)
        
        y_1 = y[mask_1].values
        y_0 = y[mask_0].values
        w_0 = self.weights_[mask_0]
        
        results = {'percentiles': {}}
        
        for p in percentiles:
            # Actual distributions
            q_1 = np.percentile(y_1, p)
            q_0 = np.percentile(y_0, p)
            
            # Counterfactual: weighted percentile
            sorted_idx = np.argsort(y_0)
            cumsum = np.cumsum(w_0[sorted_idx])
            cumsum /= cumsum[-1]
            q_cf = y_0[sorted_idx][np.searchsorted(cumsum, p/100)]
            
            total_gap = q_1 - q_0
            composition = q_cf - q_0
            wage_structure = q_1 - q_cf
            
            results['percentiles'][p] = {
                'total_gap': total_gap,
                'composition_effect': composition,
                'wage_structure_effect': wage_structure,
                'group_1_quantile': q_1,
                'group_0_quantile': q_0,
                'counterfactual_quantile': q_cf
            }
        
        # Summary statistics
        results['mean'] = {
            'total_gap': y_1.mean() - y_0.mean(),
            'composition_effect': np.average(y_0, weights=w_0) - y_0.mean(),
            'wage_structure_effect': y_1.mean() - np.average(y_0, weights=w_0)
        }
        
        return results


# =============================================================================
# 3. PROPENSITY SCORE MATCHING
# =============================================================================

class PropensityScoreMatching:
    """
    Propensity Score Matching for causal wage gap estimation.
    
    Based on Rosenbaum & Rubin (1983).
    
    Parameters
    ----------
    n_neighbors : int
        Number of neighbors for matching
    caliper : float or None
        Maximum propensity score difference for valid match
    replacement : bool
        Whether to match with replacement
    """
    
    def __init__(self, n_neighbors: int = 1, caliper: float = None,
                 replacement: bool = True):
        self.n_neighbors = n_neighbors
        self.caliper = caliper
        self.replacement = replacement
        self.propensity_scores_ = None
        self.matches_ = None
        self.att_ = None
        self.ate_ = None
    
    def fit(self, X: pd.DataFrame, treatment: pd.Series, 
            outcome: pd.Series) -> 'PropensityScoreMatching':
        """
        Estimate propensity scores and perform matching.
        
        Parameters
        ----------
        X : DataFrame
            Covariates
        treatment : Series
            Binary treatment indicator (1 = treated/female)
        outcome : Series
            Outcome variable (wages)
        """
        # Estimate propensity score
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        ps_model = LogisticRegression(max_iter=1000)
        ps_model.fit(X_scaled, treatment)
        self.propensity_scores_ = ps_model.predict_proba(X_scaled)[:, 1]
        
        # Check overlap
        ps_treated = self.propensity_scores_[treatment == 1]
        ps_control = self.propensity_scores_[treatment == 0]
        
        overlap_min = max(ps_treated.min(), ps_control.min())
        overlap_max = min(ps_treated.max(), ps_control.max())
        
        logger.info(f"Propensity score overlap: [{overlap_min:.3f}, {overlap_max:.3f}]")
        
        # Perform matching
        self._match(treatment, outcome)
        
        return self
    
    def _match(self, treatment: pd.Series, outcome: pd.Series):
        """Perform nearest neighbor matching on propensity score."""
        treated_idx = np.where(treatment == 1)[0]
        control_idx = np.where(treatment == 0)[0]
        
        ps_treated = self.propensity_scores_[treated_idx].reshape(-1, 1)
        ps_control = self.propensity_scores_[control_idx].reshape(-1, 1)
        
        # Fit nearest neighbors on control group
        nn = NearestNeighbors(n_neighbors=self.n_neighbors, metric='euclidean')
        nn.fit(ps_control)
        
        distances, indices = nn.kneighbors(ps_treated)
        
        # Apply caliper if specified
        if self.caliper:
            valid_matches = distances[:, 0] <= self.caliper
        else:
            valid_matches = np.ones(len(treated_idx), dtype=bool)
        
        # Compute treatment effects
        treated_outcomes = outcome.iloc[treated_idx].values
        matched_control_outcomes = np.array([
            outcome.iloc[control_idx[idx]].mean() 
            for idx in indices
        ])
        
        # ATT (Average Treatment Effect on Treated)
        effects = treated_outcomes - matched_control_outcomes
        self.att_ = effects[valid_matches].mean()
        self.att_se_ = effects[valid_matches].std() / np.sqrt(valid_matches.sum())
        
        # Store matches for diagnostics
        self.matches_ = {
            'treated_idx': treated_idx[valid_matches],
            'control_idx': control_idx[indices[valid_matches, 0]],
            'distances': distances[valid_matches, 0],
            'effects': effects[valid_matches]
        }
        
        logger.info(f"Matched {valid_matches.sum()}/{len(treated_idx)} treated units")
    
    def get_att(self) -> Tuple[float, float, Tuple[float, float]]:
        """
        Get Average Treatment Effect on Treated.
        
        Returns
        -------
        att : float
            Point estimate
        se : float
            Standard error
        ci : tuple
            95% confidence interval
        """
        ci = (self.att_ - 1.96 * self.att_se_, 
              self.att_ + 1.96 * self.att_se_)
        return self.att_, self.att_se_, ci
    
    def balance_check(self, X: pd.DataFrame, treatment: pd.Series) -> pd.DataFrame:
        """
        Check covariate balance before and after matching.
        
        Returns DataFrame with standardized mean differences.
        """
        results = []
        
        for col in X.columns:
            # Before matching
            treated_mean = X.loc[treatment == 1, col].mean()
            control_mean = X.loc[treatment == 0, col].mean()
            pooled_std = X[col].std()
            smd_before = (treated_mean - control_mean) / pooled_std
            
            # After matching
            matched_treated = X.iloc[self.matches_['treated_idx']][col].mean()
            matched_control = X.iloc[self.matches_['control_idx']][col].mean()
            smd_after = (matched_treated - matched_control) / pooled_std
            
            results.append({
                'variable': col,
                'smd_before': smd_before,
                'smd_after': smd_after,
                'improvement': abs(smd_before) - abs(smd_after)
            })
        
        return pd.DataFrame(results)


# =============================================================================
# 4. DOUBLY ROBUST ESTIMATION
# =============================================================================

class DoublyRobustEstimator:
    """
    Augmented Inverse Probability Weighting (AIPW) Estimator.
    
    Consistent if either propensity score or outcome model is correct.
    
    Based on Robins, Rotnitzky & Zhao (1994).
    """
    
    def __init__(self):
        self.ate_ = None
        self.att_ = None
        self.propensity_scores_ = None
        self.outcome_models_ = {}
    
    def fit(self, X: pd.DataFrame, treatment: pd.Series,
            outcome: pd.Series) -> 'DoublyRobustEstimator':
        """
        Fit doubly robust estimator.
        
        Parameters
        ----------
        X : DataFrame
            Covariates
        treatment : Series
            Binary treatment (1 = treated)
        outcome : Series
            Outcome variable
        """
        n = len(outcome)
        D = treatment.values
        Y = outcome.values
        
        # Step 1: Propensity score
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        ps_model = LogisticRegression(max_iter=1000)
        ps_model.fit(X_scaled, D)
        e = ps_model.predict_proba(X_scaled)[:, 1]
        self.propensity_scores_ = e
        
        # Clip for numerical stability
        e = np.clip(e, 0.01, 0.99)
        
        # Step 2: Outcome models
        X_const = sm.add_constant(X)
        
        # E[Y|X, D=1]
        mask_1 = (D == 1)
        model_1 = sm.OLS(Y[mask_1], X_const[mask_1]).fit()
        mu_1 = model_1.predict(X_const)
        self.outcome_models_['treated'] = model_1
        
        # E[Y|X, D=0]
        mask_0 = (D == 0)
        model_0 = sm.OLS(Y[mask_0], X_const[mask_0]).fit()
        mu_0 = model_0.predict(X_const)
        self.outcome_models_['control'] = model_0
        
        # Step 3: AIPW estimator
        # τ_DR = (1/n) Σ [μ₁(X) - μ₀(X) + D(Y-μ₁(X))/e - (1-D)(Y-μ₀(X))/(1-e)]
        
        term1 = mu_1 - mu_0
        term2 = D * (Y - mu_1) / e
        term3 = (1 - D) * (Y - mu_0) / (1 - e)
        
        influence_function = term1 + term2 - term3
        
        self.ate_ = influence_function.mean()
        self.ate_se_ = influence_function.std() / np.sqrt(n)
        
        # ATT version
        att_if = D * (Y - mu_0) / D.mean() - (1 - D) * e * (Y - mu_0) / ((1 - e) * D.mean())
        self.att_ = att_if.mean()
        self.att_se_ = att_if.std() / np.sqrt(n)
        
        return self
    
    def get_ate(self) -> Tuple[float, float, Tuple[float, float]]:
        """Get Average Treatment Effect with SE and 95% CI."""
        ci = (self.ate_ - 1.96 * self.ate_se_,
              self.ate_ + 1.96 * self.ate_se_)
        return self.ate_, self.ate_se_, ci
    
    def get_att(self) -> Tuple[float, float, Tuple[float, float]]:
        """Get Average Treatment Effect on Treated."""
        ci = (self.att_ - 1.96 * self.att_se_,
              self.att_ + 1.96 * self.att_se_)
        return self.att_, self.att_se_, ci


# =============================================================================
# 5. HECKMAN SELECTION CORRECTION
# =============================================================================

class HeckmanSelection:
    """
    Heckman (1979) Two-Step Selection Correction.
    
    Corrects for sample selection bias when outcome is observed
    only for a non-random subsample.
    
    Parameters
    ----------
    method : str
        '2step' for two-step estimator, 'mle' for maximum likelihood
    """
    
    def __init__(self, method: str = '2step'):
        self.method = method
        self.selection_model_ = None
        self.outcome_model_ = None
        self.lambda_ = None  # Inverse Mills Ratio coefficient
        self.rho_ = None     # Selection correlation
    
    @staticmethod
    def inverse_mills_ratio(z: np.ndarray) -> np.ndarray:
        """
        Compute Inverse Mills Ratio: λ(z) = φ(z) / Φ(z)
        """
        # Avoid numerical issues at extremes
        z = np.clip(z, -5, 5)
        return norm.pdf(z) / norm.cdf(z)
    
    def fit(self, X_select: pd.DataFrame, X_outcome: pd.DataFrame,
            selected: pd.Series, outcome: pd.Series,
            exclusion_vars: List[str] = None) -> 'HeckmanSelection':
        """
        Fit Heckman two-step model.
        
        Parameters
        ----------
        X_select : DataFrame
            Covariates for selection equation (should include exclusion restrictions)
        X_outcome : DataFrame
            Covariates for outcome equation
        selected : Series
            Binary selection indicator (1 = observed)
        outcome : Series
            Outcome (observed only where selected=1)
        exclusion_vars : list
            Variables in selection but not outcome (for identification)
        """
        # Step 1: Probit selection equation
        X_s = sm.add_constant(X_select)
        probit = sm.Probit(selected, X_s)
        self.selection_model_ = probit.fit(disp=0)
        
        # Compute Inverse Mills Ratio for selected observations
        z_hat = self.selection_model_.predict(X_s)
        lambda_i = self.inverse_mills_ratio(z_hat)
        
        # Step 2: OLS with IMR correction (selected sample only)
        mask = (selected == 1)
        X_o = sm.add_constant(X_outcome[mask])
        X_o['lambda'] = lambda_i[mask]
        
        self.outcome_model_ = sm.OLS(outcome[mask], X_o).fit()
        
        # Extract selection parameters
        self.lambda_ = self.outcome_model_.params['lambda']
        self.lambda_se_ = self.outcome_model_.bse['lambda']
        
        # Test for selection: H0: λ = 0 (no selection)
        self.selection_test_stat_ = self.lambda_ / self.lambda_se_
        self.selection_test_pvalue_ = 2 * (1 - norm.cdf(abs(self.selection_test_stat_)))
        
        # Estimate ρ (correlation) and σ (outcome std)
        # σρ = λ coefficient
        # Need to estimate σ from residuals
        residuals = self.outcome_model_.resid
        sigma_hat = np.sqrt(np.var(residuals) + self.lambda_**2 * np.mean(lambda_i[mask] * (lambda_i[mask] + z_hat[mask])))
        self.sigma_ = sigma_hat
        self.rho_ = self.lambda_ / sigma_hat
        
        return self
    
    def predict(self, X_outcome: pd.DataFrame, X_select: pd.DataFrame = None,
                selection_corrected: bool = True) -> np.ndarray:
        """
        Predict outcomes.
        
        Parameters
        ----------
        X_outcome : DataFrame
            Covariates for prediction
        X_select : DataFrame, optional
            Selection covariates (needed for selection-corrected predictions)
        selection_corrected : bool
            If True, includes selection correction term
        """
        X_o = sm.add_constant(X_outcome)
        
        if selection_corrected and X_select is not None:
            X_s = sm.add_constant(X_select)
            z_hat = self.selection_model_.predict(X_s)
            lambda_i = self.inverse_mills_ratio(z_hat)
            X_o['lambda'] = lambda_i
        else:
            X_o['lambda'] = 0
        
        return self.outcome_model_.predict(X_o)
    
    def summary(self) -> Dict:
        """Get summary of selection model results."""
        return {
            'lambda_coefficient': self.lambda_,
            'lambda_se': self.lambda_se_,
            'rho': self.rho_,
            'sigma': self.sigma_,
            'selection_test_stat': self.selection_test_stat_,
            'selection_test_pvalue': self.selection_test_pvalue_,
            'selection_significant': self.selection_test_pvalue_ < 0.05,
            'outcome_r_squared': self.outcome_model_.rsquared
        }


# =============================================================================
# 6. SEGREGATION INDICES
# =============================================================================

class SegregationIndices:
    """
    Occupational and Industry Segregation Measures.
    
    Implements:
    - Duncan Dissimilarity Index (D)
    - Karmel-MacLachlan Index (KM)
    - Size-Standardized Index (IP)
    - Gini Segregation Index
    """
    
    @staticmethod
    def duncan_index(df: pd.DataFrame, occupation_col: str,
                     gender_col: str, female_code: int = 2,
                     weight_col: str = None) -> float:
        """
        Duncan & Duncan (1955) Dissimilarity Index.
        
        D = (1/2) Σ |F_j/F - M_j/M|
        
        Interpretation: Proportion of women (or men) who would need to
        change occupations for perfect integration.
        """
        if weight_col:
            crosstab = df.groupby([occupation_col, gender_col])[weight_col].sum().unstack(fill_value=0)
        else:
            crosstab = pd.crosstab(df[occupation_col], df[gender_col])
        
        female_col = female_code
        male_col = 1 if female_code == 2 else 2
        
        f_share = crosstab[female_col] / crosstab[female_col].sum()
        m_share = crosstab[male_col] / crosstab[male_col].sum()
        
        return 0.5 * np.abs(f_share - m_share).sum()
    
    @staticmethod
    def karmel_maclachlan_index(df: pd.DataFrame, occupation_col: str,
                                 gender_col: str, female_code: int = 2,
                                 weight_col: str = None) -> float:
        """
        Karmel & MacLachlan (1988) Index.
        
        KM = Σ |a·F_j - (1-a)·M_j| / T
        
        Where a = M/T (male share of total).
        """
        if weight_col:
            crosstab = df.groupby([occupation_col, gender_col])[weight_col].sum().unstack(fill_value=0)
        else:
            crosstab = pd.crosstab(df[occupation_col], df[gender_col])
        
        female_col = female_code
        male_col = 1 if female_code == 2 else 2
        
        F = crosstab[female_col]
        M = crosstab[male_col]
        T = F.sum() + M.sum()
        a = M.sum() / T
        
        return np.abs(a * F - (1-a) * M).sum() / T
    
    @staticmethod
    def ip_index(df: pd.DataFrame, occupation_col: str,
                 gender_col: str, female_code: int = 2,
                 weight_col: str = None) -> float:
        """
        Size-Standardized (IP) Index.
        
        IP = (1/2) Σ |F_j/T_j - F/T| × (T_j/T)
        
        Weighted by occupation size.
        """
        if weight_col:
            crosstab = df.groupby([occupation_col, gender_col])[weight_col].sum().unstack(fill_value=0)
        else:
            crosstab = pd.crosstab(df[occupation_col], df[gender_col])
        
        female_col = female_code
        male_col = 1 if female_code == 2 else 2
        
        F_j = crosstab[female_col]
        T_j = crosstab[female_col] + crosstab[male_col]
        F = F_j.sum()
        T = T_j.sum()
        
        p_j = F_j / T_j  # Female proportion in occupation j
        p_bar = F / T     # Overall female proportion
        s_j = T_j / T     # Occupation j's share of employment
        
        return 0.5 * (np.abs(p_j - p_bar) * s_j).sum()
    
    @staticmethod
    def female_share_by_occupation(df: pd.DataFrame, occupation_col: str,
                                    gender_col: str, female_code: int = 2,
                                    weight_col: str = None) -> pd.Series:
        """
        Calculate female share for each occupation.
        
        Used for devaluation hypothesis testing.
        """
        if weight_col:
            crosstab = df.groupby([occupation_col, gender_col])[weight_col].sum().unstack(fill_value=0)
        else:
            crosstab = pd.crosstab(df[occupation_col], df[gender_col])
        
        female_col = female_code
        total = crosstab.sum(axis=1)
        
        return crosstab[female_col] / total


# =============================================================================
# 7. CONVENIENCE FUNCTIONS
# =============================================================================

def run_full_decomposition(df: pd.DataFrame, 
                           outcome_col: str,
                           gender_col: str,
                           control_cols: List[str],
                           female_code: int = 2,
                           weight_col: str = None) -> Dict:
    """
    Run comprehensive wage gap decomposition.
    
    Performs:
    1. Oaxaca-Blinder at the mean
    2. RIF decomposition at deciles
    3. DFL reweighting
    4. Propensity score matching
    5. Doubly robust estimation
    
    Returns dictionary with all results.
    """
    results = {}
    
    # Prepare data
    df = df.dropna(subset=[outcome_col, gender_col] + control_cols)
    
    female_mask = df[gender_col] == female_code
    male_mask = df[gender_col] != female_code
    
    X = df[control_cols]
    y = df[outcome_col]
    treatment = female_mask.astype(int)
    
    X_m = X[male_mask]
    X_f = X[female_mask]
    y_m = y[male_mask]
    y_f = y[female_mask]
    
    # 1. Raw gap
    results['raw_gap'] = {
        'male_mean': y_m.mean(),
        'female_mean': y_f.mean(),
        'gap': y_m.mean() - y_f.mean(),
        'gap_pct': (y_m.mean() - y_f.mean()) / y_m.mean() * 100
    }
    
    # 2. RIF decomposition
    try:
        rif = RIFRegression()
        rif_results = rif.decomposition(X_m, X_f, y_m, y_f)
        results['rif_decomposition'] = {
            tau: {
                'total_gap': r.total_gap,
                'composition': r.composition_effect,
                'wage_structure': r.wage_structure_effect
            } for tau, r in rif_results.items()
        }
    except Exception as e:
        logger.warning(f"RIF decomposition failed: {e}")
        results['rif_decomposition'] = None
    
    # 3. DFL reweighting
    try:
        dfl = DFLReweighting()
        dfl.fit(X, (df[gender_col] != female_code).astype(int))
        dfl_results = dfl.decomposition(y, (df[gender_col] != female_code).astype(int))
        results['dfl_decomposition'] = dfl_results
    except Exception as e:
        logger.warning(f"DFL decomposition failed: {e}")
        results['dfl_decomposition'] = None
    
    # 4. Propensity score matching
    try:
        psm = PropensityScoreMatching(n_neighbors=5, caliper=0.1)
        psm.fit(X, treatment, y)
        att, se, ci = psm.get_att()
        results['psm'] = {
            'att': att,
            'se': se,
            'ci_lower': ci[0],
            'ci_upper': ci[1]
        }
    except Exception as e:
        logger.warning(f"PSM failed: {e}")
        results['psm'] = None
    
    # 5. Doubly robust
    try:
        dr = DoublyRobustEstimator()
        dr.fit(X, treatment, y)
        ate, se, ci = dr.get_ate()
        results['doubly_robust'] = {
            'ate': ate,
            'se': se,
            'ci_lower': ci[0],
            'ci_upper': ci[1]
        }
    except Exception as e:
        logger.warning(f"Doubly robust estimation failed: {e}")
        results['doubly_robust'] = None
    
    return results


def compute_glass_ceiling_index(df: pd.DataFrame,
                                 outcome_col: str,
                                 gender_col: str,
                                 female_code: int = 2) -> Dict:
    """
    Compute glass ceiling indicators.
    
    Compares gender gaps at different quantiles to identify
    glass ceiling (larger gaps at top) or sticky floor (larger at bottom).
    """
    female_mask = df[gender_col] == female_code
    
    y_m = df.loc[~female_mask, outcome_col].values
    y_f = df.loc[female_mask, outcome_col].values
    
    quantiles = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    gaps = {}
    
    for q in quantiles:
        q_m = np.quantile(y_m, q)
        q_f = np.quantile(y_f, q)
        gaps[q] = {
            'male': q_m,
            'female': q_f,
            'gap': q_m - q_f,
            'gap_pct': (q_m - q_f) / q_m * 100
        }
    
    # Glass ceiling index: Gap at 90th / Gap at 50th
    glass_ceiling_index = gaps[0.90]['gap'] / gaps[0.50]['gap'] if gaps[0.50]['gap'] != 0 else np.nan
    
    # Sticky floor index: Gap at 10th / Gap at 50th  
    sticky_floor_index = gaps[0.10]['gap'] / gaps[0.50]['gap'] if gaps[0.50]['gap'] != 0 else np.nan
    
    return {
        'quantile_gaps': gaps,
        'glass_ceiling_index': glass_ceiling_index,
        'sticky_floor_index': sticky_floor_index,
        'interpretation': {
            'glass_ceiling': glass_ceiling_index > 1.1,  # Gap 10%+ larger at top
            'sticky_floor': sticky_floor_index > 1.1     # Gap 10%+ larger at bottom
        }
    }


# =============================================================================
# 8. BROWN-MOON-ZOLOTH (1980) DECOMPOSITION
# =============================================================================

class BrownMoonZoloth:
    """
    Brown, Moon & Zoloth (1980) Occupational Attainment Decomposition.
    
    Decomposes the wage gap into:
    1. Within-occupation wage differences (given occupation)
    2. Occupational access differences (between occupation)
    3. Interaction term
    
    The wage gap can be written as:
    W̄_m - W̄_f = Σⱼ P^m_j(W̄^m_j - W̄^f_j)     [Within-occupation]
               + Σⱼ W̄^f_j(P^m_j - P^f_j)       [Between-occupation]
               + Σⱼ (P^m_j - P^f_j)(W̄^m_j - W̄^f_j - (W̄_m - W̄_f))  [Interaction]
    
    References:
    -----------
    Brown, R.S., Moon, M., & Zoloth, B.S. (1980). "Incorporating Occupational
    Attainment in Studies of Male-Female Earnings Differentials."
    Journal of Human Resources, 15(1), 3-28.
    """
    
    def __init__(self):
        self.results_ = None
        self.occupation_model_ = None
        self.wage_models_ = {}
    
    def fit(self, df: pd.DataFrame, 
            wage_col: str,
            gender_col: str,
            occupation_col: str,
            control_cols: List[str],
            female_code: int = 2,
            weight_col: str = None) -> 'BrownMoonZoloth':
        """
        Fit the Brown-Moon-Zoloth decomposition.
        
        Parameters
        ----------
        df : DataFrame
            Data with wages, gender, occupation, and controls
        wage_col : str
            Wage variable (recommend log wages)
        gender_col : str
            Gender indicator
        occupation_col : str
            Occupation category variable
        control_cols : list
            Control variables for wage equations
        female_code : int
            Code for female in gender column
        weight_col : str, optional
            Sample weights
        """
        df = df.dropna(subset=[wage_col, gender_col, occupation_col] + control_cols)
        
        female_mask = df[gender_col] == female_code
        occupations = df[occupation_col].unique()
        
        # Step 1: Calculate occupation distributions
        if weight_col:
            total_m = df.loc[~female_mask, weight_col].sum()
            total_f = df.loc[female_mask, weight_col].sum()
            
            P_m = {}  # Male occupation shares
            P_f = {}  # Female occupation shares
            for occ in occupations:
                mask_occ = df[occupation_col] == occ
                P_m[occ] = df.loc[mask_occ & ~female_mask, weight_col].sum() / total_m
                P_f[occ] = df.loc[mask_occ & female_mask, weight_col].sum() / total_f
        else:
            n_m = (~female_mask).sum()
            n_f = female_mask.sum()
            P_m = {occ: ((df[occupation_col] == occ) & ~female_mask).sum() / n_m 
                   for occ in occupations}
            P_f = {occ: ((df[occupation_col] == occ) & female_mask).sum() / n_f 
                   for occ in occupations}
        
        # Step 2: Calculate mean wages by occupation and gender
        W_m_j = {}  # Male mean wage in occupation j
        W_f_j = {}  # Female mean wage in occupation j
        
        for occ in occupations:
            mask_occ = df[occupation_col] == occ
            if weight_col:
                w_m = df.loc[mask_occ & ~female_mask, wage_col]
                wt_m = df.loc[mask_occ & ~female_mask, weight_col]
                W_m_j[occ] = np.average(w_m, weights=wt_m) if len(w_m) > 0 else np.nan
                
                w_f = df.loc[mask_occ & female_mask, wage_col]
                wt_f = df.loc[mask_occ & female_mask, weight_col]
                W_f_j[occ] = np.average(w_f, weights=wt_f) if len(w_f) > 0 else np.nan
            else:
                W_m_j[occ] = df.loc[mask_occ & ~female_mask, wage_col].mean()
                W_f_j[occ] = df.loc[mask_occ & female_mask, wage_col].mean()
        
        # Overall means
        if weight_col:
            W_m = np.average(df.loc[~female_mask, wage_col], 
                            weights=df.loc[~female_mask, weight_col])
            W_f = np.average(df.loc[female_mask, wage_col], 
                            weights=df.loc[female_mask, weight_col])
        else:
            W_m = df.loc[~female_mask, wage_col].mean()
            W_f = df.loc[female_mask, wage_col].mean()
        
        # Step 3: Decomposition
        total_gap = W_m - W_f
        
        # Within-occupation: Σⱼ P^m_j (W̄^m_j - W̄^f_j)
        within = 0
        within_detail = {}
        for occ in occupations:
            if not np.isnan(W_m_j[occ]) and not np.isnan(W_f_j[occ]):
                contrib = P_m[occ] * (W_m_j[occ] - W_f_j[occ])
                within += contrib
                within_detail[occ] = contrib
        
        # Between-occupation: Σⱼ W̄^f_j (P^m_j - P^f_j)
        between = 0
        between_detail = {}
        for occ in occupations:
            if not np.isnan(W_f_j[occ]):
                contrib = W_f_j[occ] * (P_m[occ] - P_f[occ])
                between += contrib
                between_detail[occ] = contrib
        
        # Interaction: Σⱼ (P^m_j - P^f_j)(W̄^m_j - W̄^f_j - total_gap)
        interaction = 0
        interaction_detail = {}
        for occ in occupations:
            if not np.isnan(W_m_j[occ]) and not np.isnan(W_f_j[occ]):
                wage_diff = W_m_j[occ] - W_f_j[occ]
                contrib = (P_m[occ] - P_f[occ]) * (wage_diff - total_gap)
                interaction += contrib
                interaction_detail[occ] = contrib
        
        self.results_ = {
            'total_gap': total_gap,
            'within_occupation': within,
            'between_occupation': between,
            'interaction': interaction,
            'decomposition_check': within + between + interaction,  # Should ≈ total_gap
            'within_share': within / total_gap * 100 if total_gap != 0 else 0,
            'between_share': between / total_gap * 100 if total_gap != 0 else 0,
            'interaction_share': interaction / total_gap * 100 if total_gap != 0 else 0,
            'occupation_details': {
                'P_male': P_m,
                'P_female': P_f,
                'W_male': W_m_j,
                'W_female': W_f_j,
                'within_contrib': within_detail,
                'between_contrib': between_detail
            },
            'interpretation': {
                'within': "Wage discrimination within occupations",
                'between': "Occupational segregation/access barriers",
                'interaction': "Covariance of sorting and discrimination"
            }
        }
        
        return self
    
    def summary(self) -> Dict:
        """Return decomposition summary."""
        return self.results_


# =============================================================================
# 9. MACHADO-MATA (2005) QUANTILE DECOMPOSITION
# =============================================================================

class MachadoMata:
    """
    Machado & Mata (2005) Quantile Decomposition.
    
    Constructs counterfactual wage distributions by combining
    quantile regression coefficients from one group with covariate
    distribution from another.
    
    The method:
    1. Estimate quantile regressions for each group at τ ∈ (0,1)
    2. Draw random sample of θ ~ U(0,1)
    3. For each θ, predict wages using:
       - Counterfactual: Female X with male β(θ)
    4. Compare distributions
    
    References:
    -----------
    Machado, J.A.F., & Mata, J. (2005). "Counterfactual Decomposition of
    Changes in Wage Distributions using Quantile Regression."
    Journal of Applied Econometrics, 20(4), 445-465.
    """
    
    def __init__(self, n_quantiles: int = 99, n_bootstrap: int = 100):
        """
        Parameters
        ----------
        n_quantiles : int
            Number of quantiles to estimate (default 99 = percentiles)
        n_bootstrap : int
            Number of bootstrap replications for SE
        """
        self.n_quantiles = n_quantiles
        self.n_bootstrap = n_bootstrap
        self.quantiles_ = np.linspace(0.01, 0.99, n_quantiles)
        self.coef_male_ = {}
        self.coef_female_ = {}
        self.results_ = None
    
    def fit(self, df: pd.DataFrame,
            wage_col: str,
            gender_col: str,
            control_cols: List[str],
            female_code: int = 2) -> 'MachadoMata':
        """
        Fit quantile regressions and compute decomposition.
        
        Parameters
        ----------
        df : DataFrame
            Data with wages, gender, and controls
        wage_col : str
            Wage variable (log wages recommended)
        gender_col : str
            Gender indicator
        control_cols : list
            Control variables
        female_code : int
            Code for female
        """
        df = df.dropna(subset=[wage_col, gender_col] + control_cols)
        
        female_mask = df[gender_col] == female_code
        
        X_m = sm.add_constant(df.loc[~female_mask, control_cols])
        X_f = sm.add_constant(df.loc[female_mask, control_cols])
        y_m = df.loc[~female_mask, wage_col]
        y_f = df.loc[female_mask, wage_col]
        
        # Step 1: Estimate quantile regressions at each quantile
        logger.info(f"Estimating {self.n_quantiles} quantile regressions for each group...")
        
        for tau in self.quantiles_:
            try:
                # Male quantile regression
                qr_m = QuantReg(y_m, X_m).fit(q=tau, max_iter=1000)
                self.coef_male_[tau] = qr_m.params
                
                # Female quantile regression
                qr_f = QuantReg(y_f, X_f).fit(q=tau, max_iter=1000)
                self.coef_female_[tau] = qr_f.params
            except Exception as e:
                logger.warning(f"Quantile {tau:.2f} failed: {e}")
        
        # Step 2: Construct distributions
        n_draw = len(X_f)  # Sample size for counterfactual
        
        # Actual distributions
        y_m_dist = y_m.values
        y_f_dist = y_f.values
        
        # Counterfactual: Female X, Male coefficients
        # Draw random quantiles
        np.random.seed(42)
        theta_draws = np.random.uniform(0.01, 0.99, n_draw)
        
        # For each draw, find nearest estimated quantile
        y_cf = []
        for i, theta in enumerate(theta_draws):
            nearest_tau = min(self.coef_male_.keys(), key=lambda x: abs(x - theta))
            beta = self.coef_male_[nearest_tau]
            x_i = X_f.iloc[i % len(X_f)].values
            y_cf.append(x_i @ beta)
        
        y_cf = np.array(y_cf)
        
        # Step 3: Decomposition at each percentile
        percentiles = list(range(1, 100))
        decomposition = []
        
        for p in percentiles:
            q_m = np.percentile(y_m_dist, p)
            q_f = np.percentile(y_f_dist, p)
            q_cf = np.percentile(y_cf, p)
            
            total = q_m - q_f
            composition = q_cf - q_f  # Due to X differences
            wage_structure = q_m - q_cf  # Due to β differences
            
            decomposition.append({
                'percentile': p,
                'male': q_m,
                'female': q_f,
                'counterfactual': q_cf,
                'total_gap': total,
                'composition_effect': composition,
                'wage_structure_effect': wage_structure
            })
        
        self.results_ = pd.DataFrame(decomposition)
        
        # Summary statistics
        self.summary_stats_ = {
            'mean_total_gap': self.results_['total_gap'].mean(),
            'mean_composition': self.results_['composition_effect'].mean(),
            'mean_wage_structure': self.results_['wage_structure_effect'].mean(),
            'gap_at_10th': self.results_.loc[self.results_['percentile'] == 10, 'total_gap'].values[0],
            'gap_at_50th': self.results_.loc[self.results_['percentile'] == 50, 'total_gap'].values[0],
            'gap_at_90th': self.results_.loc[self.results_['percentile'] == 90, 'total_gap'].values[0],
            'n_male': len(y_m),
            'n_female': len(y_f)
        }
        
        return self
    
    def get_decomposition(self) -> pd.DataFrame:
        """Return full decomposition results."""
        return self.results_
    
    def summary(self) -> Dict:
        """Return summary statistics."""
        return self.summary_stats_


# =============================================================================
# 10. KLEVEN CHILD PENALTY METHOD
# =============================================================================

class KlevenChildPenalty:
    """
    Kleven, Landais & Søgaard (2019) Child Penalty Event Study.
    
    Estimates the impact of children on labor market outcomes using
    an event study design around first childbirth.
    
    The specification:
    Y_{ist} = Σⱼ αⱼ I[j=t] + Σₖ βₖ AgeGroupₖ + γ Year_s + ν_{ist}
    
    where t is event time (years relative to first birth), and the
    child penalty at event time j is:
    P_j = α̂_j / E[Ỹ_{ist}|j]
    
    where Ỹ is the predicted outcome absent children.
    
    References:
    -----------
    Kleven, H., Landais, C., & Søgaard, J.E. (2019). "Children and Gender
    Inequality: Evidence from Denmark." American Economic Journal: Applied
    Economics, 11(4), 181-209.
    """
    
    def __init__(self, pre_periods: int = 5, post_periods: int = 10):
        """
        Parameters
        ----------
        pre_periods : int
            Number of pre-birth periods to include
        post_periods : int
            Number of post-birth periods to include
        """
        self.pre_periods = pre_periods
        self.post_periods = post_periods
        self.event_coefs_ = None
        self.results_ = None
    
    def fit(self, df: pd.DataFrame,
            outcome_col: str,
            event_time_col: str,
            gender_col: str,
            age_col: str = None,
            year_col: str = None,
            female_code: int = 2) -> 'KlevenChildPenalty':
        """
        Estimate child penalty event study.
        
        Parameters
        ----------
        df : DataFrame
            Panel data with individual-year observations
        outcome_col : str
            Outcome (earnings, hours, employment, wage rate)
        event_time_col : str
            Years relative to first birth (negative = before, 0 = birth year)
        gender_col : str
            Gender indicator
        age_col : str, optional
            Age variable for age group controls
        year_col : str, optional
            Calendar year for year fixed effects
        female_code : int
            Code for female
        """
        df = df.copy()
        
        # Restrict to event window
        df = df[(df[event_time_col] >= -self.pre_periods) & 
                (df[event_time_col] <= self.post_periods)]
        
        # Create event time dummies (omit t=-1 as reference)
        event_times = range(-self.pre_periods, self.post_periods + 1)
        for t in event_times:
            if t != -1:  # Omit reference period
                df[f'event_{t}'] = (df[event_time_col] == t).astype(int)
        
        event_dummies = [f'event_{t}' for t in event_times if t != -1]
        
        # Create age group dummies if provided
        control_vars = []
        if age_col:
            df['age_group'] = pd.cut(df[age_col], bins=[0, 25, 30, 35, 40, 45, 50, 100],
                                      labels=['<25', '25-29', '30-34', '35-39', '40-44', '45-49', '50+'])
            age_dummies = pd.get_dummies(df['age_group'], prefix='age', drop_first=True)
            df = pd.concat([df, age_dummies], axis=1)
            control_vars.extend(age_dummies.columns.tolist())
        
        if year_col:
            year_dummies = pd.get_dummies(df[year_col], prefix='year', drop_first=True)
            df = pd.concat([df, year_dummies], axis=1)
            control_vars.extend(year_dummies.columns.tolist())
        
        # Separate by gender
        female_mask = df[gender_col] == female_code
        
        results = {'female': {}, 'male': {}}
        
        for gender, mask in [('female', female_mask), ('male', ~female_mask)]:
            df_g = df[mask].dropna(subset=[outcome_col] + event_dummies)
            
            if len(df_g) < 100:
                logger.warning(f"Insufficient data for {gender}")
                continue
            
            # Regression
            X = sm.add_constant(df_g[event_dummies + control_vars])
            y = df_g[outcome_col]
            
            model = sm.OLS(y, X).fit(cov_type='cluster', 
                                     cov_kwds={'groups': df_g.index})
            
            # Extract event coefficients
            event_coefs = {t: model.params.get(f'event_{t}', 0) 
                          for t in event_times}
            event_coefs[-1] = 0  # Reference period
            
            event_se = {t: model.bse.get(f'event_{t}', 0) 
                       for t in event_times}
            event_se[-1] = 0
            
            # Counterfactual (predicted Y without children = pre-trend extrapolation)
            pre_mean = df_g[df_g[event_time_col] < 0][outcome_col].mean()
            
            # Child penalty as percentage
            penalty = {t: (coef / pre_mean * 100) if pre_mean != 0 else 0
                      for t, coef in event_coefs.items()}
            
            results[gender] = {
                'coefficients': event_coefs,
                'standard_errors': event_se,
                'penalty_pct': penalty,
                'pre_mean': pre_mean,
                'n_obs': len(df_g)
            }
        
        # Compute gender gap in child penalty
        if 'female' in results and 'male' in results:
            penalty_gap = {}
            for t in event_times:
                f_pen = results['female']['penalty_pct'].get(t, 0)
                m_pen = results['male']['penalty_pct'].get(t, 0)
                penalty_gap[t] = f_pen - m_pen
            
            results['penalty_gap'] = penalty_gap
            
            # Long-run penalty (average of t >= 5)
            long_run_times = [t for t in event_times if t >= 5]
            if long_run_times:
                results['long_run_female_penalty'] = np.mean(
                    [results['female']['penalty_pct'][t] for t in long_run_times]
                )
                results['long_run_male_penalty'] = np.mean(
                    [results['male']['penalty_pct'][t] for t in long_run_times]
                )
                results['long_run_gap'] = (results['long_run_female_penalty'] - 
                                           results['long_run_male_penalty'])
        
        self.results_ = results
        return self
    
    def summary(self) -> Dict:
        """Return summary of child penalty estimates."""
        return self.results_
    
    def get_event_study_data(self) -> pd.DataFrame:
        """Return data for plotting event study."""
        rows = []
        for gender in ['female', 'male']:
            if gender in self.results_:
                for t, coef in self.results_[gender]['coefficients'].items():
                    rows.append({
                        'event_time': t,
                        'gender': gender,
                        'coefficient': coef,
                        'se': self.results_[gender]['standard_errors'][t],
                        'penalty_pct': self.results_[gender]['penalty_pct'][t]
                    })
        return pd.DataFrame(rows)


# =============================================================================
# 11. AGE-PERIOD-COHORT (APC) DECOMPOSITION
# =============================================================================

class APCDecomposition:
    """
    Age-Period-Cohort Decomposition for Wage Gap Analysis.
    
    Separates wage gap dynamics into:
    - Age effects: lifecycle wage growth patterns by gender
    - Period effects: time-specific shocks (recessions, policy)
    - Cohort effects: birth cohort differences (generational change)
    
    The identification problem: Age + Cohort = Period
    
    Solutions implemented:
    1. Intrinsic Estimator (Yang et al. 2004)
    2. Constrained approaches (normalize one effect)
    
    References:
    -----------
    Yang, Y., Fu, W.J., & Land, K.C. (2004). "A Methodological Comparison
    of Age-Period-Cohort Models." Sociological Methodology, 34(1), 75-118.
    """
    
    def __init__(self, method: str = 'ie'):
        """
        Parameters
        ----------
        method : str
            'ie' for Intrinsic Estimator
            'constrained' for constrained regression
        """
        self.method = method
        self.results_ = None
    
    def _create_apc_design(self, age: np.ndarray, period: np.ndarray, 
                            cohort: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Create design matrix for APC model."""
        # Get unique values
        ages = np.unique(age)
        periods = np.unique(period)
        cohorts = np.unique(cohort)
        
        n = len(age)
        
        # Create dummy matrices (dropping first category)
        A = np.zeros((n, len(ages) - 1))
        P = np.zeros((n, len(periods) - 1))
        C = np.zeros((n, len(cohorts) - 1))
        
        for i, a in enumerate(ages[1:]):
            A[:, i] = (age == a).astype(float)
        
        for i, p in enumerate(periods[1:]):
            P[:, i] = (period == p).astype(float)
        
        for i, c in enumerate(cohorts[1:]):
            C[:, i] = (cohort == c).astype(float)
        
        X = np.hstack([np.ones((n, 1)), A, P, C])
        
        info = {
            'ages': ages,
            'periods': periods,
            'cohorts': cohorts,
            'n_age': len(ages) - 1,
            'n_period': len(periods) - 1,
            'n_cohort': len(cohorts) - 1
        }
        
        return X, info
    
    def fit(self, df: pd.DataFrame,
            outcome_col: str,
            age_col: str,
            period_col: str,
            gender_col: str = None,
            female_code: int = 2) -> 'APCDecomposition':
        """
        Fit APC decomposition.
        
        Parameters
        ----------
        df : DataFrame
            Data with outcome, age, and period
        outcome_col : str
            Outcome variable (e.g., log wage)
        age_col : str
            Age variable (will be binned into 5-year groups)
        period_col : str
            Period/year variable
        gender_col : str, optional
            If provided, estimates separate APC for each gender
        female_code : int
            Code for female
        """
        df = df.copy()
        
        # Create age groups (5-year bins)
        df['age_group'] = (df[age_col] // 5) * 5
        
        # Cohort = period - age
        df['cohort'] = df[period_col] - df[age_col]
        df['cohort_group'] = (df['cohort'] // 5) * 5
        
        if gender_col:
            results = {}
            for gender_name, mask in [('female', df[gender_col] == female_code),
                                       ('male', df[gender_col] != female_code)]:
                df_g = df[mask].dropna(subset=[outcome_col, 'age_group', period_col, 'cohort_group'])
                results[gender_name] = self._fit_single(df_g, outcome_col)
            
            # Compute gap in effects
            results['gap_analysis'] = self._compute_gaps(results)
            self.results_ = results
        else:
            df = df.dropna(subset=[outcome_col, 'age_group', period_col, 'cohort_group'])
            self.results_ = self._fit_single(df, outcome_col)
        
        return self
    
    def _fit_single(self, df: pd.DataFrame, outcome_col: str) -> Dict:
        """Fit APC model for a single group."""
        age = df['age_group'].values
        period = df[outcome_col.replace(outcome_col, 'YEAR') if 'YEAR' in df.columns 
                    else df.columns[df.columns.str.contains('year|YEAR', case=False)][0]].values \
                 if any(df.columns.str.contains('year|YEAR', case=False)) else df.index.values
        
        # Use the actual period column
        for col in df.columns:
            if 'year' in col.lower() or 'period' in col.lower():
                period = df[col].values
                break
        
        cohort = df['cohort_group'].values
        y = df[outcome_col].values
        
        # Create design matrix
        X, info = self._create_apc_design(age, period, cohort)
        
        if self.method == 'ie':
            # Intrinsic Estimator via constrained least squares
            try:
                from scipy.linalg import svd
                
                # Center columns (except intercept)
                X_centered = X.copy()
                X_centered[:, 1:] = X[:, 1:] - X[:, 1:].mean(axis=0)
                
                # OLS on centered design
                coef = np.linalg.lstsq(X_centered, y, rcond=None)[0]
                
                # Reconstruct effects
                n_a = info['n_age']
                n_p = info['n_period']
                n_c = info['n_cohort']
                
                intercept = coef[0]
                age_effects = np.concatenate([[0], coef[1:1+n_a]])
                period_effects = np.concatenate([[0], coef[1+n_a:1+n_a+n_p]])
                cohort_effects = np.concatenate([[0], coef[1+n_a+n_p:]])
                
            except Exception as e:
                logger.warning(f"Intrinsic estimator failed: {e}")
                return {'error': str(e)}
        else:
            # Simple constrained: fix first cohort effect = 0
            model = sm.OLS(y, X).fit()
            coef = model.params
            
            n_a = info['n_age']
            n_p = info['n_period']
            
            intercept = coef[0]
            age_effects = np.concatenate([[0], coef[1:1+n_a]])
            period_effects = np.concatenate([[0], coef[1+n_a:1+n_a+n_p]])
            cohort_effects = np.concatenate([[0], coef[1+n_a+n_p:]])
        
        return {
            'intercept': intercept,
            'age_effects': dict(zip(info['ages'], age_effects)),
            'period_effects': dict(zip(info['periods'], period_effects)),
            'cohort_effects': dict(zip(info['cohorts'], cohort_effects)),
            'info': info,
            'n_obs': len(y)
        }
    
    def _compute_gaps(self, results: Dict) -> Dict:
        """Compute gender gaps in APC effects."""
        gaps = {}
        
        if 'female' in results and 'male' in results:
            # Age effect gaps
            ages_f = results['female']['age_effects']
            ages_m = results['male']['age_effects']
            common_ages = set(ages_f.keys()) & set(ages_m.keys())
            gaps['age_gaps'] = {a: ages_m[a] - ages_f[a] for a in common_ages}
            
            # Period effect gaps
            periods_f = results['female']['period_effects']
            periods_m = results['male']['period_effects']
            common_periods = set(periods_f.keys()) & set(periods_m.keys())
            gaps['period_gaps'] = {p: periods_m[p] - periods_f[p] for p in common_periods}
            
            # Cohort effect gaps
            cohorts_f = results['female']['cohort_effects']
            cohorts_m = results['male']['cohort_effects']
            common_cohorts = set(cohorts_f.keys()) & set(cohorts_m.keys())
            gaps['cohort_gaps'] = {c: cohorts_m[c] - cohorts_f[c] for c in common_cohorts}
        
        return gaps
    
    def summary(self) -> Dict:
        """Return APC decomposition results."""
        return self.results_

# =============================================================================
# 12. STAGGERED DIFFERENCE-IN-DIFFERENCES
# =============================================================================

class StaggeredDiD:
    """
    Staggered Difference-in-Differences with Heterogeneous Treatment Effects.
    
    Implements methods robust to treatment effect heterogeneity:
    1. Goodman-Bacon (2021) decomposition (diagnostic)
    2. Callaway-Sant'Anna (2021) group-time ATT
    3. Simple robust aggregation
    
    These methods address the "bad controls" problem in two-way fixed effects
    when treatment timing varies across units.
    
    References:
    -----------
    Goodman-Bacon, A. (2021). "Difference-in-Differences with Variation in
    Treatment Timing." Journal of Econometrics.
    
    Callaway, B. & Sant'Anna, P.H.C. (2021). "Difference-in-Differences with
    Multiple Time Periods." Journal of Econometrics.
    """
    
    def __init__(self):
        self.results_ = None
    
    def goodman_bacon_decomposition(self, df: pd.DataFrame,
                                     outcome_col: str,
                                     treat_col: str,
                                     unit_col: str,
                                     time_col: str) -> Dict:
        """
        Goodman-Bacon (2021) decomposition of TWFE estimator.
        
        Decomposes the TWFE DD estimate into a weighted average of
        all possible 2x2 DD comparisons:
        - Earlier vs Later treated
        - Later vs Earlier treated  
        - Treated vs Never treated
        
        Parameters
        ----------
        df : DataFrame
            Panel data
        outcome_col : str
            Outcome variable
        treat_col : str
            Treatment indicator (0/1)
        unit_col : str
            Unit identifier
        time_col : str
            Time period
        """
        df = df.copy()
        
        # Get treatment timing for each unit
        treat_timing = df[df[treat_col] == 1].groupby(unit_col)[time_col].min()
        df['treat_time'] = df[unit_col].map(treat_timing).fillna(np.inf)
        
        # Identify groups
        never_treated = df[df['treat_time'] == np.inf][unit_col].unique()
        ever_treated = df[df['treat_time'] != np.inf][unit_col].unique()
        
        # Group by treatment timing
        timing_groups = df[df['treat_time'] != np.inf].groupby('treat_time')[unit_col].unique().to_dict()
        
        # Calculate 2x2 DD estimates for each comparison
        comparisons = []
        
        # Treated vs Never Treated
        for timing, units in timing_groups.items():
            if len(never_treated) > 0:
                # Pre-period mean
                pre_treat = df[(df[unit_col].isin(units)) & (df[time_col] < timing)][outcome_col].mean()
                pre_control = df[(df[unit_col].isin(never_treated)) & (df[time_col] < timing)][outcome_col].mean()
                
                # Post-period mean
                post_treat = df[(df[unit_col].isin(units)) & (df[time_col] >= timing)][outcome_col].mean()
                post_control = df[(df[unit_col].isin(never_treated)) & (df[time_col] >= timing)][outcome_col].mean()
                
                dd = (post_treat - pre_treat) - (post_control - pre_control)
                
                # Weight (simplified - proportional to sample size)
                n_treat = len(units)
                n_control = len(never_treated)
                n_pre = len(df[df[time_col] < timing])
                n_post = len(df[df[time_col] >= timing])
                
                weight = (n_treat * n_control * n_pre * n_post) / (len(df) ** 2)
                
                comparisons.append({
                    'type': 'Treated vs Never',
                    'timing': timing,
                    'dd_estimate': dd,
                    'weight': weight,
                    'n_treat': n_treat,
                    'n_control': n_control
                })
        
        # Normalize weights
        total_weight = sum(c['weight'] for c in comparisons)
        for c in comparisons:
            c['weight_normalized'] = c['weight'] / total_weight if total_weight > 0 else 0
        
        # Compute overall TWFE estimate
        twfe_estimate = sum(c['dd_estimate'] * c['weight_normalized'] for c in comparisons)
        
        self.results_ = {
            'twfe_estimate': twfe_estimate,
            'comparisons': comparisons,
            'n_treated_groups': len(timing_groups),
            'n_never_treated': len(never_treated),
            'warning': 'Simplified decomposition - for full implementation use bacondecomp package'
        }
        
        return self.results_
    
    def callaway_santanna(self, df: pd.DataFrame,
                          outcome_col: str,
                          treat_col: str,
                          unit_col: str,
                          time_col: str,
                          control_cols: List[str] = None) -> Dict:
        """
        Callaway-Sant'Anna (2021) group-time ATT.
        
        Estimates treatment effects separately for each (group, time) pair,
        where group is defined by treatment timing.
        
        Parameters
        ----------
        df : DataFrame
            Panel data
        outcome_col : str
            Outcome variable
        treat_col : str
            Treatment indicator
        unit_col : str
            Unit identifier
        time_col : str
            Time period
        control_cols : list, optional
            Control variables for doubly-robust estimation
        """
        df = df.copy()
        
        # Treatment timing
        treat_timing = df[df[treat_col] == 1].groupby(unit_col)[time_col].min()
        df['g'] = df[unit_col].map(treat_timing).fillna(0)  # 0 = never treated
        
        # Get all groups and time periods
        groups = sorted([g for g in df['g'].unique() if g > 0])
        times = sorted(df[time_col].unique())
        
        # Never treated as comparison
        never_treated = df[df['g'] == 0][unit_col].unique()
        
        # Compute group-time ATT for each (g, t) pair
        att_gt = {}
        
        for g in groups:
            g_units = df[df['g'] == g][unit_col].unique()
            
            for t in times:
                if t < g:
                    continue  # Pre-treatment
                
                # Outcome change for treated group
                y_g_t = df[(df[unit_col].isin(g_units)) & (df[time_col] == t)][outcome_col].mean()
                y_g_pre = df[(df[unit_col].isin(g_units)) & (df[time_col] == g - 1)][outcome_col].mean()
                
                # Outcome change for never-treated
                y_0_t = df[(df[unit_col].isin(never_treated)) & (df[time_col] == t)][outcome_col].mean()
                y_0_pre = df[(df[unit_col].isin(never_treated)) & (df[time_col] == g - 1)][outcome_col].mean()
                
                if not np.isnan(y_g_t) and not np.isnan(y_0_t):
                    att = (y_g_t - y_g_pre) - (y_0_t - y_0_pre)
                    att_gt[(g, t)] = {
                        'att': att,
                        'n_treated': len(g_units),
                        'event_time': t - g
                    }
        
        # Aggregate to event-time effects
        event_time_effects = {}
        for (g, t), result in att_gt.items():
            e = result['event_time']
            if e not in event_time_effects:
                event_time_effects[e] = []
            event_time_effects[e].append((result['att'], result['n_treated']))
        
        # Weighted average by group size
        aggregated = {}
        for e, effects in event_time_effects.items():
            total_n = sum(n for _, n in effects)
            weighted_att = sum(att * n / total_n for att, n in effects) if total_n > 0 else 0
            aggregated[e] = weighted_att
        
        # Overall ATT
        all_effects = [(result['att'], result['n_treated']) for result in att_gt.values()]
        total_n = sum(n for _, n in all_effects)
        overall_att = sum(att * n / total_n for att, n in all_effects) if total_n > 0 else 0
        
        self.results_ = {
            'group_time_att': att_gt,
            'event_time_effects': aggregated,
            'overall_att': overall_att,
            'n_groups': len(groups),
            'n_periods': len(times)
        }
        
        return self.results_
    
    def summary(self) -> Dict:
        """Return staggered DiD results."""
        return self.results_


# =============================================================================
# 13. SYNTHETIC CONTROL METHOD
# =============================================================================

class SyntheticControl:
    """
    Abadie-Diamond-Hainmueller (2010) Synthetic Control Method.
    
    Constructs a synthetic comparison unit as a weighted combination
    of donor units to estimate treatment effects.
    
    The synthetic control minimizes pre-treatment outcome differences:
    W* = argmin (X_1 - X_0 W)'V(X_1 - X_0 W)
    
    subject to W >= 0 and sum(W) = 1.
    
    References:
    -----------
    Abadie, A., Diamond, A., & Hainmueller, J. (2010). "Synthetic Control
    Methods for Comparative Case Studies." Journal of the American
    Statistical Association.
    """
    
    def __init__(self):
        self.weights_ = None
        self.results_ = None
    
    def fit(self, df: pd.DataFrame,
            outcome_col: str,
            unit_col: str,
            time_col: str,
            treated_unit: Any,
            treatment_time: Any,
            predictor_cols: List[str] = None) -> 'SyntheticControl':
        """
        Fit synthetic control.
        
        Parameters
        ----------
        df : DataFrame
            Panel data
        outcome_col : str
            Outcome variable
        unit_col : str
            Unit identifier
        time_col : str
            Time period
        treated_unit : Any
            Identifier of treated unit
        treatment_time : Any
            Time when treatment begins
        predictor_cols : list, optional
            Predictor variables (default: pre-treatment outcomes)
        """
        df = df.copy()
        
        # Split pre/post treatment
        pre_data = df[df[time_col] < treatment_time]
        post_data = df[df[time_col] >= treatment_time]
        
        # Identify donor pool
        donors = [u for u in df[unit_col].unique() if u != treated_unit]
        
        # Create predictor matrix (pre-treatment outcomes)
        if predictor_cols is None:
            # Use pre-treatment outcome averages
            X_1 = pre_data[pre_data[unit_col] == treated_unit][outcome_col].values
            X_0 = np.column_stack([
                pre_data[pre_data[unit_col] == d][outcome_col].values[:len(X_1)]
                for d in donors
            ])
        else:
            # Use specified predictors
            X_1 = pre_data[pre_data[unit_col] == treated_unit][predictor_cols].mean().values
            X_0 = np.column_stack([
                pre_data[pre_data[unit_col] == d][predictor_cols].mean().values
                for d in donors
            ])
        
        # Optimize weights
        n_donors = len(donors)
        
        def objective(w):
            synthetic = X_0 @ w
            return np.sum((X_1 - synthetic) ** 2)
        
        # Constraints: weights sum to 1, non-negative
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        bounds = [(0, 1) for _ in range(n_donors)]
        
        # Initial weights
        w0 = np.ones(n_donors) / n_donors
        
        from scipy.optimize import minimize
        result = minimize(objective, w0, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        self.weights_ = dict(zip(donors, result.x))
        
        # Compute synthetic outcome
        y_treated = df[df[unit_col] == treated_unit].set_index(time_col)[outcome_col]
        
        y_synthetic = pd.Series(0.0, index=y_treated.index)
        for donor, weight in self.weights_.items():
            donor_outcome = df[df[unit_col] == donor].set_index(time_col)[outcome_col]
            y_synthetic += weight * donor_outcome.reindex(y_treated.index).fillna(0)
        
        # Treatment effect
        effect = y_treated - y_synthetic
        
        # Pre-treatment fit
        pre_effect = effect[effect.index < treatment_time]
        post_effect = effect[effect.index >= treatment_time]
        
        rmspe_pre = np.sqrt(np.mean(pre_effect ** 2))
        att_post = post_effect.mean()
        
        self.results_ = {
            'weights': self.weights_,
            'y_treated': y_treated,
            'y_synthetic': y_synthetic,
            'effect': effect,
            'rmspe_pre': rmspe_pre,
            'att_post': att_post,
            'pre_treatment_fit': 1 - rmspe_pre / y_treated[y_treated.index < treatment_time].std(),
            'treatment_time': treatment_time,
            'treated_unit': treated_unit
        }
        
        return self
    
    def placebo_test(self, df: pd.DataFrame,
                     outcome_col: str,
                     unit_col: str,
                     time_col: str,
                     treatment_time: Any) -> Dict:
        """
        Run placebo tests by applying synthetic control to each donor.
        
        Returns p-value based on rank of treated unit's effect.
        """
        all_units = df[unit_col].unique()
        treated_unit = self.results_['treated_unit']
        
        placebo_effects = {}
        
        for unit in all_units:
            if unit == treated_unit:
                placebo_effects[unit] = self.results_['att_post']
            else:
                try:
                    sc = SyntheticControl()
                    sc.fit(df, outcome_col, unit_col, time_col, unit, treatment_time)
                    placebo_effects[unit] = sc.results_['att_post']
                except:
                    continue
        
        # Rank of treatment effect
        effects = list(placebo_effects.values())
        treated_effect = placebo_effects[treated_unit]
        rank = sum(1 for e in effects if abs(e) >= abs(treated_effect))
        p_value = rank / len(effects)
        
        return {
            'placebo_effects': placebo_effects,
            'treated_effect': treated_effect,
            'rank': rank,
            'p_value': p_value,
            'significant': p_value <= 0.1
        }
    
    def summary(self) -> Dict:
        """Return synthetic control results."""
        return self.results_


# =============================================================================
# 14. DEVALUATION HYPOTHESIS TEST
# =============================================================================

class DevaluationTest:
    """
    Test the Devaluation Hypothesis.
    
    Tests whether wages in female-dominated occupations are lower
    controlling for skill requirements and other characteristics.
    
    The hypothesis: W_j = α + β(% Female)_j + γX_j + ε_j
    
    where β < 0 supports devaluation.
    
    References:
    -----------
    England, P. (1992). Comparable Worth: Theories and Evidence.
    Levanon, A., England, P., & Allison, P. (2009). Occupational Feminization.
    """
    
    def __init__(self):
        self.results_ = None
    
    def fit(self, df: pd.DataFrame,
            wage_col: str,
            female_share_col: str,
            control_cols: List[str] = None,
            occupation_col: str = None,
            weight_col: str = None) -> 'DevaluationTest':
        """
        Test devaluation hypothesis.
        
        Parameters
        ----------
        df : DataFrame
            Data at occupation or individual level
        wage_col : str
            Average wage or log wage
        female_share_col : str
            Proportion female in occupation (0-1)
        control_cols : list, optional
            Control variables (education requirements, skill level, etc.)
        occupation_col : str, optional
            If provided, aggregates to occupation level first
        weight_col : str, optional
            Weights (e.g., employment count)
        """
        df = df.copy()
        
        # Aggregate to occupation level if individual data
        if occupation_col:
            agg_dict = {wage_col: 'mean', female_share_col: 'mean'}
            if weight_col:
                agg_dict[weight_col] = 'sum'
            if control_cols:
                for col in control_cols:
                    agg_dict[col] = 'mean'
            
            df = df.groupby(occupation_col).agg(agg_dict).reset_index()
        
        df = df.dropna(subset=[wage_col, female_share_col])
        
        # Build regression
        if control_cols:
            X = sm.add_constant(df[[female_share_col] + control_cols])
        else:
            X = sm.add_constant(df[[female_share_col]])
        
        y = df[wage_col]
        
        if weight_col and weight_col in df.columns:
            model = sm.WLS(y, X, weights=df[weight_col]).fit(cov_type='HC1')
        else:
            model = sm.OLS(y, X).fit(cov_type='HC1')
        
        # Extract devaluation coefficient
        beta = model.params[female_share_col]
        se = model.bse[female_share_col]
        t_stat = model.tvalues[female_share_col]
        p_value = model.pvalues[female_share_col]
        
        # Interpretation
        # If wages in log: a 10pp increase in female share reduces wages by beta*10%
        # If wages in levels: direct interpretation
        
        self.results_ = {
            'devaluation_coefficient': beta,
            'standard_error': se,
            't_statistic': t_stat,
            'p_value': p_value,
            'ci_lower': beta - 1.96 * se,
            'ci_upper': beta + 1.96 * se,
            'devaluation_supported': beta < 0 and p_value < 0.05,
            'interpretation': f"10pp increase in female share associated with {beta*10:.2f}% wage change",
            'r_squared': model.rsquared,
            'n_occupations': len(df),
            'model': model
        }
        
        return self
    
    def summary(self) -> Dict:
        """Return devaluation test results."""
        return {k: v for k, v in self.results_.items() if k != 'model'}


# =============================================================================
# MASTER ANALYSIS FUNCTION
# =============================================================================

def run_complete_wage_gap_analysis(df: pd.DataFrame,
                                   wage_col: str,
                                   gender_col: str,
                                   control_cols: List[str],
                                   occupation_col: str = None,
                                   weight_col: str = None,
                                   female_code: int = 2) -> Dict:
    """
    Run comprehensive wage gap analysis using all available methods.
    
    Parameters
    ----------
    df : DataFrame
        Individual-level data with wages and characteristics
    wage_col : str
        Wage variable (ideally log hourly wage)
    gender_col : str
        Gender indicator
    control_cols : list
        Human capital and demographic controls
    occupation_col : str, optional
        Occupation variable for segregation analysis
    weight_col : str, optional
        Survey weights
    female_code : int
        Code for female in gender column
    
    Returns
    -------
    results : dict
        Comprehensive analysis results
    """
    results = {'methods_run': [], 'errors': []}
    
    logger.info("Starting comprehensive wage gap analysis...")
    
    # 1. Basic decomposition
    logger.info("Running full decomposition...")
    try:
        decomp = run_full_decomposition(df, wage_col, gender_col, control_cols, 
                                        female_code, weight_col)
        results['decomposition'] = decomp
        results['methods_run'].append('Oaxaca-Blinder + RIF + DFL + PSM + DR')
    except Exception as e:
        results['errors'].append(f'Decomposition: {e}')
    
    # 2. Glass ceiling
    logger.info("Computing glass ceiling index...")
    try:
        gc = compute_glass_ceiling_index(df, wage_col, gender_col, female_code)
        results['glass_ceiling'] = gc
        results['methods_run'].append('Glass Ceiling Index')
    except Exception as e:
        results['errors'].append(f'Glass ceiling: {e}')
    
    # 3. Machado-Mata quantile decomposition
    logger.info("Running Machado-Mata decomposition...")
    try:
        mm = MachadoMata(n_quantiles=19)  # Vigintiles for speed
        mm.fit(df, wage_col, gender_col, control_cols, female_code)
        results['machado_mata'] = mm.summary()
        results['methods_run'].append('Machado-Mata Quantile Decomposition')
    except Exception as e:
        results['errors'].append(f'Machado-Mata: {e}')
    
    # 4. Segregation analysis
    if occupation_col:
        logger.info("Computing segregation indices...")
        try:
            results['segregation'] = {
                'duncan_index': SegregationIndices.duncan_index(
                    df, occupation_col, gender_col, female_code, weight_col),
                'karmel_maclachlan': SegregationIndices.karmel_maclachlan_index(
                    df, occupation_col, gender_col, female_code, weight_col),
                'ip_index': SegregationIndices.ip_index(
                    df, occupation_col, gender_col, female_code, weight_col)
            }
            results['methods_run'].append('Segregation Indices')
            
            # Brown-Moon-Zoloth
            logger.info("Running Brown-Moon-Zoloth decomposition...")
            bmz = BrownMoonZoloth()
            bmz.fit(df, wage_col, gender_col, occupation_col, control_cols, 
                    female_code, weight_col)
            results['brown_moon_zoloth'] = bmz.summary()
            results['methods_run'].append('Brown-Moon-Zoloth Decomposition')
            
        except Exception as e:
            results['errors'].append(f'Segregation/BMZ: {e}')
    
    # 5. Summary statistics
    female_mask = df[gender_col] == female_code
    results['summary'] = {
        'n_total': len(df),
        'n_male': (~female_mask).sum(),
        'n_female': female_mask.sum(),
        'male_mean_wage': df.loc[~female_mask, wage_col].mean(),
        'female_mean_wage': df.loc[female_mask, wage_col].mean(),
        'raw_gap': df.loc[~female_mask, wage_col].mean() - df.loc[female_mask, wage_col].mean(),
        'methods_completed': len(results['methods_run']),
        'errors_encountered': len(results['errors'])
    }
    
    logger.info(f"Analysis complete. {len(results['methods_run'])} methods run, "
                f"{len(results['errors'])} errors.")
    
    return results