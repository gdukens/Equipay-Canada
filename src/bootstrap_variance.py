"""
Poisson Bootstrap Variance Estimation for LFS PUMF Data.

Implements the Statistics Canada methodology for variance estimation using
the Poisson bootstrap method as described in the LFS PUMF User Guide (January 2025).

Reference:
- Beaumont, J.-F., & Patak, Z. (2012). On the generalized bootstrap for sample 
  surveys with special attention to Poisson sampling. International Statistical 
  Review, 80(1), 127-148.

Quality Guidelines (per StatsCan):
- CV < 15%: Acceptable quality
- CV 15-35%: Marginal quality (requires warning)
- CV > 35%: Unacceptable (should not be published)
- Minimum 5 respondents required for any estimate
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Callable, List, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class QualityIndicators:
    """Quality indicators for survey estimates per StatsCan guidelines."""
    estimate: float
    variance: float
    std_error: float
    cv: float  # Coefficient of variation (%)
    ci_lower: float  # 95% confidence interval
    ci_upper: float
    sample_size: int
    quality: str  # 'acceptable', 'marginal', 'unacceptable'
    warning: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for DataFrame construction."""
        return {
            'estimate': self.estimate,
            'variance': self.variance,
            'std_error': self.std_error,
            'cv_percent': self.cv,
            'ci_lower_95': self.ci_lower,
            'ci_upper_95': self.ci_upper,
            'sample_size': self.sample_size,
            'quality': self.quality,
            'warning': self.warning
        }


def assess_quality(cv: float, sample_size: int) -> Tuple[str, Optional[str]]:
    """
    Assess estimate quality per StatsCan guidelines.
    
    Parameters:
        cv: Coefficient of variation (as percentage)
        sample_size: Number of respondents contributing to estimate
        
    Returns:
        Tuple of (quality_level, warning_message)
    """
    if sample_size < 5:
        return 'unacceptable', f"Sample size ({sample_size}) below minimum of 5"
    
    if cv > 35:
        return 'unacceptable', f"CV ({cv:.1f}%) exceeds 35% threshold"
    elif cv > 15:
        return 'marginal', f"CV ({cv:.1f}%) is between 15-35%; use with caution"
    else:
        return 'acceptable', None


class PoissonBootstrap:
    """
    Poisson Bootstrap variance estimator for LFS PUMF data.
    
    Implements the methodology from the Statistics Canada LFS PUMF User Guide
    (January 2025), Section 6.0 "Poids bootstrap de Poisson pour l'estimation 
    de la variance".
    
    The Poisson bootstrap is a practical approach for PUMF users since the full
    survey design information is not available. It provides approximate variance
    estimates that account for the complex survey design.
    
    Example:
        >>> bootstrap = PoissonBootstrap(df, n_replicates=1000, seed=42)
        >>> result = bootstrap.estimate_total('HRLYEARN', domain='PROV')
        >>> print(result['cv_percent'])
    """
    
    # Calibration domain definitions per StatsCan Appendix B
    AGE_GROUPS = {
        '15-16': lambda df: df['AGE_6'] == 1,
        '17-19': lambda df: df['AGE_6'] == 2,
        '20-24': lambda df: df['AGE_12'].isin([2, 3]),
        '25-29': lambda df: df['AGE_12'] == 3,
        '30-34': lambda df: df['AGE_12'] == 4,
        '35-44': lambda df: df['AGE_12'].isin([5, 6]),
        '45-54': lambda df: df['AGE_12'].isin([7, 8]),
        '55-59': lambda df: df['AGE_12'] == 9,
        '60-64': lambda df: df['AGE_12'] == 10,
        '65-69': lambda df: df['AGE_12'] == 11,
        '70+': lambda df: df['AGE_12'] == 12,
    }
    
    def __init__(
        self, 
        df: pd.DataFrame,
        weight_col: str = 'FINALWT',
        n_replicates: int = 1000,
        seed: Optional[int] = None,
        calibrate: bool = True
    ):
        """
        Initialize Poisson Bootstrap estimator.
        
        Parameters:
            df: DataFrame with LFS PUMF data
            weight_col: Survey weight column name (default: 'FINALWT')
            n_replicates: Number of bootstrap replicates (default: 1000)
            seed: Random seed for reproducibility
            calibrate: Whether to calibrate bootstrap weights to control totals
        """
        self.df = df.copy()
        self.weight_col = weight_col
        self.n_replicates = n_replicates
        self.calibrate = calibrate
        
        if seed is not None:
            np.random.seed(seed)
            
        if weight_col not in df.columns:
            raise ValueError(f"Weight column '{weight_col}' not found in DataFrame")
            
        self._generate_bootstrap_weights()
        
    def _generate_bootstrap_weights(self):
        """
        Generate Poisson bootstrap weights per StatsCan methodology.
        
        Formula (per Guide Section 6.1):
        adjustment_factor = 1 + poisson_factor * sqrt((finalwt - 1) / finalwt)
        bootstrap_weight = finalwt * adjustment_factor
        
        where poisson_factor = +1 or -1 with 50% probability
        """
        n = len(self.df)
        finalwt = self.df[self.weight_col].values
        
        # Generate Poisson factors: +1 or -1 with 50% probability
        poisson_factors = 2 * (np.random.random((n, self.n_replicates)) >= 0.5) - 1
        
        # Compute adjustment factors per equation (1)
        # adjustment = 1 + poisson_factor * sqrt((w - 1) / w)
        weight_factor = np.sqrt((finalwt - 1) / finalwt)[:, np.newaxis]
        adjustment_factors = 1 + poisson_factors * weight_factor
        
        # Compute uncalibrated bootstrap weights per equation (2)
        self.uncal_bsw = finalwt[:, np.newaxis] * adjustment_factors
        
        if self.calibrate:
            self._calibrate_weights()
        else:
            self.bootstrap_weights = self.uncal_bsw
            
        logger.info(f"Generated {self.n_replicates} bootstrap weight replicates")
        
    def _calibrate_weights(self):
        """
        Calibrate bootstrap weights to control totals.
        
        Per StatsCan Appendix B, calibration domains are:
        Province × Age Group × Gender (220 domains)
        
        Formula (3):
        calibrated_bsw = (sum_finalwt_by_domain / sum_bsw_by_domain) * bsw
        """
        # Create calibration domains
        self._create_age_groups()
        
        # Create domain key
        self.df['_cal_domain'] = (
            self.df['PROV'].astype(str) + '_' + 
            self.df['_age_cal'].astype(str) + '_' + 
            self.df['GENDER'].astype(str)
        )
        
        # Get domain totals of FINALWT
        domain_finalwt = self.df.groupby('_cal_domain')[self.weight_col].sum()
        
        # Calibrate each replicate
        self.bootstrap_weights = np.zeros_like(self.uncal_bsw)
        
        for domain in self.df['_cal_domain'].unique():
            mask = self.df['_cal_domain'] == domain
            domain_total = domain_finalwt[domain]
            
            # Sum of bootstrap weights in this domain for each replicate
            domain_bsw_sums = self.uncal_bsw[mask].sum(axis=0)
            
            # Calibration ratio
            cal_ratio = domain_total / domain_bsw_sums
            
            # Apply calibration
            self.bootstrap_weights[mask] = self.uncal_bsw[mask] * cal_ratio
            
        logger.info(f"Calibrated bootstrap weights to {len(domain_finalwt)} domains")
        
    def _create_age_groups(self):
        """Create age calibration groups per StatsCan specifications."""
        # Handle nullable integers by filling NA with -1
        age_6 = self.df['AGE_6'].fillna(-1).astype(int).values
        age_12 = self.df['AGE_12'].fillna(-1).astype(int).values
        
        conditions = [
            age_6 == 1,  # 15-16
            age_6 == 2,  # 17-19
            (age_12 >= 2) & (age_12 <= 3),  # 20-24
            age_12 == 3,  # 25-29 
            age_12 == 4,  # 30-34
            (age_12 >= 5) & (age_12 <= 6),  # 35-44
            (age_12 >= 7) & (age_12 <= 8),  # 45-54
            age_12 == 9,   # 55-59
            age_12 == 10,  # 60-64
            age_12 == 11,  # 65-69
            age_12 == 12,  # 70+
        ]
        choices = ['15-16', '17-19', '20-24', '25-29', '30-34', 
                   '35-44', '45-54', '55-59', '60-64', '65-69', '70+']
        
        self.df['_age_cal'] = np.select(conditions, choices, default='unknown')
        
    def estimate_total(
        self, 
        variable: str,
        domain: Optional[str] = None,
        condition: Optional[pd.Series] = None
    ) -> Union[QualityIndicators, Dict[str, QualityIndicators]]:
        """
        Estimate population total with variance.
        
        Parameters:
            variable: Column to estimate total for
            domain: Optional grouping variable for domain estimates
            condition: Optional boolean mask for subsetting
            
        Returns:
            QualityIndicators or dict of QualityIndicators by domain
        """
        df = self.df
        weights = self.df[self.weight_col].values
        bsw = self.bootstrap_weights
        
        if condition is not None:
            mask = condition.values if isinstance(condition, pd.Series) else condition
            df = df[mask]
            weights = weights[mask]
            bsw = bsw[mask]
            
        if domain is None:
            return self._compute_total_variance(df, variable, weights, bsw)
        else:
            results = {}
            for level in df[domain].unique():
                domain_mask = df[domain] == level
                results[level] = self._compute_total_variance(
                    df[domain_mask], 
                    variable, 
                    weights[domain_mask],
                    bsw[domain_mask]
                )
            return results
            
    def _compute_total_variance(
        self, 
        df: pd.DataFrame, 
        variable: str,
        weights: np.ndarray,
        bsw: np.ndarray
    ) -> QualityIndicators:
        """Compute total estimate with bootstrap variance."""
        values = df[variable].values
        sample_size = len(df)
        
        # Point estimate using survey weights
        estimate = np.sum(values * weights)
        
        # Bootstrap estimates
        bs_estimates = np.sum(values[:, np.newaxis] * bsw, axis=0)
        
        # Variance per formula in Section 6.2
        variance = np.mean((bs_estimates - estimate) ** 2)
        std_error = np.sqrt(variance)
        
        # CV as percentage
        cv = abs(std_error / estimate) * 100 if estimate != 0 else np.inf
        
        # 95% CI using t=2.0
        ci_lower = estimate - 2.0 * std_error
        ci_upper = estimate + 2.0 * std_error
        
        # Quality assessment
        quality, warning = assess_quality(cv, sample_size)
        
        return QualityIndicators(
            estimate=estimate,
            variance=variance,
            std_error=std_error,
            cv=cv,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            sample_size=sample_size,
            quality=quality,
            warning=warning
        )
        
    def estimate_mean(
        self,
        variable: str,
        domain: Optional[str] = None,
        condition: Optional[pd.Series] = None
    ) -> Union[QualityIndicators, Dict[str, QualityIndicators]]:
        """
        Estimate population mean with variance.
        
        Parameters:
            variable: Column to estimate mean for
            domain: Optional grouping variable
            condition: Optional boolean mask
            
        Returns:
            QualityIndicators or dict by domain
        """
        df = self.df
        weights = self.df[self.weight_col].values
        bsw = self.bootstrap_weights
        
        if condition is not None:
            mask = condition.values if isinstance(condition, pd.Series) else condition
            df = df[mask]
            weights = weights[mask]
            bsw = bsw[mask]
            
        if domain is None:
            return self._compute_mean_variance(df, variable, weights, bsw)
        else:
            results = {}
            for level in df[domain].unique():
                domain_mask = df[domain] == level
                results[level] = self._compute_mean_variance(
                    df[domain_mask],
                    variable,
                    weights[domain_mask],
                    bsw[domain_mask]
                )
            return results
            
    def _compute_mean_variance(
        self,
        df: pd.DataFrame,
        variable: str,
        weights: np.ndarray,
        bsw: np.ndarray
    ) -> QualityIndicators:
        """Compute mean estimate with bootstrap variance."""
        values = df[variable].values
        sample_size = len(df)
        
        # Handle missing values
        valid_mask = ~np.isnan(values)
        values = values[valid_mask]
        weights = weights[valid_mask]
        bsw = bsw[valid_mask]
        
        # Weighted mean using survey weights
        estimate = np.sum(values * weights) / np.sum(weights)
        
        # Bootstrap means
        bs_means = (np.sum(values[:, np.newaxis] * bsw, axis=0) / 
                   np.sum(bsw, axis=0))
        
        # Variance
        variance = np.mean((bs_means - estimate) ** 2)
        std_error = np.sqrt(variance)
        
        cv = abs(std_error / estimate) * 100 if estimate != 0 else np.inf
        
        ci_lower = estimate - 2.0 * std_error
        ci_upper = estimate + 2.0 * std_error
        
        quality, warning = assess_quality(cv, sample_size)
        
        return QualityIndicators(
            estimate=estimate,
            variance=variance,
            std_error=std_error,
            cv=cv,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            sample_size=sample_size,
            quality=quality,
            warning=warning
        )
        
    def estimate_ratio(
        self,
        numerator_condition: pd.Series,
        denominator_condition: Optional[pd.Series] = None,
        domain: Optional[str] = None
    ) -> Union[QualityIndicators, Dict[str, QualityIndicators]]:
        """
        Estimate ratio/proportion with variance (e.g., unemployment rate).
        
        Parameters:
            numerator_condition: Boolean mask for numerator (e.g., unemployed)
            denominator_condition: Boolean mask for denominator (e.g., labour force)
                                 If None, uses full population
            domain: Optional grouping variable
            
        Returns:
            QualityIndicators or dict by domain
        """
        if denominator_condition is None:
            denominator_condition = pd.Series(True, index=self.df.index)
            
        df = self.df[denominator_condition]
        weights = self.df.loc[denominator_condition, self.weight_col].values
        bsw = self.bootstrap_weights[denominator_condition.values]
        num_mask = numerator_condition[denominator_condition].values
        
        if domain is None:
            return self._compute_ratio_variance(num_mask, weights, bsw)
        else:
            results = {}
            for level in df[domain].unique():
                domain_mask = df[domain] == level
                results[level] = self._compute_ratio_variance(
                    num_mask[domain_mask],
                    weights[domain_mask],
                    bsw[domain_mask]
                )
            return results
            
    def _compute_ratio_variance(
        self,
        num_mask: np.ndarray,
        weights: np.ndarray,
        bsw: np.ndarray
    ) -> QualityIndicators:
        """Compute ratio estimate with bootstrap variance."""
        sample_size = np.sum(num_mask)
        
        # Point estimate
        num_wt = np.sum(weights * num_mask)
        den_wt = np.sum(weights)
        estimate = num_wt / den_wt if den_wt > 0 else 0
        
        # Bootstrap ratios
        bs_num = np.sum(bsw * num_mask[:, np.newaxis], axis=0)
        bs_den = np.sum(bsw, axis=0)
        bs_ratios = bs_num / bs_den
        
        variance = np.mean((bs_ratios - estimate) ** 2)
        std_error = np.sqrt(variance)
        
        cv = abs(std_error / estimate) * 100 if estimate != 0 else np.inf
        
        ci_lower = max(0, estimate - 2.0 * std_error)
        ci_upper = min(1, estimate + 2.0 * std_error)
        
        quality, warning = assess_quality(cv, sample_size)
        
        return QualityIndicators(
            estimate=estimate,
            variance=variance,
            std_error=std_error,
            cv=cv,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            sample_size=sample_size,
            quality=quality,
            warning=warning
        )
        
    def estimate_wage_gap(
        self,
        wage_col: str = 'HRLYEARN',
        domain: Optional[str] = None,
        condition: Optional[pd.Series] = None
    ) -> Union[Dict, Dict[str, Dict]]:
        """
        Estimate gender wage gap with variance.
        
        Gap = (male_wage - female_wage) / male_wage
        
        Parameters:
            wage_col: Wage column name
            domain: Optional grouping variable
            condition: Optional boolean mask
            
        Returns:
            Dict with male_wage, female_wage, gap, and quality indicators
        """
        df = self.df
        weights = self.df[self.weight_col].values
        bsw = self.bootstrap_weights
        
        if condition is not None:
            mask = condition.values
            df = df[mask]
            weights = weights[mask]
            bsw = bsw[mask]
            
        # Valid wage filter
        valid = df[wage_col] > 0
        
        if domain is None:
            return self._compute_gap_variance(
                df[valid], wage_col, weights[valid], bsw[valid]
            )
        else:
            results = {}
            for level in df[domain].unique():
                domain_mask = (df[domain] == level) & valid
                if domain_mask.sum() > 0:
                    results[level] = self._compute_gap_variance(
                        df[domain_mask],
                        wage_col,
                        weights[domain_mask],
                        bsw[domain_mask]
                    )
            return results
            
    def _compute_gap_variance(
        self,
        df: pd.DataFrame,
        wage_col: str,
        weights: np.ndarray,
        bsw: np.ndarray
    ) -> Dict:
        """Compute wage gap with bootstrap variance."""
        wages = df[wage_col].values
        male = (df['GENDER'] == 1).values
        female = (df['GENDER'] == 2).values
        
        sample_male = male.sum()
        sample_female = female.sum()
        
        # Point estimates
        male_wt = np.sum(weights * male)
        female_wt = np.sum(weights * female)
        
        male_wage = np.sum(wages * weights * male) / male_wt if male_wt > 0 else 0
        female_wage = np.sum(wages * weights * female) / female_wt if female_wt > 0 else 0
        
        gap = (male_wage - female_wage) / male_wage if male_wage > 0 else 0
        
        # Bootstrap gaps
        bs_male_wages = (np.sum(wages[:, np.newaxis] * bsw * male[:, np.newaxis], axis=0) /
                        np.sum(bsw * male[:, np.newaxis], axis=0))
        bs_female_wages = (np.sum(wages[:, np.newaxis] * bsw * female[:, np.newaxis], axis=0) /
                          np.sum(bsw * female[:, np.newaxis], axis=0))
        bs_gaps = (bs_male_wages - bs_female_wages) / bs_male_wages
        
        gap_variance = np.mean((bs_gaps - gap) ** 2)
        gap_std = np.sqrt(gap_variance)
        gap_cv = abs(gap_std / gap) * 100 if gap != 0 else np.inf
        
        quality, warning = assess_quality(gap_cv, min(sample_male, sample_female))
        
        return {
            'male_wage': male_wage,
            'female_wage': female_wage,
            'gap': gap,
            'gap_percent': gap * 100,
            'gap_variance': gap_variance,
            'gap_std_error': gap_std,
            'gap_cv': gap_cv,
            'gap_ci_lower': gap - 2.0 * gap_std,
            'gap_ci_upper': gap + 2.0 * gap_std,
            'sample_male': sample_male,
            'sample_female': sample_female,
            'quality': quality,
            'warning': warning
        }


def combine_monthly_weights(
    monthly_weights: List[np.ndarray],
    method: str = 'annual'
) -> np.ndarray:
    """
    Combine monthly survey weights for multi-month estimates.
    
    Per StatsCan Guide Section 5.1:
    - For 3-month moving average: FINALWT / 3
    - For annual estimate: FINALWT / 12
    
    Parameters:
        monthly_weights: List of monthly FINALWT arrays
        method: 'quarterly' (3-month) or 'annual' (12-month)
        
    Returns:
        Adjusted combined weights
    """
    combined = np.concatenate(monthly_weights)
    
    if method == 'quarterly':
        divisor = 3
    elif method == 'annual':
        divisor = 12
    else:
        divisor = len(monthly_weights)
        
    return combined / divisor


# Convenience function for quick wage gap analysis
def analyze_wage_gap_with_variance(
    df: pd.DataFrame,
    wage_col: str = 'HRLYEARN',
    domain: Optional[str] = None,
    n_replicates: int = 100,  # Use 1000 for production
    seed: int = 42
) -> pd.DataFrame:
    """
    Quick wage gap analysis with bootstrap variance.
    
    Parameters:
        df: DataFrame with LFS data
        wage_col: Wage column
        domain: Optional grouping variable
        n_replicates: Bootstrap replicates (1000 recommended for production)
        seed: Random seed
        
    Returns:
        DataFrame with gap estimates and quality indicators
    """
    bs = PoissonBootstrap(df, n_replicates=n_replicates, seed=seed)
    
    valid_wage = df[wage_col] > 0
    results = bs.estimate_wage_gap(wage_col, domain=domain, condition=valid_wage)
    
    if domain is None:
        return pd.DataFrame([results])
    else:
        return pd.DataFrame(results).T.reset_index().rename(columns={'index': domain})
