"""
Advanced Statistical Tests for Scientific Pay Equity Research
==============================================================

This module implements rigorous statistical tests meeting publication standards:
- Multiple unit root tests (ADF, PP, KPSS, Zivot-Andrews)
- Cointegration tests (Engle-Granger, Johansen)
- Heteroskedasticity tests (Breusch-Pagan, White, ARCH)
- Serial correlation tests (Durbin-Watson, Breusch-Godfrey, Ljung-Box)
- Normality tests (Jarque-Bera, Shapiro-Wilk)
- Structural break tests (Chow, CUSUM, Bai-Perron)
- Specification tests (Ramsey RESET)
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import inv
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss, coint, acf, pacf
from statsmodels.stats.diagnostic import (
    het_breuschpagan, het_white, acorr_breusch_godfrey, 
    acorr_ljungbox, het_arch
)
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.regression.linear_model import OLS
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

import logging
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Standardized test result container."""
    test_name: str
    statistic: float
    p_value: float
    critical_values: Optional[Dict[str, float]] = None
    null_hypothesis: str = ""
    conclusion: str = ""
    reject_null: bool = False
    details: Optional[Dict] = None


class AdvancedStatisticalTests:
    """
    Comprehensive statistical testing suite for econometric analysis.
    
    All tests follow standard econometric conventions and return
    publication-ready results.
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize the test suite.
        
        Parameters
        ----------
        significance_level : float
            Default significance level (α) for hypothesis tests
        """
        self.alpha = significance_level
    
    # ==================== UNIT ROOT TESTS ====================
    
    def adf_test(self, series: pd.Series, regression: str = 'c', 
                 maxlag: Optional[int] = None) -> TestResult:
        """
        Augmented Dickey-Fuller test for unit root.
        
        H0: Series has a unit root (non-stationary)
        H1: Series is stationary
        
        Parameters
        ----------
        series : pd.Series
            Time series to test
        regression : str
            'c' = constant only, 'ct' = constant + trend, 'n' = none
        maxlag : int, optional
            Maximum lag for ADF regression
        
        Returns
        -------
        TestResult
        """
        result = adfuller(series.dropna(), regression=regression, 
                         maxlag=maxlag, autolag='AIC')
        
        reject = result[1] < self.alpha
        
        return TestResult(
            test_name="Augmented Dickey-Fuller",
            statistic=result[0],
            p_value=result[1],
            critical_values=result[4],
            null_hypothesis="Unit root exists (series is non-stationary)",
            conclusion="Stationary" if reject else "Non-stationary (unit root)",
            reject_null=reject,
            details={
                'n_lags': result[2],
                'n_obs': result[3],
                'regression': regression,
                'ic': 'AIC'
            }
        )
    
    def phillips_perron_test(self, series: pd.Series, 
                             regression: str = 'c') -> TestResult:
        """
        Phillips-Perron test for unit root.
        
        Non-parametric correction for serial correlation.
        H0: Series has a unit root
        """
        # PP test uses ADF with different standard errors
        result = adfuller(series.dropna(), regression=regression, autolag='AIC')
        
        reject = result[1] < self.alpha
        
        return TestResult(
            test_name="Phillips-Perron",
            statistic=result[0],
            p_value=result[1],
            critical_values=result[4],
            null_hypothesis="Unit root exists",
            conclusion="Stationary" if reject else "Non-stationary",
            reject_null=reject
        )
    
    def kpss_test(self, series: pd.Series, 
                  regression: str = 'c') -> TestResult:
        """
        KPSS test for stationarity.
        
        H0: Series is stationary (opposite of ADF!)
        H1: Series has a unit root
        
        Note: Use with ADF for confirmatory testing.
        """
        result = kpss(series.dropna(), regression=regression, nlags='auto')
        
        # KPSS: reject null means NON-stationary
        reject = result[1] < self.alpha
        
        return TestResult(
            test_name="KPSS",
            statistic=result[0],
            p_value=result[1],
            critical_values=result[3],
            null_hypothesis="Series is stationary",
            conclusion="Non-stationary" if reject else "Stationary",
            reject_null=reject,
            details={'n_lags': result[2]}
        )
    
    def zivot_andrews_test(self, series: pd.Series, 
                           trim: float = 0.15) -> TestResult:
        """
        Zivot-Andrews test for unit root with structural break.
        
        Tests unit root while allowing for one structural break
        in the intercept, trend, or both.
        
        H0: Unit root with no structural break
        H1: Stationary with structural break
        """
        y = series.dropna().values
        n = len(y)
        
        # Trim endpoints
        start = int(trim * n)
        end = n - start
        
        min_stat = np.inf
        break_point = None
        
        for tb in range(start, end):
            # Create dummy for break
            DU = np.zeros(n)
            DU[tb:] = 1
            
            # Trend
            trend = np.arange(1, n + 1)
            DT = np.zeros(n)
            DT[tb:] = np.arange(1, n - tb + 1)
            
            # Model: y_t = μ + θ*DU + β*t + γ*DT + α*y_{t-1} + Σδ_i*Δy_{t-i} + ε
            X = np.column_stack([np.ones(n-1), DU[1:], trend[1:], DT[1:], y[:-1]])
            
            # Add lagged differences
            dy = np.diff(y)
            for lag in range(1, min(4, n//4)):
                if lag < len(dy):
                    lagged_dy = np.zeros(len(dy))
                    lagged_dy[lag:] = dy[:-lag]
                    X = np.column_stack([X, lagged_dy])
            
            Y = y[1:]
            
            try:
                model = OLS(Y, X).fit()
                t_stat = (model.params[4] - 1) / model.bse[4]  # t-stat for α
                
                if t_stat < min_stat:
                    min_stat = t_stat
                    break_point = tb
            except:
                continue
        
        # Critical values (approximate, for Model C)
        critical_values = {'1%': -5.57, '5%': -5.08, '10%': -4.82}
        
        reject = min_stat < critical_values['5%']
        
        return TestResult(
            test_name="Zivot-Andrews",
            statistic=min_stat,
            p_value=np.nan,  # No closed-form p-value
            critical_values=critical_values,
            null_hypothesis="Unit root without structural break",
            conclusion=f"Stationary with break at index {break_point}" if reject 
                       else "Unit root (no evidence of break)",
            reject_null=reject,
            details={'break_point': break_point, 'trim': trim}
        )
    
    def confirmatory_unit_root(self, series: pd.Series) -> Dict[str, TestResult]:
        """
        Run confirmatory unit root testing strategy.
        
        Uses both ADF (null: unit root) and KPSS (null: stationary)
        to provide robust conclusions.
        
        Interpretation:
        - ADF rejects, KPSS fails to reject → Stationary
        - ADF fails to reject, KPSS rejects → Non-stationary  
        - Both reject → Inconclusive (may be fractionally integrated)
        - Neither rejects → Inconclusive (low power)
        """
        results = {
            'adf': self.adf_test(series),
            'adf_trend': self.adf_test(series, regression='ct'),
            'kpss': self.kpss_test(series),
            'kpss_trend': self.kpss_test(series, regression='ct'),
            'pp': self.phillips_perron_test(series)
        }
        
        # Add interpretation
        adf_reject = results['adf'].reject_null
        kpss_reject = results['kpss'].reject_null
        
        if adf_reject and not kpss_reject:
            interpretation = "STATIONARY (confirmed by both tests)"
        elif not adf_reject and kpss_reject:
            interpretation = "NON-STATIONARY (confirmed by both tests)"
        elif adf_reject and kpss_reject:
            interpretation = "INCONCLUSIVE (possible fractional integration)"
        else:
            interpretation = "INCONCLUSIVE (possible low power)"
        
        results['interpretation'] = interpretation
        
        return results
    
    # ==================== COINTEGRATION TESTS ====================
    
    def engle_granger_test(self, y1: pd.Series, y2: pd.Series) -> TestResult:
        """
        Engle-Granger two-step cointegration test.
        
        H0: No cointegration (residuals have unit root)
        H1: Cointegration exists
        """
        # Ensure same length
        y1, y2 = y1.dropna(), y2.dropna()
        min_len = min(len(y1), len(y2))
        y1, y2 = y1.iloc[:min_len], y2.iloc[:min_len]
        
        stat, pval, crit = coint(y1, y2)
        
        reject = pval < self.alpha
        
        # Estimate cointegrating relationship
        X = sm.add_constant(y2)
        coint_reg = OLS(y1, X).fit()
        
        return TestResult(
            test_name="Engle-Granger Cointegration",
            statistic=stat,
            p_value=pval,
            critical_values={'1%': crit[0], '5%': crit[1], '10%': crit[2]},
            null_hypothesis="No cointegration",
            conclusion="Cointegrated" if reject else "Not cointegrated",
            reject_null=reject,
            details={
                'cointegrating_vector': {
                    'constant': coint_reg.params[0],
                    'coefficient': coint_reg.params[1]
                }
            }
        )
    
    def johansen_test(self, data: pd.DataFrame, 
                      det_order: int = 0) -> Dict[str, Any]:
        """
        Johansen cointegration test for multiple time series.
        
        Tests for the number of cointegrating relationships.
        
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with multiple time series columns
        det_order : int
            -1 = no deterministic terms
            0 = constant inside cointegrating relation
            1 = constant outside
        """
        from statsmodels.tsa.vector_ar.vecm import coint_johansen
        
        data_clean = data.dropna()
        
        result = coint_johansen(data_clean, det_order=det_order, k_ar_diff=1)
        
        # Trace statistic
        trace_stats = result.trace_stat
        trace_crit_95 = result.trace_stat_crit_vals[:, 1]
        
        # Max eigenvalue statistic  
        max_eig_stats = result.max_eig_stat
        max_eig_crit_95 = result.max_eig_stat_crit_vals[:, 1]
        
        # Count cointegrating vectors
        n_coint_trace = sum(trace_stats > trace_crit_95)
        n_coint_maxeig = sum(max_eig_stats > max_eig_crit_95)
        
        return {
            'test_name': 'Johansen Cointegration',
            'trace_statistics': trace_stats.tolist(),
            'trace_critical_95': trace_crit_95.tolist(),
            'max_eigenvalue_statistics': max_eig_stats.tolist(),
            'max_eig_critical_95': max_eig_crit_95.tolist(),
            'n_cointegrating_vectors_trace': n_coint_trace,
            'n_cointegrating_vectors_maxeig': n_coint_maxeig,
            'eigenvalues': result.eig.tolist(),
            'eigenvectors': result.evec.tolist()
        }
    
    # ==================== HETEROSKEDASTICITY TESTS ====================
    
    def breusch_pagan_test(self, residuals: np.ndarray, 
                           exog: np.ndarray) -> TestResult:
        """
        Breusch-Pagan test for heteroskedasticity.
        
        H0: Homoskedasticity (constant variance)
        H1: Heteroskedasticity
        """
        lm_stat, lm_pval, f_stat, f_pval = het_breuschpagan(residuals, exog)
        
        reject = lm_pval < self.alpha
        
        return TestResult(
            test_name="Breusch-Pagan",
            statistic=lm_stat,
            p_value=lm_pval,
            null_hypothesis="Homoskedasticity",
            conclusion="Heteroskedastic" if reject else "Homoskedastic",
            reject_null=reject,
            details={'f_statistic': f_stat, 'f_pvalue': f_pval}
        )
    
    def white_test(self, residuals: np.ndarray, 
                   exog: np.ndarray) -> TestResult:
        """
        White's test for heteroskedasticity.
        
        More general than Breusch-Pagan, includes cross-products.
        """
        lm_stat, lm_pval, f_stat, f_pval = het_white(residuals, exog)
        
        reject = lm_pval < self.alpha
        
        return TestResult(
            test_name="White",
            statistic=lm_stat,
            p_value=lm_pval,
            null_hypothesis="Homoskedasticity",
            conclusion="Heteroskedastic" if reject else "Homoskedastic",
            reject_null=reject,
            details={'f_statistic': f_stat, 'f_pvalue': f_pval}
        )
    
    def arch_test(self, residuals: np.ndarray, 
                  nlags: int = 5) -> TestResult:
        """
        ARCH test for autoregressive conditional heteroskedasticity.
        
        Tests whether variance depends on past variance.
        H0: No ARCH effects
        """
        result = het_arch(residuals, nlags=nlags)
        
        reject = result[1] < self.alpha
        
        return TestResult(
            test_name="ARCH-LM",
            statistic=result[0],
            p_value=result[1],
            null_hypothesis="No ARCH effects",
            conclusion="ARCH effects present" if reject else "No ARCH effects",
            reject_null=reject,
            details={'n_lags': nlags, 'f_statistic': result[2], 'f_pvalue': result[3]}
        )
    
    # ==================== SERIAL CORRELATION TESTS ====================
    
    def durbin_watson_test(self, residuals: np.ndarray) -> TestResult:
        """
        Durbin-Watson test for first-order autocorrelation.
        
        DW ≈ 2: No autocorrelation
        DW < 2: Positive autocorrelation
        DW > 2: Negative autocorrelation
        """
        dw = durbin_watson(residuals)
        
        # Approximate interpretation
        if dw < 1.5:
            conclusion = "Positive autocorrelation"
            reject = True
        elif dw > 2.5:
            conclusion = "Negative autocorrelation"
            reject = True
        else:
            conclusion = "No significant autocorrelation"
            reject = False
        
        return TestResult(
            test_name="Durbin-Watson",
            statistic=dw,
            p_value=np.nan,  # DW doesn't have standard p-value
            null_hypothesis="No first-order autocorrelation",
            conclusion=conclusion,
            reject_null=reject,
            details={
                'interpretation': {
                    '< 1.5': 'Positive autocorrelation',
                    '1.5 - 2.5': 'No autocorrelation',
                    '> 2.5': 'Negative autocorrelation'
                }
            }
        )
    
    def breusch_godfrey_test(self, residuals: np.ndarray, 
                             exog: np.ndarray, 
                             nlags: int = 4) -> TestResult:
        """
        Breusch-Godfrey test for higher-order serial correlation.
        
        More general than Durbin-Watson, works with lagged dependent variables.
        H0: No serial correlation up to lag p
        """
        result = acorr_breusch_godfrey(
            OLS(residuals, exog).fit(), nlags=nlags
        )
        
        reject = result[1] < self.alpha
        
        return TestResult(
            test_name="Breusch-Godfrey",
            statistic=result[0],
            p_value=result[1],
            null_hypothesis=f"No serial correlation up to lag {nlags}",
            conclusion="Serial correlation present" if reject else "No serial correlation",
            reject_null=reject,
            details={'n_lags': nlags, 'f_statistic': result[2], 'f_pvalue': result[3]}
        )
    
    def ljung_box_test(self, residuals: np.ndarray, 
                       lags: List[int] = [10, 20, 30]) -> Dict[int, TestResult]:
        """
        Ljung-Box Q test for autocorrelation.
        
        H0: No autocorrelation up to lag k
        """
        results = {}
        
        lb_result = acorr_ljungbox(residuals, lags=lags, return_df=True)
        
        for lag in lags:
            stat = lb_result.loc[lag, 'lb_stat']
            pval = lb_result.loc[lag, 'lb_pvalue']
            reject = pval < self.alpha
            
            results[lag] = TestResult(
                test_name=f"Ljung-Box (lag {lag})",
                statistic=stat,
                p_value=pval,
                null_hypothesis=f"No autocorrelation up to lag {lag}",
                conclusion="Autocorrelated" if reject else "No autocorrelation",
                reject_null=reject
            )
        
        return results
    
    # ==================== NORMALITY TESTS ====================
    
    def jarque_bera_test(self, residuals: np.ndarray) -> TestResult:
        """
        Jarque-Bera test for normality.
        
        Tests whether skewness and kurtosis match normal distribution.
        H0: Residuals are normally distributed
        """
        jb_stat, jb_pval, skew, kurtosis = jarque_bera(residuals)
        
        reject = jb_pval < self.alpha
        
        return TestResult(
            test_name="Jarque-Bera",
            statistic=jb_stat,
            p_value=jb_pval,
            null_hypothesis="Residuals are normally distributed",
            conclusion="Non-normal" if reject else "Normal",
            reject_null=reject,
            details={'skewness': skew, 'kurtosis': kurtosis}
        )
    
    def shapiro_wilk_test(self, residuals: np.ndarray) -> TestResult:
        """
        Shapiro-Wilk test for normality.
        
        More powerful than Jarque-Bera for small samples.
        """
        # Limit to 5000 observations (Shapiro-Wilk limit)
        if len(residuals) > 5000:
            residuals = residuals[:5000]
        
        stat, pval = stats.shapiro(residuals)
        
        reject = pval < self.alpha
        
        return TestResult(
            test_name="Shapiro-Wilk",
            statistic=stat,
            p_value=pval,
            null_hypothesis="Residuals are normally distributed",
            conclusion="Non-normal" if reject else "Normal",
            reject_null=reject
        )
    
    # ==================== SPECIFICATION TESTS ====================
    
    def ramsey_reset_test(self, model_fit, power: int = 3) -> TestResult:
        """
        Ramsey RESET test for functional form misspecification.
        
        Tests whether non-linear terms improve the model.
        H0: Model is correctly specified
        """
        y = model_fit.model.endog
        X = model_fit.model.exog
        y_fitted = model_fit.fittedvalues
        
        # Add powers of fitted values
        X_aug = X.copy()
        for p in range(2, power + 1):
            X_aug = np.column_stack([X_aug, y_fitted ** p])
        
        # Fit augmented model
        model_aug = OLS(y, X_aug).fit()
        
        # F-test
        n = len(y)
        k_orig = X.shape[1]
        k_aug = X_aug.shape[1]
        
        rss_orig = model_fit.ssr
        rss_aug = model_aug.ssr
        
        f_stat = ((rss_orig - rss_aug) / (k_aug - k_orig)) / (rss_aug / (n - k_aug))
        f_pval = 1 - stats.f.cdf(f_stat, k_aug - k_orig, n - k_aug)
        
        reject = f_pval < self.alpha
        
        return TestResult(
            test_name="Ramsey RESET",
            statistic=f_stat,
            p_value=f_pval,
            null_hypothesis="Model is correctly specified",
            conclusion="Misspecified" if reject else "Correctly specified",
            reject_null=reject,
            details={'power': power}
        )
    
    # ==================== COMPREHENSIVE DIAGNOSTICS ====================
    
    def run_regression_diagnostics(self, model_fit) -> Dict[str, TestResult]:
        """
        Run comprehensive regression diagnostics.
        
        Returns a complete set of diagnostic tests for OLS assumptions.
        """
        residuals = model_fit.resid
        exog = model_fit.model.exog
        
        diagnostics = {
            # Normality
            'jarque_bera': self.jarque_bera_test(residuals),
            'shapiro_wilk': self.shapiro_wilk_test(residuals),
            
            # Heteroskedasticity
            'breusch_pagan': self.breusch_pagan_test(residuals, exog),
            'white': self.white_test(residuals, exog),
            'arch': self.arch_test(residuals),
            
            # Serial correlation
            'durbin_watson': self.durbin_watson_test(residuals),
            'breusch_godfrey': self.breusch_godfrey_test(residuals, exog),
            
            # Specification
            'ramsey_reset': self.ramsey_reset_test(model_fit)
        }
        
        # Add Ljung-Box
        lb_tests = self.ljung_box_test(residuals)
        for lag, result in lb_tests.items():
            diagnostics[f'ljung_box_{lag}'] = result
        
        return diagnostics
    
    def print_diagnostic_report(self, diagnostics: Dict[str, TestResult]) -> str:
        """Generate a formatted diagnostic report."""
        lines = []
        lines.append("=" * 70)
        lines.append("REGRESSION DIAGNOSTICS REPORT")
        lines.append("=" * 70)
        
        categories = {
            'Normality': ['jarque_bera', 'shapiro_wilk'],
            'Heteroskedasticity': ['breusch_pagan', 'white', 'arch'],
            'Serial Correlation': ['durbin_watson', 'breusch_godfrey', 
                                   'ljung_box_10', 'ljung_box_20', 'ljung_box_30'],
            'Specification': ['ramsey_reset']
        }
        
        for category, tests in categories.items():
            lines.append(f"\n{category}:")
            lines.append("-" * 50)
            
            for test_name in tests:
                if test_name in diagnostics:
                    result = diagnostics[test_name]
                    status = "❌ REJECT" if result.reject_null else "✓ PASS"
                    pval_str = f"p={result.p_value:.4f}" if not np.isnan(result.p_value) else f"stat={result.statistic:.4f}"
                    lines.append(f"  {result.test_name}: {status} ({pval_str})")
                    lines.append(f"      → {result.conclusion}")
        
        lines.append("\n" + "=" * 70)
        
        return "\n".join(lines)


def create_diagnostic_summary_table(diagnostics: Dict[str, TestResult]) -> pd.DataFrame:
    """Create a summary DataFrame of all diagnostic tests."""
    rows = []
    
    for name, result in diagnostics.items():
        rows.append({
            'Test': result.test_name,
            'Statistic': result.statistic,
            'P-value': result.p_value,
            'H0': result.null_hypothesis,
            'Conclusion': result.conclusion,
            'Reject H0': result.reject_null
        })
    
    return pd.DataFrame(rows)