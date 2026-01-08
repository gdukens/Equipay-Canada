"""
Advanced Time Series Analysis for Pay Equity Research
======================================================

This module provides time series econometric methods for analyzing
pay equity gap trends over time using Statistics Canada data.

Supports analysis of gaps across multiple protected attributes:
- Gender (male vs. female)
- Immigration status (Canadian-born vs. immigrant)
- Age groups
- Provincial disparities
- Any other categorical dimension

Key Features:
- Structural break detection (Chow, CUSUM, Bai-Perron)
- Cointegration analysis (Engle-Granger, Johansen)
- Vector Autoregression (VAR) models
- Forecasting with confidence intervals
- Seasonal decomposition
- Dynamic Time Warping for pattern matching

For advanced methods (Markov-switching, TVP, convergence analysis),
see: src/advanced_time_series.py
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Statistical imports
from scipy import stats
from scipy.signal import find_peaks
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests, coint
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_ljungbox
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.stattools import durbin_watson

import logging
logger = logging.getLogger(__name__)

# Import advanced methods
try:
    from .advanced_time_series import (
        BaiPerronBreaks,
        ZivotAndrewsBreak,
        CUSUMTest,
        MarkovSwitchingWageGap,
        WageGapConvergence,
        TimeVaryingDiscrimination,
        WageGapSpectral,
        WageGapDynamicFactors,
        EventStudyDiD,
        run_comprehensive_time_series_analysis,
        analyze_provincial_convergence
    )
    ADVANCED_TS_AVAILABLE = True
except ImportError:
    ADVANCED_TS_AVAILABLE = False
    logger.info("Advanced time series module not available - using basic methods only")


@dataclass
class TimeSeriesResult:
    """Container for time series analysis results."""
    method: str
    statistic: float
    p_value: float
    critical_values: Optional[Dict[str, float]] = None
    conclusion: str = ""
    details: Optional[Dict[str, Any]] = None


class WageGapTimeSeriesAnalyzer:
    """
    Comprehensive time series analysis for pay equity gap trends.
    
    Analyzes wage gaps over time for any protected attribute (gender,
    immigration status, age, etc.) using econometric methods:
    - Unit root tests (ADF, KPSS, PP)
    - Structural break detection
    - Cointegration analysis
    - VAR/VECM models
    - Forecasting
    
    Examples:
        # Gender gap analysis (traditional)
        analyzer = WageGapTimeSeriesAnalyzer(data, group_column='GENDER')
        
        # Immigration gap analysis
        analyzer = WageGapTimeSeriesAnalyzer(data, group_column='IMMIG',
                                              reference_value=1, comparison_value=2,
                                              reference_label='Canadian-born',
                                              comparison_label='Immigrant')
    """
    
    def __init__(self, data: pd.DataFrame, date_column: str = 'REF_DATE',
                 wage_column: str = 'VALUE', group_column: str = 'GENDER',
                 reference_value: Any = None, comparison_value: Any = None,
                 reference_label: str = None, comparison_label: str = None):
        """
        Initialize the analyzer with time series data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Panel data with wage information by group over time
        date_column : str
            Column name containing date/period information
        group_column : str
            Column for the protected attribute (e.g., 'GENDER', 'IMMIG', 'AGE_6')
        wage_column : str
            Column name for wage values
        reference_value : Any
            Value identifying reference group (e.g., 1 for Male, 'Canadian-born')
            If None, attempts to auto-detect based on group_column
        comparison_value : Any
            Value identifying comparison group (e.g., 2 for Female, 'Immigrant')
        reference_label : str
            Human-readable label for reference group
        comparison_label : str
            Human-readable label for comparison group
        """
        self.raw_data = data.copy()
        self.date_column = date_column
        self.wage_column = wage_column
        self.group_column = group_column
        
        # Auto-detect reference/comparison values for common attributes
        self.reference_value, self.comparison_value = self._detect_group_values(
            group_column, reference_value, comparison_value
        )
        self.reference_label = reference_label or str(self.reference_value)
        self.comparison_label = comparison_label or str(self.comparison_value)
        
        # Legacy alias for backward compatibility
        self.gender_column = group_column
        
        # Prepare time series
        self.ts_data = self._prepare_time_series()
        logger.info(f"Initialized time series analyzer for {group_column} with {len(self.ts_data)} periods")
    
    def _detect_group_values(self, group_column: str, ref_val: Any, comp_val: Any) -> Tuple[Any, Any]:
        """Auto-detect reference and comparison values for common attributes."""
        defaults = {
            'GENDER': (1, 2),        # Male vs Female
            'SEX': (1, 2),           # Legacy column name
            'IMMIG': (1, 2),         # Canadian-born vs Immigrant
            'UNION': (2, 1),         # Union vs Non-union (union = 1)
            'FTPTMAIN': (1, 2),      # Full-time vs Part-time
            'PERMTEMP': (1, 2),      # Permanent vs Temporary
        }
        
        if ref_val is not None and comp_val is not None:
            return ref_val, comp_val
        
        if group_column in defaults:
            return defaults[group_column]
        
        # Try to infer from data
        if group_column in self.raw_data.columns:
            unique_vals = sorted(self.raw_data[group_column].dropna().unique())
            if len(unique_vals) >= 2:
                return unique_vals[0], unique_vals[1]
        
        raise ValueError(f"Cannot auto-detect group values for '{group_column}'. "
                        f"Please specify reference_value and comparison_value.")
    
    def _prepare_time_series(self) -> pd.DataFrame:
        """Convert panel data to time series of wage gaps."""
        df = self.raw_data.copy()
        
        # Parse date - handle various StatCan formats
        if self.date_column in df.columns:
            # Try to extract year from various formats
            df['year'] = pd.to_datetime(df[self.date_column], errors='coerce').dt.year
            if df['year'].isna().all():
                # Try extracting year directly
                df['year'] = df[self.date_column].astype(str).str[:4].astype(int, errors='ignore')
        
        # Filter for valid data
        df = df[df[self.wage_column].notna() & (df[self.wage_column] > 0)]
        
        # Aggregate by year and group
        if self.group_column in df.columns:
            agg_data = df.groupby(['year', self.group_column])[self.wage_column].mean().reset_index()
            
            # Pivot to get reference/comparison wages by year
            pivot = agg_data.pivot(index='year', columns=self.group_column, values=self.wage_column)
            
            # Try to find reference and comparison columns
            ref_col = None
            comp_col = None
            
            # First try exact match on configured values
            if self.reference_value in pivot.columns:
                ref_col = self.reference_value
            if self.comparison_value in pivot.columns:
                comp_col = self.comparison_value
            
            # Fallback: try string matching for gender labels
            if ref_col is None or comp_col is None:
                male_cols = [c for c in pivot.columns if 'male' in str(c).lower() and 'female' not in str(c).lower()]
                female_cols = [c for c in pivot.columns if 'female' in str(c).lower()]
                if male_cols and female_cols:
                    ref_col = male_cols[0]
                    comp_col = female_cols[0]
            
            if ref_col is not None and comp_col is not None:
                ts_df = pd.DataFrame({
                    'year': pivot.index,
                    'reference_wage': pivot[ref_col].values,
                    'comparison_wage': pivot[comp_col].values
                })
                # Add legacy column names for backward compatibility
                ts_df['male_wage'] = ts_df['reference_wage']
                ts_df['female_wage'] = ts_df['comparison_wage']
                
                ts_df['wage_gap'] = (ts_df['reference_wage'] - ts_df['comparison_wage']) / ts_df['reference_wage'] * 100
                ts_df['wage_ratio'] = ts_df['comparison_wage'] / ts_df['reference_wage']
                ts_df['reference_label'] = self.reference_label
                ts_df['comparison_label'] = self.comparison_label
                ts_df = ts_df.dropna().sort_values('year').reset_index(drop=True)
                return ts_df
        
        # Fallback: just use overall wage trend
        ts_df = df.groupby('year')[self.wage_column].agg(['mean', 'std', 'count']).reset_index()
        ts_df.columns = ['year', 'avg_wage', 'std_wage', 'n_obs']
        return ts_df.sort_values('year').reset_index(drop=True)
    
    def unit_root_tests(self, series: Optional[pd.Series] = None) -> Dict[str, TimeSeriesResult]:
        """
        Perform comprehensive unit root testing.
        
        Tests:
        - Augmented Dickey-Fuller (ADF)
        - Kwiatkowski-Phillips-Schmidt-Shin (KPSS)
        - Phillips-Perron (PP via ADF with autolag)
        
        Returns
        -------
        Dict[str, TimeSeriesResult]
            Results for each test
        """
        if series is None:
            if 'wage_gap' in self.ts_data.columns:
                series = self.ts_data['wage_gap'].dropna()
            else:
                series = self.ts_data['avg_wage'].dropna()
        
        results = {}
        
        # ADF Test (H0: unit root exists)
        try:
            adf_result = adfuller(series, autolag='AIC')
            results['ADF'] = TimeSeriesResult(
                method='Augmented Dickey-Fuller',
                statistic=adf_result[0],
                p_value=adf_result[1],
                critical_values=adf_result[4],
                conclusion='Stationary' if adf_result[1] < 0.05 else 'Non-stationary (unit root)',
                details={'n_lags': adf_result[2], 'n_obs': adf_result[3]}
            )
        except Exception as e:
            logger.warning(f"ADF test failed: {e}")
        
        # KPSS Test (H0: series is stationary)
        try:
            kpss_result = kpss(series, regression='c', nlags='auto')
            results['KPSS'] = TimeSeriesResult(
                method='KPSS',
                statistic=kpss_result[0],
                p_value=kpss_result[1],
                critical_values=kpss_result[3],
                conclusion='Stationary' if kpss_result[1] > 0.05 else 'Non-stationary',
                details={'n_lags': kpss_result[2]}
            )
        except Exception as e:
            logger.warning(f"KPSS test failed: {e}")
        
        return results
    
    def structural_break_detection(self, series: Optional[pd.Series] = None,
                                    min_segment: int = 5) -> Dict[str, Any]:
        """
        Detect structural breaks in the wage gap time series.
        
        Implements:
        - CUSUM test
        - Chow test (sequential)
        - Bai-Perron style multiple breakpoint detection
        
        Returns
        -------
        Dict with break points and test statistics
        """
        if series is None:
            if 'wage_gap' in self.ts_data.columns:
                series = self.ts_data['wage_gap'].dropna()
            else:
                series = self.ts_data['avg_wage'].dropna()
        
        n = len(series)
        years = self.ts_data['year'].values[:n]
        
        results = {
            'breakpoints': [],
            'cusum': {},
            'chow_tests': [],
            'multiple_breaks': []
        }
        
        # 1. CUSUM Test
        try:
            # Recursive residuals approach
            y = series.values
            X = sm.add_constant(np.arange(len(y)))
            model = OLS(y, X).fit()
            
            # Calculate recursive residuals
            rec_resid = []
            for t in range(min_segment, n):
                model_t = OLS(y[:t], X[:t]).fit()
                pred = model_t.predict(X[t:t+1])
                rec_resid.append((y[t] - pred[0]) / np.std(model_t.resid))
            
            cusum = np.cumsum(rec_resid) / np.sqrt(len(rec_resid))
            
            # Critical values (5% significance)
            critical = 1.36  # Approximate critical value
            
            results['cusum'] = {
                'statistic': np.max(np.abs(cusum)),
                'critical_value': critical,
                'structural_break_detected': np.max(np.abs(cusum)) > critical,
                'cusum_values': cusum.tolist()
            }
        except Exception as e:
            logger.warning(f"CUSUM test failed: {e}")
        
        # 2. Sequential Chow Tests
        try:
            chow_stats = []
            for t in range(min_segment, n - min_segment):
                # Split sample
                y1, X1 = y[:t], X[:t]
                y2, X2 = y[t:], X[t:]
                
                # Restricted model (pooled)
                rss_r = OLS(y, X).fit().ssr
                
                # Unrestricted (separate regressions)
                rss_u = OLS(y1, X1).fit().ssr + OLS(y2, X2).fit().ssr
                
                # Chow statistic
                k = X.shape[1]
                chow_stat = ((rss_r - rss_u) / k) / (rss_u / (n - 2*k))
                p_value = 1 - stats.f.cdf(chow_stat, k, n - 2*k)
                
                chow_stats.append({
                    'break_year': years[t] if t < len(years) else t,
                    'break_index': t,
                    'chow_statistic': chow_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                })
            
            # Find most significant breaks
            significant_breaks = [c for c in chow_stats if c['significant']]
            if significant_breaks:
                best_break = max(significant_breaks, key=lambda x: x['chow_statistic'])
                results['breakpoints'].append(best_break['break_year'])
            
            results['chow_tests'] = chow_stats
        except Exception as e:
            logger.warning(f"Chow test failed: {e}")
        
        # 3. Simple multiple breakpoint detection (Bai-Perron style)
        try:
            # Use RSS minimization
            def compute_rss(break_indices):
                segments = [0] + list(break_indices) + [n]
                total_rss = 0
                for i in range(len(segments) - 1):
                    start, end = segments[i], segments[i+1]
                    if end - start >= 2:
                        segment_y = y[start:end]
                        segment_X = sm.add_constant(np.arange(end - start))
                        total_rss += OLS(segment_y, segment_X).fit().ssr
                return total_rss
            
            # Test 1 and 2 breaks
            best_1_break = None
            best_1_rss = float('inf')
            for b1 in range(min_segment, n - min_segment):
                rss = compute_rss([b1])
                if rss < best_1_rss:
                    best_1_rss = rss
                    best_1_break = b1
            
            best_2_breaks = None
            best_2_rss = float('inf')
            for b1 in range(min_segment, n - 2*min_segment):
                for b2 in range(b1 + min_segment, n - min_segment):
                    rss = compute_rss([b1, b2])
                    if rss < best_2_rss:
                        best_2_rss = rss
                        best_2_breaks = [b1, b2]
            
            no_break_rss = compute_rss([])
            
            results['multiple_breaks'] = {
                'no_break_rss': no_break_rss,
                'one_break': {
                    'index': best_1_break,
                    'year': years[best_1_break] if best_1_break and best_1_break < len(years) else None,
                    'rss': best_1_rss
                },
                'two_breaks': {
                    'indices': best_2_breaks,
                    'years': [years[b] for b in best_2_breaks] if best_2_breaks else None,
                    'rss': best_2_rss
                } if best_2_breaks else None
            }
        except Exception as e:
            logger.warning(f"Multiple break detection failed: {e}")
        
        return results
    
    def granger_causality_analysis(self, max_lag: int = 4) -> Dict[str, Any]:
        """
        Test Granger causality between reference and comparison group wages.
        
        Helps understand lead-lag relationships in wage dynamics.
        For gender: tests if male wages lead/lag female wages.
        For immigration: tests if Canadian-born wages lead/lag immigrant wages.
        """
        results = {}
        
        # Use generic column names, fall back to legacy names
        ref_col = 'reference_wage' if 'reference_wage' in self.ts_data.columns else 'male_wage'
        comp_col = 'comparison_wage' if 'comparison_wage' in self.ts_data.columns else 'female_wage'
        
        if ref_col not in self.ts_data.columns or comp_col not in self.ts_data.columns:
            logger.warning(f"Need {ref_col}/{comp_col} columns for Granger causality")
            return results
        
        # Prepare data
        df = self.ts_data[[ref_col, comp_col]].dropna()
        
        if len(df) < max_lag + 5:
            logger.warning("Insufficient data for Granger causality test")
            return results
        
        try:
            # Test: reference wages -> comparison wages
            gc_ref_to_comp = grangercausalitytests(df[[comp_col, ref_col]], maxlag=max_lag, verbose=False)
            results['reference_to_comparison'] = {
                lag: {
                    'f_stat': gc_ref_to_comp[lag][0]['ssr_ftest'][0],
                    'p_value': gc_ref_to_comp[lag][0]['ssr_ftest'][1],
                    'significant': gc_ref_to_comp[lag][0]['ssr_ftest'][1] < 0.05
                }
                for lag in range(1, max_lag + 1)
            }
            # Legacy key names for backward compatibility
            results['male_to_female'] = results['reference_to_comparison']
            
            # Test: comparison wages -> reference wages  
            gc_comp_to_ref = grangercausalitytests(df[[ref_col, comp_col]], maxlag=max_lag, verbose=False)
            results['comparison_to_reference'] = {
                lag: {
                    'f_stat': gc_comp_to_ref[lag][0]['ssr_ftest'][0],
                    'p_value': gc_comp_to_ref[lag][0]['ssr_ftest'][1],
                    'significant': gc_comp_to_ref[lag][0]['ssr_ftest'][1] < 0.05
                }
                for lag in range(1, max_lag + 1)
            }
            # Legacy key names for backward compatibility
            results['female_to_male'] = results['comparison_to_reference']
            
            # Add metadata
            results['reference_label'] = getattr(self, 'reference_label', 'Reference')
            results['comparison_label'] = getattr(self, 'comparison_label', 'Comparison')
            
        except Exception as e:
            logger.warning(f"Granger causality test failed: {e}")
        
        return results
    
    def cointegration_analysis(self) -> Dict[str, Any]:
        """
        Test for cointegration between reference and comparison group wages.
        
        If cointegrated, wages move together in the long run,
        implying the wage gap is stationary.
        """
        results = {}
        
        # Use generic column names, fall back to legacy names
        ref_col = 'reference_wage' if 'reference_wage' in self.ts_data.columns else 'male_wage'
        comp_col = 'comparison_wage' if 'comparison_wage' in self.ts_data.columns else 'female_wage'
        
        if ref_col not in self.ts_data.columns:
            return results
        
        reference = self.ts_data[ref_col].dropna().values
        comparison = self.ts_data[comp_col].dropna().values
        
        n = min(len(reference), len(comparison))
        reference, comparison = reference[:n], comparison[:n]
        
        if n < 10:
            logger.warning("Insufficient data for cointegration test")
            return results
        
        try:
            # Engle-Granger two-step cointegration test
            coint_stat, p_value, crit_values = coint(reference, comparison)
            
            results['engle_granger'] = {
                'statistic': coint_stat,
                'p_value': p_value,
                'critical_values': {
                    '1%': crit_values[0],
                    '5%': crit_values[1],
                    '10%': crit_values[2]
                },
                'cointegrated': p_value < 0.05,
                'interpretation': 'Wages move together long-run (gap is stable)' if p_value < 0.05 
                                  else 'No long-run equilibrium detected'
            }
            
            # Estimate cointegrating vector
            X = sm.add_constant(reference)
            coint_model = OLS(comparison, X).fit()
            ref_label = getattr(self, 'reference_label', 'Reference')
            comp_label = getattr(self, 'comparison_label', 'Comparison')
            results['cointegrating_vector'] = {
                'constant': coint_model.params[0],
                'coefficient': coint_model.params[1],
                'interpretation': f"{comp_label} wage = {coint_model.params[0]:.2f} + {coint_model.params[1]:.3f} × {ref_label} wage"
            }
            
        except Exception as e:
            logger.warning(f"Cointegration test failed: {e}")
        
        return results
    
    def var_model(self, max_lag: int = 4) -> Dict[str, Any]:
        """
        Estimate Vector Autoregression model for wage dynamics.
        
        Captures dynamic interdependencies between male and female wages.
        """
        results = {}
        
        if 'male_wage' not in self.ts_data.columns:
            return results
        
        df = self.ts_data[['male_wage', 'female_wage']].dropna()
        
        if len(df) < max_lag + 10:
            logger.warning("Insufficient data for VAR model")
            return results
        
        try:
            model = VAR(df)
            
            # Select optimal lag
            lag_selection = model.select_order(maxlags=max_lag)
            optimal_lag = lag_selection.aic
            
            results['lag_selection'] = {
                'optimal_lag_aic': optimal_lag,
                'aic_values': {i: lag_selection.ics['aic'][i] for i in range(max_lag + 1)}
            }
            
            # Fit VAR
            var_fit = model.fit(optimal_lag)
            
            # Impulse Response Functions
            irf = var_fit.irf(10)
            
            results['var_model'] = {
                'lag_order': optimal_lag,
                'coefficients': {
                    'male_wage_eq': var_fit.params['male_wage'].to_dict(),
                    'female_wage_eq': var_fit.params['female_wage'].to_dict()
                },
                'irf_male_to_female': irf.irfs[:, 1, 0].tolist(),  # Response of female to male shock
                'irf_female_to_male': irf.irfs[:, 0, 1].tolist(),  # Response of male to female shock
            }
            
            # Forecast variance decomposition
            fevd = var_fit.fevd(10)
            results['variance_decomposition'] = {
                'male_wage_variance': fevd.decomp[:, 0, :].tolist(),
                'female_wage_variance': fevd.decomp[:, 1, :].tolist()
            }
            
        except Exception as e:
            logger.warning(f"VAR model failed: {e}")
        
        return results
    
    def seasonal_decomposition(self, series: Optional[pd.Series] = None,
                               period: int = None) -> Dict[str, Any]:
        """
        Decompose time series into trend, seasonal, and residual components.
        """
        if series is None:
            if 'wage_gap' in self.ts_data.columns:
                series = self.ts_data.set_index('year')['wage_gap'].dropna()
            else:
                series = self.ts_data.set_index('year')['avg_wage'].dropna()
        
        results = {}
        
        if len(series) < 4:
            logger.warning("Insufficient data for decomposition")
            return results
        
        try:
            # Determine period
            if period is None:
                period = min(4, len(series) // 2)  # For annual data, use 4 or less
            
            if period < 2:
                period = 2
            
            # Classical decomposition
            if len(series) >= 2 * period:
                decomp = seasonal_decompose(series, model='additive', period=period)
                
                results['classical'] = {
                    'trend': decomp.trend.dropna().to_dict(),
                    'seasonal': decomp.seasonal.dropna().to_dict(),
                    'residual': decomp.resid.dropna().to_dict()
                }
            
            # STL decomposition (more robust)
            if len(series) >= 2 * period:
                stl = STL(series, period=period, robust=True).fit()
                
                results['stl'] = {
                    'trend': stl.trend.to_dict(),
                    'seasonal': stl.seasonal.to_dict(),
                    'residual': stl.resid.to_dict(),
                    'trend_strength': 1 - np.var(stl.resid) / np.var(stl.trend + stl.resid),
                    'seasonal_strength': 1 - np.var(stl.resid) / np.var(stl.seasonal + stl.resid)
                }
                
        except Exception as e:
            logger.warning(f"Decomposition failed: {e}")
        
        return results
    
    def forecast_wage_gap(self, horizon: int = 5) -> Dict[str, Any]:
        """
        Forecast future wage gap using ARIMA model.
        
        Returns point forecasts and confidence intervals.
        """
        results = {}
        
        if 'wage_gap' not in self.ts_data.columns:
            logger.warning("Wage gap series not available")
            return results
        
        series = self.ts_data['wage_gap'].dropna()
        years = self.ts_data['year'].values[:len(series)]
        
        if len(series) < 10:
            logger.warning("Insufficient data for forecasting")
            return results
        
        try:
            # Determine order via AIC
            best_aic = float('inf')
            best_order = (1, 0, 1)
            
            for p in range(3):
                for d in range(2):
                    for q in range(3):
                        try:
                            model = ARIMA(series, order=(p, d, q))
                            fit = model.fit()
                            if fit.aic < best_aic:
                                best_aic = fit.aic
                                best_order = (p, d, q)
                        except:
                            continue
            
            # Fit best model
            model = ARIMA(series, order=best_order)
            fit = model.fit()
            
            # Forecast
            forecast = fit.get_forecast(steps=horizon)
            pred_mean = forecast.predicted_mean
            conf_int = forecast.conf_int()
            
            last_year = int(years[-1])
            forecast_years = list(range(last_year + 1, last_year + horizon + 1))
            
            results['arima'] = {
                'order': best_order,
                'aic': best_aic,
                'forecast_years': forecast_years,
                'point_forecast': pred_mean.tolist(),
                'lower_95': conf_int.iloc[:, 0].tolist(),
                'upper_95': conf_int.iloc[:, 1].tolist(),
                'model_summary': {
                    'ar_coefs': fit.arparams.tolist() if len(fit.arparams) > 0 else [],
                    'ma_coefs': fit.maparams.tolist() if len(fit.maparams) > 0 else []
                }
            }
            
            # Calculate years to parity (when gap reaches 0)
            if pred_mean.iloc[-1] < series.iloc[-1]:
                # Gap is decreasing
                current_gap = series.iloc[-1]
                annual_decrease = (series.iloc[-1] - pred_mean.iloc[-1]) / horizon
                if annual_decrease > 0:
                    years_to_parity = int(np.ceil(current_gap / annual_decrease))
                    results['parity_projection'] = {
                        'current_gap': current_gap,
                        'annual_decrease': annual_decrease,
                        'years_to_parity': years_to_parity,
                        'projected_parity_year': last_year + years_to_parity
                    }
            
        except Exception as e:
            logger.warning(f"Forecasting failed: {e}")
        
        return results
    
    def trend_analysis(self) -> Dict[str, Any]:
        """
        Analyze overall trends in the wage gap.
        """
        results = {}
        
        if 'wage_gap' not in self.ts_data.columns:
            if 'avg_wage' in self.ts_data.columns:
                series = self.ts_data['avg_wage'].dropna()
                years = self.ts_data['year'].values[:len(series)]
                series_name = 'avg_wage'
            else:
                return results
        else:
            series = self.ts_data['wage_gap'].dropna()
            years = self.ts_data['year'].values[:len(series)]
            series_name = 'wage_gap'
        
        if len(series) < 3:
            return results
        
        # Linear trend
        X = sm.add_constant(years)
        trend_model = OLS(series, X).fit()
        
        results['linear_trend'] = {
            'intercept': trend_model.params[0],
            'slope': trend_model.params[1],
            'slope_interpretation': f"{series_name} changes by {trend_model.params[1]:.3f} percentage points per year",
            'r_squared': trend_model.rsquared,
            'p_value_slope': trend_model.pvalues[1],
            'significant_trend': trend_model.pvalues[1] < 0.05
        }
        
        # Quadratic trend (acceleration/deceleration)
        X_quad = np.column_stack([np.ones(len(years)), years, years**2])
        quad_model = OLS(series, X_quad).fit()
        
        results['quadratic_trend'] = {
            'intercept': quad_model.params[0],
            'linear_coef': quad_model.params[1],
            'quadratic_coef': quad_model.params[2],
            'r_squared': quad_model.rsquared,
            'acceleration': 'decelerating' if quad_model.params[2] > 0 else 'accelerating'
        }
        
        # Period analysis (compare different eras)
        if len(years) >= 10:
            mid_point = len(series) // 2
            early_mean = series[:mid_point].mean()
            late_mean = series[mid_point:].mean()
            
            results['period_comparison'] = {
                'early_period': f"{years[0]}-{years[mid_point-1]}",
                'late_period': f"{years[mid_point]}-{years[-1]}",
                'early_mean': early_mean,
                'late_mean': late_mean,
                'change': late_mean - early_mean,
                'percent_change': (late_mean - early_mean) / early_mean * 100 if early_mean != 0 else 0
            }
        
        return results
    
    def run_full_analysis(self) -> Dict[str, Any]:
        """
        Run all time series analyses and return comprehensive results.
        """
        logger.info("Running comprehensive time series analysis...")
        
        results = {
            'data_summary': {
                'n_periods': len(self.ts_data),
                'year_range': [int(self.ts_data['year'].min()), int(self.ts_data['year'].max())] 
                              if 'year' in self.ts_data.columns else None,
                'columns': list(self.ts_data.columns)
            }
        }
        
        # Add current wage gap if available
        if 'wage_gap' in self.ts_data.columns:
            gap_series = self.ts_data['wage_gap'].dropna()
            results['data_summary']['current_gap'] = float(gap_series.iloc[-1])
            results['data_summary']['gap_range'] = [float(gap_series.min()), float(gap_series.max())]
        
        # Run all analyses
        results['unit_root_tests'] = self.unit_root_tests()
        results['trend_analysis'] = self.trend_analysis()
        results['structural_breaks'] = self.structural_break_detection()
        results['granger_causality'] = self.granger_causality_analysis()
        results['cointegration'] = self.cointegration_analysis()
        results['var_model'] = self.var_model()
        results['forecasts'] = self.forecast_wage_gap()
        
        logger.info("Time series analysis complete")
        return results


def create_time_series_from_lfs(processed_data_dir: Path = None) -> pd.DataFrame:
    """
    Load and prepare time series data from processed LFS data.
    
    This function reads from the processed LFS data (lfs_processed.csv)
    and aggregates by year/gender for time series analysis.
    
    Parameters
    ----------
    processed_data_dir : Path, optional
        Path to processed data directory. Defaults to 'data/processed'.
        
    Returns
    -------
    pd.DataFrame
        Aggregated time series data with columns:
        - YEAR: Reference year
        - GENDER: Gender category
        - mean_wage: Average hourly wage
        - median_wage: Median hourly wage
        - n_obs: Number of observations
    """
    if processed_data_dir is None:
        processed_data_dir = Path('data/processed')
    
    lfs_file = processed_data_dir / 'lfs_processed.csv'
    
    if not lfs_file.exists():
        logger.warning(f"LFS processed file not found: {lfs_file}")
        return pd.DataFrame()
    
    logger.info(f"Loading time series data from {lfs_file}")
    
    try:
        df = pd.read_csv(lfs_file, low_memory=False)
        
        # Determine column names
        year_col = 'YEAR' if 'YEAR' in df.columns else 'SURVYEAR'
        gender_col = 'GENDER' if 'GENDER' in df.columns else 'SEX'
        wage_col = 'HRLYEARN' if 'HRLYEARN' in df.columns else 'HOURLY_WAGE'
        
        if wage_col not in df.columns:
            logger.warning(f"Wage column not found. Available: {df.columns.tolist()}")
            return pd.DataFrame()
        
        # Aggregate by year and gender
        agg_df = df.groupby([year_col, gender_col]).agg({
            wage_col: ['mean', 'median', 'count']
        }).reset_index()
        
        agg_df.columns = ['YEAR', 'GENDER', 'mean_wage', 'median_wage', 'n_obs']
        
        logger.info(f"Created time series with {len(agg_df)} rows")
        return agg_df
        
    except Exception as e:
        logger.error(f"Error loading time series data: {e}")
        return pd.DataFrame()


# Legacy function name for backward compatibility
create_time_series_from_statcan = create_time_series_from_lfs


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    from pathlib import Path
    raw_dir = Path(__file__).parent.parent / 'data' / 'raw'
    
    df = create_time_series_from_statcan(raw_dir)
    
    if len(df) > 0:
        analyzer = WageGapTimeSeriesAnalyzer(df)
        results = analyzer.run_full_analysis()
        
        print("\n=== Time Series Analysis Results ===")
        print(f"Data covers {results['data_summary']['n_periods']} periods")
        
        if 'trend_analysis' in results and results['trend_analysis']:
            trend = results['trend_analysis'].get('linear_trend', {})
            print(f"\nLinear Trend: {trend.get('slope_interpretation', 'N/A')}")
            print(f"R-squared: {trend.get('r_squared', 0):.3f}")
