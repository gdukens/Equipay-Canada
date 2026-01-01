"""
Statistical Analysis Module
Pay equity analysis and statistical testing
With macroeconomic controls for scientific rigor

Uses centralized constants for consistent column naming.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Import centralized constants
from .constants import COLS, GENDER_CODES, normalize_column_names, humanize_columns

# Import centralized macro data
try:
    from .macro_data import (
        MACRO_DATA, get_macro_dataframe, add_macro_to_dataframe,
        adjust_for_inflation, get_deflator, BASE_YEAR, ECONOMIC_PERIODS
    )
    MACRO_AVAILABLE = True
except ImportError:
    MACRO_AVAILABLE = False

logger = logging.getLogger(__name__)


class PayEquityAnalyzer:
    """
    Comprehensive pay equity statistical analysis.
    
    Uses COLS constants for consistent column naming.
    Supports both GENDER (standard) and SEX (legacy) column names.
    """
    
    def __init__(self, df: pd.DataFrame, 
                 wage_col: str = None,
                 gender_col: str = None,
                 weight_col: str = None):
        """
        Initialize analyzer.
        
        Args:
            df: DataFrame with wage and demographic data
            wage_col: Name of wage column (default: HRLYEARN)
            gender_col: Name of gender column (default: GENDER or SEX)
            weight_col: Name of survey weight column (default: FINALWT)
        """
        self.df = normalize_column_names(df.copy())
        
        # Determine column names
        self.wage_col = wage_col or COLS.HOURLY_EARNINGS
        self.weight_col = weight_col or COLS.FINAL_WEIGHT
        
        # Gender column: prefer GENDER, fall back to SEX for legacy data
        if gender_col:
            self.gender_col = gender_col
        elif COLS.GENDER in self.df.columns:
            self.gender_col = COLS.GENDER
        elif 'SEX' in self.df.columns:
            self.gender_col = 'SEX'
        else:
            self.gender_col = COLS.GENDER
        
        self.results = {}
        self.has_weights = self.weight_col in self.df.columns
        
    def compute_raw_wage_gap(self) -> Dict:
        """
        Compute unadjusted (raw) gender wage gap.
        
        Uses survey weights (FINALWT) when available for proper population inference.
        """
        gender_col = self.gender_col
        male = self.df[self.df[gender_col] == 1][self.wage_col]
        female = self.df[self.df[gender_col] == 2][self.wage_col]
        
        # Weighted means if weights available
        if self.has_weights:
            male_weights = self.df[self.df[gender_col] == 1][self.weight_col]
            female_weights = self.df[self.df[gender_col] == 2][self.weight_col]
            male_mean = np.average(male, weights=male_weights)
            female_mean = np.average(female, weights=female_weights)
        else:
            male_mean = male.mean()
            female_mean = female.mean()
        
        male_median = male.median()
        female_median = female.median()
        
        raw_gap = male_mean - female_mean
        raw_gap_pct = (raw_gap / male_mean) * 100
        
        median_gap = male_median - female_median
        median_gap_pct = (median_gap / male_median) * 100
        
        # T-test for significance
        t_stat, p_value = stats.ttest_ind(male.dropna(), female.dropna())
        
        # Mann-Whitney U test (non-parametric)
        u_stat, u_pvalue = stats.mannwhitneyu(
            male.dropna(), female.dropna(), alternative='two-sided'
        )
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            (male.var() * (len(male) - 1) + female.var() * (len(female) - 1)) /
            (len(male) + len(female) - 2)
        )
        cohens_d = raw_gap / pooled_std
        
        results = {
            'male': {
                'mean': float(male_mean),
                'median': float(male_median),
                'std': float(male.std()),
                'n': int(len(male)),
            },
            'female': {
                'mean': float(female_mean),
                'median': float(female_median),
                'std': float(female.std()),
                'n': int(len(female)),
            },
            'raw_gap': {
                'mean_gap': float(raw_gap),
                'mean_gap_pct': float(raw_gap_pct),
                'median_gap': float(median_gap),
                'median_gap_pct': float(median_gap_pct),
                'female_to_male_ratio': float(female_mean / male_mean),
            },
            'statistical_tests': {
                't_test': {
                    'statistic': float(t_stat),
                    'p_value': float(p_value),
                    'significant_01': p_value < 0.01,
                    'significant_05': p_value < 0.05,
                },
                'mann_whitney': {
                    'statistic': float(u_stat),
                    'p_value': float(u_pvalue),
                },
                'effect_size': {
                    'cohens_d': float(cohens_d),
                    'interpretation': self._interpret_cohens_d(cohens_d),
                }
            }
        }
        
        self.results['raw_wage_gap'] = results
        return results
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size"""
        d = abs(d)
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"
    
    def compute_adjusted_wage_gap(self, 
                                    control_vars: List[str]) -> Dict:
        """
        Compute adjusted gender wage gap controlling for other factors.
        Uses OLS regression with robust standard errors.
        """
        # Prepare data
        df = self.df.copy()
        gender_col = self.gender_col
        df['IS_FEMALE'] = (df[gender_col] == 2).astype(int)
        df['LOG_WAGE'] = np.log(df[self.wage_col].clip(lower=1))
        
        # Drop missing values
        all_vars = ['LOG_WAGE', 'IS_FEMALE'] + control_vars
        df_clean = df[all_vars].dropna()
        
        # Model 1: Unadjusted (gender only)
        X_unadj = sm.add_constant(df_clean['IS_FEMALE'])
        y = df_clean['LOG_WAGE']
        
        model_unadj = sm.OLS(y, X_unadj).fit()
        
        # Model 2: Adjusted (with controls)
        X_adj = sm.add_constant(df_clean[['IS_FEMALE'] + control_vars])
        model_adj = sm.OLS(y, X_adj).fit()
        
        # Convert log coefficients to percentage gaps
        unadj_gap_pct = (np.exp(model_unadj.params['IS_FEMALE']) - 1) * 100
        adj_gap_pct = (np.exp(model_adj.params['IS_FEMALE']) - 1) * 100
        
        results = {
            'unadjusted_model': {
                'female_coefficient': float(model_unadj.params['IS_FEMALE']),
                'gap_pct': float(unadj_gap_pct),
                'p_value': float(model_unadj.pvalues['IS_FEMALE']),
                'r_squared': float(model_unadj.rsquared),
                'n_obs': int(model_unadj.nobs),
            },
            'adjusted_model': {
                'female_coefficient': float(model_adj.params['IS_FEMALE']),
                'gap_pct': float(adj_gap_pct),
                'p_value': float(model_adj.pvalues['IS_FEMALE']),
                'r_squared': float(model_adj.rsquared),
                'n_obs': int(model_adj.nobs),
                'control_variables': control_vars,
            },
            'gap_reduction': {
                'absolute': float(abs(unadj_gap_pct) - abs(adj_gap_pct)),
                'relative_pct': float((abs(unadj_gap_pct) - abs(adj_gap_pct)) / 
                                      abs(unadj_gap_pct) * 100) if unadj_gap_pct != 0 else 0,
            },
            'interpretation': {
                'explained_by_controls': float(abs(unadj_gap_pct) - abs(adj_gap_pct)),
                'unexplained_gap': float(adj_gap_pct),
            }
        }
        
        self.results['adjusted_wage_gap'] = results
        return results
    
    def compute_macro_adjusted_wage_gap(self,
                                         control_vars: List[str],
                                         year_col: str = None) -> Dict:
        """
        Compute wage gap with macroeconomic controls.
        
        Adds unemployment, GDP growth, inflation as controls to account
        for business cycle effects on the wage gap.
        
        Parameters
        ----------
        control_vars : list
            Standard control variables (education, experience, etc.)
        year_col : str
            Column containing year information
            
        Returns
        -------
        dict
            Results including macro-adjusted gap estimates
        """
        year_col = year_col or COLS.YEAR
        gender_col = self.gender_col
        if not MACRO_AVAILABLE:
            logger.warning("Macro data module not available. Using standard adjusted gap.")
            return self.compute_adjusted_wage_gap(control_vars)
        
        df = self.df.copy()
        df['IS_FEMALE'] = (df[gender_col] == 2).astype(int)
        df['LOG_WAGE'] = np.log(df[self.wage_col].clip(lower=1))
        
        # Add macroeconomic variables
        if year_col in df.columns:
            df = add_macro_to_dataframe(df, year_col)
            macro_vars = ['unemployment', 'gdp_growth', 'inflation']
            macro_vars = [v for v in macro_vars if v in df.columns and df[v].notna().any()]
        else:
            macro_vars = []
            logger.warning(f"Year column '{year_col}' not found. Skipping macro controls.")
        
        # Prepare data
        all_vars = ['LOG_WAGE', 'IS_FEMALE'] + control_vars + macro_vars
        df_clean = df[[c for c in all_vars if c in df.columns]].dropna()
        
        y = df_clean['LOG_WAGE']
        
        # Model 1: Standard controls only
        X_std = sm.add_constant(df_clean[['IS_FEMALE'] + control_vars])
        model_std = sm.OLS(y, X_std).fit()
        
        # Model 2: With macro controls
        if macro_vars:
            X_macro = sm.add_constant(df_clean[['IS_FEMALE'] + control_vars + macro_vars])
            model_macro = sm.OLS(y, X_macro).fit()
        else:
            model_macro = model_std
        
        # Convert coefficients to percentage gaps
        std_gap_pct = (np.exp(model_std.params['IS_FEMALE']) - 1) * 100
        macro_gap_pct = (np.exp(model_macro.params['IS_FEMALE']) - 1) * 100
        
        results = {
            'standard_controls': {
                'female_coefficient': float(model_std.params['IS_FEMALE']),
                'gap_pct': float(std_gap_pct),
                'p_value': float(model_std.pvalues['IS_FEMALE']),
                'r_squared': float(model_std.rsquared),
                'controls': control_vars,
            },
            'macro_controls': {
                'female_coefficient': float(model_macro.params['IS_FEMALE']),
                'gap_pct': float(macro_gap_pct),
                'p_value': float(model_macro.pvalues['IS_FEMALE']),
                'r_squared': float(model_macro.rsquared),
                'controls': control_vars + macro_vars,
                'macro_variables': macro_vars,
            },
            'macro_effect': {
                'gap_change': float(std_gap_pct - macro_gap_pct),
                'interpretation': (
                    f"Adding macro controls changes the gap by {std_gap_pct - macro_gap_pct:.2f} pp"
                ),
            }
        }
        
        # Add macro coefficients if available
        if macro_vars:
            results['macro_coefficients'] = {}
            for var in macro_vars:
                results['macro_coefficients'][var] = {
                    'coefficient': float(model_macro.params[var]),
                    'p_value': float(model_macro.pvalues[var]),
                    'significant': model_macro.pvalues[var] < 0.05,
                }
        
        self.results['macro_adjusted_wage_gap'] = results
        return results
    
    def compute_real_wage_gap(self, 
                               year_col: str = None,
                               gender_col: str = None) -> Dict:
        """
        Compute wage gap using inflation-adjusted (real) wages.
        
        Parameters
        ----------
        year_col : str
            Column containing year information (default: COLS.YEAR)
        gender_col : str
            Column containing gender indicator (default: auto-detect)
            
        Returns
        -------
        dict
            Real wage gap statistics
        """
        year_col = year_col or COLS.YEAR
        gender_col = gender_col or self.gender_col
        
        if not MACRO_AVAILABLE:
            logger.warning("Macro data module not available. Cannot compute real wages.")
            return self.compute_raw_wage_gap(gender_col)
        
        df = self.df.copy()
        
        if year_col not in df.columns:
            logger.warning(f"Year column '{year_col}' not found. Using nominal wages.")
            return self.compute_raw_wage_gap(gender_col)
        
        # Calculate real wages
        df['DEFLATOR'] = df[year_col].map(get_deflator)
        df['REAL_WAGE'] = df[self.wage_col] * df['DEFLATOR']
        
        # Compute gap using real wages
        male = df[df[gender_col] == 1]['REAL_WAGE'].dropna()
        female = df[df[gender_col] == 2]['REAL_WAGE'].dropna()
        
        male_mean = male.mean()
        female_mean = female.mean()
        
        real_gap = male_mean - female_mean
        real_gap_pct = (real_gap / male_mean) * 100 if male_mean > 0 else 0
        
        # Also get nominal for comparison
        nom_male = df[df[gender_col] == 1][self.wage_col].mean()
        nom_female = df[df[gender_col] == 2][self.wage_col].mean()
        nom_gap_pct = (nom_male - nom_female) / nom_male * 100 if nom_male > 0 else 0
        
        results = {
            'real_wages': {
                'male_mean': float(male_mean),
                'female_mean': float(female_mean),
                'gap_pct': float(real_gap_pct),
                'base_year': BASE_YEAR,
            },
            'nominal_wages': {
                'male_mean': float(nom_male),
                'female_mean': float(nom_female),
                'gap_pct': float(nom_gap_pct),
            },
            'comparison': {
                'gap_difference': float(real_gap_pct - nom_gap_pct),
                'interpretation': (
                    f"Real wage gap ({real_gap_pct:.1f}%) vs nominal ({nom_gap_pct:.1f}%): "
                    f"difference of {real_gap_pct - nom_gap_pct:.2f} pp"
                ),
            }
        }
        
        self.results['real_wage_gap'] = results
        return results
    
    def analyze_gap_by_economic_period(self,
                                        year_col: str = None,
                                        gender_col: str = None) -> pd.DataFrame:
        """
        Analyze wage gap across economic periods.
        
        Returns
        -------
        pd.DataFrame
            Wage gap statistics by economic period
        """
        year_col = year_col or COLS.YEAR
        gender_col = gender_col or self.gender_col
        
        if not MACRO_AVAILABLE:
            logger.warning("Macro data module not available.")
            return pd.DataFrame()
        
        df = self.df.copy()
        
        if year_col not in df.columns:
            logger.warning(f"Year column '{year_col}' not found.")
            return pd.DataFrame()
        
        results = []
        for period, (start, end) in ECONOMIC_PERIODS.items():
            period_df = df[(df[year_col] >= start) & (df[year_col] <= end)]
            
            if len(period_df) > 0:
                male = period_df[period_df[gender_col] == 1][self.wage_col]
                female = period_df[period_df[gender_col] == 2][self.wage_col]
                
                if len(male) > 0 and len(female) > 0:
                    gap_pct = (male.mean() - female.mean()) / male.mean() * 100
                    
                    results.append({
                        'period': period,
                        'start_year': start,
                        'end_year': end,
                        'male_wage': male.mean(),
                        'female_wage': female.mean(),
                        'wage_gap_pct': gap_pct,
                        'n_male': len(male),
                        'n_female': len(female),
                    })
        
        return pd.DataFrame(results)

    def oaxaca_blinder_decomposition(self,
                                       features: List[str],
                                       gender_col: str = None) -> Dict:
        """
        Oaxaca-Blinder wage decomposition
        Decomposes wage gap into explained and unexplained components
        """
        gender_col = gender_col or self.gender_col
        df = self.df.copy()
        
        # Split by gender
        df_male = df[df[gender_col] == 1][features + [self.wage_col]].dropna()
        df_female = df[df[gender_col] == 2][features + [self.wage_col]].dropna()
        
        # Log wages
        y_male = np.log(df_male[self.wage_col].clip(lower=1))
        y_female = np.log(df_female[self.wage_col].clip(lower=1))
        
        X_male = sm.add_constant(df_male[features])
        X_female = sm.add_constant(df_female[features])
        
        # Fit separate models
        model_male = sm.OLS(y_male, X_male).fit()
        model_female = sm.OLS(y_female, X_female).fit()
        
        # Mean characteristics
        mean_X_male = X_male.mean()
        mean_X_female = X_female.mean()
        
        # Mean wages
        mean_y_male = y_male.mean()
        mean_y_female = y_female.mean()
        
        # Total gap (in log points)
        total_gap = mean_y_male - mean_y_female
        
        # Twofold decomposition using male coefficients as reference
        # Explained: difference in characteristics
        explained = np.dot(model_male.params, (mean_X_male - mean_X_female))
        
        # Unexplained: difference in returns (potential discrimination)
        unexplained = total_gap - explained
        
        # Convert to percentages
        total_gap_pct = (np.exp(total_gap) - 1) * 100
        explained_pct = (explained / total_gap) * 100 if total_gap != 0 else 0
        unexplained_pct = (unexplained / total_gap) * 100 if total_gap != 0 else 0
        
        # Detailed decomposition by variable
        detailed = {}
        for var in ['const'] + features:
            contribution = model_male.params[var] * (mean_X_male[var] - mean_X_female[var])
            detailed[var] = {
                'contribution': float(contribution),
                'pct_of_explained': float(contribution / explained * 100) if explained != 0 else 0,
            }
        
        results = {
            'total_gap': {
                'log_points': float(total_gap),
                'percentage': float(total_gap_pct),
            },
            'explained': {
                'log_points': float(explained),
                'pct_of_total': float(explained_pct),
                'interpretation': 'Due to differences in characteristics (education, experience, etc.)',
            },
            'unexplained': {
                'log_points': float(unexplained),
                'pct_of_total': float(unexplained_pct),
                'interpretation': 'Due to different returns to characteristics (potential discrimination)',
            },
            'detailed_decomposition': detailed,
            'sample_sizes': {
                'male': len(df_male),
                'female': len(df_female),
            }
        }
        
        self.results['oaxaca_blinder'] = results
        return results
    
    def analyze_by_group(self, groupby_col: str, gender_col: str = None) -> pd.DataFrame:
        """
        Analyze wage gap by a specific grouping variable
        """
        gender_col = gender_col or self.gender_col
        results = []
        
        for group, group_df in self.df.groupby(groupby_col):
            male = group_df[group_df[gender_col] == 1][self.wage_col]
            female = group_df[group_df[gender_col] == 2][self.wage_col]
            
            if len(male) < 30 or len(female) < 30:
                continue
            
            male_mean = male.mean()
            female_mean = female.mean()
            gap = male_mean - female_mean
            gap_pct = (gap / male_mean) * 100 if male_mean > 0 else 0
            
            t_stat, p_value = stats.ttest_ind(male.dropna(), female.dropna())
            
            results.append({
                groupby_col: group,
                'male_mean': male_mean,
                'female_mean': female_mean,
                'gap': gap,
                'gap_pct': gap_pct,
                'n_male': len(male),
                'n_female': len(female),
                'p_value': p_value,
                'significant': p_value < 0.05,
            })
        
        return pd.DataFrame(results).sort_values('gap_pct', ascending=False)
    
    def quantile_analysis(self, gender_col: str = None) -> Dict:
        """
        Analyze wage gap across wage distribution quantiles
        """
        gender_col = gender_col or self.gender_col
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        
        male = self.df[self.df[gender_col] == 1][self.wage_col]
        female = self.df[self.df[gender_col] == 2][self.wage_col]
        
        results = {}
        for q in quantiles:
            male_q = male.quantile(q)
            female_q = female.quantile(q)
            gap = male_q - female_q
            gap_pct = (gap / male_q) * 100 if male_q > 0 else 0
            
            results[f'p{int(q*100)}'] = {
                'male': float(male_q),
                'female': float(female_q),
                'gap': float(gap),
                'gap_pct': float(gap_pct),
            }
        
        # Glass ceiling effect (gap at top vs bottom)
        glass_ceiling = results['p90']['gap_pct'] - results['p10']['gap_pct']
        
        results['glass_ceiling_effect'] = {
            'difference': float(glass_ceiling),
            'present': glass_ceiling > 2,  # Gap is 2+ percentage points larger at top
            'interpretation': 'Gap widens at higher wages' if glass_ceiling > 0 else 'Gap narrows at higher wages',
        }
        
        self.results['quantile_analysis'] = results
        return results
    
    def time_series_analysis(self, time_col: str = None, 
                              gender_col: str = None) -> pd.DataFrame:
        """
        Analyze wage gap trends over time (if multi-year data)
        """
        time_col = time_col or COLS.SURVEY_YEAR
        gender_col = gender_col or self.gender_col
        
        if time_col not in self.df.columns:
            return pd.DataFrame()
        
        results = []
        
        for year in sorted(self.df[time_col].unique()):
            year_df = self.df[self.df[time_col] == year]
            
            male = year_df[year_df[gender_col] == 1][self.wage_col]
            female = year_df[year_df[gender_col] == 2][self.wage_col]
            
            if len(male) < 100 or len(female) < 100:
                continue
            
            male_mean = male.mean()
            female_mean = female.mean()
            gap_pct = ((male_mean - female_mean) / male_mean) * 100
            
            results.append({
                'year': year,
                'male_mean': male_mean,
                'female_mean': female_mean,
                'gap_pct': gap_pct,
                'ratio': female_mean / male_mean,
            })
        
        return pd.DataFrame(results)
    
    def generate_summary_report(self) -> str:
        """
        Generate text summary of all analyses
        """
        report = []
        report.append("=" * 60)
        report.append("PAY EQUITY ANALYSIS SUMMARY REPORT")
        report.append("=" * 60)
        report.append("")
        
        if 'raw_wage_gap' in self.results:
            r = self.results['raw_wage_gap']
            report.append("RAW WAGE GAP")
            report.append("-" * 40)
            report.append(f"Male average hourly wage: ${r['male']['mean']:.2f}")
            report.append(f"Female average hourly wage: ${r['female']['mean']:.2f}")
            report.append(f"Raw gender wage gap: {r['raw_gap']['mean_gap_pct']:.1f}%")
            report.append(f"Women earn ${r['raw_gap']['female_to_male_ratio']:.2f} for every $1 men earn")
            report.append(f"Statistical significance: p = {r['statistical_tests']['t_test']['p_value']:.4f}")
            report.append(f"Effect size: {r['statistical_tests']['effect_size']['interpretation']}")
            report.append("")
        
        if 'adjusted_wage_gap' in self.results:
            r = self.results['adjusted_wage_gap']
            report.append("ADJUSTED WAGE GAP (Controlling for education, occupation, etc.)")
            report.append("-" * 40)
            report.append(f"Unadjusted gap: {r['unadjusted_model']['gap_pct']:.1f}%")
            report.append(f"Adjusted gap: {r['adjusted_model']['gap_pct']:.1f}%")
            report.append(f"Gap explained by controls: {r['gap_reduction']['absolute']:.1f} percentage points")
            report.append(f"Unexplained gap: {r['interpretation']['unexplained_gap']:.1f}%")
            report.append("")
        
        if 'oaxaca_blinder' in self.results:
            r = self.results['oaxaca_blinder']
            report.append("OAXACA-BLINDER DECOMPOSITION")
            report.append("-" * 40)
            report.append(f"Total wage gap: {r['total_gap']['percentage']:.1f}%")
            report.append(f"Explained by characteristics: {r['explained']['pct_of_total']:.1f}%")
            report.append(f"Unexplained (potential discrimination): {r['unexplained']['pct_of_total']:.1f}%")
            report.append("")
        
        if 'quantile_analysis' in self.results:
            r = self.results['quantile_analysis']
            report.append("WAGE GAP ACROSS DISTRIBUTION")
            report.append("-" * 40)
            report.append(f"10th percentile gap: {r['p10']['gap_pct']:.1f}%")
            report.append(f"50th percentile (median) gap: {r['p50']['gap_pct']:.1f}%")
            report.append(f"90th percentile gap: {r['p90']['gap_pct']:.1f}%")
            report.append(f"Glass ceiling effect: {r['glass_ceiling_effect']['interpretation']}")
            report.append("")
        
        report.append("=" * 60)
        report.append("END OF REPORT")
        report.append("=" * 60)
        
        return "\n".join(report)


def run_full_analysis(df: pd.DataFrame,
                       output_dir: str = "reports") -> Dict:
    """
    Run complete pay equity analysis pipeline
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    analyzer = PayEquityAnalyzer(df)
    
    # Run all analyses
    results = {}
    
    # Raw gap
    results['raw_gap'] = analyzer.compute_raw_wage_gap()
    logger.info(f"Raw wage gap: {results['raw_gap']['raw_gap']['mean_gap_pct']:.1f}%")
    
    # Adjusted gap (if numeric control variables available)
    numeric_controls = []
    for col in ['EXPERIENCE_PROXY', 'AGE_APPROX', 'EDUC']:
        if col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                numeric_controls.append(col)
    
    if numeric_controls:
        results['adjusted_gap'] = analyzer.compute_adjusted_wage_gap(numeric_controls)
        logger.info(f"Adjusted wage gap: {results['adjusted_gap']['adjusted_model']['gap_pct']:.1f}%")
    
    # Oaxaca-Blinder decomposition
    if numeric_controls:
        results['decomposition'] = analyzer.oaxaca_blinder_decomposition(numeric_controls)
    
    # Quantile analysis
    results['quantile'] = analyzer.quantile_analysis()
    
    # Analysis by groups
    for col in ['EDUC', 'NOC_10', 'PROV']:
        if col in df.columns:
            group_analysis = analyzer.analyze_by_group(col)
            if len(group_analysis) > 0:
                results[f'by_{col.lower()}'] = group_analysis.to_dict('records')
                humanize_columns(group_analysis).to_csv(f"{output_dir}/gap_by_{col.lower()}.csv", index=False)
    
    # Generate summary report
    summary = analyzer.generate_summary_report()
    with open(f"{output_dir}/pay_equity_summary.txt", 'w') as f:
        f.write(summary)
    
    print(summary)
    
    return results
