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
    
    def intersectional_analysis(self, dimensions: List[Tuple[str, str]] = None) -> Dict:
        """
        Analyze wage gaps at intersections of multiple identity dimensions.
        
        Examines how wage gaps compound when considering multiple factors
        (e.g., immigrant women, mothers with young children).
        
        Parameters
        ----------
        dimensions : list of tuples
            Pairs of binary columns to analyze, e.g., [('IS_FEMALE', 'IS_IMMIGRANT')]
            
        Returns
        -------
        dict
            Intersectional wage gap analysis results
        """
        if dimensions is None:
            dimensions = [
                ('IS_FEMALE', 'IS_IMMIGRANT'),
                ('IS_FEMALE', 'HAS_YOUNG_CHILDREN'),
                ('IS_FEMALE', 'IS_PUBLIC_SECTOR'),
                ('IS_FEMALE', 'HAS_DEGREE'),
                ('IS_FEMALE', 'IS_URBAN'),
                ('IS_IMMIGRANT', 'HAS_DEGREE'),
            ]
        
        results = {}
        
        for dim1, dim2 in dimensions:
            if dim1 not in self.df.columns or dim2 not in self.df.columns:
                continue
                
            key = f"{dim1}_x_{dim2}"
            results[key] = {}
            
            # Create intersection groups
            df = self.df.copy()
            df['_group'] = df[dim1].astype(str) + '_' + df[dim2].astype(str)
            
            # Calculate wages for each intersection
            group_stats = df.groupby('_group')[self.wage_col].agg(['mean', 'median', 'count'])
            
            # Map to meaningful labels
            label_map = {
                '0_0': f'Not {dim1[3:].lower()}, Not {dim2[3:].lower()}',
                '0_1': f'Not {dim1[3:].lower()}, {dim2[3:].lower()}',
                '1_0': f'{dim1[3:].lower()}, Not {dim2[3:].lower()}',
                '1_1': f'{dim1[3:].lower()} AND {dim2[3:].lower()}',
            }
            
            for group_code, stats in group_stats.iterrows():
                label = label_map.get(group_code, group_code)
                results[key][label] = {
                    'mean_wage': float(stats['mean']),
                    'median_wage': float(stats['median']),
                    'n': int(stats['count']),
                }
            
            # Calculate double disadvantage
            if '0_0' in group_stats.index and '1_1' in group_stats.index:
                baseline = group_stats.loc['0_0', 'mean']
                intersection = group_stats.loc['1_1', 'mean']
                results[key]['double_disadvantage'] = {
                    'gap_pct': float((baseline - intersection) / baseline * 100) if baseline > 0 else 0,
                    'interpretation': f'Compounded disadvantage of {dim1} and {dim2}',
                }
        
        self.results['intersectional'] = results
        return results
    
    def motherhood_penalty_analysis(self) -> Dict:
        """
        Analyze the wage penalty associated with having children,
        with particular focus on the gender difference (motherhood penalty vs fatherhood premium).
        """
        results = {}
        
        # Required columns
        child_cols = ['HAS_CHILDREN', 'HAS_YOUNG_CHILDREN', 'IS_MOTHER_YOUNG_CHILD', 'IS_FATHER_YOUNG_CHILD']
        available = [c for c in child_cols if c in self.df.columns]
        
        if not available or 'IS_FEMALE' not in self.df.columns:
            return {'error': 'Required parenthood columns not available'}
        
        df = self.df[self.df[self.wage_col] > 0].copy()
        
        # Overall parenthood effect
        if 'HAS_CHILDREN' in df.columns:
            parents = df[df['HAS_CHILDREN'] == 1][self.wage_col]
            non_parents = df[df['HAS_CHILDREN'] == 0][self.wage_col]
            
            if len(parents) > 100 and len(non_parents) > 100:
                results['overall'] = {
                    'parent_mean': float(parents.mean()),
                    'non_parent_mean': float(non_parents.mean()),
                    'gap_pct': float((non_parents.mean() - parents.mean()) / non_parents.mean() * 100),
                    'n_parents': len(parents),
                    'n_non_parents': len(non_parents),
                }
        
        # Gender-specific parenthood effects
        for gender, label in [(1, 'fathers'), (2, 'mothers')]:
            gender_df = df[df['IS_FEMALE'] == (gender == 2)]
            
            if 'HAS_YOUNG_CHILDREN' in gender_df.columns:
                with_kids = gender_df[gender_df['HAS_YOUNG_CHILDREN'] == 1][self.wage_col]
                without_kids = gender_df[gender_df['HAS_YOUNG_CHILDREN'] == 0][self.wage_col]
                
                if len(with_kids) > 50 and len(without_kids) > 50:
                    gap = (without_kids.mean() - with_kids.mean()) / without_kids.mean() * 100
                    results[label] = {
                        'with_young_children_mean': float(with_kids.mean()),
                        'without_young_children_mean': float(without_kids.mean()),
                        'gap_pct': float(gap),
                        'effect': 'penalty' if gap > 0 else 'premium',
                        'n_with': len(with_kids),
                        'n_without': len(without_kids),
                    }
        
        # Compare motherhood penalty vs fatherhood effect
        if 'mothers' in results and 'fathers' in results:
            mom_gap = results['mothers']['gap_pct']
            dad_gap = results['fathers']['gap_pct']
            
            results['gender_difference'] = {
                'motherhood_penalty': float(mom_gap),
                'fatherhood_effect': float(dad_gap),
                'difference': float(mom_gap - dad_gap),
                'interpretation': (
                    f"Mothers face a {abs(mom_gap):.1f}% {'penalty' if mom_gap > 0 else 'premium'}, "
                    f"while fathers see a {abs(dad_gap):.1f}% {'penalty' if dad_gap > 0 else 'premium'}. "
                    f"Gender gap in parenthood effect: {abs(mom_gap - dad_gap):.1f} pp"
                ),
            }
        
        self.results['motherhood_penalty'] = results
        return results
    
    def immigrant_wage_gap_analysis(self) -> Dict:
        """
        Analyze wage gaps by immigration status, including credential recognition effects.
        """
        results = {}
        
        if 'IS_IMMIGRANT' not in self.df.columns:
            return {'error': 'Immigration status column not available'}
        
        df = self.df[self.df[self.wage_col] > 0].copy()
        
        # Overall immigrant gap
        immigrants = df[df['IS_IMMIGRANT'] == 1][self.wage_col]
        non_immigrants = df[df['IS_IMMIGRANT'] == 0][self.wage_col]
        
        if len(immigrants) > 100 and len(non_immigrants) > 100:
            results['overall'] = {
                'immigrant_mean': float(immigrants.mean()),
                'non_immigrant_mean': float(non_immigrants.mean()),
                'gap_pct': float((non_immigrants.mean() - immigrants.mean()) / non_immigrants.mean() * 100),
                'n_immigrants': len(immigrants),
                'n_non_immigrants': len(non_immigrants),
            }
        
        # Credential recognition: immigrant wage gap by education level
        if 'HAS_DEGREE' in df.columns:
            for degree, label in [(0, 'without_degree'), (1, 'with_degree')]:
                subset = df[df['HAS_DEGREE'] == degree]
                imm = subset[subset['IS_IMMIGRANT'] == 1][self.wage_col]
                non_imm = subset[subset['IS_IMMIGRANT'] == 0][self.wage_col]
                
                if len(imm) > 50 and len(non_imm) > 50:
                    results[f'by_education_{label}'] = {
                        'immigrant_mean': float(imm.mean()),
                        'non_immigrant_mean': float(non_imm.mean()),
                        'gap_pct': float((non_imm.mean() - imm.mean()) / non_imm.mean() * 100),
                    }
            
            # Credential penalty (difference in gap)
            if 'by_education_with_degree' in results and 'by_education_without_degree' in results:
                gap_with = results['by_education_with_degree']['gap_pct']
                gap_without = results['by_education_without_degree']['gap_pct']
                
                results['credential_recognition'] = {
                    'gap_with_degree': float(gap_with),
                    'gap_without_degree': float(gap_without),
                    'credential_penalty': float(gap_with - gap_without),
                    'interpretation': (
                        f"Immigrant gap is {abs(gap_with - gap_without):.1f}pp "
                        f"{'larger' if gap_with > gap_without else 'smaller'} for degree holders, "
                        f"suggesting {'credential recognition issues' if gap_with > gap_without else 'education helps close the gap'}"
                    ),
                }
        
        # By gender
        for gender, label in [(1, 'male'), (2, 'female')]:
            gender_df = df[df['IS_FEMALE'] == (gender == 2)]
            imm = gender_df[gender_df['IS_IMMIGRANT'] == 1][self.wage_col]
            non_imm = gender_df[gender_df['IS_IMMIGRANT'] == 0][self.wage_col]
            
            if len(imm) > 50 and len(non_imm) > 50:
                results[f'{label}_immigrants'] = {
                    'immigrant_mean': float(imm.mean()),
                    'non_immigrant_mean': float(non_imm.mean()),
                    'gap_pct': float((non_imm.mean() - imm.mean()) / non_imm.mean() * 100),
                }
        
        self.results['immigrant_gap'] = results
        return results
    
    def urban_rural_gap_analysis(self) -> Dict:
        """
        Analyze wage gaps between urban and rural areas.
        """
        results = {}
        
        if 'IS_URBAN' not in self.df.columns:
            return {'error': 'Urban/rural column not available'}
        
        df = self.df[self.df[self.wage_col] > 0].copy()
        
        # Overall urban/rural gap
        urban = df[df['IS_URBAN'] == 1][self.wage_col]
        rural = df[df['IS_URBAN'] == 0][self.wage_col]
        
        if len(urban) > 100 and len(rural) > 100:
            results['overall'] = {
                'urban_mean': float(urban.mean()),
                'rural_mean': float(rural.mean()),
                'urban_premium_pct': float((urban.mean() - rural.mean()) / rural.mean() * 100),
                'n_urban': len(urban),
                'n_rural': len(rural),
            }
        
        # Gender gap within urban vs rural
        for area, label in [(1, 'urban'), (0, 'rural')]:
            area_df = df[df['IS_URBAN'] == area]
            male = area_df[area_df['IS_FEMALE'] == 0][self.wage_col]
            female = area_df[area_df['IS_FEMALE'] == 1][self.wage_col]
            
            if len(male) > 50 and len(female) > 50:
                results[f'{label}_gender_gap'] = {
                    'male_mean': float(male.mean()),
                    'female_mean': float(female.mean()),
                    'gap_pct': float((male.mean() - female.mean()) / male.mean() * 100),
                }
        
        # Compare gender gaps
        if 'urban_gender_gap' in results and 'rural_gender_gap' in results:
            results['gender_gap_comparison'] = {
                'urban_gap': results['urban_gender_gap']['gap_pct'],
                'rural_gap': results['rural_gender_gap']['gap_pct'],
                'difference': results['urban_gender_gap']['gap_pct'] - results['rural_gender_gap']['gap_pct'],
                'interpretation': (
                    f"Gender gap is {abs(results['urban_gender_gap']['gap_pct'] - results['rural_gender_gap']['gap_pct']):.1f}pp "
                    f"{'larger in urban' if results['urban_gender_gap']['gap_pct'] > results['rural_gender_gap']['gap_pct'] else 'larger in rural'} areas"
                ),
            }
        
        # Major city analysis
        if 'IS_MAJOR_CITY' in df.columns:
            major = df[df['IS_MAJOR_CITY'] == 1][self.wage_col]
            other_urban = df[(df['IS_URBAN'] == 1) & (df['IS_MAJOR_CITY'] == 0)][self.wage_col]
            
            if len(major) > 100 and len(other_urban) > 100:
                results['major_city_premium'] = {
                    'major_city_mean': float(major.mean()),
                    'other_urban_mean': float(other_urban.mean()),
                    'premium_pct': float((major.mean() - other_urban.mean()) / other_urban.mean() * 100),
                }
        
        self.results['urban_rural_gap'] = results
        return results
    
    def sector_analysis(self) -> Dict:
        """
        Analyze wage gaps by employment sector (public vs private, self-employed).
        """
        results = {}
        
        sector_cols = ['IS_PUBLIC_SECTOR', 'IS_PRIVATE_SECTOR', 'IS_SELF_EMPLOYED']
        available = [c for c in sector_cols if c in self.df.columns]
        
        if not available:
            return {'error': 'Sector columns not available'}
        
        df = self.df[self.df[self.wage_col] > 0].copy()
        
        # Public vs Private
        if 'IS_PUBLIC_SECTOR' in df.columns and 'IS_PRIVATE_SECTOR' in df.columns:
            public = df[df['IS_PUBLIC_SECTOR'] == 1][self.wage_col]
            private = df[df['IS_PRIVATE_SECTOR'] == 1][self.wage_col]
            
            if len(public) > 100 and len(private) > 100:
                results['public_vs_private'] = {
                    'public_mean': float(public.mean()),
                    'private_mean': float(private.mean()),
                    'public_premium_pct': float((public.mean() - private.mean()) / private.mean() * 100),
                    'n_public': len(public),
                    'n_private': len(private),
                }
        
        # Gender gap by sector
        for sector_col, label in [('IS_PUBLIC_SECTOR', 'public'), ('IS_PRIVATE_SECTOR', 'private')]:
            if sector_col not in df.columns:
                continue
                
            sector_df = df[df[sector_col] == 1]
            male = sector_df[sector_df['IS_FEMALE'] == 0][self.wage_col]
            female = sector_df[sector_df['IS_FEMALE'] == 1][self.wage_col]
            
            if len(male) > 50 and len(female) > 50:
                results[f'{label}_gender_gap'] = {
                    'male_mean': float(male.mean()),
                    'female_mean': float(female.mean()),
                    'gap_pct': float((male.mean() - female.mean()) / male.mean() * 100),
                }
        
        # Compare gender gaps across sectors
        if 'public_gender_gap' in results and 'private_gender_gap' in results:
            results['sector_gender_gap_comparison'] = {
                'public_gap': results['public_gender_gap']['gap_pct'],
                'private_gap': results['private_gender_gap']['gap_pct'],
                'difference': results['public_gender_gap']['gap_pct'] - results['private_gender_gap']['gap_pct'],
                'interpretation': (
                    f"Gender gap is {abs(results['public_gender_gap']['gap_pct'] - results['private_gender_gap']['gap_pct']):.1f}pp "
                    f"{'smaller in public' if results['public_gender_gap']['gap_pct'] < results['private_gender_gap']['gap_pct'] else 'smaller in private'} sector"
                ),
            }
        
        self.results['sector_analysis'] = results
        return results
    
    def precarious_work_analysis(self) -> Dict:
        """
        Analyze wage patterns for precarious workers (temporary, seasonal, involuntary part-time).
        """
        results = {}
        
        precarious_cols = ['IS_PRECARIOUS', 'IS_TEMPORARY', 'IS_SEASONAL', 'IS_INVOLUNTARY_PT']
        available = [c for c in precarious_cols if c in self.df.columns]
        
        if not available:
            return {'error': 'Precarious work columns not available'}
        
        df = self.df[self.df[self.wage_col] > 0].copy()
        
        # Overall precarious vs permanent
        if 'IS_PRECARIOUS' in df.columns:
            precarious = df[df['IS_PRECARIOUS'] == 1][self.wage_col]
            stable = df[df['IS_PRECARIOUS'] == 0][self.wage_col]
            
            if len(precarious) > 100 and len(stable) > 100:
                results['overall'] = {
                    'precarious_mean': float(precarious.mean()),
                    'stable_mean': float(stable.mean()),
                    'penalty_pct': float((stable.mean() - precarious.mean()) / stable.mean() * 100),
                    'n_precarious': len(precarious),
                    'n_stable': len(stable),
                }
        
        # Gender composition of precarious work
        if 'IS_PRECARIOUS' in df.columns and 'IS_FEMALE' in df.columns:
            precarious_pct_female = df[df['IS_PRECARIOUS'] == 1]['IS_FEMALE'].mean() * 100
            stable_pct_female = df[df['IS_PRECARIOUS'] == 0]['IS_FEMALE'].mean() * 100
            
            results['gender_composition'] = {
                'precarious_pct_female': float(precarious_pct_female),
                'stable_pct_female': float(stable_pct_female),
                'overrepresentation': float(precarious_pct_female - stable_pct_female),
                'interpretation': (
                    f"Women are {abs(precarious_pct_female - stable_pct_female):.1f}pp "
                    f"{'over' if precarious_pct_female > stable_pct_female else 'under'}represented in precarious work"
                ),
            }
        
        # Involuntary part-time
        if 'IS_INVOLUNTARY_PT' in df.columns:
            invol_pt = df[df['IS_INVOLUNTARY_PT'] == 1][self.wage_col]
            voluntary = df[df['IS_INVOLUNTARY_PT'] == 0][self.wage_col]
            
            if len(invol_pt) > 50:
                results['involuntary_pt'] = {
                    'mean_wage': float(invol_pt.mean()),
                    'comparison_mean': float(voluntary.mean()),
                    'penalty_pct': float((voluntary.mean() - invol_pt.mean()) / voluntary.mean() * 100),
                    'n': len(invol_pt),
                    'pct_female': float(df[df['IS_INVOLUNTARY_PT'] == 1]['IS_FEMALE'].mean() * 100) if 'IS_FEMALE' in df.columns else None,
                }
        
        self.results['precarious_work'] = results
        return results
    
    def overtime_analysis(self) -> Dict:
        """
        Analyze overtime patterns by gender, focusing on unpaid overtime disparities.
        """
        results = {}
        
        ot_cols = ['WORKS_OVERTIME', 'HAS_PAID_OVERTIME', 'HAS_UNPAID_OVERTIME', 'TOTAL_OT_HOURS']
        available = [c for c in ot_cols if c in self.df.columns]
        
        if not available:
            return {'error': 'Overtime columns not available'}
        
        df = self.df[self.df[self.wage_col] > 0].copy()
        
        # Overtime rates by gender
        if 'WORKS_OVERTIME' in df.columns and 'IS_FEMALE' in df.columns:
            male_ot_rate = df[df['IS_FEMALE'] == 0]['WORKS_OVERTIME'].mean() * 100
            female_ot_rate = df[df['IS_FEMALE'] == 1]['WORKS_OVERTIME'].mean() * 100
            
            results['overtime_rates'] = {
                'male_pct': float(male_ot_rate),
                'female_pct': float(female_ot_rate),
                'difference': float(male_ot_rate - female_ot_rate),
            }
        
        # Unpaid overtime by gender
        if 'HAS_UNPAID_OVERTIME' in df.columns and 'IS_FEMALE' in df.columns:
            male_unpaid = df[df['IS_FEMALE'] == 0]['HAS_UNPAID_OVERTIME'].mean() * 100
            female_unpaid = df[df['IS_FEMALE'] == 1]['HAS_UNPAID_OVERTIME'].mean() * 100
            
            results['unpaid_overtime'] = {
                'male_pct': float(male_unpaid),
                'female_pct': float(female_unpaid),
                'difference': float(male_unpaid - female_unpaid),
                'interpretation': (
                    f"{'Women' if female_unpaid > male_unpaid else 'Men'} are more likely to work unpaid overtime "
                    f"({abs(female_unpaid - male_unpaid):.1f}pp difference)"
                ),
            }
        
        # Wages of those who work unpaid OT only vs those with paid OT
        if 'UNPAID_OT_ONLY' in df.columns:
            unpaid_only = df[df['UNPAID_OT_ONLY'] == 1][self.wage_col]
            paid_ot = df[df['HAS_PAID_OVERTIME'] == 1][self.wage_col]
            
            if len(unpaid_only) > 50 and len(paid_ot) > 50:
                results['unpaid_vs_paid'] = {
                    'unpaid_only_mean': float(unpaid_only.mean()),
                    'paid_ot_mean': float(paid_ot.mean()),
                    'gap_pct': float((paid_ot.mean() - unpaid_only.mean()) / paid_ot.mean() * 100),
                }
        
        self.results['overtime_analysis'] = results
        return results

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
                       output_dir: str = "reports",
                       extended: bool = True) -> Dict:
    """
    Run complete pay equity analysis pipeline
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with wage and demographic information
    output_dir : str
        Directory to save reports
    extended : bool
        If True, run extended analyses using all available features
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    analyzer = PayEquityAnalyzer(df)
    
    # Run all analyses
    results = {}
    
    # === Core Analyses ===
    
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
    
    # === Extended Analyses (using new features) ===
    
    if extended:
        # Intersectional analysis
        if 'IS_FEMALE' in df.columns and 'IS_IMMIGRANT' in df.columns:
            results['intersectional'] = analyzer.intersectional_analysis()
            logger.info("Completed intersectional analysis")
        
        # Motherhood penalty
        if 'HAS_YOUNG_CHILDREN' in df.columns:
            results['motherhood'] = analyzer.motherhood_penalty_analysis()
            if 'gender_difference' in results.get('motherhood', {}):
                logger.info(f"Motherhood penalty: {results['motherhood']['gender_difference']['interpretation']}")
        
        # Immigrant wage gap
        if 'IS_IMMIGRANT' in df.columns:
            results['immigrant_gap'] = analyzer.immigrant_wage_gap_analysis()
            if 'overall' in results.get('immigrant_gap', {}):
                logger.info(f"Immigrant wage gap: {results['immigrant_gap']['overall']['gap_pct']:.1f}%")
        
        # Urban/rural gap
        if 'IS_URBAN' in df.columns:
            results['urban_rural'] = analyzer.urban_rural_gap_analysis()
            if 'overall' in results.get('urban_rural', {}):
                logger.info(f"Urban premium: {results['urban_rural']['overall']['urban_premium_pct']:.1f}%")
        
        # Sector analysis
        if 'IS_PUBLIC_SECTOR' in df.columns:
            results['sector'] = analyzer.sector_analysis()
            if 'public_vs_private' in results.get('sector', {}):
                logger.info(f"Public sector premium: {results['sector']['public_vs_private']['public_premium_pct']:.1f}%")
        
        # Precarious work
        if 'IS_PRECARIOUS' in df.columns:
            results['precarious'] = analyzer.precarious_work_analysis()
            if 'overall' in results.get('precarious', {}):
                logger.info(f"Precarious work penalty: {results['precarious']['overall']['penalty_pct']:.1f}%")
        
        # Overtime analysis
        if 'WORKS_OVERTIME' in df.columns:
            results['overtime'] = analyzer.overtime_analysis()
            logger.info("Completed overtime analysis")
        
        # Additional group analyses with new features
        for col in ['MARSTAT', 'IMMIG', 'COWMAIN', 'AGYOWNK', 'CMA_TYPE']:
            if col in df.columns:
                try:
                    group_analysis = analyzer.analyze_by_group(col)
                    if len(group_analysis) > 0:
                        results[f'by_{col.lower()}'] = group_analysis.to_dict('records')
                        humanize_columns(group_analysis).to_csv(f"{output_dir}/gap_by_{col.lower()}.csv", index=False)
                except Exception as e:
                    logger.warning(f"Could not analyze by {col}: {e}")
    
    # Generate summary report
    summary = analyzer.generate_summary_report()
    with open(f"{output_dir}/pay_equity_summary.txt", 'w') as f:
        f.write(summary)
    
    print(summary)
    
    return results
