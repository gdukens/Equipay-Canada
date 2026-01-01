"""
Fairness Analysis Module
Bias detection and mitigation for salary prediction

Uses centralized constants for consistent column naming.
"""

import logging
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Import centralized constants
from .constants import COLS, GENDER_CODES, normalize_column_names, PROTECTED_ATTRIBUTES

# Fairlearn imports
try:
    from fairlearn.metrics import (
        demographic_parity_difference,
        demographic_parity_ratio,
        equalized_odds_difference,
        MetricFrame,
    )
    from fairlearn.reductions import (
        ExponentiatedGradient,
        GridSearch,
        DemographicParity,
        BoundedGroupLoss,
        ErrorRate,
    )
    from fairlearn.postprocessing import ThresholdOptimizer
    HAS_FAIRLEARN = True
except ImportError:
    HAS_FAIRLEARN = False
    warnings.warn("Fairlearn not installed. Some fairness features unavailable.")

logger = logging.getLogger(__name__)


class FairnessAnalyzer:
    """
    Analyze fairness metrics for salary prediction models.
    
    Uses COLS constants for consistent column naming.
    """
    
    # Parity thresholds
    DEMOGRAPHIC_PARITY_THRESHOLD = 0.8
    EQUALIZED_ODDS_THRESHOLD = 0.1
    WAGE_GAP_THRESHOLD = 0.10  # 10% gap
    
    def __init__(self, protected_features: Optional[List[str]] = None):
        """
        Initialize fairness analyzer.
        
        Args:
            protected_features: List of protected attribute column names
        """
        self.protected_features = protected_features or PROTECTED_ATTRIBUTES
        self.metrics = {}
        self.mitigation_results = {}
        
    def compute_group_metrics(self, 
                               y_true: np.ndarray,
                               y_pred: np.ndarray,
                               sensitive_features: pd.DataFrame) -> Dict:
        """
        Compute metrics by demographic group
        """
        if not HAS_FAIRLEARN:
            return self._basic_group_metrics(y_true, y_pred, sensitive_features)
        
        results = {}
        
        for col in sensitive_features.columns:
            groups = sensitive_features[col]
            
            # Create MetricFrame
            metric_frame = MetricFrame(
                metrics={
                    'mean_prediction': lambda y_t, y_p: np.mean(y_p),
                    'mean_actual': lambda y_t, y_p: np.mean(y_t),
                    'mae': lambda y_t, y_p: np.mean(np.abs(y_t - y_p)),
                    'rmse': lambda y_t, y_p: np.sqrt(np.mean((y_t - y_p)**2)),
                },
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=groups,
            )
            
            results[col] = {
                'by_group': metric_frame.by_group.to_dict(),
                'overall': metric_frame.overall.to_dict(),
                'difference': metric_frame.difference().to_dict(),
                'ratio': metric_frame.ratio().to_dict(),
            }
        
        self.metrics['group_metrics'] = results
        return results
    
    def _basic_group_metrics(self, y_true, y_pred, sensitive_features) -> Dict:
        """Fallback when Fairlearn not available"""
        results = {}
        
        for col in sensitive_features.columns:
            groups = sensitive_features[col].unique()
            group_metrics = {}
            
            for group in groups:
                mask = sensitive_features[col] == group
                group_metrics[group] = {
                    'mean_prediction': float(np.mean(y_pred[mask])),
                    'mean_actual': float(np.mean(y_true[mask])),
                    'mae': float(np.mean(np.abs(y_true[mask] - y_pred[mask]))),
                    'count': int(mask.sum()),
                }
            
            results[col] = {'by_group': group_metrics}
        
        return results
    
    def analyze_wage_gap(self, 
                          df: pd.DataFrame,
                          y_pred: Optional[np.ndarray] = None,
                          wage_col: str = None,
                          gender_col: str = None) -> Dict:
        """
        Comprehensive wage gap analysis.
        
        Supports both GENDER (standard) and SEX (legacy) column names.
        """
        # Normalize column names
        df = normalize_column_names(df.copy())
        
        # Determine columns
        wage_col = wage_col or COLS.HOURLY_EARNINGS
        
        # Gender column: prefer GENDER, fall back to SEX
        if gender_col is None:
            gender_col = COLS.GENDER if COLS.GENDER in df.columns else 'SEX'
        
        # Actual wage gap
        male_mask = df[gender_col] == 1
        female_mask = df[gender_col] == 2
        
        male_wage = df.loc[male_mask, wage_col].mean()
        female_wage = df.loc[female_mask, wage_col].mean()
        
        raw_gap = male_wage - female_wage
        raw_gap_pct = (raw_gap / male_wage) * 100
        
        results = {
            'actual': {
                'male_mean': male_wage,
                'female_mean': female_wage,
                'raw_gap': raw_gap,
                'raw_gap_pct': raw_gap_pct,
                'female_to_male_ratio': female_wage / male_wage,
            },
            'sample_sizes': {
                'male': int(male_mask.sum()),
                'female': int(female_mask.sum()),
            }
        }
        
        # Predicted wage gap (if predictions provided)
        if y_pred is not None:
            pred_male = y_pred[male_mask].mean()
            pred_female = y_pred[female_mask].mean()
            pred_gap = pred_male - pred_female
            pred_gap_pct = (pred_gap / pred_male) * 100
            
            results['predicted'] = {
                'male_mean': pred_male,
                'female_mean': pred_female,
                'raw_gap': pred_gap,
                'raw_gap_pct': pred_gap_pct,
                'female_to_male_ratio': pred_female / pred_male,
            }
            
            # Bias amplification check
            results['bias_analysis'] = {
                'gap_amplification': pred_gap_pct - raw_gap_pct,
                'is_amplifying': pred_gap_pct > raw_gap_pct,
                'recommendation': self._get_bias_recommendation(raw_gap_pct, pred_gap_pct)
            }
        
        # Statistical significance
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(
            df.loc[male_mask, wage_col].dropna(),
            df.loc[female_mask, wage_col].dropna()
        )
        
        results['statistical_test'] = {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant_at_05': p_value < 0.05,
            'significant_at_01': p_value < 0.01,
        }
        
        self.metrics['wage_gap'] = results
        return results
    
    def _get_bias_recommendation(self, actual_gap: float, pred_gap: float) -> str:
        """Get recommendation based on gap analysis"""
        if pred_gap > actual_gap + 2:
            return "WARNING: Model amplifies gender wage gap. Consider fairness interventions."
        elif pred_gap < actual_gap - 2:
            return "Model reduces gender wage gap. Monitor for over-correction."
        else:
            return "Model maintains similar gap to observed data."
    
    def compute_fairness_metrics(self,
                                   y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   sensitive_features: pd.Series) -> Dict:
        """
        Compute comprehensive fairness metrics
        """
        # Convert regression to binary outcome for some metrics
        # (e.g., "high wage" vs "low wage")
        median_wage = np.median(y_true)
        y_true_binary = (y_true > median_wage).astype(int)
        y_pred_binary = (y_pred > median_wage).astype(int)
        
        results = {
            'threshold': float(median_wage),
        }
        
        if HAS_FAIRLEARN:
            # Demographic parity
            dp_diff = demographic_parity_difference(
                y_true_binary, y_pred_binary, 
                sensitive_features=sensitive_features
            )
            dp_ratio = demographic_parity_ratio(
                y_true_binary, y_pred_binary,
                sensitive_features=sensitive_features
            )
            
            results['demographic_parity'] = {
                'difference': float(dp_diff),
                'ratio': float(dp_ratio),
                'threshold': self.DEMOGRAPHIC_PARITY_THRESHOLD,
                'passes': dp_ratio >= self.DEMOGRAPHIC_PARITY_THRESHOLD,
            }
            
            # Equalized odds difference
            eo_diff = equalized_odds_difference(
                y_true_binary, y_pred_binary,
                sensitive_features=sensitive_features
            )
            
            results['equalized_odds'] = {
                'difference': float(eo_diff),
                'threshold': self.EQUALIZED_ODDS_THRESHOLD,
                'passes': eo_diff <= self.EQUALIZED_ODDS_THRESHOLD,
            }
        
        # Mean prediction by group (doesn't require binarization)
        groups = sensitive_features.unique()
        group_means = {}
        for group in groups:
            mask = sensitive_features == group
            group_means[str(group)] = float(np.mean(y_pred[mask]))
        
        max_mean = max(group_means.values())
        min_mean = min(group_means.values())
        
        results['prediction_parity'] = {
            'group_means': group_means,
            'ratio': min_mean / max_mean if max_mean > 0 else 0,
            'difference': max_mean - min_mean,
        }
        
        self.metrics['fairness'] = results
        return results
    
    def intersectional_analysis(self,
                                 df: pd.DataFrame,
                                 y_pred: np.ndarray,
                                 wage_col: str = 'HRLYEARN') -> pd.DataFrame:
        """
        Analyze wage gaps at intersections of protected attributes
        """
        # Create intersectional groups
        df = df.copy()
        df['predicted'] = y_pred
        
        # Determine gender column (prefer GENDER, fall back to SEX)
        gender_col = COLS.GENDER if COLS.GENDER in df.columns else 'SEX'
        
        # Gender x Education
        educ_col = COLS.EDUCATION if COLS.EDUCATION in df.columns else 'EDUC'
        if gender_col in df.columns and educ_col in df.columns:
            gender_educ = df.groupby([gender_col, educ_col]).agg({
                wage_col: ['mean', 'median', 'count'],
                'predicted': ['mean', 'median'],
            }).round(2)
            gender_educ.columns = ['actual_mean', 'actual_median', 'count',
                                   'pred_mean', 'pred_median']
        else:
            gender_educ = pd.DataFrame()
        
        # Gender x Occupation
        occ_col = COLS.OCCUPATION_10 if COLS.OCCUPATION_10 in df.columns else 'NOC_10'
        if gender_col in df.columns and occ_col in df.columns:
            gender_occ = df.groupby([gender_col, occ_col]).agg({
                wage_col: ['mean', 'count'],
            }).round(2)
            gender_occ.columns = ['actual_mean', 'count']
        else:
            gender_occ = pd.DataFrame()
        
        # Gender x Full-time status
        ftpt_col = COLS.FULLTIME_PARTTIME if COLS.FULLTIME_PARTTIME in df.columns else 'FTPTMAIN'
        if gender_col in df.columns and ftpt_col in df.columns:
            gender_ft = df.groupby([gender_col, ftpt_col]).agg({
                wage_col: ['mean', 'count'],
            }).round(2)
            gender_ft.columns = ['actual_mean', 'count']
        else:
            gender_ft = pd.DataFrame()
        
        return {
            'gender_education': gender_educ,
            'gender_occupation': gender_occ,
            'gender_fulltime': gender_ft,
        }


class FairnessMitigator:
    """
    Apply fairness mitigation techniques
    """
    
    def __init__(self, constraint: str = 'demographic_parity'):
        """
        Initialize mitigator
        
        Args:
            constraint: 'demographic_parity' or 'bounded_group_loss'
        """
        self.constraint = constraint
        self.mitigated_model = None
        self.original_metrics = {}
        self.mitigated_metrics = {}
        
    def mitigate_bias(self,
                       estimator,
                       X_train: np.ndarray,
                       y_train: np.ndarray,
                       sensitive_train: np.ndarray,
                       X_test: np.ndarray,
                       y_test: np.ndarray,
                       sensitive_test: np.ndarray) -> Dict:
        """
        Apply in-processing fairness mitigation
        """
        if not HAS_FAIRLEARN:
            logger.warning("Fairlearn not installed. Cannot apply mitigation.")
            return {}
        
        # Original model performance
        estimator.fit(X_train, y_train)
        y_pred_orig = estimator.predict(X_test)
        
        self.original_metrics = {
            'rmse': float(np.sqrt(np.mean((y_test - y_pred_orig)**2))),
            'mae': float(np.mean(np.abs(y_test - y_pred_orig))),
        }
        
        # Add group-level metrics
        analyzer = FairnessAnalyzer()
        orig_fairness = analyzer.compute_fairness_metrics(
            y_test, y_pred_orig, pd.Series(sensitive_test)
        )
        self.original_metrics['fairness'] = orig_fairness
        
        # Apply ExponentiatedGradient for regression with BoundedGroupLoss
        try:
            constraint_obj = BoundedGroupLoss(
                loss='absolute',
                upper_bound=0.01,
            )
            
            mitigator = ExponentiatedGradient(
                estimator=estimator,
                constraints=constraint_obj,
                eps=0.01,
                max_iter=50,
            )
            
            mitigator.fit(X_train, y_train, sensitive_features=sensitive_train)
            
            self.mitigated_model = mitigator
            y_pred_mitigated = mitigator.predict(X_test)
            
            self.mitigated_metrics = {
                'rmse': float(np.sqrt(np.mean((y_test - y_pred_mitigated)**2))),
                'mae': float(np.mean(np.abs(y_test - y_pred_mitigated))),
            }
            
            mitigated_fairness = analyzer.compute_fairness_metrics(
                y_test, y_pred_mitigated, pd.Series(sensitive_test)
            )
            self.mitigated_metrics['fairness'] = mitigated_fairness
            
        except Exception as e:
            logger.error(f"Mitigation failed: {e}")
            return {'error': str(e)}
        
        return {
            'original': self.original_metrics,
            'mitigated': self.mitigated_metrics,
            'improvement': self._calculate_improvement(),
        }
    
    def _calculate_improvement(self) -> Dict:
        """Calculate fairness improvement"""
        improvement = {}
        
        if 'fairness' in self.original_metrics and 'fairness' in self.mitigated_metrics:
            orig = self.original_metrics['fairness']
            mit = self.mitigated_metrics['fairness']
            
            if 'demographic_parity' in orig and 'demographic_parity' in mit:
                improvement['dp_ratio_change'] = (
                    mit['demographic_parity']['ratio'] - 
                    orig['demographic_parity']['ratio']
                )
        
        improvement['rmse_change'] = (
            self.mitigated_metrics.get('rmse', 0) - 
            self.original_metrics.get('rmse', 0)
        )
        
        return improvement


def generate_fairness_report(df: pd.DataFrame,
                              y_pred: np.ndarray,
                              output_path: str = "reports/fairness_report.html") -> str:
    """
    Generate comprehensive fairness report
    """
    analyzer = FairnessAnalyzer()
    
    # Determine gender and wage columns
    gender_col = COLS.GENDER if COLS.GENDER in df.columns else 'SEX'
    wage_col = COLS.HOURLY_EARNINGS if COLS.HOURLY_EARNINGS in df.columns else 'HRLYEARN'
    
    # Wage gap analysis
    wage_gap = analyzer.analyze_wage_gap(df, y_pred)
    
    # Group metrics
    age_col = COLS.AGE_12 if COLS.AGE_12 in df.columns else 'AGE_12'
    if age_col in df.columns:
        sensitive_df = df[[gender_col, age_col]].copy()
    else:
        sensitive_df = df[[gender_col]].copy()
        
    group_metrics = analyzer.compute_group_metrics(
        df[wage_col].values,
        y_pred,
        sensitive_df
    )
    
    # Fairness metrics
    fairness = analyzer.compute_fairness_metrics(
        df[wage_col].values,
        y_pred,
        df[gender_col]
    )
    
    # Intersectional analysis
    intersectional = analyzer.intersectional_analysis(df, y_pred)
    
    # Generate HTML report
    html = _generate_html_report(wage_gap, group_metrics, fairness, intersectional)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html)
    
    logger.info(f"Fairness report saved to {output_path}")
    return output_path


def _generate_html_report(wage_gap: Dict, group_metrics: Dict, 
                           fairness: Dict, intersectional: Dict) -> str:
    """Generate HTML fairness report"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>EquiPay Canada - Fairness Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; }
            h1 { color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }
            h2 { color: #555; margin-top: 30px; }
            .metric-box { background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 15px 0; }
            .metric-value { font-size: 24px; font-weight: bold; color: #007bff; }
            .metric-label { color: #666; font-size: 14px; }
            .warning { color: #dc3545; }
            .success { color: #28a745; }
            table { width: 100%; border-collapse: collapse; margin: 20px 0; }
            th, td { border: 1px solid #dee2e6; padding: 12px; text-align: left; }
            th { background: #007bff; color: white; }
            tr:nth-child(even) { background: #f8f9fa; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>EquiPay Canada - Pay Equity Fairness Report</h1>
    """
    
    # Wage Gap Section
    html += """
            <h2>Gender Wage Gap Analysis</h2>
            <div class="metric-box">
    """
    
    if 'actual' in wage_gap:
        gap_pct = wage_gap['actual']['raw_gap_pct']
        ratio = wage_gap['actual']['female_to_male_ratio']
        
        html += f"""
                <div class="metric-value {'warning' if gap_pct > 10 else ''}">{gap_pct:.1f}%</div>
                <div class="metric-label">Raw Gender Wage Gap</div>
                <p>Women earn ${ratio:.2f} for every $1.00 men earn</p>
                <p>Male average: ${wage_gap['actual']['male_mean']:.2f}/hr | 
                   Female average: ${wage_gap['actual']['female_mean']:.2f}/hr</p>
        """
    
    html += "</div>"
    
    # Statistical Significance
    if 'statistical_test' in wage_gap:
        sig = wage_gap['statistical_test']
        html += f"""
            <h2>Statistical Significance</h2>
            <div class="metric-box">
                <p>T-statistic: {sig['t_statistic']:.2f}</p>
                <p>P-value: {sig['p_value']:.4f}</p>
                <p class="{'success' if sig['significant_at_01'] else 'warning'}">
                    {'Statistically significant at p < 0.01' if sig['significant_at_01'] 
                     else 'Not statistically significant'}
                </p>
            </div>
        """
    
    # Fairness Metrics
    html += """
            <h2>Fairness Metrics</h2>
            <div class="metric-box">
    """
    
    if 'demographic_parity' in fairness:
        dp = fairness['demographic_parity']
        html += f"""
                <h3>Demographic Parity</h3>
                <p>Ratio: {dp['ratio']:.3f} (threshold: {dp['threshold']})</p>
                <p class="{'success' if dp['passes'] else 'warning'}">
                    {'PASSES' if dp['passes'] else 'FAILS'} demographic parity check
                </p>
        """
    
    if 'prediction_parity' in fairness:
        pp = fairness['prediction_parity']
        html += f"""
                <h3>Prediction Parity</h3>
                <p>Prediction ratio: {pp['ratio']:.3f}</p>
                <p>Group means: {pp['group_means']}</p>
        """
    
    html += "</div>"
    
    # Close HTML
    html += """
            <h2>Recommendations</h2>
            <div class="metric-box">
                <ul>
                    <li>Continue monitoring wage gaps across all demographic groups</li>
                    <li>Investigate occupation-specific gaps for targeted interventions</li>
                    <li>Consider fairness constraints in model retraining</li>
                    <li>Document compliance with pay equity legislation</li>
                </ul>
            </div>
            
            <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #dee2e6; color: #666;">
                <p>Generated by EquiPay Canada Pay Equity Analysis System</p>
            </footer>
        </div>
    </body>
    </html>
    """
    
    return html
