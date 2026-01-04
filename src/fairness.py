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
from .constants import (
    COLS, GENDER_CODES, normalize_column_names, PROTECTED_ATTRIBUTES,
    INTERSECTIONAL_ATTRIBUTES
)

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
    Supports extended analysis with new demographic features.
    """
    
    # Parity thresholds
    DEMOGRAPHIC_PARITY_THRESHOLD = 0.8
    EQUALIZED_ODDS_THRESHOLD = 0.1
    WAGE_GAP_THRESHOLD = 0.10  # 10% gap
    
    # Extended protected attributes (beyond gender)
    EXTENDED_PROTECTED = [
        'GENDER', 'IS_FEMALE',
        'IMMIG', 'IS_IMMIGRANT',
        'AGE_6',
        'PROV',
        'HAS_YOUNG_CHILDREN',
        'IS_LONE_PARENT',
        'IS_URBAN',
    ]
    
    # Intersectional groups for deeper analysis
    INTERSECTIONAL_GROUPS = [
        ('IS_FEMALE', 'IS_IMMIGRANT'),
        ('IS_FEMALE', 'HAS_YOUNG_CHILDREN'),
        ('IS_FEMALE', 'IS_PUBLIC_SECTOR'),
        ('IS_FEMALE', 'HAS_DEGREE'),
        ('IS_FEMALE', 'IS_URBAN'),
        ('IS_IMMIGRANT', 'HAS_DEGREE'),
        ('IS_FEMALE', 'IS_LONE_PARENT'),
    ]
    
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
                          gender_col: str = None,
                          weight_col: str = 'FINALWT') -> Dict:
        """
        Comprehensive wage gap analysis WITH SURVEY WEIGHTS.
        
        Supports both GENDER (standard) and SEX (legacy) column names.
        All statistics are weighted for population-level inference.
        
        Args:
            df: DataFrame with wage data and survey weights
            y_pred: Optional model predictions to analyze bias amplification
            wage_col: Column containing wages
            gender_col: Column containing gender codes
            weight_col: Column containing survey weights (FINALWT)
            
        Returns:
            Dictionary with weighted gap analysis results
        """
        # Normalize column names
        df = normalize_column_names(df.copy())
        
        # Determine columns
        wage_col = wage_col or COLS.HOURLY_EARNINGS
        
        # Gender column: prefer GENDER, fall back to SEX
        if gender_col is None:
            gender_col = COLS.GENDER if COLS.GENDER in df.columns else 'SEX'
        
        # Validate weight column
        if weight_col not in df.columns:
            warnings.warn(f"Weight column '{weight_col}' not found. Using unweighted analysis.")
            df[weight_col] = 1.0
        
        # Actual wage gap - WEIGHTED
        male_mask = df[gender_col] == 1
        female_mask = df[gender_col] == 2
        
        # Weighted means
        male_wage = np.average(
            df.loc[male_mask, wage_col],
            weights=df.loc[male_mask, weight_col]
        )
        female_wage = np.average(
            df.loc[female_mask, wage_col],
            weights=df.loc[female_mask, weight_col]
        )
        
        # Weighted population counts
        male_pop = df.loc[male_mask, weight_col].sum()
        female_pop = df.loc[female_mask, weight_col].sum()
        
        raw_gap = male_wage - female_wage
        raw_gap_pct = (raw_gap / male_wage) * 100
        
        results = {
            'actual': {
                'male_mean': float(male_wage),
                'female_mean': float(female_wage),
                'raw_gap': float(raw_gap),
                'raw_gap_pct': float(raw_gap_pct),
                'female_to_male_ratio': float(female_wage / male_wage),
            },
            'sample_sizes': {
                'male': int(male_mask.sum()),
                'female': int(female_mask.sum()),
            },
            'weighted_population': {
                'male': float(male_pop),
                'female': float(female_pop),
            },
            'weighted': True,  # Flag that results use survey weights
        }
        
        # Predicted wage gap (if predictions provided)
        if y_pred is not None:
            # Weighted prediction means
            pred_male = np.average(
                y_pred[male_mask],
                weights=df.loc[male_mask, weight_col].values
            )
            pred_female = np.average(
                y_pred[female_mask],
                weights=df.loc[female_mask, weight_col].values
            )
            pred_gap = pred_male - pred_female
            pred_gap_pct = (pred_gap / pred_male) * 100
            
            results['predicted'] = {
                'male_mean': float(pred_male),
                'female_mean': float(pred_female),
                'raw_gap': float(pred_gap),
                'raw_gap_pct': float(pred_gap_pct),
                'female_to_male_ratio': float(pred_female / pred_male),
            }
            
            # Bias amplification check
            results['bias_analysis'] = {
                'gap_amplification': float(pred_gap_pct - raw_gap_pct),
                'is_amplifying': pred_gap_pct > raw_gap_pct,
                'recommendation': self._get_bias_recommendation(raw_gap_pct, pred_gap_pct)
            }
        
        # Statistical significance (weighted t-test approximation)
        # Use weighted variance for proper inference
        from scipy import stats
        
        male_wages = df.loc[male_mask, wage_col].values
        female_wages = df.loc[female_mask, wage_col].values
        male_weights = df.loc[male_mask, weight_col].values
        female_weights = df.loc[female_mask, weight_col].values
        
        # Weighted variance
        male_var = np.average((male_wages - male_wage)**2, weights=male_weights)
        female_var = np.average((female_wages - female_wage)**2, weights=female_weights)
        
        # Effective sample sizes (for weighted data)
        n_eff_male = male_weights.sum()**2 / (male_weights**2).sum()
        n_eff_female = female_weights.sum()**2 / (female_weights**2).sum()
        
        # Weighted t-statistic
        se_diff = np.sqrt(male_var / n_eff_male + female_var / n_eff_female)
        t_stat = raw_gap / se_diff if se_diff > 0 else 0
        
        # Approximate degrees of freedom (Welch-Satterthwaite)
        df_approx = (male_var/n_eff_male + female_var/n_eff_female)**2 / (
            (male_var/n_eff_male)**2/(n_eff_male-1) + 
            (female_var/n_eff_female)**2/(n_eff_female-1)
        )
        
        # Two-tailed p-value
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df_approx))
        
        results['statistical_test'] = {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant_at_05': p_value < 0.05,
            'significant_at_01': p_value < 0.01,
            'effective_n_male': float(n_eff_male),
            'effective_n_female': float(n_eff_female),
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
                                   sensitive_features: pd.Series,
                                   sample_weight: Optional[np.ndarray] = None) -> Dict:
        """
        Compute comprehensive fairness metrics with optional survey weights.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            sensitive_features: Protected attribute (e.g., gender)
            sample_weight: Survey weights (FINALWT) for population-level metrics
        """
        # Use weighted median for threshold if weights provided
        if sample_weight is not None:
            # Weighted median
            sorted_idx = np.argsort(y_true)
            sorted_y = y_true[sorted_idx]
            sorted_w = sample_weight[sorted_idx]
            cumsum = np.cumsum(sorted_w)
            median_idx = np.searchsorted(cumsum, cumsum[-1] / 2)
            median_wage = sorted_y[median_idx]
        else:
            median_wage = np.median(y_true)
        
        # Convert regression to binary outcome for some metrics
        # (e.g., "high wage" vs "low wage")
        y_true_binary = (y_true > median_wage).astype(int)
        y_pred_binary = (y_pred > median_wage).astype(int)
        
        results = {
            'threshold': float(median_wage),
            'weighted': sample_weight is not None,
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
                                 wage_col: str = 'HRLYEARN') -> Dict:
        """
        Analyze wage gaps at intersections of protected attributes.
        
        Examines compounded disadvantages for individuals belonging to
        multiple marginalized groups.
        """
        # Create intersectional groups
        df = df.copy()
        df['predicted'] = y_pred
        
        # Determine gender column (prefer GENDER, fall back to SEX)
        gender_col = COLS.GENDER if COLS.GENDER in df.columns else 'SEX'
        
        results = {}
        
        # === Core Intersections ===
        
        # Gender x Education
        educ_col = COLS.EDUCATION if COLS.EDUCATION in df.columns else 'EDUC'
        if gender_col in df.columns and educ_col in df.columns:
            gender_educ = df.groupby([gender_col, educ_col]).agg({
                wage_col: ['mean', 'median', 'count'],
                'predicted': ['mean', 'median'],
            }).round(2)
            gender_educ.columns = ['actual_mean', 'actual_median', 'count',
                                   'pred_mean', 'pred_median']
            results['gender_education'] = gender_educ
        
        # Gender x Occupation
        occ_col = COLS.OCCUPATION_10 if COLS.OCCUPATION_10 in df.columns else 'NOC_10'
        if gender_col in df.columns and occ_col in df.columns:
            gender_occ = df.groupby([gender_col, occ_col]).agg({
                wage_col: ['mean', 'count'],
            }).round(2)
            gender_occ.columns = ['actual_mean', 'count']
            results['gender_occupation'] = gender_occ
        
        # Gender x Full-time status
        ftpt_col = COLS.FULLTIME_PARTTIME if COLS.FULLTIME_PARTTIME in df.columns else 'FTPTMAIN'
        if gender_col in df.columns and ftpt_col in df.columns:
            gender_ft = df.groupby([gender_col, ftpt_col]).agg({
                wage_col: ['mean', 'count'],
            }).round(2)
            gender_ft.columns = ['actual_mean', 'count']
            results['gender_fulltime'] = gender_ft
        
        # === Extended Intersections (New Features) ===
        
        # Gender x Immigration
        if 'IS_FEMALE' in df.columns and 'IS_IMMIGRANT' in df.columns:
            gender_immig = df.groupby(['IS_FEMALE', 'IS_IMMIGRANT']).agg({
                wage_col: ['mean', 'median', 'count'],
                'predicted': ['mean'],
            }).round(2)
            gender_immig.columns = ['actual_mean', 'actual_median', 'count', 'pred_mean']
            results['gender_immigration'] = gender_immig
            
            # Calculate double disadvantage
            try:
                baseline = df[(df['IS_FEMALE'] == 0) & (df['IS_IMMIGRANT'] == 0)][wage_col].mean()
                double = df[(df['IS_FEMALE'] == 1) & (df['IS_IMMIGRANT'] == 1)][wage_col].mean()
                results['immigrant_women_gap'] = {
                    'baseline_wage': float(baseline),
                    'immigrant_women_wage': float(double),
                    'gap_pct': float((baseline - double) / baseline * 100) if baseline > 0 else 0,
                }
            except:
                pass
        
        # Gender x Parenthood (Motherhood Penalty)
        if 'IS_FEMALE' in df.columns and 'HAS_YOUNG_CHILDREN' in df.columns:
            gender_parent = df.groupby(['IS_FEMALE', 'HAS_YOUNG_CHILDREN']).agg({
                wage_col: ['mean', 'count'],
            }).round(2)
            gender_parent.columns = ['actual_mean', 'count']
            results['gender_parenthood'] = gender_parent
            
            # Motherhood vs Fatherhood effect
            try:
                mothers_with = df[(df['IS_FEMALE'] == 1) & (df['HAS_YOUNG_CHILDREN'] == 1)][wage_col].mean()
                mothers_without = df[(df['IS_FEMALE'] == 1) & (df['HAS_YOUNG_CHILDREN'] == 0)][wage_col].mean()
                fathers_with = df[(df['IS_FEMALE'] == 0) & (df['HAS_YOUNG_CHILDREN'] == 1)][wage_col].mean()
                fathers_without = df[(df['IS_FEMALE'] == 0) & (df['HAS_YOUNG_CHILDREN'] == 0)][wage_col].mean()
                
                motherhood_penalty = (mothers_without - mothers_with) / mothers_without * 100
                fatherhood_effect = (fathers_without - fathers_with) / fathers_without * 100
                
                results['parenthood_penalty'] = {
                    'motherhood_penalty_pct': float(motherhood_penalty),
                    'fatherhood_effect_pct': float(fatherhood_effect),
                    'gender_gap_in_effect': float(motherhood_penalty - fatherhood_effect),
                    'interpretation': (
                        f"Mothers face a {abs(motherhood_penalty):.1f}% {'penalty' if motherhood_penalty > 0 else 'premium'}, "
                        f"fathers face a {abs(fatherhood_effect):.1f}% {'penalty' if fatherhood_effect > 0 else 'premium'}"
                    ),
                }
            except:
                pass
        
        # Gender x Public/Private Sector
        if 'IS_FEMALE' in df.columns and 'IS_PUBLIC_SECTOR' in df.columns:
            gender_sector = df.groupby(['IS_FEMALE', 'IS_PUBLIC_SECTOR']).agg({
                wage_col: ['mean', 'count'],
            }).round(2)
            gender_sector.columns = ['actual_mean', 'count']
            results['gender_sector'] = gender_sector
        
        # Gender x Urban/Rural
        if 'IS_FEMALE' in df.columns and 'IS_URBAN' in df.columns:
            gender_urban = df.groupby(['IS_FEMALE', 'IS_URBAN']).agg({
                wage_col: ['mean', 'count'],
            }).round(2)
            gender_urban.columns = ['actual_mean', 'count']
            results['gender_urban'] = gender_urban
        
        # Immigration x Education (Credential Recognition)
        if 'IS_IMMIGRANT' in df.columns and 'HAS_DEGREE' in df.columns:
            immig_educ = df.groupby(['IS_IMMIGRANT', 'HAS_DEGREE']).agg({
                wage_col: ['mean', 'count'],
            }).round(2)
            immig_educ.columns = ['actual_mean', 'count']
            results['immigration_education'] = immig_educ
            
            # Credential penalty
            try:
                non_imm_degree = df[(df['IS_IMMIGRANT'] == 0) & (df['HAS_DEGREE'] == 1)][wage_col].mean()
                imm_degree = df[(df['IS_IMMIGRANT'] == 1) & (df['HAS_DEGREE'] == 1)][wage_col].mean()
                non_imm_no_degree = df[(df['IS_IMMIGRANT'] == 0) & (df['HAS_DEGREE'] == 0)][wage_col].mean()
                imm_no_degree = df[(df['IS_IMMIGRANT'] == 1) & (df['HAS_DEGREE'] == 0)][wage_col].mean()
                
                gap_with_degree = (non_imm_degree - imm_degree) / non_imm_degree * 100
                gap_without_degree = (non_imm_no_degree - imm_no_degree) / non_imm_no_degree * 100
                
                results['credential_recognition'] = {
                    'immigrant_gap_with_degree': float(gap_with_degree),
                    'immigrant_gap_without_degree': float(gap_without_degree),
                    'credential_penalty': float(gap_with_degree - gap_without_degree),
                    'interpretation': (
                        f"Immigrant gap is {abs(gap_with_degree - gap_without_degree):.1f}pp "
                        f"{'larger' if gap_with_degree > gap_without_degree else 'smaller'} for degree holders"
                    ),
                }
            except:
                pass
        
        # Lone Parent x Gender
        if 'IS_FEMALE' in df.columns and 'IS_LONE_PARENT' in df.columns:
            gender_lone = df.groupby(['IS_FEMALE', 'IS_LONE_PARENT']).agg({
                wage_col: ['mean', 'count'],
            }).round(2)
            gender_lone.columns = ['actual_mean', 'count']
            results['gender_lone_parent'] = gender_lone
        
        self.metrics['intersectional'] = results
        return results


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
