"""
EquiPay Canada - Data Leakage Prevention Module
================================================

This module provides centralized safeguards against data leakage at all levels
of the analysis pipeline. It implements multiple layers of protection:

1. TARGET-DERIVED FEATURE DETECTION
   - Identifies features computed from the target variable
   - Blocks inclusion of wage-derived features in prediction

2. TEMPORAL LEAKAGE PREVENTION  
   - Ensures future data doesn't leak into training
   - Enforces chronological splitting for time series

3. CORRELATION-BASED DETECTION
   - Flags features suspiciously correlated with target
   - Provides warnings and automatic exclusion

4. PIPELINE VALIDATION
   - Validates entire ML pipelines for leakage
   - Sanity checks on model performance (R² bounds)

Usage:
------
    from src.leakage_prevention import LeakageGuard, validate_features
    
    guard = LeakageGuard(target_col='REAL_HRLYEARN')
    clean_features = guard.filter_features(df, feature_list)
    
    # Or use decorator
    @guard.protect
    def train_model(X, y):
        ...

References:
-----------
- Kaufman et al. (2012) - "Leakage in Data Mining"
- Kapoor & Narayanan (2022) - "Leakage and the Reproducibility Crisis in ML"

Author: EquiPay Canada Research Team
Version: 1.0.0
"""

import logging
from typing import List, Dict, Set, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from functools import wraps
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class LeakageConfig:
    """Configuration for leakage prevention."""
    
    # Correlation threshold for suspicious features
    correlation_threshold: float = 0.90
    
    # Maximum acceptable R² for wage prediction (sanity check)
    max_realistic_r2: float = 0.60
    
    # Warning threshold (lower than block threshold)
    warning_correlation_threshold: float = 0.70
    
    # Whether to raise exceptions or just warn
    strict_mode: bool = True
    
    # Log all checks
    verbose: bool = True


# =============================================================================
# KNOWN LEAKAGE SOURCES
# =============================================================================

# Features that are ALWAYS derived from the target (wages)
TARGET_DERIVED_FEATURES: Set[str] = {
    # Direct wage columns (the target itself)
    'HRLYEARN',
    'REAL_HRLYEARN', 
    'LOG_HRLYEARN',
    'LOG_REAL_HRLYEARN',
    'WAGE',
    'LOG_WAGE',
    'REAL_WAGE',
    'hourly_wage',
    'real_hourly_wage',
    
    # Wage zone indicators (computed from wage quantiles)
    'wage_zone_upper',
    'wage_zone_middle', 
    'wage_zone_lower',
    'wage_zone',
    
    # Wage position indicators
    'above_median',
    'below_median',
    'above_mean',
    'within_gender_percentile',
    'within_group_percentile',
    'wage_percentile',
    'wage_decile',
    'wage_quintile',
    
    # Distance from wage statistics
    'distance_from_gender_median',
    'distance_from_median',
    'distance_from_mean',
    'gap_from_mean',
    'gap_from_median',
    
    # Within-occupation wage features
    'within_occ_wage_gap',
    'occ_wage_residual',
    'wage_residual',
    
    # Ceiling/floor indicators based on wages
    'ceiling_proximity',
    'ceiling_proximity_scaled',
    'floor_proximity',
    
    # Any feature with 'wage' in the name (catch-all)
    # These are checked via pattern matching below
}

# Patterns that indicate potential leakage
LEAKAGE_PATTERNS: List[str] = [
    'wage',
    'earn',
    'salary',
    'income',
    'pay_',
    '_pay',
    'compensation',
    'percentile',
    'decile',
    'quintile',
    'quantile',
]

# Features that are SAFE (definitely not derived from target)
KNOWN_SAFE_FEATURES: Set[str] = {
    # Demographics
    'GENDER', 'IS_FEMALE', 'AGE_12', 'AGE_6', 'MARSTAT', 'IMMIG',
    'EFAMTYPE', 'AGYOWNK',
    
    # Human capital
    'EDUC', 'TENURE', 'EXPERIENCE_PROXY', 'EXPERIENCE', 'EXPERIENCE_SQ',
    'HAS_DEGREE', 'YEARS_EDUCATION',
    
    # Job characteristics
    'NOC_10', 'NOC_43', 'NAICS_21', 'COWMAIN', 'FTPTMAIN', 
    'PERMTEMP', 'UNION', 'ESTSIZE', 'FIRMSIZE', 'MJH',
    
    # Geographic
    'PROV', 'CMA', 'IS_URBAN', 'IS_MAJOR_CMA',
    
    # Hours (but NOT wage-derived)
    'UHRSMAIN', 'AHRSMAIN', 'UTOTHRS', 'ATOTHRS', 'WHYPT',
    
    # Time
    'SURVYEAR', 'SURVMNTH', 'YEAR', 'MONTH',
    
    # Weights
    'FINALWT',
    
    # Derived safe features
    'IS_PUBLIC', 'IS_PRIVATE', 'IS_SELF_EMPLOYED',
    'IS_PERMANENT', 'IS_TEMPORARY', 'IS_FULLTIME', 'IS_PARTTIME',
    'IS_UNION', 'IS_IMMIGRANT', 'IS_RECENT_IMMIG',
    'IS_MARRIED', 'HAS_YOUNG_CHILDREN',
    
    # Occupation/industry characteristics (group-level, not individual)
    'occ_female_share', 'occ_female_share_scaled',
    'female_dominated_occ', 'male_dominated_occ', 'integrated_occ',
    'occ_segregation_score', 'occ_segregation_score_scaled',
    'industry_female_share',
    
    # Propensity scores (if computed from safe features only)
    'propensity_female', 'ipw_female', 'ipw_female_scaled',
    
    # Interactions of safe features
    'FEMALE_x_EDUC', 'FEMALE_x_TENURE', 'FEMALE_x_AGE',
}


# =============================================================================
# MAIN LEAKAGE GUARD CLASS
# =============================================================================

class LeakageGuard:
    """
    Central class for preventing data leakage throughout the pipeline.
    
    This class provides multiple layers of protection:
    1. Explicit blocklist of known target-derived features
    2. Pattern-based detection of suspicious feature names
    3. Correlation-based detection of features too related to target
    4. Sanity checks on model performance
    
    Examples
    --------
    >>> guard = LeakageGuard(target_col='REAL_HRLYEARN')
    >>> 
    >>> # Check a feature list
    >>> clean_features = guard.filter_features(df, proposed_features)
    >>> 
    >>> # Validate entire pipeline
    >>> guard.validate_pipeline(X_train, y_train, X_test, y_test)
    >>> 
    >>> # Use as decorator
    >>> @guard.protect
    >>> def train_model(X, y):
    ...     return model.fit(X, y)
    """
    
    def __init__(
        self,
        target_col: str = 'REAL_HRLYEARN',
        config: LeakageConfig = None,
        additional_blocked: List[str] = None,
        additional_safe: List[str] = None
    ):
        """
        Initialize the LeakageGuard.
        
        Parameters
        ----------
        target_col : str
            Name of the target variable column
        config : LeakageConfig
            Configuration options
        additional_blocked : List[str]
            Additional features to block
        additional_safe : List[str]
            Additional features to allow
        """
        self.target_col = target_col
        self.config = config or LeakageConfig()
        
        # Build blocklist
        self.blocked_features = TARGET_DERIVED_FEATURES.copy()
        self.blocked_features.add(target_col)
        if additional_blocked:
            self.blocked_features.update(additional_blocked)
        
        # Build safelist
        self.safe_features = KNOWN_SAFE_FEATURES.copy()
        if additional_safe:
            self.safe_features.update(additional_safe)
        
        # Track detected leakage
        self.detected_leakage: List[Dict[str, Any]] = []
        
        if self.config.verbose:
            logger.info(f"LeakageGuard initialized for target: {target_col}")
            logger.info(f"  Blocked features: {len(self.blocked_features)}")
            logger.info(f"  Safe features: {len(self.safe_features)}")
    
    def is_blocked(self, feature: str) -> Tuple[bool, str]:
        """
        Check if a feature is blocked due to potential leakage.
        
        Returns
        -------
        Tuple[bool, str]
            (is_blocked, reason)
        """
        feature_lower = feature.lower()
        
        # Check explicit blocklist
        if feature in self.blocked_features:
            return True, f"Explicitly blocked (target-derived)"
        
        # Check if it's the target
        if feature == self.target_col:
            return True, "Is the target variable itself"
        
        # Check pattern matching
        for pattern in LEAKAGE_PATTERNS:
            if pattern in feature_lower and feature not in self.safe_features:
                return True, f"Matches leakage pattern: '{pattern}'"
        
        return False, ""
    
    def is_safe(self, feature: str) -> bool:
        """Check if a feature is known to be safe."""
        return feature in self.safe_features
    
    def filter_features(
        self,
        df: pd.DataFrame,
        feature_list: List[str],
        y: pd.Series = None,
        return_report: bool = False
    ) -> List[str]:
        """
        Filter a feature list to remove leaked features.
        
        Parameters
        ----------
        df : DataFrame
            Data containing the features
        feature_list : List[str]
            Proposed features to use
        y : Series, optional
            Target variable for correlation checking
        return_report : bool
            If True, return (features, report) tuple
            
        Returns
        -------
        List[str] or Tuple[List[str], Dict]
            Clean feature list (and optionally report)
        """
        clean_features = []
        blocked_features = []
        warned_features = []
        
        for feature in feature_list:
            is_blocked, reason = self.is_blocked(feature)
            
            if is_blocked:
                blocked_features.append({'feature': feature, 'reason': reason})
                self._log_leakage(feature, reason, 'blocked')
                continue
            
            # Correlation check if target provided
            if y is not None and feature in df.columns:
                try:
                    corr = abs(df[feature].corr(y))
                    
                    if corr > self.config.correlation_threshold:
                        reason = f"Suspiciously high correlation with target: r={corr:.3f}"
                        blocked_features.append({'feature': feature, 'reason': reason})
                        self._log_leakage(feature, reason, 'blocked')
                        continue
                    
                    elif corr > self.config.warning_correlation_threshold:
                        warned_features.append({
                            'feature': feature, 
                            'correlation': corr,
                            'reason': f"High correlation: r={corr:.3f}"
                        })
                        self._log_leakage(feature, f"High correlation: r={corr:.3f}", 'warning')
                        
                except Exception:
                    pass  # Skip correlation check if it fails
            
            clean_features.append(feature)
        
        # Log summary
        if self.config.verbose:
            if blocked_features:
                logger.warning(f"🔴 BLOCKED {len(blocked_features)} features due to leakage:")
                for item in blocked_features:
                    logger.warning(f"   - {item['feature']}: {item['reason']}")
            
            if warned_features:
                logger.warning(f"⚠️ WARNING on {len(warned_features)} features (high correlation):")
                for item in warned_features:
                    logger.warning(f"   - {item['feature']}: r={item['correlation']:.3f}")
            
            logger.info(f"✅ Approved {len(clean_features)} features")
        
        if return_report:
            report = {
                'approved': clean_features,
                'blocked': blocked_features,
                'warned': warned_features,
            }
            return clean_features, report
        
        return clean_features
    
    def validate_features(
        self,
        X: pd.DataFrame,
        y: pd.Series = None
    ) -> Dict[str, Any]:
        """
        Validate features and return a detailed report.
        
        This is similar to filter_features but returns a validation report
        instead of a filtered list.
        
        Parameters
        ----------
        X : DataFrame
            Feature matrix to validate
        y : Series, optional
            Target variable for correlation checking
            
        Returns
        -------
        Dict with keys:
            - is_valid: bool
            - flagged_features: List of dicts with feature info
            - safe_features: List of approved features
            - summary: String summary
        """
        flagged = []
        safe = []
        
        for col in X.columns:
            is_blocked, reason = self.is_blocked(col)
            
            if is_blocked:
                flagged.append({'feature': col, 'reason': reason, 'type': 'blocked'})
                continue
            
            # Correlation check
            if y is not None:
                try:
                    corr = abs(X[col].corr(y))
                    if corr > self.config.correlation_threshold:
                        flagged.append({
                            'feature': col,
                            'reason': f'High correlation: r={corr:.3f}',
                            'type': 'correlation',
                            'correlation': corr
                        })
                        continue
                except Exception:
                    pass
            
            safe.append(col)
        
        is_valid = len(flagged) == 0
        
        return {
            'is_valid': is_valid,
            'flagged_features': flagged,
            'safe_features': safe,
            'summary': f"{'✓ Valid' if is_valid else '⚠ Issues found'}: {len(safe)} safe, {len(flagged)} flagged"
        }
    
    def validate_pipeline(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model = None,
        predictions: np.ndarray = None
    ) -> Dict[str, Any]:
        """
        Validate an entire ML pipeline for leakage.
        
        Checks:
        1. No target-derived features in X
        2. No temporal leakage (if time column available)
        3. R² sanity check (not impossibly high)
        4. Feature correlation check
        
        Parameters
        ----------
        X_train, y_train : Training data
        X_test, y_test : Test data
        model : Fitted model (optional)
        predictions : Model predictions (optional)
        
        Returns
        -------
        Dict with validation results
        """
        results = {
            'passed': True,
            'checks': [],
            'warnings': [],
            'errors': []
        }
        
        # Check 1: No target-derived features
        features = list(X_train.columns) if hasattr(X_train, 'columns') else []
        for feature in features:
            is_blocked, reason = self.is_blocked(feature)
            if is_blocked:
                results['errors'].append(f"Leakage detected: {feature} - {reason}")
                results['passed'] = False
        
        if not results['errors']:
            results['checks'].append("✅ No target-derived features found")
        
        # Check 2: R² sanity check
        if predictions is not None:
            from sklearn.metrics import r2_score
            r2 = r2_score(y_test, predictions)
            
            if r2 > self.config.max_realistic_r2:
                results['errors'].append(
                    f"R² = {r2:.3f} exceeds realistic maximum ({self.config.max_realistic_r2}). "
                    "This strongly suggests data leakage."
                )
                results['passed'] = False
            else:
                results['checks'].append(f"✅ R² = {r2:.3f} within realistic bounds")
        
        # Check 3: Feature correlation with target
        if hasattr(X_train, 'columns'):
            for col in X_train.columns:
                try:
                    corr = abs(pd.Series(X_train[col]).corr(y_train))
                    if corr > self.config.correlation_threshold:
                        results['errors'].append(
                            f"Feature '{col}' has suspiciously high correlation "
                            f"with target (r={corr:.3f})"
                        )
                        results['passed'] = False
                except Exception:
                    pass
        
        # Log results
        if self.config.verbose:
            if results['passed']:
                logger.info("✅ Pipeline validation PASSED")
            else:
                logger.error("🔴 Pipeline validation FAILED:")
                for error in results['errors']:
                    logger.error(f"   {error}")
        
        return results
    
    def _log_leakage(self, feature: str, reason: str, level: str):
        """Log a detected leakage issue."""
        self.detected_leakage.append({
            'feature': feature,
            'reason': reason,
            'level': level,
        })
    
    def protect(self, func: Callable) -> Callable:
        """
        Decorator to protect a function from using leaked features.
        
        Usage:
            @guard.protect
            def train_model(X, y):
                ...
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check DataFrame arguments for leakage
            for arg in args:
                if isinstance(arg, pd.DataFrame):
                    self._check_dataframe(arg)
            
            for key, value in kwargs.items():
                if isinstance(value, pd.DataFrame):
                    self._check_dataframe(value)
            
            return func(*args, **kwargs)
        
        return wrapper
    
    def _check_dataframe(self, df: pd.DataFrame):
        """Check a DataFrame for leaked features."""
        for col in df.columns:
            is_blocked, reason = self.is_blocked(col)
            if is_blocked:
                msg = f"Leakage detected in DataFrame: column '{col}' - {reason}"
                if self.config.strict_mode:
                    raise ValueError(msg)
                else:
                    warnings.warn(msg)
    
    def get_report(self) -> pd.DataFrame:
        """Get a report of all detected leakage issues."""
        if not self.detected_leakage:
            return pd.DataFrame(columns=['feature', 'reason', 'level'])
        return pd.DataFrame(self.detected_leakage)
    
    def reset(self):
        """Reset the detected leakage log."""
        self.detected_leakage = []


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def validate_features(
    features: List[str],
    target_col: str = 'REAL_HRLYEARN',
    df: pd.DataFrame = None,
    y: pd.Series = None
) -> List[str]:
    """
    Quick validation of a feature list.
    
    Parameters
    ----------
    features : List[str]
        Proposed features
    target_col : str
        Target variable name
    df : DataFrame, optional
        Data for correlation checking
    y : Series, optional
        Target for correlation checking
        
    Returns
    -------
    List[str]
        Clean feature list
    """
    guard = LeakageGuard(target_col=target_col)
    
    if df is not None:
        return guard.filter_features(df, features, y)
    else:
        return [f for f in features if not guard.is_blocked(f)[0]]


def check_r2_sanity(r2: float, threshold: float = None, model_type: str = 'wage') -> Dict[str, Any]:
    """
    Check if R² is within realistic bounds for wage prediction.
    
    Economic theory and empirical evidence suggest that human capital
    and job characteristics explain at most 40-50% of wage variance.
    An R² above 0.60 strongly suggests data leakage.
    
    Parameters
    ----------
    r2 : float
        R-squared value to check
    threshold : float, optional
        Maximum acceptable R² (default based on model_type)
    model_type : str
        Type of model: 'wage' (default 0.60), 'classification' (0.95), etc.
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:
        - is_suspicious: bool
        - observed: float (the R² value)
        - threshold: float
        - message: str (interpretation)
    """
    # Set default threshold based on model type
    if threshold is None:
        thresholds = {
            'wage': 0.60,
            'classification': 0.95,
            'general': 0.80,
        }
        threshold = thresholds.get(model_type, 0.60)
    
    is_suspicious = r2 > threshold
    
    if is_suspicious:
        message = (
            f"R² = {r2:.4f} exceeds realistic maximum for {model_type} prediction ({threshold}). "
            "This strongly suggests data leakage."
        )
        warnings.warn(message)
    else:
        if r2 > 0.50:
            message = "Good model fit within realistic range."
        elif r2 > 0.30:
            message = "Moderate model fit - typical for wage models."
        else:
            message = "Lower R² - may need more features or nonlinear model."
    
    return {
        'is_suspicious': is_suspicious,
        'observed': r2,
        'threshold': threshold,
        'message': message
    }


def is_wage_derived(feature_name: str) -> bool:
    """
    Check if a feature name appears to be derived from wages.
    
    This is a fast check based on the feature name. For full validation,
    use LeakageGuard.is_blocked() which also checks patterns and safelist.
    
    Parameters
    ----------
    feature_name : str
        Name of the feature to check
        
    Returns
    -------
    bool
        True if the feature appears to be wage-derived
    """
    # Check explicit blocklist
    if feature_name in TARGET_DERIVED_FEATURES:
        return True
    
    # Check patterns
    feature_lower = feature_name.lower()
    for pattern in LEAKAGE_PATTERNS:
        if pattern in feature_lower and feature_name not in KNOWN_SAFE_FEATURES:
            return True
    
    return False


def get_safe_features(
    all_features: List[str],
    target_col: str = 'REAL_HRLYEARN'
) -> List[str]:
    """
    Get only the features known to be safe from leakage.
    
    Parameters
    ----------
    all_features : List[str]
        All available features
    target_col : str
        Target variable name
        
    Returns
    -------
    List[str]
        Features that are definitely safe
    """
    guard = LeakageGuard(target_col=target_col)
    return [f for f in all_features if guard.is_safe(f)]


# =============================================================================
# TEMPORAL LEAKAGE PREVENTION
# =============================================================================

class TemporalLeakageGuard:
    """
    Specialized guard for preventing temporal (time-based) leakage.
    
    Ensures that:
    1. Training data is from earlier periods than test data
    2. No future information leaks into historical predictions
    3. Time-based features are properly handled
    """
    
    def __init__(
        self,
        time_col: str = 'SURVYEAR',
        strict: bool = True
    ):
        self.time_col = time_col
        self.strict = strict
    
    def validate_split(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> bool:
        """
        Validate that train data is temporally before test data.
        
        Returns True if valid, raises ValueError if not (in strict mode).
        """
        if self.time_col not in train_df.columns:
            warnings.warn(f"Time column '{self.time_col}' not found. "
                         "Cannot validate temporal ordering.")
            return True
        
        train_max = train_df[self.time_col].max()
        test_min = test_df[self.time_col].min()
        
        if train_max >= test_min:
            msg = (f"Temporal leakage detected: Training data extends to {train_max} "
                   f"but test data starts at {test_min}. "
                   f"Training should use only data before test period.")
            
            if self.strict:
                raise ValueError(msg)
            else:
                warnings.warn(msg)
                return False
        
        return True
    
    def create_temporal_split(
        self,
        df: pd.DataFrame,
        test_periods: List[int],
        val_periods: List[int] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Create train/val/test split based on time periods.
        
        Parameters
        ----------
        df : DataFrame
            Full dataset
        test_periods : List[int]
            Time periods (years) for test set
        val_periods : List[int], optional
            Time periods for validation set
            
        Returns
        -------
        Dict with 'train', 'val', 'test' DataFrames
        """
        all_periods = sorted(df[self.time_col].unique())
        
        if val_periods is None:
            val_periods = []
        
        train_periods = [p for p in all_periods 
                        if p not in test_periods and p not in val_periods]
        
        result = {
            'train': df[df[self.time_col].isin(train_periods)].copy(),
            'test': df[df[self.time_col].isin(test_periods)].copy(),
        }
        
        if val_periods:
            result['val'] = df[df[self.time_col].isin(val_periods)].copy()
        
        logger.info(f"Temporal split created:")
        logger.info(f"  Train periods: {train_periods}")
        logger.info(f"  Val periods: {val_periods}")
        logger.info(f"  Test periods: {test_periods}")
        
        return result


# =============================================================================
# SINGLETON INSTANCE FOR EASY ACCESS
# =============================================================================

# Default guard instance - can be imported and used directly
default_guard = LeakageGuard()


def protect_pipeline(func: Callable) -> Callable:
    """
    Decorator using the default LeakageGuard.
    
    Usage:
        @protect_pipeline
        def train_model(X, y):
            ...
    """
    return default_guard.protect(func)
