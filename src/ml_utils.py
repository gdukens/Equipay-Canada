"""
EquiPay Canada - Machine Learning Utilities
============================================

Provides weighted train/test/validation splitting and evaluation utilities
for survey-weighted data (LFS with FINALWT).

This module ensures that:
1. Survey weights are preserved through all ML operations
2. Splits are stratified by key demographic variables
3. Evaluation metrics are properly weighted
4. Cross-validation respects the survey design

Usage:
    from src.ml_utils import WeightedMLSplitter, WeightedMetrics
    
    splitter = WeightedMLSplitter(df, target_col='HRLYEARN', weight_col='FINALWT')
    splits = splitter.create_splits(test_size=0.2, val_size=0.1)
    
    # Train with weights
    model.fit(splits['train']['X'], splits['train']['y'], 
              sample_weight=splits['train']['weights'])
    
    # Evaluate with weights
    metrics = WeightedMetrics.evaluate(
        y_true=splits['test']['y'],
        y_pred=model.predict(splits['test']['X']),
        weights=splits['test']['weights']
    )
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

from .constants import COLS

logger = logging.getLogger(__name__)


class WeightedMLSplitter:
    """
    Creates train/validation/test splits that preserve survey weights.
    
    Key Features:
    - Stratified splitting by key demographic variables
    - Weights travel with observations through all splits
    - Supports temporal splits for time series validation
    - Preserves intersectional group representation
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str = None,
        weight_col: str = 'FINALWT',
        feature_cols: Optional[List[str]] = None,
        stratify_cols: Optional[List[str]] = None,
        random_state: int = 42
    ):
        """
        Initialize the splitter.
        
        Args:
            df: DataFrame with features, target, and weights
            target_col: Name of target column (defaults to HRLYEARN)
            weight_col: Name of weight column (defaults to FINALWT)
            feature_cols: List of feature columns. If None, auto-detected.
            stratify_cols: Columns to stratify by. Defaults to ['SEX']
            random_state: Random seed for reproducibility
        """
        self.df = df.copy()
        self.target_col = target_col or COLS.HOURLY_EARNINGS
        self.weight_col = weight_col
        self.random_state = random_state
        
        # Default stratification by gender
        self.stratify_cols = stratify_cols or ['SEX']
        
        # Validate required columns
        self._validate_columns()
        
        # Auto-detect feature columns if not specified
        self.feature_cols = feature_cols or self._detect_feature_cols()
        
        logger.info(f"WeightedMLSplitter initialized with {len(self.df)} samples, "
                   f"{len(self.feature_cols)} features")
    
    def _validate_columns(self):
        """Validate that required columns exist."""
        if self.target_col not in self.df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in DataFrame")
        if self.weight_col not in self.df.columns:
            raise ValueError(f"Weight column '{self.weight_col}' not found in DataFrame. "
                           "Survey weights (FINALWT) are MANDATORY for this project.")
        
        for col in self.stratify_cols:
            if col not in self.df.columns:
                logger.warning(f"Stratification column '{col}' not found, will be skipped")
    
    def _detect_feature_cols(self) -> List[str]:
        """Auto-detect feature columns by excluding target, weight, and ID columns."""
        exclude_cols = {
            self.target_col, self.weight_col,
            'RECID', 'SURVMNTH', 'SURVYEAR', 'REC_NUM',  # ID columns
            'LFSSTAT',  # Target-related
        }
        
        feature_cols = [
            col for col in self.df.columns 
            if col not in exclude_cols 
            and self.df[col].dtype in ['int64', 'float64', 'int32', 'float32']
        ]
        
        return feature_cols
    
    def create_splits(
        self,
        test_size: float = 0.2,
        val_size: float = 0.1,
        temporal_split: bool = False,
        temporal_col: str = 'SURVYEAR',
        temporal_test_years: Optional[List[int]] = None
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Create train/validation/test splits with weights.
        
        Args:
            test_size: Fraction for test set (default 0.2)
            val_size: Fraction for validation set (default 0.1)
            temporal_split: If True, use time-based splitting
            temporal_col: Column for temporal splitting
            temporal_test_years: Years to use for test set in temporal split
            
        Returns:
            Dictionary with 'train', 'val', 'test' keys, each containing:
                - 'X': Feature array
                - 'y': Target array
                - 'weights': Sample weights
                - 'indices': Original DataFrame indices
        """
        if temporal_split:
            return self._temporal_split(temporal_col, temporal_test_years, val_size)
        else:
            return self._stratified_split(test_size, val_size)
    
    def _stratified_split(
        self,
        test_size: float,
        val_size: float
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Create stratified random splits."""
        
        # Prepare data
        X = self.df[self.feature_cols].values
        y = self.df[self.target_col].values
        weights = self.df[self.weight_col].values
        indices = self.df.index.values
        
        # Create stratification variable (combine if multiple)
        strat_cols = [c for c in self.stratify_cols if c in self.df.columns]
        if strat_cols:
            stratify = self.df[strat_cols[0]].values
            # For multiple columns, create composite key
            if len(strat_cols) > 1:
                stratify = pd.factorize(
                    self.df[strat_cols].apply(lambda x: '_'.join(x.astype(str)), axis=1)
                )[0]
        else:
            stratify = None
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test, w_temp, w_test, idx_temp, idx_test = \
            train_test_split(
                X, y, weights, indices,
                test_size=test_size,
                stratify=stratify,
                random_state=self.random_state
            )
        
        # Update stratify for remaining data
        if stratify is not None:
            temp_indices = np.isin(self.df.index, idx_temp)
            stratify_temp = stratify[temp_indices]
        else:
            stratify_temp = None
        
        # Second split: separate validation from training
        val_ratio = val_size / (1 - test_size)
        
        X_train, X_val, y_train, y_val, w_train, w_val, idx_train, idx_val = \
            train_test_split(
                X_temp, y_temp, w_temp, idx_temp,
                test_size=val_ratio,
                stratify=stratify_temp,
                random_state=self.random_state
            )
        
        # Log split statistics
        total_weight = weights.sum()
        logger.info(f"Split statistics:")
        logger.info(f"  Train: {len(y_train)} samples ({w_train.sum()/total_weight*100:.1f}% of weighted pop)")
        logger.info(f"  Val:   {len(y_val)} samples ({w_val.sum()/total_weight*100:.1f}% of weighted pop)")
        logger.info(f"  Test:  {len(y_test)} samples ({w_test.sum()/total_weight*100:.1f}% of weighted pop)")
        
        return {
            'train': {'X': X_train, 'y': y_train, 'weights': w_train, 'indices': idx_train},
            'val': {'X': X_val, 'y': y_val, 'weights': w_val, 'indices': idx_val},
            'test': {'X': X_test, 'y': y_test, 'weights': w_test, 'indices': idx_test},
            'feature_names': self.feature_cols,
        }
    
    def _temporal_split(
        self,
        temporal_col: str,
        test_years: Optional[List[int]],
        val_size: float
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Create time-based splits (recommended for forecasting)."""
        
        if temporal_col not in self.df.columns:
            raise ValueError(f"Temporal column '{temporal_col}' not found")
        
        years = sorted(self.df[temporal_col].unique())
        
        # Default: use most recent 2 years for test, 1 for validation
        if test_years is None:
            test_years = years[-2:]
            val_years = [years[-3]] if len(years) > 2 else []
        else:
            # Use year before test years for validation
            min_test_year = min(test_years)
            val_years = [min_test_year - 1] if min_test_year - 1 in years else []
        
        train_years = [y for y in years if y not in test_years and y not in val_years]
        
        logger.info(f"Temporal split: Train={train_years}, Val={val_years}, Test={test_years}")
        
        # Create masks
        train_mask = self.df[temporal_col].isin(train_years)
        val_mask = self.df[temporal_col].isin(val_years)
        test_mask = self.df[temporal_col].isin(test_years)
        
        def extract_split(mask):
            return {
                'X': self.df.loc[mask, self.feature_cols].values,
                'y': self.df.loc[mask, self.target_col].values,
                'weights': self.df.loc[mask, self.weight_col].values,
                'indices': self.df.index[mask].values,
            }
        
        result = {
            'train': extract_split(train_mask),
            'test': extract_split(test_mask),
            'feature_names': self.feature_cols,
        }
        
        if val_years:
            result['val'] = extract_split(val_mask)
        else:
            # Use stratified split of training data for validation
            train_data = result['train']
            X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
                train_data['X'], train_data['y'], train_data['weights'],
                test_size=val_size,
                random_state=self.random_state
            )
            result['train'] = {'X': X_train, 'y': y_train, 'weights': w_train}
            result['val'] = {'X': X_val, 'y': y_val, 'weights': w_val}
        
        return result
    
    def create_cv_folds(
        self,
        n_folds: int = 5,
        stratify: bool = True
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create cross-validation folds that preserve stratification.
        
        Returns list of (train_indices, val_indices) tuples.
        """
        X = self.df[self.feature_cols].values
        
        if stratify:
            strat_col = self.stratify_cols[0] if self.stratify_cols else None
            if strat_col and strat_col in self.df.columns:
                stratify_y = self.df[strat_col].values
            else:
                stratify_y = None
        else:
            stratify_y = None
        
        if stratify_y is not None:
            kfold = StratifiedKFold(
                n_splits=n_folds,
                shuffle=True,
                random_state=self.random_state
            )
            return list(kfold.split(X, stratify_y))
        else:
            from sklearn.model_selection import KFold
            kfold = KFold(
                n_splits=n_folds,
                shuffle=True,
                random_state=self.random_state
            )
            return list(kfold.split(X))
    
    def get_fold_data(
        self,
        fold_indices: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[Dict, Dict]:
        """
        Get train/val data for a specific fold.
        
        Returns:
            (train_dict, val_dict) each with X, y, weights keys
        """
        train_idx, val_idx = fold_indices
        
        train = {
            'X': self.df.iloc[train_idx][self.feature_cols].values,
            'y': self.df.iloc[train_idx][self.target_col].values,
            'weights': self.df.iloc[train_idx][self.weight_col].values,
        }
        
        val = {
            'X': self.df.iloc[val_idx][self.feature_cols].values,
            'y': self.df.iloc[val_idx][self.target_col].values,
            'weights': self.df.iloc[val_idx][self.weight_col].values,
        }
        
        return train, val


class WeightedMetrics:
    """
    Compute weighted evaluation metrics for survey-weighted predictions.
    
    All metrics properly incorporate survey weights for population-level inference.
    """
    
    @staticmethod
    def weighted_mse(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        weights: np.ndarray
    ) -> float:
        """Weighted Mean Squared Error."""
        return np.average((y_true - y_pred) ** 2, weights=weights)
    
    @staticmethod
    def weighted_rmse(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        weights: np.ndarray
    ) -> float:
        """Weighted Root Mean Squared Error."""
        return np.sqrt(WeightedMetrics.weighted_mse(y_true, y_pred, weights))
    
    @staticmethod
    def weighted_mae(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        weights: np.ndarray
    ) -> float:
        """Weighted Mean Absolute Error."""
        return np.average(np.abs(y_true - y_pred), weights=weights)
    
    @staticmethod
    def weighted_r2(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        weights: np.ndarray
    ) -> float:
        """
        Weighted R-squared (coefficient of determination).
        
        R² = 1 - SS_res / SS_tot
        where both are weighted.
        """
        y_mean = np.average(y_true, weights=weights)
        ss_res = np.average((y_true - y_pred) ** 2, weights=weights)
        ss_tot = np.average((y_true - y_mean) ** 2, weights=weights)
        
        if ss_tot == 0:
            return 0.0
        
        return 1 - (ss_res / ss_tot)
    
    @staticmethod
    def weighted_mape(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        weights: np.ndarray,
        epsilon: float = 1e-8
    ) -> float:
        """Weighted Mean Absolute Percentage Error."""
        return np.average(
            np.abs((y_true - y_pred) / (y_true + epsilon)),
            weights=weights
        ) * 100
    
    @staticmethod
    def weighted_median_ae(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        weights: np.ndarray
    ) -> float:
        """Weighted Median Absolute Error."""
        errors = np.abs(y_true - y_pred)
        # Sort by errors and find weighted median
        sorted_idx = np.argsort(errors)
        sorted_errors = errors[sorted_idx]
        sorted_weights = weights[sorted_idx]
        cumsum = np.cumsum(sorted_weights)
        median_idx = np.searchsorted(cumsum, cumsum[-1] / 2)
        return sorted_errors[median_idx]
    
    @classmethod
    def evaluate(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        weights: np.ndarray,
        include_unweighted: bool = True
    ) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            weights: Sample weights (FINALWT)
            include_unweighted: Also compute unweighted metrics for comparison
            
        Returns:
            Dictionary of metric names to values
        """
        results = {
            # Weighted metrics (population-level)
            'weighted_rmse': cls.weighted_rmse(y_true, y_pred, weights),
            'weighted_mae': cls.weighted_mae(y_true, y_pred, weights),
            'weighted_r2': cls.weighted_r2(y_true, y_pred, weights),
            'weighted_mape': cls.weighted_mape(y_true, y_pred, weights),
            'weighted_median_ae': cls.weighted_median_ae(y_true, y_pred, weights),
        }
        
        if include_unweighted:
            # Unweighted metrics (sample-level) for comparison
            results.update({
                'unweighted_rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'unweighted_mae': mean_absolute_error(y_true, y_pred),
                'unweighted_r2': r2_score(y_true, y_pred),
            })
        
        return results
    
    @classmethod
    def evaluate_by_group(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        weights: np.ndarray,
        groups: np.ndarray,
        group_names: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Compute metrics by demographic group.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            weights: Sample weights
            groups: Group membership array (e.g., gender codes)
            group_names: Optional mapping from codes to names
            
        Returns:
            DataFrame with metrics by group
        """
        results = []
        
        for group in np.unique(groups):
            mask = groups == group
            group_name = group_names.get(group, str(group)) if group_names else str(group)
            
            group_metrics = cls.evaluate(
                y_true[mask], y_pred[mask], weights[mask],
                include_unweighted=False
            )
            group_metrics['group'] = group_name
            group_metrics['n_samples'] = mask.sum()
            group_metrics['weighted_n'] = weights[mask].sum()
            results.append(group_metrics)
        
        return pd.DataFrame(results).set_index('group')


class WeightedGapAnalysis:
    """
    Analyze prediction gaps across demographic groups using survey weights.
    """
    
    @staticmethod
    def compute_gap(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        weights: np.ndarray,
        groups: np.ndarray,
        reference_group: Any = 1  # Default: male (code 1)
    ) -> Dict[str, float]:
        """
        Compute weighted wage gap relative to reference group.
        
        Returns gap statistics for actual values and predictions.
        """
        ref_mask = groups == reference_group
        other_mask = ~ref_mask
        
        # Weighted means
        ref_actual = np.average(y_true[ref_mask], weights=weights[ref_mask])
        other_actual = np.average(y_true[other_mask], weights=weights[other_mask])
        
        ref_pred = np.average(y_pred[ref_mask], weights=weights[ref_mask])
        other_pred = np.average(y_pred[other_mask], weights=weights[other_mask])
        
        actual_gap = ref_actual - other_actual
        actual_gap_pct = (actual_gap / ref_actual) * 100
        
        pred_gap = ref_pred - other_pred
        pred_gap_pct = (pred_gap / ref_pred) * 100
        
        return {
            'reference_group': reference_group,
            'actual_gap': actual_gap,
            'actual_gap_pct': actual_gap_pct,
            'predicted_gap': pred_gap,
            'predicted_gap_pct': pred_gap_pct,
            'gap_amplification': pred_gap_pct - actual_gap_pct,
            'reference_actual_mean': ref_actual,
            'reference_pred_mean': ref_pred,
            'other_actual_mean': other_actual,
            'other_pred_mean': other_pred,
        }
    
    @staticmethod
    def check_bias_amplification(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        weights: np.ndarray,
        groups: np.ndarray,
        tolerance: float = 2.0  # percentage points
    ) -> Dict[str, Any]:
        """
        Check if model amplifies or reduces existing biases.
        
        Args:
            tolerance: Acceptable deviation in percentage points
            
        Returns:
            Analysis results with recommendations
        """
        gap_analysis = WeightedGapAnalysis.compute_gap(
            y_true, y_pred, weights, groups
        )
        
        amplification = gap_analysis['gap_amplification']
        
        if amplification > tolerance:
            status = 'AMPLIFYING'
            recommendation = (
                f"WARNING: Model amplifies the wage gap by {amplification:.1f} percentage points. "
                "Consider: 1) Fairness constraints during training, 2) Post-processing calibration, "
                "3) Removing or reweighting biased features."
            )
        elif amplification < -tolerance:
            status = 'REDUCING'
            recommendation = (
                f"Model reduces the wage gap by {-amplification:.1f} percentage points. "
                "While this may seem positive, verify this reflects legitimate factors "
                "and not over-correction that could lead to reverse discrimination."
            )
        else:
            status = 'NEUTRAL'
            recommendation = (
                "Model maintains similar gap to observed data. "
                "The model is neither amplifying nor significantly reducing existing disparities."
            )
        
        return {
            **gap_analysis,
            'status': status,
            'recommendation': recommendation,
        }


def prepare_weighted_training_data(
    df: pd.DataFrame,
    target_col: str = None,
    weight_col: str = 'FINALWT',
    feature_cols: Optional[List[str]] = None,
    test_size: float = 0.2,
    val_size: float = 0.1,
    stratify_by: str = 'SEX',
    random_state: int = 42
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Convenience function to prepare data for weighted ML training.
    
    This is the RECOMMENDED entry point for preparing LFS data for model training.
    
    Args:
        df: DataFrame with LFS data (must include FINALWT)
        target_col: Target column name (default: HRLYEARN)
        weight_col: Weight column name (default: FINALWT)
        feature_cols: Feature columns. If None, auto-detected.
        test_size: Fraction for test set
        val_size: Fraction for validation set
        stratify_by: Column to stratify splits by
        random_state: Random seed
        
    Returns:
        Dictionary with train/val/test splits, each containing X, y, weights
        
    Example:
        >>> splits = prepare_weighted_training_data(df)
        >>> model.fit(splits['train']['X'], splits['train']['y'],
        ...           sample_weight=splits['train']['weights'])
        >>> metrics = WeightedMetrics.evaluate(
        ...     splits['test']['y'],
        ...     model.predict(splits['test']['X']),
        ...     splits['test']['weights']
        ... )
    """
    splitter = WeightedMLSplitter(
        df=df,
        target_col=target_col,
        weight_col=weight_col,
        feature_cols=feature_cols,
        stratify_cols=[stratify_by] if stratify_by else None,
        random_state=random_state
    )
    
    return splitter.create_splits(test_size=test_size, val_size=val_size)
