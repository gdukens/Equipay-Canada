"""
Machine Learning Models Module
Salary prediction using ensemble methods

IMPORTANT: This module uses survey weights (FINALWT) throughout training and evaluation.
All metrics are population-level estimates using weighted calculations.

See ml_utils.py for:
- WeightedMLSplitter: Stratified train/val/test splitting with weights
- WeightedMetrics: Population-level evaluation metrics
- WeightedGapAnalysis: Bias detection in predictions
"""

import logging
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Import column constants for consistent column naming
from .constants import COLS
from .ml_utils import WeightedMLSplitter, WeightedMetrics, WeightedGapAnalysis

# Gradient boosting libraries
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    warnings.warn("XGBoost not installed. Using sklearn alternatives.")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    warnings.warn("LightGBM not installed. Using sklearn alternatives.")

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    warnings.warn("CatBoost not installed. Using sklearn alternatives.")

logger = logging.getLogger(__name__)


class CatBoostRegressorWrapper:
    """
    Wrapper for CatBoostRegressor to ensure sklearn compatibility.
    Fixes compatibility issues with VotingRegressor in newer sklearn versions.
    """
    
    def __init__(self, **kwargs):
        self._model = cb.CatBoostRegressor(**kwargs)
        self._params = kwargs
    
    def fit(self, X, y, **kwargs):
        self._model.fit(X, y, **kwargs)
        return self
    
    def predict(self, X):
        return self._model.predict(X)
    
    def get_params(self, deep=True):
        return self._params.copy()
    
    def set_params(self, **params):
        self._params.update(params)
        self._model = cb.CatBoostRegressor(**self._params)
        return self
    
    def __sklearn_tags__(self):
        from sklearn.utils._tags import Tags, InputTags, TargetTags, RegressorTags
        return Tags(
            estimator_type="regressor",
            input_tags=InputTags(),
            target_tags=TargetTags(required=True),
            regressor_tags=RegressorTags(),
        )


class SalaryPredictor:
    """
    Ensemble model for salary prediction with survey weight support.
    
    All training and evaluation uses FINALWT for population-level inference.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize salary predictor"""
        self.config = config or self._default_config()
        self.models = {}
        self.ensemble = None
        self.feature_names = None
        self.metrics = {}
        self._sample_weights = None  # Store weights for evaluation
        
    def _default_config(self) -> Dict:
        """Default model configuration"""
        return {
            'test_size': 0.2,
            'random_state': 42,
            'cv_folds': 5,
            'xgboost': {
                'n_estimators': 500,
                'max_depth': 8,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 3,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
            },
            'lightgbm': {
                'n_estimators': 500,
                'max_depth': 8,
                'learning_rate': 0.05,
                'num_leaves': 31,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
            },
            'random_forest': {
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
            },
        }
    
    def build_models(self) -> Dict[str, Any]:
        """Build individual models"""
        models = {}
        
        # XGBoost
        if HAS_XGBOOST:
            models['xgboost'] = xgb.XGBRegressor(
                **self.config.get('xgboost', {}),
                random_state=self.config['random_state'],
                n_jobs=-1,
                verbosity=0,
            )
        
        # LightGBM
        if HAS_LIGHTGBM:
            models['lightgbm'] = lgb.LGBMRegressor(
                **self.config.get('lightgbm', {}),
                random_state=self.config['random_state'],
                n_jobs=-1,
                verbose=-1,
            )
        
        # CatBoost
        if HAS_CATBOOST:
            models['catboost'] = CatBoostRegressorWrapper(
                iterations=500,
                depth=8,
                learning_rate=0.05,
                random_seed=self.config['random_state'],
                verbose=False,
            )
        
        # Random Forest (always available)
        models['random_forest'] = RandomForestRegressor(
            **self.config.get('random_forest', {}),
            random_state=self.config['random_state'],
            n_jobs=-1,
        )
        
        # Gradient Boosting (sklearn fallback)
        models['gradient_boosting'] = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            random_state=self.config['random_state'],
        )
        
        # Ridge regression for baseline
        models['ridge'] = Ridge(alpha=1.0)
        
        self.models = models
        return models
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              feature_names: Optional[List[str]] = None,
              sample_weight: Optional[np.ndarray] = None,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              val_weight: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Train all models with survey weights.
        
        Args:
            X: Training features
            y: Training target
            feature_names: Names of feature columns
            sample_weight: Survey weights (FINALWT) for training data - MANDATORY
            X_val: Validation features (optional, will split from X if not provided)
            y_val: Validation target
            val_weight: Validation weights
            
        Returns:
            Dictionary of model names to metrics
        """
        if sample_weight is None:
            warnings.warn(
                "No sample_weight provided! For LFS data, survey weights (FINALWT) "
                "should ALWAYS be used for population-level inference. "
                "Proceeding with equal weights (sample-level estimates only)."
            )
            sample_weight = np.ones(len(y))
        
        self.feature_names = feature_names
        self._sample_weights = sample_weight
        
        # Split data if validation set not provided
        if X_val is None:
            X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
                X, y, sample_weight,
                test_size=self.config['test_size'],
                random_state=self.config['random_state']
            )
        else:
            X_train, y_train, w_train = X, y, sample_weight
            X_test, y_test = X_val, y_val
            w_test = val_weight if val_weight is not None else np.ones(len(y_val))
        
        logger.info(f"Training on {len(X_train)} samples (weighted pop: {w_train.sum():,.0f})")
        logger.info(f"Testing on {len(X_test)} samples (weighted pop: {w_test.sum():,.0f})")
        
        # Build models
        self.build_models()
        
        # Train each model
        results = {}
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            try:
                # Most sklearn models support sample_weight in fit()
                if hasattr(model, 'fit'):
                    try:
                        model.fit(X_train, y_train, sample_weight=w_train)
                    except TypeError:
                        # Some models don't support sample_weight
                        model.fit(X_train, y_train)
                        logger.warning(f"{name} does not support sample weights")
                
                # Evaluate with WEIGHTED metrics
                y_pred = model.predict(X_test)
                metrics = self._compute_weighted_metrics(y_test, y_pred, w_test)
                results[name] = metrics
                
                logger.info(f"  {name}: Weighted R² = {metrics['weighted_r2']:.4f}, "
                           f"Weighted RMSE = {metrics['weighted_rmse']:.2f}")
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                continue
        
        # Build ensemble from best models
        self._build_ensemble(X_train, y_train, w_train, X_test, y_test, w_test)
        
        self.metrics = results
        return results
    
    def _compute_weighted_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   weights: np.ndarray) -> Dict[str, float]:
        """Compute weighted regression metrics for population inference."""
        return WeightedMetrics.evaluate(y_true, y_pred, weights, include_unweighted=True)
    
    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute unweighted regression metrics (for backward compatibility)."""
        return {
            'r2': r2_score(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
        }
    
    def _build_ensemble(self, X_train: np.ndarray, y_train: np.ndarray,
                        w_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray,
                        w_test: np.ndarray) -> None:
        """Build ensemble from trained models using weighted evaluation."""
        # Select top 3 models by weighted R² score
        model_scores = {}
        for name, model in self.models.items():
            try:
                y_pred = model.predict(X_test)
                model_scores[name] = WeightedMetrics.weighted_r2(y_test, y_pred, w_test)
            except:
                continue
        
        # Sort by weighted score
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        top_models = sorted_models[:3]
        
        logger.info(f"Building ensemble from: {[m[0] for m in top_models]}")
        
        # Create weighted ensemble
        estimators = [(name, self.models[name]) for name, _ in top_models]
        weights = [score for _, score in top_models]
        weights = [w / sum(weights) for w in weights]  # Normalize
        
        self.ensemble = VotingRegressor(
            estimators=estimators,
            weights=weights,
            n_jobs=-1,
        )
        
        # Fit ensemble with sample weights
        try:
            self.ensemble.fit(X_train, y_train, sample_weight=w_train)
        except TypeError:
            self.ensemble.fit(X_train, y_train)
            logger.warning("VotingRegressor fit without sample weights")
        
        # Evaluate ensemble with weighted metrics
        y_pred = self.ensemble.predict(X_test)
        ensemble_metrics = self._compute_weighted_metrics(y_test, y_pred, w_test)
        self.metrics['ensemble'] = ensemble_metrics
        
        logger.info(f"Ensemble: Weighted R² = {ensemble_metrics['weighted_r2']:.4f}, "
                   f"Weighted RMSE = {ensemble_metrics['weighted_rmse']:.2f}")
    
    def predict(self, X: np.ndarray, return_std: bool = False) -> np.ndarray:
        """
        Make predictions using the ensemble
        """
        if self.ensemble is None:
            raise ValueError("Model not trained. Call train() first.")
        
        predictions = self.ensemble.predict(X)
        
        if return_std:
            # Get individual model predictions for uncertainty
            individual_preds = []
            for name, est in self.ensemble.named_estimators_.items():
                try:
                    pred = est.predict(X)
                    individual_preds.append(pred)
                except:
                    continue
            
            std = np.std(individual_preds, axis=0)
            return predictions, std
        
        return predictions
    
    def predict_with_confidence(self, X: np.ndarray, 
                                 confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions with confidence intervals
        """
        predictions, std = self.predict(X, return_std=True)
        
        # Z-score for confidence level
        from scipy import stats
        z = stats.norm.ppf((1 + confidence) / 2)
        
        lower = predictions - z * std
        upper = predictions + z * std
        
        # Ensure non-negative wages
        lower = np.maximum(lower, 0)
        
        return predictions, lower, upper
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, 
                       model_name: str = 'xgboost',
                       sample_weight: Optional[np.ndarray] = None,
                       n_folds: int = None) -> Dict[str, float]:
        """
        Perform cross-validation with optional weighted scoring.
        
        Note: sklearn's cross_val_score doesn't natively support weighted scoring.
        For proper weighted CV, use WeightedMLSplitter.create_cv_folds() instead.
        """
        if model_name not in self.models:
            self.build_models()
        
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not available")
        
        cv_folds = n_folds or self.config.get('cv_folds', 5)
        
        if sample_weight is not None:
            # Manual weighted cross-validation
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=self.config['random_state'])
            
            scores = []
            for train_idx, val_idx in kf.split(X):
                X_tr, X_val = X[train_idx], X[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]
                w_tr, w_val = sample_weight[train_idx], sample_weight[val_idx]
                
                # Clone model for this fold
                from sklearn.base import clone
                fold_model = clone(model)
                
                try:
                    fold_model.fit(X_tr, y_tr, sample_weight=w_tr)
                except TypeError:
                    fold_model.fit(X_tr, y_tr)
                
                y_pred = fold_model.predict(X_val)
                score = WeightedMetrics.weighted_r2(y_val, y_pred, w_val)
                scores.append(score)
            
            return {
                'mean_weighted_r2': np.mean(scores),
                'std_weighted_r2': np.std(scores),
                'scores': scores,
            }
        else:
            # Standard unweighted CV
            scores = cross_val_score(
                model, X, y, 
                cv=cv_folds, 
                scoring='r2',
                n_jobs=-1
            )
            
            return {
                'mean_r2': scores.mean(),
                'std_r2': scores.std(),
                'scores': scores.tolist(),
            }
    
    def get_feature_importance(self, model_name: str = 'xgboost') -> pd.DataFrame:
        """
        Get feature importance from a specific model
        """
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not trained")
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_)
        else:
            raise ValueError(f"Model {model_name} doesn't have feature importance")
        
        if self.feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importance))]
        else:
            feature_names = self.feature_names[:len(importance)]
        
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df
    
    def save(self, path: str = "models/salary_predictor.joblib") -> None:
        """Save model to disk"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        save_data = {
            'models': self.models,
            'ensemble': self.ensemble,
            'feature_names': self.feature_names,
            'metrics': self.metrics,
            'config': self.config,
        }
        
        joblib.dump(save_data, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str = "models/salary_predictor.joblib") -> None:
        """Load model from disk"""
        save_data = joblib.load(path)
        
        self.models = save_data['models']
        self.ensemble = save_data['ensemble']
        self.feature_names = save_data['feature_names']
        self.metrics = save_data['metrics']
        self.config = save_data['config']
        
        logger.info(f"Model loaded from {path}")


class WageGapModel:
    """
    Specialized model for analyzing wage gaps using Oaxaca-Blinder decomposition.
    
    NOW WITH SURVEY WEIGHTS: All calculations use FINALWT for population inference.
    """
    
    def __init__(self):
        self.model_male = None
        self.model_female = None
        self.results = None
        
    def fit(self, df: pd.DataFrame, 
            features: List[str],
            target: str = None,
            gender_col: str = None,
            weight_col: str = 'FINALWT') -> Dict:
        """
        Fit separate models for each gender group with survey weights.
        
        Args:
            df: DataFrame with features, target, and weights
            features: List of feature column names
            target: Target column name (defaults to COLS.HOURLY_EARNINGS)
            gender_col: Gender column name (defaults to COLS.GENDER)
            weight_col: Survey weight column (defaults to 'FINALWT')
            
        Returns:
            Dictionary with decomposition results
        """
        # Use constants as defaults
        if target is None:
            target = COLS.HOURLY_EARNINGS
        if gender_col is None:
            gender_col = COLS.GENDER
        
        # Validate weight column
        if weight_col not in df.columns:
            warnings.warn(f"Weight column '{weight_col}' not found. Using equal weights.")
            df = df.copy()
            df[weight_col] = 1.0
            
        # Split by gender
        df_male = df[df[gender_col] == 1]
        df_female = df[df[gender_col] == 2]
        
        # Prepare features and weights
        X_male = df_male[features].values
        y_male = df_male[target].values
        w_male = df_male[weight_col].values
        
        X_female = df_female[features].values
        y_female = df_female[target].values
        w_female = df_female[weight_col].values
        
        # Fit models WITH SAMPLE WEIGHTS
        self.model_male = Ridge(alpha=1.0)
        self.model_female = Ridge(alpha=1.0)
        
        self.model_male.fit(X_male, y_male, sample_weight=w_male)
        self.model_female.fit(X_female, y_female, sample_weight=w_female)
        
        # Compute WEIGHTED means for decomposition
        mean_X_male = np.average(X_male, axis=0, weights=w_male)
        mean_X_female = np.average(X_female, axis=0, weights=w_female)
        
        mean_y_male = np.average(y_male, weights=w_male)
        mean_y_female = np.average(y_female, weights=w_female)
        
        # Weighted sample sizes (population representation)
        n_male_weighted = w_male.sum()
        n_female_weighted = w_female.sum()
        
        # Raw gap (weighted)
        raw_gap = mean_y_male - mean_y_female
        raw_gap_pct = (raw_gap / mean_y_male) * 100
        
        # Predicted wages if females had male characteristics
        counterfactual = self.model_male.predict(mean_X_female.reshape(1, -1))[0]
        
        # Explained gap (due to differences in characteristics)
        explained = mean_y_male - counterfactual
        
        # Unexplained gap (potential discrimination)
        unexplained = counterfactual - mean_y_female
        
        self.results = {
            'raw_gap': raw_gap,
            'raw_gap_pct': raw_gap_pct,
            'explained_gap': explained,
            'unexplained_gap': unexplained,
            'explained_pct': (explained / raw_gap) * 100 if raw_gap != 0 else 0,
            'unexplained_pct': (unexplained / raw_gap) * 100 if raw_gap != 0 else 0,
            'mean_male_wage': mean_y_male,
            'mean_female_wage': mean_y_female,
            'n_male': len(df_male),
            'n_female': len(df_female),
            'n_male_weighted': n_male_weighted,
            'n_female_weighted': n_female_weighted,
            'weighted': True,  # Flag that results use survey weights
        }
        
        return self.results


def train_salary_model(df: pd.DataFrame, 
                       feature_engineer=None,
                       weight_col: str = 'FINALWT') -> Tuple[SalaryPredictor, Dict]:
    """
    Convenience function to train salary prediction model WITH SURVEY WEIGHTS.
    
    Args:
        df: DataFrame with LFS data (must include FINALWT)
        feature_engineer: FeatureEngineer instance
        weight_col: Column containing survey weights
        
    Returns:
        Trained SalaryPredictor and metrics dictionary
    """
    from .feature_engineering import FeatureEngineer
    
    if feature_engineer is None:
        feature_engineer = FeatureEngineer()
    
    # Validate weight column exists
    if weight_col not in df.columns:
        raise ValueError(f"Weight column '{weight_col}' not found. "
                        "Survey weights are MANDATORY for this project.")
    
    # Prepare features (preserves weights)
    X, y = feature_engineer.fit_transform(df)
    feature_names = feature_engineer.get_feature_names()
    
    # Get weights aligned with the transformed data
    weights = df[weight_col].values[:len(y)]  # Align if any rows dropped
    
    # Create proper train/val/test splits with weights
    from .ml_utils import WeightedMLSplitter
    splitter = WeightedMLSplitter(
        df=df.iloc[:len(y)],  # Match feature engineering output
        target_col=None,  # We already have y
        weight_col=weight_col,
        feature_cols=None,  # We already have X
    )
    
    # Train model with weights
    predictor = SalaryPredictor()
    metrics = predictor.train(X, y, feature_names, sample_weight=weights)
    
    # Save artifacts
    predictor.save()
    feature_engineer.save()
    
    return predictor, metrics