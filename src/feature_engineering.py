"""
Feature Engineering Module
Handles feature creation, encoding, and transformation for ML models

Uses centralized constants for consistent column naming and code mappings.
"""

import logging
from typing import List, Tuple, Dict, Any, Optional

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path

# Import centralized constants
from .constants import (
    COLS, GENDER_CODES, PROVINCE_CODES, EDUCATION_CODES,
    AGE_6_CODES, AGE_12_CODES, NOC_10_CODES, FTPT_CODES,
    UNION_CODES, PERMTEMP_CODES, ESTSIZE_CODES, MARSTAT_CODES,
    CORE_NUMERIC_FEATURES, CORE_CATEGORICAL_FEATURES, BINARY_FEATURES,
    get_all_mappings
)

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering for compensation analysis.
    
    Uses centralized constants for consistent column naming.
    """
    
    # Feature groups - using constants where possible
    NUMERIC_FEATURES = [
        'AGE_APPROX',
        'EXPERIENCE_PROXY',
        'TENURE',
        'UHRSMAIN',
    ]
    
    CATEGORICAL_FEATURES = [
        COLS.GENDER,
        COLS.EDUCATION,
        COLS.OCCUPATION_10,
        COLS.INDUSTRY,
        COLS.PROVINCE,
        COLS.FULLTIME_PARTTIME,
        COLS.PERMANENT_TEMP,
        COLS.UNION,
        COLS.ESTABLISHMENT_SIZE,
        'MARSTAT',
    ]
    
    BINARY_FEATURES = BINARY_FEATURES  # From constants
    
    TARGET = COLS.HOURLY_EARNINGS
    LOG_TARGET = COLS.LOG_HOURLY_EARNINGS
    
    # Human-readable labels - use centralized mappings
    FEATURE_LABELS = get_all_mappings()
    
    def __init__(self):
        """Initialize feature engineer"""
        self.preprocessor = None
        self.label_encoders = {}
        self.feature_names = None
        
    def get_feature_columns(self, include_binary: bool = True) -> List[str]:
        """Get list of feature columns"""
        features = self.NUMERIC_FEATURES + self.CATEGORICAL_FEATURES
        if include_binary:
            features += self.BINARY_FEATURES
        return features
    
    def prepare_features(self, df: pd.DataFrame, 
                         features: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Prepare features for modeling
        """
        if features is None:
            features = self.get_feature_columns(include_binary=False)
        
        # Select only available features
        available_features = [f for f in features if f in df.columns]
        
        X = df[available_features].copy()
        
        # Handle missing values
        for col in X.columns:
            if X[col].dtype in ['int64', 'float64']:
                X[col] = X[col].fillna(X[col].median())
            else:
                X[col] = X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 'Unknown')
        
        return X
    
    def create_preprocessor(self, 
                            numeric_features: Optional[List[str]] = None,
                            categorical_features: Optional[List[str]] = None) -> ColumnTransformer:
        """
        Create sklearn preprocessing pipeline
        """
        if numeric_features is None:
            numeric_features = self.NUMERIC_FEATURES
        if categorical_features is None:
            categorical_features = self.CATEGORICAL_FEATURES
            
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), 
                 categorical_features)
            ],
            remainder='passthrough'
        )
        
        self.preprocessor = preprocessor
        return preprocessor
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit preprocessor and transform data
        Returns X (features) and y (target)
        """
        # Prepare features
        numeric_cols = [c for c in self.NUMERIC_FEATURES if c in df.columns]
        categorical_cols = [c for c in self.CATEGORICAL_FEATURES if c in df.columns]
        
        X = df[numeric_cols + categorical_cols].copy()
        y = df[self.TARGET].values
        
        # Handle missing values
        for col in numeric_cols:
            X[col] = X[col].fillna(X[col].median())
        for col in categorical_cols:
            X[col] = X[col].fillna(-1).astype(str)
        
        # Create and fit preprocessor
        self.create_preprocessor(numeric_cols, categorical_cols)
        X_transformed = self.preprocessor.fit_transform(X)
        
        # Store feature names
        self._extract_feature_names(numeric_cols, categorical_cols)
        
        return X_transformed, y
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted preprocessor"""
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        
        numeric_cols = [c for c in self.NUMERIC_FEATURES if c in df.columns]
        categorical_cols = [c for c in self.CATEGORICAL_FEATURES if c in df.columns]
        
        X = df[numeric_cols + categorical_cols].copy()
        
        # Handle missing values
        for col in numeric_cols:
            X[col] = X[col].fillna(X[col].median())
        for col in categorical_cols:
            X[col] = X[col].fillna(-1).astype(str)
        
        return self.preprocessor.transform(X)
    
    def _extract_feature_names(self, numeric_cols: List[str], 
                                categorical_cols: List[str]) -> None:
        """Extract feature names after transformation"""
        feature_names = list(numeric_cols)
        
        # Get one-hot encoded feature names
        if hasattr(self.preprocessor, 'transformers_'):
            ohe = self.preprocessor.named_transformers_['cat']
            if hasattr(ohe, 'get_feature_names_out'):
                cat_features = ohe.get_feature_names_out(categorical_cols)
                feature_names.extend(cat_features)
        
        self.feature_names = feature_names
    
    def get_feature_names(self) -> List[str]:
        """Get feature names after transformation"""
        return self.feature_names or []
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features"""
        df = df.copy()
        
        # Education x Experience
        if 'EDUC' in df.columns and 'EXPERIENCE_PROXY' in df.columns:
            df['EDUC_X_EXP'] = df['EDUC'] * df['EXPERIENCE_PROXY']
        
        # Gender x Occupation
        if 'IS_FEMALE' in df.columns and 'NOC_10' in df.columns:
            df['FEMALE_X_OCC'] = df['IS_FEMALE'].astype(str) + '_' + df['NOC_10'].astype(str)
        
        # Full-time x Union
        if 'IS_FULLTIME' in df.columns and 'IS_UNION' in df.columns:
            df['FT_X_UNION'] = df['IS_FULLTIME'] * df['IS_UNION']
        
        return df
    
    def encode_labels(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Encode categorical columns with label encoder"""
        df = df.copy()
        
        for col in columns:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(
                        df[col].astype(str)
                    )
                else:
                    df[f'{col}_encoded'] = self.label_encoders[col].transform(
                        df[col].astype(str)
                    )
        
        return df
    
    def save(self, path: str = "models/feature_engineer.joblib") -> None:
        """Save feature engineer to disk"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'preprocessor': self.preprocessor,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
        }, path)
        logger.info(f"Saved feature engineer to {path}")
    
    def load(self, path: str = "models/feature_engineer.joblib") -> None:
        """Load feature engineer from disk"""
        data = joblib.load(path)
        self.preprocessor = data['preprocessor']
        self.label_encoders = data['label_encoders']
        self.feature_names = data['feature_names']
        logger.info(f"Loaded feature engineer from {path}")


def compute_feature_importance(model, feature_names: List[str]) -> pd.DataFrame:
    """
    Compute feature importance from a trained model
    """
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_).flatten()
    else:
        raise ValueError("Model does not have feature importance attributes")
    
    # Ensure lengths match
    min_len = min(len(feature_names), len(importance))
    
    importance_df = pd.DataFrame({
        'feature': feature_names[:min_len],
        'importance': importance[:min_len]
    }).sort_values('importance', ascending=False)
    
    return importance_df


def get_feature_label(feature: str, value: Any, 
                      labels: Dict = None) -> str:
    """Get human-readable label for a feature value"""
    if labels is None:
        labels = FeatureEngineer.FEATURE_LABELS
    
    if feature in labels and value in labels[feature]:
        return labels[feature][value]
    return str(value)
