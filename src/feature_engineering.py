"""
Feature Engineering Module
Handles feature creation, encoding, and transformation for ML models

Uses centralized constants for consistent column naming and code mappings.
Supports both core and extended feature sets for comprehensive analysis.

Now includes integration with econometric-theory-driven features:
- Distributional features (RIF, quantile position)
- Segregation features (Duncan index, occupation typicality)
- Propensity features (DFL reweighting)
- Counterfactual features (Oaxaca-Blinder derived)
- Glass ceiling features
- Selection features (Heckman)
- Temporal features (structural breaks)

Anti-Overfitting Measures:
- Hierarchical complexity levels
- Built-in VIF multicollinearity detection
- Automatic feature selection
- Cross-validation ready
"""

import logging
from typing import List, Tuple, Dict, Any, Optional

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import mutual_info_regression
import joblib
from pathlib import Path

# Import econometric feature engineering
try:
    from .econometric_features import (
        EconometricFeatureEngineer, 
        FeatureConfig, 
        FeatureComplexity,
        create_econometric_features,
        get_feature_groups
    )
    ECONOMETRIC_FEATURES_AVAILABLE = True
except ImportError:
    ECONOMETRIC_FEATURES_AVAILABLE = False

# Import centralized constants
from .constants import (
    COLS, GENDER_CODES, PROVINCE_CODES, EDUCATION_CODES,
    AGE_6_CODES, AGE_12_CODES, NOC_10_CODES, FTPT_CODES,
    UNION_CODES, PERMTEMP_CODES, ESTSIZE_CODES, MARSTAT_CODES,
    IMMIG_CODES, COWMAIN_CODES, FIRMSIZE_CODES, SCHOOLN_CODES,
    CORE_NUMERIC_FEATURES, CORE_CATEGORICAL_FEATURES, BINARY_FEATURES,
    EXTENDED_NUMERIC_FEATURES, EXTENDED_CATEGORICAL_FEATURES, EXTENDED_BINARY_FEATURES,
    INTERSECTIONAL_ATTRIBUTES,
    get_all_mappings
)

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering for compensation analysis.
    
    Supports two modes:
    - 'core': Original feature set for backward compatibility
    - 'extended': Full feature set exploiting all LFS columns
    
    Uses centralized constants for consistent column naming.
    """
    
    # Core feature groups (backward compatible)
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
    
    # Extended feature groups (exploit all columns)
    EXTENDED_NUMERIC = [
        'AGE_APPROX',
        'EXPERIENCE_PROXY',
        'TENURE',
        'PREVTEN',           # Previous job tenure
        'UHRSMAIN',          # Usual hours
        'AHRSMAIN',          # Actual hours
        'HOURS_GAP',         # Difference between actual and usual
        'TOTAL_OT_HOURS',    # Total overtime hours
    ]
    
    EXTENDED_CATEGORICAL = [
        COLS.GENDER,
        COLS.EDUCATION,
        COLS.OCCUPATION_10,
        'NOC_43',            # Detailed occupation
        COLS.INDUSTRY,
        COLS.PROVINCE,
        'CMA_TYPE',          # Urban/Rural/Major City
        COLS.FULLTIME_PARTTIME,
        COLS.PERMANENT_TEMP,
        COLS.UNION,
        COLS.ESTABLISHMENT_SIZE,
        'FIRMSIZE',          # Firm size
        'MARSTAT',
        'IMMIG',             # Immigration status
        'COWMAIN',           # Class of worker
        'EFAMTYPE',          # Family type
        'AGYOWNK',           # Age of youngest child
        'SCHOOLN',           # Student status
    ]
    
    EXTENDED_BINARY = [
        'IS_FEMALE',
        'IS_FULLTIME',
        'IS_PERMANENT',
        'IS_UNION',
        'HAS_DEGREE',
        'IS_IMMIGRANT',
        'IS_URBAN',
        'IS_MAJOR_CITY',
        'IS_PUBLIC_SECTOR',
        'IS_SELF_EMPLOYED',
        'IS_MULTIPLE_JOBS',
        'IS_MARRIED',
        'HAS_CHILDREN',
        'HAS_YOUNG_CHILDREN',
        'IS_STUDENT',
        'WORKS_OVERTIME',
        'HAS_UNPAID_OVERTIME',
        'IS_INVOLUNTARY_PT',
        'IS_PRECARIOUS',
        'IS_LONE_PARENT',
        'IS_LARGE_FIRM',
    ]
    
    # Intersectional features for equity analysis
    INTERSECTIONAL_FEATURES = [
        'IS_IMMIGRANT_FEMALE',
        'IS_MOTHER_YOUNG_CHILD',
        'IS_FATHER_YOUNG_CHILD',
    ]
    
    BINARY_FEATURES = BINARY_FEATURES  # From constants (core set)
    
    TARGET = COLS.HOURLY_EARNINGS
    LOG_TARGET = COLS.LOG_HOURLY_EARNINGS
    REAL_TARGET = 'REAL_HRLYEARN'
    
    # Human-readable labels - use centralized mappings
    FEATURE_LABELS = get_all_mappings()
    
    def __init__(self, mode: str = 'extended', 
                 econometric_complexity: str = 'standard',
                 enable_econometric_features: bool = True):
        """
        Initialize feature engineer
        
        Args:
            mode: 'core' for backward compatible features, 
                  'extended' for full feature exploitation
            econometric_complexity: 'minimal', 'standard', 'comprehensive', 'experimental'
                Controls the number of theory-driven features
            enable_econometric_features: Whether to compute econometric features
        """
        self.mode = mode
        self.preprocessor = None
        self.label_encoders = {}
        self.feature_names = None
        self.econometric_complexity = econometric_complexity
        self.enable_econometric_features = enable_econometric_features
        
        # Econometric feature engineer
        self.econometric_engineer = None
        self.econometric_selected_features = []
        
        # Set feature lists based on mode
        if mode == 'extended':
            self._numeric_features = self.EXTENDED_NUMERIC
            self._categorical_features = self.EXTENDED_CATEGORICAL
            self._binary_features = self.EXTENDED_BINARY
        else:
            self._numeric_features = self.NUMERIC_FEATURES
            self._categorical_features = self.CATEGORICAL_FEATURES
            self._binary_features = self.BINARY_FEATURES
    
    def create_econometric_features(self, df: pd.DataFrame,
                                     gender_col: str = 'GENDER',
                                     wage_col: str = 'LOG_WAGE',
                                     female_code: int = 2,
                                     select_features: bool = True) -> pd.DataFrame:
        """
        Create theory-driven econometric features.
        
        This method adds features grounded in labor economics theory:
        - Distributional: quantile position, RIF values
        - Segregation: occupation gender shares, Duncan-derived
        - Propensity: DFL weights, gender typicality
        - Counterfactual: Oaxaca-Blinder derived gaps
        - Glass ceiling: upper-tail position features
        - Selection: Heckman-style corrections
        - Temporal: regime indicators, trends
        
        Anti-overfitting measures are built in:
        - VIF multicollinearity detection
        - Low-variance feature removal
        - Highly correlated pair removal
        - Automatic feature selection
        
        Args:
            df: Input DataFrame
            gender_col: Name of gender column
            wage_col: Name of wage column (log scale preferred)
            female_code: Code indicating female
            select_features: Whether to run feature selection
            
        Returns:
            DataFrame with econometric features added
        """
        if not ECONOMETRIC_FEATURES_AVAILABLE:
            logger.warning("Econometric features module not available")
            return df
        
        # Map complexity string to enum
        complexity_map = {
            'minimal': FeatureComplexity.MINIMAL,
            'standard': FeatureComplexity.STANDARD,
            'comprehensive': FeatureComplexity.COMPREHENSIVE,
            'experimental': FeatureComplexity.EXPERIMENTAL
        }
        
        config = FeatureConfig(
            complexity=complexity_map.get(self.econometric_complexity, 
                                         FeatureComplexity.STANDARD)
        )
        
        self.econometric_engineer = EconometricFeatureEngineer(
            config=config,
            gender_col=gender_col,
            wage_col=wage_col,
            female_code=female_code
        )
        
        # Fit and transform
        result = self.econometric_engineer.fit_transform(df)
        
        # Run feature selection if requested
        if select_features and wage_col in df.columns:
            self.econometric_selected_features = self.econometric_engineer.select_features(
                result, df[wage_col]
            )
            logger.info(f"Selected {len(self.econometric_selected_features)} econometric features")
        else:
            self.econometric_selected_features = list(
                self.econometric_engineer.feature_metadata_.keys()
            )
        
        return result
    
    def get_econometric_feature_report(self) -> pd.DataFrame:
        """Get report on econometric features with metadata and selection status."""
        if self.econometric_engineer is None:
            return pd.DataFrame()
        return self.econometric_engineer.get_feature_report()
    
    def get_all_selected_features(self) -> List[str]:
        """Get all selected features (basic + econometric)."""
        basic = self._numeric_features + self._categorical_features + self._binary_features
        return basic + self.econometric_selected_features
        
    def get_feature_columns(self, include_binary: bool = True, 
                             include_intersectional: bool = False) -> List[str]:
        """Get list of feature columns based on current mode"""
        features = self._numeric_features + self._categorical_features
        if include_binary:
            features += self._binary_features
        if include_intersectional and self.mode == 'extended':
            features += self.INTERSECTIONAL_FEATURES
        return features
    
    def get_available_features(self, df: pd.DataFrame, 
                                include_binary: bool = True) -> Dict[str, List[str]]:
        """
        Get features that are actually available in the DataFrame
        
        Returns dict with 'numeric', 'categorical', 'binary' keys
        """
        numeric = [f for f in self._numeric_features if f in df.columns]
        categorical = [f for f in self._categorical_features if f in df.columns]
        binary = [f for f in self._binary_features if f in df.columns] if include_binary else []
        
        return {
            'numeric': numeric,
            'categorical': categorical,
            'binary': binary,
            'all': numeric + categorical + binary
        }
    
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
        
        NOTE: Only uses records with valid wages for training.
        """
        # Filter to records with valid wages for training
        if 'HAS_VALID_WAGE' in df.columns:
            train_df = df[df['HAS_VALID_WAGE']].copy()
            logger.info(f"Using {len(train_df):,} records with valid wages for training "
                       f"(out of {len(df):,} total)")
        else:
            # Fallback: filter by wage > 0
            train_df = df[df[self.TARGET] > 0].copy()
            logger.info(f"Filtered to {len(train_df):,} records with positive wages")
        
        # Prepare features
        numeric_cols = [c for c in self.NUMERIC_FEATURES if c in train_df.columns]
        categorical_cols = [c for c in self.CATEGORICAL_FEATURES if c in train_df.columns]
        
        X = train_df[numeric_cols + categorical_cols].copy()
        y = train_df[self.TARGET].values
        
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
        """
        Create interaction features for deeper analysis
        
        Includes standard interactions and equity-focused intersectional features.
        """
        df = df.copy()
        
        # === Standard Interactions ===
        
        # Education x Experience (human capital)
        if 'EDUC' in df.columns and 'EXPERIENCE_PROXY' in df.columns:
            df['EDUC_X_EXP'] = df['EDUC'] * df['EXPERIENCE_PROXY']
        
        # Full-time x Union (job quality)
        if 'IS_FULLTIME' in df.columns and 'IS_UNION' in df.columns:
            df['FT_X_UNION'] = df['IS_FULLTIME'] * df['IS_UNION']
        
        # Education x Full-time
        if 'HAS_DEGREE' in df.columns and 'IS_FULLTIME' in df.columns:
            df['DEGREE_X_FT'] = df['HAS_DEGREE'] * df['IS_FULLTIME']
        
        # === Equity-Focused Intersectional Interactions ===
        
        # Gender x Immigration (double disadvantage)
        if 'IS_FEMALE' in df.columns and 'IS_IMMIGRANT' in df.columns:
            df['FEMALE_X_IMMIGRANT'] = df['IS_FEMALE'] * df['IS_IMMIGRANT']
        
        # Gender x Parenthood (motherhood penalty)
        if 'IS_FEMALE' in df.columns and 'HAS_YOUNG_CHILDREN' in df.columns:
            df['FEMALE_X_YOUNG_CHILD'] = df['IS_FEMALE'] * df['HAS_YOUNG_CHILDREN']
        
        # Gender x Education (returns to education by gender)
        if 'IS_FEMALE' in df.columns and 'HAS_DEGREE' in df.columns:
            df['FEMALE_X_DEGREE'] = df['IS_FEMALE'] * df['HAS_DEGREE']
        
        # Gender x Public Sector (sector segregation)
        if 'IS_FEMALE' in df.columns and 'IS_PUBLIC_SECTOR' in df.columns:
            df['FEMALE_X_PUBLIC'] = df['IS_FEMALE'] * df['IS_PUBLIC_SECTOR']
        
        # Gender x Urban (geographic effects)
        if 'IS_FEMALE' in df.columns and 'IS_URBAN' in df.columns:
            df['FEMALE_X_URBAN'] = df['IS_FEMALE'] * df['IS_URBAN']
        
        # Gender x Occupation (occupational segregation)
        if 'IS_FEMALE' in df.columns and 'NOC_10' in df.columns:
            df['FEMALE_X_OCC'] = df['IS_FEMALE'].astype(str) + '_' + df['NOC_10'].astype(str)
        
        # Immigration x Education (credential recognition)
        if 'IS_IMMIGRANT' in df.columns and 'HAS_DEGREE' in df.columns:
            df['IMMIGRANT_X_DEGREE'] = df['IS_IMMIGRANT'] * df['HAS_DEGREE']
        
        # Immigration x Urban (settlement patterns)
        if 'IS_IMMIGRANT' in df.columns and 'IS_URBAN' in df.columns:
            df['IMMIGRANT_X_URBAN'] = df['IS_IMMIGRANT'] * df['IS_URBAN']
        
        # Parenthood x Full-time (work-life balance)
        if 'HAS_YOUNG_CHILDREN' in df.columns and 'IS_FULLTIME' in df.columns:
            df['PARENT_X_FT'] = df['HAS_YOUNG_CHILDREN'] * df['IS_FULLTIME']
        
        # Lone parent x Female (single mothers)
        if 'IS_LONE_PARENT' in df.columns and 'IS_FEMALE' in df.columns:
            df['FEMALE_LONE_PARENT'] = df['IS_FEMALE'] * df['IS_LONE_PARENT']
        
        # Precarious work x Gender
        if 'IS_PRECARIOUS' in df.columns and 'IS_FEMALE' in df.columns:
            df['FEMALE_X_PRECARIOUS'] = df['IS_FEMALE'] * df['IS_PRECARIOUS']
        
        # Overtime x Gender (unpaid labor)
        if 'HAS_UNPAID_OVERTIME' in df.columns and 'IS_FEMALE' in df.columns:
            df['FEMALE_X_UNPAID_OT'] = df['IS_FEMALE'] * df['HAS_UNPAID_OVERTIME']
        
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
