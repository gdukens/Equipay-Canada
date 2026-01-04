"""
Econometric & Theory-Driven Feature Engineering Module
=======================================================

This module creates features grounded in labor economics theory, ensuring
that ML models capture the TRUE mechanisms driving pay gaps.

Feature Categories:
1. Distributional Features (RIF, quantile position)
2. Segregation Features (Duncan index, occupation typicality)  
3. Propensity & Selection Features (propensity scores, Mills ratio)
4. Counterfactual Features (Oaxaca-Blinder derived)
5. Occupational Sorting Features (Brown-Moon-Zoloth)
6. Glass Ceiling Features (upper-tail effects)
7. Temporal & Regime Features (structural breaks)
8. Convergence Features (gap dynamics)

Anti-Overfitting Measures:
- Feature groups with hierarchical complexity
- Built-in multicollinearity detection (VIF)
- Automatic feature selection via information criteria
- Cross-validation ready feature subsets
- Regularization-ready scaling

References:
-----------
- Firpo, Fortin & Lemieux (2009) - RIF Regression
- Oaxaca (1973), Blinder (1973) - Wage Decomposition
- DiNardo, Fortin & Lemieux (1996) - Reweighting
- Heckman (1979) - Selection Correction
- Brown, Moon & Zoloth (1980) - Occupational Attainment
"""

import logging
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression, VarianceThreshold
from sklearn.model_selection import cross_val_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

class FeatureComplexity(Enum):
    """Feature complexity levels for controlling overfitting risk."""
    MINIMAL = 1      # Only essential, low-risk features
    STANDARD = 2     # Balanced feature set
    COMPREHENSIVE = 3  # Full feature set with all theory-driven features
    EXPERIMENTAL = 4   # Including high-dimensional/risky features


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    complexity: FeatureComplexity = FeatureComplexity.STANDARD
    max_vif: float = 10.0  # Maximum variance inflation factor
    min_variance: float = 0.01  # Minimum variance threshold
    min_correlation_with_target: float = 0.01  # Minimum correlation with target
    max_pairwise_correlation: float = 0.95  # Max correlation between features
    remove_collinear: bool = True
    scale_features: bool = True
    compute_selection_features: bool = True
    compute_temporal_features: bool = True
    n_quantiles: int = 9  # For RIF computation
    propensity_clip: Tuple[float, float] = (0.01, 0.99)
    random_state: int = 42


@dataclass
class FeatureMetadata:
    """Metadata about generated features for interpretability."""
    name: str
    category: str
    theory_source: str
    complexity: FeatureComplexity
    description: str
    interpretation: str
    vif: Optional[float] = None
    correlation_with_target: Optional[float] = None


# =============================================================================
# MAIN FEATURE ENGINEERING CLASS
# =============================================================================

class EconometricFeatureEngineer:
    """
    Theory-driven feature engineering with anti-overfitting safeguards.
    
    Creates features grounded in labor economics theory while automatically
    detecting and mitigating overfitting risks.
    
    Parameters
    ----------
    config : FeatureConfig
        Configuration controlling complexity and safeguards
    gender_col : str
        Column name for gender indicator
    wage_col : str
        Column name for wage variable
    female_code : int
        Code indicating female in gender column
        
    Examples
    --------
    >>> engineer = EconometricFeatureEngineer(
    ...     config=FeatureConfig(complexity=FeatureComplexity.STANDARD)
    ... )
    >>> df_features = engineer.fit_transform(df)
    >>> selected_features = engineer.get_selected_features()
    """
    
    def __init__(self, 
                 config: FeatureConfig = None,
                 gender_col: str = 'GENDER',
                 wage_col: str = 'LOG_WAGE',
                 female_code: int = 2):
        self.config = config or FeatureConfig()
        self.gender_col = gender_col
        self.wage_col = wage_col
        self.female_code = female_code
        
        # Fitted state
        self.is_fitted_ = False
        self.feature_metadata_: Dict[str, FeatureMetadata] = {}
        self.selected_features_: List[str] = []
        self.removed_features_: Dict[str, str] = {}  # feature -> removal reason
        self.scalers_: Dict[str, StandardScaler] = {}
        
        # Learned parameters for transform
        self.group_means_: Dict = {}
        self.quantile_values_: Dict = {}
        self.propensity_model_ = None
        self.occupation_shares_: Dict = {}
        self.within_occ_params_: Dict = {}
        self.regime_breaks_: List = []
        
    def fit(self, df: pd.DataFrame, y: pd.Series = None) -> 'EconometricFeatureEngineer':
        """
        Fit the feature engineer on training data.
        
        Learns all necessary parameters for feature creation:
        - Group means and quantiles
        - Propensity score model
        - Occupation segregation measures
        - Within-occupation wage parameters
        
        Parameters
        ----------
        df : DataFrame
            Training data
        y : Series, optional
            Target variable for feature selection
        """
        logger.info("Fitting EconometricFeatureEngineer...")
        
        if y is None and self.wage_col in df.columns:
            y = df[self.wage_col]
        
        # Learn group statistics
        self._fit_group_statistics(df)
        
        # Learn propensity model
        self._fit_propensity_model(df)
        
        # Learn occupation parameters
        self._fit_occupation_parameters(df)
        
        # Learn temporal parameters if available
        if 'SURVYEAR' in df.columns or 'year' in df.columns:
            self._fit_temporal_parameters(df)
        
        self.is_fitted_ = True
        logger.info("Feature engineer fitted successfully")
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by creating all theory-driven features.
        
        Parameters
        ----------
        df : DataFrame
            Data to transform
            
        Returns
        -------
        DataFrame with new features added
        """
        if not self.is_fitted_:
            raise ValueError("Feature engineer not fitted. Call fit() first.")
        
        result = df.copy()
        
        # Create features based on complexity level
        if self.config.complexity.value >= FeatureComplexity.MINIMAL.value:
            result = self._create_distributional_features(result)
            result = self._create_basic_segregation_features(result)
            
        if self.config.complexity.value >= FeatureComplexity.STANDARD.value:
            result = self._create_propensity_features(result)
            result = self._create_counterfactual_features(result)
            result = self._create_glass_ceiling_features(result)
            
        if self.config.complexity.value >= FeatureComplexity.COMPREHENSIVE.value:
            result = self._create_advanced_segregation_features(result)
            result = self._create_selection_features(result)
            result = self._create_occupational_sorting_features(result)
            
        if self.config.complexity.value >= FeatureComplexity.EXPERIMENTAL.value:
            result = self._create_rif_features(result)
            result = self._create_temporal_regime_features(result)
        
        # Apply scaling if configured
        if self.config.scale_features:
            result = self._scale_new_features(result)
        
        return result
    
    def fit_transform(self, df: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df, y).transform(df)
    
    def select_features(self, df: pd.DataFrame, y: pd.Series,
                        method: str = 'vif_and_correlation') -> List[str]:
        """
        Select features using anti-overfitting criteria.
        
        Parameters
        ----------
        df : DataFrame
            Feature matrix
        y : Series
            Target variable
        method : str
            Selection method: 'vif_and_correlation', 'mutual_info', 'all'
            
        Returns
        -------
        List of selected feature names
        """
        new_features = [c for c in df.columns if c not in 
                       ['SURVYEAR', 'year', self.gender_col, self.wage_col, 
                        'HRLYEARN', 'FINALWT', 'WAGE']]
        
        # Get numeric features only
        numeric_features = [c for c in new_features 
                          if df[c].dtype in ['int64', 'float64', 'int32', 'float32']]
        
        if len(numeric_features) == 0:
            return []
        
        X = df[numeric_features].copy()
        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
        
        selected = set(numeric_features)
        
        # Step 1: Remove low variance features
        variances = X.var()
        low_var = variances[variances < self.config.min_variance].index.tolist()
        for f in low_var:
            selected.discard(f)
            self.removed_features_[f] = 'low_variance'
        
        # Step 2: Remove features with low correlation to target
        if y is not None:
            y_clean = y.loc[X.index].fillna(y.median())
            correlations = X[list(selected)].corrwith(y_clean).abs()
            low_corr = correlations[correlations < self.config.min_correlation_with_target].index.tolist()
            for f in low_corr:
                selected.discard(f)
                self.removed_features_[f] = 'low_target_correlation'
        
        # Step 3: Remove highly correlated pairs (keep the one more correlated with target)
        if self.config.remove_collinear and len(selected) > 1:
            X_selected = X[list(selected)]
            corr_matrix = X_selected.corr().abs()
            
            # Find highly correlated pairs
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            high_corr_pairs = [(col, row) for col in upper.columns for row in upper.index 
                              if upper.loc[row, col] > self.config.max_pairwise_correlation]
            
            for f1, f2 in high_corr_pairs:
                if f1 in selected and f2 in selected:
                    # Keep the one more correlated with target
                    if y is not None:
                        corr1 = abs(X[f1].corr(y_clean))
                        corr2 = abs(X[f2].corr(y_clean))
                        remove = f2 if corr1 >= corr2 else f1
                    else:
                        remove = f2
                    selected.discard(remove)
                    self.removed_features_[remove] = f'collinear_with_{f1 if remove == f2 else f2}'
        
        # Step 4: VIF check for remaining features
        if self.config.remove_collinear and len(selected) > 2:
            selected = self._vif_selection(X[list(selected)])
        
        self.selected_features_ = list(selected)
        logger.info(f"Selected {len(self.selected_features_)} features, "
                   f"removed {len(self.removed_features_)}")
        
        return self.selected_features_
    
    def get_feature_report(self) -> pd.DataFrame:
        """Get a report of all features with metadata."""
        if not self.feature_metadata_:
            return pd.DataFrame()
        
        records = []
        for name, meta in self.feature_metadata_.items():
            records.append({
                'feature': name,
                'category': meta.category,
                'theory': meta.theory_source,
                'complexity': meta.complexity.name,
                'description': meta.description,
                'vif': meta.vif,
                'target_corr': meta.correlation_with_target,
                'selected': name in self.selected_features_,
                'removal_reason': self.removed_features_.get(name, None)
            })
        
        return pd.DataFrame(records)
    
    # =========================================================================
    # FITTING METHODS
    # =========================================================================
    
    def _fit_group_statistics(self, df: pd.DataFrame):
        """Learn group means, quantiles, and distributions."""
        gender = df[self.gender_col]
        
        # Overall and group-specific means for key variables
        for col in ['EDUC', 'AGE_12', 'TENURE', 'EXPERIENCE_PROXY', self.wage_col]:
            if col in df.columns:
                self.group_means_[f'{col}_overall'] = df[col].mean()
                self.group_means_[f'{col}_male'] = df.loc[gender != self.female_code, col].mean()
                self.group_means_[f'{col}_female'] = df.loc[gender == self.female_code, col].mean()
        
        # Quantiles for wage distribution
        if self.wage_col in df.columns:
            for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
                self.quantile_values_[q] = df[self.wage_col].quantile(q)
                self.quantile_values_[f'{q}_male'] = df.loc[gender != self.female_code, self.wage_col].quantile(q)
                self.quantile_values_[f'{q}_female'] = df.loc[gender == self.female_code, self.wage_col].quantile(q)
    
    def _fit_propensity_model(self, df: pd.DataFrame):
        """Fit propensity score model P(Female | X)."""
        # Select covariates for propensity model
        covariates = []
        for col in ['EDUC', 'AGE_12', 'PROV', 'NOC_10', 'TENURE']:
            if col in df.columns:
                covariates.append(col)
        
        if not covariates:
            logger.warning("No covariates available for propensity model")
            return
        
        # Prepare data
        X = df[covariates].copy()
        X = X.fillna(X.median())
        y = (df[self.gender_col] == self.female_code).astype(int)
        
        # Fit logistic regression
        self.propensity_model_ = LogisticRegression(max_iter=1000, random_state=self.config.random_state)
        self.propensity_model_.fit(X, y)
        self.propensity_covariates_ = covariates
    
    def _fit_occupation_parameters(self, df: pd.DataFrame):
        """Learn occupation-level parameters for segregation features."""
        if 'NOC_10' not in df.columns:
            return
        
        gender = df[self.gender_col]
        
        # Female share by occupation
        occ_stats = df.groupby('NOC_10').agg({
            self.gender_col: lambda x: (x == self.female_code).mean(),
            self.wage_col: ['mean', 'std', 'count'] if self.wage_col in df.columns else 'count'
        })
        
        if self.wage_col in df.columns:
            occ_stats.columns = ['female_share', 'mean_wage', 'std_wage', 'count']
        else:
            occ_stats.columns = ['female_share', 'count']
        
        self.occupation_shares_ = occ_stats.to_dict('index')
        
        # Within-occupation wage parameters by gender
        if self.wage_col in df.columns:
            for occ in df['NOC_10'].unique():
                occ_data = df[df['NOC_10'] == occ]
                self.within_occ_params_[occ] = {
                    'male_mean': occ_data.loc[gender != self.female_code, self.wage_col].mean(),
                    'female_mean': occ_data.loc[gender == self.female_code, self.wage_col].mean(),
                    'overall_mean': occ_data[self.wage_col].mean(),
                    'overall_std': occ_data[self.wage_col].std()
                }
    
    def _fit_temporal_parameters(self, df: pd.DataFrame):
        """Learn temporal regime breaks."""
        year_col = 'SURVYEAR' if 'SURVYEAR' in df.columns else 'year'
        if year_col not in df.columns:
            return
        
        # Simple regime detection: pre/post COVID
        self.regime_breaks_ = [2019]  # Could be extended with Bai-Perron
        
        # Compute trend parameters
        years = df[year_col].unique()
        if len(years) > 3 and self.wage_col in df.columns:
            yearly_gap = df.groupby(year_col).apply(
                lambda x: x.loc[x[self.gender_col] != self.female_code, self.wage_col].mean() - 
                         x.loc[x[self.gender_col] == self.female_code, self.wage_col].mean()
            )
            self.gap_trend_ = np.polyfit(yearly_gap.index, yearly_gap.values, 1)[0]
        else:
            self.gap_trend_ = 0
    
    # =========================================================================
    # FEATURE CREATION METHODS
    # =========================================================================
    
    def _create_distributional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features based on distributional position."""
        result = df.copy()
        gender = df[self.gender_col]
        
        # 1. Within-gender percentile rank
        if self.wage_col in df.columns:
            result['within_gender_percentile'] = df.groupby(self.gender_col)[self.wage_col].rank(pct=True)
            self._add_metadata('within_gender_percentile', 'distributional', 'Machado-Mata',
                             FeatureComplexity.MINIMAL,
                             'Position within own gender wage distribution',
                             'Higher = higher relative position')
            
            # 2. Distance from group median
            male_median = self.quantile_values_.get('0.5_male', df[self.wage_col].median())
            female_median = self.quantile_values_.get('0.5_female', df[self.wage_col].median())
            
            result['distance_from_gender_median'] = np.where(
                gender == self.female_code,
                df[self.wage_col] - female_median,
                df[self.wage_col] - male_median
            )
            self._add_metadata('distance_from_gender_median', 'distributional', 'Quantile Analysis',
                             FeatureComplexity.MINIMAL,
                             'Distance from own gender median wage',
                             'Positive = above median')
            
            # 3. Wage distribution zone
            q25 = self.quantile_values_.get(0.25, df[self.wage_col].quantile(0.25))
            q75 = self.quantile_values_.get(0.75, df[self.wage_col].quantile(0.75))
            
            result['wage_zone_lower'] = (df[self.wage_col] < q25).astype(int)
            result['wage_zone_upper'] = (df[self.wage_col] > q75).astype(int)
            result['wage_zone_middle'] = ((df[self.wage_col] >= q25) & 
                                          (df[self.wage_col] <= q75)).astype(int)
            
            self._add_metadata('wage_zone_upper', 'distributional', 'Glass Ceiling',
                             FeatureComplexity.MINIMAL,
                             'In upper quartile of wage distribution',
                             '1 = in top 25%')
        
        # 4. Standardized characteristics (distance from group mean)
        for col in ['EDUC', 'AGE_12']:
            if col in df.columns and f'{col}_overall' in self.group_means_:
                overall_mean = self.group_means_[f'{col}_overall']
                overall_std = df[col].std()
                if overall_std > 0:
                    result[f'{col}_standardized'] = (df[col] - overall_mean) / overall_std
                    self._add_metadata(f'{col}_standardized', 'distributional', 'Human Capital',
                                     FeatureComplexity.MINIMAL,
                                     f'Standardized {col} (z-score)',
                                     'How many SDs from mean')
        
        return result
    
    def _create_basic_segregation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic segregation features."""
        result = df.copy()
        
        if 'NOC_10' not in df.columns or not self.occupation_shares_:
            return result
        
        # 1. Female share in occupation
        result['occ_female_share'] = df['NOC_10'].map(
            lambda x: self.occupation_shares_.get(x, {}).get('female_share', 0.5)
        )
        self._add_metadata('occ_female_share', 'segregation', 'Duncan Index',
                         FeatureComplexity.MINIMAL,
                         'Share of females in occupation',
                         '0-1, higher = more female-dominated')
        
        # 2. Occupation type indicators
        result['female_dominated_occ'] = (result['occ_female_share'] > 0.7).astype(int)
        result['male_dominated_occ'] = (result['occ_female_share'] < 0.3).astype(int)
        result['integrated_occ'] = ((result['occ_female_share'] >= 0.3) & 
                                     (result['occ_female_share'] <= 0.7)).astype(int)
        
        self._add_metadata('female_dominated_occ', 'segregation', 'Occupational Segregation',
                         FeatureComplexity.MINIMAL,
                         'Occupation is >70% female',
                         '1 = female-dominated occupation')
        
        # 3. Segregation distance from overall mean
        overall_female_share = (df[self.gender_col] == self.female_code).mean()
        result['occ_segregation_score'] = abs(result['occ_female_share'] - overall_female_share)
        
        self._add_metadata('occ_segregation_score', 'segregation', 'Duncan Index',
                         FeatureComplexity.MINIMAL,
                         'Absolute deviation from overall gender ratio',
                         'Higher = more segregated occupation')
        
        return result
    
    def _create_propensity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create propensity score based features."""
        result = df.copy()
        
        if self.propensity_model_ is None:
            return result
        
        # Get propensity scores
        X = df[self.propensity_covariates_].copy().fillna(df[self.propensity_covariates_].median())
        propensity = self.propensity_model_.predict_proba(X)[:, 1]
        propensity = np.clip(propensity, *self.config.propensity_clip)
        
        # 1. Raw propensity score
        result['propensity_female'] = propensity
        self._add_metadata('propensity_female', 'propensity', 'DFL Reweighting',
                         FeatureComplexity.STANDARD,
                         'Probability of being female given characteristics',
                         '0-1, based on logistic regression')
        
        # 2. Gender typicality score
        gender = df[self.gender_col]
        result['gender_typicality'] = np.where(
            gender == self.female_code,
            1 - propensity,  # Atypical female (low propensity but female)
            propensity       # Atypical male (high propensity but male)
        )
        self._add_metadata('gender_typicality', 'propensity', 'DFL Reweighting',
                         FeatureComplexity.STANDARD,
                         'How atypical is this person for their gender',
                         'Higher = more atypical characteristics for their gender')
        
        # 3. Inverse probability weight
        result['ipw_female'] = np.where(
            gender == self.female_code,
            1 / propensity,
            1 / (1 - propensity)
        )
        # Clip extreme weights
        result['ipw_female'] = np.clip(result['ipw_female'], 0.1, 10)
        
        self._add_metadata('ipw_female', 'propensity', 'IPW Estimation',
                         FeatureComplexity.STANDARD,
                         'Inverse probability weight for causal inference',
                         'Used for reweighting in decomposition')
        
        # 4. Common support indicator
        result['common_support'] = ((propensity > 0.1) & (propensity < 0.9)).astype(int)
        self._add_metadata('common_support', 'propensity', 'PSM',
                         FeatureComplexity.STANDARD,
                         'In propensity score overlap region',
                         '1 = good counterfactual match exists')
        
        return result
    
    def _create_counterfactual_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create counterfactual/decomposition features."""
        result = df.copy()
        gender = df[self.gender_col]
        
        # 1. Distance from own-gender mean for key characteristics
        for col in ['EDUC', 'AGE_12', 'TENURE']:
            if col in df.columns:
                male_mean = self.group_means_.get(f'{col}_male', df[col].mean())
                female_mean = self.group_means_.get(f'{col}_female', df[col].mean())
                
                result[f'{col}_gap_from_gender_mean'] = np.where(
                    gender == self.female_code,
                    df[col] - female_mean,
                    df[col] - male_mean
                )
                self._add_metadata(f'{col}_gap_from_gender_mean', 'counterfactual', 'Oaxaca-Blinder',
                                 FeatureComplexity.STANDARD,
                                 f'{col} relative to own gender mean',
                                 'Positive = above gender average')
                
                # Distance from opposite gender mean (endowment gap)
                result[f'{col}_vs_opposite_gender'] = np.where(
                    gender == self.female_code,
                    df[col] - male_mean,  # How does female compare to male average
                    df[col] - female_mean
                )
                self._add_metadata(f'{col}_vs_opposite_gender', 'counterfactual', 'Oaxaca-Blinder',
                                 FeatureComplexity.STANDARD,
                                 f'{col} relative to opposite gender mean',
                                 'Captures endowment differences')
        
        # 2. Predicted wage using opposite gender's distribution
        if self.wage_col in df.columns:
            # Simple counterfactual: what would wage be at opposite gender's median?
            male_median = self.quantile_values_.get('0.5_male', 0)
            female_median = self.quantile_values_.get('0.5_female', 0)
            gap = male_median - female_median
            
            result['counterfactual_wage_adjustment'] = np.where(
                gender == self.female_code,
                gap,   # Add gap to female wages
                -gap   # Subtract gap from male wages
            )
            self._add_metadata('counterfactual_wage_adjustment', 'counterfactual', 'DFL',
                             FeatureComplexity.STANDARD,
                             'Adjustment to match opposite gender distribution',
                             'What would wage be under opposite gender structure')
        
        return result
    
    def _create_glass_ceiling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create glass ceiling related features."""
        result = df.copy()
        
        if self.wage_col not in df.columns:
            return result
        
        gender = df[self.gender_col]
        wage = df[self.wage_col]
        
        # 1. Position relative to 90th percentile (ceiling proximity)
        p90 = self.quantile_values_.get(0.9, wage.quantile(0.9))
        result['ceiling_proximity'] = wage / p90
        self._add_metadata('ceiling_proximity', 'glass_ceiling', 'Glass Ceiling Index',
                         FeatureComplexity.STANDARD,
                         'Wage as fraction of 90th percentile',
                         'Higher = closer to the ceiling')
        
        # 2. Above median indicator (for ceiling zone)
        p50 = self.quantile_values_.get(0.5, wage.median())
        result['above_median'] = (wage > p50).astype(int)
        
        # 3. Glass ceiling zone interaction
        result['female_ceiling_zone'] = ((gender == self.female_code) & 
                                          (wage > p50)).astype(int)
        self._add_metadata('female_ceiling_zone', 'glass_ceiling', 'Glass Ceiling Index',
                         FeatureComplexity.STANDARD,
                         'Female in upper half of wage distribution',
                         'Area where glass ceiling effects are strongest')
        
        # 4. Distance from gender-specific ceiling
        p90_male = self.quantile_values_.get('0.9_male', p90)
        p90_female = self.quantile_values_.get('0.9_female', p90)
        
        result['distance_to_ceiling'] = np.where(
            gender == self.female_code,
            p90_female - wage,
            p90_male - wage
        )
        self._add_metadata('distance_to_ceiling', 'glass_ceiling', 'Glass Ceiling Index',
                         FeatureComplexity.STANDARD,
                         'Gap to own gender 90th percentile',
                         'Room for wage growth within current structure')
        
        return result
    
    def _create_advanced_segregation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced segregation features."""
        result = df.copy()
        
        if 'NOC_10' not in df.columns or not self.occupation_shares_:
            return result
        
        gender = df[self.gender_col]
        
        # 1. Within-occupation gender wage gap exposure
        def get_within_occ_gap(occ):
            params = self.within_occ_params_.get(occ, {})
            male_mean = params.get('male_mean', 0)
            female_mean = params.get('female_mean', 0)
            return male_mean - female_mean if male_mean and female_mean else 0
        
        result['within_occ_wage_gap'] = df['NOC_10'].map(get_within_occ_gap)
        self._add_metadata('within_occ_wage_gap', 'segregation', 'Brown-Moon-Zoloth',
                         FeatureComplexity.COMPREHENSIVE,
                         'Gender wage gap within this occupation',
                         'Higher = larger within-occupation gap')
        
        # 2. Occupation wage premium relative to overall
        overall_mean = self.group_means_.get(f'{self.wage_col}_overall', 0)
        
        def get_occ_premium(occ):
            params = self.within_occ_params_.get(occ, {})
            return params.get('overall_mean', overall_mean) - overall_mean
        
        result['occupation_wage_premium'] = df['NOC_10'].map(get_occ_premium)
        self._add_metadata('occupation_wage_premium', 'segregation', 'Occupational Decomposition',
                         FeatureComplexity.COMPREHENSIVE,
                         'Occupation mean wage relative to overall',
                         'Positive = high-paying occupation')
        
        # 3. Working in "wrong" occupation (gender mismatch)
        result['gender_occ_mismatch'] = np.where(
            gender == self.female_code,
            result.get('male_dominated_occ', 0),  # Female in male-dominated
            result.get('female_dominated_occ', 0)  # Male in female-dominated
        )
        self._add_metadata('gender_occ_mismatch', 'segregation', 'Occupational Attainment',
                         FeatureComplexity.COMPREHENSIVE,
                         'In occupation atypical for gender',
                         '1 = in opposite-gender-dominated occupation')
        
        return result
    
    def _create_selection_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create selection correction related features."""
        result = df.copy()
        
        if not self.config.compute_selection_features:
            return result
        
        # Inverse Mills ratio proxy (for sample selection)
        # This is a simplified version - full Heckman requires selection equation
        
        if 'propensity_female' in result.columns:
            propensity = result['propensity_female'].values
            
            # Mills ratio approximation for labor force participation
            # Higher propensity females might have different selection patterns
            xb = stats.norm.ppf(np.clip(propensity, 0.01, 0.99))
            result['mills_ratio_proxy'] = stats.norm.pdf(xb) / stats.norm.cdf(xb)
            result['mills_ratio_proxy'] = result['mills_ratio_proxy'].replace([np.inf, -np.inf], np.nan)
            result['mills_ratio_proxy'] = result['mills_ratio_proxy'].fillna(0)
            
            self._add_metadata('mills_ratio_proxy', 'selection', 'Heckman Selection',
                             FeatureComplexity.COMPREHENSIVE,
                             'Inverse Mills ratio approximation',
                             'Captures selection into sample')
        
        return result
    
    def _create_occupational_sorting_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create occupational sorting features (Brown-Moon-Zoloth)."""
        result = df.copy()
        
        if 'NOC_10' not in df.columns:
            return result
        
        # 1. Expected vs actual occupation wage
        if self.wage_col in df.columns and self.within_occ_params_:
            def get_expected_wage(row):
                occ = row['NOC_10']
                params = self.within_occ_params_.get(occ, {})
                if row[self.gender_col] == self.female_code:
                    return params.get('female_mean', row[self.wage_col])
                else:
                    return params.get('male_mean', row[self.wage_col])
            
            result['expected_wage_for_occ'] = df.apply(get_expected_wage, axis=1)
            result['wage_vs_occ_expected'] = df[self.wage_col] - result['expected_wage_for_occ']
            
            self._add_metadata('wage_vs_occ_expected', 'sorting', 'Brown-Moon-Zoloth',
                             FeatureComplexity.COMPREHENSIVE,
                             'Actual wage minus expected for occupation/gender',
                             'Positive = earning above occupation-gender average')
        
        return result
    
    def _create_rif_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create RIF (Recentered Influence Function) features."""
        result = df.copy()
        
        if self.wage_col not in df.columns:
            return result
        
        wage = df[self.wage_col].values
        n = len(wage)
        
        # Silverman's rule for bandwidth
        h = 1.06 * np.std(wage) * n ** (-1/5)
        
        for tau in [0.1, 0.5, 0.9]:
            q_tau = np.quantile(wage, tau)
            
            # Kernel density at quantile
            kernel_vals = norm.pdf((wage - q_tau) / h) / h
            f_tau = max(np.mean(kernel_vals), 1e-10)
            
            # RIF formula
            indicator = (wage <= q_tau).astype(float)
            rif = q_tau + (tau - indicator) / f_tau
            
            result[f'rif_q{int(tau*100)}'] = rif
            self._add_metadata(f'rif_q{int(tau*100)}', 'distributional', 'RIF-OLS',
                             FeatureComplexity.EXPERIMENTAL,
                             f'Recentered Influence Function at {int(tau*100)}th percentile',
                             'Captures influence on distributional statistics')
        
        # RIF spread (uncertainty proxy)
        if 'rif_q90' in result.columns and 'rif_q10' in result.columns:
            result['rif_spread'] = result['rif_q90'] - result['rif_q10']
            self._add_metadata('rif_spread', 'distributional', 'RIF-OLS',
                             FeatureComplexity.EXPERIMENTAL,
                             'Spread of RIF values across quantiles',
                             'Higher = more influence on distribution tails')
        
        return result
    
    def _create_temporal_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal and regime-based features."""
        result = df.copy()
        
        year_col = 'SURVYEAR' if 'SURVYEAR' in df.columns else 'year'
        if year_col not in df.columns:
            return result
        
        years = df[year_col]
        min_year = years.min()
        
        # 1. Trend variable
        result['trend'] = years - min_year
        self._add_metadata('trend', 'temporal', 'Time Series',
                         FeatureComplexity.EXPERIMENTAL,
                         'Linear time trend (years since start)',
                         'Captures secular changes')
        
        # 2. Regime indicators based on structural breaks
        for break_year in self.regime_breaks_:
            result[f'post_{break_year}'] = (years >= break_year).astype(int)
            self._add_metadata(f'post_{break_year}', 'temporal', 'Structural Breaks',
                             FeatureComplexity.EXPERIMENTAL,
                             f'Period after {break_year}',
                             'Captures regime change')
        
        # 3. Trend interactions with gender
        gender = df[self.gender_col]
        result['trend_x_female'] = result['trend'] * (gender == self.female_code).astype(int)
        self._add_metadata('trend_x_female', 'temporal', 'Time Series',
                         FeatureComplexity.EXPERIMENTAL,
                         'Time trend interacted with female',
                         'How gap changes over time')
        
        # 4. Gap closing rate (if we have it)
        if hasattr(self, 'gap_trend_'):
            result['gap_trend_exposure'] = result['trend'] * self.gap_trend_
            self._add_metadata('gap_trend_exposure', 'temporal', 'Convergence',
                             FeatureComplexity.EXPERIMENTAL,
                             'Exposure to gap trend over time',
                             'Based on estimated gap trajectory')
        
        return result
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _add_metadata(self, name: str, category: str, theory: str,
                     complexity: FeatureComplexity, description: str,
                     interpretation: str):
        """Add metadata for a feature."""
        self.feature_metadata_[name] = FeatureMetadata(
            name=name,
            category=category,
            theory_source=theory,
            complexity=complexity,
            description=description,
            interpretation=interpretation
        )
    
    def _vif_selection(self, X: pd.DataFrame) -> List[str]:
        """Select features using VIF (remove high multicollinearity)."""
        X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
        features = list(X.columns)
        
        # Add constant for VIF calculation
        X_const = sm.add_constant(X_clean)
        
        while len(features) > 1:
            try:
                vif_values = [variance_inflation_factor(X_const.values, i) 
                             for i in range(1, len(features) + 1)]
                max_vif_idx = np.argmax(vif_values)
                max_vif = vif_values[max_vif_idx]
                
                if max_vif > self.config.max_vif:
                    removed = features.pop(max_vif_idx)
                    self.removed_features_[removed] = f'high_vif_{max_vif:.1f}'
                    X_const = sm.add_constant(X_clean[features])
                else:
                    break
            except Exception as e:
                logger.warning(f"VIF calculation error: {e}")
                break
        
        return features
    
    def _scale_new_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numeric features."""
        result = df.copy()
        
        # Get newly created features (in metadata)
        new_features = [f for f in self.feature_metadata_.keys() if f in result.columns]
        numeric_new = [f for f in new_features 
                      if result[f].dtype in ['int64', 'float64', 'int32', 'float32']]
        
        for col in numeric_new:
            if col not in self.scalers_:
                scaler = StandardScaler()
                valid_data = result[col].replace([np.inf, -np.inf], np.nan).dropna()
                if len(valid_data) > 0:
                    scaler.fit(valid_data.values.reshape(-1, 1))
                    self.scalers_[col] = scaler
            
            if col in self.scalers_:
                values = result[col].replace([np.inf, -np.inf], np.nan)
                mask = ~values.isna()
                if mask.any():
                    result.loc[mask, f'{col}_scaled'] = self.scalers_[col].transform(
                        values[mask].values.reshape(-1, 1)
                    ).flatten()
        
        return result


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_econometric_features(df: pd.DataFrame,
                                 complexity: str = 'standard',
                                 gender_col: str = 'GENDER',
                                 wage_col: str = 'LOG_WAGE',
                                 female_code: int = 2,
                                 select_features: bool = True) -> Tuple[pd.DataFrame, List[str]]:
    """
    Convenience function to create econometric features.
    
    Parameters
    ----------
    df : DataFrame
        Input data
    complexity : str
        'minimal', 'standard', 'comprehensive', or 'experimental'
    gender_col : str
        Gender column name
    wage_col : str
        Wage column name
    female_code : int
        Code for female
    select_features : bool
        Whether to run feature selection
        
    Returns
    -------
    Tuple of (DataFrame with features, list of selected feature names)
    """
    complexity_map = {
        'minimal': FeatureComplexity.MINIMAL,
        'standard': FeatureComplexity.STANDARD,
        'comprehensive': FeatureComplexity.COMPREHENSIVE,
        'experimental': FeatureComplexity.EXPERIMENTAL
    }
    
    config = FeatureConfig(complexity=complexity_map.get(complexity, FeatureComplexity.STANDARD))
    
    engineer = EconometricFeatureEngineer(
        config=config,
        gender_col=gender_col,
        wage_col=wage_col,
        female_code=female_code
    )
    
    result = engineer.fit_transform(df)
    
    if select_features and wage_col in df.columns:
        selected = engineer.select_features(result, df[wage_col])
    else:
        selected = list(engineer.feature_metadata_.keys())
    
    return result, selected


def get_feature_groups() -> Dict[str, List[str]]:
    """
    Get feature groups for hierarchical feature selection.
    
    Returns dict mapping group name to list of feature name patterns.
    """
    return {
        'distributional': [
            'within_gender_percentile', 'distance_from_gender_median',
            'wage_zone_*', '*_standardized'
        ],
        'segregation': [
            'occ_female_share', '*_dominated_occ', 'integrated_occ',
            'occ_segregation_score', 'within_occ_wage_gap', 'occupation_wage_premium'
        ],
        'propensity': [
            'propensity_female', 'gender_typicality', 'ipw_female', 'common_support'
        ],
        'counterfactual': [
            '*_gap_from_gender_mean', '*_vs_opposite_gender', 'counterfactual_wage_adjustment'
        ],
        'glass_ceiling': [
            'ceiling_proximity', 'above_median', 'female_ceiling_zone', 'distance_to_ceiling'
        ],
        'selection': [
            'mills_ratio_proxy'
        ],
        'sorting': [
            'expected_wage_for_occ', 'wage_vs_occ_expected', 'gender_occ_mismatch'
        ],
        'rif': [
            'rif_q*', 'rif_spread'
        ],
        'temporal': [
            'trend', 'post_*', 'trend_x_female', 'gap_trend_exposure'
        ]
    }
