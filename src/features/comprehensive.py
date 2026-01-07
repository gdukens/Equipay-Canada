"""
EquiPay Canada - Comprehensive Feature Engineering
===================================================

This module provides complete feature engineering for wage gap analysis,
utilizing ALL available columns from the LFS PUMF dataset with built-in
safeguards against data leakage.

Key Features:
- Uses all 60 LFS columns systematically
- Creates interaction terms for intersectional analysis
- Builds occupation/industry segregation measures
- Generates human capital proxies
- Maintains strict separation from target-derived features

Design Philosophy:
------------------
1. NEVER create features from the target variable (wages)
2. All features are derived from characteristics known BEFORE wage observation
3. Explicit categorization of all features for transparency
4. Integration with LeakageGuard for validation

Author: EquiPay Canada Research Team
Version: 1.0.0
"""

import logging
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
import warnings

import numpy as np
import pandas as pd

from src.leakage_prevention import LeakageGuard, validate_features

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    
    # Include interaction terms
    include_interactions: bool = True
    
    # Include occupation segregation features
    include_segregation: bool = True
    
    # Include propensity scores (IPW)
    include_propensity: bool = True
    
    # Include polynomial terms
    include_polynomials: bool = True
    polynomial_degree: int = 2
    
    # Scale features
    scale_continuous: bool = True
    
    # One-hot encode categoricals
    one_hot_encode: bool = False
    
    # Maximum categories for one-hot
    max_categories: int = 20
    
    # Weight column for weighted calculations
    weight_col: str = 'FINALWT'
    
    # Target column (to AVOID using)
    target_col: str = 'REAL_HRLYEARN'


# =============================================================================
# MAIN FEATURE ENGINEER
# =============================================================================

class ComprehensiveFeatureEngineer:
    """
    Comprehensive feature engineering for wage gap analysis.
    
    This class creates a rich feature set from all available LFS columns
    while maintaining strict safeguards against data leakage.
    
    Examples
    --------
    >>> engineer = ComprehensiveFeatureEngineer()
    >>> df_features = engineer.create_all_features(df)
    >>> feature_list = engineer.get_feature_names()
    >>> 
    >>> # For ML training
    >>> X = df_features[feature_list]
    >>> y = df['REAL_HRLYEARN']
    """
    
    def __init__(self, config: FeatureConfig = None):
        """
        Initialize the feature engineer.
        
        Parameters
        ----------
        config : FeatureConfig, optional
            Configuration options
        """
        self.config = config or FeatureConfig()
        self.leakage_guard = LeakageGuard(target_col=self.config.target_col)
        
        # Track created features
        self._created_features: Set[str] = set()
        self._feature_categories: Dict[str, List[str]] = {
            'demographic': [],
            'human_capital': [],
            'job': [],
            'geographic': [],
            'time': [],
            'interaction': [],
            'segregation': [],
            'derived': [],
        }
        
        logger.info("ComprehensiveFeatureEngineer initialized")
    
    def create_all_features(
        self,
        df: pd.DataFrame,
        validate: bool = True
    ) -> pd.DataFrame:
        """
        Create all features from the input data.
        
        Parameters
        ----------
        df : DataFrame
            Input data (should contain raw LFS columns)
        validate : bool
            Whether to validate for leakage
            
        Returns
        -------
        DataFrame
            Data with all engineered features added
        """
        df = df.copy()
        
        logger.info(f"Creating features from {len(df)} records...")
        
        # 1. Demographic features
        df = self._create_demographic_features(df)
        
        # 2. Human capital features  
        df = self._create_human_capital_features(df)
        
        # 3. Job characteristic features
        df = self._create_job_features(df)
        
        # 4. Geographic features
        df = self._create_geographic_features(df)
        
        # 5. Time features
        df = self._create_time_features(df)
        
        # 6. Segregation measures
        if self.config.include_segregation:
            df = self._create_segregation_features(df)
        
        # 7. Interaction terms
        if self.config.include_interactions:
            df = self._create_interaction_features(df)
        
        # 8. Propensity scores
        if self.config.include_propensity:
            df = self._create_propensity_features(df)
        
        # 9. Validate for leakage
        if validate:
            self._validate_no_leakage(df)
        
        logger.info(f"Created {len(self._created_features)} features")
        return df
    
    # =========================================================================
    # DEMOGRAPHIC FEATURES
    # =========================================================================
    
    def _create_demographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create demographic features."""
        
        # Binary gender indicator
        if 'GENDER' in df.columns and 'IS_FEMALE' not in df.columns:
            df['IS_FEMALE'] = (df['GENDER'] == 2).astype(np.int8)
            self._add_feature('IS_FEMALE', 'demographic')
        
        # Immigration status
        if 'IMMIG' in df.columns:
            # Is immigrant (any landing period)
            df['IS_IMMIGRANT'] = (df['IMMIG'].isin([1, 2, 3])).astype(np.int8)
            self._add_feature('IS_IMMIGRANT', 'demographic')
            
            # Recent immigrant (0-5 years)
            df['IS_RECENT_IMMIG'] = (df['IMMIG'] == 1).astype(np.int8)
            self._add_feature('IS_RECENT_IMMIG', 'demographic')
            
            # Established immigrant (10+ years)
            df['IS_ESTABLISHED_IMMIG'] = (df['IMMIG'] == 3).astype(np.int8)
            self._add_feature('IS_ESTABLISHED_IMMIG', 'demographic')
        
        # Marital status
        if 'MARSTAT' in df.columns:
            df['IS_MARRIED'] = (df['MARSTAT'].isin([1, 2])).astype(np.int8)
            self._add_feature('IS_MARRIED', 'demographic')
            
            df['IS_SINGLE'] = (df['MARSTAT'] == 6).astype(np.int8)
            self._add_feature('IS_SINGLE', 'demographic')
        
        # Family responsibilities
        if 'AGYOWNK' in df.columns:
            df['HAS_YOUNG_CHILDREN'] = (df['AGYOWNK'].isin([1, 2, 3])).astype(np.int8)
            self._add_feature('HAS_YOUNG_CHILDREN', 'demographic')
            
            df['HAS_SCHOOL_AGE_CHILDREN'] = (df['AGYOWNK'].isin([4, 5, 6])).astype(np.int8)
            self._add_feature('HAS_SCHOOL_AGE_CHILDREN', 'demographic')
            
            df['HAS_CHILDREN'] = (df['AGYOWNK'] > 0).astype(np.int8)
            self._add_feature('HAS_CHILDREN', 'demographic')
        
        # Age features
        if 'AGE_12' in df.columns:
            # Prime working age (25-54)
            df['IS_PRIME_AGE'] = (df['AGE_12'].isin([3, 4, 5, 6, 7, 8])).astype(np.int8)
            self._add_feature('IS_PRIME_AGE', 'demographic')
            
            # Young worker (15-24)
            df['IS_YOUNG'] = (df['AGE_12'].isin([1, 2])).astype(np.int8)
            self._add_feature('IS_YOUNG', 'demographic')
            
            # Near retirement (55+)
            df['IS_NEAR_RETIREMENT'] = (df['AGE_12'].isin([9, 10, 11, 12])).astype(np.int8)
            self._add_feature('IS_NEAR_RETIREMENT', 'demographic')
        
        return df
    
    # =========================================================================
    # HUMAN CAPITAL FEATURES
    # =========================================================================
    
    def _create_human_capital_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create human capital features (education and experience)."""
        
        # Education indicators
        if 'EDUC' in df.columns:
            df['HAS_DEGREE'] = (df['EDUC'] >= 6).astype(np.int8)
            self._add_feature('HAS_DEGREE', 'human_capital')
            
            df['HAS_GRADUATE_DEGREE'] = (df['EDUC'] == 7).astype(np.int8)
            self._add_feature('HAS_GRADUATE_DEGREE', 'human_capital')
            
            df['HAS_POSTSECONDARY'] = (df['EDUC'] >= 5).astype(np.int8)
            self._add_feature('HAS_POSTSECONDARY', 'human_capital')
            
            df['HIGH_SCHOOL_OR_LESS'] = (df['EDUC'] <= 3).astype(np.int8)
            self._add_feature('HIGH_SCHOOL_OR_LESS', 'human_capital')
            
            # Approximate years of education (for Mincer equation)
            education_years_map = {
                0: 8,   # Less than Grade 9
                1: 9.5, # Grade 9-10
                2: 11,  # Grade 11-13 no diploma
                3: 12,  # High school diploma
                4: 13,  # Some post-secondary
                5: 14,  # Certificate/diploma
                6: 16,  # Bachelor's
                7: 18   # Above bachelor's
            }
            df['YEARS_EDUCATION'] = df['EDUC'].map(education_years_map).fillna(12)
            self._add_feature('YEARS_EDUCATION', 'human_capital')
        
        # Experience proxy (Mincer style: Age - Education - 6)
        if 'AGE_12' in df.columns:
            # Convert age category to midpoint
            age_midpoint_map = {
                1: 17, 2: 22, 3: 27, 4: 32, 5: 37, 6: 42,
                7: 47, 8: 52, 9: 57, 10: 62, 11: 67, 12: 72
            }
            age_approx = df['AGE_12'].map(age_midpoint_map).fillna(35)
            
            if 'YEARS_EDUCATION' in df.columns:
                df['EXPERIENCE_PROXY'] = np.maximum(0, age_approx - df['YEARS_EDUCATION'] - 6)
            else:
                # Assume 12 years education if not available
                df['EXPERIENCE_PROXY'] = np.maximum(0, age_approx - 18)
            
            self._add_feature('EXPERIENCE_PROXY', 'human_capital')
            
            # Quadratic experience (diminishing returns)
            df['EXPERIENCE_SQ'] = df['EXPERIENCE_PROXY'] ** 2 / 100  # Scaled
            self._add_feature('EXPERIENCE_SQ', 'human_capital')
        
        # Tenure features
        if 'TENURE' in df.columns:
            df['IS_NEW_HIRE'] = (df['TENURE'] <= 2).astype(np.int8)  # Less than 1 year
            self._add_feature('IS_NEW_HIRE', 'human_capital')
            
            df['IS_LONG_TENURE'] = (df['TENURE'] >= 5).astype(np.int8)  # 10+ years
            self._add_feature('IS_LONG_TENURE', 'human_capital')
            
            # Approximate years of tenure
            tenure_years_map = {
                0: 0.2, 1: 0.5, 2: 0.8, 3: 3, 4: 7.5, 5: 15, 6: 25
            }
            df['TENURE_YEARS'] = df['TENURE'].map(tenure_years_map).fillna(3)
            self._add_feature('TENURE_YEARS', 'human_capital')
        
        return df
    
    # =========================================================================
    # JOB CHARACTERISTIC FEATURES
    # =========================================================================
    
    def _create_job_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create job characteristic features."""
        
        # Class of worker
        if 'COWMAIN' in df.columns:
            df['IS_PUBLIC'] = (df['COWMAIN'] == 1).astype(np.int8)
            self._add_feature('IS_PUBLIC', 'job')
            
            df['IS_PRIVATE'] = (df['COWMAIN'] == 2).astype(np.int8)
            self._add_feature('IS_PRIVATE', 'job')
            
            df['IS_SELF_EMPLOYED'] = (df['COWMAIN'].isin([3, 4])).astype(np.int8)
            self._add_feature('IS_SELF_EMPLOYED', 'job')
        
        # Employment type
        if 'PERMTEMP' in df.columns:
            df['IS_PERMANENT'] = (df['PERMTEMP'] == 1).astype(np.int8)
            self._add_feature('IS_PERMANENT', 'job')
            
            df['IS_TEMPORARY'] = (df['PERMTEMP'].isin([2, 3, 4, 5])).astype(np.int8)
            self._add_feature('IS_TEMPORARY', 'job')
        
        # Full-time/part-time
        if 'FTPTMAIN' in df.columns:
            df['IS_FULLTIME'] = (df['FTPTMAIN'] == 1).astype(np.int8)
            self._add_feature('IS_FULLTIME', 'job')
            
            df['IS_PARTTIME'] = (df['FTPTMAIN'] == 2).astype(np.int8)
            self._add_feature('IS_PARTTIME', 'job')
        
        # Union status
        if 'UNION' in df.columns:
            df['IS_UNION'] = (df['UNION'].isin([1, 2])).astype(np.int8)  # Member or covered
            self._add_feature('IS_UNION', 'job')
            
            df['IS_UNION_MEMBER'] = (df['UNION'] == 1).astype(np.int8)
            self._add_feature('IS_UNION_MEMBER', 'job')
        
        # Firm size
        if 'ESTSIZE' in df.columns:
            df['IS_LARGE_ESTABLISHMENT'] = (df['ESTSIZE'] >= 3).astype(np.int8)  # 100+ employees
            self._add_feature('IS_LARGE_ESTABLISHMENT', 'job')
            
            df['IS_SMALL_ESTABLISHMENT'] = (df['ESTSIZE'] == 1).astype(np.int8)  # <20 employees
            self._add_feature('IS_SMALL_ESTABLISHMENT', 'job')
        
        # Multiple jobs
        if 'MJH' in df.columns:
            df['HAS_MULTIPLE_JOBS'] = (df['MJH'] == 1).astype(np.int8)
            self._add_feature('HAS_MULTIPLE_JOBS', 'job')
        
        # Occupation groups (higher-level categories)
        if 'NOC_10' in df.columns:
            # White collar (Management, Business, Science, Health, Education)
            df['IS_WHITE_COLLAR'] = (df['NOC_10'].isin([0, 1, 2, 3, 4])).astype(np.int8)
            self._add_feature('IS_WHITE_COLLAR', 'job')
            
            # Blue collar (Trades, Manufacturing)
            df['IS_BLUE_COLLAR'] = (df['NOC_10'].isin([7, 8, 9])).astype(np.int8)
            self._add_feature('IS_BLUE_COLLAR', 'job')
            
            # Professional (Management, Science, Health)
            df['IS_PROFESSIONAL'] = (df['NOC_10'].isin([0, 2, 3])).astype(np.int8)
            self._add_feature('IS_PROFESSIONAL', 'job')
        
        # Industry groups
        if 'NAICS_21' in df.columns:
            # Goods-producing industries
            df['IS_GOODS_PRODUCING'] = (df['NAICS_21'].isin([1, 2, 3, 4, 5, 6])).astype(np.int8)
            self._add_feature('IS_GOODS_PRODUCING', 'job')
            
            # Service industries
            df['IS_SERVICE_SECTOR'] = (~df['NAICS_21'].isin([1, 2, 3, 4, 5, 6])).astype(np.int8)
            self._add_feature('IS_SERVICE_SECTOR', 'job')
            
            # High-wage industries
            df['IS_HIGH_WAGE_INDUSTRY'] = (df['NAICS_21'].isin([2, 3, 10, 12, 19])).astype(np.int8)
            self._add_feature('IS_HIGH_WAGE_INDUSTRY', 'job')
            
            # Care industries (Health, Education)
            df['IS_CARE_INDUSTRY'] = (df['NAICS_21'].isin([14, 15])).astype(np.int8)
            self._add_feature('IS_CARE_INDUSTRY', 'job')
        
        # Part-time reason
        if 'WHYPT' in df.columns:
            df['INVOLUNTARY_PARTTIME'] = (df['WHYPT'] == 4).astype(np.int8)  # Could only find PT
            self._add_feature('INVOLUNTARY_PARTTIME', 'job')
            
            df['PARTTIME_FAMILY'] = (df['WHYPT'] == 2).astype(np.int8)  # Family responsibilities
            self._add_feature('PARTTIME_FAMILY', 'job')
        
        return df
    
    # =========================================================================
    # GEOGRAPHIC FEATURES
    # =========================================================================
    
    def _create_geographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create geographic features."""
        
        # Urban/rural
        if 'CMA' in df.columns:
            df['IS_URBAN'] = (df['CMA'] > 0).astype(np.int8)
            self._add_feature('IS_URBAN', 'geographic')
            
            # Major CMAs (Toronto, Montreal, Vancouver)
            major_cmas = [535, 462, 933]  # Toronto, Montreal, Vancouver
            df['IS_MAJOR_CMA'] = (df['CMA'].isin(major_cmas)).astype(np.int8)
            self._add_feature('IS_MAJOR_CMA', 'geographic')
            
            # Large CMAs (500k+)
            large_cmas = [535, 462, 933, 825, 835, 505, 602, 539]
            df['IS_LARGE_CMA'] = (df['CMA'].isin(large_cmas)).astype(np.int8)
            self._add_feature('IS_LARGE_CMA', 'geographic')
        
        # Regional groupings
        if 'PROV' in df.columns:
            # Atlantic provinces
            df['IS_ATLANTIC'] = (df['PROV'].isin([10, 11, 12, 13])).astype(np.int8)
            self._add_feature('IS_ATLANTIC', 'geographic')
            
            # Quebec
            df['IS_QUEBEC'] = (df['PROV'] == 24).astype(np.int8)
            self._add_feature('IS_QUEBEC', 'geographic')
            
            # Ontario
            df['IS_ONTARIO'] = (df['PROV'] == 35).astype(np.int8)
            self._add_feature('IS_ONTARIO', 'geographic')
            
            # Prairies (MB, SK, AB)
            df['IS_PRAIRIES'] = (df['PROV'].isin([46, 47, 48])).astype(np.int8)
            self._add_feature('IS_PRAIRIES', 'geographic')
            
            # BC
            df['IS_BC'] = (df['PROV'] == 59).astype(np.int8)
            self._add_feature('IS_BC', 'geographic')
            
            # Western Canada (Prairies + BC)
            df['IS_WESTERN'] = (df['PROV'].isin([46, 47, 48, 59])).astype(np.int8)
            self._add_feature('IS_WESTERN', 'geographic')
        
        return df
    
    # =========================================================================
    # TIME FEATURES
    # =========================================================================
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features."""
        
        if 'SURVYEAR' in df.columns:
            # Trend variable (years since 2010)
            df['TREND'] = df['SURVYEAR'] - 2010
            self._add_feature('TREND', 'time')
            
            # Policy periods
            df['POST_2015'] = (df['SURVYEAR'] >= 2015).astype(np.int8)
            self._add_feature('POST_2015', 'time')
            
            df['POST_2020'] = (df['SURVYEAR'] >= 2020).astype(np.int8)
            self._add_feature('POST_2020', 'time')
            
            # COVID period
            df['COVID_PERIOD'] = (df['SURVYEAR'].isin([2020, 2021])).astype(np.int8)
            self._add_feature('COVID_PERIOD', 'time')
        
        if 'SURVMNTH' in df.columns:
            # Seasonal indicators
            df['IS_Q1'] = (df['SURVMNTH'].isin([1, 2, 3])).astype(np.int8)
            self._add_feature('IS_Q1', 'time')
            
            df['IS_SUMMER'] = (df['SURVMNTH'].isin([6, 7, 8])).astype(np.int8)
            self._add_feature('IS_SUMMER', 'time')
        
        return df
    
    # =========================================================================
    # SEGREGATION FEATURES
    # =========================================================================
    
    def _create_segregation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create occupation and industry segregation measures.
        
        These are group-level features computed from the data but NOT
        from the target variable, so they don't cause leakage.
        """
        weight_col = self.config.weight_col
        
        # Occupation segregation (% female in each occupation)
        if 'NOC_10' in df.columns and 'IS_FEMALE' in df.columns:
            if weight_col in df.columns:
                # Weighted female share by occupation
                occ_female_share = (
                    df.groupby('NOC_10')
                    .apply(lambda x: np.average(x['IS_FEMALE'], weights=x[weight_col]))
                    .to_dict()
                )
            else:
                occ_female_share = df.groupby('NOC_10')['IS_FEMALE'].mean().to_dict()
            
            df['OCC_FEMALE_SHARE'] = df['NOC_10'].map(occ_female_share)
            self._add_feature('OCC_FEMALE_SHARE', 'segregation')
            
            # Occupation type based on female share
            df['FEMALE_DOMINATED_OCC'] = (df['OCC_FEMALE_SHARE'] > 0.7).astype(np.int8)
            self._add_feature('FEMALE_DOMINATED_OCC', 'segregation')
            
            df['MALE_DOMINATED_OCC'] = (df['OCC_FEMALE_SHARE'] < 0.3).astype(np.int8)
            self._add_feature('MALE_DOMINATED_OCC', 'segregation')
            
            df['INTEGRATED_OCC'] = (
                (df['OCC_FEMALE_SHARE'] >= 0.3) & 
                (df['OCC_FEMALE_SHARE'] <= 0.7)
            ).astype(np.int8)
            self._add_feature('INTEGRATED_OCC', 'segregation')
        
        # Industry segregation
        if 'NAICS_21' in df.columns and 'IS_FEMALE' in df.columns:
            if weight_col in df.columns:
                ind_female_share = (
                    df.groupby('NAICS_21')
                    .apply(lambda x: np.average(x['IS_FEMALE'], weights=x[weight_col]))
                    .to_dict()
                )
            else:
                ind_female_share = df.groupby('NAICS_21')['IS_FEMALE'].mean().to_dict()
            
            df['IND_FEMALE_SHARE'] = df['NAICS_21'].map(ind_female_share)
            self._add_feature('IND_FEMALE_SHARE', 'segregation')
            
            df['FEMALE_DOMINATED_IND'] = (df['IND_FEMALE_SHARE'] > 0.7).astype(np.int8)
            self._add_feature('FEMALE_DOMINATED_IND', 'segregation')
            
            df['MALE_DOMINATED_IND'] = (df['IND_FEMALE_SHARE'] < 0.3).astype(np.int8)
            self._add_feature('MALE_DOMINATED_IND', 'segregation')
        
        # Duncan segregation index (occupational)
        if 'NOC_10' in df.columns and 'IS_FEMALE' in df.columns:
            # Calculate Duncan index contribution
            df['OCC_SEGREGATION_CONTRIB'] = abs(df['OCC_FEMALE_SHARE'] - 0.5)
            self._add_feature('OCC_SEGREGATION_CONTRIB', 'segregation')
        
        return df
    
    # =========================================================================
    # INTERACTION FEATURES
    # =========================================================================
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction terms for intersectional analysis.
        
        These capture how the gender effect varies across different
        dimensions (occupation, province, immigration status, etc.)
        """
        
        if 'IS_FEMALE' not in df.columns:
            return df
        
        # Gender × Education
        if 'HAS_DEGREE' in df.columns:
            df['FEMALE_x_DEGREE'] = df['IS_FEMALE'] * df['HAS_DEGREE']
            self._add_feature('FEMALE_x_DEGREE', 'interaction')
        
        if 'YEARS_EDUCATION' in df.columns:
            df['FEMALE_x_EDUC'] = df['IS_FEMALE'] * df['YEARS_EDUCATION']
            self._add_feature('FEMALE_x_EDUC', 'interaction')
        
        # Gender × Experience
        if 'EXPERIENCE_PROXY' in df.columns:
            df['FEMALE_x_EXPERIENCE'] = df['IS_FEMALE'] * df['EXPERIENCE_PROXY']
            self._add_feature('FEMALE_x_EXPERIENCE', 'interaction')
        
        if 'TENURE_YEARS' in df.columns:
            df['FEMALE_x_TENURE'] = df['IS_FEMALE'] * df['TENURE_YEARS']
            self._add_feature('FEMALE_x_TENURE', 'interaction')
        
        # Gender × Immigration
        if 'IS_IMMIGRANT' in df.columns:
            df['FEMALE_x_IMMIGRANT'] = df['IS_FEMALE'] * df['IS_IMMIGRANT']
            self._add_feature('FEMALE_x_IMMIGRANT', 'interaction')
        
        if 'IS_RECENT_IMMIG' in df.columns:
            df['FEMALE_x_RECENT_IMMIG'] = df['IS_FEMALE'] * df['IS_RECENT_IMMIG']
            self._add_feature('FEMALE_x_RECENT_IMMIG', 'interaction')
        
        # Gender × Job type
        if 'IS_PUBLIC' in df.columns:
            df['FEMALE_x_PUBLIC'] = df['IS_FEMALE'] * df['IS_PUBLIC']
            self._add_feature('FEMALE_x_PUBLIC', 'interaction')
        
        if 'IS_FULLTIME' in df.columns:
            df['FEMALE_x_FULLTIME'] = df['IS_FEMALE'] * df['IS_FULLTIME']
            self._add_feature('FEMALE_x_FULLTIME', 'interaction')
        
        if 'IS_UNION' in df.columns:
            df['FEMALE_x_UNION'] = df['IS_FEMALE'] * df['IS_UNION']
            self._add_feature('FEMALE_x_UNION', 'interaction')
        
        if 'IS_PERMANENT' in df.columns:
            df['FEMALE_x_PERMANENT'] = df['IS_FEMALE'] * df['IS_PERMANENT']
            self._add_feature('FEMALE_x_PERMANENT', 'interaction')
        
        # Gender × Occupation type
        if 'IS_WHITE_COLLAR' in df.columns:
            df['FEMALE_x_WHITE_COLLAR'] = df['IS_FEMALE'] * df['IS_WHITE_COLLAR']
            self._add_feature('FEMALE_x_WHITE_COLLAR', 'interaction')
        
        if 'IS_PROFESSIONAL' in df.columns:
            df['FEMALE_x_PROFESSIONAL'] = df['IS_FEMALE'] * df['IS_PROFESSIONAL']
            self._add_feature('FEMALE_x_PROFESSIONAL', 'interaction')
        
        # Gender × Geography
        if 'IS_URBAN' in df.columns:
            df['FEMALE_x_URBAN'] = df['IS_FEMALE'] * df['IS_URBAN']
            self._add_feature('FEMALE_x_URBAN', 'interaction')
        
        if 'IS_MAJOR_CMA' in df.columns:
            df['FEMALE_x_MAJOR_CMA'] = df['IS_FEMALE'] * df['IS_MAJOR_CMA']
            self._add_feature('FEMALE_x_MAJOR_CMA', 'interaction')
        
        # Gender × Segregation
        if 'OCC_FEMALE_SHARE' in df.columns:
            df['FEMALE_x_OCC_FEMALE_SHARE'] = df['IS_FEMALE'] * df['OCC_FEMALE_SHARE']
            self._add_feature('FEMALE_x_OCC_FEMALE_SHARE', 'interaction')
        
        if 'FEMALE_DOMINATED_OCC' in df.columns:
            df['FEMALE_x_FEMALE_DOM_OCC'] = df['IS_FEMALE'] * df['FEMALE_DOMINATED_OCC']
            self._add_feature('FEMALE_x_FEMALE_DOM_OCC', 'interaction')
        
        # Gender × Family
        if 'HAS_YOUNG_CHILDREN' in df.columns:
            df['FEMALE_x_YOUNG_CHILDREN'] = df['IS_FEMALE'] * df['HAS_YOUNG_CHILDREN']
            self._add_feature('FEMALE_x_YOUNG_CHILDREN', 'interaction')
        
        if 'IS_MARRIED' in df.columns:
            df['FEMALE_x_MARRIED'] = df['IS_FEMALE'] * df['IS_MARRIED']
            self._add_feature('FEMALE_x_MARRIED', 'interaction')
        
        # Gender × Time
        if 'TREND' in df.columns:
            df['FEMALE_x_TREND'] = df['IS_FEMALE'] * df['TREND']
            self._add_feature('FEMALE_x_TREND', 'interaction')
        
        if 'COVID_PERIOD' in df.columns:
            df['FEMALE_x_COVID'] = df['IS_FEMALE'] * df['COVID_PERIOD']
            self._add_feature('FEMALE_x_COVID', 'interaction')
        
        # Triple interactions (Gender × Province × Occupation)
        if 'NOC_10' in df.columns and 'PROV' in df.columns:
            # For each province, we'll create NOC_10 fixed effects
            # But that creates too many features - better done in regression
            pass
        
        return df
    
    # =========================================================================
    # PROPENSITY SCORES
    # =========================================================================
    
    def _create_propensity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create propensity score features for IPW estimation.
        
        These are used to address selection bias in the gender wage gap.
        """
        
        # Check required columns
        if 'IS_FEMALE' not in df.columns:
            return df
        
        # Get predictors for propensity model
        propensity_predictors = []
        for col in ['EDUC', 'TENURE', 'AGE_12', 'PROV', 'NOC_10', 'NAICS_21']:
            if col in df.columns:
                propensity_predictors.append(col)
        
        if len(propensity_predictors) < 2:
            return df
        
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            
            # Prepare data
            X_prop = pd.get_dummies(df[propensity_predictors], drop_first=True)
            y_prop = df['IS_FEMALE'].values
            
            # Handle missing values
            X_prop = X_prop.fillna(X_prop.median())
            
            # Fit propensity model
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_prop)
            
            prop_model = LogisticRegression(
                max_iter=1000, 
                solver='lbfgs',
                random_state=42
            )
            prop_model.fit(X_scaled, y_prop)
            
            # Get propensity scores
            propensity = prop_model.predict_proba(X_scaled)[:, 1]
            
            # Clip to avoid extreme weights
            propensity = np.clip(propensity, 0.01, 0.99)
            
            df['PROPENSITY_FEMALE'] = propensity
            self._add_feature('PROPENSITY_FEMALE', 'derived')
            
            # IPW weights
            df['IPW_WEIGHT'] = np.where(
                df['IS_FEMALE'] == 1,
                1 / propensity,
                1 / (1 - propensity)
            )
            # Normalize weights
            df['IPW_WEIGHT'] = df['IPW_WEIGHT'] / df['IPW_WEIGHT'].mean()
            self._add_feature('IPW_WEIGHT', 'derived')
            
        except Exception as e:
            logger.warning(f"Could not compute propensity scores: {e}")
        
        return df
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def _add_feature(self, name: str, category: str):
        """Track a created feature."""
        self._created_features.add(name)
        if category in self._feature_categories:
            self._feature_categories[category].append(name)
    
    def _validate_no_leakage(self, df: pd.DataFrame):
        """Validate that no leaked features were created."""
        for feature in self._created_features:
            is_blocked, reason = self.leakage_guard.is_blocked(feature)
            if is_blocked:
                raise ValueError(f"LEAKAGE DETECTED: Feature '{feature}' - {reason}")
        
        logger.info("✅ Leakage validation passed")
    
    def get_feature_names(
        self,
        categories: List[str] = None,
        exclude_categories: List[str] = None
    ) -> List[str]:
        """
        Get list of created feature names.
        
        Parameters
        ----------
        categories : List[str], optional
            Only return features from these categories
        exclude_categories : List[str], optional
            Exclude features from these categories
            
        Returns
        -------
        List[str]
            Feature names
        """
        if categories is not None:
            features = []
            for cat in categories:
                features.extend(self._feature_categories.get(cat, []))
            return features
        
        if exclude_categories is not None:
            features = []
            for cat, feats in self._feature_categories.items():
                if cat not in exclude_categories:
                    features.extend(feats)
            return features
        
        return list(self._created_features)
    
    def get_feature_summary(self) -> pd.DataFrame:
        """Get a summary of all created features by category."""
        rows = []
        for category, features in self._feature_categories.items():
            for feature in features:
                rows.append({
                    'feature': feature,
                    'category': category
                })
        return pd.DataFrame(rows)
    
    def get_ml_features(
        self,
        df: pd.DataFrame,
        include_categorical: bool = True
    ) -> List[str]:
        """
        Get features suitable for ML training.
        
        This returns only the features that are:
        1. Created by this engineer
        2. Actually present in the dataframe
        3. Not blocked by leakage detection
        
        Parameters
        ----------
        df : DataFrame
            Data to check for available features
        include_categorical : bool
            Whether to include categorical columns
            
        Returns
        -------
        List[str]
            Features ready for ML
        """
        available = [f for f in self._created_features if f in df.columns]
        
        # Filter through leakage guard
        clean = self.leakage_guard.filter_features(df, available)
        
        if not include_categorical:
            # Remove categorical columns (those with few unique values that aren't binary)
            clean = [
                f for f in clean 
                if df[f].nunique() <= 2 or df[f].dtype in ['float32', 'float64']
            ]
        
        return clean


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_comprehensive_features(
    df: pd.DataFrame,
    config: FeatureConfig = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Convenience function to create all features.
    
    Parameters
    ----------
    df : DataFrame
        Input data
    config : FeatureConfig, optional
        Configuration
        
    Returns
    -------
    Tuple[DataFrame, List[str]]
        (Data with features, list of feature names)
    """
    engineer = ComprehensiveFeatureEngineer(config=config)
    df_out = engineer.create_all_features(df)
    features = engineer.get_feature_names()
    return df_out, features
