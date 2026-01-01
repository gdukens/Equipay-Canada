"""
Unit Tests for EquiPay Canada
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline import LFSDataPipeline
from src.feature_engineering import FeatureEngineer
from src.models import SalaryPredictor, WageGapModel
from src.analysis import PayEquityAnalyzer
from src.fairness import FairnessAnalyzer
from src.utils import format_currency, format_percentage, calculate_confidence_interval


class TestDataPipeline:
    """Tests for data pipeline"""
    
    def test_generate_synthetic_data(self):
        """Test synthetic data generation"""
        pipeline = LFSDataPipeline()
        df = pipeline.generate_synthetic_data(n_samples=1000)
        
        assert len(df) == 1000
        assert 'HRLYEARN' in df.columns
        # Check for GENDER column (standard) or SEX (legacy)
        assert 'GENDER' in df.columns or 'SEX' in df.columns
        gender_col = 'GENDER' if 'GENDER' in df.columns else 'SEX'
        assert df[gender_col].isin([1, 2]).all()
        
    def test_clean_data(self):
        """Test data cleaning"""
        pipeline = LFSDataPipeline()
        df = pipeline.generate_synthetic_data(n_samples=1000)
        df_clean = pipeline.clean_data(df)
        
        # Should have valid wages
        assert df_clean['HRLYEARN'].notna().all()
        assert (df_clean['HRLYEARN'] > 0).all()
        
    def test_derived_features(self):
        """Test derived feature creation"""
        pipeline = LFSDataPipeline()
        df = pipeline.generate_synthetic_data(n_samples=1000)
        df = pipeline.clean_data(df)
        df = pipeline.create_derived_features(df)
        
        assert 'IS_FEMALE' in df.columns
        assert 'LOG_HRLYEARN' in df.columns


class TestFeatureEngineering:
    """Tests for feature engineering"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data"""
        pipeline = LFSDataPipeline()
        df = pipeline.generate_synthetic_data(n_samples=500)
        df = pipeline.clean_data(df)
        df = pipeline.create_derived_features(df)
        return df
    
    def test_fit_transform(self, sample_data):
        """Test feature transformation"""
        fe = FeatureEngineer()
        X, y = fe.fit_transform(sample_data)
        
        assert X is not None
        assert y is not None
        assert len(X) == len(y)
        assert not np.isnan(X).any()
        
    def test_feature_names(self, sample_data):
        """Test feature name extraction"""
        fe = FeatureEngineer()
        fe.fit_transform(sample_data)
        
        names = fe.get_feature_names()
        assert names is not None
        assert len(names) > 0


class TestModels:
    """Tests for ML models"""
    
    @pytest.fixture
    def training_data(self):
        """Create training data"""
        pipeline = LFSDataPipeline()
        df = pipeline.generate_synthetic_data(n_samples=1000)
        df = pipeline.clean_data(df)
        df = pipeline.create_derived_features(df)
        
        fe = FeatureEngineer()
        X, y = fe.fit_transform(df)
        return X, y, fe.get_feature_names()
    
    def test_model_training(self, training_data):
        """Test model training"""
        X, y, feature_names = training_data
        
        predictor = SalaryPredictor()
        metrics = predictor.train(X, y, feature_names)
        
        assert 'ensemble' in metrics
        assert metrics['ensemble']['r2'] > 0
        
    def test_prediction(self, training_data):
        """Test predictions"""
        X, y, feature_names = training_data
        
        predictor = SalaryPredictor()
        predictor.train(X, y, feature_names)
        
        predictions = predictor.predict(X[:10])
        
        assert len(predictions) == 10
        assert (predictions > 0).all()


class TestPayEquityAnalysis:
    """Tests for pay equity analysis"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data"""
        pipeline = LFSDataPipeline()
        df = pipeline.generate_synthetic_data(n_samples=1000)
        df = pipeline.clean_data(df)
        df = pipeline.create_derived_features(df)
        return df
    
    def test_raw_wage_gap(self, sample_data):
        """Test raw wage gap calculation"""
        analyzer = PayEquityAnalyzer(sample_data)
        results = analyzer.compute_raw_wage_gap()
        
        assert 'male' in results
        assert 'female' in results
        assert 'raw_gap' in results
        assert 'mean_gap_pct' in results['raw_gap']
        
    def test_quantile_analysis(self, sample_data):
        """Test quantile analysis"""
        analyzer = PayEquityAnalyzer(sample_data)
        results = analyzer.quantile_analysis()
        
        assert 'p10' in results
        assert 'p50' in results
        assert 'p90' in results
        assert 'glass_ceiling_effect' in results


class TestFairnessAnalysis:
    """Tests for fairness analysis"""
    
    @pytest.fixture
    def prediction_data(self):
        """Create prediction data"""
        pipeline = LFSDataPipeline()
        df = pipeline.generate_synthetic_data(n_samples=1000)
        df = pipeline.clean_data(df)
        df = pipeline.create_derived_features(df)
        
        # Mock predictions (with slight bias)
        np.random.seed(42)
        y_pred = df['HRLYEARN'].values + np.random.normal(0, 2, len(df))
        
        return df, y_pred
    
    def test_wage_gap_analysis(self, prediction_data):
        """Test wage gap analysis"""
        df, y_pred = prediction_data
        
        analyzer = FairnessAnalyzer()
        results = analyzer.analyze_wage_gap(df, y_pred)
        
        assert 'actual' in results
        assert 'predicted' in results
        assert 'bias_analysis' in results
        
    def test_fairness_metrics(self, prediction_data):
        """Test fairness metrics computation"""
        df, y_pred = prediction_data
        
        # Get gender column (GENDER or SEX)
        gender_col = 'GENDER' if 'GENDER' in df.columns else 'SEX'
        
        analyzer = FairnessAnalyzer()
        results = analyzer.compute_fairness_metrics(
            y_true=df['HRLYEARN'].values,
            y_pred=y_pred,
            sensitive_features=df[gender_col]
        )
        
        assert 'prediction_parity' in results


class TestUtils:
    """Tests for utility functions"""
    
    def test_format_currency(self):
        """Test currency formatting"""
        assert format_currency(1234.56) == "$1,234.56"
        assert format_currency(0) == "$0.00"
        
    def test_format_percentage(self):
        """Test percentage formatting"""
        assert format_percentage(15.5) == "15.5%"
        assert format_percentage(15.567, decimals=2) == "15.57%"
        
    def test_confidence_interval(self):
        """Test confidence interval calculation"""
        data = np.array([10, 12, 14, 11, 13, 15, 12, 14])
        ci_lower, ci_upper = calculate_confidence_interval(data, confidence=0.95)
        
        mean = np.mean(data)
        assert ci_lower < mean < ci_upper


class TestIntegration:
    """Integration tests for full pipeline"""
    
    def test_full_pipeline(self):
        """Test complete pipeline execution"""
        # Data
        pipeline = LFSDataPipeline()
        df = pipeline.generate_synthetic_data(n_samples=500)
        df = pipeline.clean_data(df)
        df = pipeline.create_derived_features(df)
        
        # Features
        fe = FeatureEngineer()
        X, y = fe.fit_transform(df)
        
        # Model
        predictor = SalaryPredictor()
        metrics = predictor.train(X, y, fe.get_feature_names())
        
        # Predictions
        y_pred = predictor.predict(X)
        
        # Analysis
        pay_analyzer = PayEquityAnalyzer(df)
        gap = pay_analyzer.compute_raw_wage_gap()
        
        # Fairness
        fair_analyzer = FairnessAnalyzer()
        fairness = fair_analyzer.analyze_wage_gap(df, y_pred)
        
        # Assertions
        assert metrics['ensemble']['r2'] > 0
        assert 'mean_gap_pct' in gap['raw_gap']
        assert 'actual' in fairness


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
