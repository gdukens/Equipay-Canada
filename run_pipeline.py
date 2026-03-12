#!/usr/bin/env python
"""
EquiPay Canada - Main Pipeline Runner
=====================================

Automated pipeline for comprehensive pay equity analysis.

DATA SCOPE:
This project uses ONLY:
- LFS PUMF microdata (2010-2025)
- Macroeconomic data (CPI, GDP, unemployment, interest rates)

Features:
- Flexible data sources (LFS PUMF preferred, synthetic fallback)
- Macro-adjusted real wage calculations
- Weighted statistics using survey weights
- ML model training with fairness evaluation
- Publication-ready reports

Usage:
    python run_pipeline.py                     # Auto-detect data source
    python run_pipeline.py --source=pumf      # Use real LFS microdata
    python run_pipeline.py --source=synthetic # Use synthetic data
"""

import logging
import argparse
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_pipeline import LFSDataPipeline
from src.feature_engineering import FeatureEngineer
from src.models import SalaryPredictor
from src.analysis import PayEquityAnalyzer, run_full_analysis
from src.fairness import FairnessAnalyzer, generate_fairness_report
from src.utils import setup_logging, Timer, create_output_directories
from src.constants import COLS, DATA_SCOPE_START, DATA_SCOPE_END


def main():
    """Run the complete pay equity analysis pipeline."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='EquiPay Canada - Pay Equity Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Data Sources:
  pumf      Real LFS PUMF microdata (preferred)
  synthetic Generated synthetic data (for testing)
  auto      Try pumf first, fall back to synthetic (default)

Examples:
  python run_pipeline.py
  python run_pipeline.py --source=pumf --samples=100000
  python run_pipeline.py --source=synthetic --skip-training
        """
    )
    parser.add_argument('--source', 
                        choices=['auto', 'pumf', 'synthetic'],
                        default='auto',
                        help='Data source: pumf (LFS microdata) or synthetic')
    parser.add_argument('--samples', type=int, default=50000,
                        help='Number of samples for synthetic data')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip model training if model exists')
    parser.add_argument('--output-dir', type=str, default='reports',
                        help='Output directory for reports')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    args = parser.parse_args()
    
    # Setup
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)
    create_output_directories()
    
    logger.info("=" * 70)
    logger.info("EQUIPAY CANADA - PAY EQUITY ANALYSIS PIPELINE")
    logger.info(f"Data Scope: {DATA_SCOPE_START}-{DATA_SCOPE_END}")
    logger.info("=" * 70)
    
    # =========================================================================
    # Step 1: Data Pipeline
    # =========================================================================
    with Timer("Data Pipeline"):
        logger.info("\n[1/5] Running Data Pipeline...")
        
        pipeline = LFSDataPipeline()
        
        processed_path = Path('data/processed/lfs_processed.csv')
        
        # Check if we can use existing processed data
        if processed_path.exists() and args.source == 'auto':
            logger.info("Loading existing processed data...")
            import pandas as pd
            df = pd.read_csv(processed_path)
            
            # Verify data is from expected source
            if COLS.SOURCE in df.columns:
                source = df[COLS.SOURCE].iloc[0] if len(df) > 0 else 'Unknown'
                logger.info(f"Data source: {source}")
        else:
            # Run pipeline with specified data source
            df = pipeline.run_pipeline(
                data_source=args.source,
                n_samples=args.samples,
                save=True
            )
        
        # Report data summary
        logger.info(f"Dataset: {len(df):,} records, {len(df.columns)} features")
        
        if COLS.YEAR in df.columns:
            year_range = f"{df[COLS.YEAR].min()}-{df[COLS.YEAR].max()}"
            logger.info(f"Year range: {year_range}")
        
        # Report gender distribution
        gender_col = COLS.GENDER if COLS.GENDER in df.columns else 'SEX'
        if gender_col in df.columns:
            male_n = (df[gender_col] == 1).sum()
            female_n = (df[gender_col] == 2).sum()
            logger.info(f"Gender: Male={male_n:,}, Female={female_n:,}")
    
    # =========================================================================
    # Step 2: Feature Engineering
    # =========================================================================
    with Timer("Feature Engineering"):
        logger.info("\n[2/5] Feature Engineering...")
        
        feature_engineer = FeatureEngineer()
        X, y = feature_engineer.fit_transform(df)
        
        logger.info(f"Features: {X.shape[1]}")
        logger.info(f"Target: {COLS.HOURLY_EARNINGS} (hourly earnings)")
        
        # Report weighted mean wage if weights available
        if COLS.FINAL_WEIGHT in df.columns:
            import numpy as np
            weighted_mean = np.average(df[COLS.HOURLY_EARNINGS], 
                                       weights=df[COLS.FINAL_WEIGHT])
            logger.info(f"Weighted mean wage: ${weighted_mean:.2f}/hr")
    
    # =========================================================================
    # Step 3: Model Training
    # =========================================================================
    with Timer("Model Training"):
        logger.info("\n[3/5] Training Salary Prediction Models...")
        
        model_path = Path('models/salary_predictor.joblib')
        
        if args.skip_training and model_path.exists():
            logger.info("Loading existing model...")
            predictor = SalaryPredictor()
            predictor.load(str(model_path))
        else:
            predictor = SalaryPredictor()
            metrics = predictor.train(X, y, feature_engineer.get_feature_names())
            
            # Display results
            logger.info("\nModel Performance:")
            for name, m in metrics.items():
                logger.info(f"  {name}: R²={m['weighted_r2']:.4f}, RMSE=${m['weighted_rmse']:.2f}")
            
            # Save model
            predictor.save(str(model_path))
            feature_engineer.save('models/feature_engineer.joblib')
        
        # Generate predictions
        y_pred = predictor.predict(X)
    
    # =========================================================================
    # Step 4: Pay Equity Analysis
    # =========================================================================
    with Timer("Pay Equity Analysis"):
        logger.info("\n[4/5] Pay Equity Statistical Analysis...")
        
        # Run comprehensive analysis
        analysis_results = run_full_analysis(df, output_dir=args.output_dir)
        
        # Key findings
        if 'raw_gap' in analysis_results:
            gap = analysis_results['raw_gap']['raw_gap']['mean_gap_pct']
            logger.info(f"\n*** Key Finding: Raw Gender Wage Gap = {gap:.1f}% ***")
        
        # Time series analysis if multiple years
        if COLS.YEAR in df.columns:
            years = df[COLS.YEAR].nunique()
            if years > 1:
                logger.info(f"Time series: {years} years of data available")
    
    # =========================================================================
    # Step 5: Fairness Evaluation
    # =========================================================================
    with Timer("Fairness Evaluation"):
        logger.info("\n[5/5] Fairness Evaluation...")
        
        try:
            # Generate fairness report
            report_path = generate_fairness_report(
                df=df,
                y_pred=y_pred,
                output_path=f"{args.output_dir}/fairness_report.html"
            )
            
            # Wage gap analysis
            analyzer = FairnessAnalyzer()
            wage_gap = analyzer.analyze_wage_gap(df, y_pred)
            
            logger.info(f"Actual wage gap: {wage_gap['actual']['raw_gap_pct']:.1f}%")
            if 'predicted' in wage_gap:
                logger.info(f"Predicted wage gap: {wage_gap['predicted']['raw_gap_pct']:.1f}%")
        except Exception as e:
            logger.warning(f"Fairness evaluation error: {e}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)
    
    logger.info(f"""
    
OUTPUTS GENERATED:
------------------
✓ Processed data: data/processed/lfs_processed.csv
✓ Trained model: models/salary_predictor.joblib
✓ Feature engineer: models/feature_engineer.joblib
✓ Pay equity report: {args.output_dir}/pay_equity_summary.txt
✓ Fairness report: {args.output_dir}/fairness_report.html

DATA SUMMARY:
-------------
• Records: {len(df):,}
• Data source: {df.get(COLS.SOURCE, ['Unknown']).iloc[0] if COLS.SOURCE in df.columns else 'Unknown'}
• Scope: {DATA_SCOPE_START}-{DATA_SCOPE_END}

NEXT STEPS:
-----------
1. Review the pay equity summary report
2. Open the fairness HTML report in a browser
3. Run the dashboard: streamlit run app/dashboard.py
4. Start the API: uvicorn api.main:app --reload
5. Run econometric analysis: python run_econometrics.py

    """)


if __name__ == "__main__":
    main()
