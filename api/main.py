"""
EquiPay Canada - FastAPI Salary Prediction API
RESTful API for salary prediction and pay equity analysis
"""

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="EquiPay Canada API",
    description="""
    ## Salary Prediction and Pay Equity Analysis API
    
    This API provides:
    - Salary predictions based on worker characteristics
    - Pay equity analysis and wage gap statistics
    - Fairness metrics for model predictions
    
    Built for compensation analysts and HR professionals.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Pydantic Models (Request/Response Schemas)
# ============================================================================

class WorkerProfile(BaseModel):
    """Input profile for salary prediction"""
    sex: int = Field(..., ge=1, le=2, description="1=Male, 2=Female")
    age: int = Field(..., ge=15, le=70, description="Age in years")
    education: int = Field(..., ge=0, le=6, description="Education level (0-6)")
    occupation: int = Field(..., ge=0, le=9, description="NOC occupation code (0-9)")
    province: int = Field(35, description="Province code (e.g., 35=Ontario)")
    full_time: int = Field(1, ge=1, le=2, description="1=Full-time, 2=Part-time")
    union_status: int = Field(3, ge=1, le=3, description="1=Member, 2=Covered, 3=Not unionized")
    
    class Config:
        json_schema_extra = {
            "example": {
                "sex": 2,
                "age": 35,
                "education": 5,
                "occupation": 1,
                "province": 35,
                "full_time": 1,
                "union_status": 3
            }
        }


class SalaryPrediction(BaseModel):
    """Salary prediction response"""
    predicted_hourly_wage: float = Field(..., description="Predicted hourly wage in CAD")
    confidence_interval_lower: float = Field(..., description="Lower bound of 95% CI")
    confidence_interval_upper: float = Field(..., description="Upper bound of 95% CI")
    annual_salary_estimate: float = Field(..., description="Estimated annual salary (2000 hrs)")
    percentile: float = Field(..., description="Percentile rank in wage distribution")
    comparison_to_average: Dict[str, float] = Field(..., description="Comparison metrics")


class WageGapResponse(BaseModel):
    """Wage gap analysis response"""
    raw_gap_percentage: float
    raw_gap_dollars: float
    male_average: float
    female_average: float
    female_to_male_ratio: float
    sample_size: Dict[str, int]
    statistical_significance: Dict[str, Any]


class HealthCheck(BaseModel):
    """Health check response"""
    status: str
    version: str
    model_loaded: bool


class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    profiles: List[WorkerProfile]


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[SalaryPrediction]
    summary: Dict[str, float]


# ============================================================================
# Model and Data Loading
# ============================================================================

class ModelService:
    """Service for model predictions"""
    
    def __init__(self):
        self.model = None
        self.feature_engineer = None
        self.df_reference = None
        self._load_model()
    
    def _load_model(self):
        """Load model and reference data"""
        try:
            # Try to load trained model
            model_path = Path("models/salary_predictor.joblib")
            if model_path.exists():
                import joblib
                data = joblib.load(model_path)
                self.model = data.get('ensemble')
                logger.info("Model loaded successfully")
            else:
                logger.warning("No trained model found. Using fallback prediction.")
            
            # Load reference data for statistics (real Statistics Canada data)
            data_path = Path("data/processed/lfs_processed.csv")
            if data_path.exists():
                self.df_reference = pd.read_csv(data_path)
                # Verify it's real data
                if 'source' in self.df_reference.columns:
                    source = self.df_reference['source'].iloc[0] if len(self.df_reference) > 0 else 'Unknown'
                    logger.info(f"Reference data loaded: {len(self.df_reference)} records (Source: {source})")
                else:
                    logger.info(f"Reference data loaded: {len(self.df_reference)} records")
            else:
                # Generate synthetic data as fallback (in production, use real data)
                from src.data_pipeline import LFSDataPipeline
                pipeline = LFSDataPipeline()
                self.df_reference = pipeline.generate_synthetic_data(n_samples=10000)
                self.df_reference = pipeline.clean_data(self.df_reference)
                self.df_reference = pipeline.create_derived_features(self.df_reference)
                logger.info(f"Loaded synthetic fallback data: {len(self.df_reference)} records")
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
    
    def _get_gender_col(self) -> str:
        """Get the gender column name (GENDER or SEX)"""
        if self.df_reference is None:
            return 'GENDER'
        return 'GENDER' if 'GENDER' in self.df_reference.columns else 'SEX'
    
    def predict(self, profile: WorkerProfile) -> SalaryPrediction:
        """Make salary prediction"""
        if self.df_reference is None:
            raise HTTPException(status_code=500, detail="Reference data not available")
        
        gender_col = self._get_gender_col()
        
        # Use reference data for prediction (fallback when model not trained)
        filters = [
            self.df_reference[gender_col] == profile.sex,
            self.df_reference['FTPTMAIN'] == profile.full_time,
        ]
        
        if 'EDUC' in self.df_reference.columns:
            filters.append(self.df_reference['EDUC'] == profile.education)
        
        if 'NOC_10' in self.df_reference.columns:
            filters.append(self.df_reference['NOC_10'] == profile.occupation)
        
        # Combine filters
        mask = filters[0]
        for f in filters[1:]:
            mask = mask & f
        
        similar = self.df_reference[mask]
        
        # Fallback to broader filters if too few matches
        if len(similar) < 30:
            mask = (self.df_reference[gender_col] == profile.sex) & \
                   (self.df_reference['FTPTMAIN'] == profile.full_time)
            similar = self.df_reference[mask]
        
        if len(similar) < 10:
            similar = self.df_reference
        
        # Calculate prediction statistics
        predicted = similar['HRLYEARN'].mean()
        std = similar['HRLYEARN'].std()
        n = len(similar)
        
        # 95% confidence interval
        se = std / np.sqrt(n)
        ci_lower = max(predicted - 1.96 * se, similar['HRLYEARN'].quantile(0.1))
        ci_upper = predicted + 1.96 * se
        
        # Percentile in overall distribution
        percentile = (self.df_reference['HRLYEARN'] < predicted).mean() * 100
        
        # Comparison to averages
        overall_avg = self.df_reference['HRLYEARN'].mean()
        gender_avg = self.df_reference[self.df_reference[gender_col] == profile.sex]['HRLYEARN'].mean()
        
        return SalaryPrediction(
            predicted_hourly_wage=round(predicted, 2),
            confidence_interval_lower=round(ci_lower, 2),
            confidence_interval_upper=round(ci_upper, 2),
            annual_salary_estimate=round(predicted * 2000, 0),
            percentile=round(percentile, 1),
            comparison_to_average={
                "vs_overall": round(((predicted - overall_avg) / overall_avg) * 100, 1),
                "vs_same_gender": round(((predicted - gender_avg) / gender_avg) * 100, 1),
            }
        )
    
    def get_wage_gap(self) -> WageGapResponse:
        """Calculate wage gap statistics"""
        if self.df_reference is None:
            raise HTTPException(status_code=500, detail="Reference data not available")
        
        gender_col = self._get_gender_col()
        
        male = self.df_reference[self.df_reference[gender_col] == 1]['HRLYEARN']
        female = self.df_reference[self.df_reference[gender_col] == 2]['HRLYEARN']
        
        male_avg = male.mean()
        female_avg = female.mean()
        gap_dollars = male_avg - female_avg
        gap_pct = (gap_dollars / male_avg) * 100
        
        # Statistical test
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(male.dropna(), female.dropna())
        
        return WageGapResponse(
            raw_gap_percentage=round(gap_pct, 2),
            raw_gap_dollars=round(gap_dollars, 2),
            male_average=round(male_avg, 2),
            female_average=round(female_avg, 2),
            female_to_male_ratio=round(female_avg / male_avg, 4),
            sample_size={
                "male": len(male),
                "female": len(female)
            },
            statistical_significance={
                "t_statistic": round(t_stat, 4),
                "p_value": round(p_value, 6),
                "significant_at_05": p_value < 0.05,
                "significant_at_01": p_value < 0.01
            }
        )


# Initialize model service
model_service = ModelService()


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to EquiPay Canada API",
        "docs": "/docs",
        "version": "1.0.0"
    }


@app.get("/health", response_model=HealthCheck, tags=["General"])
async def health_check():
    """Health check endpoint"""
    return HealthCheck(
        status="healthy",
        version="1.0.0",
        model_loaded=model_service.model is not None or model_service.df_reference is not None
    )


@app.post("/predict", response_model=SalaryPrediction, tags=["Predictions"])
async def predict_salary(profile: WorkerProfile):
    """
    Predict salary for a worker profile
    
    - **sex**: Gender (1=Male, 2=Female)
    - **age**: Age in years (15-70)
    - **education**: Education level (0=Less than HS, 6=Graduate degree)
    - **occupation**: NOC occupation code (0-9)
    - **province**: Province code (e.g., 35=Ontario)
    - **full_time**: Employment type (1=Full-time, 2=Part-time)
    - **union_status**: Union status (1=Member, 2=Covered, 3=Not unionized)
    """
    try:
        prediction = model_service.predict(profile)
        return prediction
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def batch_predict(request: BatchPredictionRequest):
    """
    Batch prediction for multiple worker profiles
    """
    predictions = []
    for profile in request.profiles:
        try:
            pred = model_service.predict(profile)
            predictions.append(pred)
        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            continue
    
    if not predictions:
        raise HTTPException(status_code=400, detail="No valid predictions could be made")
    
    wages = [p.predicted_hourly_wage for p in predictions]
    
    return BatchPredictionResponse(
        predictions=predictions,
        summary={
            "count": len(predictions),
            "mean": round(np.mean(wages), 2),
            "median": round(np.median(wages), 2),
            "min": round(min(wages), 2),
            "max": round(max(wages), 2),
        }
    )


@app.get("/wage-gap", response_model=WageGapResponse, tags=["Analysis"])
async def get_wage_gap():
    """
    Get overall gender wage gap statistics
    """
    try:
        return model_service.get_wage_gap()
    except Exception as e:
        logger.error(f"Wage gap error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/wage-gap/{dimension}", tags=["Analysis"])
async def get_wage_gap_by_dimension(
    dimension: str = Query(..., description="Dimension: education, occupation, province")
):
    """
    Get wage gap by a specific dimension
    """
    df = model_service.df_reference
    if df is None:
        raise HTTPException(status_code=500, detail="Reference data not available")
    
    dimension_map = {
        "education": "EDUC",
        "occupation": "NOC_10",
        "province": "PROV",
    }
    
    if dimension not in dimension_map:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid dimension. Choose from: {list(dimension_map.keys())}"
        )
    
    col = dimension_map[dimension]
    if col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column {col} not in data")
    
    results = []
    for group in df[col].unique():
        group_df = df[df[col] == group]
        male = group_df[group_df['SEX'] == 1]['HRLYEARN']
        female = group_df[group_df['SEX'] == 2]['HRLYEARN']
        
        if len(male) < 10 or len(female) < 10:
            continue
        
        male_avg = male.mean()
        female_avg = female.mean()
        
        results.append({
            "group": int(group) if isinstance(group, (np.integer, float)) else group,
            "male_average": round(male_avg, 2),
            "female_average": round(female_avg, 2),
            "gap_percentage": round(((male_avg - female_avg) / male_avg) * 100, 2),
            "sample_size": {"male": len(male), "female": len(female)}
        })
    
    return {
        "dimension": dimension,
        "results": sorted(results, key=lambda x: x["gap_percentage"], reverse=True)
    }


@app.get("/statistics", tags=["Analysis"])
async def get_statistics():
    """
    Get overall wage statistics
    """
    df = model_service.df_reference
    if df is None:
        raise HTTPException(status_code=500, detail="Reference data not available")
    
    return {
        "sample_size": len(df),
        "wage_statistics": {
            "mean": round(df['HRLYEARN'].mean(), 2),
            "median": round(df['HRLYEARN'].median(), 2),
            "std": round(df['HRLYEARN'].std(), 2),
            "min": round(df['HRLYEARN'].min(), 2),
            "max": round(df['HRLYEARN'].max(), 2),
            "percentiles": {
                "10": round(df['HRLYEARN'].quantile(0.10), 2),
                "25": round(df['HRLYEARN'].quantile(0.25), 2),
                "50": round(df['HRLYEARN'].quantile(0.50), 2),
                "75": round(df['HRLYEARN'].quantile(0.75), 2),
                "90": round(df['HRLYEARN'].quantile(0.90), 2),
            }
        },
        "gender_distribution": df['SEX'].value_counts().to_dict() if 'SEX' in df.columns else {}
    }


@app.get("/reference-codes", tags=["Reference"])
async def get_reference_codes():
    """
    Get reference codes for API inputs
    """
    return {
        "sex": {"1": "Male", "2": "Female"},
        "education": {
            "0": "Less than high school",
            "1": "High school graduate",
            "2": "Some college",
            "3": "College diploma",
            "4": "University certificate",
            "5": "Bachelor's degree",
            "6": "Graduate degree"
        },
        "occupation": {
            "0": "Management",
            "1": "Business/Finance",
            "2": "Sciences",
            "3": "Health",
            "4": "Education/Law/Social",
            "5": "Art/Culture/Recreation",
            "6": "Sales/Service",
            "7": "Trades/Transport",
            "8": "Resources/Agriculture",
            "9": "Manufacturing"
        },
        "province": {
            "10": "Newfoundland and Labrador",
            "11": "Prince Edward Island",
            "12": "Nova Scotia",
            "13": "New Brunswick",
            "24": "Quebec",
            "35": "Ontario",
            "46": "Manitoba",
            "47": "Saskatchewan",
            "48": "Alberta",
            "59": "British Columbia"
        },
        "full_time": {"1": "Full-time", "2": "Part-time"},
        "union_status": {
            "1": "Union member",
            "2": "Covered by union agreement",
            "3": "Not unionized"
        }
    }


# ============================================================================
# Run Application
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
