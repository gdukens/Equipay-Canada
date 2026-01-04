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
from pathlib import Path
import sys
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_store import EquiPayDataStore

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
    """Service for model predictions using DuckDB data store"""
    
    def __init__(self):
        self.model = None
        self.feature_engineer = None
        self.data_store = None
        self._load_model()
    
    def _load_model(self):
        """Load model and initialize data store"""
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
            
            # Initialize DuckDB data store (memory-efficient)
            self.data_store = EquiPayDataStore(memory_limit='3GB')
            stats = self.data_store.get_summary_stats()
            logger.info(f"Data store initialized: {stats['total_records']:,} records")
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
    
    def predict(self, profile: WorkerProfile) -> SalaryPrediction:
        """Make salary prediction using DuckDB queries"""
        if self.data_store is None:
            raise HTTPException(status_code=500, detail="Data store not available")
        
        # Query similar workers using SQL (memory-efficient)
        query = f"""
            SELECT HRLYEARN / 100.0 as HRLYEARN
            FROM lfs
            WHERE GENDER = {profile.sex}
              AND FTPTMAIN = {profile.full_time}
              AND EDUC = {profile.education}
              AND NOC_10 = {profile.occupation}
              AND HRLYEARN IS NOT NULL AND HRLYEARN > 0
        """
        similar = self.data_store.query(query)
        
        # Fallback to broader filters if too few matches
        if len(similar) < 30:
            query = f"""
                SELECT HRLYEARN / 100.0 as HRLYEARN
                FROM lfs
                WHERE GENDER = {profile.sex}
                  AND FTPTMAIN = {profile.full_time}
                  AND HRLYEARN IS NOT NULL AND HRLYEARN > 0
            """
            similar = self.data_store.query(query)
        
        if len(similar) < 10:
            # Get overall wages (converting from cents)
            similar = self.data_store.query("SELECT HRLYEARN / 100.0 as HRLYEARN FROM lfs WHERE HRLYEARN > 0")
        
        # Calculate prediction statistics
        wages = similar['HRLYEARN']
        predicted = wages.mean()
        std = wages.std()
        n = len(wages)
        
        # 95% confidence interval
        se = std / np.sqrt(n) if n > 0 else 0
        ci_lower = max(predicted - 1.96 * se, wages.quantile(0.1))
        ci_upper = predicted + 1.96 * se
        
        # Percentile in overall distribution (converting from cents)
        overall_wages = self.data_store.query("SELECT HRLYEARN / 100.0 as HRLYEARN FROM lfs WHERE HRLYEARN > 0")['HRLYEARN']
        percentile = (overall_wages < predicted).mean() * 100
        
        # Comparison to averages
        overall_avg = overall_wages.mean()
        gender_wages = self.data_store.query(f"""
            SELECT HRLYEARN / 100.0 as HRLYEARN FROM lfs 
            WHERE GENDER = {profile.sex} AND HRLYEARN IS NOT NULL AND HRLYEARN > 0
        """)
        gender_avg = gender_wages['HRLYEARN'].mean()
        
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
        """Calculate wage gap statistics using DuckDB"""
        if self.data_store is None:
            raise HTTPException(status_code=500, detail="Data store not available")
        
        # Get detailed stats for statistical test (convert cents to dollars)
        stats = self.data_store.query("""
            SELECT 
                GENDER,
                AVG(HRLYEARN) / 100.0 as avg_wage,
                STDDEV(HRLYEARN) / 100.0 as std_wage,
                COUNT(*) as n
            FROM lfs
            WHERE HRLYEARN IS NOT NULL AND HRLYEARN > 0
            GROUP BY GENDER
        """)
        
        male_stats = stats[stats['GENDER'] == 1].iloc[0] if len(stats[stats['GENDER'] == 1]) > 0 else None
        female_stats = stats[stats['GENDER'] == 2].iloc[0] if len(stats[stats['GENDER'] == 2]) > 0 else None
        
        if male_stats is None or female_stats is None:
            raise HTTPException(status_code=500, detail="Insufficient data for wage gap calculation")
        
        male_avg = male_stats['avg_wage']
        female_avg = female_stats['avg_wage']
        gap_dollars = male_avg - female_avg
        gap_pct = (gap_dollars / male_avg) * 100
        
        # Welch's t-test (approximate using summary stats)
        n1, n2 = int(male_stats['n']), int(female_stats['n'])
        s1, s2 = male_stats['std_wage'], female_stats['std_wage']
        
        se = np.sqrt((s1**2 / n1) + (s2**2 / n2))
        t_stat = gap_dollars / se if se > 0 else 0
        
        # Approximate p-value using normal distribution for large samples
        from scipy import stats as scipy_stats
        p_value = 2 * (1 - scipy_stats.norm.cdf(abs(t_stat)))
        
        return WageGapResponse(
            raw_gap_percentage=round(gap_pct, 2),
            raw_gap_dollars=round(gap_dollars, 2),
            male_average=round(male_avg, 2),
            female_average=round(female_avg, 2),
            female_to_male_ratio=round(female_avg / male_avg, 4),
            sample_size={
                "male": n1,
                "female": n2
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
    if model_service.data_store is None:
        raise HTTPException(status_code=500, detail="Data store not available")
    
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
    
    # Use SQL aggregation (memory-efficient, convert cents to dollars)
    query = f"""
        SELECT 
            {col} as group_val,
            GENDER,
            AVG(HRLYEARN) / 100.0 as avg_wage,
            COUNT(*) as n
        FROM lfs
        WHERE HRLYEARN IS NOT NULL AND HRLYEARN > 0
        GROUP BY {col}, GENDER
    """
    stats = model_service.data_store.query(query)
    
    results = []
    for group_val in stats['group_val'].unique():
        group_data = stats[stats['group_val'] == group_val]
        male_row = group_data[group_data['GENDER'] == 1]
        female_row = group_data[group_data['GENDER'] == 2]
        
        if len(male_row) == 0 or len(female_row) == 0:
            continue
        
        male_n = int(male_row['n'].iloc[0])
        female_n = int(female_row['n'].iloc[0])
        
        if male_n < 10 or female_n < 10:
            continue
        
        male_avg = float(male_row['avg_wage'].iloc[0])
        female_avg = float(female_row['avg_wage'].iloc[0])
        
        results.append({
            "group": int(group_val) if isinstance(group_val, (np.integer, float)) else group_val,
            "male_average": round(male_avg, 2),
            "female_average": round(female_avg, 2),
            "gap_percentage": round(((male_avg - female_avg) / male_avg) * 100, 2),
            "sample_size": {"male": male_n, "female": female_n}
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
    if model_service.data_store is None:
        raise HTTPException(status_code=500, detail="Data store not available")
    
    # Use SQL for all statistics (memory-efficient, convert cents to dollars)
    stats = model_service.data_store.query("""
        SELECT 
            COUNT(*) as sample_size,
            AVG(HRLYEARN) / 100.0 as mean_wage,
            MEDIAN(HRLYEARN) / 100.0 as median_wage,
            STDDEV(HRLYEARN) / 100.0 as std_wage,
            MIN(HRLYEARN) / 100.0 as min_wage,
            MAX(HRLYEARN) / 100.0 as max_wage,
            QUANTILE_CONT(HRLYEARN, 0.10) / 100.0 as p10,
            QUANTILE_CONT(HRLYEARN, 0.25) / 100.0 as p25,
            QUANTILE_CONT(HRLYEARN, 0.50) / 100.0 as p50,
            QUANTILE_CONT(HRLYEARN, 0.75) / 100.0 as p75,
            QUANTILE_CONT(HRLYEARN, 0.90) / 100.0 as p90
        FROM lfs
        WHERE HRLYEARN IS NOT NULL AND HRLYEARN > 0
    """)
    
    gender_stats = model_service.data_store.query("""
        SELECT GENDER, COUNT(*) as n
        FROM lfs
        WHERE HRLYEARN IS NOT NULL
        GROUP BY GENDER
    """)
    
    gender_dist = {int(row['GENDER']): int(row['n']) for _, row in gender_stats.iterrows()}
    
    s = stats.iloc[0]
    return {
        "sample_size": int(s['sample_size']),
        "wage_statistics": {
            "mean": round(s['mean_wage'], 2),
            "median": round(s['median_wage'], 2),
            "std": round(s['std_wage'], 2),
            "min": round(s['min_wage'], 2),
            "max": round(s['max_wage'], 2),
            "percentiles": {
                "10": round(s['p10'], 2),
                "25": round(s['p25'], 2),
                "50": round(s['p50'], 2),
                "75": round(s['p75'], 2),
                "90": round(s['p90'], 2),
            }
        },
        "gender_distribution": gender_dist
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
