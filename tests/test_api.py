"""
API Tests for EquiPay Canada
============================

Tests for FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.main import app


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


class TestHealthCheck:
    """Tests for health check endpoint"""
    
    def test_health_check(self, client):
        """Test health endpoint returns OK"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data


class TestSalaryPrediction:
    """Tests for salary prediction endpoints"""
    
    def test_predict_salary_valid(self, client):
        """Test prediction with valid input"""
        payload = {
            "sex": 2,
            "age": 35,
            "education": 5,
            "occupation": 1,
            "province": 35,
            "full_time": 1,
            "union_status": 3
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "predicted_hourly_wage" in data
        assert data["predicted_hourly_wage"] > 0
    
    def test_predict_salary_invalid_sex(self, client):
        """Test prediction with invalid sex value"""
        payload = {
            "sex": 3,  # Invalid
            "age": 35,
            "education": 5,
            "occupation": 1,
            "province": 35,
            "full_time": 1,
            "union_status": 3
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Validation error
    
    def test_predict_salary_missing_field(self, client):
        """Test prediction with missing required field"""
        payload = {
            "age": 35,
            "education": 5,
            # Missing sex
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422


class TestWageGap:
    """Tests for wage gap endpoints"""
    
    def test_get_wage_gap(self, client):
        """Test wage gap endpoint"""
        response = client.get("/wage-gap")
        # May return 200 or 503 depending on data availability
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "raw_gap_percentage" in data
    
    def test_wage_gap_by_occupation(self, client):
        """Test wage gap by occupation"""
        response = client.get("/wage-gap/by-occupation")
        # Check response structure
        assert response.status_code in [200, 404, 503]


class TestBatchPrediction:
    """Tests for batch prediction endpoint"""
    
    def test_batch_predict(self, client):
        """Test batch prediction"""
        payload = {
            "profiles": [
                {
                    "sex": 1,
                    "age": 30,
                    "education": 4,
                    "occupation": 0,
                    "province": 35,
                    "full_time": 1,
                    "union_status": 3
                },
                {
                    "sex": 2,
                    "age": 40,
                    "education": 5,
                    "occupation": 1,
                    "province": 24,
                    "full_time": 1,
                    "union_status": 1
                }
            ]
        }
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 2


class TestAPIDocumentation:
    """Tests for API documentation endpoints"""
    
    def test_openapi_schema(self, client):
        """Test OpenAPI schema is accessible"""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "info" in data
    
    def test_docs_endpoint(self, client):
        """Test Swagger docs are accessible"""
        response = client.get("/docs")
        assert response.status_code == 200
    
    def test_redoc_endpoint(self, client):
        """Test ReDoc is accessible"""
        response = client.get("/redoc")
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
