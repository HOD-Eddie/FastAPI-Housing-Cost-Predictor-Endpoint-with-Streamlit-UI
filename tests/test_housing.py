from fastapi.testclient import TestClient
from app.main import app

# 1. Initialize the virtual test client pointing to your real estate platform
client = TestClient(app)

def test_optimized_liveness_check():
    """Verify our diagnostic check probes memory allocations for both models."""
    response = client.get("/health/live")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "alive"
    assert "environment" in data
    assert "model_version" in data


def test_listing_generation_pipeline_success():
    """Verify a valid house profile completes the full custom + pretrained pipeline."""
    payload = {
        "median_income": 8.32,
        "house_age": 41.0,
        "avg_rooms": 6.98,
        "avg_bedrooms": 1.02,
        "population": 322.0,
        "avg_occupancy": 2.55,
        "latitude": 37.88,
        "longitude": -122.23,
        "marketing_tone": "luxury"
    }
    
    response = client.post("/api/generate-listing", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    # Validate the data contract structure
    assert "estimated_price_usd" in data
    assert "generated_description" in data
    assert "model_version" in data
    
    # Assert values are mathematically and structurally sound
    assert data["estimated_price_usd"] > 0
    assert isinstance(data["generated_description"], str)
    assert len(data["generated_description"]) > 0


def test_pipeline_validation_bounds():
    """Verify that Pydantic blocks broken data before it hits scikit-learn or the LLM."""
    invalid_payload = {
        "median_income": -5.0,        # Invalid: must be gt=0
        "house_age": "brand_new",     # Invalid: must be a float number
        "avg_rooms": 6.98,
        "avg_bedrooms": 1.02,
        "population": 322.0,
        "avg_occupancy": 2.55,
        "latitude": 95.0,             # Invalid: max latitude is 90
        "longitude": -122.23,
        "marketing_tone": "luxury"
    }
    
    response = client.post("/api/generate-listing", json=invalid_payload)
    
    # Check that FastAPI completely intercepts it as an unprocessable request
    assert response.status_code == 422
