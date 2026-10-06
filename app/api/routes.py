# app/api/routes.py
from fastapi import APIRouter, HTTPException
from app.schemas.housing_schema import HouseFeaturesInput, HouseListingResponse
from app.core.config import settings

# Import both of our operational service singletons
from app.services.price_service import pricing_service
from app.services.description_service import description_service

# 1. Initialize the router package
router = APIRouter()

@router.post(
    "/generate-listing", 
    response_model=HouseListingResponse,
    summary="Calculate property price and generate marketing copy"
)
def create_house_listing(payload: HouseFeaturesInput):
    """
    Takes comprehensive housing dimensions, predicts the market valuation 
    using a custom Random Forest Regressor, and pipes the results into a 
    pre-trained language transformer to write an advertisement script.
    """
    try:
        # Step A: Convert Pydantic data into a plain Python dictionary
        input_data = payload.model_dump()
        
        # Step B: Station 1 - Compute the fair market price using our custom model
        estimated_price = pricing_service.predict_price(input_data)
        
        # Step C: Station 2 - Pass data and the calculated price to the pre-trained LLM
        marketing_copy = description_service.generate_listing(input_data, estimated_price)
        
        # Step D: Structure the final response to match our HouseListingResponse schema
        return HouseListingResponse(
            estimated_price_usd=estimated_price,
            generated_description=marketing_copy,
            model_version=settings.MODEL_VERSION
        )
        
    except Exception as e:
        # Safeguard our endpoints against internal calculations faults
        raise HTTPException(
            status_code=500, 
            detail=f"Real Estate Pipeline Processing Error: {str(e)}"
        )
 