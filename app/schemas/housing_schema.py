from pydantic import BaseModel, Field

class HouseFeaturesInput(BaseModel):
    # Enforcing strict, logical real-world constraints for our 8 parameters
    median_income: float = Field(..., gt=0, description="Median income in tens of thousands (e.g., 8.3 = $83,000)", examples=[8.32], strict=True)
    house_age: float = Field(..., ge=0, description="Median age of houses in the block", examples=[41.0], strict=True)
    avg_rooms: float = Field(..., gt=0, description="Average number of rooms per household", examples=[6.98], strict=True)
    avg_bedrooms: float = Field(..., gt=0, description="Average number of bedrooms per household", examples=[1.02], strict=True)
    population: float = Field(..., gt=0, description="Block group population total", examples=[322.0], strict=True)
    avg_occupancy: float = Field(..., gt=0, description="Average household members", examples=[2.55], strict=True)
    latitude: float = Field(..., min_value=-90, max_value=90, description="Geographical latitude", examples=[37.88], strict=True)
    longitude: float = Field(..., min_value=-180, max_value=180, description="Geographical longitude", examples=[-122.23], strict=True)
    
    # Adding a custom configuration field for our Station 2 (The LLM)
    marketing_tone: str = Field("enthusiastic", description="The copywriting tone (e.g., professional, luxury, rustic)", examples=["luxury"])


class HouseListingResponse(BaseModel):
    estimated_price_usd: float = Field(..., description="The calculated fair market price in raw USD", strict=True)
    generated_description: str = Field(..., description="The pre-trained AI marketing description copy")
    model_version: str = Field(..., description="The operational model tag version")
