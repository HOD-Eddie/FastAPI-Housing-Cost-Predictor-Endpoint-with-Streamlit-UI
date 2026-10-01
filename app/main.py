# app/main.py
from fastapi import FastAPI, Response, status
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router as housing_router
from app.core.config import settings

# Import our singletons to check their internal states during health probes
from app.services.price_service import pricing_service
from app.services.description_service import description_service

# 1. Instantiate the central FastAPI engine
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Chained Pipeline: Custom Random Forest Regressor + Pre-trained Text Transformer.",
    version="1.0.0"
)

# 2. Wire standard cross-origin resource mapping
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Mount our active real estate generation pipeline router
app.include_router(housing_router, prefix="/api", tags=["Real Estate Operations"])


# 4. OPTIMIZED LIVENESS PROBE
@app.get("/health/live", summary="Perform operational liveness diagnostics")
def check_liveness(response: Response):
    """
    Highly optimized health check. Instead of a dummy text string, it active probes 
    our memory allocations to confirm both models are loaded and ready to serve math.
    """
    # Verify Station 1: Check if scikit-learn model object is loaded in memory
    is_pricing_ready = pricing_service.model is not None
    
    # Verify Station 2: Check if our pre-trained text generator container exists
    is_generator_ready = description_service.generator is not None

    if not is_pricing_ready or not is_generator_ready:
        # If any component failed to load on boot, switch status to 503 Service Unavailable
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {
            "status": "unhealthy",
            "diagnostics": {
                "custom_regression_loaded": is_pricing_ready,
                "pretrained_llm_loaded": is_generator_ready
            }
        }

    # System is 100% operational (Defaults to HTTP 200 OK)
    return {
        "status": "alive",
        "environment": settings.ENVIRONMENT,
        "model_version": settings.MODEL_VERSION
    }
