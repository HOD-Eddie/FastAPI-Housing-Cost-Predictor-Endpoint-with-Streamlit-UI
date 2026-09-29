from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.core.config import settings
from app.services.ml_service import model_service
from prometheus_fastapi_instrumentator import Instrumentator

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model during startup so it stays in memory
    model_service.load_model()
    yield
    # Clean up operations can go here on shutdown

app = FastAPI(title=settings.PROJECT_NAME, lifespan=lifespan)

# Instrument Prometheus metrics
Instrumentator().instrument(app).expose(app)

@app.post("/predict")
def get_prediction(data: list[float]):
    predictions = model_service.predict(data)
    return {"predictions": predictions, "model_version": settings.MODEL_VERSION}
