from app.core.config import settings
from loguru import logger

class MLModelService:
    def __init__(self):
        self.model = None

    def load_model(self):
        logger.info(f"Loading model version {settings.MODEL_VERSION}...")
        # Simulate loading a model file (e.g., joblib.load or onnxruntime.InferenceSession)
        self.model = "MockModelInstance"
        logger.info("Model loaded successfully.")

    def predict(self, input_data: list):
        if not self.model:
            raise RuntimeError("Model is not loaded.")
        # Dummy inference logic
        logger.info(f"Running inference on {len(input_data)} items")
        return [x * 2.5 for x in input_data]

model_service = MLModelService()
