from pathlib import Path
import joblib
import numpy as np

class PricingService:
    DEFAULT_MODEL_PATH = Path("app/models/housing_model.joblib")

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH):
        self.model = None
        self.load_model(model_path)

    def load_model(self, model_path: Path):
        if not model_path.exists():
            raise FileNotFoundError(f"Housing model file not found at: {model_path.resolve()}")
        self.model = joblib.load(model_path)
        print(f"--> Custom Regression Model successfully loaded from {model_path}")

    def predict_price(self, input_data: dict) -> float:
        if not self.model:
            raise RuntimeError("Regression model is not loaded.")

        # Crucial Step: Arrange the Pydantic data into the EXACT structural 
        # positional sequence your training script used:
        # ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
        input_array = np.array([
            [
                input_data["median_income"],
                input_data["house_age"],
                input_data["avg_rooms"],
                input_data["avg_bedrooms"],
                input_data["population"],
                input_data["avg_occupancy"],
                input_data["latitude"],
                input_data["longitude"],
            ]
        ])

        # Run model inference
        # The model returns an array containing a single float value (e.g., [3.4523])
        raw_prediction = float(self.model.predict(input_array)[0])

        # Convert the raw prediction back to real-world USD.
        # Remember, the dataset target is scaled in hundreds of thousands of dollars!
        real_usd_price = raw_prediction * 100000

        return round(real_usd_price, 2)

# Instantiate the service singleton instance for imports
pricing_service = PricingService()
