# FastAPI Housing Cost Predictor with Streamlit UI

This repo is a testament to my journey as an MLOps engineer. The code is not the most optimized or the most accurate. It shows my commitment to learning and deploying without waiting for perfection. As I grow beyond this stage, I want to keep learning from my own documented milestones.

## What it does

You enter details about a neighborhood. The app does two things:

1. A Random Forest Regressor, trained on the California Housing dataset, estimates the property price from 8 features.
2. A pre-trained Hugging Face model (distilgpt2) writes a short listing description around that price, in the tone you choose.

Pydantic validates every request before it reaches either model.

## Project structure

```text
├── app/
│   ├── api/routes.py                  # Endpoints
│   ├── core/config.py                 # pydantic-settings configuration
│   ├── schemas/housing_schema.py      # Request and response models
│   ├── services/
│   │   ├── price_service.py           # Loads and runs the regression model
│   │   └── description_service.py     # Text generation pipeline
│   └── main.py                        # App entry point and health check
├── tests/test_housing.py              # Pytest tests
├── dashboard.py                       # Streamlit interface
├── train_model.py                     # Trains and saves the model
├── Dockerfile                         # API image
├── Dockerfile.streamlit               # Dashboard image
└── docker-compose.yml                 # Runs both services
```

## Run it locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Train the model first. The model file is not stored in the repo.
python train_model.py

# Start the API
PYTHONPATH=. uvicorn app.main:app --reload
```

- API docs: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health/live

In a second terminal, start the dashboard:

```bash
streamlit run dashboard.py
```

Dashboard: http://localhost:8501

## Run it with Docker

Train the model first, so the file exists when the image is built:

```bash
python train_model.py
docker compose up --build
```

## Tests

```bash
python -m pytest tests/
```

## What's next

CI on every push, experiment tracking, cloud deployment, monitoring, and a stronger model for the text generation.