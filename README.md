# FastAPI-Housing-Cost-Predictor-Endpoint-with-Streamlit-UI
This repo is a testament to my journey, as a MLOps Engineer. The codebase is not the most optimized, or most accurate, but instead shows my commitment to learning, and deploying without waiting for perfection. Even as I evolve beyond this stage, I seek to always make reference, and learn from my own documented milestones.

# 🏡 Smart Real Estate Valuation & AI Listing Platform
This is an MLOps pipeline built with **FastAPI** and **Streamlit**. The system takes physical and economic neighborhood metrics, calculates an accurate property valuation using a custom regression model, and instantly pipes those insights into a pre-trained language transformer to generate custom real estate marketing descriptions.

---

## 🏗️ Architecture & Core Data Flow

The platform is designed as an isolated microservices assembly line:

1. **Station 1: Custom Valuation (Scikit-Learn Regression)**
   * Uses a **Random Forest Regressor** trained on the California Housing Dataset (achieving a **0.89 R² accuracy score**).
   * Maps 8 spatial-economic features to calculate absolute property values in real USD.
2. **Station 2: Automated Copywriting (Pre-trained LLM)**
   * Leverages the open-source Hugging Face **`distilgpt2`** model (~350MB local memory footprint).
   * Employs prompt engineering templates adjusted with creative sampling and repetition penalties to generate tailored advertisement scripts.
3. **Data Protection Layer (Pydantic)**
   * Enforces strict operational boundaries (e.g., house dimensions must be `gt=0`) to protect internal tensor layers from parsing crashes.

---

## 🛠️ Project Structure

```text
my_new_project_name/
├── app/
│   ├── api/
│   │   └── routes.py             # Route controller and endpoints
│   ├── core/
│   │   └── config.py             # Pydantic BaseSettings management
│   ├── schemas/
│   │   └── housing_schema.py     # Pydantic Request/Response models
│   └── services/
│   |    ├── price_service.py     # Station 1 logic (Custom model handler)
│   |    └── description_service.py # Station 2 logic (Pre-trained HuggingFace Pipeline)
|   └── models/
│        └── housing_model.joblib      # Serialized Random Forest model binary
├── tests/
│   └── test_pipeline.py          # Pytest integration & validation suites
├── dashboard.py                  # Streamlit user interface client
├── train_housing.py              # Offline model training pipeline script
└── README.md
```

---

## ⚡ Getting Started

### 1. Environment Setup & Installation
Clone the repository and spin up your local Python environment:

```bash
# Activate your virtual environment
source .venv/bin/activate

# Install all frozen dependencies
pip install -r requirements.txt
```

### 2. Train the Custom Regressor
Generate your local production model weights binary file:

```bash
python3 train_housing.py
```

### 3. Launch the Backend API
Run the Uvicorn engine. On initial boot, the app will automatically download the pre-trained `safetensors` model weights to your local machine cache.

```bash
PYTHONPATH=. uvicorn app.main:app --reload
```
* Interactive API Documentation (Swagger UI): `http://127.0.0`
* Optimized Memory Liveness Diagnostics: `http://127.0.0`

### 4. Run the Visual Frontend
Open a separate terminal window, activate your virtual environment, and fire up the UI client:

```bashapp/data_load.py/dapp/data_load.py/dataset_load.pyataset_load.py
streamlit run dashboard.py
```
* Dashboard URL: `http://localhost:8501`

---

## 🧪 Automated Testing

The platform features an automated quality-assurance validation suite that evaluates system health configurations and contract type constraints. Run them via the console root:

```bash
python3 -m pytest tests/
```
