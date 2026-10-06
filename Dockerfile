# 1. Use an official lightweight Python image
FROM python:3.13-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy and install dependencies first (optimizes Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the project files into the container
COPY . .

# 5. Train the model inside the image so a fresh clone works
RUN python train_model.py

# 6. Download the text model at build time, not on first startup
RUN python -c "from transformers import pipeline; pipeline('text-generation', model='distilgpt2')"

# 7. Expose the API port
EXPOSE 8000

# 8. Run Uvicorn listening on all network interfaces to serve the model
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]