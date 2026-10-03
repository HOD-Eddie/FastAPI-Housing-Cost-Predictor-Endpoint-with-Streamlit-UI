# 1. Use an official lightweight Python image
FROM python:3.13-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy and install dependencies first (optimizes Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the entire 'app' directory and files into the container
COPY . .

# 5. Expose the API port
EXPOSE 8000

# 6. Run Uvicorn listening on all network interfaces to serve the model
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
