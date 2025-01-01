# Base image with Python and GPU support
FROM python:3.9-slim as base

# Switch to non-root user
RUN apt-get update && apt-get install -y \
    git curl libgl1-mesa-glx cmake build-essential && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements to the container
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . /app

# Make the download script executable and run it
RUN chmod +x download_weights.sh && \
    mkdir -p /root/.cache/torch/hub/checkpoints/ && \
    ./download_weights.sh

# Expose FastAPI port
EXPOSE 8000

# Set the entry point for running FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
