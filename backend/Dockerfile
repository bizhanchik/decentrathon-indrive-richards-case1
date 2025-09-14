# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --timeout 300 -r requirements.txt

# Copy the application code
COPY . .

# Create necessary directories for model weights
RUN mkdir -p ../runs/train/yolov8s/weights \
    && mkdir -p ../runs/train/model2/weights \
    && mkdir -p ../runs/train/model3/weights \
    && mkdir -p ../runs/train/model4/weights

# Expose the port the app runs on
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Command to run the application
CMD ["python", "app.py"]