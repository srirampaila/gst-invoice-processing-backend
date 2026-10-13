# ==============================================================================
# Production Dockerfile for GST Invoice Processing Backend API
# ==============================================================================
FROM python:3.12-slim

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=8000 \
    HOST=0.0.0.0

# Install system dependencies: Tesseract OCR, Leptonica, and OpenCV headless runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency specifications first for Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and configuration
COPY src/ ./src/
COPY app/ ./app/
COPY tests/ ./tests/
COPY .env.example .env.example

# Create storage directories
RUN mkdir -p data/raw data/processed data/outputs data/samples

# Expose the default API port
EXPOSE 8000

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/api/health || exit 1

# Start Uvicorn ASGI server
CMD ["sh", "-c", "uvicorn src.api.app:app --host ${HOST:-0.0.0.0} --port ${PORT:-8000}"]
