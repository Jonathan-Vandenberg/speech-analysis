# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    ffmpeg \
    libsndfile1 \
    espeak-ng espeak-ng-data \
    wget tar ca-certificates \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download Allosaurus model so it's available at runtime
RUN python - <<'PY'
from allosaurus.pretrained import Pretrained
try:
    Pretrained('latest')
    print('Allosaurus model cached successfully')
except Exception as e:
    print('Warning: failed to cache Allosaurus model:', e)
PY

# Copy application code
COPY main/ ./main/
COPY *.py ./

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash speech
RUN chown -R speech:speech /app
USER speech

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

# Start the application
CMD ["uvicorn", "main.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
