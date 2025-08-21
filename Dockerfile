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

# Note: Allosaurus model will be downloaded on first run.
# We keep the image build robust by not failing if optional model fetch fails.

# Copy application code
COPY main/ ./main/
COPY *.py ./

# Optional: copy tenant schema SQL if present in build context
# In this repo, it's committed at app/prisma/tenant_schema.sql
RUN mkdir -p prisma
COPY app/prisma/tenant_schema.sql ./prisma/tenant_schema.sql

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
