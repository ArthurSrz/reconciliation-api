FROM python:3.11-slim

# Install system dependencies for compilation
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    cmake \
    pkg-config \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with more memory and better caching
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app:/app/nano_graphrag

# Health check (Railway will set PORT, default to 8080 for health check)
HEALTHCHECK --interval=30s --timeout=30s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8080}/health || exit 1

# Run the application (Railway will provide PORT variable)
CMD gunicorn --bind 0.0.0.0:$PORT --timeout 300 --workers 1 --worker-class sync reconciliation_api:app
