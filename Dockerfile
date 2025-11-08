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

# Make entrypoint script executable
RUN chmod +x entrypoint.sh

# Set environment variables
ENV PYTHONPATH=/app:/app/nano_graphrag

# Health check - disable for now due to PORT variable issues
# HEALTHCHECK --interval=30s --timeout=30s --start-period=40s --retries=3 \
#     CMD curl -f http://localhost:8080/health || exit 1

# Run the application using entrypoint script
CMD ["./entrypoint.sh"]
