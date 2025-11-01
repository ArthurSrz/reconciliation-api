FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 5002

# Run with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5002", "--workers", "2", "--worker-class", "gthread", "--threads", "4", "--timeout", "120", "reconciliation_api:app"]