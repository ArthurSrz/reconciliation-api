#!/bin/bash
# entrypoint.sh - Railway deployment script

# Set default port if not provided by Railway
export PORT=${PORT:-8080}

# Start the application
exec gunicorn --bind 0.0.0.0:$PORT --timeout 300 --workers 1 --worker-class sync reconciliation_api:app
