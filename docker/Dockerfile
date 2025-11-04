FROM python:3.11-slim

ARG MODE=prod
ENV MODE=${MODE}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    RUNNING_IN_DOCKER=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ .

# Copy models to root level
COPY models/ /models/

# Create logs directory
RUN mkdir -p /app/logs

# Don't copy .env - use env_file in docker-compose instead
# COPY .env /.env  # ❌ Remove this line

EXPOSE 8000

# Run as non-root user for better security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app /models
USER appuser

# Use exec form for better signal handling
CMD ["sh", "-c", "if [ \"$MODE\" = \"dev\" ]; then uvicorn main:app --host 0.0.0.0 --port 8000 --reload; else uvicorn main:app --host 0.0.0.0 --port 8000; fi"]