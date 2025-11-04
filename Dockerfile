FROM python:3.11-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (this layer will be cached)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.11-slim

ARG MODE=prod
ENV MODE=${MODE}
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/app \
    RUNNING_IN_DOCKER=1

WORKDIR /app

# Copy system dependencies from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy application code (changes frequently)
COPY app/ ./app/

# Create directories
RUN mkdir -p /app/logs /models

# Run as non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app /models
USER appuser

EXPOSE 8000

CMD ["sh", "-c", "cd app && if [ \"$MODE\" = \"dev\" ]; then uvicorn main:app --host 0.0.0.0 --port 8000 --reload; else uvicorn main:app --host 0.0.0.0 --port 8000; fi"]