FROM python:3.11-slim

ARG MODE=prod
ENV MODE=${MODE}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    RUNNING_IN_DOCKER=1

WORKDIR /app

# Install system dependencies including wget/curl for downloading
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ .

# Create directories
RUN mkdir -p /app/logs /models

# Download models at build time (if publicly accessible)
# Or mount them at runtime via volumes
RUN echo "Models will be mounted via volume or downloaded at runtime"

EXPOSE 8000

# Run as non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app /models
USER appuser

CMD ["sh", "-c", "if [ \"$MODE\" = \"dev\" ]; then uvicorn main:app --host 0.0.0.0 --port 8000 --reload; else uvicorn main:app --host 0.0.0.0 --port 8000; fi"]