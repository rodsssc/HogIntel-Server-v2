FROM python:3.11-slim as builder

# Install system dependencies needed for building Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    wget \
    gcc \
    g++ \
    cmake \
    pkg-config \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to latest version
RUN pip install --upgrade pip setuptools wheel

# Install Python dependencies in stages to better identify issues
COPY requirements.txt .

# Install dependencies with better error handling and compatibility
RUN pip install --no-cache-dir \
    fastapi==0.109.0 \
    uvicorn[standard]==0.27.0 \
    pydantic==2.5.3 \
    pydantic-settings==2.1.0 && \
    pip install --no-cache-dir \
    statsmodels==0.14.1 \
    numpy==1.26.3 \
    pillow==10.2.0 && \
    pip install --no-cache-dir \
    torch==2.1.2 \
    torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
    opencv-python-headless==4.9.0.80 \
    ultralytics>=8.0.200 && \
    pip install --no-cache-dir \
    pandas==2.1.4 \
    scikit-learn==1.4.0 \
    xgboost==2.0.3 && \
    pip install --no-cache-dir \
    prophet==1.1.5 && \
    pip install --no-cache-dir \
    python-multipart==0.0.6 \
    python-jose[cryptography]==3.3.0 \
    passlib[bcrypt]==1.7.4 \
    python-dotenv==1.0.0 \
    python-dateutil==2.8.2 \
    pytz==2023.3 \
    openpyxl==3.1.2

# Final stage
FROM python:3.11-slim

ARG MODE=prod
ENV MODE=${MODE}
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/app \
    RUNNING_IN_DOCKER=1

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY app/ ./app/

# Create directories
RUN mkdir -p /app/logs /models

# Run as non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app /models
USER appuser

EXPOSE 8000

CMD ["sh", "-c", "cd app && if [ \"$MODE\" = \"dev\" ]; then uvicorn main:app --host 0.0.0.0 --port 8000 --reload; else uvicorn main:app --host 0.0.0.0 --port 8000; fi"]