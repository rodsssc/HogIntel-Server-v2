# app/main.py
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# ============================================================
# Environment Detection & Configuration Loading
# ============================================================
# Determine which .env file to load based on environment
project_root = Path(__file__).parent.parent

# Check if running in Docker
in_docker = os.path.exists('/.dockerenv') or os.environ.get('RUNNING_IN_DOCKER') == '1'

if in_docker:
    env_file = project_root / '.env'
    print(f"🐳 Running in Docker - Loading: {env_file}")
else:
    # Running locally - prefer .env.local if it exists
    local_env = project_root / '.env.local'
    if local_env.exists():
        env_file = local_env
        print(f"💻 Running locally - Loading: {env_file}")
    else:
        env_file = project_root / '.env'
        print(f"⚠️  .env.local not found, falling back to: {env_file}")

# Load the appropriate environment file
load_dotenv(env_file, override=True)
print(f"✅ Environment variables loaded from: {env_file}")

# ============================================================
# Now import everything else
# ============================================================
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import uvicorn

from config import settings 
from logger import setup_logger
from routers import scan, confirm, price

# Setup logger
logger = setup_logger(__name__)

# ============================================================
# Application Lifespan
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events for startup and shutdown"""
    # Startup
    logger.info("=" * 60)
    logger.info("Starting HogIntel API Server")
    logger.info("=" * 60)
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Debug mode: {settings.DEBUG}")
    logger.info(f"Running in Docker: {in_docker}")
    logger.info(f"Host: {settings.HOST}:{settings.PORT}")
    logger.info(f"CORS origins: {settings.ALLOWED_ORIGINS}")
    logger.info("-" * 60)
    logger.info("Model Paths:")
    logger.info(f"  YOLO: {settings.YOLO_MODEL_PATH}")
    logger.info(f"  CNN: {settings.CNN_MODEL_PATH}")
    logger.info(f"  Price: {settings.PRICE_MODEL_PATH}")
    logger.info("=" * 60)
    
    yield
    
    # Shutdown
    logger.info("=" * 60)
    logger.info("Shutting down HogIntel API Server")
    logger.info("=" * 60)

# ============================================================
# FastAPI Application
# ============================================================
app = FastAPI(
    title="HogIntel API",
    description="Pig Weight & Price Estimation API using YOLOv8, CNN Regressor, and Ridge Regression",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# ============================================================
# CORS Middleware
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Include Routers
# ============================================================
app.include_router(scan.router, prefix="/api/v1", tags=["scan"])
app.include_router(confirm.router, prefix="/api/v1", tags=["confirm"])
app.include_router(price.router, prefix="/api/v1", tags=["price"])

# ============================================================
# Root Endpoints
# ============================================================
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "HogIntel API Server",
        "version": "1.0.0",
        "status": "healthy",
        "environment": settings.ENVIRONMENT,
        "running_in_docker": in_docker,
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/health",
            "scan": "/api/v1/scan",
            "confirm": "/api/v1/confirm",
            "price": "/api/v1/price"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint with model status"""
    
    # Check if model files exist
    model_status = {
        "yolo": {
            "path": settings.YOLO_MODEL_PATH,
            "exists": os.path.exists(settings.YOLO_MODEL_PATH) if settings.YOLO_MODEL_PATH else False
        },
        "cnn": {
            "path": settings.CNN_MODEL_PATH,
            "exists": os.path.exists(settings.CNN_MODEL_PATH) if settings.CNN_MODEL_PATH else False
        },
        "price": {
            "path": settings.PRICE_MODEL_PATH,
            "exists": os.path.exists(settings.PRICE_MODEL_PATH) if settings.PRICE_MODEL_PATH else False
        }
    }
    
    # Overall health status
    all_models_loaded = all(model["exists"] for model in model_status.values())
    
    return {
        "status": "healthy" if all_models_loaded else "degraded",
        "service": "hogintel-api",
        "version": "1.0.0",
        "environment": settings.ENVIRONMENT,
        "running_in_docker": in_docker,
        "models": model_status,
        "models_loaded": all_models_loaded
    }

@app.get("/api/v1/info")
async def api_info():
    """Get API configuration information"""
    return {
        "api": {
            "title": "HogIntel API",
            "version": "1.0.0",
            "environment": settings.ENVIRONMENT,
            "debug_mode": settings.DEBUG
        },
        "server": {
            "host": settings.HOST,
            "port": settings.PORT,
            "running_in_docker": in_docker
        },
        "models": {
            "yolo_path": settings.YOLO_MODEL_PATH,
            "cnn_path": settings.CNN_MODEL_PATH,
            "price_path": settings.PRICE_MODEL_PATH
        },
        "config": {
            "min_confidence": settings.MIN_CONFIDENCE,
            "target_mae": settings.TARGET_MAE,
            "price_error_threshold": settings.PRICE_ERROR_THRESHOLD,
            "max_image_size": settings.MAX_IMAGE_SIZE
        }
    }

# ============================================================
# Main Entry Point
# ============================================================
if __name__ == "__main__":
    logger.info("Starting server via uvicorn...")
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )