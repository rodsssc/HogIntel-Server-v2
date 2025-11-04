# app/main.py
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import urllib.request
import shutil

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
# Model Download Utility
# ============================================================
def download_file(url: str, destination: str, description: str = "file") -> bool:
    """
    Download a file from URL to destination path
    
    Args:
        url: Source URL
        destination: Destination file path
        description: Human-readable description for logging
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"📥 Downloading {description}...")
        print(f"   URL: {url}")
        print(f"   Destination: {destination}")
        
        # Create parent directory if it doesn't exist
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        
        # Download with progress indication
        def reporthook(count, block_size, total_size):
            if total_size > 0:
                percent = int(count * block_size * 100 / total_size)
                if count % 50 == 0:  # Print every ~5MB for typical block sizes
                    print(f"   Progress: {percent}%")
        
        urllib.request.urlretrieve(url, destination, reporthook=reporthook)
        print(f"✅ Successfully downloaded {description}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to download {description}: {e}")
        return False

def ensure_models_downloaded():
    """
    Download models from cloud storage if they don't exist locally
    Reads model URLs from environment variables
    """
    print("\n" + "=" * 60)
    print("🔍 Checking Model Files...")
    print("=" * 60)
    
    # Define models to check/download
    models_config = [
        {
            "env_path": "YOLO_MODEL_PATH",
            "env_url": "YOLO_MODEL_URL",
            "description": "YOLO Detection Model"
        },
        {
            "env_path": "CNN_MODEL_PATH",
            "env_url": "CNN_MODEL_URL",
            "description": "CNN Weight Prediction Model"
        },
        {
            "env_path": "PRICE_MODEL_PATH",
            "env_url": "PRICE_MODEL_URL",
            "description": "Price Prediction Model"
        },
        {
            "env_path": "PRICE_SCALER_PATH",
            "env_url": "PRICE_SCALER_URL",
            "description": "Price Scaler"
        },
        {
            "env_path": "PRICE_FEATURES_PATH",
            "env_url": "PRICE_FEATURES_URL",
            "description": "Price Features Config"
        },
        {
            "env_path": "PRICE_METRICS_PATH",
            "env_url": "PRICE_METRICS_URL",
            "description": "Price Model Metrics"
        }
    ]
    
    all_models_ready = True
    
    for model in models_config:
        local_path = os.getenv(model["env_path"])
        model_url = os.getenv(model["env_url"])
        
        if not local_path:
            print(f"⚠️  {model['description']}: Path not configured (skipping)")
            continue
        
        # Check if file already exists
        if os.path.exists(local_path):
            file_size = os.path.getsize(local_path)
            print(f"✅ {model['description']}: Found ({file_size:,} bytes)")
            print(f"   Path: {local_path}")
        else:
            print(f"❌ {model['description']}: Not found locally")
            
            # Try to download if URL is provided
            if model_url:
                success = download_file(model_url, local_path, model['description'])
                if not success:
                    all_models_ready = False
                    print(f"⚠️  WARNING: Failed to download {model['description']}")
            else:
                all_models_ready = False
                print(f"⚠️  WARNING: No download URL configured for {model['description']}")
                print(f"   Set {model['env_url']} in your .env file")
    
    print("=" * 60)
    
    if not all_models_ready:
        print("⚠️  WARNING: Some models are missing!")
        print("   The API will start but endpoints may fail.")
        print("   Please configure model URLs in your .env file:")
        print("   - YOLO_MODEL_URL")
        print("   - CNN_MODEL_URL")
        print("   - PRICE_MODEL_URL")
        print("   - PRICE_SCALER_URL")
        print("   - PRICE_FEATURES_URL")
        print("   - PRICE_METRICS_URL")
        print("=" * 60)
    else:
        print("✅ All models are ready!")
        print("=" * 60)
    
    return all_models_ready

# ============================================================
# Download models BEFORE importing other modules
# ============================================================
models_ready = ensure_models_downloaded()

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
    logger.info(f"  Models Ready: {models_ready}")
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
        "status": "healthy" if models_ready else "degraded",
        "environment": settings.ENVIRONMENT,
        "running_in_docker": in_docker,
        "models_ready": models_ready,
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
            "exists": os.path.exists(settings.YOLO_MODEL_PATH) if settings.YOLO_MODEL_PATH else False,
            "size": os.path.getsize(settings.YOLO_MODEL_PATH) if settings.YOLO_MODEL_PATH and os.path.exists(settings.YOLO_MODEL_PATH) else 0
        },
        "cnn": {
            "path": settings.CNN_MODEL_PATH,
            "exists": os.path.exists(settings.CNN_MODEL_PATH) if settings.CNN_MODEL_PATH else False,
            "size": os.path.getsize(settings.CNN_MODEL_PATH) if settings.CNN_MODEL_PATH and os.path.exists(settings.CNN_MODEL_PATH) else 0
        },
        "price": {
            "path": settings.PRICE_MODEL_PATH,
            "exists": os.path.exists(settings.PRICE_MODEL_PATH) if settings.PRICE_MODEL_PATH else False,
            "size": os.path.getsize(settings.PRICE_MODEL_PATH) if settings.PRICE_MODEL_PATH and os.path.exists(settings.PRICE_MODEL_PATH) else 0
        },
        "price_scaler": {
            "path": settings.PRICE_SCALER_PATH,
            "exists": os.path.exists(settings.PRICE_SCALER_PATH) if settings.PRICE_SCALER_PATH else False,
            "size": os.path.getsize(settings.PRICE_SCALER_PATH) if settings.PRICE_SCALER_PATH and os.path.exists(settings.PRICE_SCALER_PATH) else 0
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
        "models_loaded": all_models_loaded,
        "models_ready": models_ready
    }

@app.get("/api/v1/info")
async def api_info():
    """Get API configuration information"""
    return {
        "api": {
            "title": "HogIntel API",
            "version": "1.0.0",
            "environment": settings.ENVIRONMENT,
            "debug_mode": settings.DEBUG,
            "models_ready": models_ready
        },
        "server": {
            "host": settings.HOST,
            "port": settings.PORT,
            "running_in_docker": in_docker
        },
        "models": {
            "yolo_path": settings.YOLO_MODEL_PATH,
            "cnn_path": settings.CNN_MODEL_PATH,
            "price_path": settings.PRICE_MODEL_PATH,
            "price_scaler_path": settings.PRICE_SCALER_PATH,
            "price_features_path": settings.PRICE_FEATURES_PATH,
            "price_metrics_path": settings.PRICE_METRICS_PATH
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