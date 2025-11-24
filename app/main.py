# app/main.py
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import urllib.request
import shutil
import time
from typing import Dict, List, Optional

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
# Enhanced Model Download Utility
# ============================================================
class ModelDownloader:
    """Enhanced model downloader with retry logic and progress tracking"""
    
    def __init__(self):
        self.download_attempts = {}
        self.max_retries = 3
        self.retry_delay = 5  # seconds
    
    def download_file(self, url: str, destination: str, description: str = "file") -> bool:
        """
        Download a file from URL to destination path with retry logic
        
        Args:
            url: Source URL
            destination: Destination file path
            description: Human-readable description for logging
        
        Returns:
            bool: True if successful, False otherwise
        """
        if url not in self.download_attempts:
            self.download_attempts[url] = 0
        
        while self.download_attempts[url] < self.max_retries:
            try:
                self.download_attempts[url] += 1
                print(f"📥 Downloading {description} (attempt {self.download_attempts[url]}/{self.max_retries})...")
                print(f"   URL: {url}")
                print(f"   Destination: {destination}")
                
                # Create parent directory if it doesn't exist
                Path(destination).parent.mkdir(parents=True, exist_ok=True)
                
                # Download with progress indication
                def reporthook(count, block_size, total_size):
                    if total_size > 0:
                        percent = int(count * block_size * 100 / total_size)
                        if count % 50 == 0 or percent == 100:  # Print every ~5MB and at completion
                            size_mb = (count * block_size) / (1024 * 1024)
                            total_mb = total_size / (1024 * 1024)
                            print(f"   Progress: {percent}% ({size_mb:.1f}/{total_mb:.1f} MB)")
                
                urllib.request.urlretrieve(url, destination, reporthook=reporthook)
                
                # Verify file was downloaded successfully
                if os.path.exists(destination) and os.path.getsize(destination) > 0:
                    file_size = os.path.getsize(destination) / (1024 * 1024)  # MB
                    print(f"✅ Successfully downloaded {description} ({file_size:.1f} MB)")
                    return True
                else:
                    print(f"⚠️  Downloaded file is empty or missing: {destination}")
                    if os.path.exists(destination):
                        os.remove(destination)  # Clean up corrupted file
                    
            except Exception as e:
                print(f"❌ Download attempt {self.download_attempts[url]} failed: {e}")
                
                # Clean up partially downloaded file
                if os.path.exists(destination):
                    os.remove(destination)
                
                # Wait before retry
                if self.download_attempts[url] < self.max_retries:
                    print(f"⏳ Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
        
        print(f"💥 Failed to download {description} after {self.max_retries} attempts")
        return False

def ensure_models_downloaded():
    """
    Download models from cloud storage if they don't exist locally
    Enhanced for SARIMA model support
    """
    print("\n" + "=" * 60)
    print("🔍 Checking Model Files...")
    print("=" * 60)
    
    # Define models to check/download - Updated for SARIMA
    models_config = [
        # Weight Estimation Models
        {
            "env_path": "YOLO_MODEL_PATH",
            "env_url": "YOLO_MODEL_URL",
            "description": "YOLO Detection Model",
            "required": True
        },
        {
            "env_path": "CNN_MODEL_PATH",
            "env_url": "CNN_MODEL_URL",
            "description": "CNN Weight Prediction Model",
            "required": True
        },
        # SARIMA Price Prediction Models
        {
            "env_path": "PRICE_MODEL_PATH",
            "env_url": "SARIMA_MODEL_URL",
            "description": "SARIMA Price Prediction Model",
            "required": True
        },
        {
            "env_path": "PRICE_METADATA_PATH",
            "env_url": "SARIMA_METADATA_URL",
            "description": "SARIMA Model Metadata",
            "required": True
        },
        {
            "env_path": "PRICE_DIAGNOSTICS_PATH",
            "env_url": "SARIMA_DIAGNOSTICS_URL",
            "description": "SARIMA Diagnostics Report",
            "required": False
        },
        {
            "env_path": "PRICE_DATA_QUALITY_PATH",
            "env_url": "SARIMA_DATA_QUALITY_URL",
            "description": "SARIMA Data Quality Report",
            "required": False
        }
    ]
    
    downloader = ModelDownloader()
    all_models_ready = True
    required_models_missing = False
    
    for model in models_config:
        local_path = os.getenv(model["env_path"])
        model_url = os.getenv(model["env_url"])
        is_required = model.get("required", True)
        
        if not local_path:
            print(f"⚠️  {model['description']}: Path not configured in {model['env_path']}")
            if is_required:
                all_models_ready = False
                required_models_missing = True
            continue
        
        # Check if file already exists
        if os.path.exists(local_path):
            file_size = os.path.getsize(local_path) / (1024 * 1024)  # MB
            print(f"✅ {model['description']}: Found ({file_size:.1f} MB)")
            print(f"   Path: {local_path}")
            
            # Verify file is not corrupted (basic check)
            if file_size == 0:
                print(f"⚠️  File is empty, will re-download: {local_path}")
                os.remove(local_path)
                if model_url:
                    success = downloader.download_file(model_url, local_path, model['description'])
                    if not success and is_required:
                        all_models_ready = False
                        required_models_missing = True
        else:
            print(f"❌ {model['description']}: Not found locally")
            print(f"   Expected: {local_path}")
            
            # Try to download if URL is provided
            if model_url:
                success = downloader.download_file(model_url, local_path, model['description'])
                if not success and is_required:
                    all_models_ready = False
                    required_models_missing = True
            else:
                print(f"⚠️  No download URL configured for {model['description']}")
                print(f"   Set {model['env_url']} in your .env file")
                if is_required:
                    all_models_ready = False
                    required_models_missing = True
    
    print("=" * 60)
    
    if required_models_missing:
        print("💥 CRITICAL: Required models are missing!")
        print("   The API will start but price prediction endpoints will fail.")
        print("   Please configure these environment variables:")
        print("   - YOLO_MODEL_URL")
        print("   - CNN_MODEL_URL")
        print("   - SARIMA_MODEL_URL")
        print("   - SARIMA_METADATA_URL")
        print("=" * 60)
    elif not all_models_ready:
        print("⚠️  WARNING: Some optional models are missing!")
        print("   The API will start but some features may be limited.")
        print("=" * 60)
    else:
        print("✅ All models are ready!")
        print("=" * 60)
    
    return all_models_ready

def validate_model_files() -> Dict[str, bool]:
    """
    Validate that all model files exist and are accessible
    Returns dictionary with validation results
    """
    validation_results = {}
    
    # Model paths to validate
    model_paths = {
        "yolo_model": os.getenv("YOLO_MODEL_PATH"),
        "cnn_model": os.getenv("CNN_MODEL_PATH"),
        "sarima_model": os.getenv("PRICE_MODEL_PATH"),
        "sarima_metadata": os.getenv("PRICE_METADATA_PATH"),
        "sarima_diagnostics": os.getenv("PRICE_DIAGNOSTICS_PATH"),
        "sarima_data_quality": os.getenv("PRICE_DATA_QUALITY_PATH")
    }
    
    print("\n" + "=" * 60)
    print("🔍 Validating Model Files...")
    print("=" * 60)
    
    for model_name, model_path in model_paths.items():
        if not model_path:
            print(f"❌ {model_name}: Path not configured")
            validation_results[model_name] = False
            continue
            
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path)
            status = "✅" if file_size > 0 else "❌"
            print(f"{status} {model_name}: Found ({file_size:,} bytes) - {model_path}")
            validation_results[model_name] = file_size > 0
        else:
            print(f"❌ {model_name}: Not found - {model_path}")
            validation_results[model_name] = False
    
    print("=" * 60)
    return validation_results

# ============================================================
# Download and validate models BEFORE importing other modules
# ============================================================
print("\n" + "🚀 Initializing HogIntel API Server")
print("=" * 50)

# Download models if needed
models_ready = ensure_models_downloaded()

# Validate model files
model_validation = validate_model_files()

# Check if critical models are available
critical_models = ["yolo_model", "cnn_model", "sarima_model", "sarima_metadata"]
critical_models_ready = all(model_validation.get(model, False) for model in critical_models)

# ============================================================
# Now import everything else
# ============================================================
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
import psutil
import datetime

from config import settings 
from logger import setup_logger
from routers import scan, confirm, price

# Setup logger
logger = setup_logger(__name__)

# ============================================================
# Enhanced Application Lifespan
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Enhanced lifespan events for startup and shutdown"""
    # Startup
    startup_time = datetime.datetime.now()
    logger.info("=" * 60)
    logger.info("🚀 Starting HogIntel API Server")
    logger.info("=" * 60)
    logger.info(f"📅 Startup Time: {startup_time}")
    logger.info(f"🌍 Environment: {settings.ENVIRONMENT}")
    logger.info(f"🐛 Debug mode: {settings.DEBUG}")
    logger.info(f"🐳 Running in Docker: {in_docker}")
    logger.info(f"🌐 Host: {settings.HOST}:{settings.PORT}")
    logger.info(f"🔗 CORS origins: {settings.ALLOWED_ORIGINS}")
    logger.info(f"📊 API Version: {settings.API_VERSION}")
    logger.info("-" * 60)
    logger.info("🤖 Model Status:")
    logger.info(f"  YOLO: {settings.YOLO_MODEL_PATH} - {'✅' if model_validation.get('yolo_model') else '❌'}")
    logger.info(f"  CNN: {settings.CNN_MODEL_PATH} - {'✅' if model_validation.get('cnn_model') else '❌'}")
    logger.info(f"  SARIMA: {settings.PRICE_MODEL_PATH} - {'✅' if model_validation.get('sarima_model') else '❌'}")
    logger.info(f"  SARIMA Metadata: {settings.PRICE_METADATA_PATH} - {'✅' if model_validation.get('sarima_metadata') else '❌'}")
    logger.info(f"  Critical Models Ready: {'✅' if critical_models_ready else '❌'}")
    logger.info(f"  All Models Ready: {'✅' if models_ready else '❌'}")
    logger.info("-" * 60)
    logger.info("⚙️  Configuration:")
    logger.info(f"  SARIMA Min Data Points: {settings.SARIMA_MIN_DATA_POINTS}")
    logger.info(f"  SARIMA Max Forecast Days: {settings.SARIMA_MAX_FORECAST_DAYS}")
    logger.info(f"  Price Range: ₱{settings.MIN_PRICE_PER_KG} - ₱{settings.MAX_PRICE_PER_KG}")
    logger.info("=" * 60)
    
    # Record startup metrics
    startup_metrics = {
        "startup_time": startup_time,
        "critical_models_ready": critical_models_ready,
        "all_models_ready": models_ready,
        "model_validation": model_validation
    }
    
    app.state.startup_metrics = startup_metrics
    app.state.startup_time = startup_time
    app.state.request_count = 0
    
    yield
    
    # Shutdown
    shutdown_time = datetime.datetime.now()
    uptime = shutdown_time - startup_time
    logger.info("=" * 60)
    logger.info("🛑 Shutting down HogIntel API Server")
    logger.info(f"📅 Shutdown Time: {shutdown_time}")
    logger.info(f"⏱️  Total Uptime: {uptime}")
    logger.info(f"📊 Total Requests: {getattr(app.state, 'request_count', 0)}")
    logger.info("=" * 60)

# ============================================================
# Enhanced FastAPI Application
# ============================================================
app = FastAPI(
    title="HogIntel API",
    description="""Pig Weight & Price Estimation API using Computer Vision and Time Series Forecasting.
    
## Features
- 🐖 **Weight Estimation**: YOLOv8 detection + CNN regression for accurate weight prediction
- 📈 **Price Forecasting**: SARIMA time series model for market price predictions
- 🔍 **Batch Processing**: Support for multiple predictions in single request
- 📊 **Analytics**: Comprehensive market analysis and trend detection
- 🏥 **Health Monitoring**: Real-time system health and model status

## Models
- **YOLOv8**: Object detection for hog localization
- **CNN Regressor**: Weight prediction from detected regions
- **SARIMA**: Seasonal ARIMA for price time series forecasting
""",
    version=settings.API_VERSION,
    lifespan=lifespan,
    docs_url="/docs" if settings.ENABLE_SWAGGER else None,
    redoc_url="/redoc" if settings.ENABLE_SWAGGER else None,
    contact={
        "name": "HogIntel Team",
        "email": "support@hogintel.com",
    },
    license_info={
        "name": "Proprietary",
        "url": "https://hogintel.com/license",
    }
)

# ============================================================
# Enhanced Middleware
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    max_age=settings.CORS_MAX_AGE,
)

# Trusted Host middleware for security
if settings.ENVIRONMENT == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]  # Configure specific hosts in production
    )

# Request counting middleware
@app.middleware("http")
async def count_requests(request: Request, call_next):
    app.state.request_count = getattr(app.state, 'request_count', 0) + 1
    response = await call_next(request)
    return response

# ============================================================
# Include Routers
# ============================================================
app.include_router(scan.router, prefix="/api/v1", tags=["Weight Estimation"])
app.include_router(confirm.router, prefix="/api/v1", tags=["Weight Confirmation"])
app.include_router(price.router, prefix="/api/v1", tags=["Price Prediction"])

# ============================================================
# Enhanced Root Endpoints
# ============================================================
@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint with comprehensive API information"""
    current_time = datetime.datetime.now()
    uptime = current_time - getattr(app.state, 'startup_time', current_time)
    
    return {
        "message": "🚀 HogIntel API Server - Pig Weight & Price Estimation",
        "version": settings.API_VERSION,
        "status": "healthy" if critical_models_ready else "degraded",
        "environment": settings.ENVIRONMENT,
        "timestamp": current_time.isoformat(),
        "uptime_seconds": uptime.total_seconds(),
        "running_in_docker": in_docker,
        "models_status": {
            "critical_models_ready": critical_models_ready,
            "all_models_ready": models_ready,
            "validation_details": model_validation
        },
        "endpoints": {
            "documentation": "/docs",
            "health": "/health",
            "system": "/system",
            "weight_estimation": "/api/v1/scan",
            "weight_confirmation": "/api/v1/confirm",
            "price_prediction": "/api/v1/price",
            "batch_prediction": "/api/v1/price/batch",
            "price_forecast": "/api/v1/price/forecast"
        }
    }

@app.get("/health", tags=["System"])
async def health_check():
    """Enhanced health check endpoint with system metrics"""
    current_time = datetime.datetime.now()
    uptime = current_time - getattr(app.state, 'startup_time', current_time)
    
    # System metrics
    memory = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=0.1)
    disk_usage = psutil.disk_usage('/')
    
    # Enhanced model status
    model_status = {
        "yolo": {
            "path": settings.YOLO_MODEL_PATH,
            "exists": model_validation.get('yolo_model', False),
            "size_mb": os.path.getsize(settings.YOLO_MODEL_PATH) / (1024 * 1024) if model_validation.get('yolo_model') else 0,
            "required": True
        },
        "cnn": {
            "path": settings.CNN_MODEL_PATH,
            "exists": model_validation.get('cnn_model', False),
            "size_mb": os.path.getsize(settings.CNN_MODEL_PATH) / (1024 * 1024) if model_validation.get('cnn_model') else 0,
            "required": True
        },
        "sarima": {
            "path": settings.PRICE_MODEL_PATH,
            "exists": model_validation.get('sarima_model', False),
            "size_mb": os.path.getsize(settings.PRICE_MODEL_PATH) / (1024 * 1024) if model_validation.get('sarima_model') else 0,
            "required": True
        },
        "sarima_metadata": {
            "path": settings.PRICE_METADATA_PATH,
            "exists": model_validation.get('sarima_metadata', False),
            "size_mb": os.path.getsize(settings.PRICE_METADATA_PATH) / (1024 * 1024) if model_validation.get('sarima_metadata') else 0,
            "required": True
        }
    }
    
    # Overall health status
    all_critical_models_loaded = all(
        model_status[model]["exists"] 
        for model in ["yolo", "cnn", "sarima", "sarima_metadata"]
    )
    
    health_status = "healthy" if all_critical_models_loaded else "degraded"
    
    return {
        "status": health_status,
        "service": "hogintel-api",
        "version": settings.API_VERSION,
        "timestamp": current_time.isoformat(),
        "uptime_seconds": uptime.total_seconds(),
        "environment": settings.ENVIRONMENT,
        "system": {
            "memory_usage_percent": memory.percent,
            "memory_available_mb": memory.available / (1024 * 1024),
            "cpu_usage_percent": cpu_percent,
            "disk_usage_percent": disk_usage.percent,
            "total_requests": getattr(app.state, 'request_count', 0)
        },
        "models": model_status,
        "critical_models_ready": all_critical_models_loaded,
        "all_models_ready": models_ready
    }

@app.get("/system", tags=["System"])
async def system_info():
    """Comprehensive system information endpoint"""
    current_time = datetime.datetime.now()
    uptime = current_time - getattr(app.state, 'startup_time', current_time)
    
    # System information
    memory = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=0.1)
    disk_usage = psutil.disk_usage('/')
    
    return {
        "api": {
            "title": "HogIntel API",
            "version": settings.API_VERSION,
            "environment": settings.ENVIRONMENT,
            "debug_mode": settings.DEBUG,
            "startup_time": getattr(app.state, 'startup_time', current_time).isoformat(),
            "uptime_seconds": uptime.total_seconds(),
            "total_requests": getattr(app.state, 'request_count', 0)
        },
        "server": {
            "host": settings.HOST,
            "port": settings.PORT,
            "running_in_docker": in_docker,
            "python_version": sys.version
        },
        "system": {
            "memory": {
                "total_mb": memory.total / (1024 * 1024),
                "available_mb": memory.available / (1024 * 1024),
                "used_percent": memory.percent
            },
            "cpu": {
                "cores": psutil.cpu_count(),
                "usage_percent": cpu_percent
            },
            "disk": {
                "total_gb": disk_usage.total / (1024 * 1024 * 1024),
                "used_gb": disk_usage.used / (1024 * 1024 * 1024),
                "free_gb": disk_usage.free / (1024 * 1024 * 1024),
                "usage_percent": disk_usage.percent
            }
        },
        "models": {
            "yolo_path": settings.YOLO_MODEL_PATH,
            "cnn_path": settings.CNN_MODEL_PATH,
            "sarima_path": settings.PRICE_MODEL_PATH,
            "sarima_metadata_path": settings.PRICE_METADATA_PATH,
            "sarima_diagnostics_path": settings.PRICE_DIAGNOSTICS_PATH,
            "sarima_data_quality_path": settings.PRICE_DATA_QUALITY_PATH
        },
        "configuration": {
            "weight_estimation": {
                "min_confidence": settings.MIN_CONFIDENCE,
                "target_mae": settings.TARGET_MAE,
                "max_weight_kg": settings.MAX_WEIGHT_KG,
                "min_weight_kg": settings.MIN_WEIGHT_KG
            },
            "price_prediction": {
                "model_type": settings.PRICE_MODEL_TYPE,
                "error_threshold": settings.PRICE_ERROR_THRESHOLD,
                "sarima_min_data_points": settings.SARIMA_MIN_DATA_POINTS,
                "sarima_recommended_points": settings.SARIMA_RECOMMENDED_POINTS,
                "sarima_max_forecast_days": settings.SARIMA_MAX_FORECAST_DAYS,
                "min_price_per_kg": settings.MIN_PRICE_PER_KG,
                "max_price_per_kg": settings.MAX_PRICE_PER_KG
            },
            "image_processing": {
                "max_image_size": settings.MAX_IMAGE_SIZE,
                "crop_padding": settings.CROP_PADDING,
                "supported_formats": settings.SUPPORTED_FORMATS,
                "max_image_file_size": settings.MAX_IMAGE_FILE_SIZE
            }
        }
    }

@app.get("/api/v1/info", tags=["System"])
async def api_info():
    """Legacy API info endpoint - redirects to /system"""
    return await system_info()

# ============================================================
# Global Exception Handler
# ============================================================
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for uncaught exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "code": "INTERNAL_ERROR",
            "details": str(exc) if settings.DEBUG else "An internal error occurred",
            "timestamp": datetime.datetime.now().isoformat()
        }
    )

# ============================================================
# Main Entry Point
# ============================================================
if __name__ == "__main__":
    logger.info("Starting server via uvicorn...")
    
    uvicorn_config = {
        "app": "main:app",
        "host": settings.HOST,
        "port": settings.PORT,
        "log_level": "info",
        "access_log": True
    }
    
    # Development-specific settings
    if settings.DEBUG:
        uvicorn_config.update({
            "reload": settings.ENABLE_AUTO_RELOAD,
            "reload_dirs": ["app"],
            "reload_excludes": ["*.pyc", "*.tmp", "logs/*"]
        })
    
    # Production-specific settings
    if settings.ENVIRONMENT == "production":
        uvicorn_config.update({
            "workers": 1,  # Can be increased based on available CPU cores
            "timeout_keep_alive": 5,
            "limit_max_requests": 1000
        })
    
    uvicorn.run(**uvicorn_config)