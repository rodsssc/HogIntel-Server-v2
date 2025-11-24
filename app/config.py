# config.py

import os
from typing import List, Union
from pydantic_settings import BaseSettings
from pydantic import field_validator, Field
from typing import Any

class Settings(BaseSettings):
    # ========================
    # 🚀 Application Settings
    # ========================
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    API_VERSION: str = "1.2.0"  # ✅ ADDED

    # ========================
    # 🔗 API & Server Config
    # ========================
    API_V1_PREFIX: str = "/api/v1"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    REQUEST_TIMEOUT: int = 30  # ✅ ADDED
    API_RATE_LIMIT: str = "100/minute"  # ✅ ADDED

    # ========================
    # 🌐 CORS - Store as List directly
    # ========================
    ALLOWED_ORIGINS: Any = Field(default=["http://localhost:3000", "*"])

    # ========================
    # 🤖 Model Paths
    # ========================
    YOLO_MODEL_PATH: str = "/models/yolo_model/best.pt"
    CNN_MODEL_PATH: str = "/models/weight_model/best_model.pt"
    
    # SARIMA Model Paths - ✅ UPDATED
    PRICE_MODEL_PATH: str = "/models/enhanced_sarima_price_model/enhanced_sarima_model.pkl"
    PRICE_METADATA_PATH: str = "/models/enhanced_sarima_price_model/model_metadata.json"  # ✅ ADDED
    PRICE_DIAGNOSTICS_PATH: str = "/models/enhanced_sarima_price_model/enhanced_diagnostics.png"  # ✅ ADDED
    PRICE_DATA_QUALITY_PATH: str = "/models/enhanced_sarima_price_model/data_quality_report.json"  # ✅ ADDED

    # ========================
    # 📥 Model Download URLs
    # ========================
    YOLO_MODEL_URL: str = ""
    CNN_MODEL_URL: str = ""
    
    # SARIMA Model URLs - ✅ UPDATED
    SARIMA_MODEL_URL: str = ""  # ✅ ADDED
    SARIMA_METADATA_URL: str = ""  # ✅ ADDED
    SARIMA_DIAGNOSTICS_URL: str = ""  # ✅ ADDED
    SARIMA_DATA_QUALITY_URL: str = ""  # ✅ ADDED

    # ========================
    # ⚖️ Weight Prediction
    # ========================
    MIN_CONFIDENCE: float = 0.5
    TARGET_MAE: float = 3.0
    MAX_WEIGHT_KG: float = 500.0  # ✅ ADDED
    MIN_WEIGHT_KG: float = 0.0  # ✅ ADDED

    # ========================
    # 💰 SARIMA Price Prediction
    # ========================
    PRICE_MODEL_TYPE: str = "sarima"  # ✅ UPDATED
    PRICE_ERROR_THRESHOLD: float = 0.10
    PRICE_DATA_CSV: str = "data/pig-price.csv"
    
    # SARIMA Configuration - ✅ ADDED
    SARIMA_MIN_DATA_POINTS: int = 12
    SARIMA_RECOMMENDED_POINTS: int = 24
    SARIMA_MAX_FORECAST_DAYS: int = 365
    DEFAULT_CONFIDENCE: float = 0.85
    MIN_PRICE_PER_KG: float = 120.0
    MAX_PRICE_PER_KG: float = 250.0

    # ========================
    # 🖼️ Image Processing
    # ========================
    MAX_IMAGE_SIZE: int = 1024
    CROP_PADDING: int = 20
    SUPPORTED_FORMATS: str = "jpeg,jpg,png"  # ✅ ADDED
    MAX_IMAGE_FILE_SIZE: int = 10485760  # 10MB - ✅ ADDED

    # ========================
    # 📊 Data & Analytics
    # ========================
    HISTORICAL_DATA_RETENTION_DAYS: int = 730  # ✅ ADDED
    MIN_HISTORICAL_RECORDS: int = 12  # ✅ ADDED
    MAX_BATCH_REQUESTS: int = 100  # ✅ ADDED
    ANALYTICS_ENABLED: bool = True  # ✅ ADDED

    # ========================
    # 📝 Logging & Monitoring
    # ========================
    LOG_FILE: str = "/app/logs/hogintel.log"
    LOG_RETENTION_DAYS: int = 30  # ✅ ADDED
    METRICS_ENABLED: bool = True  # ✅ ADDED
    HEALTH_CHECK_INTERVAL: int = 60  # ✅ ADDED

    # ========================
    # 🔐 Security
    # ========================
    API_KEY_HEADER: str = "X-API-Key"  # ✅ ADDED
    CORS_MAX_AGE: int = 600  # ✅ ADDED

    # ========================
    # 🎯 Development Features
    # ========================
    ENABLE_SWAGGER: bool = True  # ✅ ADDED
    ENABLE_AUTO_RELOAD: bool = True  # ✅ ADDED
    ENABLE_SYNTHETIC_DATA: bool = True  # ✅ ADDED
    DEBUG_PREDICTIONS: bool = True  # ✅ ADDED

    @field_validator('ALLOWED_ORIGINS', mode='before')
    @classmethod
    def parse_origins(cls, v: Union[str, List[str], None]) -> List[str]:
        """
        Parse comma-separated origins string into a list.
        Handles empty strings and None values gracefully.
        """
        # If already a list, return it
        if isinstance(v, list):
            return v
        
        # If None or empty string, return default
        if v is None or v == "" or v.strip() == "":
            return ["http://localhost:3000", "http://localhost:8080", "*"]
        
        # If string, split by comma
        if isinstance(v, str):
            origins = [origin.strip() for origin in v.split(",") if origin.strip()]
            return origins if origins else ["*"]
        
        # Fallback
        return ["*"]

    @field_validator('DEBUG', mode='before')
    @classmethod
    def parse_debug(cls, v: Union[str, bool, None]) -> bool:
        """Parse debug boolean from string"""
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.lower() in ('true', '1', 'yes')
        return False

    @field_validator('ENABLE_SWAGGER', 'ENABLE_AUTO_RELOAD', 'ENABLE_SYNTHETIC_DATA', 'DEBUG_PREDICTIONS', 
                    'ANALYTICS_ENABLED', 'METRICS_ENABLED', mode='before')
    @classmethod
    def parse_boolean_flags(cls, v: Union[str, bool, None]) -> bool:
        """Parse boolean flags from string"""
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.lower() in ('true', '1', 'yes', 'enabled')
        return False

    @field_validator('SARIMA_MIN_DATA_POINTS', 'SARIMA_RECOMMENDED_POINTS', 'SARIMA_MAX_FORECAST_DAYS',
                    'MIN_HISTORICAL_RECORDS', 'MAX_BATCH_REQUESTS', 'LOG_RETENTION_DAYS', 
                    'HEALTH_CHECK_INTERVAL', 'CORS_MAX_AGE', 'REQUEST_TIMEOUT', mode='before')
    @classmethod
    def parse_positive_integers(cls, v: Union[str, int, None]) -> int:
        """Parse positive integers from string"""
        if isinstance(v, int):
            return max(0, v)
        if isinstance(v, str):
            try:
                return max(0, int(v))
            except ValueError:
                return 0
        return 0

    @field_validator('MIN_CONFIDENCE', 'TARGET_MAE', 'PRICE_ERROR_THRESHOLD', 'DEFAULT_CONFIDENCE',
                    'MAX_WEIGHT_KG', 'MIN_WEIGHT_KG', 'MIN_PRICE_PER_KG', 'MAX_PRICE_PER_KG', mode='before')
    @classmethod
    def parse_positive_floats(cls, v: Union[str, float, None]) -> float:
        """Parse positive floats from string"""
        if isinstance(v, float):
            return max(0.0, v)
        if isinstance(v, (int, str)):
            try:
                return max(0.0, float(v))
            except ValueError:
                return 0.0
        return 0.0

    class Config:
        # Get the directory where config.py is located
        env_file = os.path.join(os.path.dirname(__file__), ".env")
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"


# Initialize settings - wrap in try/except for better error messages
try:
    settings = Settings()
    print(f"✅ Settings loaded successfully from: {os.path.join(os.path.dirname(__file__), '.env')}")
    print(f"🌍 Environment: {settings.ENVIRONMENT}")
    print(f"🔧 Debug Mode: {settings.DEBUG}")
    print(f"🤖 Price Model Type: {settings.PRICE_MODEL_TYPE}")
except Exception as e:
    print(f"❌ Error loading settings: {e}")
    print(f"📁 Looking for .env in: {os.path.dirname(__file__)}")
    print(f"📂 Current working directory: {os.getcwd()}")
    
    # Try to load with defaults
    print("🔄 Attempting to load with default settings...")
    settings = Settings(_env_file=None)  # Force use of defaults