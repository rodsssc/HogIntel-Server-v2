"""
HogIntel Configuration
Environment variables and application settings
"""
from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    # Environment
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    
    # API Settings
    API_V1_PREFIX: str = "/api/v1"
    ALLOWED_ORIGINS: List[str] = ["*"]
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    
    # Model Paths
    YOLO_MODEL_PATH: str = r"C:\Users\Acer\OneDrive\Desktop\HogIntel-Price&Weight-Estimation\models\weights\best.pt"
    CNN_MODEL_PATH: str = r"C:\Users\Acer\OneDrive\Desktop\HogIntel-Price&Weight-Estimation\models\outputs\training_run1\best_model.pt"
    PRICE_MODEL_PATH: str = r"C:\Users\Acer\OneDrive\Desktop\HogIntel-Price&Weight-Estimation\models\improved_price_model\best_price_model.pkl"
    
    # Calibration
    CAMERA_HEIGHT: float = 1.5  # meters
    MARKER_SIZE: float = 0.3  # meters (calibration marker size)
    
    # Weight Prediction Thresholds
    MIN_CONFIDENCE: float = 0.5  # Minimum detection confidence
    TARGET_MAE: float = 3.0  # Target MAE in kg
    
    # Price Model Settings
    PRICE_MODEL_TYPE: str = "xgboost"  # "xgboost" or "prophet"
    PRICE_ERROR_THRESHOLD: float = 0.10  # 10% error threshold
    

    PRICE_DATA_CSV: str = "data/pig-price.csv"
    CALIBRATION_FILE: str = "data/calibration.json"
    
    # Image Processing
    MAX_IMAGE_SIZE: int = 1024  # Max dimension for processing
    CROP_PADDING: int = 20  # Padding around ROI in pixels
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/hogintel.log"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"

settings = Settings()