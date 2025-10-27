"""
HogIntel Configuration
Environment variables and application settings
"""

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

    # ========================
    # 🔗 API & Server Config
    # ========================
    API_V1_PREFIX: str = "/api/v1"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True

    # ========================
    # 🌐 CORS - Store as List directly
    # ========================
    ALLOWED_ORIGINS: Any = Field(default=["http://localhost:3000", "*"])

    # ========================
    # 🤖 Model Paths
    # ========================
    YOLO_MODEL_PATH: str = r"C:\Users\Acer\OneDrive\Desktop\HogIntel-Price&Weight-Estimation\models\yolo_model\best.pt"
    CNN_MODEL_PATH: str = r"C:\Users\Acer\OneDrive\Desktop\HogIntel-Price&Weight-Estimation\models\weight_model\best_model.pt"
    PRICE_MODEL_PATH: str = r"C:\Users\Acer\OneDrive\Desktop\HogIntel-Price&Weight-Estimation\models\price_model\best_price_model.pkl"

    # ========================
    # ⚖️ Weight Prediction
    # ========================
    MIN_CONFIDENCE: float = 0.5
    TARGET_MAE: float = 3.0

    # ========================
    # 💰 Price Prediction
    # ========================
    PRICE_MODEL_TYPE: str = "ridgeregressor"
    PRICE_ERROR_THRESHOLD: float = 0.10
    PRICE_DATA_CSV: str = "data/pig-price.csv"

    # ========================
    # 🖼️ Image Processing
    # ========================
    MAX_IMAGE_SIZE: int = 1024
    CROP_PADDING: int = 20

    # ========================
    # 📝 Logging
    # ========================
    LOG_FILE: str = "logs/hogintel.log"

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

    class Config:
        # Get the directory where config.py is located
        env_file = os.path.join(os.path.dirname(__file__), ".env")
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"


# Initialize settings - wrap in try/except for better error messages
try:
    settings = Settings()
except Exception as e:
    print(f"❌ Error loading settings: {e}")
    print(f"📁 Looking for .env in: {os.path.dirname(__file__)}")
    print(f"📂 Current working directory: {os.getcwd()}")
    raise