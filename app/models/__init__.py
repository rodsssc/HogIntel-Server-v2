"""
Models package initialization
Location: app/models/__init__.py
"""

from models.yolo_detector import HogDetector
from models.cnn_regressor import WeightRegressor
from models.price_model import PricePredictor

__all__ = [
    "HogDetector",
    "WeightRegressor",
    "PricePredictor",
]
