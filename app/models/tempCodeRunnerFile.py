# app/models/price_model.py
import joblib
import pandas as pd
from logger import setup_logger

logger = setup_logger(__name__)

class PricePredictor:
    def __init__(self, model_path=r'C:\Users\Acer\OneDrive\Desktop\HogIntel-Price&Weight-Estimation\models\pig_price_model.pkl'):
        try:
            self.model = joblib.load(model_path)
            logger.info(f"Price model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load price model: {e}")
            self.model = None
    
    def predict_price(self, weight, features=None):
        """
        Predict price per kg based on weight and market features
        Returns: price_per_kg, total_value, confidence
        """
        if self.model is None:
            # Fallback to simple lookup
            return self._fallback_price(weight)
        
        try:
            # Prepare features
            data = {'weight': weight}
            if features:
                data.update(features)
            
            df = pd.DataFrame([data])
            price_per_kg = self.model.predict(df)[0]
            total_value = price_per_kg * weight
            
            return price_per_kg, total_value, 0.88
        except Exception as e:
            logger.error(f"Price prediction failed: {e}")
            return self._fallback_price(weight)
    
    def _fallback_price(self, weight):
        """Simple fallback pricing"""
        base_price = 180.0  # PHP per kg
        total = base_price * weight
        return base_price, total, 0.75