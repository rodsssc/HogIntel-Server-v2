# app/models/price_model.py
import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path
from logger import setup_logger
from typing import Dict, Optional, Any, List
from dataclasses import dataclass
from datetime import datetime

logger = setup_logger(__name__)

@dataclass
class PricePrediction:
    """Structure for price prediction results"""
    price_per_kg: float
    confidence: float
    model_used: str
    features_used: Optional[List[str]] = None
    market_conditions: Optional[Dict[str, Any]] = None
    prediction_metadata: Optional[Dict[str, Any]] = None

class PricePredictor:
    """
    Enhanced Price Predictor using Ridge Regression
    Supports feature engineering and multiple fallback strategies
    """
    
    def __init__(self, 
                 model_path: str = r'C:\Users\Acer\OneDrive\Desktop\HogIntel-Price&Weight-Estimation\models\improved_price_model\best_price_model.pkl',
                 scaler_path: str = r'C:\Users\Acer\OneDrive\Desktop\HogIntel-Price&Weight-Estimation\models\improved_price_model\price_scaler.pkl',
                 features_path: str = r'C:\Users\Acer\OneDrive\Desktop\HogIntel-Price&Weight-Estimation\models\improved_price_model\selected_features.json',
                 metrics_path: str = r'C:\Users\Acer\OneDrive\Desktop\HogIntel-Price&Weight-Estimation\models\improved_price_model\model_metrics.json'):
        
        self.model = None
        self.scaler = None
        self.selected_features = None
        self.model_metrics = None
        self.model_name = "ridge"
        self.fallback_available = True
        
        # Load main model
        try:
            self.model = joblib.load(model_path)
            logger.info(f"Ridge regression model loaded from {model_path}")
            self.model_loaded = True
        except Exception as e:
            logger.error(f"Failed to load price model: {e}")
            self.model_loaded = False
        
        # Load scaler
        try:
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Scaler loaded from {scaler_path}")
        except Exception as e:
            logger.warning(f"Failed to load scaler: {e}")
            self.scaler = None
        
        # Load selected features
        try:
            with open(features_path, 'r') as f:
                self.selected_features = json.load(f)
            logger.info(f"Selected features loaded: {self.selected_features}")
        except Exception as e:
            logger.warning(f"Failed to load selected features: {e}")
            self.selected_features = None
        
        # Load model metrics
        try:
            with open(metrics_path, 'r') as f:
                self.model_metrics = json.load(f)
            logger.info(f"Model metrics loaded (Test MAPE: {self.model_metrics.get('test_mape', 'N/A')}%)")
        except Exception as e:
            logger.warning(f"Failed to load model metrics: {e}")
            self.model_metrics = None
        
        # Historical price data for feature engineering (will be updated from database)
        self.historical_prices = []
        self.last_known_price = 180.0  # Default fallback
    
    def update_historical_prices(self, prices: List[Dict[str, Any]]):
        """
        Update historical prices for feature engineering
        
        Args:
            prices: List of dicts with 'date' and 'price' keys
        """
        try:
            self.historical_prices = sorted(prices, key=lambda x: x['date'])
            if self.historical_prices:
                self.last_known_price = self.historical_prices[-1]['price']
            logger.info(f"Historical prices updated: {len(self.historical_prices)} records")
        except Exception as e:
            logger.error(f"Failed to update historical prices: {e}")
    
    def engineer_features(self, 
                         base_weight: Optional[float] = None,
                         current_date: Optional[datetime] = None) -> Dict[str, float]:
        """
        Engineer features for prediction matching training script
        
        Args:
            base_weight: Weight in kg (optional, for future weight-based features)
            current_date: Date for temporal features
            
        Returns:
            Dictionary of engineered features
        """
        if current_date is None:
            current_date = datetime.now()
        
        features = {}
        
        # Temporal features
        features['year'] = current_date.year
        features['month_num'] = current_date.month
        features['quarter'] = (current_date.month - 1) // 3 + 1
        
        # Cyclical encoding
        features['month_sin'] = np.sin(2 * np.pi * current_date.month / 12)
        features['month_cos'] = np.cos(2 * np.pi * current_date.month / 12)
        
        # Price lag features (based on historical data)
        if len(self.historical_prices) >= 1:
            features['price_lag_1'] = self.historical_prices[-1]['price']
        else:
            features['price_lag_1'] = self.last_known_price
        
        if len(self.historical_prices) >= 3:
            features['price_lag_3'] = self.historical_prices[-3]['price']
        else:
            features['price_lag_3'] = features['price_lag_1']
        
        if len(self.historical_prices) >= 6:
            features['price_lag_6'] = self.historical_prices[-6]['price']
        else:
            features['price_lag_6'] = features['price_lag_1']
        
        if len(self.historical_prices) >= 12:
            features['price_lag_12'] = self.historical_prices[-12]['price']
        else:
            features['price_lag_12'] = features['price_lag_1']
        
        # Rolling statistics
        if len(self.historical_prices) >= 3:
            recent_3 = [p['price'] for p in self.historical_prices[-3:]]
            features['price_rolling_mean_3'] = np.mean(recent_3)
            features['price_rolling_std_3'] = np.std(recent_3) if len(recent_3) > 1 else 0
        else:
            features['price_rolling_mean_3'] = features['price_lag_1']
            features['price_rolling_std_3'] = 0
        
        if len(self.historical_prices) >= 6:
            recent_6 = [p['price'] for p in self.historical_prices[-6:]]
            features['price_rolling_mean_6'] = np.mean(recent_6)
        else:
            features['price_rolling_mean_6'] = features['price_rolling_mean_3']
        
        # Price changes
        if len(self.historical_prices) >= 2:
            price_current = self.historical_prices[-1]['price']
            price_prev_1 = self.historical_prices[-2]['price']
            features['price_pct_change_1'] = (price_current - price_prev_1) / price_prev_1
        else:
            features['price_pct_change_1'] = 0
        
        if len(self.historical_prices) >= 4:
            price_current = self.historical_prices[-1]['price']
            price_prev_3 = self.historical_prices[-4]['price']
            features['price_pct_change_3'] = (price_current - price_prev_3) / price_prev_3
        else:
            features['price_pct_change_3'] = 0
        
        if len(self.historical_prices) >= 13:
            price_current = self.historical_prices[-1]['price']
            price_prev_12 = self.historical_prices[-13]['price']
            features['price_pct_change_12'] = (price_current - price_prev_12) / price_prev_12
            features['price_yoy_change'] = price_current - price_prev_12
        else:
            features['price_pct_change_12'] = 0
            features['price_yoy_change'] = 0
        
        return features
    
    def predict(self, 
                weight_kg: Optional[float] = None, 
                market_data: Optional[Dict] = None, 
                use_fallback: bool = False,
                current_date: Optional[datetime] = None) -> PricePrediction:
        """
        Predict price per kg using Ridge regression model
        
        Args:
            weight_kg: Weight in kilograms (currently not used, for future enhancement)
            market_data: Optional dict with additional market information
            use_fallback: Force use of fallback model
            current_date: Date for prediction (defaults to now)
            
        Returns:
            PricePrediction object with comprehensive prediction details
        """
        if not self.model_loaded or use_fallback or self.model is None:
            return self._fallback_price(weight_kg, market_data)
        
        try:
            # Engineer features
            features = self.engineer_features(weight_kg, current_date)
            
            # Select only the features used by the model
            if self.selected_features:
                feature_values = [features.get(feat, 0) for feat in self.selected_features]
                feature_names = self.selected_features
            else:
                feature_values = list(features.values())
                feature_names = list(features.keys())
            
            # Create DataFrame
            df = pd.DataFrame([feature_values], columns=feature_names)
            
            # Scale features
            if self.scaler:
                df_scaled = self.scaler.transform(df)
            else:
                df_scaled = df.values
            
            # Make prediction
            price_per_kg = float(self.model.predict(df_scaled)[0])
            
            # Apply reasonable bounds (PHP 120-250 per kg)
            price_per_kg = np.clip(price_per_kg, 120.0, 250.0)
            
            # Calculate confidence based on model metrics
            if self.model_metrics:
                test_mape = self.model_metrics.get('test_mape', 15.0)
                confidence = max(0.5, min(0.95, 1.0 - (test_mape / 100)))
            else:
                confidence = 0.85
            
            # Prepare metadata
            metadata = {
                'features_count': len(feature_names),
                'scaler_used': self.scaler is not None,
                'prediction_date': current_date.isoformat() if current_date else datetime.now().isoformat(),
                'historical_data_points': len(self.historical_prices)
            }
            
            if self.model_metrics:
                metadata['model_test_mape'] = self.model_metrics.get('test_mape')
                metadata['model_test_mae'] = self.model_metrics.get('test_mae')
            
            return PricePrediction(
                price_per_kg=price_per_kg,
                confidence=confidence,
                model_used="ridge_regression",
                features_used=feature_names,
                market_conditions=self._get_market_conditions(weight_kg, market_data, features),
                prediction_metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Ridge regression prediction failed: {e}, falling back to simple model")
            return self._fallback_price(weight_kg, market_data)
    
    def _fallback_price(self, 
                       weight_kg: Optional[float] = None, 
                       market_data: Optional[Dict] = None) -> PricePrediction:
        """
        Simple fallback pricing based on last known price
        """
        base_price = self.last_known_price
        
        # Adjust based on weight if provided
        if weight_kg:
            if weight_kg > 100:
                base_price += 5.0
            elif weight_kg < 50:
                base_price -= 5.0
        
        # Adjust based on market conditions if provided
        if market_data and 'trend' in market_data:
            trend = market_data['trend'].lower()
            if trend == 'increasing':
                base_price *= 1.02
            elif trend == 'decreasing':
                base_price *= 0.98
        
        return PricePrediction(
            price_per_kg=float(base_price),
            confidence=0.70,
            model_used="fallback",
            features_used=None,
            market_conditions=self._get_market_conditions(weight_kg, market_data, {}),
            prediction_metadata={
                'reason': 'Ridge model unavailable or forced fallback',
                'base_price_source': 'last_known_price'
            }
        )
    
    def _get_market_conditions(self, 
                               weight_kg: Optional[float], 
                               market_data: Optional[Dict],
                               features: Dict) -> Dict[str, Any]:
        """Generate market conditions summary"""
        conditions = {
            "price_trend": self._analyze_price_trend(features),
            "market_stability": self._calculate_stability(features),
            "data_quality": "good" if len(self.historical_prices) >= 12 else "limited"
        }
        
        if weight_kg:
            conditions["weight_category"] = self._get_weight_category(weight_kg)
        
        if market_data:
            conditions["external_factors"] = market_data.get("factors", [])
            conditions["market_sentiment"] = market_data.get("trend", "neutral")
        
        return conditions
    
    def _analyze_price_trend(self, features: Dict) -> str:
        """Analyze price trend from features"""
        if not features or 'price_pct_change_3' not in features:
            return "stable"
        
        pct_change = features.get('price_pct_change_3', 0)
        
        if pct_change > 0.03:
            return "increasing"
        elif pct_change < -0.03:
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_stability(self, features: Dict) -> str:
        """Calculate market stability from price volatility"""
        if not features or 'price_rolling_std_3' not in features:
            return "unknown"
        
        std = features.get('price_rolling_std_3', 0)
        mean = features.get('price_rolling_mean_3', 180)
        
        if mean == 0:
            return "unknown"
        
        cv = std / mean  # Coefficient of variation
        
        if cv < 0.02:
            return "very_stable"
        elif cv < 0.05:
            return "stable"
        elif cv < 0.10:
            return "moderate"
        else:
            return "volatile"
    
    def _get_weight_category(self, weight_kg: float) -> str:
        """Categorize weight for market analysis"""
        if weight_kg < 50:
            return "lightweight"
        elif weight_kg < 80:
            return "standard"
        elif weight_kg < 100:
            return "heavy"
        else:
            return "premium"
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get information about available models"""
        models = [
            {
                "name": "ridge_regression",
                "status": "available" if self.model_loaded else "unavailable",
                "description": "Primary Ridge Regression model with L2 regularization",
                "features": len(self.selected_features) if self.selected_features else "unknown",
                "performance": self.model_metrics if self.model_metrics else None
            },
            {
                "name": "fallback",
                "status": "available",
                "description": "Simple rule-based fallback using last known price"
            }
        ]
        return models
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        return {
            "model_type": "ridge_regression",
            "model_loaded": self.model_loaded,
            "scaler_loaded": self.scaler is not None,
            "selected_features": self.selected_features,
            "metrics": self.model_metrics,
            "historical_data_points": len(self.historical_prices),
            "last_known_price": self.last_known_price,
            "fallback_available": self.fallback_available
        }