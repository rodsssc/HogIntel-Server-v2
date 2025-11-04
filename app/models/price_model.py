# app/models/price_model.py
import os
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
    model_name: str
    features_used: Optional[List[str]] = None
    market_conditions: Optional[Dict[str, Any]] = None
    prediction_metadata: Optional[Dict[str, Any]] = None

class PricePredictor:
    """
    Enhanced Price Predictor using Ridge Regression
    Supports feature engineering and multiple fallback strategies
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 scaler_path: Optional[str] = None,
                 features_path: Optional[str] = None,
                 metrics_path: Optional[str] = None,
                 db_connection=None):
        
        # Use environment variables or fallback to container defaults
        self.model_path = model_path or os.getenv(
            'PRICE_MODEL_PATH', 
            '/models/price_model/best_price_model.pkl'
        )
        self.scaler_path = scaler_path or os.getenv(
            'PRICE_SCALER_PATH',
            '/models/price_model/price_scaler.pkl'
        )
        self.features_path = features_path or os.getenv(
            'PRICE_FEATURES_PATH',
            '/models/price_model/selected_features.json'
        )
        self.metrics_path = metrics_path or os.getenv(
            'PRICE_METRICS_PATH',
            '/models/price_model/model_metrics.json'
        )
        
        logger.info("=" * 60)
        logger.info("Initializing PricePredictor")
        logger.info("=" * 60)
        logger.info(f"Model Path:    {self.model_path}")
        logger.info(f"Scaler Path:   {self.scaler_path}")
        logger.info(f"Features Path: {self.features_path}")
        logger.info(f"Metrics Path:  {self.metrics_path}")
        logger.info("-" * 60)
        
        # Initialize attributes
        self.model = None
        self.scaler = None
        self.selected_features = None
        self.model_metrics = None
        self.model_name = "ridge"
        self.fallback_available = True
        self.model_loaded = False
        
        # Load main model
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            self.model = joblib.load(self.model_path)
            logger.info(f"✅ Ridge regression model loaded from {self.model_path}")
            logger.info(f"   Model type: {type(self.model).__name__}")
            self.model_loaded = True
            
        except FileNotFoundError as e:
            logger.error(f"❌ Model file not found: {e}")
            self.model_loaded = False
        except Exception as e:
            logger.error(f"❌ Failed to load price model: {e}")
            logger.exception("Full traceback:")
            self.model_loaded = False
        
        # Load scaler
        try:
            if not os.path.exists(self.scaler_path):
                raise FileNotFoundError(f"Scaler file not found: {self.scaler_path}")
            
            self.scaler = joblib.load(self.scaler_path)
            logger.info(f"✅ Scaler loaded from {self.scaler_path}")
            logger.info(f"   Scaler type: {type(self.scaler).__name__}")
            
        except FileNotFoundError as e:
            logger.warning(f"⚠️  Scaler file not found: {e}")
            logger.warning("   Predictions will work without scaling (may be less accurate)")
            self.scaler = None
        except Exception as e:
            logger.warning(f"⚠️  Failed to load scaler: {e}")
            self.scaler = None
        
        # Load selected features
        try:
            if not os.path.exists(self.features_path):
                raise FileNotFoundError(f"Features file not found: {self.features_path}")
            
            with open(self.features_path, 'r') as f:
                self.selected_features = json.load(f)
            logger.info(f"✅ Selected features loaded: {len(self.selected_features)} features")
            logger.info(f"   Features: {', '.join(self.selected_features[:5])}...")
            
        except FileNotFoundError as e:
            logger.warning(f"⚠️  Features file not found: {e}")
            logger.warning("   Will use all available features")
            self.selected_features = None
        except Exception as e:
            logger.warning(f"⚠️  Failed to load selected features: {e}")
            self.selected_features = None
        
        # Load model metrics
        try:
            if not os.path.exists(self.metrics_path):
                raise FileNotFoundError(f"Metrics file not found: {self.metrics_path}")
            
            with open(self.metrics_path, 'r') as f:
                self.model_metrics = json.load(f)
            
            test_mape = self.model_metrics.get('test_mape', 'N/A')
            test_mae = self.model_metrics.get('test_mae', 'N/A')
            logger.info(f"✅ Model metrics loaded")
            logger.info(f"   Test MAPE: {test_mape}%")
            logger.info(f"   Test MAE:  {test_mae}")
            
        except FileNotFoundError as e:
            logger.warning(f"⚠️  Metrics file not found: {e}")
            self.model_metrics = None
        except Exception as e:
            logger.warning(f"⚠️  Failed to load model metrics: {e}")
            self.model_metrics = None
        
        logger.info("-" * 60)
        logger.info(f"PricePredictor Status:")
        logger.info(f"  Model Loaded:    {'✅ YES' if self.model_loaded else '❌ NO'}")
        logger.info(f"  Scaler Loaded:   {'✅ YES' if self.scaler else '⚠️  NO'}")
        logger.info(f"  Features Loaded: {'✅ YES' if self.selected_features else '⚠️  NO'}")
        logger.info(f"  Metrics Loaded:  {'✅ YES' if self.model_metrics else '⚠️  NO'}")
        logger.info("=" * 60)
        
        # Historical price data for feature engineering
        self.historical_prices = []
        self.last_known_price = 180.0  # Default fallback price (PHP)
        
        # 🔧 AUTO-UPDATE: Try to fetch historical prices on initialization
        try:
            self.auto_update_historical_prices(db_connection)
        except Exception as e:
            logger.warning(f"Could not auto-update historical prices: {e}")
            logger.warning("Using default fallback price. Call update_historical_prices() manually.")
    
    def auto_update_historical_prices(self, db_connection=None):
        """
        Automatically fetch and update historical prices from database
        
        Args:
            db_connection: Database connection object (optional)
        """
        try:
            if db_connection:
                # Fetch from database
                query = """
                    SELECT date, avg_price as price 
                    FROM historical_prices 
                    WHERE date >= DATE_SUB(NOW(), INTERVAL 12 MONTH)
                    ORDER BY date ASC
                """
                results = db_connection.execute(query)
                prices = [
                    {"date": row.date, "price": float(row.price)} 
                    for row in results
                ]
                self.update_historical_prices(prices)
                logger.info(f"✅ Auto-updated {len(prices)} historical price records from database")
            else:
                # Generate synthetic data for testing (REPLACE THIS IN PRODUCTION)
                logger.warning("⚠️  No database connection - generating synthetic historical data")
                logger.warning("⚠️  REPLACE THIS WITH ACTUAL DATABASE QUERIES IN PRODUCTION!")
                
                from datetime import timedelta
                
                # Generate 12 months of synthetic price data with realistic variation
                base_price = 170.0
                prices = []
                current_date = datetime.now()
                
                for i in range(12):
                    date = current_date - timedelta(days=30 * (12 - i))
                    # Add seasonal variation and trend
                    seasonal = 5 * np.sin(2 * np.pi * date.month / 12)
                    trend = (i - 6) * 0.5  # Slight upward trend
                    noise = np.random.uniform(-3, 3)
                    price = base_price + seasonal + trend + noise
                    prices.append({"date": date, "price": float(price)})
                
                self.update_historical_prices(prices)
                logger.info(f"⚠️  Generated {len(prices)} synthetic price records for testing")
                
        except Exception as e:
            logger.error(f"❌ Failed to auto-update historical prices: {e}")
            logger.warning("⚠️  Continuing with default last_known_price")
    
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
            logger.info(f"📊 Historical prices updated: {len(self.historical_prices)} records")
            logger.info(f"   Last known price: PHP {self.last_known_price:.2f}/kg")
            logger.info(f"   Date range: {self.historical_prices[0]['date'].strftime('%Y-%m-%d')} to {self.historical_prices[-1]['date'].strftime('%Y-%m-%d')}")
        except Exception as e:
            logger.error(f"❌ Failed to update historical prices: {e}")
    
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
        features['year'] = float(current_date.year)
        features['month_num'] = float(current_date.month)
        features['quarter'] = float((current_date.month - 1) // 3 + 1)
        
        # Cyclical encoding
        features['month_sin'] = np.sin(2 * np.pi * current_date.month / 12)
        features['month_cos'] = np.cos(2 * np.pi * current_date.month / 12)
        
        # Price lag features (based on historical data)
        if len(self.historical_prices) >= 1:
            features['price_lag_1'] = float(self.historical_prices[-1]['price'])
        else:
            features['price_lag_1'] = float(self.last_known_price)
        
        if len(self.historical_prices) >= 3:
            features['price_lag_3'] = float(self.historical_prices[-3]['price'])
        else:
            features['price_lag_3'] = features['price_lag_1']
        
        if len(self.historical_prices) >= 6:
            features['price_lag_6'] = float(self.historical_prices[-6]['price'])
        else:
            features['price_lag_6'] = features['price_lag_1']
        
        if len(self.historical_prices) >= 12:
            features['price_lag_12'] = float(self.historical_prices[-12]['price'])
        else:
            features['price_lag_12'] = features['price_lag_1']
        
        # Rolling statistics
        if len(self.historical_prices) >= 3:
            recent_3 = [p['price'] for p in self.historical_prices[-3:]]
            features['price_rolling_mean_3'] = float(np.mean(recent_3))
            features['price_rolling_std_3'] = float(np.std(recent_3)) if len(recent_3) > 1 else 0.0
        else:
            features['price_rolling_mean_3'] = features['price_lag_1']
            features['price_rolling_std_3'] = 0.0
        
        if len(self.historical_prices) >= 6:
            recent_6 = [p['price'] for p in self.historical_prices[-6:]]
            features['price_rolling_mean_6'] = float(np.mean(recent_6))
        else:
            features['price_rolling_mean_6'] = features['price_rolling_mean_3']
        
        # Price changes
        if len(self.historical_prices) >= 2:
            price_current = self.historical_prices[-1]['price']
            price_prev_1 = self.historical_prices[-2]['price']
            features['price_pct_change_1'] = float((price_current - price_prev_1) / price_prev_1) if price_prev_1 != 0 else 0.0
        else:
            features['price_pct_change_1'] = 0.0
        
        if len(self.historical_prices) >= 4:
            price_current = self.historical_prices[-1]['price']
            price_prev_3 = self.historical_prices[-4]['price']
            features['price_pct_change_3'] = float((price_current - price_prev_3) / price_prev_3) if price_prev_3 != 0 else 0.0
        else:
            features['price_pct_change_3'] = 0.0
        
        if len(self.historical_prices) >= 13:
            price_current = self.historical_prices[-1]['price']
            price_prev_12 = self.historical_prices[-13]['price']
            features['price_pct_change_12'] = float((price_current - price_prev_12) / price_prev_12) if price_prev_12 != 0 else 0.0
            features['price_yoy_change'] = float(price_current - price_prev_12)
        else:
            features['price_pct_change_12'] = 0.0
            features['price_yoy_change'] = 0.0
        
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
            logger.warning("⚠️  Using fallback price prediction")
            return self._fallback_price(weight_kg, market_data)
        
        try:
            # Engineer features
            features = self.engineer_features(weight_kg, current_date)
            
            # 🔍 DEBUG: Log all feature values
            logger.info("=" * 80)
            logger.info("🔍 FEATURE VALUES FOR PREDICTION")
            logger.info("=" * 80)
            logger.info(f"   Prediction Date: {(current_date or datetime.now()).strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"   Historical Data Points: {len(self.historical_prices)}")
            logger.info("-" * 80)
            
            for feat_name, feat_value in features.items():
                logger.info(f"   {feat_name:30s}: {feat_value:>12.4f}")
            
            logger.info("=" * 80)
            
            # Select only the features used by the model
            if self.selected_features:
                feature_values = [features.get(feat, 0.0) for feat in self.selected_features]
                feature_names = self.selected_features
                
                # 🔍 DEBUG: Log selected features being sent to model
                logger.info("🎯 SELECTED FEATURES FOR MODEL:")
                logger.info("-" * 80)
                for fname, fval in zip(feature_names, feature_values):
                    logger.info(f"   {fname:30s}: {fval:>12.4f}")
                logger.info("=" * 80)
            else:
                feature_values = list(features.values())
                feature_names = list(features.keys())
                logger.warning("⚠️  Using all features (selected_features not loaded)")
            
            # Create DataFrame
            df = pd.DataFrame([feature_values], columns=feature_names)
            logger.debug(f"📊 DataFrame shape: {df.shape}")
            
            # Scale features
            if self.scaler:
                df_scaled = self.scaler.transform(df.values)
                logger.debug("✅ Features scaled")
            else:
                df_scaled = df.values
                logger.warning("⚠️  No scaling applied (scaler not loaded)")
            
            # Make prediction
            prediction = self.model.predict(df_scaled)
            
            # Handle different prediction formats
            if isinstance(prediction, np.ndarray):
                price_per_kg = float(prediction[0])
            else:
                price_per_kg = float(prediction)
            
            # Apply reasonable bounds (PHP 120-250 per kg)
            original_price = price_per_kg
            price_per_kg = float(np.clip(price_per_kg, 120.0, 250.0))
            
            if abs(original_price - price_per_kg) > 0.01:
                logger.warning(f"⚠️  Price clipped from ₱{original_price:.2f} to ₱{price_per_kg:.2f}")
            
            # Calculate confidence based on model metrics
            if self.model_metrics:
                test_mape = self.model_metrics.get('test_mape', 15.0)
                confidence = float(max(0.5, min(0.95, 1.0 - (test_mape / 100))))
            else:
                confidence = 0.85
            
            logger.info("=" * 80)
            logger.info(f"💰 PREDICTION RESULT: ₱{price_per_kg:.2f}/kg")
            logger.info(f"📊 Confidence: {confidence:.2%}")
            logger.info(f"🤖 Model: {type(self.model).__name__}")
            logger.info("=" * 80)
            
            # Prepare metadata
            metadata = {
                'features_count': len(feature_names),
                'scaler_used': self.scaler is not None,
                'prediction_date': current_date.isoformat() if current_date else datetime.now().isoformat(),
                'historical_data_points': len(self.historical_prices),
                'model_type': type(self.model).__name__,
                'original_prediction': original_price,
                'clipped': abs(original_price - price_per_kg) > 0.01
            }
            
            if self.model_metrics:
                metadata['model_test_mape'] = self.model_metrics.get('test_mape')
                metadata['model_test_mae'] = self.model_metrics.get('test_mae')
            
            return PricePrediction(
                price_per_kg=price_per_kg,
                confidence=confidence,
                model_name="ridge_regression",
                features_used=feature_names,
                market_conditions=self._get_market_conditions(weight_kg, market_data, features),
                prediction_metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"❌ Ridge regression prediction failed: {e}")
            logger.exception("Full traceback:")
            logger.warning("⚠️  Falling back to simple price model")
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
        
        # Apply bounds
        base_price = float(np.clip(base_price, 120.0, 250.0))
        
        logger.info(f"💰 Fallback price: PHP {base_price:.2f}/kg")
        
        return PricePrediction(
            price_per_kg=base_price,
            confidence=0.70,
            model_name="fallback",
            features_used=None,
            market_conditions=self._get_market_conditions(weight_kg, market_data, {}),
            prediction_metadata={
                'reason': 'Ridge model unavailable or forced fallback',
                'base_price_source': 'last_known_price',
                'last_known_price': self.last_known_price
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
            "features_loaded": self.selected_features is not None,
            "metrics_loaded": self.model_metrics is not None,
            "selected_features": self.selected_features,
            "feature_count": len(self.selected_features) if self.selected_features else 0,
            "metrics": self.model_metrics,
            "historical_data_points": len(self.historical_prices),
            "last_known_price": self.last_known_price,
            "fallback_available": self.fallback_available,
            "paths": {
                "model": self.model_path,
                "scaler": self.scaler_path,
                "features": self.features_path,
                "metrics": self.metrics_path
            }
        }