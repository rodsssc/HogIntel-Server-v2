# app/models/price_model.py
import os
import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path
from logger import setup_logger
from typing import Dict, Optional, Any, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io
import base64

# Suppress warnings
warnings.filterwarnings('ignore')

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

@dataclass
class ForecastResult:
    """Structure for forecast results"""
    dates: List[str]
    predictions: List[float]
    confidence_intervals: List[Dict[str, float]]
    model_metadata: Dict[str, Any]

class PricePredictor:
    """
    Enhanced Price Predictor using SARIMA (Seasonal ARIMA)
    Supports time series forecasting with seasonality and trends
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 metadata_path: Optional[str] = None,
                 db_connection=None):
        
        # Use environment variables or fallback to container defaults
        self.model_path = model_path or os.getenv(
            'PRICE_MODEL_PATH', 
            'C:/Users/Acer/OneDrive/Desktop/HogIntel-Price&Weight-Estimation/models/enhanced_sarima_price_model/enhanced_sarima_model.pkl'
        )
        self.metadata_path = metadata_path or os.getenv(
            'PRICE_METADATA_PATH',
            'C:/Users/Acer/OneDrive/Desktop/HogIntel-Price&Weight-Estimation/models/enhanced_sarima_price_model/model_metadata.json'
        )
        
        logger.info("=" * 60)
        logger.info("Initializing SARIMA PricePredictor")
        logger.info("=" * 60)
        logger.info(f"Model Path:    {self.model_path}")
        logger.info(f"Metadata Path: {self.metadata_path}")
        logger.info("-" * 60)
        
        # Initialize attributes
        self.model = None
        self.model_metadata = None
        self.model_name = "sarima"
        self.fallback_available = True
        self.model_loaded = False
        self.sarima_order = None
        self.sarima_seasonal_order = None
        
        # Load SARIMA model
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"SARIMA model file not found: {self.model_path}")
            
            self.model = joblib.load(self.model_path)
            logger.info(f"✅ SARIMA model loaded from {self.model_path}")
            logger.info(f"   Model type: {type(self.model).__name__}")
            self.model_loaded = True
            
        except FileNotFoundError as e:
            logger.error(f"❌ SARIMA model file not found: {e}")
            self.model_loaded = False
        except Exception as e:
            logger.error(f"❌ Failed to load SARIMA model: {e}")
            logger.exception("Full traceback:")
            self.model_loaded = False
        
        # Load model metadata
        try:
            if not os.path.exists(self.metadata_path):
                raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
            
            with open(self.metadata_path, 'r') as f:
                self.model_metadata = json.load(f)
            
            # Extract SARIMA parameters
            self.sarima_order = self.model_metadata.get('order', (1, 1, 1))
            self.sarima_seasonal_order = self.model_metadata.get('seasonal_order', (1, 1, 1, 12))
            
            logger.info(f"✅ Model metadata loaded")
            logger.info(f"   SARIMA Order: {self.sarima_order}")
            logger.info(f"   Seasonal Order: {self.sarima_seasonal_order}")
            logger.info(f"   AIC: {self.model_metadata.get('aic', 'N/A')}")
            logger.info(f"   BIC: {self.model_metadata.get('bic', 'N/A')}")
            
        except FileNotFoundError as e:
            logger.warning(f"⚠️  Metadata file not found: {e}")
            self.model_metadata = None
        except Exception as e:
            logger.warning(f"⚠️  Failed to load model metadata: {e}")
            self.model_metadata = None
        
        logger.info("-" * 60)
        logger.info(f"SARIMA PricePredictor Status:")
        logger.info(f"  Model Loaded:    {'✅ YES' if self.model_loaded else '❌ NO'}")
        logger.info(f"  Metadata Loaded: {'✅ YES' if self.model_metadata else '⚠️  NO'}")
        logger.info(f"  Fallback Available: {'✅ YES' if self.fallback_available else '❌ NO'}")
        logger.info("=" * 60)
        
        # Historical price data for time series analysis
        self.historical_prices = []
        self.last_known_price = 180.0  # Default fallback price (PHP)
        self.price_series = None
        
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
                    WHERE date >= DATE_SUB(NOW(), INTERVAL 24 MONTH)
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
                
                # Generate 24 months of synthetic price data with realistic variation
                base_price = 170.0
                prices = []
                current_date = datetime.now()
                
                for i in range(24):
                    date = current_date - timedelta(days=30 * (24 - i))
                    # Add seasonal variation and trend
                    seasonal = 8 * np.sin(2 * np.pi * date.month / 12)
                    trend = (i - 12) * 0.3  # Slight upward trend
                    noise = np.random.uniform(-5, 5)
                    price = base_price + seasonal + trend + noise
                    prices.append({"date": date, "price": float(price)})
                
                self.update_historical_prices(prices)
                logger.info(f"⚠️  Generated {len(prices)} synthetic price records for SARIMA testing")
                
        except Exception as e:
            logger.error(f"❌ Failed to auto-update historical prices: {e}")
            logger.warning("⚠️  Continuing with default last_known_price")
    
    def update_historical_prices(self, prices: List[Dict[str, Any]]):
        """
        Update historical prices for SARIMA time series analysis
        
        Args:
            prices: List of dicts with 'date' and 'price' keys
        """
        try:
            self.historical_prices = sorted(prices, key=lambda x: x['date'])
            if self.historical_prices:
                self.last_known_price = self.historical_prices[-1]['price']
                
                # Create pandas Series for SARIMA
                dates = [p['date'] for p in self.historical_prices]
                price_values = [p['price'] for p in self.historical_prices]
                self.price_series = pd.Series(price_values, index=dates)
                
            logger.info(f"📊 Historical prices updated for SARIMA: {len(self.historical_prices)} records")
            logger.info(f"   Last known price: PHP {self.last_known_price:.2f}/kg")
            if self.historical_prices:
                logger.info(f"   Date range: {self.historical_prices[0]['date'].strftime('%Y-%m-%d')} to {self.historical_prices[-1]['date'].strftime('%Y-%m-%d')}")
                
            # Check data sufficiency for SARIMA
            if len(self.historical_prices) < 12:
                logger.warning(f"⚠️  Limited historical data: {len(self.historical_prices)} records. SARIMA works best with 24+ months.")
            elif len(self.historical_prices) < 24:
                logger.info(f"📈 Historical data: {len(self.historical_prices)} records (moderate)")
            else:
                logger.info(f"📈 Historical data: {len(self.historical_prices)} records (sufficient)")
                
        except Exception as e:
            logger.error(f"❌ Failed to update historical prices: {e}")
    
    def prepare_time_series_data(self, current_date: Optional[datetime] = None) -> Optional[pd.Series]:
        """
        Prepare time series data for SARIMA prediction
        
        Args:
            current_date: Date for prediction
            
        Returns:
            Prepared time series data
        """
        if self.price_series is None or len(self.price_series) == 0:
            return None
        
        try:
            # Ensure the series is properly indexed and sorted
            series = self.price_series.sort_index()
            
            # If we have a current date, we might need to extend the series
            if current_date and current_date > series.index[-1]:
                # For future dates, we'll use the existing series and let SARIMA forecast
                pass
            
            return series
            
        except Exception as e:
            logger.error(f"❌ Failed to prepare time series data: {e}")
            return None
    
    def predict(self, 
                weight_kg: Optional[float] = None, 
                market_data: Optional[Dict] = None, 
                use_fallback: bool = False,
                current_date: Optional[datetime] = None) -> PricePrediction:
        """
        Predict price per kg using SARIMA time series model
        
        Args:
            weight_kg: Weight in kilograms (for market context)
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
            if current_date is None:
                current_date = datetime.now()
            
            # Prepare time series data
            time_series = self.prepare_time_series_data(current_date)
            
            if time_series is None or len(time_series) < 6:
                logger.warning("⚠️  Insufficient historical data for SARIMA, using fallback")
                return self._fallback_price(weight_kg, market_data)
            
            logger.info("=" * 80)
            logger.info("🔮 SARIMA TIME SERIES PREDICTION")
            logger.info("=" * 80)
            logger.info(f"   Prediction Date: {current_date.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"   Historical Data Points: {len(time_series)}")
            logger.info(f"   Date Range: {time_series.index[0].strftime('%Y-%m-%d')} to {time_series.index[-1].strftime('%Y-%m-%d')}")
            logger.info("-" * 80)
            
            # Calculate days ahead for forecast
            last_date = time_series.index[-1]
            days_ahead = (current_date - last_date).days
            
            if days_ahead <= 0:
                # Use the last known price if predicting for a date in the historical range
                price_per_kg = float(time_series.iloc[-1])
                forecast_horizon = 0
                logger.info("📅 Predicting for historical date, using last known price")
            else:
                # Generate forecast for future date
                forecast_horizon = max(1, min(days_ahead, 365))  # Limit to 1 year
                logger.info(f"🔮 Generating {forecast_horizon}-day ahead forecast")
                
                try:
                    # Use the loaded SARIMA model to forecast
                    forecast = self.model.get_forecast(steps=forecast_horizon)
                    forecast_values = forecast.predicted_mean
                    price_per_kg = float(forecast_values.iloc[-1])
                    
                    logger.info(f"📈 SARIMA forecast successful: ₱{price_per_kg:.2f}/kg")
                    
                except Exception as forecast_error:
                    logger.error(f"❌ SARIMA forecast failed: {forecast_error}")
                    logger.warning("⚠️  Falling back to trend-based prediction")
                    # Fallback: use simple trend projection
                    price_per_kg = self._trend_based_prediction(time_series, forecast_horizon)
                    forecast_horizon = -1  # Indicates fallback was used
            
            # Apply reasonable bounds (PHP 120-250 per kg)
            original_price = price_per_kg
            price_per_kg = float(np.clip(price_per_kg, 120.0, 250.0))
            
            if abs(original_price - price_per_kg) > 0.01:
                logger.warning(f"⚠️  Price clipped from ₱{original_price:.2f} to ₱{price_per_kg:.2f}")
            
            # Calculate confidence based on data quality and forecast horizon
            confidence = self._calculate_sarima_confidence(time_series, forecast_horizon)
            
            logger.info("=" * 80)
            logger.info(f"💰 SARIMA PREDICTION RESULT: ₱{price_per_kg:.2f}/kg")
            logger.info(f"📊 Confidence: {confidence:.2%}")
            logger.info(f"📅 Forecast Horizon: {forecast_horizon} days")
            logger.info(f"🤖 Model: SARIMA{self.sarima_order}{self.sarima_seasonal_order}")
            logger.info("=" * 80)
            
            # Prepare metadata
            metadata = {
                'forecast_horizon': forecast_horizon,
                'prediction_date': current_date.isoformat(),
                'historical_data_points': len(time_series),
                'model_type': 'SARIMA',
                'sarima_order': self.sarima_order,
                'seasonal_order': self.sarima_seasonal_order,
                'original_prediction': original_price,
                'clipped': abs(original_price - price_per_kg) > 0.01,
                'data_sufficiency': 'sufficient' if len(time_series) >= 24 else 'limited'
            }
            
            if self.model_metadata:
                metadata.update({
                    'model_aic': self.model_metadata.get('aic'),
                    'model_bic': self.model_metadata.get('bic'),
                    'model_training_date': self.model_metadata.get('training_date')
                })
            
            return PricePrediction(
                price_per_kg=price_per_kg,
                confidence=confidence,
                model_name="sarima",
                features_used=["time_series", "seasonality", "trend"],
                market_conditions=self._get_market_conditions(weight_kg, market_data, time_series),
                prediction_metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"❌ SARIMA prediction failed: {e}")
            logger.exception("Full traceback:")
            logger.warning("⚠️  Falling back to simple price model")
            return self._fallback_price(weight_kg, market_data)
    
    def _trend_based_prediction(self, time_series: pd.Series, days_ahead: int) -> float:
        """
        Simple trend-based prediction as fallback when SARIMA fails
        
        Args:
            time_series: Historical price data
            days_ahead: Number of days to forecast ahead
            
        Returns:
            Predicted price
        """
        try:
            # Use simple linear regression on recent data
            recent_data = time_series.tail(30)  # Last 30 days
            if len(recent_data) < 10:
                return float(time_series.iloc[-1])
            
            # Calculate trend
            x = np.arange(len(recent_data))
            y = recent_data.values
            slope, intercept = np.polyfit(x, y, 1)
            
            # Project forward
            future_x = len(recent_data) + days_ahead - 1
            predicted_price = slope * future_x + intercept
            
            logger.info(f"📊 Trend-based fallback: slope={slope:.4f}, predicted=₱{predicted_price:.2f}")
            
            return float(predicted_price)
            
        except Exception as e:
            logger.warning(f"⚠️  Trend-based prediction failed: {e}")
            return float(time_series.iloc[-1])
    
    def _calculate_sarima_confidence(self, time_series: pd.Series, forecast_horizon: int) -> float:
        """
        Calculate prediction confidence based on data quality and forecast horizon
        
        Args:
            time_series: Historical price data
            forecast_horizon: Number of days ahead for forecast
            
        Returns:
            Confidence score between 0.5 and 0.95
        """
        base_confidence = 0.85
        
        # Adjust for data quantity
        data_points = len(time_series)
        if data_points >= 24:
            data_factor = 1.0
        elif data_points >= 12:
            data_factor = 0.9
        elif data_points >= 6:
            data_factor = 0.8
        else:
            data_factor = 0.7
        
        # Adjust for forecast horizon
        if forecast_horizon <= 0:
            horizon_factor = 1.0  # Historical data
        elif forecast_horizon <= 7:
            horizon_factor = 0.95
        elif forecast_horizon <= 30:
            horizon_factor = 0.85
        elif forecast_horizon <= 90:
            horizon_factor = 0.75
        else:
            horizon_factor = 0.65
        
        # Calculate final confidence
        confidence = base_confidence * data_factor * horizon_factor
        return float(max(0.5, min(0.95, confidence)))
    
    def generate_forecast(self, periods: int = 30) -> Dict[str, Any]:
        """
        Generate future price forecasts using SARIMA model
        
        Args:
            periods: Number of periods to forecast
            
        Returns:
            Forecast results with confidence intervals
        """
        if not self.model_loaded or self.model is None:
            raise ValueError("SARIMA model not loaded")
        
        if self.price_series is None or len(self.price_series) < 6:
            raise ValueError("Insufficient historical data for forecasting")
        
        try:
            logger.info(f"🔮 Generating {periods}-period SARIMA forecast")
            
            # Generate forecast
            forecast = self.model.get_forecast(steps=periods)
            forecast_mean = forecast.predicted_mean
            confidence_int = forecast.conf_int()
            
            # Generate future dates
            last_date = self.price_series.index[-1]
            if isinstance(last_date, str):
                last_date = datetime.fromisoformat(last_date)
            
            future_dates = [last_date + timedelta(days=i+1) for i in range(periods)]
            
            # Prepare results
            forecasts = []
            confidence_intervals = []
            
            for i, date in enumerate(future_dates):
                pred = float(forecast_mean.iloc[i])
                ci_lower = float(confidence_int.iloc[i, 0])
                ci_upper = float(confidence_int.iloc[i, 1])
                
                forecasts.append({
                    'date': date.isoformat(),
                    'predicted_price': pred,
                    'confidence_lower': ci_lower,
                    'confidence_upper': ci_upper
                })
                
                confidence_intervals.append({
                    'date': date.isoformat(),
                    'lower': ci_lower,
                    'upper': ci_upper
                })
            
            # Calculate overall statistics
            avg_prediction = float(np.mean(forecast_mean))
            trend = "increasing" if forecast_mean.iloc[-1] > forecast_mean.iloc[0] else "decreasing"
            
            result = {
                'forecast_periods': periods,
                'average_predicted_price': avg_prediction,
                'trend': trend,
                'forecasts': forecasts,
                'confidence_intervals': confidence_intervals,
                'model_metadata': {
                    'sarima_order': self.sarima_order,
                    'seasonal_order': self.sarima_seasonal_order,
                    'historical_data_points': len(self.price_series),
                    'forecast_generated_at': datetime.now().isoformat()
                }
            }
            
            logger.info(f"✅ SARIMA forecast generated: {periods} periods, avg price: ₱{avg_prediction:.2f}, trend: {trend}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ SARIMA forecast generation failed: {e}")
            raise
    
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
            market_conditions=self._get_market_conditions(weight_kg, market_data, None),
            prediction_metadata={
                'reason': 'SARIMA model unavailable or forced fallback',
                'base_price_source': 'last_known_price',
                'last_known_price': self.last_known_price
            }
        )
    
    def _get_market_conditions(self, 
                               weight_kg: Optional[float], 
                               market_data: Optional[Dict],
                               time_series: Optional[pd.Series]) -> Dict[str, Any]:
        """Generate market conditions summary"""
        conditions = {
            "model_type": "SARIMA",
            "data_quality": "good" if time_series is not None and len(time_series) >= 24 else "limited"
        }
        
        if time_series is not None and len(time_series) >= 6:
            conditions["price_trend"] = self._analyze_price_trend(time_series)
            conditions["market_stability"] = self._calculate_volatility(time_series)
            conditions["seasonality_strength"] = self._assess_seasonality(time_series)
        else:
            conditions["price_trend"] = "unknown"
            conditions["market_stability"] = "unknown"
            conditions["seasonality_strength"] = "unknown"
        
        if weight_kg:
            conditions["weight_category"] = self._get_weight_category(weight_kg)
        
        if market_data:
            conditions["external_factors"] = market_data.get("factors", [])
            conditions["market_sentiment"] = market_data.get("trend", "neutral")
        
        return conditions
    
    def _analyze_price_trend(self, time_series: pd.Series) -> str:
        """Analyze price trend from time series data"""
        if len(time_series) < 6:
            return "unknown"
        
        try:
            # Use linear regression to determine trend
            x = np.arange(len(time_series))
            y = time_series.values
            slope, _ = np.polyfit(x, y, 1)
            
            # Calculate percentage change
            pct_change = (time_series.iloc[-1] - time_series.iloc[0]) / time_series.iloc[0]
            
            if slope > 0.05 or pct_change > 0.03:
                return "increasing"
            elif slope < -0.05 or pct_change < -0.03:
                return "decreasing"
            else:
                return "stable"
                
        except Exception:
            return "unknown"
    
    def _calculate_volatility(self, time_series: pd.Series) -> str:
        """Calculate market stability from price volatility"""
        if len(time_series) < 6:
            return "unknown"
        
        try:
            returns = time_series.pct_change().dropna()
            volatility = returns.std()
            
            if volatility < 0.02:
                return "very_stable"
            elif volatility < 0.05:
                return "stable"
            elif volatility < 0.10:
                return "moderate"
            else:
                return "volatile"
                
        except Exception:
            return "unknown"
    
    def _assess_seasonality(self, time_series: pd.Series) -> str:
        """Assess strength of seasonality in price data"""
        if len(time_series) < 12:
            return "insufficient_data"
        
        try:
            # Simple seasonality assessment using monthly averages
            monthly_avg = time_series.groupby(time_series.index.month).mean()
            seasonal_variation = monthly_avg.std() / monthly_avg.mean()
            
            if seasonal_variation > 0.1:
                return "strong"
            elif seasonal_variation > 0.05:
                return "moderate"
            else:
                return "weak"
                
        except Exception:
            return "unknown"
    
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
                "name": "sarima",
                "status": "available" if self.model_loaded else "unavailable",
                "description": "Seasonal ARIMA time series model for price forecasting",
                "order": self.sarima_order,
                "seasonal_order": self.sarima_seasonal_order,
                "performance": {
                    "aic": self.model_metadata.get('aic') if self.model_metadata else None,
                    "bic": self.model_metadata.get('bic') if self.model_metadata else None
                } if self.model_metadata else None
            },
            {
                "name": "fallback",
                "status": "available",
                "description": "Simple rule-based fallback using last known price"
            }
        ]
        return models
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive SARIMA model information"""
        return {
            "model_type": "sarima",
            "model_loaded": self.model_loaded,
            "sarima_parameters": {
                "order": self.sarima_order,
                "seasonal_order": self.sarima_seasonal_order
            },
            "model_metadata": self.model_metadata,
            "historical_data_points": len(self.historical_prices),
            "last_known_price": self.last_known_price,
            "fallback_available": self.fallback_available,
            "data_sufficiency": "sufficient" if len(self.historical_prices) >= 24 else "limited",
            "minimum_recommended_data": 24,
            "paths": {
                "model": self.model_path,
                "metadata": self.metadata_path
            }
        }