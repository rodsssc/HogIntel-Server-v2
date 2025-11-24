# schema.py
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum
import re

class ScanStatus(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    ADJUSTED = "adjusted"

class ModelType(str, Enum):
    SARIMA = "sarima"
    RIDGE = "ridge"
    FALLBACK = "fallback"
    PROPHET = "prophet"

class MarketTrend(str, Enum):
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    VOLATILE = "volatile"

class DataQuality(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    LIMITED = "limited"
    INSUFFICIENT = "insufficient"

class WeightRequest(BaseModel):
    image_data: str = Field(..., description="Base64 encoded image data", min_length=100, example="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ...")
    calibration_data: Optional[Dict[str, Any]] = Field(None, description="Camera calibration parameters")
    user_id: Optional[str] = Field(None, description="Optional user identifier", max_length=50)
    
    @validator('image_data')
    def validate_image_data(cls, v):
        if not v.startswith(('data:image/jpeg;base64,', 'data:image/png;base64,', 'data:image/jpg;base64,')):
            raise ValueError('Image data must be base64 encoded JPEG or PNG')
        return v

class WeightResponse(BaseModel):
    weight_kg: float = Field(..., description="Predicted weight in kilograms", ge=0, le=500, example=85.5)
    confidence: float = Field(..., description="Confidence score (0-1)", ge=0, le=1, example=0.92)
    detection_bbox: List[float] = Field(..., description="Bounding box [x1, y1, x2, y2]", min_items=4, max_items=4, example=[0.1, 0.2, 0.8, 0.9])
    roi_cropped: bool = Field(..., description="Whether ROI was successfully cropped", example=True)
    processing_time: float = Field(..., description="Processing time in seconds", ge=0, example=2.34)
    scan_id: str = Field(..., description="Unique scan identifier", example="scan_123456789")
    model_used: str = Field(..., description="Model used for prediction", example="yolo_cnn_ensemble")
    quality_metrics: Optional[Dict[str, Any]] = Field(None, description="Image quality assessment metrics")

class ConfirmRequest(BaseModel):
    scan_id: str = Field(..., description="Scan identifier from weight prediction", example="scan_123456789")
    confirmed_weight: Optional[float] = Field(None, description="User confirmed/adjusted weight", ge=0, le=500, example=87.0)
    status: ScanStatus = Field(..., description="User action on weight prediction", example=ScanStatus.ADJUSTED)
    notes: Optional[str] = Field(None, description="Additional notes from user", max_length=500)

class ConfirmResponse(BaseModel):
    success: bool = Field(..., description="Confirmation recorded successfully", example=True)
    confirmed_weight: float = Field(..., description="Final confirmed weight", ge=0, le=500, example=87.0)
    scan_id: str = Field(..., description="Scan identifier", example="scan_123456789")
    timestamp: datetime = Field(..., description="Confirmation timestamp")
    previous_weight: Optional[float] = Field(None, description="Original predicted weight", ge=0, le=500)
    adjustment: Optional[float] = Field(None, description="Weight adjustment amount")

class PriceRequest(BaseModel):
    scan_id: str = Field(..., description="Scan identifier from confirmed weight", example="scan_123456789")
    confirmed_weight: float = Field(..., description="Confirmed weight in kg", ge=0, le=500, example=87.0)
    market_data: Optional[Dict[str, Any]] = Field(None, description="Additional market parameters")
    use_fallback: bool = Field(False, description="Use fallback model if available")
    prediction_date: Optional[datetime] = Field(None, description="Specific date for price prediction")
    
    @validator('confirmed_weight')
    def validate_weight(cls, v):
        if v <= 0:
            raise ValueError('Weight must be greater than 0')
        if v > 500:
            raise ValueError('Weight exceeds reasonable maximum (500kg)')
        return v

class PriceResponse(BaseModel):
    price_per_kg: float = Field(..., description="Predicted price per kilogram", ge=0, example=185.50)
    total_value: float = Field(..., description="Total value (price_per_kg * weight)", ge=0, example=16138.50)
    confidence: float = Field(..., description="Price prediction confidence", ge=0, le=1, example=0.88)
    model_used: ModelType = Field(..., description="Which model was used", example=ModelType.SARIMA)
    market_conditions: Dict[str, Any] = Field(..., description="Market context and analysis")
    timestamp: datetime = Field(..., description="Prediction timestamp")
    forecast_horizon: Optional[int] = Field(None, description="Days ahead forecasted for SARIMA", ge=0, example=7)
    prediction_metadata: Optional[Dict[str, Any]] = Field(None, description="Additional prediction details")
    data_quality: DataQuality = Field(default=DataQuality.GOOD, description="Quality of input data for prediction")  # FIXED: Added default value

    class Config:
        # This makes the enum values serialize properly
        use_enum_values = True
        
class ForecastRequest(BaseModel):
    periods: int = Field(default=30, ge=1, le=365, description="Number of periods to forecast", example=30)
    include_confidence: bool = Field(default=True, description="Include confidence intervals")
    start_date: Optional[datetime] = Field(None, description="Start date for forecast")

class ForecastResponse(BaseModel):
    forecast_periods: int = Field(..., description="Number of periods forecasted", example=30)
    average_predicted_price: float = Field(..., description="Average predicted price across forecast", ge=0, example=182.75)
    trend: MarketTrend = Field(..., description="Overall price trend in forecast")
    forecasts: List[Dict[str, Any]] = Field(..., description="Detailed forecast data")
    confidence_intervals: Optional[List[Dict[str, Any]]] = Field(None, description="Confidence intervals for forecasts")
    model_metadata: Dict[str, Any] = Field(..., description="Model information and parameters")
    timestamp: datetime = Field(..., description="Forecast generation timestamp")
    data_sufficiency: DataQuality = Field(..., description="Quality of historical data used")

class ModelInfoResponse(BaseModel):
    model_type: str = Field(..., description="Type of model", example="sarima")
    model_loaded: bool = Field(..., description="Whether model is successfully loaded", example=True)
    sarima_parameters: Optional[Dict[str, Any]] = Field(None, description="SARIMA model parameters")
    historical_data_points: int = Field(..., description="Number of historical data points", ge=0, example=24)
    last_known_price: float = Field(..., description="Most recent price in historical data", ge=0, example=180.50)
    fallback_available: bool = Field(..., description="Whether fallback model is available", example=True)
    data_sufficiency: DataQuality = Field(..., description="Quality of available data")
    performance_metrics: Optional[Dict[str, Any]] = Field(None, description="Model performance metrics")

class HealthCheckResponse(BaseModel):
    status: str = Field(..., description="Service status", example="healthy")
    message: str = Field(..., description="Status message", example="All systems operational")
    timestamp: datetime = Field(..., description="Health check timestamp")
    details: Dict[str, Any] = Field(..., description="Detailed health information")
    version: str = Field(..., description="API version", example="1.2.0")

class BatchPriceRequest(BaseModel):
    requests: List[PriceRequest] = Field(..., description="List of price prediction requests", max_items=100)
    
    @validator('requests')
    def validate_batch_size(cls, v):
        if len(v) > 100:
            raise ValueError('Batch size cannot exceed 100 requests')
        return v

class BatchPriceResponse(BaseModel):
    successful: int = Field(..., description="Number of successful predictions", ge=0, example=95)
    failed: int = Field(..., description="Number of failed predictions", ge=0, example=5)
    results: List[Dict[str, Any]] = Field(..., description="Successful prediction results")
    errors: Optional[List[Dict[str, Any]]] = Field(None, description="Error details for failed predictions")
    timestamp: datetime = Field(..., description="Batch processing timestamp")
    processing_time: float = Field(..., description="Total processing time in seconds", ge=0)

class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message", example="Insufficient historical data")
    code: str = Field(..., description="Error code", example="INSUFFICIENT_DATA")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Unique request identifier")

# FIXED: Changed 'date' field name to avoid conflict with datetime.date type
class HistoricalPriceUpdate(BaseModel):
    price_date: datetime = Field(..., description="Price date", example="2024-01-15")  # Changed from 'date' to 'price_date'
    price: float = Field(..., description="Price per kg", ge=0, example=175.50)
    source: Optional[str] = Field(None, description="Data source", example="market_api")
    volume: Optional[float] = Field(None, description="Trading volume", ge=0)

class HistoricalUpdateResponse(BaseModel):
    success: bool = Field(..., description="Update successful", example=True)
    records_updated: int = Field(..., description="Number of records updated", ge=0, example=150)
    date_range: Dict[str, str] = Field(..., description="Date range of updated data")
    last_known_price: float = Field(..., description="Updated last known price", ge=0, example=182.25)
    data_sufficiency: DataQuality = Field(..., description="Updated data quality assessment")
    message: str = Field(..., description="Update status message")
    timestamp: datetime = Field(..., description="Update timestamp")

# ========================
# Analytics and Monitoring
# ========================

class PredictionAnalytics(BaseModel):
    scan_id: str = Field(..., description="Scan identifier")
    model_used: str = Field(..., description="Model used for prediction")
    prediction_time: float = Field(..., description="Time taken for prediction")
    confidence: float = Field(..., description="Prediction confidence")
    input_features: Optional[Dict[str, Any]] = Field(None, description="Input features used")
    timestamp: datetime = Field(..., description="Analytics timestamp")

class SystemMetrics(BaseModel):
    memory_usage_mb: float = Field(..., description="Memory usage in MB")
    cpu_percent: float = Field(..., description="CPU usage percentage")
    active_connections: int = Field(..., description="Active API connections")
    model_inference_time: float = Field(..., description="Average model inference time")
    timestamp: datetime = Field(..., description="Metrics timestamp")