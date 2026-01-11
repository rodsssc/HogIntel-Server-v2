# app/schemas.py
"""
Complete Pydantic schemas for HogIntel API
Includes weight estimation, price prediction, age validation, and confirmation schemas
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


# ============================================================
# ENUMS
# ============================================================

class PredictionMethod(str, Enum):
    """Method used for weight prediction"""
    IMAGE_ONLY_CNN = "image_only_cnn"
    MULTIMODAL_CNN = "multimodal_cnn"
    MULTIMODAL_CNN_FALLBACK = "multimodal_cnn_fallback"
    BLENDED_IMAGE_AGE = "blended_image_age"
    AGE_ONLY = "age_only"


class ConfirmationStatus(str, Enum):
    """Status for weight confirmation"""
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    ADJUSTED = "adjusted"


class BreedType(str, Enum):
    """Pig breed types"""
    COMMERCIAL = "commercial"
    DUROC = "duroc"
    YORKSHIRE = "yorkshire"
    LANDRACE = "landrace"
    BERKSHIRE = "berkshire"
    HAMPSHIRE = "hampshire"


class GenderType(str, Enum):
    """Pig gender types"""
    MALE = "male"
    FEMALE = "female"
    CASTRATED = "castrated"
    UNKNOWN = "unknown"


class AgreementLevel(str, Enum):
    """Agreement between image and age predictions"""
    EXCELLENT = "excellent"  # <5% deviation
    GOOD = "good"  # 5-10% deviation
    ACCEPTABLE = "acceptable"  # 10-15% deviation
    UNDERWEIGHT_CONCERN = "underweight_concern"  # >15% below expected
    OVERWEIGHT_CONCERN = "overweight_concern"  # >15% above expected


class DataQuality(str, Enum):
    """Data quality levels for price predictions"""
    EXCELLENT = "excellent"
    GOOD = "good"
    LIMITED = "limited"
    INSUFFICIENT = "insufficient"


# ============================================================
# WEIGHT ESTIMATION SCHEMAS
# ============================================================

class WeightRange(BaseModel):
    """Expected weight range"""
    minimum: float = Field(..., ge=0, description="Minimum expected weight")
    maximum: float = Field(..., ge=0, description="Maximum expected weight")
    average: float = Field(..., ge=0, description="Average expected weight")
    
    class Config:
        schema_extra = {
            "example": {
                "minimum": 75.0,
                "maximum": 95.0,
                "average": 85.0
            }
        }


class AgeValidation(BaseModel):
    """Age-based validation results"""
    age_based_estimate: WeightRange = Field(
        ...,
        description="Expected weight range based on age"
    )
    within_expected_range: bool = Field(
        ...,
        description="Whether predicted weight is within expected range"
    )
    deviation_percent: float = Field(
        ...,
        description="Percentage deviation from age-based average"
    )
    agreement_level: AgreementLevel = Field(
        ...,
        description="Classification of agreement between predictions"
    )
    blended_weight: float = Field(
        ...,
        description="Weighted average of image and age predictions"
    )
    recommendation: str = Field(
        ...,
        description="Recommendation based on comparison"
    )
    
    class Config:
        use_enum_values = True


class WeightRequest(BaseModel):
    """Basic weight prediction request"""
    image_data: str = Field(
        ..., 
        description="Base64 encoded image data",
        min_length=100
    )
    
    @validator('image_data')
    def validate_image_data(cls, v):
        if not v.startswith(('data:image/jpeg;base64,', 'data:image/png;base64,', 'data:image/jpg;base64,')):
            raise ValueError('Image data must be base64 encoded JPEG or PNG')
        return v


class WeightRequestEnhanced(BaseModel):
    """Enhanced weight prediction request with age support"""
    image_data: str = Field(
        ..., 
        description="Base64 encoded image data",
        min_length=100,
        example="data:image/jpeg;base64,/9j/4AAQ..."
    )
    age_months: Optional[float] = Field(
        None,
        ge=0.5,
        le=24,
        description="Age of hog in months (optional but recommended)",
        example=5.5
    )
    breed: Optional[BreedType] = Field(
        BreedType.COMMERCIAL,
        description="Breed type for age-based adjustments"
    )
    gender: Optional[GenderType] = Field(
        GenderType.UNKNOWN,
        description="Gender for growth rate adjustments"
    )
    calibration_data: Optional[Dict[str, Any]] = Field(
        None,
        description="Camera calibration parameters"
    )
    user_id: Optional[str] = Field(
        None,
        max_length=50,
        description="User identifier for tracking"
    )
    
    @validator('image_data')
    def validate_image_data(cls, v):
        if not v.startswith(('data:image/jpeg;base64,', 'data:image/png;base64,', 'data:image/jpg;base64,')):
            raise ValueError('Image data must be base64 encoded JPEG or PNG')
        return v
    
    @validator('age_months')
    def validate_age(cls, v):
        if v is not None:
            if v < 0.5:
                raise ValueError('Age must be at least 0.5 months (2 weeks)')
            if v > 24:
                raise ValueError('Age exceeds reasonable maximum (24 months)')
        return v

    class Config:
        use_enum_values = True
        schema_extra = {
            "example": {
                "image_data": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
                "age_months": 5.5,
                "breed": "duroc",
                "gender": "male",
                "user_id": "farm_001"
            }
        }


class WeightResponse(BaseModel):
    """Basic weight prediction response"""
    estimated_weight: float = Field(
        ...,
        ge=0,
        le=500,
        description="Predicted weight in kilograms"
    )
    confidence: float = Field(
        ...,
        ge=0,
        le=1,
        description="Prediction confidence score"
    )
    detection_id: str = Field(
        ...,
        description="Detection identifier"
    )
    unit: str = Field(
        default="kg",
        description="Weight unit"
    )


class WeightResponseEnhanced(BaseModel):
    """Enhanced weight prediction response with age validation"""
    weight_kg: float = Field(
        ...,
        ge=0,
        le=500,
        description="Predicted weight in kilograms",
        example=87.5
    )
    confidence: float = Field(
        ...,
        ge=0,
        le=1,
        description="Prediction confidence score",
        example=0.92
    )
    detection_bbox: List[float] = Field(
        ...,
        min_items=4,
        max_items=4,
        description="Bounding box coordinates [x1, y1, x2, y2]",
        example=[0.1, 0.2, 0.8, 0.9]
    )
    roi_cropped: bool = Field(
        ...,
        description="Whether ROI was successfully extracted",
        example=True
    )
    processing_time: float = Field(
        ...,
        ge=0,
        description="Total processing time in seconds",
        example=2.34
    )
    scan_id: str = Field(
        ...,
        description="Unique scan identifier",
        example="scan_123abc456def"
    )
    model_used: str = Field(
        ...,
        description="Model architecture used",
        example="multimodal_cnn_resnet50"
    )
    quality_metrics: Optional[Dict[str, Any]] = Field(
        None,
        description="Image quality assessment metrics"
    )
    
    # Age-related fields
    age_provided: bool = Field(
        ...,
        description="Whether age was provided in request",
        example=True
    )
    age_months: Optional[float] = Field(
        None,
        description="Age used in prediction (provided or fallback)",
        example=5.5
    )
    age_validation: Optional[AgeValidation] = Field(
        None,
        description="Age-based validation and comparison results"
    )
    prediction_method: PredictionMethod = Field(
        ...,
        description="Method used for weight prediction"
    )
    
    class Config:
        use_enum_values = True
        schema_extra = {
            "example": {
                "weight_kg": 87.5,
                "confidence": 0.92,
                "detection_bbox": [100.5, 150.2, 450.8, 550.9],
                "roi_cropped": True,
                "processing_time": 2.34,
                "scan_id": "scan_123abc456def",
                "model_used": "multimodal_cnn_resnet50",
                "quality_metrics": {
                    "detection_confidence": 0.95,
                    "image_resolution": "1920x1080",
                    "roi_size": "350x400",
                    "brightness": 127.5,
                    "sharpness": 142.3
                },
                "age_provided": True,
                "age_months": 5.5,
                "age_validation": {
                    "age_based_estimate": {
                        "minimum": 75.0,
                        "maximum": 95.0,
                        "average": 85.0
                    },
                    "within_expected_range": True,
                    "deviation_percent": 2.9,
                    "agreement_level": "excellent",
                    "blended_weight": 86.8,
                    "recommendation": "Predictions align well. Weight is within expected range for age."
                },
                "prediction_method": "multimodal_cnn"
            }
        }


# ============================================================
# CONFIRMATION SCHEMAS
# ============================================================

class ConfirmRequest(BaseModel):
    """Request model for weight confirmation"""
    scan_id: str = Field(
        ..., 
        description="Scan identifier from weight prediction",
        example="scan_123abc456def"
    )
    confirmed_weight: float = Field(
        ..., 
        ge=0, 
        le=500, 
        description="User confirmed/adjusted weight in kg",
        example=87.5
    )
    status: ConfirmationStatus = Field(
        ..., 
        description="Confirmation status: accepted, rejected, or adjusted",
        example="accepted"
    )
    notes: Optional[str] = Field(
        None, 
        max_length=500, 
        description="Additional notes from user",
        example="Age-enhanced prediction accepted"
    )
    age_months: Optional[float] = Field(
        None, 
        ge=0.5, 
        le=24, 
        description="Age of hog in months",
        example=5.5
    )
    breed: Optional[BreedType] = Field(
        BreedType.COMMERCIAL, 
        description="Breed type: commercial, duroc, yorkshire, etc."
    )
    gender: Optional[GenderType] = Field(
        GenderType.UNKNOWN, 
        description="Gender: male, female, castrated, unknown"
    )
    
    @validator('confirmed_weight')
    def validate_weight(cls, v):
        if v <= 0:
            raise ValueError('Weight must be greater than 0')
        if v > 500:
            raise ValueError('Weight exceeds reasonable maximum (500 kg)')
        return v
    
    class Config:
        use_enum_values = True
        schema_extra = {
            "example": {
                "scan_id": "scan_123abc456def",
                "confirmed_weight": 87.5,
                "status": "accepted",
                "notes": "Age-enhanced prediction accepted",
                "age_months": 5.5,
                "breed": "commercial",
                "gender": "unknown"
            }
        }


class ConfirmResponse(BaseModel):
    """Response model for weight confirmation"""
    success: bool = Field(
        ..., 
        description="Whether confirmation was successful",
        example=True
    )
    confirmed_weight: float = Field(
        ..., 
        description="Confirmed weight in kg",
        example=87.5
    )
    scan_id: str = Field(
        ..., 
        description="Scan identifier",
        example="scan_123abc456def"
    )
    timestamp: datetime = Field(
        ..., 
        description="Confirmation timestamp"
    )
    status: str = Field(
        ..., 
        description="Confirmation status",
        example="accepted"
    )
    previous_weight: Optional[float] = Field(
        None, 
        description="Previous predicted weight before adjustment",
        example=85.0
    )
    adjustment: Optional[float] = Field(
        None, 
        description="Weight adjustment amount (if adjusted)",
        example=2.5
    )
    age_validation: Optional[Dict[str, Any]] = Field(
        None, 
        description="Age-based validation results"
    )
    age_months: Optional[float] = Field(
        None,
        description="Age in months if provided",
        example=5.5
    )
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "confirmed_weight": 87.5,
                "scan_id": "scan_123abc456def",
                "timestamp": "2024-12-21T10:30:00",
                "status": "accepted",
                "previous_weight": None,
                "adjustment": None,
                "age_validation": {
                    "age_based_estimate": {
                        "minimum": 75.0,
                        "maximum": 95.0,
                        "average": 85.0
                    },
                    "within_expected_range": True,
                    "deviation_percent": 2.9,
                    "agreement_level": "excellent",
                    "blended_weight": 86.8,
                    "recommendation": "Weight is within expected range for age."
                },
                "age_months": 5.5
            }
        }


# ============================================================
# AGE ESTIMATION SCHEMAS
# ============================================================

class AgeEstimateRequest(BaseModel):
    """Request model for age-based weight estimation"""
    age_months: float = Field(
        ..., 
        ge=0.5, 
        le=24, 
        description="Age of hog in months",
        example=5.5
    )
    breed: Optional[BreedType] = Field(
        BreedType.COMMERCIAL, 
        description="Breed type"
    )
    gender: Optional[GenderType] = Field(
        GenderType.UNKNOWN, 
        description="Gender"
    )
    
    class Config:
        use_enum_values = True
        schema_extra = {
            "example": {
                "age_months": 5.5,
                "breed": "commercial",
                "gender": "male"
            }
        }


class AgeEstimateResponse(BaseModel):
    """Response model for age-based weight estimation"""
    estimated_weight_kg: Dict[str, float] = Field(
        ..., 
        description="Estimated weight range (min, max, avg)",
        example={"minimum": 75.0, "maximum": 95.0, "average": 85.0}
    )
    age_months: float = Field(
        ..., 
        description="Age in months",
        example=5.5
    )
    breed: str = Field(
        ..., 
        description="Breed type used",
        example="commercial"
    )
    gender: str = Field(
        ..., 
        description="Gender used",
        example="male"
    )
    confidence: float = Field(
        ..., 
        ge=0, 
        le=1, 
        description="Estimation confidence",
        example=0.85
    )
    breed_factor: float = Field(
        ..., 
        description="Breed adjustment factor applied",
        example=1.0
    )
    gender_factor: float = Field(
        ..., 
        description="Gender adjustment factor applied",
        example=1.05
    )
    growth_stage: str = Field(
        ..., 
        description="Growth stage classification",
        example="finisher"
    )
    market_ready: bool = Field(
        ..., 
        description="Whether hog is market ready",
        example=True
    )
    timestamp: str = Field(
        ..., 
        description="Response timestamp"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "estimated_weight_kg": {
                    "minimum": 75.0,
                    "maximum": 95.0,
                    "average": 85.0
                },
                "age_months": 5.5,
                "breed": "commercial",
                "gender": "male",
                "confidence": 0.85,
                "breed_factor": 1.0,
                "gender_factor": 1.05,
                "growth_stage": "finisher",
                "market_ready": True,
                "timestamp": "2024-12-21T10:30:00Z"
            }
        }


class WeightComparisonRequest(BaseModel):
    """Request for comparing image and age-based predictions"""
    age_months: float = Field(..., ge=0.5, le=24)
    image_predicted_weight: float = Field(..., ge=0, le=500)
    breed: Optional[BreedType] = Field(BreedType.COMMERCIAL)
    gender: Optional[GenderType] = Field(GenderType.UNKNOWN)
    
    class Config:
        use_enum_values = True


class WeightComparisonResponse(BaseModel):
    """Response for weight comparison"""
    age_based_estimate: Dict[str, float] = Field(
        ...,
        description="Age-based weight estimate"
    )
    image_predicted_weight: float = Field(
        ...,
        description="Image-based prediction"
    )
    blended_weight: float = Field(
        ...,
        description="Blended weight recommendation"
    )
    within_expected_range: bool = Field(
        ...,
        description="Whether prediction is within expected range"
    )
    deviation_kg: float = Field(
        ...,
        description="Deviation from expected in kg"
    )
    deviation_percent: float = Field(
        ...,
        description="Deviation percentage"
    )
    agreement_level: str = Field(
        ...,
        description="Agreement level classification"
    )
    recommendation: str = Field(
        ...,
        description="Recommendation based on comparison"
    )
    confidence_scores: Dict[str, float] = Field(
        ...,
        description="Confidence scores for different methods"
    )


class GrowthRecommendationRequest(BaseModel):
    """Request for growth recommendations"""
    age_months: float = Field(..., ge=0.5, le=24)
    current_weight: float = Field(..., ge=0, le=500)


class GrowthRecommendationResponse(BaseModel):
    """Response with growth recommendations"""
    current_status: Dict[str, Any] = Field(
        ...,
        description="Current growth status"
    )
    market_projection: Dict[str, Any] = Field(
        ...,
        description="Market readiness projection"
    )
    recommendations: List[str] = Field(
        ...,
        description="Growth and management recommendations"
    )


# ============================================================
# PRICE PREDICTION SCHEMAS
# ============================================================

class PriceRequest(BaseModel):
    """Request model for price prediction"""
    scan_id: str = Field(
        ..., 
        description="Scan identifier from weight prediction",
        example="scan_123abc456def"
    )
    confirmed_weight: float = Field(
        ..., 
        ge=0, 
        le=500, 
        description="Confirmed weight in kilograms",
        example=87.5
    )
    market_data: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional market condition data"
    )
    use_fallback: bool = Field(
        False,
        description="Whether to use fallback prediction if SARIMA fails"
    )
    prediction_date: Optional[datetime] = Field(
        None,
        description="Date for which to predict price (defaults to current date)"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "scan_id": "scan_123abc456def",
                "confirmed_weight": 87.5,
                "market_data": {
                    "region": "Davao",
                    "market_type": "local"
                },
                "use_fallback": False
            }
        }


class PriceResponse(BaseModel):
    """Response model for price prediction"""
    price_per_kg: float = Field(
        ...,
        ge=0,
        description="Predicted price per kilogram in PHP",
        example=185.50
    )
    total_value: float = Field(
        ...,
        ge=0,
        description="Total value (price_per_kg * weight) in PHP",
        example=16231.25
    )
    confidence: float = Field(
        ...,
        ge=0,
        le=1,
        description="Prediction confidence score",
        example=0.85
    )
    model_used: str = Field(
        ...,
        description="Model used for prediction",
        example="SARIMA(1,1,1)(1,1,1,12)"
    )
    market_conditions: Optional[Dict[str, Any]] = Field(
        None,
        description="Market condition analysis"
    )
    timestamp: datetime = Field(
        ...,
        description="Prediction timestamp"
    )
    features_used: Optional[List[str]] = Field(
        None,
        description="Features used in prediction"
    )
    prediction_metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional prediction metadata"
    )
    data_quality: DataQuality = Field(
        ...,
        description="Quality of historical data used"
    )
    
    class Config:
        use_enum_values = True
        schema_extra = {
            "example": {
                "price_per_kg": 185.50,
                "total_value": 16231.25,
                "confidence": 0.85,
                "model_used": "SARIMA(1,1,1)(1,1,1,12)",
                "market_conditions": {
                    "price_trend": "stable",
                    "market_stability": "moderate",
                    "seasonality_strength": "medium"
                },
                "timestamp": "2024-12-20T10:30:00",
                "features_used": ["temporal_patterns", "seasonality", "trends"],
                "prediction_metadata": {
                    "forecast_horizon": 0,
                    "sarima_order": "(1,1,1)",
                    "seasonal_order": "(1,1,1,12)",
                    "data_sufficiency": "good"
                },
                "data_quality": "good"
            }
        }


class HistoricalPriceUpdate(BaseModel):
    """Model for updating historical prices"""
    price_date: str = Field(
        ...,
        description="Date in ISO format (YYYY-MM-DD)",
        example="2024-01-15"
    )
    price: float = Field(
        ...,
        ge=0,
        description="Price per kg in PHP",
        example=180.50
    )
    source: Optional[str] = Field(
        None,
        description="Data source",
        example="PSA"
    )
    volume: Optional[float] = Field(
        None,
        description="Trading volume if available",
        example=1500.0
    )


# ============================================================
# ERROR RESPONSE SCHEMAS
# ============================================================

class ErrorResponse(BaseModel):
    """Standard error response model"""
    error: str = Field(
        ...,
        description="Error message",
        example="Invalid input data"
    )
    code: str = Field(
        ...,
        description="Error code",
        example="VALIDATION_ERROR"
    )
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional error details"
    )
    timestamp: Optional[datetime] = Field(
        None,
        description="Error timestamp"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "error": "Invalid input data",
                "code": "VALIDATION_ERROR",
                "details": {
                    "field": "confirmed_weight",
                    "message": "Weight must be greater than 0"
                },
                "timestamp": "2024-12-20T10:30:00"
            }
        }


# ============================================================
# BATCH PROCESSING SCHEMAS
# ============================================================

class BatchWeightRequest(BaseModel):
    """Batch weight prediction request"""
    images: List[WeightRequestEnhanced] = Field(
        ...,
        max_items=50,
        description="List of weight prediction requests"
    )
    
    @validator('images')
    def validate_batch_size(cls, v):
        if len(v) > 50:
            raise ValueError('Batch size cannot exceed 50 requests')
        return v


class BatchWeightResponse(BaseModel):
    """Batch weight prediction response"""
    successful: int = Field(..., ge=0, description="Number of successful predictions")
    failed: int = Field(..., ge=0, description="Number of failed predictions")
    results: List[WeightResponseEnhanced] = Field(..., description="Successful predictions")
    errors: Optional[List[Dict[str, Any]]] = Field(None, description="Failed predictions")
    total_processing_time: float = Field(..., ge=0, description="Total batch processing time")
    average_per_image: float = Field(..., ge=0, description="Average time per image")
    timestamp: datetime = Field(..., description="Batch completion timestamp")


class BatchPriceRequest(BaseModel):
    """Request model for batch price predictions"""
    requests: List[PriceRequest] = Field(
        ...,
        max_items=100,
        description="List of price prediction requests"
    )


class BatchPriceResponse(BaseModel):
    """Response model for batch price predictions"""
    successful: int = Field(
        ...,
        description="Number of successful predictions"
    )
    failed: int = Field(
        ...,
        description="Number of failed predictions"
    )
    results: List[Dict[str, Any]] = Field(
        ...,
        description="Successful predictions"
    )
    errors: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Failed predictions with errors"
    )
    timestamp: datetime = Field(
        ...,
        description="Batch completion timestamp"
    )


# ============================================================
# FORECAST SCHEMAS
# ============================================================

class ForecastRequest(BaseModel):
    """Request model for price forecasting"""
    periods: int = Field(
        30,
        ge=1,
        le=365,
        description="Number of future periods to forecast",
        example=30
    )
    include_confidence: bool = Field(
        True,
        description="Include confidence intervals in forecast"
    )
    start_date: Optional[datetime] = Field(
        None,
        description="Start date for forecast"
    )


class ForecastResponse(BaseModel):
    """Response model for price forecasting"""
    forecasts: List[Dict[str, Any]] = Field(
        ...,
        description="Forecasted prices with dates"
    )
    confidence_intervals: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Confidence intervals for each forecast"
    )
    model_used: str = Field(
        ...,
        description="Model used for forecasting"
    )
    timestamp: datetime = Field(
        ...,
        description="Forecast generation timestamp"
    )


# ============================================================
# MODEL INFO SCHEMAS
# ============================================================

class ModelCapabilities(BaseModel):
    """Model capability information"""
    image_only: bool = Field(..., description="Supports image-only prediction")
    image_with_age: bool = Field(..., description="Supports multi-modal prediction")
    age_fallback: bool = Field(..., description="Can use age fallback if not provided")


class ModelInfoResponseEnhanced(BaseModel):
    """Enhanced model information response"""
    detector: Dict[str, Any] = Field(..., description="Detection model info")
    regressor: Dict[str, Any] = Field(..., description="Weight regression model info")
    age_estimator: Dict[str, Any] = Field(..., description="Age estimation capabilities")
    recommendations: Dict[str, str] = Field(..., description="Usage recommendations")
    timestamp: str = Field(..., description="Response timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "detector": {
                    "model_loaded": True,
                    "model_type": "YOLOv8",
                    "confidence_threshold": 0.5
                },
                "regressor": {
                    "model_loaded": True,
                    "is_multimodal": True,
                    "requires_age": False,
                    "age_scaler_available": True,
                    "capabilities": {
                        "image_only": True,
                        "image_with_age": True,
                        "age_fallback": True
                    }
                },
                "age_estimator": {
                    "available": True,
                    "age_range": "0.5-24 months",
                    "breeds_supported": ["commercial", "duroc", "yorkshire", "landrace"],
                    "genders_supported": ["male", "female", "castrated", "unknown"]
                },
                "recommendations": {
                    "best_accuracy": "Provide both image and age for best results (MAE <3kg)",
                    "acceptable": "Image only provides good estimates (MAE ~4-5kg)",
                    "age_validation": "Age cross-validation helps detect outliers"
                },
                "timestamp": "2024-12-20T10:30:00Z"
            }
        }


# ============================================================
# CALIBRATION SCHEMAS
# ============================================================

class CalibrationRequest(BaseModel):
    """Camera calibration for improved accuracy"""
    camera_height_cm: float = Field(..., gt=0, description="Camera height from ground")
    camera_angle_degrees: float = Field(..., ge=0, le=90, description="Camera angle")
    focal_length_mm: Optional[float] = Field(None, gt=0, description="Camera focal length")
    sensor_size_mm: Optional[float] = Field(None, gt=0, description="Sensor size")
    reference_measurements: Optional[List[Dict[str, float]]] = Field(
        None,
        description="Reference measurements for calibration"
    )


class CalibrationResponse(BaseModel):
    """Camera calibration response"""
    calibrated: bool = Field(..., description="Calibration successful")
    scale_factor: float = Field(..., description="Calculated scale factor")
    confidence: float = Field(..., ge=0, le=1, description="Calibration confidence")
    recommendations: List[str] = Field(..., description="Setup recommendations")
    timestamp: datetime = Field(..., description="Calibration timestamp")


# ============================================================
# COMPARISON SCHEMAS
# ============================================================

class PredictionComparison(BaseModel):
    """Comparison between different prediction methods"""
    image_only: Optional[float] = Field(None, description="Weight from image-only model")
    image_with_age: Optional[float] = Field(None, description="Weight from multi-modal model")
    age_based: Optional[WeightRange] = Field(None, description="Age-based estimate")
    blended: Optional[float] = Field(None, description="Blended prediction")
    recommended: float = Field(..., description="Recommended final weight")
    method_used: PredictionMethod = Field(..., description="Method used for final prediction")
    
    class Config:
        use_enum_values = True