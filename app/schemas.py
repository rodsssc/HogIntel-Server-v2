from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class ScanStatus(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    ADJUSTED = "adjusted"

class WeightRequest(BaseModel):
    image_data: str = Field(..., description="Base64 encoded image data")
    calibration_data: Optional[Dict[str, Any]] = Field(None, description="Camera calibration parameters")
    user_id: Optional[str] = Field(None, description="Optional user identifier")

class WeightResponse(BaseModel):
    weight_kg: float = Field(..., description="Predicted weight in kilograms")
    confidence: float = Field(..., description="Confidence score (0-1)")
    detection_bbox: List[float] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    roi_cropped: bool = Field(..., description="Whether ROI was successfully cropped")
    processing_time: float = Field(..., description="Processing time in seconds")
    scan_id: str = Field(..., description="Unique scan identifier")

class ConfirmRequest(BaseModel):
    scan_id: str = Field(..., description="Scan identifier from weight prediction")
    confirmed_weight: Optional[float] = Field(None, description="User confirmed/adjusted weight")
    status: ScanStatus = Field(..., description="User action on weight prediction")

class ConfirmResponse(BaseModel):
    success: bool = Field(..., description="Confirmation recorded successfully")
    confirmed_weight: float = Field(..., description="Final confirmed weight")
    scan_id: str = Field(..., description="Scan identifier")
    timestamp: datetime = Field(..., description="Confirmation timestamp")

class PriceRequest(BaseModel):
    scan_id: str = Field(..., description="Scan identifier from confirmed weight")
    confirmed_weight: float = Field(..., description="Confirmed weight in kg")
    market_data: Optional[Dict[str, Any]] = Field(None, description="Additional market parameters")
    use_fallback: bool = Field(False, description="Use Prophet fallback if available")

class PriceResponse(BaseModel):
    price_per_kg: float = Field(..., description="Predicted price per kilogram")
    total_value: float = Field(..., description="Total value (price_per_kg * weight)")
    confidence: float = Field(..., description="Price prediction confidence")
    model_used: str = Field(..., description="Which model was used (XGBoost/Prophet)")
    market_conditions: Optional[Dict[str, Any]] = Field(None, description="Market context")
    timestamp: datetime = Field(..., description="Prediction timestamp")

class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    code: str = Field(..., description="Error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")