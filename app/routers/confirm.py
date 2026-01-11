# app/routers/confirm.py - Enhanced Error Handling
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any
from datetime import datetime
from logger import setup_logger
from models.age_weight_estimator import AgeWeightEstimator
import json

logger = setup_logger(__name__)
router = APIRouter()

# Initialize age weight estimator
age_estimator = AgeWeightEstimator()

class ConfirmRequest(BaseModel):
    scan_id: str = Field(..., description="Scan identifier from weight prediction")
    confirmed_weight: float = Field(..., ge=0, le=500, description="User confirmed/adjusted weight in kg")
    status: str = Field(..., description="accepted, rejected, or adjusted")
    notes: Optional[str] = Field(None, max_length=500, description="Additional notes")
    age_months: Optional[float] = Field(None, ge=0.5, le=24, description="Age of hog in months")
    breed: Optional[str] = Field("commercial", description="Breed type: commercial, duroc, yorkshire, etc.")
    gender: Optional[str] = Field("unknown", description="Gender: male, female, castrated, unknown")
    
    @validator('status')
    def validate_status(cls, v):
        valid_statuses = ["accepted", "rejected", "adjusted"]
        if v not in valid_statuses:
            raise ValueError(
                f"Invalid status: '{v}'. Must be one of: {', '.join(valid_statuses)}"
            )
        return v

class ConfirmResponse(BaseModel):
    success: bool
    confirmed_weight: float
    scan_id: str
    timestamp: datetime
    status: str
    previous_weight: Optional[float] = None
    adjustment: Optional[float] = None
    age_validation: Optional[Dict[str, Any]] = None
    age_months: Optional[float] = None

class AgeEstimateRequest(BaseModel):
    age_months: float = Field(..., ge=0.5, le=24, description="Age of hog in months")
    breed: Optional[str] = Field("commercial", description="Breed type")
    gender: Optional[str] = Field("unknown", description="Gender")

class AgeEstimateResponse(BaseModel):
    estimated_weight_kg: Dict[str, float]
    age_months: float
    breed: str
    gender: str
    confidence: float
    breed_factor: float
    gender_factor: float
    growth_stage: str
    market_ready: bool
    timestamp: str

class WeightComparisonRequest(BaseModel):
    age_months: float = Field(..., ge=0.5, le=24)
    image_predicted_weight: float = Field(..., ge=0, le=500)
    breed: Optional[str] = Field("commercial")
    gender: Optional[str] = Field("unknown")

class WeightComparisonResponse(BaseModel):
    age_based_estimate: Dict[str, float]
    image_predicted_weight: float
    blended_weight: float
    within_expected_range: bool
    deviation_kg: float
    deviation_percent: float
    agreement_level: str
    recommendation: str
    confidence_scores: Dict[str, float]

class GrowthRecommendationRequest(BaseModel):
    age_months: float = Field(..., ge=0.5, le=24)
    current_weight: float = Field(..., ge=0, le=500)

class GrowthRecommendationResponse(BaseModel):
    current_status: Dict[str, Any]
    market_projection: Dict[str, Any]
    recommendations: list

@router.post(
    "/confirm",
    response_model=ConfirmResponse,
    summary="Confirm weight prediction with optional age validation",
    description="Stage 1c: User confirms, rejects, or adjusts the predicted weight. Optionally validate against age-based estimates."
)
async def confirm_weight(request: ConfirmRequest, raw_request: Request):
    """
    Stage 1c: User confirms/adjusts weight with optional age-based validation
    
    Valid status values: 'accepted', 'rejected', 'adjusted'
    
    If age_months is provided, the system will:
    1. Estimate expected weight based on age
    2. Compare with confirmed weight
    3. Flag any significant deviations
    4. Provide recommendations
    """
    try:
        # Log incoming request for debugging
        body = await raw_request.body()
        logger.info(f"📥 Confirm request received: {body.decode()}")
        
        # Additional validation
        if request.confirmed_weight <= 0:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Weight must be greater than 0",
                    "received_weight": request.confirmed_weight
                }
            )
        
        # Perform age-based validation if age is provided
        age_validation = None
        if request.age_months is not None:
            try:
                comparison = age_estimator.compare_with_image_prediction(
                    age_months=request.age_months,
                    image_predicted_weight=request.confirmed_weight,
                    breed=request.breed or "commercial",
                    gender=request.gender or "unknown"
                )
                
                age_validation = {
                    "age_based_estimate": comparison["age_based_estimate"],
                    "within_expected_range": comparison["within_expected_range"],
                    "deviation_percent": comparison["deviation_percent"],
                    "agreement_level": comparison["agreement_level"],
                    "recommendation": comparison["recommendation"],
                    "blended_weight": comparison["blended_weight"]
                }
                
                # Log if there's a significant deviation
                if comparison["agreement_level"] in ["underweight_concern", "overweight_concern"]:
                    logger.warning(
                        f"⚠️ Weight deviation detected for scan {request.scan_id}: "
                        f"{comparison['agreement_level']} - {comparison['recommendation']}"
                    )
                    
            except Exception as e:
                logger.error(f"Age validation error: {e}")
                # Don't fail the confirmation if age validation fails
                age_validation = {"error": str(e)}
        
        # Store confirmation for feedback loop
        logger.info(
            f"✅ Weight {request.status}: {request.confirmed_weight}kg for scan {request.scan_id}"
            + (f" (Age: {request.age_months} months)" if request.age_months else "")
        )
        
        # In a real implementation, save to database here
        # database.store_confirmation(request)
        
        response_data = {
            "success": True,
            "confirmed_weight": request.confirmed_weight,
            "scan_id": request.scan_id,
            "timestamp": datetime.now(),
            "status": request.status,
            "age_validation": age_validation,
            "age_months": request.age_months
        }
        
        # Add adjustment info if status is adjusted
        if request.status == "adjusted" and request.notes:
            logger.info(f"📝 User notes: {request.notes}")
        
        logger.info(f"✅ Confirmation successful for scan {request.scan_id}")
        return ConfirmResponse(**response_data)
        
    except HTTPException:
        raise
    except ValueError as e:
        # Catch validation errors from Pydantic
        logger.error(f"❌ Validation error: {e}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Validation error",
                "message": str(e),
                "valid_status_values": ["accepted", "rejected", "adjusted"]
            }
        )
    except Exception as e:
        logger.error(f"❌ Confirmation error: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Confirmation failed: {str(e)}"
        )

@router.post(
    "/estimate-by-age",
    response_model=AgeEstimateResponse,
    summary="Estimate weight based on age",
    description="Get expected weight range for a hog based on its age, breed, and gender"
)
async def estimate_weight_by_age(request: AgeEstimateRequest):
    """
    Estimate weight based on age
    
    Returns expected weight range based on:
    - Age in months
    - Breed type (affects growth rate)
    - Gender (males typically heavier)
    
    Example:
    - 3 months old commercial pig: 30-45 kg (avg: 37.5 kg)
    - 6 months old duroc boar: 79-105 kg (avg: 92 kg)
    """
    try:
        result = age_estimator.estimate_weight_from_age(
            age_months=request.age_months,
            breed=request.breed,
            gender=request.gender
        )
        
        return AgeEstimateResponse(**result)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Age estimation error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Age estimation failed: {str(e)}"
        )

@router.post(
    "/compare-weights",
    response_model=WeightComparisonResponse,
    summary="Compare image prediction with age-based estimate",
    description="Validate image-based weight prediction against age-based expectations"
)
async def compare_weight_predictions(request: WeightComparisonRequest):
    """
    Compare image-based weight prediction with age-based estimate
    
    Useful for:
    - Validating image predictions
    - Detecting outliers
    - Getting blended/averaged predictions
    - Identifying potential health concerns
    
    Returns:
    - Comparison analysis
    - Agreement level
    - Blended weight recommendation
    - Deviation metrics
    """
    try:
        result = age_estimator.compare_with_image_prediction(
            age_months=request.age_months,
            image_predicted_weight=request.image_predicted_weight,
            breed=request.breed,
            gender=request.gender
        )
        
        return WeightComparisonResponse(**result)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Weight comparison error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Weight comparison failed: {str(e)}"
        )

@router.post(
    "/growth-recommendations",
    response_model=GrowthRecommendationResponse,
    summary="Get growth recommendations",
    description="Get growth analysis and recommendations based on current age and weight"
)
async def get_growth_recommendations(request: GrowthRecommendationRequest):
    """
    Get growth recommendations based on current age and weight
    
    Provides:
    - Current growth status vs expected
    - Days to market weight (100kg)
    - Growth recommendations
    - Feeding and management suggestions
    
    Useful for:
    - Growth monitoring
    - Feed program optimization
    - Market timing decisions
    - Health issue detection
    """
    try:
        result = age_estimator.get_growth_recommendations(
            age_months=request.age_months,
            current_weight=request.current_weight
        )
        
        return GrowthRecommendationResponse(**result)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Growth recommendations error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Growth recommendations failed: {str(e)}"
        )

@router.get("/age-chart", summary="Get complete age-weight reference chart")
async def get_age_weight_chart():
    """
    Get the complete age-weight reference chart
    
    Returns standard growth curves for commercial pigs including:
    - Monthly weight ranges (1-12 months)
    - Breed adjustment factors
    - Gender adjustment factors
    - Growth stage classifications
    """
    return {
        "age_weight_chart": age_estimator.age_weight_chart,
        "breed_factors": age_estimator.breed_factors,
        "gender_factors": age_estimator.gender_factors,
        "description": {
            "age_ranges": "Standard commercial pig growth from 1-12 months",
            "weights": "Format: (minimum_kg, maximum_kg, average_kg)",
            "breed_factors": "Multipliers applied to base weights",
            "gender_factors": "Multipliers for male/female differences"
        },
        "growth_stages": {
            "nursing": "< 1 month",
            "weaning": "1-2.5 months",
            "grower": "2.5-4 months",
            "finisher": "4-7 months",
            "market_ready": "7+ months"
        }
    }

@router.get("/confirm/health")
async def health_check():
    """Health check for confirmation endpoint"""
    return {
        "status": "healthy",
        "service": "weight_confirmation",
        "features": [
            "weight_confirmation",
            "age_based_estimation",
            "weight_comparison",
            "growth_recommendations"
        ],
        "valid_status_values": ["accepted", "rejected", "adjusted"],
        "age_estimator_ready": True,
        "timestamp": datetime.now()
    }