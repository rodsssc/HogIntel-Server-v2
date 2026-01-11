# app/routers/weight.py
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
import base64
import numpy as np
import cv2
from datetime import datetime
import time
import uuid
from logger import setup_logger
from models.yolo_detector import HogDetector
from models.cnn_regressor import WeightRegressor
from models.age_weight_estimator import AgeWeightEstimator

logger = setup_logger(__name__)
router = APIRouter()

# Initialize models
hog_detector = HogDetector()
weight_regressor = WeightRegressor()
age_estimator = AgeWeightEstimator()


class WeightRequest(BaseModel):
    image_data: str = Field(..., description="Base64 encoded image", min_length=100)
    age_months: Optional[float] = Field(None, ge=0.5, le=24, description="Age of hog in months")
    breed: Optional[str] = Field("commercial", description="Breed type")
    gender: Optional[str] = Field("unknown", description="Gender")
    calibration_data: Optional[Dict[str, Any]] = Field(None, description="Camera calibration")
    user_id: Optional[str] = Field(None, max_length=50, description="User identifier")
    
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


class WeightResponse(BaseModel):
    weight_kg: float = Field(..., description="Predicted weight in kg")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    detection_bbox: List[float] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    roi_cropped: bool = Field(..., description="ROI successfully cropped")
    processing_time: float = Field(..., description="Processing time in seconds")
    scan_id: str = Field(..., description="Unique scan identifier")
    model_used: str = Field(..., description="Model architecture used")
    quality_metrics: Optional[Dict[str, Any]] = Field(None, description="Image quality metrics")
    
    # Age-related fields
    age_provided: bool = Field(..., description="Whether age was provided")
    age_months: Optional[float] = Field(None, description="Age used in prediction")
    age_validation: Optional[Dict[str, Any]] = Field(None, description="Age-based validation")
    prediction_method: str = Field(..., description="Prediction method used")


@router.post(
    "/predict",
    response_model=WeightResponse,
    summary="Predict hog weight from image with optional age",
    description="Stage 1b: Enhanced weight prediction using image + optional age information"
)
async def predict_weight(request: WeightRequest):
    """
    Enhanced weight prediction with multi-modal support
    
    Features:
    - Image-only prediction (legacy support)
    - Image + Age prediction (recommended for best accuracy)
    - Age-based validation and cross-checking
    - Blended predictions when both available
    
    Target Performance:
    - Image only: MAE ~4-5kg, R² ~0.88
    - Image + Age: MAE <3kg, R² >0.92
    """
    start_time = time.time()
    
    try:
        # Generate scan ID
        scan_id = f"scan_{uuid.uuid4().hex[:12]}"
        
        # Decode image
        logger.info(f"🔍 Processing scan {scan_id}")
        image_data = request.image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Stage 1: Detect hog and get ROI
        logger.info("🎯 Stage 1a: Detecting hog...")
        detections = hog_detector.detect(image)
        
        if not detections or len(detections) == 0:
            raise HTTPException(
                status_code=404,
                detail="No hog detected in image. Please ensure the hog is clearly visible."
            )
        
        # Get best detection
        best_detection = max(detections, key=lambda x: x['confidence'])
        bbox = best_detection['bbox']
        detection_conf = best_detection['confidence']
        
        logger.info(f"✅ Hog detected with confidence: {detection_conf:.2f}")
        
        # Stage 2: Weight prediction
        logger.info("⚖️ Stage 1b: Predicting weight...")
        
        # Check if age is provided
        age_provided = request.age_months is not None
        prediction_method = "unknown"
        age_validation = None
        
        # Get CNN prediction
        weight_kg, cnn_confidence, cnn_metadata = weight_regressor.predict_weight(
            image_rgb,
            bbox=bbox,
            age_months=request.age_months,
            use_age_fallback=not age_provided
        )
        
        # Determine prediction method
        if cnn_metadata.get("model_type") == "multimodal" and age_provided:
            prediction_method = "multimodal_cnn"
            logger.info("🔥 Using multi-modal CNN (Image + Age)")
        elif cnn_metadata.get("model_type") == "multimodal" and not age_provided:
            prediction_method = "multimodal_cnn_fallback"
            logger.warning("⚠️ Multi-modal model using age fallback")
        else:
            prediction_method = "image_only_cnn"
            logger.info("📷 Using image-only CNN")
        
        # If age provided, do cross-validation
        if age_provided:
            try:
                comparison = age_estimator.compare_with_image_prediction(
                    age_months=request.age_months,
                    image_predicted_weight=weight_kg,
                    breed=request.breed or "commercial",
                    gender=request.gender or "unknown"
                )
                
                age_validation = {
                    "age_based_estimate": comparison["age_based_estimate"],
                    "within_expected_range": comparison["within_expected_range"],
                    "deviation_percent": comparison["deviation_percent"],
                    "agreement_level": comparison["agreement_level"],
                    "blended_weight": comparison["blended_weight"],
                    "recommendation": comparison["recommendation"]
                }
                
                # Log if predictions don't align
                if comparison["agreement_level"] in ["underweight_concern", "overweight_concern"]:
                    logger.warning(
                        f"⚠️ Prediction mismatch: {comparison['agreement_level']} - "
                        f"Image: {weight_kg:.1f}kg, Age-based: {comparison['age_based_estimate']['average']:.1f}kg"
                    )
                
                # Use blended weight if predictions are very different
                if comparison["agreement_level"] in ["underweight_concern", "overweight_concern"]:
                    if prediction_method == "image_only_cnn":
                        # For image-only models, prefer the blended weight
                        logger.info(f"📊 Using blended weight: {comparison['blended_weight']:.2f}kg")
                        weight_kg = comparison['blended_weight']
                        prediction_method = "blended_image_age"
                        
            except Exception as e:
                logger.error(f"Age validation error: {e}")
                age_validation = {"error": str(e)}
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Quality metrics
        quality_metrics = {
            "detection_confidence": float(detection_conf),
            "image_resolution": f"{image.shape[1]}x{image.shape[0]}",
            "roi_size": f"{int(bbox['width'])}x{int(bbox['height'])}",
            "brightness": float(np.mean(image_rgb)),
            "sharpness": float(cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
        }
        
        # Build response
        response = WeightResponse(
            weight_kg=round(weight_kg, 2),
            confidence=round(cnn_confidence, 2),
            detection_bbox=[
                float(bbox['x1']),
                float(bbox['y1']),
                float(bbox['x2']),
                float(bbox['y2'])
            ],
            roi_cropped=True,
            processing_time=round(processing_time, 2),
            scan_id=scan_id,
            model_used=weight_regressor.get_model_info()["model_loaded"] and "cnn_regressor" or "fallback",
            quality_metrics=quality_metrics,
            age_provided=age_provided,
            age_months=request.age_months,
            age_validation=age_validation,
            prediction_method=prediction_method
        )
        
        logger.info(
            f"✅ Prediction complete: {weight_kg:.2f}kg "
            f"(confidence: {cnn_confidence:.2f}, method: {prediction_method}, time: {processing_time:.2f}s)"
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Weight prediction failed: {e}")
        logger.exception("Full traceback:")
        raise HTTPException(
            status_code=500,
            detail=f"Weight prediction failed: {str(e)}"
        )


@router.get("/model-info")
async def get_model_info():
    """
    Get information about loaded models
    
    Returns details about:
    - Detection model (YOLO)
    - Weight prediction model (CNN)
    - Model capabilities (image-only vs multi-modal)
    """
    detector_info = hog_detector.get_model_info()
    regressor_info = weight_regressor.get_model_info()
    
    return {
        "detector": detector_info,
        "regressor": regressor_info,
        "age_estimator": {
            "available": True,
            "age_range": "0.5-24 months",
            "breeds_supported": list(age_estimator.breed_factors.keys()),
            "genders_supported": list(age_estimator.gender_factors.keys())
        },
        "recommendations": {
            "best_accuracy": "Provide both image and age for best results (MAE <3kg)",
            "acceptable": "Image only provides good estimates (MAE ~4-5kg)",
            "age_validation": "Age cross-validation helps detect outliers"
        },
        "timestamp": datetime.now().isoformat()
    }


@router.get("/health")
async def health_check():
    """Health check for weight prediction service"""
    detector_status = hog_detector.model is not None
    regressor_status = weight_regressor.model_loaded
    
    return {
        "status": "healthy" if (detector_status and regressor_status) else "degraded",
        "detector_loaded": detector_status,
        "regressor_loaded": regressor_status,
        "regressor_type": "multimodal" if weight_regressor.is_multimodal else "image_only",
        "age_estimator_available": True,
        "features": {
            "detection": detector_status,
            "weight_prediction": regressor_status,
            "age_support": weight_regressor.is_multimodal,
            "age_validation": True,
            "blended_predictions": True
        },
        "timestamp": datetime.now().isoformat()
    }


@router.post("/predict-with-file")
async def predict_weight_from_file(
    file: UploadFile = File(...),
    age_months: Optional[float] = Form(None),
    breed: Optional[str] = Form("commercial"),
    gender: Optional[str] = Form("unknown")
):
    """
    Alternative endpoint accepting file upload instead of base64
    
    Useful for:
    - Testing with tools like Postman
    - Mobile apps that work better with multipart/form-data
    - Large images (no base64 overhead)
    """
    try:
        # Read file
        contents = await file.read()
        
        # Convert to base64 for consistency with main endpoint
        image_b64 = base64.b64encode(contents).decode('utf-8')
        
        # Determine content type
        content_type = file.content_type or "image/jpeg"
        if "jpeg" in content_type or "jpg" in content_type:
            image_data = f"data:image/jpeg;base64,{image_b64}"
        elif "png" in content_type:
            image_data = f"data:image/png;base64,{image_b64}"
        else:
            raise HTTPException(status_code=400, detail="Unsupported image format. Use JPEG or PNG.")
        
        # Create request object
        request = WeightRequest(
            image_data=image_data,
            age_months=age_months,
            breed=breed,
            gender=gender
        )
        
        # Use main prediction logic
        return await predict_weight(request)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File upload prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))