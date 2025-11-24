# app/routers/scan.py - ENHANCED VERSION
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import Optional
import io
from PIL import Image
import numpy as np
from datetime import datetime
import uuid
import json

from logger import setup_logger
from models.yolo_detector import HogDetector
from models.cnn_regressor import WeightRegressor

logger = setup_logger(__name__)
router = APIRouter()

# Initialize models
detector = HogDetector()
weight_estimator = WeightRegressor()

def generate_detection_id():
    """Generate unique detection ID"""
    return f"det_{uuid.uuid4().hex[:12]}"

def generate_scan_id():
    """Generate unique scan ID"""
    return f"scan_{uuid.uuid4().hex[:12]}"


@router.post(
    "/detect",
    summary="Stage 1a: Detect hogs using YOLO",
    description="Detect all hogs in image and return bounding boxes with coordinates"
)
async def detect_hogs(
    image: UploadFile = File(..., description="Image file to analyze"),
    confidence_threshold: Optional[float] = Form(0.5, description="Minimum confidence threshold (0.0-1.0)"),
    iou_threshold: Optional[float] = Form(0.4, description="IoU threshold for NMS (0.0-1.0)"),
    max_detections: Optional[int] = Form(10, description="Maximum number of detections to return")
):
    """
    Stage 1a: Detect hogs in image using YOLO model.
    
    This endpoint performs object detection to find all hogs in the provided image.
    
    **Process:**
    1. Receives image file
    2. Runs YOLO detection model
    3. Returns bounding boxes for all detected hogs
    
    **Returns:**
    - bounding_boxes: List of detected hog bounding boxes with x, y, width, height
    - overall_confidence: Average confidence across all detections
    - detection_id: Unique ID for this detection session
    - total_detections: Number of hogs detected
    - metadata: Additional information (image size, timestamp, model info)
    
    **Next Step:** Use the detection_id and select a bounding box to call /scan for weight estimation
    """
    try:
        logger.info("=" * 60)
        logger.info("🔍 STAGE 1a: YOLO HOG DETECTION STARTED")
        logger.info(f"   Confidence threshold: {confidence_threshold}")
        logger.info(f"   IoU threshold: {iou_threshold}")
        logger.info(f"   Max detections: {max_detections}")
        
        # Validate parameters
        if not (0.0 <= confidence_threshold <= 1.0):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid confidence threshold",
                    "code": "INVALID_CONFIDENCE",
                    "message": "Confidence threshold must be between 0.0 and 1.0"
                }
            )
        
        if not (0.0 <= iou_threshold <= 1.0):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid IoU threshold",
                    "code": "INVALID_IOU",
                    "message": "IoU threshold must be between 0.0 and 1.0"
                }
            )
        
        if max_detections < 1:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid max detections",
                    "code": "INVALID_MAX_DET",
                    "message": "Max detections must be at least 1"
                }
            )
        
        # Read and validate image
        try:
            contents = await image.read()
            img = Image.open(io.BytesIO(contents))
            logger.info(f"📷 Image loaded: {img.size[0]}x{img.size[1]}, mode: {img.mode}")
        except Exception as e:
            logger.error(f"❌ Failed to read image: {e}")
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid image file",
                    "code": "INVALID_IMAGE",
                    "message": "Could not read image file. Please ensure it's a valid image format."
                }
            )
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            logger.info(f"🔄 Converting image from {img.mode} to RGB")
            img = img.convert('RGB')
        
        img_array = np.array(img)
        logger.info(f"✓ Image array shape: {img_array.shape}")
        
        # Check if YOLO model is loaded
        if not detector.model_loaded:
            logger.error("❌ YOLO model not loaded!")
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "Detection service unavailable",
                    "code": "MODEL_NOT_LOADED",
                    "message": "YOLO model failed to load. Please check server logs and ensure model file exists.",
                    "model_path": detector.model_path
                }
            )
        
        # Run YOLO detection
        logger.info("🤖 Running YOLO detection...")
        detections = detector.detect(
            img_array,
            conf_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            max_det=max_detections
        )
        
        # Log detection results
        logger.info(f"✅ YOLO detection complete - {len(detections)} hog(s) detected")
        
        if len(detections) == 0:
            logger.warning("⚠️  No hogs detected in image")
        else:
            # Log each detection with details
            for i, det in enumerate(detections):
                logger.info(
                    f"   Detection {i+1}: "
                    f"x={det['x']:.1f}, y={det['y']:.1f}, "
                    f"w={det['width']:.1f}, h={det['height']:.1f}, "
                    f"confidence={det['confidence']:.2%}, "
                    f"class={det['class_name']}"
                )
        
        # Calculate overall confidence
        overall_confidence = (
            sum(d['confidence'] for d in detections) / len(detections) 
            if detections else 0.0
        )
        
        detection_id = generate_detection_id()
        
        # Prepare response
        response = {
            "bounding_boxes": detections,
            "overall_confidence": round(overall_confidence, 4),
            "detection_id": detection_id,
            "total_detections": len(detections),
            "metadata": {
                "image_size": list(img.size),
                "timestamp": datetime.now().isoformat(),
                "model_type": "YOLOv8",
                "model_path": detector.model_path,
                "confidence_threshold": confidence_threshold,
                "iou_threshold": iou_threshold,
                "max_detections_requested": max_detections
            }
        }
        
        logger.info(f"📦 Returning detection result: {detection_id}")
        logger.info(f"   Overall confidence: {overall_confidence:.2%}")
        logger.info("=" * 60)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected detection error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Detection failed",
                "code": "DETECTION_ERROR",
                "message": f"An unexpected error occurred during detection: {str(e)}"
            }
        )


@router.post(
    "/scan",
    summary="Stage 1b: Estimate weight for detected hog",
    description="Predict weight from detected hog using CNN regressor model"
)
async def scan_image(
    image: UploadFile = File(..., description="Image file to analyze"),
    selected_hog_id: Optional[str] = Form(None, description="Detection ID from previous /detect call"),
    bbox: Optional[str] = Form(None, description="JSON string of bounding box coordinates {x, y, width, height}")
):
    """
    Stage 1b: Predict weight from detected hog.
    
    This endpoint should be called after /detect to estimate the weight
    of a specific detected hog using its bounding box coordinates.
    
    **Process:**
    1. Receives image and bounding box from detection
    2. Crops/processes the region of interest
    3. Runs CNN weight estimation model
    4. Returns estimated weight with confidence
    
    **Returns:**
    - estimated_weight: Predicted weight in kg
    - confidence: Confidence score of the prediction (0.0-1.0)
    - detection_id: Unique ID for tracking (or scan_id)
    - unit: Weight unit (kg)
    - metadata: Additional information
    
    **Next Step:** Call /confirm to confirm the weight and proceed to price prediction
    """
    try:
        logger.info("=" * 60)
        logger.info("⚖️  STAGE 1b: WEIGHT ESTIMATION STARTED")
        if selected_hog_id:
            logger.info(f"   Detection ID: {selected_hog_id}")
        
        # Read and validate image
        try:
            contents = await image.read()
            img = Image.open(io.BytesIO(contents))
            logger.info(f"📷 Image loaded: {img.size[0]}x{img.size[1]}, mode: {img.mode}")
        except Exception as e:
            logger.error(f"❌ Failed to read image: {e}")
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid image file",
                    "code": "INVALID_IMAGE",
                    "message": "Could not read image file"
                }
            )
        
        # Convert to RGB
        if img.mode != 'RGB':
            logger.info(f"🔄 Converting image from {img.mode} to RGB")
            img = img.convert('RGB')
        
        img_array = np.array(img)
        
        # Parse and validate bbox if provided
        bbox_dict = None
        if bbox:
            try:
                bbox_dict = json.loads(bbox)
                logger.info(
                    f"📐 Using bounding box: "
                    f"x={bbox_dict.get('x', 0):.1f}, "
                    f"y={bbox_dict.get('y', 0):.1f}, "
                    f"w={bbox_dict.get('width', 0):.1f}, "
                    f"h={bbox_dict.get('height', 0):.1f}"
                )
                
                # Validate bbox coordinates
                if any(k not in bbox_dict for k in ['x', 'y', 'width', 'height']):
                    raise ValueError("Bounding box must contain x, y, width, and height")
                
                # Check if bbox is within image bounds
                if (bbox_dict['x'] < 0 or bbox_dict['y'] < 0 or
                    bbox_dict['x'] + bbox_dict['width'] > img.size[0] or
                    bbox_dict['y'] + bbox_dict['height'] > img.size[1]):
                    logger.warning("⚠️  Bounding box extends outside image bounds")
                    
            except json.JSONDecodeError as e:
                logger.error(f"❌ Invalid bbox JSON: {e}")
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Invalid bounding box format",
                        "code": "INVALID_BBOX",
                        "message": "Bounding box must be valid JSON with x, y, width, height fields"
                    }
                )
            except ValueError as e:
                logger.error(f"❌ Invalid bbox data: {e}")
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Invalid bounding box data",
                        "code": "INVALID_BBOX_DATA",
                        "message": str(e)
                    }
                )
        else:
            logger.info("📐 No bounding box provided - using full image")
        
        # Check if weight model is loaded
        if not weight_estimator.model_loaded:
            logger.error("❌ Weight estimation model not loaded!")
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "Weight estimation service unavailable",
                    "code": "MODEL_NOT_LOADED",
                    "message": "CNN weight model failed to load. Please check server logs."
                }
            )
        
        # Predict weight
        logger.info("🤖 Running CNN weight estimation...")
        weight, confidence = weight_estimator.predict_weight(img_array, bbox_dict)
        
        # Validate weight result
        if weight <= 0:
            logger.warning(f"⚠️  Invalid weight prediction: {weight} kg")
            weight = 0.0
            confidence = 0.0
        
        logger.info(f"✅ Weight estimation complete: {weight:.2f} kg (confidence: {confidence:.2%})")
        
        scan_id = selected_hog_id or generate_scan_id()
        
        # Prepare response
        response = {
            "estimated_weight": round(weight, 2),
            "confidence": round(confidence, 4),
            "scan_id": scan_id,  # Use scan_id for consistency with SARIMA
            "detection_id": scan_id,  # Keep for backward compatibility
            "unit": "kg",
            "metadata": {
                "image_size": list(img.size),
                "timestamp": datetime.now().isoformat(),
                "bbox_used": bbox_dict is not None,
                "model_type": "CNN_Regressor",
                "roi_cropped": bbox_dict is not None
            }
        }
        
        logger.info(f"📦 Returning weight result: {scan_id}")
        logger.info("=" * 60)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected weight estimation error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Weight estimation failed",
                "code": "SCAN_ERROR",
                "message": f"An unexpected error occurred during weight estimation: {str(e)}"
            }
        )


@router.get(
    "/models/status",
    summary="Check detection models status",
    description="Get status and information about YOLO and CNN weight estimation models"
)
async def get_models_status():
    """
    Check if detection models are loaded and operational.
    
    **Returns:**
    - yolo_model: Status and info about YOLO detection model
    - weight_model: Status and info about CNN weight estimation model
    - overall_status: Overall system health status
    """
    yolo_status = "operational" if detector.model_loaded else "unavailable"
    weight_status = "operational" if weight_estimator.model_loaded else "unavailable"
    
    overall_status = "healthy" if (detector.model_loaded and weight_estimator.model_loaded) else "degraded"
    
    response = {
        "yolo_model": {
            "loaded": detector.model_loaded,
            "model_path": detector.model_path,
            "model_type": "YOLOv8",
            "status": yolo_status,
            "target_class": "hog"
        },
        "weight_model": {
            "loaded": weight_estimator.model_loaded,
            "model_path": getattr(weight_estimator, 'model_path', None),
            "model_type": "CNN_Regressor",
            "status": weight_status
        },
        "overall_status": overall_status,
        "timestamp": datetime.now().isoformat()
    }
    
    logger.info(f"📊 Models status check: YOLO={yolo_status}, Weight={weight_status}, Overall={overall_status}")
    
    return response