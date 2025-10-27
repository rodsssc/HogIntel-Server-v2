# app/routers/scan.py
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import Optional
import io
from PIL import Image
import numpy as np
from datetime import datetime
import uuid

from logger import setup_logger
from models.yolo_detector import HogDetector
from models.cnn_regressor import WeightRegressor

logger = setup_logger(__name__)
router = APIRouter()

# Initialize models
detector = HogDetector()
weight_estimator = WeightRegressor()

def generate_detection_id():
    return f"det_{uuid.uuid4().hex[:12]}"

@router.post("/detect")
async def detect_hogs(
    image: UploadFile = File(...),
    confidence_threshold: Optional[float] = Form(0.5),
    iou_threshold: Optional[float] = Form(0.4),
    max_detections: Optional[int] = Form(10)
):
    """Stage 1a: Detect hogs in image"""
    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img_array = np.array(img)
        
        # Run detection
        detections = detector.detect(
            img_array,
            conf_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            max_det=max_detections
        )
        
        return {
            "bounding_boxes": detections,
            "overall_confidence": sum(d['confidence'] for d in detections) / len(detections) if detections else 0,
            "detection_id": generate_detection_id(),
            "total_detections": len(detections),
            "metadata": {
                "image_size": list(img.size),
                "timestamp": datetime.now().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Detection error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/scan")
async def scan_image(
    image: UploadFile = File(...),
    selected_hog_id: Optional[str] = Form(None),
    bbox: Optional[str] = Form(None)  # JSON string of bbox
):
    """Stage 1b: Predict weight from detected hog"""
    try:
        import json
        
        contents = await image.read()
        img = Image.open(io.BytesIO(contents))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img_array = np.array(img)
        
        # Parse bbox if provided
        bbox_dict = json.loads(bbox) if bbox else None
        
        # Predict weight
        weight, confidence = weight_estimator.predict_weight(img_array, bbox_dict)
        
        return {
            "estimated_weight": round(weight, 2),
            "confidence": round(confidence, 2),
            "detection_id": generate_detection_id(),
            "unit": "kg",
            "metadata": {
                "image_size": list(img.size),
                "timestamp": datetime.now().isoformat(),
                "bbox_used": bbox_dict is not None
            }
        }
    except Exception as e:
        logger.error(f"Scan error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))