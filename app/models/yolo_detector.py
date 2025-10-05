# app/models/yolo_detector.py
from ultralytics import YOLO
import numpy as np
from logger import setup_logger

logger = setup_logger(__name__)

class HogDetector:
    def __init__(self, model_path=r'C:\Users\Acer\OneDrive\Desktop\HogIntel-Price&Weight-Estimation\models\weights\best.pt'):
        try:
            self.model = YOLO(model_path)
            logger.info(f"YOLO model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self.model = None
    
    def detect(self, image_array, conf_threshold=0.5, iou_threshold=0.4, max_det=10):
        """
        Detect hogs in image
        Returns: list of bounding boxes with coordinates
        """
        if self.model is None:
            return []
        
        try:
            results = self.model.predict(
                image_array,
                conf=conf_threshold,
                iou=iou_threshold,
                max_det=max_det,
                verbose=False
            )
            
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    detections.append({
                        "x": float(x1),
                        "y": float(y1),
                        "width": float(x2 - x1),
                        "height": float(y2 - y1),
                        "confidence": float(box.conf[0]),
                        "class_name": "hog",
                        "class_id": int(box.cls[0])
                    })
            
            return detections
        except Exception as e:
            logger.error(f"Detection failed: {e}", exc_info=True)
            return []