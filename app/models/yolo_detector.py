# app/models/yolo_detector.py - ENHANCED VERSION
from ultralytics import YOLO
import numpy as np
import os
from logger import setup_logger
from typing import Optional, Dict, Any, List

logger = setup_logger(__name__)


class HogDetector:
    """
    YOLO-based hog detection model.
    
    This class wraps the YOLOv8 model for detecting hogs in images.
    It handles model loading, inference, and post-processing of detections.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the HogDetector.
        
        Args:
            model_path: Path to YOLO model file (.pt). If None, uses environment variable
                       or default container path.
        """
        # Determine model path
        self.model_path = model_path or os.getenv(
            'YOLO_MODEL_PATH', 
            '/models/yolo_model/best.pt'
        )
        
        logger.info("=" * 60)
        logger.info("🚀 INITIALIZING YOLO HOG DETECTOR")
        logger.info(f"   Model path: {self.model_path}")
        
        # Initialize model state
        self.model = None
        self.model_loaded = False
        self.class_names = []
        
        # Load the model
        self._load_model()
        
        logger.info("=" * 60)
    
    def _load_model(self):
        """Load the YOLO model from disk."""
        try:
            # Check if model file exists
            if not os.path.exists(self.model_path):
                logger.error(f"❌ Model file not found: {self.model_path}")
                logger.error(f"   Current directory: {os.getcwd()}")
                logger.error(f"   Directory contents: {os.listdir(os.path.dirname(self.model_path)) if os.path.exists(os.path.dirname(self.model_path)) else 'Directory does not exist'}")
                self.model_loaded = False
                return
            
            # Get model file size
            model_size_mb = os.path.getsize(self.model_path) / (1024 * 1024)
            logger.info(f"📦 Model file size: {model_size_mb:.2f} MB")
            
            # Load YOLO model
            logger.info("⏳ Loading YOLO model...")
            self.model = YOLO(self.model_path)
            
            # Get model information
            if hasattr(self.model, 'names'):
                self.class_names = list(self.model.names.values())
                logger.info(f"✅ Model classes: {self.class_names}")
            
            self.model_loaded = True
            logger.info(f"✅ YOLO model loaded successfully from {self.model_path}")
            logger.info(f"   Model type: {type(self.model).__name__}")
            
        except FileNotFoundError as e:
            logger.error(f"❌ Model file not found: {e}")
            self.model_loaded = False
        except Exception as e:
            logger.error(f"❌ Failed to load YOLO model: {e}", exc_info=True)
            self.model_loaded = False
    
    def detect(
        self, 
        image_array: np.ndarray, 
        conf_threshold: float = 0.5, 
        iou_threshold: float = 0.4, 
        max_det: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Detect hogs in image.
        
        Args:
            image_array: Input image as numpy array (H, W, C) in RGB format
            conf_threshold: Confidence threshold for detections (0.0-1.0)
            iou_threshold: IoU threshold for Non-Maximum Suppression
            max_det: Maximum number of detections to return
        
        Returns:
            List of detection dictionaries with keys:
            - x: X coordinate of top-left corner
            - y: Y coordinate of top-left corner
            - width: Width of bounding box
            - height: Height of bounding box
            - confidence: Detection confidence score
            - class_name: Class name ('hog')
            - class_id: Class ID
        """
        # Check if model is loaded
        if not self.model_loaded or self.model is None:
            logger.error("❌ Cannot detect: Model not loaded")
            return []
        
        # Validate input
        if not isinstance(image_array, np.ndarray):
            logger.error(f"❌ Invalid input type: {type(image_array)}, expected np.ndarray")
            return []
        
        if len(image_array.shape) != 3:
            logger.error(f"❌ Invalid image shape: {image_array.shape}, expected (H, W, C)")
            return []
        
        logger.info(f"🔍 Running YOLO detection on image shape: {image_array.shape}")
        logger.info(f"   Parameters: conf={conf_threshold}, iou={iou_threshold}, max_det={max_det}")
        
        try:
            # Run YOLO prediction
            results = self.model.predict(
                image_array,
                conf=conf_threshold,
                iou=iou_threshold,
                max_det=max_det,
                verbose=False  # Suppress YOLO's verbose output
            )
            
            detections = []
            total_boxes = 0
            
            # Process results
            for result in results:
                boxes = result.boxes
                total_boxes += len(boxes)
                
                for box in boxes:
                    # Extract coordinates (xyxy format from YOLO)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Convert to x, y, width, height format
                    x = float(x1)
                    y = float(y1)
                    width = float(x2 - x1)
                    height = float(y2 - y1)
                    
                    # Extract confidence and class
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    
                    # Get class name
                    class_name = "hog"
                    if hasattr(self.model, 'names') and class_id in self.model.names:
                        class_name = self.model.names[class_id]
                    
                    # Validate detection
                    if width <= 0 or height <= 0:
                        logger.warning(f"⚠️  Invalid bounding box dimensions: w={width}, h={height}")
                        continue
                    
                    detection = {
                        "x": x,
                        "y": y,
                        "width": width,
                        "height": height,
                        "confidence": confidence,
                        "class_name": class_name,
                        "class_id": class_id
                    }
                    
                    detections.append(detection)
            
            logger.info(f"✅ Detection complete: {len(detections)} valid detections from {total_boxes} total boxes")
            
            # Sort by confidence (highest first)
            detections.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Log detection statistics
            if detections:
                confidences = [d['confidence'] for d in detections]
                logger.info(f"   Confidence range: {min(confidences):.2%} - {max(confidences):.2%}")
                logger.info(f"   Average confidence: {sum(confidences)/len(confidences):.2%}")
            
            return detections
            
        except Exception as e:
            logger.error(f"❌ Detection failed: {e}", exc_info=True)
            return []
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "model_loaded": self.model_loaded,
            "model_path": self.model_path,
            "model_exists": os.path.exists(self.model_path),
            "model_type": "YOLOv8",
            "class_names": self.class_names,
            "num_classes": len(self.class_names)
        }
    
    def __repr__(self) -> str:
        """String representation of the detector."""
        status = "loaded" if self.model_loaded else "not loaded"
        return f"HogDetector(model_path='{self.model_path}', status='{status}')"