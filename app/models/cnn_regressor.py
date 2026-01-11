# app/models/cnn_regressor.py
import torch
import torch.nn as nn
import os
import json
from PIL import Image
import torchvision.transforms as transforms
from typing import Optional, Dict, Any, List
from torchvision.models import resnet50, efficientnet_b0
from logger import setup_logger
import numpy as np

logger = setup_logger(__name__)


class MultiModalWeightPredictor(nn.Module):
    """Multi-modal CNN model for hog weight prediction (Image + Age)"""
    
    def __init__(self, backbone='resnet50', pretrained=False, dropout=0.3):
        super(MultiModalWeightPredictor, self).__init__()
        
        # Image feature extractor
        if backbone == 'resnet50':
            self.backbone = resnet50(weights=None)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove classification head
        elif backbone == 'efficientnet_b0':
            self.backbone = efficientnet_b0(weights=None)
            feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Age feature processor
        self.age_processor = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        
        # Combined features dimension
        combined_dim = feature_dim + 128
        
        # Fusion and regression head
        self.regressor = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)  # Single output for weight
        )
    
    def forward(self, image, age):
        # Extract image features
        image_features = self.backbone(image)
        
        # Process age feature
        age_features = self.age_processor(age.unsqueeze(1))
        
        # Concatenate features
        combined_features = torch.cat([image_features, age_features], dim=1)
        
        # Predict weight
        weight = self.regressor(combined_features)
        return weight.squeeze()


class WeightPredictor(nn.Module):
    """Legacy CNN model for backward compatibility (Image only)"""
    
    def __init__(self, backbone='resnet50', pretrained=False, dropout=0.3):
        super(WeightPredictor, self).__init__()
        
        if backbone == 'resnet50':
            self.backbone = resnet50(weights=None)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        weight = self.regressor(features)
        return weight.squeeze()


class WeightRegressor:
    """Enhanced weight regressor supporting both image-only and multi-modal predictions"""
    
    def __init__(self, model_path: Optional[str] = None):
        # Use environment variable or fallback to container default
        self.model_path = model_path or os.getenv(
            'CNN_MODEL_PATH', 
            '/models/weight_model/best_model.pt'
        )
        
        self.model = None
        self.model_loaded = False
        self.is_multimodal = False
        self.age_scaler = None
        
        try:
            logger.info(f"Loading CNN model from: {self.model_path}")
            
            # Check if file exists
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=torch.device('cpu'))
            
            # Detect model type and load accordingly
            if isinstance(checkpoint, dict):
                # Check if this is a multi-modal model
                config = checkpoint.get('config', {})
                self.is_multimodal = 'age' in str(config).lower() or 'multimodal' in str(config).lower()
                
                # Try to detect from state dict keys
                if not self.is_multimodal and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    self.is_multimodal = any('age_processor' in key for key in state_dict.keys())
                
                # Initialize appropriate model architecture
                if self.is_multimodal:
                    logger.info("🔥 Detected multi-modal model (Image + Age)")
                    self.model = MultiModalWeightPredictor(
                        backbone=config.get('backbone', 'resnet50'),
                        dropout=config.get('dropout', 0.3)
                    )
                    
                    # Load age scaler if available
                    if 'age_scaler' in checkpoint:
                        self.age_scaler = checkpoint['age_scaler']
                        logger.info(f"Age scaler loaded: mean={self.age_scaler['mean']:.2f}, std={self.age_scaler['std']:.2f}")
                    else:
                        # Try to load from separate file
                        scaler_path = os.path.join(os.path.dirname(self.model_path), 'age_scaler.json')
                        if os.path.exists(scaler_path):
                            with open(scaler_path, 'r') as f:
                                self.age_scaler = json.load(f)
                            logger.info(f"Age scaler loaded from file: mean={self.age_scaler['mean']:.2f}, std={self.age_scaler['std']:.2f}")
                else:
                    logger.info("📷 Detected image-only model")
                    self.model = WeightPredictor(
                        backbone=config.get('backbone', 'resnet50'),
                        dropout=config.get('dropout', 0.3)
                    )
                
                # Load state dict
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
            else:
                # Old format - assume image-only model
                self.model = checkpoint
                self.is_multimodal = hasattr(self.model, 'age_processor')
                logger.info(f"Loaded legacy model: {'multi-modal' if self.is_multimodal else 'image-only'}")
            
            # Set to evaluation mode
            self.model.eval()
            
            logger.info(f"✅ CNN weight model loaded successfully")
            logger.info(f"   Model type: {'Multi-modal (Image + Age)' if self.is_multimodal else 'Image-only'}")
            self.model_loaded = True
            
        except FileNotFoundError as e:
            logger.error(f"❌ Model file not found: {e}")
            self.model_loaded = False
        except Exception as e:
            logger.error(f"❌ Failed to load CNN model: {e}")
            logger.exception("Full traceback:")
            self.model_loaded = False
        
        # Define image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _scale_age(self, age_months: float) -> float:
        """Scale age using stored scaler parameters"""
        if self.age_scaler is None:
            logger.warning("Age scaler not available, using raw age value")
            return age_months
        
        # Standardize: (x - mean) / std
        scaled_age = (age_months - self.age_scaler['mean']) / self.age_scaler['std']
        return scaled_age
    
    def predict_weight(
        self, 
        image_array, 
        bbox=None, 
        age_months: Optional[float] = None,
        use_age_fallback: bool = True
    ) -> tuple:
        """
        Predict weight from image and optionally age
        
        Args:
            image_array: Input image as numpy array
            bbox: Optional bounding box for ROI cropping
            age_months: Age of hog in months (required for multi-modal models)
            use_age_fallback: If True and age not provided, estimate from typical growth curve
        
        Returns:
            (weight_kg, confidence, metadata)
        """
        if self.model is None or not self.model_loaded:
            logger.warning("Model not loaded, returning default values")
            return 0.0, 0.0, {"error": "model_not_loaded"}
        
        try:
            # Crop if bbox provided
            if bbox:
                x = int(bbox.get('x', 0))
                y = int(bbox.get('y', 0))
                w = int(bbox.get('width', image_array.shape[1]))
                h = int(bbox.get('height', image_array.shape[0]))
                
                # Ensure coordinates are within bounds
                x = max(0, x)
                y = max(0, y)
                w = min(w, image_array.shape[1] - x)
                h = min(h, image_array.shape[0] - y)
                
                image_array = image_array[y:y+h, x:x+w]
            
            # Convert to PIL Image
            img = Image.fromarray(image_array)
            
            # Apply transforms
            img_tensor = self.transform(img).unsqueeze(0)
            
            # Prepare metadata
            metadata = {
                "model_type": "multimodal" if self.is_multimodal else "image_only",
                "age_provided": age_months is not None
            }
            
            # Predict based on model type
            with torch.no_grad():
                if self.is_multimodal:
                    # Multi-modal prediction
                    if age_months is None:
                        if use_age_fallback:
                            # Estimate age from typical market hog (4-6 months, avg 5)
                            age_months = 5.0
                            logger.warning(f"⚠️ Age not provided for multi-modal model, using fallback: {age_months} months")
                            metadata["age_fallback"] = True
                            metadata["age_used"] = age_months
                        else:
                            raise ValueError("Age is required for multi-modal model predictions")
                    
                    # Scale age
                    scaled_age = self._scale_age(age_months)
                    age_tensor = torch.tensor([scaled_age], dtype=torch.float32)
                    
                    # Predict
                    weight = self.model(img_tensor, age_tensor)
                    
                    # Higher confidence when age is provided
                    confidence = 0.92 if not metadata.get("age_fallback", False) else 0.78
                    metadata["age_months"] = age_months
                    metadata["scaled_age"] = float(scaled_age)
                    
                else:
                    # Image-only prediction
                    weight = self.model(img_tensor)
                    confidence = 0.85
                    
                    if age_months is not None:
                        logger.info(f"ℹ️ Age provided ({age_months} months) but model is image-only")
                        metadata["age_ignored"] = age_months
                
                # Handle both scalar and tensor outputs
                if isinstance(weight, torch.Tensor):
                    weight = weight.item()
                
                # Ensure reasonable weight range (10-300 kg for hogs)
                weight = max(10.0, min(300.0, weight))
            
            logger.info(
                f"✅ Predicted weight: {weight:.2f} kg "
                f"(confidence: {confidence:.2f}, model: {metadata['model_type']})"
            )
            
            return float(weight), float(confidence), metadata
            
        except Exception as e:
            logger.error(f"❌ Weight prediction failed: {e}")
            logger.exception("Full traceback:")
            return 0.0, 0.0, {"error": str(e)}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "model_loaded": self.model_loaded,
            "is_multimodal": self.is_multimodal,
            "requires_age": self.is_multimodal,
            "age_scaler_available": self.age_scaler is not None,
            "model_path": self.model_path,
            "capabilities": {
                "image_only": True,
                "image_with_age": self.is_multimodal,
                "age_fallback": True
            }
        }


# Standalone function for backward compatibility
def predict_weight_from_image(
    image_array, 
    model_path: Optional[str] = None,
    bbox: Optional[Dict] = None,
    age_months: Optional[float] = None
) -> tuple:
    """
    Convenience function for weight prediction
    
    Args:
        image_array: Input image
        model_path: Path to model checkpoint
        bbox: Optional bounding box
        age_months: Optional age in months
    
    Returns:
        (weight_kg, confidence, metadata)
    """
    regressor = WeightRegressor(model_path)
    return regressor.predict_weight(image_array, bbox, age_months)