# app/models/cnn_regressor.py
import torch
import torch.nn as nn
import os
from PIL import Image
import torchvision.transforms as transforms
from typing import Optional, Dict, Any, List
from torchvision.models import resnet50
from logger import setup_logger

logger = setup_logger(__name__)

# You need to define the same model architecture
class WeightPredictor(nn.Module):
    """CNN model for hog weight prediction - must match training script"""
    
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
    def __init__(self, model_path: Optional[str] = None):
        # Use environment variable or fallback to container default
        self.model_path = model_path or os.getenv(
            'CNN_MODEL_PATH', 
            '/models/weight_model/best_model.pt'
        )
        
        self.model = None
        self.model_loaded = False
        
        try:
            # ✅ FIX: Initialize the model architecture first
            self.model = WeightPredictor(backbone='resnet50', dropout=0.3)
            
            # ✅ FIX: Load the state dictionary instead of the whole object
            logger.info(f"Loading CNN model from: {self.model_path}")
            
            # Check if file exists
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=torch.device('cpu'))
            
            # Handle different save formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    # Assume the dict itself is the state_dict
                    self.model.load_state_dict(checkpoint)
            else:
                # If it's the model object itself (old format)
                self.model = checkpoint
            
            # Set to evaluation mode
            self.model.eval()
            
            logger.info(f"✅ CNN weight model loaded successfully from {self.model_path}")
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
    
    def predict_weight(self, image_array, bbox=None):
        """
        Predict weight from image or cropped ROI
        Returns: weight in kg, confidence score
        """
        if self.model is None or not self.model_loaded:
            logger.warning("Model not loaded, returning default values")
            return 0.0, 0.0
        
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
            
            # Predict
            with torch.no_grad():
                weight = self.model(img_tensor)
                
                # Handle both scalar and tensor outputs
                if isinstance(weight, torch.Tensor):
                    weight = weight.item()
                
                confidence = 0.85  # You can add uncertainty estimation later
            
            logger.info(f"Predicted weight: {weight:.2f} kg (confidence: {confidence:.2f})")
            return float(weight), float(confidence)
            
        except Exception as e:
            logger.error(f"Weight prediction failed: {e}")
            logger.exception("Full traceback:")
            return 0.0, 0.0