# app/models/cnn_regressor.py
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
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
    def __init__(self, model_path=r'C:\Users\Acer\OneDrive\Desktop\HogIntel-Price&Weight-Estimation\models\outputs\training_run1\best_model.pt'):
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Get config from checkpoint if available
            config = checkpoint.get('config', {})
            backbone = config.get('backbone', 'resnet50')
            dropout = config.get('dropout', 0.3)
            
            # Create model with same architecture
            self.model = WeightPredictor(
                backbone=backbone,
                pretrained=False,
                dropout=dropout
            )
            
            # Load weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            logger.info(f"CNN regressor loaded from {model_path}")
            logger.info(f"Model MAE: {checkpoint.get('val_mae', 'N/A')}, R²: {checkpoint.get('val_r2', 'N/A')}")
            
        except Exception as e:
            logger.error(f"Failed to load CNN regressor: {e}")
            self.model = None
        
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
        if self.model is None:
            return 0.0, 0.0
        
        try:
            # Crop if bbox provided
            if bbox:
                x, y, w, h = int(bbox['x']), int(bbox['y']), int(bbox['width']), int(bbox['height'])
                image_array = image_array[y:y+h, x:x+w]
            
            img = Image.fromarray(image_array)
            img_tensor = self.transform(img).unsqueeze(0)
            
            with torch.no_grad():
                weight = self.model(img_tensor).item()
                confidence = 0.85  # You can add uncertainty estimation
            
            return weight, confidence
        except Exception as e:
            logger.error(f"Weight prediction failed: {e}")
            return 0.0, 0.0