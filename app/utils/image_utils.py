"""
HogIntel Image Utilities
Image preprocessing and transformation functions
"""
import cv2
import numpy as np
import base64
from PIL import Image
from io import BytesIO
from typing import Tuple, List, Optional

def preprocess_image(image_data: bytes) -> np.ndarray:
    """
    Main preprocessing function for incoming image data
    
    Args:
        image_data: Raw image bytes from request
        
    Returns:
        Preprocessed OpenCV image (BGR format)
    """
    try:
        # Convert bytes to base64 string for decoding
        base64_string = base64.b64encode(image_data).decode('utf-8')
        
        # Decode base64 to OpenCV image
        image = decode_base64_image(base64_string)
        
        # Resize if too large
        image = resize_image(image, max_size=1024)
        
        # Enhance image quality
        image = enhance_image(image)
        
        return image
        
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {str(e)}")

def extract_roi(image: np.ndarray, bbox: List[float]) -> np.ndarray:
    """
    Extract region of interest - alias for crop_roi for compatibility
    
    Args:
        image: Full image
        bbox: Bounding box [x1, y1, x2, y2]
        
    Returns:
        Cropped ROI image
    """
    return crop_roi(image, bbox, padding=20)

def decode_base64_image(base64_string: str) -> np.ndarray:
    """
    Decode base64 string to OpenCV image
    
    Args:
        base64_string: Base64 encoded image string
        
    Returns:
        numpy array (BGR format)
    """
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        img_data = base64.b64decode(base64_string)
        
        # Convert to PIL Image
        pil_image = Image.open(BytesIO(img_data))
        
        # Convert to numpy array
        img_array = np.array(pil_image)
        
        # Convert RGB to BGR (OpenCV format)
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        return img_array
        
    except Exception as e:
        raise ValueError(f"Failed to decode image: {str(e)}")

def encode_image_base64(image: np.ndarray) -> str:
    """
    Encode OpenCV image to base64 string
    
    Args:
        image: OpenCV image (BGR format)
        
    Returns:
        Base64 encoded string
    """
    try:
        # Convert BGR to RGB
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # Encode to bytes
        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG", quality=85)
        
        # Encode to base64
        img_bytes = buffer.getvalue()
        base64_string = base64.b64encode(img_bytes).decode('utf-8')
        
        return f"data:image/jpeg;base64,{base64_string}"
        
    except Exception as e:
        raise ValueError(f"Failed to encode image: {str(e)}")

def crop_roi(image: np.ndarray, bbox: List[float], padding: int = 20) -> np.ndarray:
    """
    Crop region of interest from image with padding
    
    Args:
        image: Full image
        bbox: Bounding box [x1, y1, x2, y2]
        padding: Padding in pixels
        
    Returns:
        Cropped image
    """
    h, w = image.shape[:2]
    
    # Extract coordinates
    x1, y1, x2, y2 = map(int, bbox)
    
    # Add padding
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    
    # Ensure valid dimensions
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bounding box after padding: {[x1, y1, x2, y2]}")
    
    # Crop
    cropped = image[y1:y2, x1:x2]
    
    return cropped

def resize_image(image: np.ndarray, max_size: int = 1024) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio
    
    Args:
        image: Input image
        max_size: Maximum dimension size
        
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    
    if max(h, w) <= max_size:
        return image
    
    # Calculate new dimensions
    if h > w:
        new_h = max_size
        new_w = int(w * (max_size / h))
    else:
        new_w = max_size
        new_h = int(h * (max_size / w))
    
    # Resize
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return resized

def preprocess_for_cnn(image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Preprocess image for CNN input
    
    Args:
        image: Input image
        target_size: Target dimensions (width, height)
        
    Returns:
        Preprocessed image tensor
    """
    # Resize
    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    # Normalize to [0, 1]
    normalized = resized.astype(np.float32) / 255.0
    
    # Convert to tensor format (CHW)
    if len(normalized.shape) == 3:
        tensor = np.transpose(normalized, (2, 0, 1))
    else:
        tensor = np.expand_dims(normalized, axis=0)
    
    # Add batch dimension
    tensor = np.expand_dims(tensor, axis=0)
    
    return tensor

def enhance_image(image: np.ndarray) -> np.ndarray:
    """
    Enhance image quality (contrast, brightness)
    
    Args:
        image: Input image
        
    Returns:
        Enhanced image
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Split channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Merge channels
    enhanced_lab = cv2.merge([l, a, b])
    
    # Convert back to BGR
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    return enhanced

def draw_bbox(image: np.ndarray, bbox: List[float], label: str = "", confidence: float = 0.0) -> np.ndarray:
    """
    Draw bounding box on image
    
    Args:
        image: Input image
        bbox: Bounding box [x1, y1, x2, y2]
        label: Label text
        confidence: Detection confidence
        
    Returns:
        Image with bounding box
    """
    img_copy = image.copy()
    
    x1, y1, x2, y2 = map(int, bbox)
    
    # Draw rectangle
    color = (0, 255, 0)  # Green
    thickness = 2
    cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)
    
    # Draw label
    if label:
        text = f"{label} {confidence:.2f}" if confidence > 0 else label
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        
        # Get text size
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        
        # Draw background rectangle
        cv2.rectangle(img_copy, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
        
        # Draw text
        cv2.putText(img_copy, text, (x1, y1 - 5), font, font_scale, (0, 0, 0), font_thickness)
    
    return img_copy

def validate_image_size(image: np.ndarray, min_size: int = 100, max_size: int = 4096) -> bool:
    """
    Validate image dimensions
    
    Args:
        image: Input image
        min_size: Minimum dimension size
        max_size: Maximum dimension size
        
    Returns:
        Boolean indicating if image size is valid
    """
    h, w = image.shape[:2]
    return min_size <= h <= max_size and min_size <= w <= max_size

def get_image_info(image: np.ndarray) -> dict:
    """
    Get image information
    
    Args:
        image: Input image
        
    Returns:
        Dictionary with image information
    """
    h, w = image.shape[:2]
    channels = 1 if len(image.shape) == 2 else image.shape[2]
    dtype = str(image.dtype)
    
    return {
        "height": h,
        "width": w,
        "channels": channels,
        "dtype": dtype,
        "size_kb": (image.nbytes / 1024)
    }