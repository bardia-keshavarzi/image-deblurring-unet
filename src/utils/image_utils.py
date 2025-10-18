# src/utils/image_utils.py
import cv2
import numpy as np
from pathlib import Path

def load_image(path, color_mode='BGR'):
    """
    Load image from file
    
    Args:
        path: Path to image file
        color_mode: 'BGR' (OpenCV default), 'RGB', or 'GRAY'
    
    Returns:
        Image as numpy array
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    
    # Load image
    if color_mode == 'GRAY':
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError(f"Could not load image: {path}")
    
    # OpenCV loads as BGR, convert if needed
    if color_mode == 'RGB' and len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image

def save_image(image, path, color_mode='BGR'):
    """
    Save image to file
    
    Args:
        image: Image array
        path: Output path
        color_mode: Color mode of INPUT image
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)  # Create folder if needed
    
    # Convert RGB to BGR for saving
    save_img = image.copy()
    if color_mode == 'RGB' and len(image.shape) == 3:
        save_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    success = cv2.imwrite(str(path), save_img)
    if not success:
        raise ValueError(f"Could not save image to {path}")

# Add to image_utils.py

def normalize_image(image):
    """
    Normalize image to [0, 1] range
    
    Args:
        image: uint8 image
    
    Returns:
        float32 image in [0, 1]
    """
    return image.astype(np.float32) / 255.0

def denormalize_image(image):
    """
    Convert normalized image back to uint8
    
    Args:
        image: float32 image in [0, 1]
    
    Returns:
        uint8 image in [0, 255]
    """
    return (image * 255).clip(0, 255).astype(np.uint8)

# Add to image_utils.py
def bgr_to_rgb(image):
    """Convert BGR to RGB"""
    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def rgb_to_bgr(image):
    """Convert RGB to BGR"""
    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

def resize_image(image, size, interpolation='lanczos'):
    """
    Resize image
    
    Args:
        image: Input image
        size: (width, height)
        interpolation: 'nearest', 'linear', 'cubic', 'lanczos'
    """
    methods = {
        'nearest': cv2.INTER_NEAREST,
        'linear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC,
        'lanczos': cv2.INTER_LANCZOS4
    }
    method = methods.get(interpolation, cv2.INTER_LANCZOS4)
    return cv2.resize(image, size, interpolation=method)