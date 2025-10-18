# src/data/transforms.py
"""
Data Transforms and Augmentation

WHAT THIS DOES:
- Defines augmentation pipeline for training
- Applies random transformations to images
- Normalizes images for neural networks

WHY WE NEED THIS:
- Data augmentation prevents overfitting
- Makes model robust to variations
- Standard deep learning practice
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from typing import Dict, Any


class DeblurTransforms:
    """
    Transformation pipeline for deblurring dataset
    
    USAGE:
        transforms = DeblurTransforms(image_size=256, augment=True)
        transformed = transforms(blurred=blurred_img, sharp=sharp_img)
        blurred_tensor = transformed['blurred']
        sharp_tensor = transformed['sharp']
    """
    
    def __init__(self, image_size: int = 256, augment: bool = True):
        """
        Initialize transforms
        
        Args:
            image_size: Target image size (will crop/resize to this)
            augment: Whether to apply augmentation (True for train, False for val)
        """
        self.image_size = image_size
        self.augment = augment
        
        # Build transformation pipeline
        if augment:
            self.transform = self._build_train_transforms()
        else:
            self.transform = self._build_val_transforms()
        
        print(f"✓ Transforms initialized: size={image_size}, augment={augment}")
    
    def _build_train_transforms(self) -> A.Compose:
        """
        Build training augmentation pipeline
        
        WHAT EACH TRANSFORM DOES:
        - RandomCrop: Take random patch from image
        - HorizontalFlip: Flip left-right (50% chance)
        - VerticalFlip: Flip up-down (30% chance)
        - RandomRotate90: Rotate 0/90/180/270 degrees
        - RandomBrightnessContrast: Vary lighting
        - Normalize: Scale to [-1, 1] or [0, 1]
        - ToTensorV2: Convert to PyTorch tensor
        """
        return A.Compose([
            # Spatial transforms (apply same to both blurred and sharp)
            A.RandomCrop(self.image_size, self.image_size, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            
            # Color transforms (subtle, to handle lighting variations)
            A.RandomBrightnessContrast(
                brightness_limit=0.2,  # ±20% brightness
                contrast_limit=0.2,     # ±20% contrast
                p=0.3                   # 30% chance to apply
            ),
            
            # Normalization (IMPORTANT for neural networks)
            A.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),  
                max_pixel_value=255.0
            ),
            
            # Convert to PyTorch tensor (H,W,C) -> (C,H,W)
            ToTensorV2()
        ], 
        # CRITICAL: Apply same transform to both images
        additional_targets={'sharp': 'image'}
        )
    
    def _build_val_transforms(self) -> A.Compose:
        """
        Build validation transforms (NO augmentation)
        
        For validation/testing, we want consistent results:
        - Just resize to target size
        - Normalize
        - Convert to tensor
        """
        return A.Compose([
            # Resize instead of random crop (deterministic)
            A.Resize(self.image_size, self.image_size),
            
            # Normalize (same as training)
            A.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
                max_pixel_value=255.0
            ),
            
            # Convert to tensor
            ToTensorV2()
        ],
        additional_targets={'sharp': 'image'}
        )
    
    def __call__(self, blurred: np.ndarray, sharp: np.ndarray) -> Dict[str, Any]:
        """
        Apply transforms to image pair
        
        Args:
            blurred: Blurred image (H, W, 3) uint8
            sharp: Sharp image (H, W, 3) uint8
        
        Returns:
            Dictionary with 'blurred' and 'sharp' tensors
        
        Example:
            >>> transform = DeblurTransforms(256, augment=True)
            >>> result = transform(blurred_img, sharp_img)
            >>> blurred_tensor = result['blurred']  # (3, 256, 256)
            >>> sharp_tensor = result['sharp']      # (3, 256, 256)
        """
        # Apply transforms
        # albumentations automatically applies to both images
        transformed = self.transform(image=blurred, sharp=sharp)
        
        return {
            'blurred': transformed['image'],
            'sharp': transformed['sharp']
        }
