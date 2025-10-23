# src/data/transforms.py
"""Data Transforms - Updated to ensure identical spatial sizes and tanh scaling"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from typing import Dict, Any


class DeblurTransforms:
    """Transformation pipeline for deblurring dataset"""
    
    def __init__(self, image_size: int = 256, augment: bool = True, tanh: bool = True):
        self.image_size = image_size
        self.augment = augment
        self.tanh = tanh
        
        if augment:
            self.transform = self._build_train_transforms()
        else:
            self.transform = self._build_val_transforms()
        
        print(f"✓ Transforms initialized: size={image_size}, augment={augment}, tanh={tanh}")
    
    def _common_norm(self):
        # Normalize to [0,1]
        ops = [A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0)]
        # Optional: map to [-1,1] expected by tanh
        if self.tanh:
            ops.append(A.Lambda(image=lambda x: x * 2.0 - 1.0, always_apply=True))
        ops.append(ToTensorV2())
        return ops
    
    def _build_train_transforms(self) -> A.Compose:
        """Build training augmentation pipeline ensuring same crop for both images"""
        return A.Compose([
            A.RandomCrop(self.image_size, self.image_size, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            *self._common_norm(),
        ], additional_targets={'sharp': 'image'})
    
    def _build_val_transforms(self) -> A.Compose:
        """Build validation transforms with identical resizing and normalization"""
        return A.Compose([
            A.Resize(self.image_size, self.image_size),
            *self._common_norm(),
        ], additional_targets={'sharp': 'image'})
    
    def __call__(self, blurred: np.ndarray, sharp: np.ndarray) -> Dict[str, Any]:
        transformed = self.transform(image=blurred, sharp=sharp)
        return {
            'blurred': transformed['image'],
            'sharp': transformed['sharp']
        }
