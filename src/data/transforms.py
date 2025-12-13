# src/data/transforms.py
"""Data Transforms - FIXED VERSION"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from typing import Dict, Any


class DeblurTransforms:
    """Transformation pipeline for deblurring dataset"""
    
    def __init__(self, image_size: int = 256, augment: bool = True):
        self.image_size = image_size
        self.augment = augment
        
        if augment:
            self.transform = self._build_train_transforms()
        else:
            self.transform = self._build_val_transforms()
        
        print(f"✓ Transforms initialized: size={image_size}, augment={augment}")
    
    def _build_train_transforms(self) -> A.Compose:
        """Build training augmentation pipeline"""
        return A.Compose([
            # Spatial transforms
            A.RandomCrop(self.image_size, self.image_size, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            
           
            A.Normalize(
                mean=(0.0, 0.0, 0.0),  # Images will be in [0, 1] range
                std=(1.0, 1.0, 1.0),
                max_pixel_value=255.0
            ),
            
            ToTensorV2() # numpy array to pytorch tensor hwc to chw
        ], 
        additional_targets={'sharp': 'image'}
        )
    
    def _build_val_transforms(self) -> A.Compose:

        return A.Compose([
            A.Resize(self.image_size, self.image_size),
            

            A.Normalize(
                mean=(0.0, 0.0, 0.0),  
                std=(1.0, 1.0, 1.0),    
                max_pixel_value=255.0
            ),
            
            ToTensorV2()
        ],
        additional_targets={'sharp': 'image'}
        )
    
    def __call__(self, blurred: np.ndarray, sharp: np.ndarray) -> Dict[str, Any]:
        """Apply transforms to image pair"""
        transformed = self.transform(image=blurred, sharp=sharp)
        
        return {
            'blurred': transformed['image'], #tensor
            'sharp': transformed['sharp']
        }
