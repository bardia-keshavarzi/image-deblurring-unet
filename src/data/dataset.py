# src/data/dataset.py
"""
Simplified PyTorch Dataset for GoPro Dataset
Loads pre-generated blurred/sharp pairs
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np

from .transforms import DeblurTransforms


class DeblurDataset(Dataset):
    """
    Dataset for GoPro pre-generated blur/sharp pairs
    
    Usage:
        dataset = DeblurDataset(
            sharp_paths=sharp_image_paths,
            blurred_paths=blurred_image_paths,
            image_size=256,
            augment=True
        )
    """
    
    def __init__(self,
                 sharp_paths: List[Path],
                 blurred_paths: List[Path],
                 image_size: int = 256,
                 augment: bool = True):
        """
        Initialize dataset
        
        Args:
            sharp_paths: List of paths to sharp images
            blurred_paths: List of paths to blurred images (must match sharp_paths)
            image_size: Target image size
            augment: Whether to apply augmentation
        """
        self.sharp_paths = sharp_paths
        self.blurred_paths = blurred_paths
        self.image_size = image_size
        self.augment = augment
        
        # Validate
        if len(sharp_paths) != len(blurred_paths):
            raise ValueError(
                f"Number of sharp ({len(sharp_paths)}) and "
                f"blurred ({len(blurred_paths)}) images must match"
            )
        
        # Initialize transforms
        self.transforms = DeblurTransforms(image_size=image_size, augment=augment)
        
        print(f"✓ Dataset initialized:")
        print(f"  Images: {len(sharp_paths)}")
        print(f"  Size: {image_size}")
        print(f"  Augment: {augment}")
    
    def __len__(self) -> int:
        return len(self.sharp_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get one sample
        
        Returns:
            Tuple of (blurred_tensor, sharp_tensor)
        """
        # Load images
        sharp_img = self._load_image(self.sharp_paths[idx])
        blurred_img = self._load_image(self.blurred_paths[idx])
        
        # Apply transforms
        transformed = self.transforms(blurred_img, sharp_img)
        
        return transformed['blurred'], transformed['sharp']
    
    @staticmethod
    def _load_image(path: Path) -> np.ndarray:
        """Load image as RGB uint8"""
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Could not load image: {path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img


def create_dataloaders(
    train_sharp_paths: List[Path],
    val_sharp_paths: List[Path],
    train_blurred_paths: List[Path],
    val_blurred_paths: List[Path],
    image_size: int = 256,
    batch_size: int = 8,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders for GoPro
    
    Args:
        train_sharp_paths: Training sharp image paths
        val_sharp_paths: Validation sharp image paths
        train_blurred_paths: Training blurred image paths
        val_blurred_paths: Validation blurred image paths
        image_size: Image size
        batch_size: Batch size
        num_workers: Number of data loading workers
    
    Returns:
        (train_loader, val_loader)
    
    Example:
        >>> train_sharp = sorted(Path('data/gopro/train/sharp').glob('*.png'))
        >>> train_blur = sorted(Path('data/gopro/train/blurred').glob('*.png'))
        >>> val_sharp = sorted(Path('data/gopro/test/sharp').glob('*.png'))
        >>> val_blur = sorted(Path('data/gopro/test/blurred').glob('*.png'))
        >>> train_loader, val_loader = create_dataloaders(
        ...     train_sharp, val_sharp, train_blur, val_blur
        ... )
    """
    # Create datasets
    train_dataset = DeblurDataset(
        sharp_paths=train_sharp_paths,
        blurred_paths=train_blurred_paths,
        image_size=image_size,
        augment=True  # Augmentation for training
    )
    
    val_dataset = DeblurDataset(
        sharp_paths=val_sharp_paths,
        blurred_paths=val_blurred_paths,
        image_size=image_size,
        augment=False  # NO augmentation for validation
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    print(f"\n✓ DataLoaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Batch size: {batch_size}")
    
    return train_loader, val_loader


if __name__ == '__main__':
    """Test dataset loading"""
    print("Testing GoPro dataset loading...")
    
    # Example usage
    gopro_train_sharp = Path('data/gopro/train/sharp')
    gopro_train_blur = Path('data/gopro/train/blurred')
    
    if gopro_train_sharp.exists():
        sharp_paths = sorted(gopro_train_sharp.glob('*.png'))[:10]
        blur_paths = sorted(gopro_train_blur.glob('*.png'))[:10]
        
        dataset = DeblurDataset(sharp_paths, blur_paths, image_size=256)
        
        blurred, sharp = dataset[0]
        print(f"✓ Sample loaded: blurred={blurred.shape}, sharp={sharp.shape}")
    else:
        print("⚠️ GoPro dataset not found. Please organize first.")
