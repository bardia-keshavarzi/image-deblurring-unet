
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np
from .transforms import DeblurTransforms


class DeblurDataset(Dataset):
    def __init__(
        self,
        sharp_paths: List[Path],
        blurred_paths: List[Path],
        image_size: int = 256,
        augment: bool = True,
    ):

        sharp_files = {p.name: p for p in sharp_paths}
        blur_files = {p.name: p for p in blurred_paths}
        common = sorted(list(sharp_files.keys() & blur_files.keys()))

        self.sharp_paths = [sharp_files[name] for name in common]
        self.blurred_paths = [blur_files[name] for name in common]

        if len(self.sharp_paths) == 0:
            raise ValueError("No matched sharp/blurred pairs found!")

        self.image_size = image_size
        self.augment = augment
        self.transforms = DeblurTransforms(image_size=image_size, augment=augment)

        print(f"✓ Dataset initialized:")
        print(f"  Total pairs: {len(self.sharp_paths)}")
        print(f"  Image size:  {image_size}")
        print(f"  Augment:     {augment}")

    def __len__(self):   
        return len(self.sharp_paths)

    def __getitem__(self, idx): 
        sharp_img = self._load_image(self.sharp_paths[idx])
        blurred_img = self._load_image(self.blurred_paths[idx])
        transformed = self.transforms(blurred_img, sharp_img)
        return transformed["blurred"], transformed["sharp"]

    @staticmethod
    def _load_image(path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not load image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img


def _load_image_paths(folder: str) -> List[Path]:
    exts = ["*.png", "*.jpg", "*.jpeg"]
    files = []
    for ext in exts:
        files += list(Path(folder).glob(ext))
    files = sorted([p for p in files if p.is_file()])
    return files


def create_dataloaders(
    train_sharp_dir: str,
    train_blur_dir: str,
    val_sharp_dir: str,
    val_blur_dir: str,
    image_size: int = 256,
    batch_size: int = 8,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:

    train_sharp = _load_image_paths(train_sharp_dir)
    train_blur = _load_image_paths(train_blur_dir)
    val_sharp = _load_image_paths(val_sharp_dir)
    val_blur = _load_image_paths(val_blur_dir)

    train_dataset = DeblurDataset(train_sharp, train_blur, image_size=image_size, augment=True)
    val_dataset = DeblurDataset(val_sharp, val_blur, image_size=image_size, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print(f"\n✓ DataLoaders created:")
    print(f"  Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    print(f"  Batch size: {batch_size}")
    return train_loader, val_loader

############################## test ########################################
if __name__ == "__main__":
    print("Testing GoPro dataset loading...")

    gopro_train_sharp = "data/gopro/train/sharp"
    gopro_train_blur = "data/gopro/train/blurred"

    train_loader, _ = create_dataloaders(gopro_train_sharp, gopro_train_blur, gopro_train_sharp, gopro_train_blur)
    blurred, sharp = next(iter(train_loader))
    print(f"Batch loaded successfully: {blurred.shape}, {sharp.shape}")
