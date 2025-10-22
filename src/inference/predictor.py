# src/inference/predictor.py
"""
Simple inference for trained U-Net model

Usage:
    predictor = DeblurPredictor('checkpoints/best_model.pth')
    deblurred = predictor.predict(blurred_image)
"""

import torch
import numpy as np
from pathlib import Path
import cv2


class DeblurPredictor:
    """
    Load trained U-Net and deblur images
    
    Simple, minimal version for 8-week project
    """
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        """
        Initialize predictor
        
        Args:
            model_path: Path to checkpoint (.pth file)
            device: 'cuda' or 'cpu'
        """
        from src.models.unet import UNet
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = UNet(in_channels=3, out_channels=3)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
            print(f"  Best PSNR: {checkpoint.get('best_psnr', 'unknown'):.2f} dB")
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Model loaded on {self.device}")
    
    @torch.inference_mode()
    def predict(self, blurred_image: np.ndarray) -> np.ndarray:
        """
        Deblur a single image
        
        Args:
            blurred_image: Input blurred image (H, W, 3) RGB, uint8 [0, 255]
        
        Returns:
            Deblurred image (H, W, 3) RGB, uint8 [0, 255]
        """
        # Validate input
        if not isinstance(blurred_image, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(blurred_image)}")
        
        if blurred_image.ndim != 3 or blurred_image.shape[2] != 3:
            raise ValueError(f"Expected (H, W, 3) RGB image, got shape {blurred_image.shape}")
        
        if blurred_image.dtype != np.uint8:
            raise TypeError(f"Expected uint8, got {blurred_image.dtype}")
        
        # Preprocess
        input_tensor = self._preprocess(blurred_image)
        
        # Predict (no need for torch.no_grad(), using @torch.inference_mode())
        output_tensor = self.model(input_tensor)
        
        # Postprocess
        deblurred = self._postprocess(output_tensor)
        
        return deblurred
    
    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Convert image to tensor and pad to divisible by 16
        
        Args:
            image: (H, W, 3) uint8 [0, 255]
        
        Returns:
            tensor: (1, 3, H_padded, W_padded) float32 [0, 1]
        """
        # Store original size for later cropping
        self.original_h, self.original_w = image.shape[:2]
        
        # Calculate padding to make dimensions divisible by 16
        pad_h = (16 - self.original_h % 16) % 16
        pad_w = (16 - self.original_w % 16) % 16
        
        # Apply padding if needed
        if pad_h > 0 or pad_w > 0:
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            print(f"Padded image from ({self.original_h}, {self.original_w}) to {image.shape[:2]}")
        
        # Normalize to [0, 1]
        image_float = image.astype(np.float32) / 255.0
        
        # Convert to tensor (H, W, C) -> (C, H, W)
        tensor = torch.from_numpy(image_float).permute(2, 0, 1)
        
        # Add batch dimension (C, H, W) -> (1, C, H, W)
        tensor = tensor.unsqueeze(0)
        
        # Move to device
        tensor = tensor.to(self.device)
        
        return tensor
    
    def _postprocess(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Convert tensor back to image and remove padding
        
        Args:
            tensor: (1, 3, H_padded, W_padded) float32 [0, 1]
        
        Returns:
            image: (H, W, 3) uint8 [0, 255]
        """
        # Remove batch dimension
        tensor = tensor.squeeze(0)
        
        # Move to CPU
        tensor = tensor.cpu()
        
        # Convert (C, H, W) -> (H, W, C)
        image = tensor.permute(1, 2, 0).numpy()
        
        # Crop back to original size (remove padding)
        image = image[:self.original_h, :self.original_w, :]
        
        # Clip and convert to uint8
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        
        return image
    
    def predict_file(self, input_path: str, output_path: str):
        """
        Deblur image from file and save result
        
        Args:
            input_path: Path to blurred image
            output_path: Path to save deblurred image
        """
        # Load image
        blurred = cv2.imread(input_path)
        if blurred is None:
            raise ValueError(f"Cannot load image: {input_path}")
        
        # Convert BGR to RGB
        blurred_rgb = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
        
        # Predict
        deblurred_rgb = self.predict(blurred_rgb)
        
        # Convert RGB to BGR for saving
        deblurred_bgr = cv2.cvtColor(deblurred_rgb, cv2.COLOR_RGB2BGR)
        
        # Save
        cv2.imwrite(output_path, deblurred_bgr)
        print(f"✓ Saved: {output_path}")


# Test
if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    print("Testing DeblurPredictor...")
    
    # Check if model exists
    model_path = 'checkpoints/best_model.pth'
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        print("Train a model first!")
        sys.exit(1)
    
    # Load predictor
    predictor = DeblurPredictor(model_path)
    
    # Test on random image
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    result = predictor.predict(test_image)
    
    print(f"✓ Input shape: {test_image.shape}")
    print(f"✓ Output shape: {result.shape}")
    print(f"✓ Output range: [{result.min()}, {result.max()}]")
    
    print("\n✅ Predictor works!")
    print("\nUsage:")
    print("  predictor = DeblurPredictor('checkpoints/best_model.pth')")
    print("  deblurred = predictor.predict(blurred_image)")
