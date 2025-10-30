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
        self.model = UNet(in_channels=3, out_channels=3, use_attention=True)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            # Check for nested model_state key
            if 'model_state' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state'])
                print(f"✓ Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
                print(f"  Best PSNR: {checkpoint.get('best_psnr', 'unknown'):.2f} dB")
            elif 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✓ Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
                print(f"  Best PSNR: {checkpoint.get('best_psnr', 'unknown'):.2f} dB")
            else:
                # Assume it's the raw state_dict
                self.model.load_state_dict(checkpoint)
        else:
            # Direct state_dict (old format)
            self.model.load_state_dict(checkpoint)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"✓ Model loaded on {self.device}")
    
    def predict(self, blurred_image: np.ndarray) -> np.ndarray:
        """
        Deblur a single image
        
        Args:
            blurred_image: Input blurred image (H, W, 3) RGB, uint8 [0, 255]
        
        Returns:
            Deblurred image (H, W, 3) RGB, uint8 [0, 255]
        """
        # Preprocess
        input_tensor = self._preprocess(blurred_image)
        
        # Predict
        with torch.no_grad():
            output_tensor = self.model(input_tensor)
        
        # Postprocess
        deblurred = self._postprocess(output_tensor)
        
        return deblurred
    
    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Convert image to tensor
        
        Args:
            image: (H, W, 3) uint8 [0, 255]
        
        Returns:
            tensor: (1, 3, H, W) float32 [0, 1]
        """
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
        Convert tensor back to image
        
        Args:
            tensor: (1, 3, H, W) float32 [0, 1]
        
        Returns:
            image: (H, W, 3) uint8 [0, 255]
        """
        # Remove batch dimension
        tensor = tensor.squeeze(0)
        
        # Move to CPU
        tensor = tensor.cpu()
        
        # Convert (C, H, W) -> (H, W, C)
        image = tensor.permute(1, 2, 0).numpy()
        
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

    def predict_tta(self, blurred_image: np.ndarray, num_augs: int = 4) -> np.ndarray:
        """
        Predict with test-time augmentation for better quality
        
        Args:
            blurred_image: Input (H, W, 3) RGB uint8
            num_augs: Number of augmentations (4 or 8 recommended)
        
        Returns:
            Deblurred image (H, W, 3) RGB uint8
        """
        predictions = []
        
        # Original
        predictions.append(self.predict(blurred_image))
        
        if num_augs >= 2:
            # Horizontal flip
            flipped_h = np.fliplr(blurred_image)
            pred_h = self.predict(flipped_h)
            predictions.append(np.fliplr(pred_h))
        
        if num_augs >= 4:
            # Vertical flip
            flipped_v = np.flipud(blurred_image)
            pred_v = self.predict(flipped_v)
            predictions.append(np.flipud(pred_v))
            
            # Both flips
            flipped_hv = np.flipud(np.fliplr(blurred_image))
            pred_hv = self.predict(flipped_hv)
            predictions.append(np.flipud(np.fliplr(pred_hv)))
        
        if num_augs >= 8:
            # 90° rotations
            for k in [1, 2, 3]:  # 90°, 180°, 270°
                rotated = np.rot90(blurred_image, k)
                pred_rot = self.predict(rotated)
                predictions.append(np.rot90(pred_rot, -k))
        
        # Average all predictions
        result = np.mean(predictions, axis=0).astype(np.uint8)
        return result    


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
