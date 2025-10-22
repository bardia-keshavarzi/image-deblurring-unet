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
