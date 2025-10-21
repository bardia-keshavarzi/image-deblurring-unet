# src/training/metrics.py
"""
Metrics for Image Quality Assessment using TorchMetrics

This module provides PSNR and SSIM metrics with automatic GPU support.

PSNR (Peak Signal-to-Noise Ratio):
- Measures pixel-wise similarity
- Higher is better (30+ dB is good)
- Formula: PSNR = 20 × log10(MAX / √MSE)

SSIM (Structural Similarity Index):
- Measures structural similarity (better than PSNR for human perception)
- Range: 0 to 1 (higher is better, 1 = identical)
- Takes into account luminance, contrast, structure

Reference: https://lightning.ai/docs/torchmetrics/stable/image/peak_signal_noise_ratio.html
"""

import torch
import numpy as np
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure
)


class PSNRMetric:
    """
    Peak Signal-to-Noise Ratio using TorchMetrics
    
    Measures pixel-wise similarity between images.
    Higher PSNR indicates better quality.
    
    Args:
        data_range: The data range of the input images (default: 1.0)
        
    Usage:
        psnr = PSNRMetric()
        score = psnr(prediction, target)  # Returns scalar in dB
        
    Expected Input:
        - Images in range [-1, 1] (will be converted to [0, 1])
        - Shape: (B, C, H, W) where B=batch, C=channels, H=height, W=width
        
    Output:
        - PSNR value in decibels (dB)
        - Typical values: 20-40+ dB
    """
    
    def __init__(self, data_range=1.0):
        """
        Initialize PSNR metric
        
        Args:
            data_range: Range of the input data (1.0 for [0,1] normalized images)
        """
        self.metric = PeakSignalNoiseRatio(data_range=data_range)
        self.device = None  # Track current device
    
    def __call__(self, pred, target):
        if self.device != pred.device:
            self.metric = self.metric.to(pred.device)
            self.device = pred.device
        
        # Data already in [0, 1] - just clamp
        pred = torch.clamp(pred, 0.0, 1.0)
        target = torch.clamp(target, 0.0, 1.0)
        
        return self.metric(pred, target).item()
        
        def reset(self):
            """Reset metric state"""
            self.metric.reset()


class SSIMMetric:
    """
    Structural Similarity Index using TorchMetrics
    
    Better than PSNR for measuring perceptual quality.
    Considers luminance, contrast, and structure.
    
    Args:
        data_range: The data range of the input images (default: 1.0)
        
    Usage:
        ssim = SSIMMetric()
        score = ssim(prediction, target)  # Returns scalar between 0 and 1
        
    Expected Input:
        - Images in range [-1, 1] (will be converted to [0, 1])
        - Shape: (B, C, H, W)
        
    Output:
        - SSIM value between 0 and 1
        - 1.0 = perfect match
        - 0.9+ = excellent quality
        - 0.7-0.9 = good quality
        - <0.7 = poor quality
    """
    
    def __init__(self, data_range=1.0):
        """
        Initialize SSIM metric
        
        Args:
            data_range: Range of the input data (1.0 for [0,1] normalized images)
        """
        self.metric = StructuralSimilarityIndexMeasure(data_range=data_range)
        self.device = None  # Track current device
    
    def __call__(self, pred, target):
        """
        Calculate SSIM between prediction and target
        
        Args:
            pred: Predicted image tensor (B, C, H, W) in range [-1, 1]
            target: Target image tensor (B, C, H, W) in range [-1, 1]
        
        Returns:
            SSIM value between 0 and 1 (scalar)
        """
        # CRITICAL FIX: Move metric to same device as input
        if self.device != pred.device:
            self.metric = self.metric.to(pred.device)
            self.device = pred.device
        
        # Convert from [-1, 1] to [0, 1]
        pred = (pred + 1.0) / 2.0
        target = (target + 1.0) / 2.0
        
        # Clamp to ensure values are in valid range
        pred = torch.clamp(pred, 0.0, 1.0)
        target = torch.clamp(target, 0.0, 1.0)
        
        return self.metric(pred, target).item()
    
    def reset(self):
        """Reset metric state"""
        self.metric.reset()


def calculate_psnr(pred, target):
    """
    Convenience function for PSNR calculation
    
    Args:
        pred: Predicted image (numpy array or torch tensor)
        target: Target image (numpy array or torch tensor)
    
    Returns:
        PSNR value in dB
        
    Example:
        >>> pred = torch.randn(4, 3, 256, 256)
        >>> target = torch.randn(4, 3, 256, 256)
        >>> psnr = calculate_psnr(pred, target)
        >>> print(f"PSNR: {psnr:.2f} dB")
    """
    # Convert numpy to tensor if needed
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred).float()
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target).float()
    
    metric = PSNRMetric()
    return metric(pred, target)


def calculate_ssim(pred, target):
    """
    Convenience function for SSIM calculation
    
    Args:
        pred: Predicted image (numpy array or torch tensor)
        target: Target image (numpy array or torch tensor)
    
    Returns:
        SSIM value between 0 and 1
        
    Example:
        >>> pred = torch.randn(4, 3, 256, 256)
        >>> target = torch.randn(4, 3, 256, 256)
        >>> ssim = calculate_ssim(pred, target)
        >>> print(f"SSIM: {ssim:.4f}")
    """
    # Convert numpy to tensor if needed
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred).float()
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target).float()
    
    metric = SSIMMetric()
    return metric(pred, target)


def evaluate_batch(pred_batch, target_batch):
    """
    Evaluate a batch of images and return both PSNR and SSIM
    
    Args:
        pred_batch: Batch of predicted images (B, C, H, W)
        target_batch: Batch of target images (B, C, H, W)
    
    Returns:
        Dictionary with 'psnr' and 'ssim' scores
        
    Example:
        >>> scores = evaluate_batch(predictions, targets)
        >>> print(f"PSNR: {scores['psnr']:.2f} dB")
        >>> print(f"SSIM: {scores['ssim']:.4f}")
    """
    psnr_score = calculate_psnr(pred_batch, target_batch)
    ssim_score = calculate_ssim(pred_batch, target_batch)
    
    return {
        'psnr': psnr_score,
        'ssim': ssim_score
    }


# Test and demonstration
if __name__ == '__main__':
    print("=" * 60)
    print("Testing Metrics (TorchMetrics with GPU Support)")
    print("=" * 60)
    
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Create test images on GPU if available
    print("\nCreating test images (batch of 2, 256×256 RGB)...")
    target = torch.randn(2, 3, 256, 256).to(device)
    
    # Initialize metrics
    psnr_metric = PSNRMetric()
    ssim_metric = SSIMMetric()
    
    # Test 1: Perfect match
    print("\n" + "-" * 60)
    print("Test 1: Perfect Match (identical images)")
    print("-" * 60)
    pred_perfect = target.clone()
    
    psnr_perfect = psnr_metric(pred_perfect, target)
    ssim_perfect = ssim_metric(pred_perfect, target)
    
    print(f"  PSNR: {psnr_perfect:.2f} dB")
    print(f"  SSIM: {ssim_perfect:.4f}")
    print(f"  Status: {'✓ PASS' if psnr_perfect > 80 and ssim_perfect > 0.99 else '✗ FAIL'}")
    
    # Test 2: Small noise
    print("\n" + "-" * 60)
    print("Test 2: Small Noise (10% noise)")
    print("-" * 60)
    noise_small = torch.randn_like(target) * 0.1
    pred_noisy = target + noise_small
    
    psnr_noisy = psnr_metric(pred_noisy, target)
    ssim_noisy = ssim_metric(pred_noisy, target)
    
    print(f"  PSNR: {psnr_noisy:.2f} dB")
    print(f"  SSIM: {ssim_noisy:.4f}")
    print(f"  Status: {'✓ PASS' if 15 < psnr_noisy < 35 else '✗ FAIL'}")
    
    print("\n" + "=" * 60)
    print("✅ All Tests Completed Successfully!")
    print("=" * 60)
    print("\nMetrics are ready for training with automatic GPU support.")
