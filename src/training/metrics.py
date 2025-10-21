# src/training/metrics.py
"""
Metrics for Image Quality Assessment using TorchMetrics
FIXED VERSION - Correct indentation
"""

import torch
import numpy as np
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure
)


class PSNRMetric:
    """Peak Signal-to-Noise Ratio using TorchMetrics"""
    
    def __init__(self, data_range=1.0):
        """Initialize PSNR metric"""
        self.metric = PeakSignalNoiseRatio(data_range=1.0)
        self.device = None
    
    def __call__(self, pred, target):
        """Calculate PSNR"""
        # Move metric to device
        if self.device != pred.device:
            self.metric = self.metric.to(pred.device)
            self.device = pred.device
        
        # Clamp to [0, 1]
        pred = torch.clamp(pred, 0.0, 1.0)
        target = torch.clamp(target, 0.0, 1.0)
        
        return self.metric(pred, target).item()
    
    def reset(self):  # ✅ CORRECT INDENTATION - at class level!
        """Reset metric state"""
        self.metric.reset()


class SSIMMetric:
    """Structural Similarity Index using TorchMetrics"""
    
    def __init__(self, data_range=1.0):
        """Initialize SSIM metric"""
        self.metric = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.device = None
    
    def __call__(self, pred, target):
        """Calculate SSIM"""
        # Move metric to device
        if self.device != pred.device:
            self.metric = self.metric.to(pred.device)
            self.device = pred.device
        
        # Clamp to [0, 1]
        pred = torch.clamp(pred, 0.0, 1.0)
        target = torch.clamp(target, 0.0, 1.0)
        
        return self.metric(pred, target).item()
    
    def reset(self):  
        """Reset metric state"""
        self.metric.reset()


def calculate_psnr(pred, target):
    """Convenience function for PSNR"""
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred).float()
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target).float()
    
    metric = PSNRMetric()
    return metric(pred, target)


def calculate_ssim(pred, target):
    """Convenience function for SSIM"""
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred).float()
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target).float()
    
    metric = SSIMMetric()
    return metric(pred, target)


def evaluate_batch(pred_batch, target_batch):
    """Evaluate a batch"""
    psnr_score = calculate_psnr(pred_batch, target_batch)
    ssim_score = calculate_ssim(pred_batch, target_batch)
    
    return {
        'psnr': psnr_score,
        'ssim': ssim_score
    }


# Test
if __name__ == '__main__':
    print("=" * 60)
    print("Testing Metrics - FIXED VERSION")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Create test images in [0, 1] range
    print("\nCreating test images in [0, 1] range...")
    target = torch.rand(2, 3, 256, 256).to(device)
    
    psnr_metric = PSNRMetric()
    ssim_metric = SSIMMetric()
    
    # Test 1: Perfect match
    print("\n" + "-" * 60)
    print("Test 1: Perfect Match")
    pred_perfect = target.clone()
    
    psnr = psnr_metric(pred_perfect, target)
    ssim = ssim_metric(pred_perfect, target)
    
    print(f"  PSNR: {psnr:.2f} dB (should be > 40)")
    print(f"  SSIM: {ssim:.4f} (should be ~1.0)")
    print(f"  Status: {'✅ PASS' if psnr > 40 and ssim > 0.99 else '❌ FAIL'}")
    
    # Test 2: Small noise
    print("\n" + "-" * 60)
    print("Test 2: Small Noise (5% noise)")
    noise = torch.randn_like(target) * 0.05
    pred_noisy = torch.clamp(target + noise, 0, 1)
    
    psnr = psnr_metric(pred_noisy, target)
    ssim = ssim_metric(pred_noisy, target)
    
    print(f"  PSNR: {psnr:.2f} dB (should be 20-35)")
    print(f"  SSIM: {ssim:.4f} (should be 0.7-0.9)")
    print(f"  Status: {'✅ PASS' if 20 < psnr < 35 and 0.7 < ssim < 0.95 else '❌ FAIL'}")
    
    print("\n" + "=" * 60)
    print("✅ Metrics Test Complete")
    print("=" * 60)
