# src/training/metrics.py
import torch
import numpy as np
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure
)


class PSNRMetric:    
    def __init__(self, data_range=1.0):
        """
        Initialize PSNR metric
        
        Args:
            data_range: Range of the input data (1.0 for [0,1] normalized images)
        """
        self.metric = PeakSignalNoiseRatio(data_range=data_range)
    
    def __call__(self, pred, target):
        """
        Calculate PSNR between prediction and target
        
        Args:
            pred: Predicted image tensor (B, C, H, W) in range [-1, 1]
            target: Target image tensor (B, C, H, W) in range [-1, 1]
        
        Returns:
            PSNR value in dB (scalar)
        """
        # Convert from [-1, 1] to [0, 1] for TorchMetrics
        pred = (pred + 1.0) / 2.0
        target = (target + 1.0) / 2.0
        
        # Clamp to ensure values are in valid range
        pred = torch.clamp(pred, 0.0, 1.0)
        target = torch.clamp(target, 0.0, 1.0)
        
        return self.metric(pred, target).item()
    
    def reset(self):
        """Reset metric state"""
        self.metric.reset()


class SSIMMetric:  
    def __init__(self, data_range=1.0):
        self.metric = StructuralSimilarityIndexMeasure(data_range=data_range)
    
    def __call__(self, pred, target):

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
    # Convert numpy to tensor if needed
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred).float()
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target).float()
    
    metric = PSNRMetric()
    return metric(pred, target)


def calculate_ssim(pred, target):
    # Convert numpy to tensor if needed
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred).float()
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target).float()
    
    metric = SSIMMetric()
    return metric(pred, target)


def evaluate_batch(pred_batch, target_batch):
    psnr_score = calculate_psnr(pred_batch, target_batch)
    ssim_score = calculate_ssim(pred_batch, target_batch)
    
    return {
        'psnr': psnr_score,
        'ssim': ssim_score
    }
