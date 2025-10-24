"""
Image quality metrics (PSNR / SSIM)
Optimized, reliable for mixed precision & batch calculation
"""

import torch
import numpy as np
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure


# --------------------------------------------------------------
# PSNR Metric (batched, GPU-safe, AMP-safe)
# --------------------------------------------------------------
class PSNRMetric:
    """Lightweight PSNR metric using torchmetrics.functional"""

    def __init__(self, data_range=1.0):
        self.data_range = data_range

    def __call__(self, pred, target):
        # Clamp predicted and target images
        pred = torch.clamp(pred, 0.0, 1.0)
        target = torch.clamp(target, 0.0, 1.0)

        # Functional metric is faster for batched ops and AMP
        return peak_signal_noise_ratio(pred, target, data_range=self.data_range).detach().item()


# --------------------------------------------------------------
# SSIM Metric (batched, GPU-safe, AMP-safe)
# --------------------------------------------------------------
class SSIMMetric:
    """Lightweight SSIM metric using torchmetrics.functional"""

    def __init__(self, data_range=1.0):
        self.data_range = data_range

    def __call__(self, pred, target):
        pred = torch.clamp(pred, 0.0, 1.0)
        target = torch.clamp(target, 0.0, 1.0)

        return structural_similarity_index_measure(pred, target, data_range=self.data_range).detach().item()


# --------------------------------------------------------------
# Convenience batch evaluators
# --------------------------------------------------------------
def calculate_psnr(pred, target):
    """Convenience wrapper for PSNR"""
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred).float().div(255.0)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target).float().div(255.0)

    pred, target = pred.unsqueeze(0) if pred.ndim == 3 else pred, target.unsqueeze(0) if target.ndim == 3 else target
    return PSNRMetric()(pred, target)


def calculate_ssim(pred, target):
    """Convenience wrapper for SSIM"""
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred).float().div(255.0)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target).float().div(255.0)

    pred, target = pred.unsqueeze(0) if pred.ndim == 3 else pred, target.unsqueeze(0) if target.ndim == 3 else target
    return SSIMMetric()(pred, target)


def evaluate_batch(pred_batch, target_batch):
    """Simultaneously compute PSNR + SSIM on a batch"""
    psnr_score = PSNRMetric()(pred_batch, target_batch)
    ssim_score = SSIMMetric()(pred_batch, target_batch)
    return {"psnr": psnr_score, "ssim": ssim_score}


# --------------------------------------------------------------
# Test script
# --------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Optimized PSNR & SSIM Metrics")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    target = torch.rand(2, 3, 256, 256, device=device)
    psnr_metric = PSNRMetric()
    ssim_metric = SSIMMetric()

    # perfect reconstruction
    print("\nPerfect Reconstruction Test")
    pred = target.clone()
    psnr_val = psnr_metric(pred, target)
    ssim_val = ssim_metric(pred, target)
    print(f"  PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}")

    # small noise test
    print("\nSmall Noise Test (σ ≈ 0.05)")
    noise = torch.randn_like(target) * 0.05
    pred_noisy = torch.clamp(target + noise, 0, 1)
    psnr_val = psnr_metric(pred_noisy, target)
    ssim_val = ssim_metric(pred_noisy, target)
    print(f"  PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}")

    print("=" * 60)
    print("✅ Metric tests complete.")
    print("=" * 60)
