# src/training/metrics.py
"""
Metrics - PSNR / SSIM wrappers with explicit device placement and shape safety
"""

import torch
import torch.nn.functional as F
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure


def _match_size(x, ref):
    if x.shape[-2:] != ref.shape[-2:]:
        x = F.interpolate(x, size=ref.shape[-2:], mode='bilinear', align_corners=True)
    return x


def _match_device(x, ref):
    if x.device != ref.device:
        x = x.to(ref.device)
    return x


class PSNRMetric:
    def __init__(self, data_range=2.0, device: torch.device | None = None):
        # Create metric and optionally place on a device
        self.metric = PeakSignalNoiseRatio(data_range=data_range)
        if device is not None:
            self.metric = self.metric.to(device)

    def to(self, device):
        self.metric = self.metric.to(device)
        return self

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        # Ensure metric is on the same device as tensors
        if next(self.metric.parameters(), None) is not None:
            if next(self.metric.parameters()).device != target.device:
                self.metric = self.metric.to(target.device)
        # Align shapes/devices for inputs
        pred = _match_size(pred, target)
        pred = _match_device(pred, target)
        return self.metric(pred, target).item()


class SSIMMetric:
    def __init__(self, data_range=2.0, device: torch.device | None = None):
        self.metric = StructuralSimilarityIndexMeasure(data_range=data_range)
        if device is not None:
            self.metric = self.metric.to(device)

    def to(self, device):
        self.metric = self.metric.to(device)
        return self

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        if next(self.metric.parameters(), None) is not None:
            if next(self.metric.parameters()).device != target.device:
                self.metric = self.metric.to(target.device)
        pred = _match_size(pred, target)
        pred = _match_device(pred, target)
        return self.metric(pred, target).item()
