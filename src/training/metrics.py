# src/training/metrics.py
"""
Metrics - PSNR / SSIM wrappers with shape/device safety
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
    def __init__(self, data_range=2.0):  # tanh outputs in [-1,1] → range=2
        self.metric = PeakSignalNoiseRatio(data_range=data_range)

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        pred = _match_size(pred, target)
        pred = _match_device(pred, target)
        return self.metric(pred, target).item()


class SSIMMetric:
    def __init__(self, data_range=2.0):  # tanh outputs in [-1,1]
        # Default kernel/gaussian settings from torchmetrics
        self.metric = StructuralSimilarityIndexMeasure(data_range=data_range)

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        pred = _match_size(pred, target)
        pred = _match_device(pred, target)
        return self.metric(pred, target).item()
