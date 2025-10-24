"""
Combined SSIM + L1 + Perceptual Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16
from torchmetrics.functional import structural_similarity_index_measure


class PerceptualLoss(nn.Module):
    def __init__(self, layer_ids=[3, 8, 15]):
        super().__init__()
        vgg = vgg16(weights="IMAGENET1K_V1").features.eval()
        self.vgg = nn.Sequential(*list(vgg.children())[:max(layer_ids) + 1])
        for p in self.vgg.parameters():
            p.requires_grad = False
        self.layer_ids = layer_ids
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def forward(self, pred, target):
        device = pred.device
        mean, std = self.mean.to(device), self.std.to(device)
        x, y = (pred - mean) / std, (target - mean) / std
        loss = 0
        for i, layer in enumerate(self.vgg):
            x, y = layer(x), layer(y)
            if i in self.layer_ids:
                loss += F.l1_loss(x, y)
        return loss / len(self.layer_ids)


class CombinedLoss(nn.Module):
    def __init__(self, l1_w=0.5, perc_w=0.3, ssim_w=0.2):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.perc = PerceptualLoss()
        self.l1_w = l1_w
        self.perc_w = perc_w
        self.ssim_w = ssim_w

    def forward(self, pred, target):
        l1_loss = self.l1(pred, target)
        perc_loss = self.perc(pred, target)
        ssim_val = structural_similarity_index_measure(pred, target)
        ssim_loss = 1 - ssim_val
        return (
            self.l1_w * l1_loss
            + self.perc_w * perc_loss
            + self.ssim_w * ssim_loss
        )


def get_loss_function():
    return CombinedLoss()
