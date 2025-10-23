# src/training/losses.py
"""
Loss Functions - IMPROVED

Adds perceptual loss for better visual quality
"""

import torch
import torch.nn as nn
import torchvision.models as models


class PerceptualLoss(nn.Module):
    """VGG-based perceptual loss"""
    
    def __init__(self):
        super().__init__()
        
        # Use VGG16 features (pretrained)
        vgg = models.vgg16(pretrained=True).features[:16]  # Up to relu3_3
        self.vgg = vgg.eval()
        
        # Freeze VGG
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        # Normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        self.l1 = nn.L1Loss()
    
    def forward(self, pred, target):
        # Normalize
        pred_norm = (pred - self.mean) / self.std
        target_norm = (target - self.mean) / self.std
        
        # Extract features
        pred_features = self.vgg(pred_norm)
        target_features = self.vgg(target_norm)
        
        return self.l1(pred_features, target_features)


class CombinedLoss(nn.Module):
    """L1 + Perceptual Loss"""
    
    def __init__(self, alpha=0.8, beta=0.2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.l1 = nn.L1Loss()
        self.perceptual = PerceptualLoss()
    
    def forward(self, pred, target):
        l1_loss = self.l1(pred, target)
        perceptual_loss = self.perceptual(pred, target)
        return self.alpha * l1_loss + self.beta * perceptual_loss


def get_loss_function():
    """Get combined loss function"""
    return CombinedLoss(alpha=0.8, beta=0.2)
