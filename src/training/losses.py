import torch.nn as nn
import torch
from torchvision.models import vgg16

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg16(pretrained=True).features[:16]
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg.eval()
    def forward(self, pred, target):
        # inputs must be 0-1, 3 channels
        return nn.functional.l1_loss(
            self.vgg((pred+1)/2), self.vgg((target+1)/2))

def get_loss_function():
    l1 = nn.L1Loss()
    perceptual = PerceptualLoss()
    def loss_fn(pred, target):
        return l1(pred, target) + 0.1 * perceptual(pred, target)
    return loss_fn