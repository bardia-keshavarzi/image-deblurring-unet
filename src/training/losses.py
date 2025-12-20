import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure as ssim


def gradient_loss(pred, target):
    gx_pred = pred[:, :, :, :-1] - pred[:, :, :, 1:]
    gy_pred = pred[:, :, :-1, :] - pred[:, :, 1:, :]
    gx_targ = target[:, :, :, :-1] - target[:, :, :, 1:]
    gy_targ = target[:, :, :-1, :] - target[:, :, 1:, :]
    return F.l1_loss(gx_pred, gx_targ) + F.l1_loss(gy_pred, gy_targ)


class CombinedLoss(nn.Module):
    def __init__(self, l1_w=0.6, ssim_w=0.1, grad_w=0.05):
        super().__init__()
        self.l1_w = l1_w
        self.ssim_w = ssim_w
        self.grad_w = grad_w

        self.l1 = nn.L1Loss()


    def forward(self, pred, target):
        pred = torch.clamp(pred, 0, 1)
        target = torch.clamp(target, 0, 1)

        l1_loss = self.l1(pred, target)

        ssim_loss = 1 - ssim(pred, target)
        grad_l = gradient_loss(pred, target)

        return (
            self.l1_w * l1_loss
            + self.ssim_w * ssim_loss
            + self.grad_w * grad_l
        )


def get_loss_function():
    return CombinedLoss()