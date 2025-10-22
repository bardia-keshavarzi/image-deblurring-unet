# src/training/losses.py
import torch.nn as nn

def get_loss_function():
    """Return L1 loss only"""
    return nn.L1Loss()