# src/inference/__init__.py
"""Inference module for trained U-Net"""

from .predictor import DeblurPredictor

__all__ = ['DeblurPredictor']