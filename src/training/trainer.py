# src/training/trainer.py
"""
Trainer - IMPROVED

Added:
- Learning rate scheduler
- Better early stopping
- Loss combination support
- MAIN: runnable entrypoint for `python -m src.training.trainer`
- FIX: Properly glob files from organized GoPro folders and support Colab /content/data root
- FIX: Use ReduceLROnPlateau arguments compatible with Colab's torch version (no verbose, use threshold_mode)
- FIX: Ensure TorchMetrics (PSNR/SSIM) are moved to the active training device
"""

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau  # NEW
from pathlib import Path
from tqdm import tqdm
import time
import argparse
import yaml
import os

from .losses import get_loss_function
from .metrics import PSNRMetric, SSIMMetric

# Optional imports inside __main__ path modifications
try:
    # Support running as module and script
    import sys
    from ..models.unet import create_deblur_unet
    from ..data.dataset import DeblurDataset
    from torch.utils.data import DataLoader
except Exception:
    pass


class Trainer:
    """Training manager"""
    
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        print(f"\n{'='*60}")
        print(f"Training Device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"{'='*60}\n")
        
        # Optimizer
        lr = (
            self.config.get('training', {}).get('learning_rate')
            if isinstance(self.config.get('training'), dict)
            else self.config.get('training.learning_rate', 0.0002)
        ) or 0.0002
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # NEW: Learning rate scheduler (compatible args)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',          # Maximize PSNR
            factor=0.5,          # Reduce LR by half
            patience=5,          # After 5 epochs no improvement
            threshold=1e-4,      # Significant change threshold
            threshold_mode='rel',# Relative improvements
            min_lr=1e-6
        )
        
        # Loss and metrics
        self.criterion = get_loss_function()
        self.psnr_metric = PSNRMetric(device=self.device)  # ensure created on device
        self.ssim_metric = SSIMMetric(device=self.device)  # ensure created on device
        
        # Also provide an explicit .to(device) just in case
        self.psnr_metric.to(self.device)
        self.ssim_metric.to(self.device)
        
        # Checkpointing
        save_dir = (
            self.config.get('training', {}).get('save_dir', 'checkpoints')
            if isinstance(self.config.get('training'), dict)
            else self.config.get('training.save_dir', 'checkpoints')
        )
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        # Logging
        log_dir = (
            self.config.get('training', {}).get('log_dir', 'runs')
            if isinstance(self.config.get('training'), dict)
            else self.config.get('training.log_dir', 'runs')
        )
        self.writer = SummaryWriter(log_dir)
        
        # Tracking
        self.epoch = 0
        self.best_psnr = 0.0
        self.patience = (
            self.config.get('training', {}).get('patience', 20)
            if isinstance(self.config.get('training'), dict)
            else self.config.get('training.patience', 20)
        )
        self.epochs_no_improve = 0
        
        print("✓ Trainer initialized")
        print(f"  Optimizer: Adam (lr={lr})")
        print(f"  Loss: Combined (L1 + Perceptual)")
        print(f"  LR Scheduler: ReduceLROnPlateau")
        print(f"  Save dir: {self.save_dir}\n")
    
    def train_one_epoch(self):
        self.model.train()
        total_loss = 0.0
        total_psnr = 0.0
        num_batches = len(self.train_loader)
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch+1} [Train]")
        for _, (blurred, sharp) in enumerate(pbar):
            blurred = blurred.to(self.device)
            sharp = sharp.to(self.device)
            pred = self.model(blurred)
            loss = self.criterion(pred, sharp)
            self.optimizer.zero_grad()
            loss.backward()
            grad_clip = (
                self.config.get('training', {}).get('gradient_clipping')
                if isinstance(self.config.get('training'), dict)
                else self.config.get('training.gradient_clipping', None)
            )
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float(grad_clip))
            self.optimizer.step()
            with torch.no_grad():
                # Ensure metrics are on the correct device
                self.psnr_metric.to(self.device)
                self.ssim_metric.to(self.device)
                psnr = self.psnr_metric(pred, sharp)
            total_loss += loss.item()
            total_psnr += psnr
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'psnr': f'{psnr:.2f}'})
        return {'loss': total_loss / max(1, num_batches), 'psnr': total_psnr / max(1, num_batches)}
    
    def validate(self):
        self.model.eval()
        total_psnr = 0.0
        total_ssim = 0.0
        num_batches = len(self.val_loader)
        pbar = tqdm(self.val_loader, desc=f"Epoch {self.epoch+1} [Val]")
        with torch.no_grad():
            # Ensure metrics on device
            self.psnr_metric.to(self.device)
            self.ssim_metric.to(self.device)
            for blurred, sharp in pbar:
                blurred = blurred.to(self.device)
                sharp = sharp.to(self.device)
                pred = self.model(blurred)
                psnr = self.psnr_metric(pred, sharp)
                ssim = self.ssim_metric(pred, sharp)
                total_psnr += psnr
                total_ssim += ssim
                pbar.set_postfix({'psnr': f'{psnr:.2f}', 'ssim': f'{ssim:.4f}'})
        return {'psnr': total_psnr / max(1, num_batches), 'ssim': total_ssim / max(1, num_batches)}

# ... rest of file remains unchanged ...
