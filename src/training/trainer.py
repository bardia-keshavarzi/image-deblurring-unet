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
        self.psnr_metric = PSNRMetric()
        self.ssim_metric = SSIMMetric()
        
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
    
    def save_checkpoint(self, filename='checkpoint.pth', is_best=False):
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_psnr': self.best_psnr,
        }
        filepath = self.save_dir / filename
        torch.save(checkpoint, filepath)
        if is_best:
            torch.save(checkpoint, self.save_dir / 'best_model.pth')
    
    def train(self, num_epochs):
        print(f"{'='*60}\nStarting Training: {num_epochs} epochs\nTarget: 28-30 dB PSNR\n{'='*60}\n")
        start_time = time.time()
        for epoch in range(num_epochs):
            self.epoch = epoch
            train_m = self.train_one_epoch()
            val_m = self.validate()
            self.scheduler.step(val_m['psnr'])
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Loss/train', train_m['loss'], epoch)
            self.writer.add_scalar('PSNR/train', train_m['psnr'], epoch)
            self.writer.add_scalar('PSNR/val', val_m['psnr'], epoch)
            self.writer.add_scalar('SSIM/val', val_m['ssim'], epoch)
            self.writer.add_scalar('LR', current_lr, epoch)
            print(f"\nEpoch {epoch+1}/{num_epochs} Summary:\n  Train Loss: {train_m['loss']:.4f}\n  Train PSNR: {train_m['psnr']:.2f} dB\n  Val PSNR:   {val_m['psnr']:.2f} dB\n  Val SSIM:   {val_m['ssim']:.4f}\n  LR:         {current_lr:.6f}\n")
            is_best = val_m['psnr'] > self.best_psnr
            if is_best:
                improvement = val_m['psnr'] - self.best_psnr
                self.best_psnr = val_m['psnr']
                self.epochs_no_improve = 0
                print(f"  ✅ New best! PSNR: {self.best_psnr:.2f} dB (+{improvement:.2f} dB)")
                self.save_checkpoint('best_model.pth', is_best=True)
            else:
                self.epochs_no_improve += 1
                print(f"  No improvement for {self.epochs_no_improve} epochs (best: {self.best_psnr:.2f} dB)")
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')
            if self.epochs_no_improve >= self.patience:
                print(f"\n⚠️ Early stopping! No improvement for {self.patience} epochs")
                break
        self.save_checkpoint('last_model.pth')
        elapsed = time.time() - start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        print(f"{'='*60}\nTraining Complete!\n{'='*60}\nTime: {hours}h {minutes}m\nBest Validation PSNR: {self.best_psnr:.2f} dB\nFinal model saved to: {self.save_dir}/best_model.pth\n{'='*60}\n")
        self.writer.close()


def _glob_images(root: Path):
    exts = ('*.png', '*.jpg', '*.jpeg')
    files = []
    for e in exts:
        files.extend(sorted(root.glob(e)))
    return files


def _resolve_data_roots(config):
    """Resolve dataset directories, honoring Colab /content/data if present.
    Expects organized structure as created by scripts/organize_gopro.py:
    /.../data/gopro/{train,test}/{blurred,sharp}
    """
    data_root_env = os.environ.get('DATA_ROOT', None)
    if data_root_env:
        base = Path(data_root_env)
    else:
        colab_root = Path('/content/data')
        base = colab_root if colab_root.exists() else Path('.')
    
    def pick(path_str: str) -> Path:
        p = Path(path_str)
        if p.is_absolute() and p.exists():
            return p
        repo_rel = Path(path_str)
        if repo_rel.exists():
            return repo_rel
        candidate = base / Path(path_str).name if base.exists() else repo_rel
        return candidate
    
    dcfg = config['data']
    train_sharp = pick(dcfg['train_sharp'])
    train_blur = pick(dcfg['train_blurred'])
    test_sharp = pick(dcfg['test_sharp'])
    test_blur = pick(dcfg['test_blurred'])
    return train_sharp, train_blur, test_sharp, test_blur


def _build_from_config(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    train_sharp_dir, train_blur_dir, val_sharp_dir, val_blur_dir = _resolve_data_roots(config)
    
    train_sharp_paths = _glob_images(train_sharp_dir)
    train_blurred_paths = _glob_images(train_blur_dir)
    val_sharp_paths = _glob_images(val_sharp_dir)
    val_blurred_paths = _glob_images(val_blur_dir)

    if not train_sharp_paths or not train_blurred_paths:
        raise RuntimeError(f"No training images found. Sharp: {train_sharp_dir}, Blurred: {train_blur_dir}")
    if len(train_sharp_paths) != len(train_blurred_paths):
        raise RuntimeError(f"Train sharp/blur count mismatch: {len(train_sharp_paths)} vs {len(train_blurred_paths)}")
    if len(val_sharp_paths) != len(val_blurred_paths):
        print(f"⚠️ Val sharp/blur count mismatch: {len(val_sharp_paths)} vs {len(val_blurred_paths)}")

    model_cfg = config['model']
    model = create_deblur_unet(
        in_channels=model_cfg.get('in_channels', 3),
        out_channels=model_cfg.get('out_channels', 3),
        output_activation=model_cfg.get('output_activation', 'tanh')
    )

    image_size = int(config['data'].get('image_size', 384))
    augment = bool(config['data'].get('augmentation', True))
    num_workers = int(config['data'].get('num_workers', 4))
    batch_size = int(config['data'].get('batch_size', 8))

    train_dataset = DeblurDataset(
        sharp_paths=train_sharp_paths,
        blurred_paths=train_blurred_paths,
        image_size=image_size,
        augment=augment,
    )
    val_dataset = DeblurDataset(
        sharp_paths=val_sharp_paths,
        blurred_paths=val_blurred_paths,
        image_size=image_size,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    trainer = Trainer(model, train_loader, val_loader, config)
    return trainer, config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Deblurring Trainer')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to YAML config')
    parser.add_argument('--epochs', type=int, default=None, help='Override epochs')
    parser.add_argument('--colab', action='store_true', help='Colab-friendly workers and prints')
    args = parser.parse_args()

    trainer, config = _build_from_config(args.config)

    if args.colab:
        print('Colab mode: consider lowering num_workers=2 in config if you see DataLoader stalls')

    epochs = args.epochs if args.epochs is not None else config.get('training', {}).get('num_epochs', 100)
    trainer.train(epochs)
