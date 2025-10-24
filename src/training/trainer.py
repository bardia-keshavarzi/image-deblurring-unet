# src/training/trainer.py
"""
Trainer - IMPROVED

Added:
- Learning rate scheduler
- Better early stopping
- Loss combination support
"""

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau  # NEW
from pathlib import Path
from tqdm import tqdm
import time

from .losses import get_loss_function
from .metrics import PSNRMetric, SSIMMetric


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
        lr = config.get('training.learning_rate', 0.0002)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # NEW: Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',  # Maximize PSNR
            factor=0.5,  # Reduce LR by half
            patience=5,  # After 5 epochs no improvement
            min_lr=1e-6,
        )
        
        # Loss and metrics
        self.criterion = get_loss_function()
        self.psnr_metric = PSNRMetric()
        self.ssim_metric = SSIMMetric()
        
        # Checkpointing
        self.save_dir = Path(config.get('training.save_dir', 'checkpoints'))
        self.save_dir.mkdir(exist_ok=True)
        
        # Logging
        log_dir = Path(config.get('training.log_dir', 'runs'))
        self.writer = SummaryWriter(log_dir)
        
        # Tracking
        self.epoch = 0
        self.best_psnr = 0.0
        self.patience = config.get('training.patience', 20)
        self.epochs_no_improve = 0
        
        print("✓ Trainer initialized")
        print(f"  Optimizer: Adam (lr={lr})")
        print(f"  Loss: Combined (L1 + Perceptual)")
        print(f"  LR Scheduler: ReduceLROnPlateau")
        print(f"  Save dir: {self.save_dir}\n")
    
    def train_one_epoch(self):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_psnr = 0.0
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch+1} [Train]")
        
        for batch_idx, (blurred, sharp) in enumerate(pbar):
            blurred = blurred.to(self.device)
            sharp = sharp.to(self.device)
            
            # Forward pass
            pred = self.model(blurred)
            loss = self.criterion(pred, sharp)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                psnr = self.psnr_metric(pred, sharp)
            
            # Accumulate
            total_loss += loss.item()
            total_psnr += psnr
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'psnr': f'{psnr:.2f}'
            })
        
        # Average metrics
        avg_loss = total_loss / num_batches
        avg_psnr = total_psnr / num_batches
        
        return {
            'loss': avg_loss,
            'psnr': avg_psnr
        }
    
    def validate(self):
        """Validate model"""
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

                # Compute metrics correctly
                psnr = self.psnr_metric(pred, sharp)
                ssim = self.ssim_metric(pred, sharp)  # ✅ fixed call

                total_psnr += psnr
                total_ssim += ssim

                # Update progress bar properly
                pbar.set_postfix({
                    'psnr': f'{psnr:.2f}',
                    'ssim': f'{ssim:.4f}'
                })
        
        avg_psnr = total_psnr / num_batches
        avg_ssim = total_ssim / num_batches
        
        return {
            'psnr': avg_psnr,
            'ssim': avg_ssim
        }
    
    def save_checkpoint(self, filename='checkpoint.pth', is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),  # NEW
            'best_psnr': self.best_psnr
        }
        
        filepath = self.save_dir / filename
        torch.save(checkpoint, filepath)
        
        if is_best:
            best_path = self.save_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
    
    def train(self, num_epochs):
        """Main training loop"""
        print(f"{'='*60}")
        print(f"Starting Training: {num_epochs} epochs")
        print(f"Target: 28-30 dB PSNR")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_one_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # NEW: Update learning rate based on validation PSNR
            self.scheduler.step(val_metrics['psnr'])
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log to TensorBoard
            self.writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
            self.writer.add_scalar('PSNR/train', train_metrics['psnr'], epoch)
            self.writer.add_scalar('PSNR/val', val_metrics['psnr'], epoch)
            self.writer.add_scalar('SSIM/val', val_metrics['ssim'], epoch)
            self.writer.add_scalar('LR', current_lr, epoch)  # NEW
            
            # Print summary
            print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Train PSNR: {train_metrics['psnr']:.2f} dB")
            print(f"  Val PSNR:   {val_metrics['psnr']:.2f} dB")
            print(f"  Val SSIM:   {val_metrics['ssim']:.4f}")
            print(f"  LR:         {current_lr:.6f}")  # NEW
            
            # Check if best model
            is_best = val_metrics['psnr'] > self.best_psnr
            
            if is_best:
                improvement = val_metrics['psnr'] - self.best_psnr
                self.best_psnr = val_metrics['psnr']
                self.epochs_no_improve = 0
                print(f"  ✅ New best! PSNR: {self.best_psnr:.2f} dB (+{improvement:.2f} dB)")
                self.save_checkpoint('best_model.pth', is_best=True)
            else:
                self.epochs_no_improve += 1
                print(f"  No improvement for {self.epochs_no_improve} epochs (best: {self.best_psnr:.2f} dB)")
            
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')
            
            # Early stopping
            if self.epochs_no_improve >= self.patience:
                print(f"\n⚠️ Early stopping! No improvement for {self.patience} epochs")
                break
            
            print()
        
        # Save final model
        self.save_checkpoint('last_model.pth')
        
        # Training complete
        elapsed = time.time() - start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        
        print(f"{'='*60}")
        print(f"Training Complete!")
        print(f"{'='*60}")
        print(f"Time: {hours}h {minutes}m")
        print(f"Best Validation PSNR: {self.best_psnr:.2f} dB")
        print(f"Improvement from baseline: +{self.best_psnr - 23.89:.2f} dB")
        print(f"Final model saved to: {self.save_dir}/best_model.pth")
        print(f"{'='*60}\n")
        
        self.writer.close()
