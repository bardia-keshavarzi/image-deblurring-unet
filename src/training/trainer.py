# src/training/train.py
"""
Training Script for U-Net Deblurring

Main training loop with:
- Progress tracking
- Checkpointing (save best model)
- Validation after each epoch
- TensorBoard logging
"""

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import time

from .losses import L1Loss
from .metrics import PSNRMetric, SSIMMetric


class Trainer:
    """
    Training manager for U-Net
    
    Handles:
    - Training loop
    - Validation
    - Checkpointing
    - Logging
    """
    
    def __init__(self, model, train_loader, val_loader, config):
        """
        Initialize trainer
        
        Args:
            model: U-Net model
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            config: Configuration object
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Get device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        print(f"\n{'='*60}")
        print(f"Training Device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"{'='*60}\n")
        
        # Setup optimizer
        lr = config.get('training.learning_rate', 0.0001)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Setup loss and metrics
        self.criterion = L1Loss()
        self.psnr_metric = PSNRMetric()
        self.ssim_metric = SSIMMetric()
        
        # Setup checkpointing
        self.save_dir = Path(config.get('training.save_dir', 'checkpoints'))
        self.save_dir.mkdir(exist_ok=True)
        
        # Setup logging
        log_dir = Path(config.get('training.log_dir', 'runs'))
        self.writer = SummaryWriter(log_dir)
        
        # Tracking
        self.epoch = 0
        self.best_psnr = 0.0
        self.patience = config.get('training.patience', 10)
        self.epochs_no_improve = 0
        
        print("✓ Trainer initialized")
        print(f"  Optimizer: Adam (lr={lr})")
        print(f"  Loss: L1")
        print(f"  Save dir: {self.save_dir}")
        print(f"  Log dir: {log_dir}\n")
    
    def train_one_epoch(self):
        """
        Train for one epoch
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        total_psnr = 0.0
        num_batches = len(self.train_loader)
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch+1} [Train]")
        
        for batch_idx, (blurred, sharp) in enumerate(pbar):
            # Move to device
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
        """
        Validate model
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        total_psnr = 0.0
        total_ssim = 0.0
        num_batches = len(self.val_loader)
        
        # Progress bar
        pbar = tqdm(self.val_loader, desc=f"Epoch {self.epoch+1} [Val]")
        
        with torch.no_grad():
            for blurred, sharp in pbar:
                # Move to device
                blurred = blurred.to(self.device)
                sharp = sharp.to(self.device)
                
                # Forward pass
                pred = self.model(blurred)
                
                # Calculate metrics
                psnr = self.psnr_metric(pred, sharp)
                ssim = self.ssim_metric(pred, sharp)
                
                total_psnr += psnr
                total_ssim += ssim
                
                # Update progress bar
                pbar.set_postfix({
                    'psnr': f'{psnr:.2f}',
                    'ssim': f'{ssim:.4f}'
                })
        
        # Average metrics
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
            'best_psnr': self.best_psnr
        }
        
        filepath = self.save_dir / filename
        torch.save(checkpoint, filepath)
        
        if is_best:
            best_path = self.save_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
    
    def train(self, num_epochs):
        """
        Main training loop
        
        Args:
            num_epochs: Number of epochs to train
        """
        print(f"{'='*60}")
        print(f"Starting Training: {num_epochs} epochs")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_one_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Log to TensorBoard
            self.writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
            self.writer.add_scalar('PSNR/train', train_metrics['psnr'], epoch)
            self.writer.add_scalar('PSNR/val', val_metrics['psnr'], epoch)
            self.writer.add_scalar('SSIM/val', val_metrics['ssim'], epoch)
            
            # Print summary
            print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Train PSNR: {train_metrics['psnr']:.2f} dB")
            print(f"  Val PSNR:   {val_metrics['psnr']:.2f} dB")
            print(f"  Val SSIM:   {val_metrics['ssim']:.4f}")
            
            # Check if best model
            is_best = val_metrics['psnr'] > self.best_psnr
            
            if is_best:
                self.best_psnr = val_metrics['psnr']
                self.epochs_no_improve = 0
                print(f"  ✅ New best! PSNR: {self.best_psnr:.2f} dB")
                self.save_checkpoint('best_model.pth', is_best=True)
            else:
                self.epochs_no_improve += 1
                print(f"  No improvement for {self.epochs_no_improve} epochs (best: {self.best_psnr:.2f} dB)")
            
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')
            
            # Early stopping
            if self.epochs_no_improve >= self.patience:
                print(f"\n⚠️  Early stopping! No improvement for {self.patience} epochs")
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
        print(f"Final model saved to: {self.save_dir}/last_model.pth")
        print(f"Best model saved to: {self.save_dir}/best_model.pth")
        print(f"{'='*60}\n")
        
        self.writer.close()


def main():
    """Main training function"""
    import sys
    from pathlib import Path
    
    # Add project root to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from src.utils.config import Config
    from src.models.unet import UNet
    from src.data import create_dataloaders
    
    print("="*60)
    print("U-Net Training on GoPro Dataset")
    print("="*60)
    
    # Load config
    config = Config()
    
    # Load data
    print("\n[1/4] Loading dataset...")
    
    train_sharp_dir = Path(config.get('data.train_sharp'))
    train_blur_dir = Path(config.get('data.train_blurred'))
    test_sharp_dir = Path(config.get('data.test_sharp'))
    test_blur_dir = Path(config.get('data.test_blurred'))
    
    train_sharp_paths = sorted(train_sharp_dir.glob('*.png'))
    train_blur_paths = sorted(train_blur_dir.glob('*.png'))
    test_sharp_paths = sorted(test_sharp_dir.glob('*.png'))
    test_blur_paths = sorted(test_blur_dir.glob('*.png'))
    
    print(f"  Train: {len(train_sharp_paths)} pairs")
    print(f"  Test: {len(test_sharp_paths)} pairs")
    
    train_loader, val_loader = create_dataloaders(
        train_sharp_paths, test_sharp_paths,
        train_blur_paths, test_blur_paths,
        image_size=config.get('data.image_size'),
        batch_size=config.get('data.batch_size'),
        num_workers=config.get('data.num_workers')
    )
    
    # Create model
    print("\n[2/4] Creating model...")
    model = UNet(
        in_channels=config.get('model.in_channels'),
        out_channels=config.get('model.out_channels')
    )
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {num_params:,} (~{num_params/1e6:.1f}M)")
    
    # Create trainer
    print("\n[3/4] Setting up trainer...")
    trainer = Trainer(model, train_loader, val_loader, config)
    
    # Train!
    print("\n[4/4] Starting training...")
    num_epochs = config.get('training.num_epochs', 50)
    trainer.train(num_epochs)
    
    print("\n✅ Training complete! Check checkpoints/ for saved models.")
    print("   View training progress: tensorboard --logdir=runs")


if __name__ == '__main__':
    main()


at first explain the overview
