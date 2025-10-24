"""
Trainer with LR scheduling + early stopping
"""

import torch, time
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.training.losses import get_loss_function
from src.training.metrics import PSNRMetric, SSIMMetric
from src.models.unet import UNet
from src.data.dataset import create_dataloaders


class Trainer:
    def __init__(self, cfg):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UNet().to(self.device)
        self.criterion = get_loss_function()
        self.optimizer = Adam(self.model.parameters(), lr=cfg["optimizer"]["lr"])
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
        )
        self.metrics = {"psnr": PSNRMetric().to(self.device), "ssim": SSIMMetric().to(self.device)}
        self.writer = SummaryWriter(log_dir="runs/experiment")
        self.cfg = cfg
        self.best_psnr = -1

    def train_one_epoch(self, loader):
        self.model.train()
        total_loss, total_psnr = 0, 0
        for blurred, sharp in loader:
            blurred, sharp = blurred.to(self.device), sharp.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(blurred)
            loss = self.criterion(pred, sharp)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            total_psnr += self.metrics["psnr"](pred, sharp)
        return total_loss / len(loader), total_psnr / len(loader)

    def validate(self, loader):
        self.model.eval()
        total_psnr, total_ssim = 0, 0
        with torch.no_grad():
            for blurred, sharp in loader:
                blurred, sharp = blurred.to(self.device), sharp.to(self.device)
                pred = self.model(blurred)
                total_psnr += self.metrics["psnr"](pred, sharp)
                total_ssim += self.metrics["ssim"](pred, sharp)
        return total_psnr / len(loader), total_ssim / len(loader)

    def train(self, epochs=100):
        train_loader, val_loader = create_dataloaders(self.cfg)
        patience, wait = 10, 0
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss, train_psnr = self.train_one_epoch(train_loader)
            val_psnr, val_ssim = self.validate(val_loader)
            self.scheduler.step(val_psnr)
            lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch}/{epochs} "
                f"| Loss: {train_loss:.4f} "
                f"| Val PSNR: {val_psnr:.2f} "
                f"| Val SSIM: {val_ssim:.3f} "
                f"| LR: {lr:.6f}"
            )

            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("PSNR/val", val_psnr, epoch)
            self.writer.add_scalar("SSIM/val", val_ssim, epoch)

            if val_psnr > self.best_psnr:
                self.best_psnr = val_psnr
                torch.save(self.model.state_dict(), "checkpoints/best_model.pth")
                wait = 0
                print(f"✅  New best model ({val_psnr:.2f} dB)")
            else:
                wait += 1
                if wait > patience:
                    print("⚠️ Early stopping – no improvement")
                    break
            print(f"⏱️ Epoch time : {(time.time()-t0):.1f}s\n")


if __name__ == "__main__":
    from src.config import load_config
    cfg = load_config("config.yaml")
    Trainer(cfg).train(cfg["training"]["num_epochs"])
