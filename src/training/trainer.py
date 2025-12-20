import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from tqdm import tqdm
import time

from src.training.losses import get_loss_function
from src.training.metrics import PSNRMetric, SSIMMetric
from src.models.unet import UNet
from src.utils.config import Config
from src.data.dataset import create_dataloaders


class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        lr = config.get("training.learning_rate", 0.0002)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.get("training.num_epochs", 100), eta_min=1e-6)

        self.criterion = get_loss_function()
        self.psnr_metric = PSNRMetric()
        self.ssim_metric = SSIMMetric()
        self.scaler = torch.cuda.amp.GradScaler()  # AMP setup

        self.save_dir = Path(config.get("training.save_dir", "checkpoints"))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(Path(config.get("training.log_dir", "runs")))

        self.best_psnr = 0.0
        self.epochs_no_improve = 0
        self.patience = config.get("training.patience", 20)
        print(f"Trainer initialized on {self.device}")


    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss, total_psnr = 0.0, 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")

        for blurred, sharp in pbar:
            blurred, sharp = blurred.to(self.device), sharp.to(self.device)

            with torch.cuda.amp.autocast(): 
                pred = self.model(blurred)
                loss = self.criterion(pred, sharp)

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            psnr = self.psnr_metric(pred, sharp)
            total_loss += loss.item()
            total_psnr += psnr

            pbar.set_postfix(loss=f"{loss.item():.4f}", psnr=f"{psnr:.2f}")

        return {"loss": total_loss / len(self.train_loader),
                "psnr": total_psnr / len(self.train_loader)}


    def validate(self, epoch):
        self.model.eval()
        total_psnr, total_ssim = 0.0, 0.0
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]")

        with torch.no_grad():
            for blurred, sharp in pbar:
                blurred, sharp = blurred.to(self.device), sharp.to(self.device)
                with torch.cuda.amp.autocast():
                    pred = self.model(blurred)
                psnr = self.psnr_metric(pred, sharp)
                ssim = self.ssim_metric(pred, sharp)
                total_psnr += psnr
                total_ssim += ssim
                pbar.set_postfix(psnr=f"{psnr:.2f}", ssim=f"{ssim:.4f}")

        return {"psnr": total_psnr / len(self.val_loader),
                "ssim": total_ssim / len(self.val_loader)}

    def save_checkpoint(self, filename="checkpoint.pth", is_best=False):
        state = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "best_psnr": self.best_psnr,
        }
        torch.save(state, self.save_dir / filename)
        if is_best:
            torch.save(state, self.save_dir / "best_model.pth")


    def train(self, epochs):
        for epoch in range(epochs):
            start_time = time.time()
            train_metrics = self.train_one_epoch(epoch)
            val_metrics = self.validate(epoch)
            self.scheduler.step()
            lr = self.optimizer.param_groups[0]["lr"]

            self.writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
            self.writer.add_scalar("PSNR/val", val_metrics["psnr"], epoch)
            self.writer.add_scalar("SSIM/val", val_metrics["ssim"], epoch)
            self.writer.add_scalar("LR", lr, epoch)

            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_metrics['loss']:.4f}, "
                  f"Val PSNR: {val_metrics['psnr']:.2f}, "
                  f"Val SSIM: {val_metrics['ssim']:.4f}, LR: {lr:.6f}")


            if val_metrics["psnr"] > self.best_psnr:
                improvement = val_metrics["psnr"] - self.best_psnr
                self.best_psnr = val_metrics["psnr"]
                self.epochs_no_improve = 0
                print(f"New best model: {self.best_psnr:.2f} dB (+{improvement:.2f})")
                self.save_checkpoint(is_best=True)
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.patience:
                    print("Early stopping triggered.")
                    break

            print(f"Epoch duration: {(time.time() - start_time)/60:.2f} min\n")

        print(f"Training complete. Best PSNR: {self.best_psnr:.2f} dB")
        self.save_checkpoint("final_model.pth")
        self.writer.close()


if __name__ == "__main__":
    cfg = Config("configs/config.yaml")
    train_loader, val_loader = create_dataloaders(
        cfg.get("data.train_sharp"),
        cfg.get("data.train_blurred"),
        cfg.get("data.test_sharp"),
        cfg.get("data.test_blurred"),
        cfg.get("data.image_size"),
        cfg.get("data.batch_size"),
        cfg.get("data.num_workers"),
    )

    model = UNet(in_channels=3, out_channels=3, use_attention=True)
    trainer = Trainer(model, train_loader, val_loader, cfg)
    trainer.train(cfg.get("training.num_epochs"))
