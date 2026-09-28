"""
Sentinel-2 SRGAN Fine-Tuning and Training Engine
Optimized for Google Colab (T4 / V100 GPU) and Local Workstations.
"""

import os
import sys
import yaml
import time
import argparse
from pathlib import Path
import numpy as np

# Ensure ml-service root is in sys.path
current_dir = Path(__file__).resolve().parent
ml_root = current_dir.parent
if str(ml_root) not in sys.path:
    sys.path.insert(0, str(ml_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.srgan.generator import SRGANGenerator
from models.srgan.discriminator import SRGANDiscriminator
from models.srgan.losses import SpectralAngleMapperLoss, GradientEdgeLoss
from training.dataset import SatelliteSRDataset

def calculate_psnr(img1, img2):
    """Computes PSNR between two batches of tensors in [0, 1]."""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    return 20.0 * torch.log10(1.0 / torch.sqrt(mse))

def main():
    parser = argparse.ArgumentParser(description="Train Sentinel-2 SRGAN Super-Resolution Model")
    parser.add_argument("--config", type=str, default="ml-service/configs/srgan_config.yaml", help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs from config")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size from config")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_file():
        # Fallback to local path relative to root
        config_path = ml_root / "configs" / "srgan_config.yaml"

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Overrides
    epochs = args.epochs or cfg["training"].get("epochs", 25)
    batch_size = args.batch_size or cfg["training"].get("batch_size", 8)
    lr_g = float(cfg["training"].get("lr_generator", 1e-4))
    lr_d = float(cfg["training"].get("lr_discriminator", 1e-4))
    b1 = float(cfg["training"].get("b1", 0.9))
    b2 = float(cfg["training"].get("b2", 0.999))

    in_channels = cfg["data"].get("in_channels", 4)
    scale_factor = cfg["data"].get("scale_factor", 4)

    save_dir = Path(cfg["experiment"].get("save_dir", "checkpoints"))
    save_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["training"].get("device", "cuda") == "cuda" else "cpu")

    print("================================================================")
    print("🛰️ Nexus Sentinel-2 SRGAN Fine-Tuning Pipeline")
    print(f"⚡ Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"📊 Hyperparameters: Epochs={epochs}, BatchSize={batch_size}, LR_G={lr_g}, LR_D={lr_d}")
    print(f"📡 Multi-Spectral Channels: {in_channels} (RGB + NIR), Scale: {scale_factor}x")
    print(f"💾 Checkpoints Output: {save_dir}")
    print("================================================================")

    # Datasets
    train_lr = cfg["data"]["train_lr_dir"]
    train_hr = cfg["data"]["train_hr_dir"]
    val_lr = cfg["data"].get("val_lr_dir", train_lr)
    val_hr = cfg["data"].get("val_hr_dir", train_hr)

    train_dataset = SatelliteSRDataset(train_lr, train_hr, in_channels=in_channels)
    if len(train_dataset) == 0:
        print(f"❌ Error: No training pairs found in {train_lr} and {train_hr}.")
        print("👉 Run first: python scripts/prepare_dataset.py --source data/raw/sentinel2")
        sys.exit(1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg["training"].get("num_workers", 2),
        drop_last=True if len(train_dataset) > batch_size else False
    )

    # Initialize Networks
    generator = SRGANGenerator(in_c=in_channels, out_c=in_channels, scale=scale_factor).to(device)
    discriminator = SRGANDiscriminator(in_channels=in_channels).to(device)

    # Losses
    criterion_content = nn.L1Loss().to(device)
    criterion_adv = nn.BCEWithLogitsLoss().to(device)
    criterion_sam = SpectralAngleMapperLoss().to(device)
    criterion_edge = GradientEdgeLoss().to(device)

    lambda_content = float(cfg["training"].get("lambda_content", 1.0))
    lambda_adv = float(cfg["training"].get("lambda_adv", 1e-3))
    lambda_sam = float(cfg["training"].get("lambda_sam", 0.05))
    lambda_edge = float(cfg["training"].get("lambda_edge", 0.05))

    # Optimizers
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(b1, b2))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr_d, betas=(b1, b2))

    best_psnr = 0.0

    print(f"\n🚀 Starting training for {epochs} epochs on {len(train_dataset)} patches...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        generator.train()
        discriminator.train()

        running_g_loss = 0.0
        running_d_loss = 0.0
        running_sam = 0.0
        running_psnr = 0.0
        batch_count = 0

        for lr_imgs, hr_imgs in train_loader:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            current_bs = lr_imgs.size(0)

            # Ground truth labels for discriminator
            real_labels = torch.ones(current_bs, 1, device=device)
            fake_labels = torch.zeros(current_bs, 1, device=device)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_d.zero_grad()

            # Real high-res loss
            pred_real = discriminator(hr_imgs)
            d_real_loss = criterion_adv(pred_real, real_labels)

            # Fake super-res loss
            sr_imgs = generator(lr_imgs)
            pred_fake = discriminator(sr_imgs.detach())
            d_fake_loss = criterion_adv(pred_fake, fake_labels)

            d_loss = (d_real_loss + d_fake_loss) * 0.5
            d_loss.backward()
            optimizer_d.step()

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_g.zero_grad()

            pred_fake_g = discriminator(sr_imgs)
            adv_loss = criterion_adv(pred_fake_g, real_labels)
            content_loss = criterion_content(sr_imgs, hr_imgs)
            sam_loss = criterion_sam(sr_imgs, hr_imgs)
            edge_loss = criterion_edge(sr_imgs, hr_imgs)

            g_loss = (
                lambda_content * content_loss +
                lambda_adv * adv_loss +
                lambda_sam * sam_loss +
                lambda_edge * edge_loss
            )
            g_loss.backward()
            optimizer_g.step()

            # Metric tracking
            with torch.no_grad():
                psnr_val = calculate_psnr(sr_imgs, hr_imgs).item()

            running_g_loss += g_loss.item()
            running_d_loss += d_loss.item()
            running_sam += sam_loss.item()
            running_psnr += psnr_val
            batch_count += 1

        avg_g_loss = running_g_loss / max(1, batch_count)
        avg_d_loss = running_d_loss / max(1, batch_count)
        avg_sam = running_sam / max(1, batch_count)
        avg_psnr = running_psnr / max(1, batch_count)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Loss_G: {avg_g_loss:.4f} | Loss_D: {avg_d_loss:.4f} | "
            f"SAM: {avg_sam:.4f} rad | PSNR: {avg_psnr:.2f} dB"
        )

        # Save best model
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save(generator.state_dict(), save_dir / "generator_best.pth")
            torch.save(discriminator.state_dict(), save_dir / "discriminator_best.pth")
            # Also save as default weights for ml-service
            torch.save(generator.state_dict(), save_dir / "srgan_sentinel2_x4.pth")

        if epoch % cfg["training"].get("save_every", 5) == 0:
            torch.save(generator.state_dict(), save_dir / f"generator_epoch_{epoch}.pth")

    total_time = (time.time() - start_time) / 60.0
    print("\n================================================================")
    print(f"🎉 Training finished in {total_time:.2f} minutes!")
    print(f"🏆 Best PSNR: {best_psnr:.2f} dB")
    print(f"💾 Best checkpoint saved to: {save_dir / 'generator_best.pth'}")
    print(f"💾 Production weights saved to: {save_dir / 'srgan_sentinel2_x4.pth'}")
    print("================================================================")

if __name__ == "__main__":
    main()
