import os
import glob
import time
import argparse
import random
import numpy as np
import rasterio
from rasterio.windows import Window
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from PIL import Image

from models.srgan.generator import SRGANGenerator
from evaluation.metrics import RemoteSensingMetrics
from preprocessing.normalize import normalize_sentinel2

# Canonical band config
CANONICAL_BANDS = 4

class SentinelWaldDataset(Dataset):
    def __init__(self, tiles, tile_size=256, scale_factor=4, augment=False):
        self.tiles = tiles
        self.tile_size = tile_size
        self.scale_factor = scale_factor
        self.augment = augment

    def __len__(self):
        return len(self.tiles)

    def _degrade(self, hr_tensor):
        sigma = 1.0
        kernel_size = 5
        x = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        gaussian = torch.exp(-(x**2) / (2 * sigma**2))
        kernel1d = gaussian / gaussian.sum()
        kernel2d = (kernel1d.unsqueeze(1) @ kernel1d.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
        kernel2d = kernel2d.expand(hr_tensor.shape[0], 1, kernel_size, kernel_size).to(hr_tensor.device)
        
        padded = F.pad(hr_tensor.unsqueeze(0), (kernel_size//2,)*4, mode='reflect')
        blurred = F.conv2d(padded, kernel2d, groups=hr_tensor.shape[0]).squeeze(0)
        lr_tensor = F.interpolate(blurred.unsqueeze(0), scale_factor=1.0/self.scale_factor, mode='bicubic', align_corners=False).squeeze(0)
        return torch.clamp(lr_tensor, 0.0, 1.0)

    def __getitem__(self, idx):
        file_path, x, y, tags = self.tiles[idx]
        with rasterio.open(file_path) as src:
            window = Window(x, y, self.tile_size, self.tile_size)
            arr = src.read(window=window)
            
            norm_arr, _ = normalize_sentinel2(arr, tags=tags)
            hr_tensor = torch.from_numpy(norm_arr)

        if self.augment:
            if random.random() > 0.5:
                hr_tensor = torch.flip(hr_tensor, [2])
            if random.random() > 0.5:
                hr_tensor = torch.flip(hr_tensor, [1])
            k = random.randint(0, 3)
            hr_tensor = torch.rot90(hr_tensor, k, [1, 2])
            
        lr_tensor = self._degrade(hr_tensor)
        return lr_tensor, hr_tensor

def discover_tiles(data_dir, tile_size, expected_bands, smoke_test=False):
    files = glob.glob(os.path.join(data_dir, "**/*.tif"), recursive=True)
    if not files:
        return [], []

    valid_files = []
    for f in files:
        try:
            with rasterio.open(f) as src:
                if src.count == expected_bands:
                    valid_files.append(f)
        except Exception:
            pass

    if not valid_files:
        return [], []

    # Split by source file strictly to avoid spatial leakage
    random.seed(42)
    random.shuffle(valid_files)
    
    if len(valid_files) < 2:
        if smoke_test:
            print("[SMOKE TEST] Only 1 valid file found. Duplicating it into train and val. Metrics will NOT BE VALID.")
            train_files = valid_files
            val_files = valid_files
        else:
            raise RuntimeError(f"Found {len(valid_files)} file(s). Need at least 2 distinct files for a strict spatial train/val split. Pass --smoke-test to bypass.")
    else:
        val_count = max(1, int(len(valid_files) * 0.2))
        val_files = valid_files[:val_count]
        train_files = valid_files[val_count:]

    def extract_tiles(flist):
        tiles = []
        for f in flist:
            with rasterio.open(f) as src:
                w, h = src.width, src.height
                tags = src.tags()
                for y in range(0, h - tile_size + 1, tile_size):
                    for x in range(0, w - tile_size + 1, tile_size):
                        tiles.append((f, x, y, tags))
        return tiles

    train_tiles = extract_tiles(train_files)
    val_tiles = extract_tiles(val_files)
    
    print(f"Discovered {len(train_tiles)} train patches and {len(val_tiles)} validation patches.")
    if len(train_tiles) < 200 and not smoke_test:
        print(f"WARNING: Only {len(train_tiles)} training patches found. < 200 patches is not recommended.")
        
    return train_tiles, val_tiles

def evaluate(model, dataloader, device, scale_factor, epoch, out_dir):
    model.eval()
    psnr_model, ssim_model = 0.0, 0.0
    psnr_bicubic, ssim_bicubic = 0.0, 0.0
    
    saved_image = False
    
    with torch.no_grad():
        for i, (lr, hr) in enumerate(dataloader):
            lr, hr = lr.to(device), hr.to(device)
            sr = model(lr)
            
            sr_np = sr.cpu().numpy()
            hr_np = hr.cpu().numpy()
            lr_np = lr.cpu().numpy()
            
            for j in range(sr.shape[0]):
                m_sr = RemoteSensingMetrics.evaluate_all(sr_np[j], hr_np[j], scale_factor)
                psnr_model += m_sr.get('psnr', 0)
                ssim_model += m_sr.get('ssim', 0)
                
                lr_t = torch.from_numpy(lr_np[j]).unsqueeze(0)
                bicubic = F.interpolate(lr_t, scale_factor=scale_factor, mode='bicubic', align_corners=False).squeeze(0).numpy()
                bicubic = np.clip(bicubic, 0.0, 1.0)
                m_bi = RemoteSensingMetrics.evaluate_all(bicubic, hr_np[j], scale_factor)
                
                psnr_bicubic += m_bi.get('psnr', 0)
                ssim_bicubic += m_bi.get('ssim', 0)
                
                if not saved_image and j == 0:
                    from preprocessing.normalize import denormalize_to_uint8
                    hr_img = Image.fromarray(denormalize_to_uint8(hr_np[j][:3]).transpose(1,2,0))
                    sr_img = Image.fromarray(denormalize_to_uint8(sr_np[j][:3]).transpose(1,2,0))
                    bi_img = Image.fromarray(denormalize_to_uint8(bicubic[:3]).transpose(1,2,0))
                    
                    combined = Image.new('RGB', (hr_img.width * 3, hr_img.height))
                    combined.paste(hr_img, (0, 0))
                    combined.paste(bi_img, (hr_img.width, 0))
                    combined.paste(sr_img, (hr_img.width*2, 0))
                    combined.save(os.path.join(out_dir, f"val_epoch_{epoch}_hr_bi_sr.png"))
                    saved_image = True

    n = max(1, len(dataloader.dataset))
    return psnr_model/n, ssim_model/n, psnr_bicubic/n, ssim_bicubic/n

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Training on device: {device}")
    
    train_tiles, val_tiles = discover_tiles(args.data_dir, args.patch_size, args.bands, args.smoke_test)
    if not train_tiles:
        print("No valid tiles found. Aborting.")
        return
        
    train_dataset = SentinelWaldDataset(train_tiles, tile_size=args.patch_size, scale_factor=args.scale, augment=True)
    val_dataset = SentinelWaldDataset(val_tiles, tile_size=args.patch_size, scale_factor=args.scale, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    model = SRGANGenerator(in_c=args.bands, out_c=args.bands, scale=args.scale).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss()
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    best_psnr = 0.0
    
    print(f"Starting training for {args.epochs} epochs... (In_channels: {args.bands})")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        
        for batch_idx, (lr, hr) in enumerate(train_loader):
            lr, hr = lr.to(device), hr.to(device)
            optimizer.zero_grad()
            sr = model(lr)
            loss = criterion(sr, hr)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader)
        
        psnr_m, ssim_m, psnr_b, ssim_b = evaluate(model, val_loader, device, args.scale, epoch, args.checkpoint_dir)
        
        elapsed = time.time() - start_time
        
        valid_str = "[NOT VALID] " if args.smoke_test else ""
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Time: {elapsed:.2f}s")
        print(f"  {valid_str}[Model]   Val PSNR: {psnr_m:.2f} dB, SSIM: {ssim_m:.4f}")
        print(f"  {valid_str}[Bicubic] Val PSNR: {psnr_b:.2f} dB, SSIM: {ssim_b:.4f}")
        
        if psnr_m > best_psnr:
            best_psnr = psnr_m
            best_path = os.path.join(args.checkpoint_dir, "generator_best.pth")
            checkpoint = {
                "state_dict": model.state_dict(),
                "in_channels": args.bands,
                "scale_factor": args.scale
            }
            torch.save(checkpoint, best_path)
            print(f"  --> New best PSNR! Saved checkpoint to {best_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wald Protocol Self-Supervised Training for SRGAN")
    parser.add_argument("--data_dir", type=str, default="../../data/raw/sentinel2", help="Directory containing raw GeoTIFFs")
    parser.add_argument("--checkpoint_dir", type=str, default="../checkpoints", help="Directory to save weights")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--patch_size", type=int, default=256, help="Tile size for HR images")
    parser.add_argument("--scale", type=int, default=4, help="Super-resolution scale factor")
    parser.add_argument("--bands", type=int, default=CANONICAL_BANDS, help="Number of bands the generator expects")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--cpu", action="store_true", help="Force training on CPU")
    parser.add_argument("--smoke-test", action="store_true", help="Bypass strict valid split validation")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    args = parser.parse_args()
    args.data_dir = os.path.normpath(os.path.join(base_dir, args.data_dir))
    args.checkpoint_dir = os.path.normpath(os.path.join(base_dir, args.checkpoint_dir))
    
    train(args)
