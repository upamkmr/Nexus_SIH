"""
Sentinel-2 Data Preparation & Patch Slicing Script
Slices large multi-spectral satellite imagery into training/validation patches.
Produces matching High-Resolution (HR) and Low-Resolution (LR) pairs.
"""

import os
import sys
import glob
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

def generate_synthetic_sentinel_tile(h=1024, w=1024, num_bands=4):
    """
    Generates synthetic 4-band satellite imagery (R, G, B, NIR) simulating
    agricultural fields, water bodies, and terrain features for prototyping.
    """
    x = np.linspace(0, 10, w)
    y = np.linspace(0, 10, h)
    xx, yy = np.meshgrid(x, y)

    # Base geographic terrain elevation/variation
    terrain = np.sin(xx * 0.5) * np.cos(yy * 0.5) + 0.5 * np.sin(xx * 1.5 + yy * 1.2)
    terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min() + 1e-6)

    # Agricultural field block patterns
    grid_x = (np.sin(xx * 3.0) > 0.1).astype(np.float32)
    grid_y = (np.cos(yy * 3.0) > 0.1).astype(np.float32)
    fields = grid_x * grid_y

    tile = np.zeros((num_bands, h, w), dtype=np.float32)
    # Band 1: Red (B04)
    tile[0] = np.clip(0.3 * terrain + 0.2 * fields + np.random.normal(0, 0.02, (h, w)), 0.05, 0.95)
    # Band 2: Green (B03)
    tile[1] = np.clip(0.35 * terrain + 0.3 * fields + np.random.normal(0, 0.02, (h, w)), 0.05, 0.95)
    # Band 3: Blue (B02)
    tile[2] = np.clip(0.25 * terrain + 0.1 * fields + np.random.normal(0, 0.02, (h, w)), 0.05, 0.95)
    # Band 4: Near-Infrared (B08) - High reflection in vegetation fields
    tile[3] = np.clip(0.4 * terrain + 0.5 * fields + np.random.normal(0, 0.02, (h, w)), 0.05, 0.95)

    return tile

def load_image_or_raster(file_path):
    """Loads GeoTIFF, NumPy, or standard image into (C, H, W) normalized [0, 1] float32."""
    ext = Path(file_path).suffix.lower()
    if ext in [".npy"]:
        data = np.load(file_path)
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        elif data.ndim == 3 and data.shape[-1] in [3, 4]:
            data = np.transpose(data, (2, 0, 1))
        return data.astype(np.float32)

    if ext in [".tif", ".tiff"]:
        try:
            import tifffile
            data = tifffile.imread(file_path)
            if data.ndim == 2:
                data = data[np.newaxis, ...]
            elif data.ndim == 3 and data.shape[-1] in [3, 4]:
                data = np.transpose(data, (2, 0, 1))
            data = data.astype(np.float32)
            if data.max() > 1.0:
                data = (data - data.min()) / (data.max() - data.min() + 1e-6)
            return data
        except Exception:
            pass

    # Standard image (RGB)
    img = Image.open(file_path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1)) # (3, H, W)
    # If 4 bands expected, synthesize NIR band from Green/Red
    nir = np.clip(arr[1] * 1.2 - arr[0] * 0.2, 0.0, 1.0)[np.newaxis, ...]
    return np.concatenate([arr, nir], axis=0)

def downsample_patch(patch_c_h_w, scale=4):
    """Downsamples (C, H, W) patch by scale factor using bicubic interpolation."""
    c, h, w = patch_c_h_w.shape
    lr_h, lr_w = h // scale, w // scale
    lr = np.zeros((c, lr_h, lr_w), dtype=np.float32)
    for i in range(c):
        band_img = Image.fromarray((patch_c_h_w[i] * 255).astype(np.uint8))
        band_lr = band_img.resize((lr_w, lr_h), Image.Resampling.BICUBIC)
        lr[i] = np.array(band_lr, dtype=np.float32) / 255.0
    return lr

def main():
    parser = argparse.ArgumentParser(description="Prepare and slice Sentinel-2 dataset into HR/LR patch pairs.")
    parser.add_argument("--source", type=str, default="data/raw/sentinel2", help="Source folder containing raw images")
    parser.add_argument("--output", type=str, default="data/processed/train", help="Base output directory")
    parser.add_argument("--patch-size", type=int, default=64, help="LR patch size (HR patch size will be patch_size * scale)")
    parser.add_argument("--scale", type=int, default=4, help="Super-resolution scale factor (e.g. 4 for 10m -> 2.5m)")
    parser.add_argument("--val-split", type=float, default=0.15, help="Validation split fraction")
    parser.add_argument("--num-synthetic", type=int, default=4, help="Number of synthetic tiles to generate if source is empty")
    args = parser.parse_args()

    hr_size = args.patch_size * args.scale
    lr_size = args.patch_size

    out_train_hr = Path(args.output) / "hr"
    out_train_lr = Path(args.output) / "lr"
    val_base = Path(args.output).parent / "val"
    out_val_hr = val_base / "hr"
    out_val_lr = val_base / "lr"

    for d in [out_train_hr, out_train_lr, out_val_hr, out_val_lr]:
        d.mkdir(parents=True, exist_ok=True)

    print("================================================================")
    print("🛰️ Nexus Sentinel-2 Dataset Slicer")
    print(f"📁 Source: {args.source}")
    print(f"📦 Output Train HR: {out_train_hr}")
    print(f"📦 Output Train LR: {out_train_lr}")
    print(f"🔍 HR Patch Size: {hr_size}x{hr_size} | LR Patch Size: {lr_size}x{lr_size} (Scale: {args.scale}x)")
    print("================================================================")

    # Search for files
    supported_exts = ["*.tif", "*.tiff", "*.npy", "*.png", "*.jpg", "*.jpeg"]
    raw_files = []
    if os.path.isdir(args.source):
        for ext in supported_exts:
            raw_files.extend(glob.glob(os.path.join(args.source, "**", ext), recursive=True))

    tiles = []
    if not raw_files:
        print(f"ℹ️ No raw image files found in '{args.source}'.")
        print(f"✨ Generating {args.num_synthetic} realistic multi-spectral 4-band simulation tiles (1024x1024)...")
        for i in range(args.num_synthetic):
            tile = generate_synthetic_sentinel_tile(1024, 1024, num_bands=4)
            tiles.append((f"synthetic_tile_{i+1}", tile))
    else:
        print(f"Found {len(raw_files)} raw satellite files. Slicing...")
        for p in raw_files:
            try:
                tile = load_image_or_raster(p)
                tiles.append((Path(p).stem, tile))
            except Exception as e:
                print(f"Warning: Failed to load {p}: {e}")

    total_patches = 0
    for name, tile in tiles:
        _, h, w = tile.shape
        # Slide window
        step = hr_size
        patch_idx = 0
        for y in range(0, h - hr_size + 1, step):
            for x in range(0, w - hr_size + 1, step):
                hr_patch = tile[:, y:y + hr_size, x:x + hr_size]
                lr_patch = downsample_patch(hr_patch, scale=args.scale)

                is_val = (np.random.rand() < args.val_split)
                hr_dest = out_val_hr if is_val else out_train_hr
                lr_dest = out_val_lr if is_val else out_train_lr

                filename = f"{name}_p{patch_idx:04d}.npy"
                np.save(hr_dest / filename, hr_patch.astype(np.float32))
                np.save(lr_dest / filename, lr_patch.astype(np.float32))

                patch_idx += 1
                total_patches += 1

    train_count = len(list(out_train_hr.glob("*.npy")))
    val_count = len(list(out_val_hr.glob("*.npy")))
    print("\n✅ Dataset slicing complete!")
    print(f"   📊 Training pairs: {train_count}")
    print(f"   📊 Validation pairs: {val_count}")
    print(f"   Total patches: {total_patches}")

if __name__ == "__main__":
    main()
