"""
PyTorch Dataset for Multi-Spectral Sentinel-2 Super-Resolution
Loads paired LR and HR patches (.npy, .tif, or images).
"""

import os
import glob
import numpy as np
from pathlib import Path

try:
    import torch
    from torch.utils.data import Dataset

    class SatelliteSRDataset(Dataset):
        def __init__(self, lr_dir: str, hr_dir: str, in_channels: int = 4):
            self.lr_dir = Path(lr_dir)
            self.hr_dir = Path(hr_dir)
            self.in_channels = in_channels

            self.lr_files = sorted(list(self.lr_dir.glob("*.npy")))
            if not self.lr_files:
                # Try images/tifs
                for ext in ["*.tif", "*.png", "*.jpg"]:
                    self.lr_files.extend(sorted(list(self.lr_dir.glob(ext))))

        def __len__(self):
            return len(self.lr_files)

        def __getitem__(self, idx):
            lr_path = self.lr_files[idx]
            hr_path = self.hr_dir / lr_path.name

            if not hr_path.exists():
                raise FileNotFoundError(f"Missing matching HR file for {lr_path.name} in {self.hr_dir}")

            if lr_path.suffix == ".npy":
                lr_arr = np.load(lr_path).astype(np.float32)
                hr_arr = np.load(hr_path).astype(np.float32)
            else:
                from PIL import Image
                lr_img = Image.open(lr_path).convert("RGB")
                hr_img = Image.open(hr_path).convert("RGB")
                lr_arr = np.transpose(np.array(lr_img, dtype=np.float32) / 255.0, (2, 0, 1))
                hr_arr = np.transpose(np.array(hr_img, dtype=np.float32) / 255.0, (2, 0, 1))

            # Ensure channel count matches in_channels
            if lr_arr.shape[0] > self.in_channels:
                lr_arr = lr_arr[:self.in_channels]
                hr_arr = hr_arr[:self.in_channels]
            elif lr_arr.shape[0] < self.in_channels:
                pad_lr = np.repeat(lr_arr[:1], self.in_channels - lr_arr.shape[0], axis=0)
                pad_hr = np.repeat(hr_arr[:1], self.in_channels - hr_arr.shape[0], axis=0)
                lr_arr = np.concatenate([lr_arr, pad_lr], axis=0)
                hr_arr = np.concatenate([hr_arr, pad_hr], axis=0)

            return torch.from_numpy(lr_arr), torch.from_numpy(hr_arr)

except ImportError:
    Dataset = object
    SatelliteSRDataset = None
