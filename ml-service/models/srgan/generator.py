"""
Sentinel-2 Super-Resolution Generative Adversarial Network (SRGAN)
Optimized for multi-spectral Earth Observation (RGB + NIR 10m bands).
Upscales 4x (10m -> 2.5m GSD).
"""

import os
import numpy as np
from typing import Optional, Dict, Any, Tuple
try:
    from ..base_model import BaseSuperResolutionModel
except Exception:
    try:
        from models.base_model import BaseSuperResolutionModel
    except Exception:
        class BaseSuperResolutionModel:
            def __init__(self, name, scale_factor=4, in_channels=4):
                self.name = name
                self.scale_factor = scale_factor
                self.in_channels = in_channels

try:
    import torch
    import torch.nn as nn

    class ResidualBlock(nn.Module):
        def __init__(self, channels=64):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(channels)
            self.prelu = nn.PReLU()
            self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(channels)

        def forward(self, x):
            return x + self.bn2(self.conv2(self.prelu(self.bn1(self.conv1(x)))))

    class SRGANGenerator(nn.Module):
        """
        Deep Residual Network for Multi-Spectral Super-Resolution.
        Accepts N-channel input (default 4: R, G, B, NIR) and produces 4x super-resolved output.
        """
        def __init__(self, in_c=4, out_c=4, n_res_blocks=16, scale=4):
            super().__init__()
            self.scale = scale
            self.head = nn.Sequential(
                nn.Conv2d(in_c, 64, kernel_size=9, padding=4),
                nn.PReLU()
            )
            self.body = nn.Sequential(*[ResidualBlock(64) for _ in range(n_res_blocks)])
            self.trunk = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64)
            )
            # 4x upsampling via two 2x PixelShuffle layers
            self.upsample = nn.Sequential(
                nn.Conv2d(64, 256, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.PReLU(),
                nn.Conv2d(64, 256, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.PReLU()
            )
            self.tail = nn.Conv2d(64, out_c, kernel_size=9, padding=4)
            nn.init.zeros_(self.tail.weight)
            nn.init.zeros_(self.tail.bias)

        def forward(self, x):
            skip = torch.nn.functional.interpolate(x, scale_factor=self.scale, mode='bicubic', align_corners=False)
            h = self.head(x)
            b = self.body(h)
            t = self.trunk(b) + h
            u = self.upsample(t)
            residual = self.tail(u)
            out = skip + residual
            return torch.clamp(out, 0.0, 1.0)

except ImportError:
    nn = None
    torch = None
    ResidualBlock = None
    SRGANGenerator = None


class SentinelSRGAN(BaseSuperResolutionModel):
    def __init__(self, scale_factor: int = 4, in_channels: int = 4, weights_path: Optional[str] = None):
        super().__init__(name="Sentinel-2 SRGAN", scale_factor=scale_factor, in_channels=in_channels)
        if weights_path is None:
            # Auto-discover available checkpoints
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            candidates = [
                os.path.join(base_dir, "checkpoints", "srgan_sentinel2_x4.pth"),
                os.path.join(base_dir, "checkpoints", "generator_best.pth")
            ]
            for c in candidates:
                if os.path.exists(c):
                    weights_path = c
                    break
        self.weights_path = weights_path
        self.model = None
        self._init_network()

    def _init_network(self):
        """Initializes PyTorch deep residual model if available and loads pretrained weights."""
        if torch is not None and SRGANGenerator is not None:
            if self.weights_path and os.path.exists(self.weights_path):
                try:
                    checkpoint = torch.load(self.weights_path, map_location="cpu")
                    if isinstance(checkpoint, dict) and "state_dict" in checkpoint and "in_channels" in checkpoint:
                        ckpt_channels = checkpoint["in_channels"]
                        if ckpt_channels != self.in_channels:
                            print(f"[SRGAN Warning] Mismatch! Checkpoint built for {ckpt_channels} bands, input has {self.in_channels} bands.")
                            self.weights_path = None
                            return
                        self.model = SRGANGenerator(in_c=ckpt_channels, out_c=ckpt_channels, scale=self.scale_factor)
                        self.model.load_state_dict(checkpoint["state_dict"])
                        print(f"[SRGAN] Loaded weights from: {self.weights_path} ({ckpt_channels} channels)")
                    else:
                        # Legacy format fallback
                        # Strip "module." prefix if saved with DataParallel
                        new_state_dict = {}
                        for k, v in checkpoint.items():
                            name = k[7:] if k.startswith("module.") else k
                            new_state_dict[name] = v
                        checkpoint = new_state_dict
                        
                        ckpt_channels = None
                        for v in checkpoint.values():
                            if getattr(v, "ndim", 0) == 4:
                                ckpt_channels = v.shape[1]
                                break
                                
                        if ckpt_channels is None:
                            print("[SRGAN Warning] Could not infer channels from legacy checkpoint. Fallback.")
                            self.weights_path = None
                            return
                        
                        if self.in_channels != ckpt_channels:
                            print(f"[SRGAN Warning] Legacy checkpoint requires {ckpt_channels} bands, input has {self.in_channels}. Fallback.")
                            self.weights_path = None
                            return
                        
                        self.model = SRGANGenerator(in_c=ckpt_channels, out_c=ckpt_channels, scale=self.scale_factor)
                        self.model.load_state_dict(checkpoint)
                        print(f"[SRGAN] Loaded legacy weights from: {self.weights_path} (inferred {ckpt_channels} channels)")
                except Exception as e:
                    print(f"[SRGAN Warning] Could not load state_dict from {self.weights_path}: {e}")
                    self.weights_path = None
            else:
                self.model = SRGANGenerator(in_c=self.in_channels, out_c=self.in_channels, scale=self.scale_factor)
            self.model.eval()
        else:
            self.model = None

    def predict_tile(self, tile: np.ndarray) -> np.ndarray:
        """
        Runs super-resolution on a single multi-spectral tile.
        Input: (C, H, W) float32 in [0, 1]
        Output: (C, H*4, W*4) float32 in [0, 1]
        """
        channels, h, w = tile.shape

        if self.model is not None:
            try:
                # Adapt channel count for 4-band deep network
                adapted_tile = tile
                was_3_channel = False
                if channels == 3:
                    was_3_channel = True
                    # Estimate 4th NIR band using remote sensing proxy (or replicate)
                    nir_proxy = np.clip(1.1 * tile[1] - 0.1 * tile[0], 0.0, 1.0)
                    adapted_tile = np.concatenate([tile, np.expand_dims(nir_proxy, axis=0)], axis=0)

                with torch.no_grad():
                    tensor = torch.from_numpy(adapted_tile).unsqueeze(0).float()
                    sr_tensor = self.model(tensor)
                    sr_np = sr_tensor.squeeze(0).cpu().numpy()
                    if was_3_channel:
                        return sr_np[:3]
                    return sr_np
            except Exception as e:
                print(f"[SRGAN Inference Warning] PyTorch inference encountered: {e}. Falling back to analytical filter.")

        # Analytical Filter Fallback:
        # High-order bicubic interpolation + unsharp masking high-frequency detail synthesis
        target_h = h * self.scale_factor
        target_w = w * self.scale_factor
        sr_out = np.zeros((channels, target_h, target_w), dtype=np.float32)

        from scipy.ndimage import map_coordinates, gaussian_filter

        y_coords = np.linspace(0, h - 1, target_h)
        x_coords = np.linspace(0, w - 1, target_w)
        grid_y, grid_x = np.meshgrid(y_coords, x_coords, indexing='ij')

        for c in range(channels):
            band = tile[c]
            upscaled = map_coordinates(band, [grid_y, grid_x], order=3, mode='reflect')
            blurred = gaussian_filter(upscaled, sigma=1.0)
            high_pass = upscaled - blurred
            enhanced = upscaled + 1.25 * high_pass
            sr_out[c] = np.clip(enhanced, 0.0, 1.0)

        return sr_out