"""
Composite Multi-Spectral Loss Functions for Satellite Super-Resolution
Balances:
- L1 Content loss (pixel-level fidelity)
- Adversarial loss (sharp textures)
- Spectral Angle Mapper (SAM) loss (protects multi-band color and NDVI ratios)
- Gradient/Edge loss (reconstructs narrow roads, coastlines, and field boundaries)
"""

import numpy as np

try:
    import torch
    import torch.nn as nn

    class SpectralAngleMapperLoss(nn.Module):
        """
        Differentiable Spectral Angle Mapper (SAM) loss for PyTorch.
        Computes the spectral angle between predicted and ground-truth pixel vectors across spectral bands.
        """
        def __init__(self, eps: float = 1e-7):
            super().__init__()
            self.eps = eps

        def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
            # sr, hr shape: (B, C, H, W)
            dot = torch.sum(sr * hr, dim=1) # (B, H, W)
            norm_sr = torch.norm(sr, p=2, dim=1)
            norm_hr = torch.norm(hr, p=2, dim=1)
            cos_theta = dot / (norm_sr * norm_hr + self.eps)
            cos_theta = torch.clamp(cos_theta, -1.0 + self.eps, 1.0 - self.eps)
            return torch.mean(torch.acos(cos_theta))

    class GradientEdgeLoss(nn.Module):
        """
        Calculates spatial gradient differences to enforce crisp object boundaries.
        """
        def __init__(self):
            super().__init__()

        def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
            dy_sr = torch.abs(sr[:, :, 1:, :] - sr[:, :, :-1, :])
            dx_sr = torch.abs(sr[:, :, :, 1:] - sr[:, :, :, :-1])
            dy_hr = torch.abs(hr[:, :, 1:, :] - hr[:, :, :-1, :])
            dx_hr = torch.abs(hr[:, :, :, 1:] - hr[:, :, :, :-1])
            return torch.mean(torch.abs(dy_sr - dy_hr)) + torch.mean(torch.abs(dx_sr - dx_hr))

except ImportError:
    torch = None
    nn = None
    SpectralAngleMapperLoss = None
    GradientEdgeLoss = None


def compute_spectral_angle_loss(sr_tensor, hr_tensor):
    """
    Penalizes deviations in the multi-spectral angle vector (NumPy).
    Enforces that ratio between B04 (Red) and B08 (NIR) remains physically consistent.
    """
    dot = np.sum(sr_tensor * hr_tensor, axis=0)
    norm_sr = np.linalg.norm(sr_tensor, axis=0)
    norm_hr = np.linalg.norm(hr_tensor, axis=0)
    cos = np.clip(dot / (norm_sr * norm_hr + 1e-6), -1.0, 1.0)
    return float(np.mean(np.arccos(cos)))


def compute_gradient_edge_loss(sr_tensor, hr_tensor):
    """Computes gradient difference between SR and HR along X and Y axes (NumPy)."""
    gy_sr, gx_sr = np.gradient(sr_tensor, axis=(-2, -1))
    gy_hr, gx_hr = np.gradient(hr_tensor, axis=(-2, -1))
    return float(np.mean(np.abs(gx_sr - gx_hr) + np.abs(gy_sr - gy_hr)))