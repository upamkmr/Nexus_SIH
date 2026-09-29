"""
High-Frequency Spline & Unsharp Masking Refinement (Classical Baseline)
Implements high-order bicubic spline interpolation paired with spatial high-pass
unsharp detail synthesis for remote sensing feature enhancement.
(Note: Pretrained Transformer SwinIR fine-tuning is documented in the research roadmap).
"""

import numpy as np
from typing import Optional
from ..base_model import BaseSuperResolutionModel

class HighFrequencySplineRefiner(BaseSuperResolutionModel):
    """
    Classical High-Frequency Spline and Edge Synthesis baseline.
    Serves as an analytical spatial filter baseline before deep neural inference.
    """
    def __init__(self, scale_factor: int = 4, in_channels: int = 4, weights_path: Optional[str] = None):
        super().__init__(name="High-Frequency Spline Refiner (Classical Baseline)", scale_factor=scale_factor, in_channels=in_channels)
        self.weights_path = weights_path

    def predict_tile(self, tile: np.ndarray) -> np.ndarray:
        """
        Runs high-order spatial spline reconstruction on a single tile with high-pass edge refinement.
        Preserves physical spectral balance while restoring edge contrast.
        """
        channels, h, w = tile.shape
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

            # Unsharp masking detail enhancement:
            # Preserves smooth gradients on agricultural fields while sharpening road edges
            fine_detail = upscaled - gaussian_filter(upscaled, sigma=0.8)
            refined = upscaled + 1.25 * fine_detail
            sr_out[c] = np.clip(refined, 0.0, 1.0)

        return sr_out

# Backward-compatibility alias
SwinIRRemoteSensing = HighFrequencySplineRefiner