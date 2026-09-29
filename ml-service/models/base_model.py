"""
Base Super-Resolution Model Interface
Standardizes inference, tiling, and uncertainty sampling across all model architectures
(SRGAN, GeoDiffusion, SwinIR Transformer).
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from preprocessing.tiling import RasterTiler
from postprocessing.merge_tiles import TileMerger
from postprocessing.uncertainty import SpatialUncertaintyEstimator

class BaseSuperResolutionModel(ABC):
    def __init__(self, name: str, scale_factor: int = 4, in_channels: int = 4):
        self.name = name
        self.scale_factor = scale_factor
        self.in_channels = in_channels

    @abstractmethod
    def predict_tile(self, tile: np.ndarray) -> np.ndarray:
        """
        Runs super-resolution on a single tile patch.
        Input: (C, H, W)
        Output: (C, H * scale_factor, W * scale_factor)
        """
        pass

    def predict_scene(
        self,
        image: np.ndarray,
        tile_size: int = 256,
        overlap: int = 32
    ) -> np.ndarray:
        """
        Processes a full satellite scene by tiling, running inference, and seamlessly merging.
        For scenes <= 384x384, executes single-pass inference directly to eliminate tiling overhead.
        """
        c, h, w = image.shape
        if h <= 384 and w <= 384:
            return self.predict_tile(image)

        tiler = RasterTiler(tile_size=tile_size, overlap=overlap)
        tiles, meta = tiler.split_into_tiles(image)

        sr_tiles = []
        for t in tiles:
            sr_tile = self.predict_tile(t)
            sr_tiles.append(sr_tile)

        merged = TileMerger.merge_super_resolved_tiles(sr_tiles, meta, scale_factor=self.scale_factor)
        return merged

    def predict_with_uncertainty(
        self,
        image: np.ndarray,
        num_samples: int = 2,
        tile_size: int = 256,
        overlap: int = 32
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Runs Test-Time Augmentation (TTA) and multi-pass spatial ensemble inference
        to quantify pixel-wise reconstruction uncertainty.
        
        Tests geometric invariance and perturbation sensitivity:
        Pass 1: Canonical input
        Pass 2: Horizontal flip -> Predict -> Inverse flip
        (Optional Pass 3 & 4 if num_samples >= 4)
        
        Variance across passes highlights inferred/hallucinated sub-pixel details,
        while invariant regions (uniform water, bare ground) maintain low variance.
        
        Returns:
            mean_prediction: (C, H * scale_factor, W * scale_factor)
            uncertainty_map: (H * scale_factor, W * scale_factor) in [0, 1]
            stats: Summary uncertainty metrics
        """
        sample_runs = []

        # Pass 1: Canonical inference
        run_orig = self.predict_scene(image, tile_size=tile_size, overlap=overlap)
        sample_runs.append(run_orig)

        # Pass 2: Horizontal flip TTA (with inverse reflection)
        try:
            img_h = np.flip(image, axis=2).copy()
            pred_h = self.predict_scene(img_h, tile_size=tile_size, overlap=overlap)
            run_h = np.flip(pred_h, axis=2).copy()
            sample_runs.append(run_h)
        except Exception:
            pass

        if num_samples >= 3:
            # Pass 3: Vertical flip TTA
            try:
                img_v = np.flip(image, axis=1).copy()
                pred_v = self.predict_scene(img_v, tile_size=tile_size, overlap=overlap)
                run_v = np.flip(pred_v, axis=1).copy()
                sample_runs.append(run_v)
            except Exception:
                pass

        if num_samples >= 4:
            # Pass 4: 90-degree rotation TTA
            try:
                img_rot = np.rot90(image, 1, (1, 2)).copy()
                pred_rot = self.predict_scene(img_rot, tile_size=tile_size, overlap=overlap)
                run_rot = np.rot90(pred_rot, -1, (1, 2)).copy()
                sample_runs.append(run_rot)
            except Exception:
                pass

        # Ensure all runs match dimensions exactly
        target_shape = run_orig.shape
        valid_runs = [r for r in sample_runs if r.shape == target_shape]

        if len(valid_runs) < 2:
            valid_runs = [run_orig]

        # Ensemble mean prediction (standard super-resolution TTA boosting)
        mean_pred = np.mean(np.stack(valid_runs, axis=0), axis=0).astype(np.float32)

        # Quantify pixel-wise variance
        u_map, stats = SpatialUncertaintyEstimator.compute_variance_map(valid_runs)

        return mean_pred, u_map, stats


class BicubicBaseline(BaseSuperResolutionModel):
    """
    Certified Bicubic Interpolation Baseline.
    Standard remote sensing reference benchmark used to establish
    baseline metrics (PSNR, SSIM, SAM, ERGAS) that ML generative models must beat.
    """
    def __init__(self, scale_factor: int = 4, in_channels: int = 4):
        super().__init__(name="Bicubic Baseline (Standard)", scale_factor=scale_factor, in_channels=in_channels)

    def predict_tile(self, tile: np.ndarray) -> np.ndarray:
        channels, h, w = tile.shape
        target_h = h * self.scale_factor
        target_w = w * self.scale_factor
        sr_out = np.zeros((channels, target_h, target_w), dtype=np.float32)

        from scipy.ndimage import map_coordinates
        y_coords = np.linspace(0, h - 1, target_h)
        x_coords = np.linspace(0, w - 1, target_w)
        grid_y, grid_x = np.meshgrid(y_coords, x_coords, indexing='ij')

        for c in range(channels):
            sr_out[c] = np.clip(map_coordinates(tile[c], [grid_y, grid_x], order=3, mode='reflect'), 0.0, 1.0)

        return sr_out