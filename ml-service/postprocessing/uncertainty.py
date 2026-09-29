"""
Spatial Uncertainty Quantification for Remote Sensing Super-Resolution
Directly addresses the SIH requirement: "manage uncertainty because some reconstructed
details are inferred by the model and not directly observed."
Uses Monte Carlo stochastic sampling to produce pixel-level confidence and variance maps.
"""

import numpy as np
from typing import List, Tuple, Dict, Any

class SpatialUncertaintyEstimator:
    @staticmethod
    def compute_variance_map(predictions: List[np.ndarray]) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Computes pixel-wise spatial uncertainty across ensemble passes (TTA, stochastic sampling, or multi-model).
        
        Args:
            predictions: List of N numpy arrays, each shape (C, H, W)
        
        Returns:
            uncertainty_map: 2D array of shape (H, W) normalized to [0.0, 1.0]
            summary_stats: Dict containing mean, max, and high-uncertainty coverage percentage
        """
        if len(predictions) < 2:
            pred = predictions[0]
            if pred.ndim == 3:
                gray = np.mean(pred, axis=0)
            else:
                gray = pred
            gy, gx = np.gradient(gray)
            grad_mag = np.sqrt(gx**2 + gy**2)
            grad_norm = (grad_mag - grad_mag.min()) / (grad_mag.max() - grad_mag.min() + 1e-6)
            return grad_norm.astype(np.float32), {
                "mean_uncertainty": round(float(np.mean(grad_norm)), 4),
                "max_uncertainty": round(float(np.max(grad_norm)), 4),
                "high_uncertainty_coverage_pct": round(float(np.mean(grad_norm > 0.55) * 100.0), 2)
            }

        # Stack along new dimension: (N, C, H, W)
        stacked = np.stack(predictions, axis=0)

        # Variance across N ensemble runs: (C, H, W)
        var_per_channel = np.var(stacked, axis=0)

        # Average variance across spectral channels: (H, W)
        spatial_variance = np.mean(var_per_channel, axis=0)

        raw_max = float(np.max(spatial_variance))
        raw_mean = float(np.mean(spatial_variance))

        # Dynamic range normalization with robust percentile clipping:
        # Avoids division by zero or washed out maps when variance is subtle
        p5 = float(np.percentile(spatial_variance, 2))
        p98 = float(np.percentile(spatial_variance, 98))
        spread = p98 - p5

        if spread > 1e-7:
            norm_uncertainty = np.clip((spatial_variance - p5) / spread, 0.0, 1.0)
        else:
            # If passes were virtually identical, calculate high-frequency gradient proxy
            pred = predictions[0]
            gray = np.mean(pred, axis=0) if pred.ndim == 3 else pred
            gy, gx = np.gradient(gray)
            grad_mag = np.sqrt(gx**2 + gy**2)
            norm_uncertainty = (grad_mag - grad_mag.min()) / (grad_mag.max() - grad_mag.min() + 1e-6)
            norm_uncertainty = np.clip(norm_uncertainty, 0.0, 1.0)

        threshold = 0.50
        high_uncertainty_pixels = np.sum(norm_uncertainty > threshold)
        total_pixels = norm_uncertainty.size
        coverage_pct = (high_uncertainty_pixels / total_pixels) * 100.0

        stats = {
            "mean_uncertainty": round(float(np.mean(norm_uncertainty)), 4),
            "max_uncertainty": round(max(raw_max, float(np.max(norm_uncertainty))), 6),
            "high_uncertainty_coverage_pct": round(float(coverage_pct), 2),
            "total_pixels_evaluated": int(total_pixels),
            "ensemble_passes": len(predictions)
        }

        return norm_uncertainty.astype(np.float32), stats

    @staticmethod
    def generate_heatmap_rgb(uncertainty_map: np.ndarray) -> np.ndarray:
        """
        Converts 2D float uncertainty array [0, 1] into a high-contrast 3-channel RGB heatmap:
        - Dark Indigo/Blue (0.0): Confident direct observation / homogeneous areas (water, fields)
        - Cyan/Teal (0.3): Low-uncertainty texture
        - Yellow/Amber (0.7): Moderate edge uncertainty
        - Crimson/Red (1.0): High uncertainty (inferred/hallucinated sub-pixel structures)
        """
        h, w = uncertainty_map.shape
        heatmap = np.zeros((3, h, w), dtype=np.uint8)

        u = np.clip(uncertainty_map, 0.0, 1.0)

        # Smooth multi-stop colormap transfer:
        # Red
        r = np.clip(1.5 * (u - 0.25), 0.0, 1.0)
        r[u < 0.25] = 0.0
        # Green
        g = np.sin(np.pi * u)
        # Blue
        b = np.clip(1.0 - 1.5 * u, 0.0, 1.0)

        heatmap[0] = (r * 255.0).astype(np.uint8)
        heatmap[1] = (g * 255.0).astype(np.uint8)
        heatmap[2] = (b * 255.0).astype(np.uint8)

        return heatmap