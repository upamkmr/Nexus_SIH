"""
Scientific Remote Sensing Super-Resolution Evaluation Metrics
Implements standard Earth Observation metrics:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- SAM (Spectral Angle Mapper) - Validates multi-spectral color preservation
- ERGAS (Relative Dimensionless Global Error in Synthesis)
- RMSE (Root Mean Square Error)
"""

import numpy as np
from typing import Dict, Any, Optional

class RemoteSensingMetrics:
    @staticmethod
    def calculate_psnr(target: np.ndarray, reference: np.ndarray, max_val: float = 1.0) -> float:
        """Computes Peak Signal-to-Noise Ratio in decibels (dB)."""
        mse = np.mean((target - reference) ** 2)
        if mse == 0:
            return float("inf")
        return float(20 * np.log10(max_val / np.sqrt(mse)))

    @staticmethod
    def calculate_ssim(target: np.ndarray, reference: np.ndarray, max_val: float = 1.0) -> float:
        """Computes Structural Similarity Index (SSIM) between target and reference arrays."""
        c1 = (0.01 * max_val) ** 2
        c2 = (0.03 * max_val) ** 2

        mu_x = np.mean(target)
        mu_y = np.mean(reference)

        sigma_x = np.var(target)
        sigma_y = np.var(reference)
        sigma_xy = np.mean((target - mu_x) * (reference - mu_y))

        numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
        return float(numerator / (denominator + 1e-7))

    @staticmethod
    def calculate_sam(target: np.ndarray, reference: np.ndarray) -> float:
        """
        Spectral Angle Mapper (SAM):
        Computes the spectral angle between two multi-spectral vectors across all bands.
        Output in degrees. A SAM < 3.0° signifies near-lossless spectral fidelity.
        
        Expected shape: (C, H, W)
        """
        if target.ndim != 3 or reference.ndim != 3:
            return 0.0

        channels, height, width = target.shape
        t_flat = target.reshape(channels, -1) # (C, N)
        r_flat = reference.reshape(channels, -1) # (C, N)

        dot_product = np.sum(t_flat * r_flat, axis=0)
        norm_t = np.linalg.norm(t_flat, axis=0)
        norm_r = np.linalg.norm(r_flat, axis=0)

        denominator = norm_t * norm_r
        valid_mask = denominator > 1e-6

        cos_angles = np.zeros(dot_product.shape[0])
        cos_angles[valid_mask] = dot_product[valid_mask] / denominator[valid_mask]
        cos_angles = np.clip(cos_angles, -1.0, 1.0)

        sam_rad = np.mean(np.arccos(cos_angles[valid_mask])) if np.any(valid_mask) else 0.0
        sam_deg = float(np.degrees(sam_rad))
        return sam_deg

    @staticmethod
    def calculate_ergas(target: np.ndarray, reference: np.ndarray, scale_factor: int = 4) -> float:
        """
        Relative Dimensionless Global Error in Synthesis (ERGAS):
        Standard metric in satellite pansharpening and super-resolution.
        Values < 3.0 denote acceptable synthesis quality.
        """
        if target.ndim != 3:
            return 0.0

        channels = target.shape[0]
        sum_err = 0.0

        for c in range(channels):
            rmse_c = np.sqrt(np.mean((target[c] - reference[c]) ** 2))
            mean_c = np.mean(reference[c]) + 1e-6
            sum_err += (rmse_c / mean_c) ** 2

        ergas = 100.0 * (1.0 / scale_factor) * np.sqrt(sum_err / channels)
        return float(ergas)

    @classmethod
    def evaluate_all(
        cls,
        target: np.ndarray,
        reference: np.ndarray,
        scale_factor: int = 4
    ) -> Dict[str, Any]:
        """Runs full suite of remote sensing validation metrics against reference."""
        # Align dimensions if slight mismatch
        t = target
        r = reference
        if t.shape != r.shape:
            min_c = min(t.shape[0], r.shape[0])
            min_h = min(t.shape[1], r.shape[1])
            min_w = min(t.shape[2], r.shape[2])
            t = t[:min_c, :min_h, :min_w]
            r = r[:min_c, :min_h, :min_w]

        return {
            "has_reference": True,
            "psnr": round(cls.calculate_psnr(t, r), 2),
            "ssim": round(cls.calculate_ssim(t, r), 4),
            "sam_deg": round(cls.calculate_sam(t, r), 2),
            "ergas": round(cls.calculate_ergas(t, r, scale_factor), 2),
            "rmse": round(float(np.sqrt(np.mean((t - r) ** 2))), 4)
        }

    @classmethod
    def evaluate_with_baseline(
        cls,
        target: np.ndarray,
        lr_input: np.ndarray,
        reference: np.ndarray,
        scale_factor: int = 4
    ) -> Dict[str, Any]:
        """
        Evaluates model output against certified HR reference raster AND benchmarks
        directly against a standard Bicubic baseline on the same input.
        Reports exact metric deltas (+Δ dB PSNR, +Δ SSIM) that the model must beat.
        """
        model_scores = cls.evaluate_all(target, reference, scale_factor=scale_factor)

        # Generate bicubic baseline
        channels, lr_h, lr_w = lr_input.shape
        target_h, target_w = reference.shape[1], reference.shape[2]
        bicubic_upscaled = np.zeros((channels, target_h, target_w), dtype=np.float32)

        from scipy.ndimage import map_coordinates
        y_coords = np.linspace(0, lr_h - 1, target_h)
        x_coords = np.linspace(0, lr_w - 1, target_w)
        grid_y, grid_x = np.meshgrid(y_coords, x_coords, indexing='ij')

        for c in range(channels):
            bicubic_upscaled[c] = np.clip(
                map_coordinates(lr_input[c], [grid_y, grid_x], order=3, mode='reflect'),
                0.0, 1.0
            )

        baseline_scores = cls.evaluate_all(bicubic_upscaled, reference, scale_factor=scale_factor)

        psnr_delta = round(model_scores["psnr"] - baseline_scores["psnr"], 2)
        ssim_delta = round(model_scores["ssim"] - baseline_scores["ssim"], 4)
        sam_delta = round(baseline_scores["sam_deg"] - model_scores["sam_deg"], 2) # positive = model has smaller spectral distortion

        return {
            "has_reference": True,
            "reference_type": "Paired High-Resolution Ground Truth",
            "model_metrics": model_scores,
            "bicubic_baseline": baseline_scores,
            "psnr": model_scores["psnr"],
            "ssim": model_scores["ssim"],
            "sam_deg": model_scores["sam_deg"],
            "ergas": model_scores["ergas"],
            "baseline_comparison": {
                "psnr_delta_db": psnr_delta,
                "ssim_delta": ssim_delta,
                "sam_delta_deg": sam_delta,
                "beats_baseline": bool(psnr_delta >= 0.0 and ssim_delta >= 0.0)
            }
        }

    @staticmethod
    def evaluate_no_reference(target: np.ndarray, lr_input: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Provides objective no-reference spatial sharpness and spectral preservation metrics
        when no paired high-resolution ground truth image is provided.
        Never fabricates reference-based numbers like PSNR or SSIM.
        """
        # Tenengrad Gradient Density (High-frequency sharpness indicator)
        gray = np.mean(target, axis=0) if target.ndim == 3 else target
        gy, gx = np.gradient(gray)
        tenengrad = float(np.mean(gx**2 + gy**2))

        # Spatial Frequency (SF)
        row_freq = np.sqrt(np.mean(np.diff(gray, axis=0) ** 2))
        col_freq = np.sqrt(np.mean(np.diff(gray, axis=1) ** 2))
        spatial_freq = float(np.sqrt(row_freq**2 + col_freq**2))

        result = {
            "has_reference": False,
            "reference_status": "No high-resolution reference raster provided. Reference-based metrics (PSNR, SSIM, SAM, ERGAS) require paired ground truth.",
            "psnr": None,
            "ssim": None,
            "sam_deg": None,
            "ergas": None,
            "no_reference_assessment": {
                "tenengrad_sharpness_density": round(tenengrad, 6),
                "spatial_frequency": round(spatial_freq, 4),
                "dynamic_range": f"{float(target.min()):.3f} - {float(target.max()):.3f}"
            }
        }

        # Spectral NDVI preservation check if NIR and Red bands exist (Bands 4 and 8 in Sentinel-2)
        if target.ndim == 3 and target.shape[0] >= 4 and lr_input is not None and lr_input.shape[0] >= 4:
            # Sentinel-2: Band 2=Blue, 3=Green, 4=Red, 8=NIR
            nir_sr, red_sr = target[3], target[2]
            ndvi_sr = (nir_sr - red_sr) / (nir_sr + red_sr + 1e-6)

            nir_lr, red_lr = lr_input[3], lr_input[2]
            ndvi_lr = (nir_lr - red_lr) / (nir_lr + red_lr + 1e-6)

            from scipy.ndimage import zoom
            z_factors = (ndvi_sr.shape[0] / ndvi_lr.shape[0], ndvi_sr.shape[1] / ndvi_lr.shape[1])
            ndvi_lr_upscaled = zoom(ndvi_lr, z_factors, order=1)

            ndvi_mae = float(np.mean(np.abs(ndvi_sr - ndvi_lr_upscaled)))
            result["no_reference_assessment"]["ndvi_spectral_consistency_error"] = round(ndvi_mae, 4)

        return result