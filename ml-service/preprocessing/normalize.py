"""
Remote Sensing Radiometric Normalization for Sentinel-2 MSI L2A Imagery
Handles Bottom-of-Atmosphere (BOA) surface reflectance scaling and percentile contrast stretching.
"""

import numpy as np
from typing import Tuple, Optional

def normalize_sentinel2(
    arr: np.ndarray,
    method: str = "auto",
    percentiles: Tuple[float, float] = (2.0, 98.0)
) -> Tuple[np.ndarray, dict]:
    """
    Normalizes Sentinel-2 multi-spectral data into standard [0.0, 1.0] float32 arrays
    while preserving radiometric and physical surface reflectance integrity.
    
    Sentinel-2 L2A Bottom-of-Atmosphere (BOA) products use a quantification value of 10000
    (DN = Reflectance * 10000). A value of 10000 corresponds to 100% surface reflectance (1.0).
    
    Args:
        arr: Input array of shape (C, H, W) or (H, W) or (H, W, C)
        method: "auto", "reflectance" (divide by 10000 DN / 255 uint8), or "percentile" (display only)
        percentiles: Low and high percentiles (only used if method="percentile")
    
    Returns:
        normalized_array (float32 in [0, 1]), stats_dict for exact radiometric inversion
    """
    arr = arr.astype(np.float32)
    stats = {"method": method, "original_dtype": str(arr.dtype)}

    if method == "auto":
        # Detect radiometric scale:
        # Sentinel-2 L2A 12-bit DNs typically exceed 255 and max out around 10000
        max_val = float(np.max(arr))
        if max_val > 255.0:
            method = "reflectance"
        elif max_val > 1.0:
            method = "uint8"
        else:
            method = "passthrough"
        stats["detected_method"] = method

    if method == "reflectance":
        # Physical Sentinel-2 L2A BOA reflectance quantification factor: 10000
        # Clip to [0.0, 1.0] to maintain physically valid reflectance range
        norm_arr = np.clip(arr / 10000.0, 0.0, 1.0)
        stats["scale_factor"] = 10000.0
    elif method == "uint8":
        # Standard 8-bit image [0, 255]
        norm_arr = np.clip(arr / 255.0, 0.0, 1.0)
        stats["scale_factor"] = 255.0
    elif method == "passthrough":
        norm_arr = np.clip(arr, 0.0, 1.0)
        stats["scale_factor"] = 1.0
    elif method == "percentile":
        # Preserved ONLY for non-calibrated visual contrast preview enhancement
        if arr.ndim == 3 and arr.shape[0] in [1, 3, 4, 8, 12]:
            norm_bands = []
            mins, maxs = [], []
            for b in range(arr.shape[0]):
                p_low = np.percentile(arr[b], percentiles[0])
                p_high = np.percentile(arr[b], percentiles[1])
                p_high = max(p_high, p_low + 1e-5)
                stretched = np.clip((arr[b] - p_low) / (p_high - p_low), 0.0, 1.0)
                norm_bands.append(stretched)
                mins.append(float(p_low))
                maxs.append(float(p_high))
            norm_arr = np.stack(norm_bands, axis=0)
            stats["mins"] = mins
            stats["maxs"] = maxs
        else:
            p_low = np.percentile(arr, percentiles[0])
            p_high = np.percentile(arr, percentiles[1])
            p_high = max(p_high, p_low + 1e-5)
            norm_arr = np.clip((arr - p_low) / (p_high - p_low), 0.0, 1.0)
            stats["min"] = float(p_low)
            stats["max"] = float(p_high)
    else:
        p_min = float(arr.min())
        p_max = float(max(arr.max(), p_min + 1e-5))
        norm_arr = (arr - p_min) / (p_max - p_min)
        stats["min"] = p_min
        stats["max"] = p_max

    return norm_arr.astype(np.float32), stats

def denormalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Converts normalized [0, 1] float array to 8-bit [0, 255] RGB visualization array."""
    clipped = np.clip(arr * 255.0, 0.0, 255.0)
    return clipped.astype(np.uint8)

def denormalize_to_sentinel2_dn(arr: np.ndarray, stats: Optional[dict] = None) -> np.ndarray:
    """Restores normalized float array to original 16-bit ESA Sentinel-2 DN scale [0, 10000]."""
    if stats and stats.get("method") == "reflectance":
        return np.clip(arr * 10000.0, 0, 10000).astype(np.uint16)
    return np.clip(arr * 10000.0, 0, 65535).astype(np.uint16)