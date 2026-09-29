"""
Remote Sensing Radiometric Normalization for Sentinel-2 MSI L2A Imagery
Handles Bottom-of-Atmosphere (BOA) surface reflectance scaling and percentile contrast stretching.
"""

import numpy as np
from typing import Tuple, Optional

def normalize_sentinel2(
    arr: np.ndarray,
    method: str = "auto",
    percentiles: Tuple[float, float] = (2.0, 98.0),
    tags: Optional[dict] = None,
    baseline_offset: Optional[float] = None
) -> Tuple[np.ndarray, dict]:
    """
    Normalizes Sentinel-2 multi-spectral data into standard [0.0, 1.0] float32 arrays
    while preserving radiometric and physical surface reflectance integrity.
    """
    original_dtype = str(arr.dtype)
    arr = arr.astype(np.float32)
    stats = {"method": method, "original_dtype": original_dtype}

    if method == "auto":
        if "uint8" in original_dtype:
            method = "uint8"
        else:
            max_val = float(np.max(arr))
            if max_val > 255.0:
                method = "reflectance"
            elif max_val > 1.0:
                method = "uint8"
            else:
                method = "passthrough"
        stats["detected_method"] = method

    if method == "reflectance":
        offset = 0.0
        if baseline_offset is not None:
            offset = baseline_offset
        elif tags is not None and "BOA_ADD_OFFSET" in tags:
            offset = float(tags["BOA_ADD_OFFSET"])
        elif "uint16" in original_dtype:
            offset = 1000.0
            warn_msg = "No BOA_ADD_OFFSET tag found in uint16 GeoTIFF. Assuming newer baseline (>= 04.00) with offset=1000."
            import warnings
            warnings.warn(warn_msg)
            stats["warning"] = warn_msg

        # Treat DN == 0 as NoData (or 0 reflectance)
        mask = arr > 0
        norm_arr = np.zeros_like(arr)
        norm_arr[mask] = np.clip((arr[mask] - offset) / 10000.0, 0.0, 1.0)
        
        stats["offset"] = offset
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