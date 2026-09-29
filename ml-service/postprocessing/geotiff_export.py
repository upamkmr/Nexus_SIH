"""
GeoTIFF and Image Export Pipeline
Exports super-resolved products as multi-spectral GeoTIFFs with georeferencing
and PNG previews for web and dashboard consumption.
"""

import os
import numpy as np
from PIL import Image as PILImage
from typing import Optional, Dict, Any

class GeoTiffExporter:
    @staticmethod
    def export_preview_png(
        array: np.ndarray,
        output_filepath: str,
        is_normalized: bool = True
    ) -> str:
        """
        Exports a 3-channel (RGB) or 1-channel array as a viewable PNG image.
        array shape expected: (C, H, W) or (H, W)
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)

        if array.ndim == 3:
            if array.shape[0] >= 3:
                rgb = array[:3] # First 3 channels
            else:
                rgb = np.repeat(array[:1], 3, axis=0)
            
            if is_normalized:
                rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
            else:
                rgb = rgb.astype(np.uint8)
            
            # Transpose to (H, W, C) for PIL
            img_hwc = np.transpose(rgb, (1, 2, 0))
        elif array.ndim == 2:
            if is_normalized:
                img_hwc = np.clip(array * 255.0, 0, 255).astype(np.uint8)
            else:
                img_hwc = array.astype(np.uint8)
        else:
            raise ValueError(f"Unsupported array shape: {array.shape}")

        img = PILImage.fromarray(img_hwc)
        img.save(output_filepath, format="PNG")
        return output_filepath

    @staticmethod
    def export_geotiff(
        multi_band_array: np.ndarray,
        output_filepath: str,
        geo_meta: Optional[Dict[str, Any]] = None,
        uncertainty_band: Optional[np.ndarray] = None,
        scale_factor: int = 4
    ) -> str:
        """
        Exports the super-resolved array as a standard georeferenced GeoTIFF raster.
        Uses rasterio to write CRS, Affine geotransform (scaled from 10m to 2.5m),
        and includes multi-spectral bands plus the uncertainty variance channel.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)

        # Standardize input shape to (C, H, W)
        if multi_band_array.ndim == 2:
            arr_chw = np.expand_dims(multi_band_array, axis=0)
        else:
            arr_chw = multi_band_array

        channels, height, width = arr_chw.shape

        # Append uncertainty as auxiliary channel if provided
        band_names = []
        if channels == 1:
            band_names = ["Band_1"]
        elif channels == 3:
            band_names = ["B04_Red", "B03_Green", "B02_Blue"]
        elif channels >= 4:
            band_names = ["B02_Blue", "B03_Green", "B04_Red", "B08_NIR"][:channels]

        if uncertainty_band is not None:
            # Resize uncertainty map if shape mismatch
            u_band = uncertainty_band
            if u_band.shape != (height, width):
                from scipy.ndimage import zoom
                zoom_factors = (height / u_band.shape[0], width / u_band.shape[1])
                u_band = zoom(u_band, zoom_factors, order=1)
            u_expanded = np.expand_dims(u_band, axis=0)
            data_to_save = np.concatenate([arr_chw, u_expanded], axis=0).astype(np.float32)
            band_names.append("Uncertainty_Variance_Map")
        else:
            data_to_save = arr_chw.astype(np.float32)

        total_bands, out_h, out_w = data_to_save.shape

        # Attempt export with rasterio for true GIS interoperability
        try:
            import rasterio
            from rasterio.transform import Affine
            from preprocessing.georeference import GeoReferenceHandler

            crs = "EPSG:32643"
            transform = Affine(2.5, 0.0, 700000.0, 0.0, -2.5, 3100000.0)

            if geo_meta:
                if geo_meta.get("crs"):
                    crs = geo_meta["crs"]
                
                # Check for rasterio Affine transform in metadata
                if "transform" in geo_meta and isinstance(geo_meta["transform"], Affine):
                    orig_transform = geo_meta["transform"]
                    transform = GeoReferenceHandler.adjust_affine_transform(orig_transform, scale_factor=scale_factor)
                elif "geotransform" in geo_meta:
                    orig_gt = geo_meta["geotransform"]
                    adj_gt = GeoReferenceHandler.adjust_geotransform_for_super_resolution(orig_gt, scale_factor=scale_factor)
                    # GDAL (c, a, b, f, d, e) -> Affine(a, b, c, d, e, f)
                    transform = Affine(adj_gt[1], adj_gt[2], adj_gt[0], adj_gt[4], adj_gt[5], adj_gt[3])

            with rasterio.open(
                output_filepath,
                'w',
                driver='GTiff',
                height=out_h,
                width=out_w,
                count=total_bands,
                dtype='float32',
                crs=crs,
                transform=transform,
                nodata=0.0
            ) as dst:
                for idx in range(total_bands):
                    dst.write(data_to_save[idx], idx + 1)
                    if idx < len(band_names):
                        dst.set_band_description(idx + 1, band_names[idx])

            return output_filepath
        except Exception as e:
            print(f"[GeoTIFF Export Warning] Rasterio export failed ({e}). Falling back to tifffile.")

        # Fallback to tifffile
        try:
            import tifffile
            tifffile.imwrite(output_filepath, data_to_save)
            return output_filepath
        except ImportError:
            # Fallback to PIL standard TIFF
            if data_to_save.ndim == 3:
                rgb = data_to_save[:3] if data_to_save.shape[0] >= 3 else np.repeat(data_to_save[:1], 3, axis=0)
                hwc = np.transpose(rgb, (1, 2, 0))
                hwc_uint8 = np.clip(hwc * 255.0, 0, 255).astype(np.uint8)
                img = PILImage.fromarray(hwc_uint8)
            else:
                img = PILImage.fromarray(np.clip(data_to_save * 255.0, 0, 255).astype(np.uint8))
            img.save(output_filepath, format="TIFF")
            return output_filepath