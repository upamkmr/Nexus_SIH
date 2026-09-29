"""
End-to-End Satellite Super-Resolution Pipeline Orchestrator
Executes: Load (Rasterio/PIL) -> Radiometric Normalization -> Model Inference -> Uncertainty Quantification
-> Seamless Tile Merge -> Georeferenced GeoTIFF & Preview Export -> Scientific Baseline Validation
"""

import os
import time
import numpy as np
from PIL import Image as PILImage
PILImage.MAX_IMAGE_PIXELS = None # Enable large satellite raster processing

from typing import Dict, Any, Optional, Tuple

from preprocessing.normalize import normalize_sentinel2, denormalize_to_uint8
from preprocessing.band_selection import extract_true_color, calculate_ndvi
from preprocessing.tiling import RasterTiler
from preprocessing.georeference import GeoReferenceHandler
from postprocessing.merge_tiles import TileMerger
from postprocessing.uncertainty import SpatialUncertaintyEstimator
from postprocessing.geotiff_export import GeoTiffExporter
from evaluation.metrics import RemoteSensingMetrics
from models.srgan.generator import SentinelSRGAN
from models.transformer.swin_ir import HighFrequencySplineRefiner, SwinIRRemoteSensing
from models.base_model import BicubicBaseline

class SatelliteSuperResolutionPipeline:
    def __init__(self, output_dir: Optional[str] = None):
        if output_dir is None:
            output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/outputs"))
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

    def load_raster(self, image_path: str, max_extent: int = 384) -> Tuple[np.ndarray, Dict[str, Any], bool]:
        """
        Returns (array, geo_meta, used_synthetic).
        used_synthetic=True when the real file could not be loaded.
        """
        if not os.path.exists(image_path):
            arr = self._generate_synthetic_sentinel2_scene()
            meta = GeoReferenceHandler.get_default_sentinel_metadata(arr.shape[1], arr.shape[2])
            return arr, meta, True

        # Try loading with rasterio for multi-spectral GIS rasters
        try:
            import rasterio
            with rasterio.open(image_path) as src:
                w, h = src.width, src.height
                meta = {
                    "crs": str(src.crs) if src.crs else "EPSG:32643",
                    "transform": src.transform,
                    "nodata": src.nodata,
                    "count": src.count,
                    "driver": src.driver,
                    "dtype": str(src.dtypes[0]),
                    "tags": src.tags()
                }

                if w > max_extent or h > max_extent:
                    # Read centered window of interest
                    col_off = max(0, (w - max_extent) // 2)
                    row_off = max(0, (h - max_extent) // 2)
                    read_w = min(w - col_off, max_extent)
                    read_h = min(h - row_off, max_extent)
                    from rasterio.windows import Window
                    win = Window(col_off, row_off, read_w, read_h)
                    arr = src.read(window=win)
                    meta["transform"] = rasterio.windows.transform(win, src.transform)
                else:
                    arr = src.read()

                # Ensure shape is (C, H, W)
                if arr.ndim == 2:
                    arr = np.expand_dims(arr, axis=0)

                return arr, meta, False
        except Exception as e:
            print(f"[Rasterio Warning] Could not open with rasterio: {e}. Trying PIL...")

        try:
            with PILImage.open(image_path) as img:
                w, h = img.size
                if w > max_extent or h > max_extent:
                    left = (w - max_extent) // 2
                    top = (h - max_extent) // 2
                    img = img.crop((left, top, left + max_extent, top + max_extent))
                
                arr = np.array(img)
                if arr.ndim == 2:
                    arr_chw = np.expand_dims(arr, axis=0)
                elif arr.ndim == 3:
                    # (H, W, C) -> (C, H, W)
                    arr_chw = np.transpose(arr[:, :, :4], (2, 0, 1))
                else:
                    arr_chw = np.expand_dims(arr, axis=0)

                meta = GeoReferenceHandler.get_default_sentinel_metadata(arr_chw.shape[1], arr_chw.shape[2])
                return arr_chw, meta, False
        except Exception as e:
            print(f"[Loader Warning] {e}. Falling back to synthetic scene.")
            arr = self._generate_synthetic_sentinel2_scene()
            meta = GeoReferenceHandler.get_default_sentinel_metadata(arr.shape[1], arr.shape[2])
            return arr, meta, True

    def _generate_synthetic_sentinel2_scene(self) -> np.ndarray:
        """Creates a 4-band synthetic 10m Sentinel-2 scene (256x256)."""
        h, w = 256, 256
        scene = np.zeros((4, h, w), dtype=np.float32)
        scene[0] = 0.15 # Blue
        scene[1] = 0.25 # Green
        scene[2] = 0.20 # Red
        scene[3] = 0.45 # NIR
        scene[1, 30:110, 20:100] = 0.40
        scene[3, 30:110, 20:100] = 0.75
        scene[2, 30:110, 20:100] = 0.12
        scene[:, 120:124, :] = 0.55
        scene[:, 150:190, 140:180] = 0.65
        noise = np.random.normal(0, 0.02, scene.shape).astype(np.float32)
        return np.clip(scene + noise, 0.0, 1.0)

    def run(
        self,
        image_path: str,
        model_name: str = "srgan",
        scale_factor: int = 4,
        estimate_uncertainty: bool = True,
        reference_path: Optional[str] = None,
        run_wald_validation: bool = False,
        tile_size: int = 256,
        overlap: int = 32,
        baseline_offset: Optional[float] = None
    ) -> Dict[str, Any]:
        start_time = time.time()

        # 1. Load raster and geospatial metadata
        raw_data, geo_meta, used_synthetic = self.load_raster(image_path)
        pipeline_warnings = []
        if used_synthetic:
            pipeline_warnings.append(
                "Input file could not be loaded — results are based on a synthetic Sentinel-2 scene and are NOT representative of real data."
            )
        channels, orig_h, orig_w = raw_data.shape

        # 2. Normalize reflectance (preserves Sentinel-2 BOA 10000 DN calibration)
        norm_data, norm_stats = normalize_sentinel2(
            raw_data, 
            tags=geo_meta.get("tags"),
            baseline_offset=baseline_offset
        )
        
        if norm_stats.get("warning"):
            pipeline_warnings.append(norm_stats["warning"])
        print(f"Applied BOA offset: {norm_stats.get('offset', 0.0)}")

        # 3. Model selection
        model_untrained = False
        model_key = model_name.lower().replace("-", "_")
        if model_key == "srgan":
            model = SentinelSRGAN(scale_factor=scale_factor, in_channels=channels)
            if getattr(model, 'weights_path', None) is None:
                model_untrained = True
                pipeline_warnings.append("SRGAN checkpoint not found. Model is untrained. Falling back to HighFrequencySplineRefiner.")
                model = HighFrequencySplineRefiner(scale_factor=scale_factor, in_channels=channels)
        elif model_key == "bicubic":
            model = BicubicBaseline(scale_factor=scale_factor, in_channels=channels)
        else:
            model = HighFrequencySplineRefiner(scale_factor=scale_factor, in_channels=channels)

        # 4. Super-Resolution Inference with TTA Uncertainty (2-pass TTA for high CPU throughput)
        if estimate_uncertainty:
            sr_result, u_map, u_stats = model.predict_with_uncertainty(
                norm_data,
                num_samples=2,
                tile_size=tile_size,
                overlap=overlap
            )
        else:
            sr_result = model.predict_scene(
                norm_data,
                tile_size=tile_size,
                overlap=overlap
            )
            u_map, u_stats = None, None

        # 5. Export Files with GeoReferencing & Visual Previews
        job_tag = int(time.time())
        
        # Export input preview for comparison first to capture 2-98 stretch bounds
        input_filename = f"input_raw_{job_tag}.png"
        input_filepath = os.path.join(self.output_dir, input_filename)
        _, stretch_bounds = GeoTiffExporter.export_preview_png(norm_data, input_filepath)
        
        sr_filename = f"enhanced_sr_{model_name}_{job_tag}.png"
        sr_filepath = os.path.join(self.output_dir, sr_filename)
        GeoTiffExporter.export_preview_png(sr_result, sr_filepath, vmin_vmax=stretch_bounds)

        # Export full GeoTIFF with CRS and updated affine geotransform (10m -> 2.5m)
        tif_filename = f"enhanced_sr_{model_name}_{job_tag}.tif"
        tif_filepath = os.path.join(self.output_dir, tif_filename)
        GeoTiffExporter.export_geotiff(
            sr_result,
            tif_filepath,
            geo_meta=geo_meta,
            uncertainty_band=u_map,
            scale_factor=scale_factor
        )

        u_filename = None
        if u_map is not None:
            u_rgb = SpatialUncertaintyEstimator.generate_heatmap_rgb(u_map)
            u_filename = f"uncertainty_heatmap_{job_tag}.png"
            u_filepath = os.path.join(self.output_dir, u_filename)
            GeoTiffExporter.export_preview_png(u_rgb, u_filepath, stretch_contrast=False)

        # 6. Scientific Validation & Baseline Comparison (NO fabricated noise simulation!)
        if reference_path and os.path.exists(reference_path):
            # True paired HR reference evaluation with bicubic baseline benchmarking
            ref_raw, ref_meta, _ = self.load_raster(reference_path)
            ref_norm, _ = normalize_sentinel2(ref_raw, tags=ref_meta.get("tags"))
            metrics = RemoteSensingMetrics.evaluate_with_baseline(
                sr_result,
                norm_data,
                ref_norm,
                scale_factor=scale_factor
            )
        elif run_wald_validation:
            # Wald Protocol: Degrade 10m Sentinel-2 input to 40m, super-resolve back to 10m,
            # and evaluate against original 10m input as the certified reference!
            from scipy.ndimage import zoom
            z_down = (1.0, 1.0 / scale_factor, 1.0 / scale_factor)
            wald_lr = zoom(norm_data, z_down, order=3)
            wald_sr = model.predict_scene(wald_lr, tile_size=tile_size, overlap=overlap)
            metrics = RemoteSensingMetrics.evaluate_with_baseline(
                wald_sr,
                wald_lr,
                norm_data,
                scale_factor=scale_factor
            )
            metrics["validation_protocol"] = "Wald Degradation Protocol (40m -> 10m benchmark against original Sentinel-2)"
        else:
            # Objective No-Reference Remote Sensing Quality Assessment
            metrics = RemoteSensingMetrics.evaluate_no_reference(sr_result, norm_data)

        exec_time = round(time.time() - start_time, 3)

        return {
            "status": "success",
            "model_used": model.name,
            "original_resolution": "10.0m",
            "target_resolution": f"{10.0 / scale_factor:.1f}m",
            "scale_factor": scale_factor,
            "output_path": sr_filepath,
            "geotiff_path": tif_filepath,
            "preview_url": f"/static/outputs/{sr_filename}",
            "geotiff_url": f"/static/outputs/{tif_filename}",
            "input_preview_url": f"/static/outputs/{input_filename}",
            "uncertainty_map_url": f"/static/outputs/{u_filename}" if u_filename else None,
            "metrics": metrics,
            "uncertainty": u_stats,
            "used_synthetic_data": used_synthetic,
            "model_untrained": model_untrained,
            "warnings": pipeline_warnings,
            "execution_time_seconds": exec_time,
            "metadata": {
                "source": "Copernicus Data Space Ecosystem (CDSE) Sentinel-2",
                "portal_url": "https://browser.dataspace.copernicus.eu",
                "bands_processed": ["B02", "B03", "B04", "B08"] if channels >= 4 else ["B04", "B03", "B02"],
                "target_gsd_meters": 10.0 / scale_factor,
                "input_dimensions": f"{orig_w}x{orig_h}",
                "output_dimensions": f"{orig_w * scale_factor}x{orig_h * scale_factor}",
                "tile_dimensions": {"tile_size": tile_size, "overlap": overlap},
                "crs": geo_meta.get("crs", "EPSG:32643"),
                "radiometric_normalization": norm_stats.get("detected_method", norm_stats.get("method"))
            }
        }