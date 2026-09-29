"""
Geospatial Metadata & Affine Transformation Management
Preserves CRS, BBOX, and adjusts pixel resolution from 10m down to <4m (2.5m)
for scientific GIS applications (QGIS, ArcGIS, Google Earth Engine).
"""

from typing import Dict, Any, Tuple, Optional

class GeoReferenceHandler:
    @staticmethod
    def adjust_geotransform_for_super_resolution(
        original_geotransform: Tuple[float, float, float, float, float, float],
        scale_factor: int = 4
    ) -> Tuple[float, float, float, float, float, float]:
        """
        Updates GDAL GeoTransform tuple (c, a, b, f, d, e):
        a = pixel width (e.g. 10m) -> becomes 10.0 / scale_factor (e.g. 2.5m)
        e = pixel height (e.g. -10m) -> becomes -10.0 / scale_factor (e.g. -2.5m)
        c = top-left X coordinate (unchanged)
        f = top-left Y coordinate (unchanged)
        b, d = rotation parameters (usually 0.0)
        """
        c, a, b, f, d, e = original_geotransform
        new_a = a / float(scale_factor)
        new_e = e / float(scale_factor)
        return (c, new_a, b, f, d, new_e)

    @staticmethod
    def adjust_affine_transform(
        affine_transform: Any,
        scale_factor: int = 4
    ) -> Any:
        """
        Adjusts a rasterio.transform.Affine object by scaling pixel dimensions.
        Affine(a, b, c, d, e, f):
        a = x resolution, e = y resolution (negative), c = x origin, f = y origin
        """
        try:
            from rasterio.transform import Affine
            if isinstance(affine_transform, Affine):
                return Affine(
                    affine_transform.a / float(scale_factor),
                    affine_transform.b,
                    affine_transform.c,
                    affine_transform.d,
                    affine_transform.e / float(scale_factor),
                    affine_transform.f
                )
        except ImportError:
            pass
        return affine_transform

    @staticmethod
    def get_default_sentinel_metadata(
        height: int,
        width: int,
        crs: str = "EPSG:32643",
        scale_factor: int = 4
    ) -> Dict[str, Any]:
        """Generates standard UTM georeferencing metadata for simulated or processed rasters."""
        input_res = 10.0
        output_res = input_res / float(scale_factor)

        gdal_gt = (700000.0, output_res, 0.0, 3100000.0, 0.0, -output_res)

        meta = {
            "crs": crs,
            "input_resolution_m": input_res,
            "output_resolution_m": output_res,
            "scale_factor": scale_factor,
            "geotransform": gdal_gt,
            "driver": "GTiff",
            "nodata": 0
        }

        try:
            from rasterio.transform import Affine
            meta["transform"] = Affine(output_res, 0.0, 700000.0, 0.0, -output_res, 3100000.0)
        except ImportError:
            pass

        return meta