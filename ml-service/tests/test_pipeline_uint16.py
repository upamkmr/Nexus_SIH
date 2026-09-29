import os
import numpy as np
import rasterio
from rasterio.transform import from_origin
from pipeline import SatelliteSuperResolutionPipeline
from preprocessing.normalize import normalize_sentinel2
from models.base_model import BicubicBaseline
from models.transformer.swin_ir import HighFrequencySplineRefiner
from postprocessing.geotiff_export import GeoTiffExporter

import tempfile
import shutil
import pytest

@pytest.fixture
def synthetic_s2_tiff(tmp_path):
    file_path = str(tmp_path / "test_s2_uint16.tif")
    data = np.random.randint(1000, 5000, size=(4, 64, 64), dtype=np.uint16)
    data[:, 0, 0] = 0
    data[:, 1, 1] = 12000
    
    transform = from_origin(500000.0, 4600000.0, 10.0, 10.0)
    
    with rasterio.open(
        file_path, 'w',
        driver='GTiff',
        height=64, width=64,
        count=4, dtype='uint16',
        crs='EPSG:32643',
        transform=transform
    ) as dst:
        dst.update_tags(PROCESSING_BASELINE="4.00", BOA_ADD_OFFSET="1000")
        dst.write(data)
        
    return file_path

def test_pipeline_uint16_end_to_end(synthetic_s2_tiff, tmp_path):
    pipeline = SatelliteSuperResolutionPipeline()
    pipeline.output_dir = str(tmp_path)
    
    result = pipeline.run(
        image_path=synthetic_s2_tiff,
        model_name="swin_ir",
        scale_factor=4,
        estimate_uncertainty=False
    )
    
    assert result["status"] == "success"
    out_tif = os.path.join(str(tmp_path), os.path.basename(result["geotiff_url"]))
    out_png = os.path.join(str(tmp_path), os.path.basename(result["preview_url"]))
    input_png = os.path.join(str(tmp_path), os.path.basename(result["input_preview_url"]))
    
    assert os.path.exists(out_tif), "GeoTIFF not saved"
    assert os.path.exists(out_png), "PNG preview not saved"
    assert os.path.exists(input_png), "Input preview not saved"
    
    with rasterio.open(out_tif) as src:
        assert src.count == 4, "Output does not have 4 bands"
        assert src.crs.to_string() == "EPSG:32643", "Output CRS mismatch"
        # 10.0 / 4 = 2.5
        assert src.transform[0] == 2.5, "Transform scale X mismatch"
        assert src.transform[4] == -2.5, "Transform scale Y mismatch"

@pytest.fixture
def synthetic_s2_untagged_tiff(tmp_path):
    file_path = str(tmp_path / "test_s2_untagged.tif")
    data = np.random.randint(1000, 5000, size=(4, 64, 64), dtype=np.uint16)
    
    transform = from_origin(500000.0, 4600000.0, 10.0, 10.0)
    
    with rasterio.open(
        file_path, 'w',
        driver='GTiff',
        height=64, width=64,
        count=4, dtype='uint16',
        crs='EPSG:32643',
        transform=transform
    ) as dst:
        # Deliberately do NOT add BOA_ADD_OFFSET tag
        dst.write(data)
        
    return file_path

def test_pipeline_untagged_warning(synthetic_s2_untagged_tiff, tmp_path):
    pipeline = SatelliteSuperResolutionPipeline()
    pipeline.output_dir = str(tmp_path)
    
    result = pipeline.run(
        image_path=synthetic_s2_untagged_tiff,
        model_name="swin_ir",
        scale_factor=4,
        estimate_uncertainty=False
    )
    
    assert result["status"] == "success"
    warnings_list = result.get("warnings", [])
    assert len(warnings_list) > 0, "Expected a warning for untagged uint16, got none"
    
    warning_reported = warnings_list[0]
    print(f"REPORTED WARNING: {warning_reported}")
    assert "No BOA_ADD_OFFSET tag found" in warning_reported
