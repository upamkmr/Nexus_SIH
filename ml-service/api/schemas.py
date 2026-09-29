from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum

class ModelArchitecture(str, Enum):
    SRGAN = "srgan"
    SWIN_IR = "swin_ir"
    BICUBIC = "bicubic"

class PredictRequest(BaseModel):
    image_path: str = Field(..., description="Absolute or relative path to the Sentinel-2 image")
    reference_path: Optional[str] = Field(None, description="Optional path to paired High-Resolution reference raster (e.g. SPOT 1.5m / aerial / ground truth)")
    run_wald_validation: bool = Field(default=False, description="Run Wald degradation protocol (benchmark 40m -> 10m against original Sentinel-2)")
    output_filename: Optional[str] = Field(None, description="Custom name for the super-resolved output")
    model_type: ModelArchitecture = Field(default=ModelArchitecture.SRGAN, description="Selected super-resolution model")
    scale_factor: int = Field(default=4, ge=2, le=8, description="Spatial scale factor (e.g. 4 for 10m -> 2.5m)")
    bands: List[str] = Field(default=["B04", "B03", "B02"], description="Bands to process (RGB or RGB+NIR)")
    estimate_uncertainty: bool = Field(default=True, description="Whether to produce uncertainty variance map via TTA ensemble")
    preserve_georeference: bool = Field(default=True, description="Preserve CRS and Affine transform in output GeoTIFF")
    baseline_offset: Optional[float] = Field(None, description="Force a specific BOA DN offset if tags are missing")

class UncertaintySummary(BaseModel):
    mean_uncertainty: float
    max_uncertainty: float
    high_uncertainty_coverage_pct: float
    uncertainty_map_path: Optional[str] = None

class PredictResponse(BaseModel):
    status: str
    job_id: Optional[str] = None
    original_resolution: str = "10.0m"
    target_resolution: str = "2.5m"
    model_used: str
    output_path: str
    preview_url: Optional[str] = None
    geotiff_url: Optional[str] = None
    input_preview_url: Optional[str] = None
    uncertainty_map_url: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    uncertainty: Optional[UncertaintySummary] = None
    used_synthetic_data: bool = False
    model_untrained: bool = False
    warnings: List[str] = []
    execution_time_seconds: float
    metadata: Dict[str, Any] = {}

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    device: str
    cuda_available: bool
    supported_models: List[str]
