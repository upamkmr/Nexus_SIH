import time
import os
from fastapi import APIRouter, HTTPException
from .schemas import (
    PredictRequest,
    PredictResponse,
    HealthResponse,
    UncertaintySummary,
    ModelArchitecture,
)
from pipeline import SatelliteSuperResolutionPipeline

router = APIRouter()
pipeline = SatelliteSuperResolutionPipeline()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    has_cuda = False
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if has_cuda else "CPU (PyTorch 2.x)"
    except ImportError:
        device_name = "CPU (PyTorch not available)"

    return HealthResponse(
        status="online",
        service="Nexus Super-Resolution ML Engine",
        version="1.0.0",
        device=device_name,
        cuda_available=has_cuda,
        supported_models=[m.value for m in ModelArchitecture]
    )

@router.get("/models")
async def list_models():
    return {
        "models": [
            {
                "id": "srgan",
                "name": "Sentinel-2 SRGAN (PyTorch)",
                "description": "Deep residual convolutional network with pixel-shuffle upsampling trained on multi-spectral satellite imagery.",
                "scale_factor": 4,
                "input_resolution": "10m",
                "output_resolution": "2.5m",
                "speed": "Fast (~0.6s/tile)",
                "status": "Operational (Checkpoint active)"
            },
            {
                "id": "swin_ir",
                "name": "High-Frequency Spline Refiner (Classical Baseline)",
                "description": "High-order spline interpolation with unsharp high-pass spatial detail synthesis. Pretrained SwinIR fine-tuning is on the roadmap.",
                "scale_factor": 4,
                "input_resolution": "10m",
                "output_resolution": "2.5m",
                "speed": "Very Fast (~0.3s/tile)",
                "status": "Operational (Analytical Baseline)"
            },
            {
                "id": "bicubic",
                "name": "Bicubic Interpolation Baseline (Reference Standard)",
                "description": "Standard spatial interpolation baseline used to compute objective delta scores (+Δ dB PSNR, +Δ SSIM).",
                "scale_factor": 4,
                "input_resolution": "10m",
                "output_resolution": "2.5m",
                "speed": "Ultra Fast (<0.1s/tile)",
                "status": "Operational (Benchmark Standard)"
            }
        ],
        "roadmap_models": [
            {
                "id": "diffusion",
                "name": "GeoDiffusion-SR",
                "description": "Latent diffusion model for remote sensing, fine-tuned on paired WorldStrat/SEN2VENµS datasets.",
                "status": "Research Roadmap"
            }
        ]
    }

@router.post("/predict", response_model=PredictResponse)
async def predict_super_resolution(req: PredictRequest):
    try:
        res = pipeline.run(
            image_path=req.image_path,
            model_name=req.model_type.value,
            scale_factor=req.scale_factor,
            estimate_uncertainty=req.estimate_uncertainty,
            reference_path=req.reference_path,
            run_wald_validation=req.run_wald_validation
        )

        uncertainty = None
        if res.get("uncertainty"):
            uncertainty = UncertaintySummary(
                mean_uncertainty=res["uncertainty"]["mean_uncertainty"],
                max_uncertainty=res["uncertainty"]["max_uncertainty"],
                high_uncertainty_coverage_pct=res["uncertainty"]["high_uncertainty_coverage_pct"],
                uncertainty_map_path=res.get("uncertainty_map_url")
            )

        return PredictResponse(
            status="success",
            job_id="job_" + str(int(time.time())),
            original_resolution=res["original_resolution"],
            target_resolution=res["target_resolution"],
            model_used=res["model_used"],
            output_path=res["output_path"],
            preview_url=res["preview_url"],
            geotiff_url=res.get("geotiff_url"),
            input_preview_url=res.get("input_preview_url"),
            uncertainty_map_url=res.get("uncertainty_map_url"),
            metrics=res.get("metrics"),
            uncertainty=uncertainty,
            execution_time_seconds=res["execution_time_seconds"],
            metadata=res["metadata"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline inference failed: {str(e)}")