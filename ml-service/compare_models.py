import os
from PIL import Image
from pipeline import SatelliteSuperResolutionPipeline
import warnings
warnings.filterwarnings("ignore")

pipeline = SatelliteSuperResolutionPipeline()
image_path = "C:/sih/Nexus_SIH/data/raw/sentinel2/TCI.tif"
scale_factor = 4
# We only have 3 bands in the TCI image (RGB). If we ask for 4, rasterio might fail.
# TCI has 3 bands.
bands = ["B04", "B03", "B02"]

results = {}
models = ["bicubic", "swin_ir", "srgan"] # swin_ir is mapped to HighFrequencySplineRefiner

from models.srgan.generator import SRGANGenerator
import torch
import torch.nn.functional as F
from preprocessing.normalize import normalize_sentinel2

for m in ["bicubic", "swin_ir"]:
    print(f"Running {m}...")
    res = pipeline.run(
        image_path=image_path,
        model_name=m,
        scale_factor=scale_factor,
        estimate_uncertainty=False,
    )
    results[m] = res
    print(f"{m} done.")

print("Running pure untrained srgan directly...")
import rasterio
with rasterio.open(image_path) as src:
    arr = src.read()
norm_arr, _ = normalize_sentinel2(arr)
# SRGAN expects [B, C, H, W]
t = torch.from_numpy(norm_arr).unsqueeze(0)
# TCI has 3 bands, SRGAN defaults to 4. We pass in_c=3, out_c=3
gen = SRGANGenerator(in_c=3, out_c=3, scale=scale_factor)
gen.eval()
with torch.no_grad():
    out = gen(t).squeeze(0).numpy()
# Save to disk
out_path_srgan = "C:/sih/Nexus_SIH/data/outputs/enhanced_srgan_pure.png"
from preprocessing.normalize import denormalize_to_uint8
out_img = denormalize_to_uint8(out).transpose(1,2,0)
Image.fromarray(out_img).save(out_path_srgan)

# Compute metrics
from evaluation.metrics import RemoteSensingMetrics
no_ref = RemoteSensingMetrics.evaluate_no_reference(out, norm_arr)
results["srgan_pure"] = {
    "output_path": out_path_srgan,
    "metrics": no_ref
}

models = ["bicubic", "swin_ir", "srgan_pure"]

print("\n=== METRICS TABLE ===")
print(f"{'Model':<15} | {'Tenengrad':<12} | {'Spatial Freq':<12} | {'NDVI Err':<10}")
print("-" * 55)
for m in models:
    no_ref = results[m]["metrics"]["no_reference_assessment"]
    t = no_ref.get("tenengrad_sharpness_density", 0)
    sf = no_ref.get("spatial_frequency", 0)
    ndvi = no_ref.get("ndvi_spectral_consistency_error", "N/A")
    print(f"{m:<15} | {t:<12.6f} | {sf:<12.6f} | {ndvi}")

# Save side by side
imgs = []
for m in models:
    path = results[m]["output_path"]
    imgs.append(Image.open(path))

total_width = sum(i.width for i in imgs)
max_height = max(i.height for i in imgs)

combined = Image.new('RGB', (total_width, max_height))
x_offset = 0
for im in imgs:
    combined.paste(im, (x_offset,0))
    x_offset += im.width

out_path = "C:/sih/Nexus_SIH/data/outputs/comparison_3_models.png"
combined.save(out_path)
print(f"\nSaved side-by-side comparison to {out_path}")
