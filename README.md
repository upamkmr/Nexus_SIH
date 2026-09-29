# 🛰️ Nexus SR | Sentinel-2 Satellite Super-Resolution Framework
**NTRO Problem Statement 26142**: *Deep Learning Based Super Resolution Mapping (SRM) from Medium Resolution Satellite Imageries*

An Earth Observation super-resolution framework engineered to enhance **10m Sentinel-2 multi-spectral imagery** into sharp **<4.0m spatial resolution products** (2.5m GSD target) with certified **radiometric integrity**, **georeferencing preservation**, and **pixel-level uncertainty quantification**.

---

## 🏛️ System Architecture & Model Pipeline

1. **Sentinel-2 SRGAN (PyTorch Deep ResNet)**:
   - Deep Residual Convolutional Generator with 16 residual blocks and dual $2\times$ sub-pixel PixelShuffle layers for continuous $4\times$ spatial upsampling ($10\text{m} \to 2.5\text{m}$).
   - Natively operates on multi-spectral remote sensing bands (B02 Blue, B03 Green, B04 Red, B08 NIR) with adaptive 3-band / 4-band tensor routing.

2. **High-Frequency Spline Refiner (Classical Baseline)**:
   - High-order bicubic spline interpolation combined with spatial high-pass edge synthesis. Provides an analytical benchmark against deep neural networks.

3. **Bicubic Baseline Standard**:
   - The established remote sensing baseline standard. All candidate models are evaluated directly against this baseline to report true metric gains ($+\Delta\text{dB PSNR}, +\Delta\text{SSIM}$).

4. **Extended Research Roadmap**:
   - SwinIR Shifted-Window Transformer and Remote Sensing Latent Diffusion fine-tuning using paired real-world datasets (**WorldStrat** SPOT 1.5m / **SEN2VENµS** 5m).

---

## 🔬 Scientific Rigor & Remote Sensing Integrity

### 1. Radiometric Calibration (BOA Surface Reflectance)
- **Sentinel-2 L2A BOA Reflectance**: Multi-spectral imagery is scaled using the ESA quantification factor of $10000$ ($\text{Reflectance} = \text{DN} / 10000.0$).
- **No Destructive Stretching**: Absolute surface reflectance is strictly preserved, preventing distortion of spectral indices such as **NDVI** (Normalized Difference Vegetation Index) and **SAM** (Spectral Angle Mapper).

### 2. Geospatial Co-Registration (GIS Integration)
- **Rasterio Multi-Band I/O**: Reads native GeoTIFF headers, Coordinate Reference Systems (e.g. `EPSG:32643`), and Affine transformation matrices.
- **Affine Geotransform Scaling**: Updates pixel width and height via `GeoReferenceHandler` ($a' = a / 4.0$, $e' = e / 4.0$) so exported $2.5\text{m}$ GeoTIFF products align with sub-pixel precision in **QGIS**, **ArcGIS**, and **Google Earth Engine**.
- **Auxiliary Uncertainty Channel**: Appends the spatial variance map as an additional GeoTIFF band.

### 3. Spatial Uncertainty Quantification (TTA Ensemble)
- **Problem**: Reconstructed high-frequency details are inferred by the generative model and not directly observed.
- **Solution**: Uses **Test-Time Augmentation (TTA) multi-pass spatial ensemble scaffolding** across geometric symmetry groups (canonical, horizontal reflection, vertical reflection, $90^\circ$ rotation).
- **Invariance vs Hallucination**:
  - Invariant homogeneous terrain (water bodies, open fields) yields near-zero variance ($\sim 0$).
  - Fine sub-pixel structures, building boundaries, and model-inferred textures exhibit directional variance, generating a calibrated spatial uncertainty heatmap (Indigo = Confident, Red = Model Inferred).

### 4. Objective Validation & Wald Protocol
- **Zero Fabricated Metrics**: Simulated noise scoring has been deleted.
- **Wald Protocol Degradation Benchmark**: Degrades real $10\text{m}$ Sentinel-2 rasters $4\times$ to $40\text{m}$, super-resolves them back to $10\text{m}$, and validates against the original $10\text{m}$ Sentinel-2 raster as certified ground truth reference.
- **Bicubic Baseline Benchmarking**: Every evaluation reports:
  - Model PSNR, SSIM, SAM, ERGAS
  - Bicubic Baseline PSNR, SSIM, SAM, ERGAS
  - Gain ($+\Delta\text{dB PSNR}, +\Delta\text{SSIM}$) that the model must beat.
- **No-Reference Assessment**: Unpaired inferences provide Tenengrad gradient density, spatial frequency, and NDVI spectral consistency without fake numbers.

---

## 🚀 Quick Start

### 1. Prerequisites
- **Node.js**: v18+ (tested on v20+)
- **Python**: 3.10+ (tested on 3.14+)
- **Rasterio & GDAL** / **PyTorch**

### 2. Install Dependencies

```bash
# Server dependencies
cd server && npm install

# Client dependencies
cd ../client && npm install

# Python ML Microservice
cd ../ml-service
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt rasterio
```

### 3. Running the Stack

```bash
# 1. Express API (Port 5000)
cd server && npm start

# 2. React Vite UI (Port 5173)
cd client && npm run dev

# 3. FastAPI ML Engine (Port 8000)
cd ml-service
./venv/bin/uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload
```

---

## 📊 Evaluation Standards

| Metric | Target | Formula / Description |
| :--- | :---: | :--- |
| **PSNR** | $> 30.0\text{ dB}$ | Peak Signal-to-Noise Ratio (Spatial fidelity) |
| **SSIM** | $> 0.85$ | Structural Similarity Index (Boundary preservation) |
| **SAM** | $< 3.0^\circ$ | Spectral Angle Mapper (Physical multi-band color integrity) |
| **ERGAS** | $< 3.0$ | Dimensionless Global Synthesis Error across all bands |
| **TTA Uncertainty** | Calibrated Map | Pixel-wise ensemble variance identifying inferred features |