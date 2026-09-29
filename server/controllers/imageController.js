const path = require('path');
const fs = require('fs');
const mlBridge = require('../services/mlBridge');
const { UPLOAD_PATH, OUTPUT_PATH } = require('../config/env');

// In-memory cache for recent super-resolution jobs
let recentJobs = [];
let latestJob = null;

class ImageController {
  // Process uploaded image or selected sample raster
  async processImage(req, res, next) {
    try {
      const io = req.app.get('io');
      let targetFilePath = null;
      let filename = '';

      if (req.file) {
        targetFilePath = req.file.path;
        filename = req.file.filename;
      } else if (req.body.sample_file) {
        // User selected an existing Copernicus Sentinel-2 sample raster
        const cleanSample = path.basename(req.body.sample_file);
        targetFilePath = path.join(UPLOAD_PATH, cleanSample);
        filename = cleanSample;

        if (!fs.existsSync(targetFilePath)) {
          return res.status(404).json({ error: `Sample file "${cleanSample}" not found in data repository.` });
        }
      } else {
        // Fallback to default Sentinel-2 TCI scene if available
        const defaultSample = path.join(UPLOAD_PATH, 'TCI.tif');
        if (fs.existsSync(defaultSample)) {
          targetFilePath = defaultSample;
          filename = 'TCI.tif';
        } else {
          return res.status(400).json({
            error: 'No image file uploaded or sample specified. Please provide a GeoTIFF or select a Copernicus sample.'
          });
        }
      }

      const modelType = req.body.model_type || req.body.modelType || 'srgan';
      const scaleFactor = parseInt(req.body.scale_factor || req.body.scaleFactor || '4', 10);
      const estimateUncertainty = req.body.estimate_uncertainty !== 'false' && req.body.estimate_uncertainty !== false;
      const runWaldValidation = req.body.run_wald_validation === 'true' || req.body.run_wald_validation === true;
      const referencePath = req.body.reference_path || null;
      const bands = req.body.bands ? (Array.isArray(req.body.bands) ? req.body.bands : [req.body.bands]) : ['B04', 'B03', 'B02'];

      const jobId = 'job_' + Date.now();

      // Emit starting progress over WebSocket
      if (io) {
        io.emit('job_started', {
          jobId,
          filename,
          modelType,
          scaleFactor,
          timestamp: new Date().toISOString()
        });
        io.emit('job_progress', {
          jobId,
          progress: 25,
          message: `Parsing raster bands and tiling with ${scaleFactor}x super-resolution grid...`
        });
      }

      console.log(`[ImageController] Running super-resolution on: ${targetFilePath} using ${modelType}`);

      // Run ML Pipeline via microservice
      const mlResult = await mlBridge.runSuperResolution({
        imagePath: targetFilePath,
        modelType,
        scaleFactor,
        estimateUncertainty,
        bands,
        referencePath,
        runWaldValidation
      });

      const responsePayload = {
        jobId,
        filename,
        originalImage: `/static/uploads/${filename}`,
        ...mlResult,
        timestamp: new Date().toISOString()
      };

      latestJob = responsePayload;
      recentJobs.unshift(responsePayload);
      if (recentJobs.length > 50) recentJobs.pop();

      // Emit completion via WebSocket
      if (io) {
        io.emit('job_complete', {
          jobId,
          status: 'success',
          result: responsePayload
        });
      }

      return res.status(200).json({
        success: true,
        message: 'Super-resolution inference successfully generated',
        jobId,
        data: responsePayload
      });
    } catch (error) {
      console.error('[ImageController Error]', error.message);
      const io = req.app.get('io');
      if (io) {
        io.emit('job_error', {
          error: error.message || 'Super-resolution pipeline failed'
        });
      }
      return res.status(500).json({
        success: false,
        error: error.message || 'Super-resolution processing failed. Ensure ML service is online.'
      });
    }
  }

  // Get the most recent super-resolution result
  async getLatestResult(req, res) {
    if (latestJob) {
      return res.json({ success: true, data: latestJob });
    }

    // If no job was run during this Node session, find recent outputs on disk
    try {
      if (fs.existsSync(OUTPUT_PATH)) {
        const files = fs.readdirSync(OUTPUT_PATH)
          .filter(f => f.startsWith('enhanced_sr_') && f.endsWith('.png'))
          .sort((a, b) => {
            const timeA = fs.statSync(path.join(OUTPUT_PATH, a)).mtimeMs;
            const timeB = fs.statSync(path.join(OUTPUT_PATH, b)).mtimeMs;
            return timeB - timeA;
          });

        if (files.length > 0) {
          const latestFile = files[0];
          const latestTif = latestFile.replace('.png', '.tif');
          // Find matching uncertainty if available
          const uFiles = fs.readdirSync(OUTPUT_PATH).filter(f => f.startsWith('uncertainty_heatmap_'));
          const latestU = uFiles.length > 0 ? uFiles[0] : null;

          // Find matching input preview if available
          const inFiles = fs.readdirSync(OUTPUT_PATH).filter(f => f.startsWith('input_raw_'));
          const latestIn = inFiles.length > 0 ? inFiles[0] : null;

          const fallbackResult = {
            jobId: 'disk_latest',
            model_used: 'Sentinel-2 SRGAN (PyTorch)',
            original_resolution: '10.0m',
            target_resolution: '2.5m',
            scale_factor: 4,
            preview_url: `/static/outputs/${latestFile}`,
            geotiff_url: `/static/outputs/${latestTif}`,
            input_preview_url: latestIn ? `/static/outputs/${latestIn}` : '/sample_input_10m.png',
            uncertainty_map_url: latestU ? `/static/outputs/${latestU}` : null,
            metrics: {
              has_reference: false,
              reference_status: "No paired high-resolution reference raster provided. Reference-based metrics (PSNR, SSIM, SAM, ERGAS) require paired ground truth.",
              psnr: null,
              ssim: null,
              sam_deg: null,
              ergas: null
            },
            metadata: {
              source: 'Copernicus Data Space Ecosystem (CDSE) Sentinel-2',
              bands_processed: ['B04', 'B03', 'B02', 'B08'],
              target_gsd_meters: 2.5
            }
          };
          return res.json({ success: true, data: fallbackResult });
        }
      }
    } catch (e) {
      console.warn('Error reading output dir for latest result:', e.message);
    }

    return res.status(404).json({ success: false, message: 'No super-resolution results available yet.' });
  }

  // List all available Copernicus sample files for instant testing
  async getSamples(req, res) {
    try {
      const samples = [];
      if (fs.existsSync(UPLOAD_PATH)) {
        const files = fs.readdirSync(UPLOAD_PATH);
        for (const file of files) {
          if (file.endsWith('.tif') || file.endsWith('.png') || file.endsWith('.jp2')) {
            const stat = fs.statSync(path.join(UPLOAD_PATH, file));
            let label = file;
            let description = 'Sentinel-2 Level-2A Raster';

            if (file === 'Sentinel2_Airport_Runways_10m.tif') {
              label = 'Airport Runways & Taxiways (Infrastructure)';
              description = 'Ideal test: Sharp linear asphalt edges, runway markings & high-contrast terminal aprons (4 bands, 10m GSD)';
            } else if (file === 'Sentinel2_Agricultural_Canals_10m.tif') {
              label = 'Agricultural Parcels & Irrigation Canal';
              description = 'Ideal test: Rectilinear crop boundaries, multi-spectral NDVI contrasts & water boundary (4 bands, 10m GSD)';
            } else if (file === 'TCI.tif') {
              label = 'Sentinel-2 True Color Image (TCI)';
              description = 'Natural Color RGB 10m Granule from Copernicus Data Space Browser';
            } else if (file === 'B04.tif') {
              label = 'Sentinel-2 Band 04 (Red - 665nm)';
              description = '10m Surface Reflectance Red Band';
            } else if (file === 'B08.tif') {
              label = 'Sentinel-2 Band 08 (NIR - 842nm)';
              description = '10m Surface Reflectance Near-Infrared Band';
            }

            samples.push({
              filename: file,
              label,
              description,
              sizeBytes: stat.size,
              sizeMb: (stat.size / (1024 * 1024)).toFixed(2)
            });
          }
        }
      }

      return res.json({ success: true, samples });
    } catch (err) {
      return res.status(500).json({ success: false, error: err.message });
    }
  }

  // Get job history
  async getHistory(req, res) {
    return res.json({ success: true, history: recentJobs });
  }
}

module.exports = new ImageController();
