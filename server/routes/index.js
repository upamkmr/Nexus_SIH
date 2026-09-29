const express = require('express');
const path = require('path');
const router = express.Router();
const imageRoutes = require('./imageRoutes');
const { ML_SERVICE_URL } = require('../config/env');
const axios = require('axios');

// Mount Image & Super-Resolution routes
router.use('/images', imageRoutes);

// System Health Check
router.get('/health', async (req, res) => {
  let mlServiceStatus = 'offline';
  let mlData = null;
  try {
    const mlHealth = await axios.get(`${ML_SERVICE_URL}/api/v1/health`, { timeout: 2000 });
    if (mlHealth.data && mlHealth.data.status === 'online') {
      mlServiceStatus = 'online';
      mlData = mlHealth.data;
    }
  } catch (err) {
    mlServiceStatus = 'unreachable';
  }

  res.json({
    status: 'online',
    service: 'Nexus Express API Server',
    timestamp: new Date().toISOString(),
    ml_service: {
      status: mlServiceStatus,
      url: ML_SERVICE_URL,
      details: mlData
    },
    version: '1.0.0'
  });
});

const fs = require('fs');
const { OUTPUT_PATH } = require('../config/env');

// Dynamic Dashboard Stats (Derived from real processed products)
router.get('/dashboard/stats', (req, res) => {
  let fileCount = 0;
  let storageMb = 0;

  try {
    if (fs.existsSync(OUTPUT_PATH)) {
      const files = fs.readdirSync(OUTPUT_PATH);
      const enhancedFiles = files.filter(f => f.startsWith('enhanced_sr_') && f.endsWith('.png'));
      fileCount = enhancedFiles.length;
      
      for (const f of files) {
        try {
          const st = fs.statSync(path.join(OUTPUT_PATH, f));
          storageMb += st.size / (1024 * 1024);
        } catch (_) {}
      }
    }
  } catch (err) {
    console.warn('Could not read outputs dir for stats:', err.message);
  }

  res.json({
    totalImagesProcessed: fileCount,
    activeJobs: 0,
    benchmarkNote: "Metrics require paired ground truth or Wald protocol degradation benchmark",
    avgPsnr: null,
    avgSsim: null,
    storageUsedGb: Number((storageMb / 1024).toFixed(3)),
    modelsAvailable: ['Sentinel-2 SRGAN (PyTorch)', 'High-Frequency Spline Refiner', 'Bicubic Baseline']
  });
});

module.exports = router;
