const axios = require('axios');
const { ML_SERVICE_URL } = require('../config/env');

class MLBridgeService {
  constructor() {
    this.client = axios.create({
      baseURL: ML_SERVICE_URL,
      timeout: 300000 // 5 minutes timeout for deep learning inference
    });
  }

  async checkHealth() {
    try {
      const response = await this.client.get('/api/v1/health');
      return { online: true, data: response.data };
    } catch (error) {
      return { online: false, error: error.message };
    }
  }

  async getModels() {
    try {
      const response = await this.client.get('/api/v1/models');
      return response.data;
    } catch (error) {
      return { models: [], error: error.message };
    }
  }

  async runSuperResolution({
    imagePath,
    modelType = 'srgan',
    scaleFactor = 4,
    estimateUncertainty = true,
    bands = ['B04', 'B03', 'B02'],
    referencePath = null,
    runWaldValidation = false,
    baselineOffset = null
  }) {
    const payload = {
      image_path: imagePath,
      model_type: modelType,
      scale_factor: Number(scaleFactor),
      estimate_uncertainty: Boolean(estimateUncertainty),
      bands: Array.isArray(bands) ? bands : ['B04', 'B03', 'B02'],
      reference_path: referencePath || null,
      run_wald_validation: Boolean(runWaldValidation)
    };
    if (baselineOffset !== null) {
      payload.baseline_offset = baselineOffset;
    }

    const response = await this.client.post('/api/v1/predict', payload);
    return response.data;
  }
}

module.exports = new MLBridgeService();
