const dotenv = require('dotenv');
const path = require('path');

dotenv.config({ path: path.resolve(__dirname, '../.env') });

const NODE_ENV = process.env.NODE_ENV || 'development';

// JWT_SECRET must be set in production; warn loudly in development
let JWT_SECRET = process.env.JWT_SECRET;
if (!JWT_SECRET) {
  if (NODE_ENV === 'production') {
    throw new Error('[FATAL] JWT_SECRET environment variable is not set. Refusing to start in production without it.');
  } else {
    console.warn('[WARNING] JWT_SECRET not set — using insecure dev placeholder. Set JWT_SECRET in server/.env before deploying.');
    JWT_SECRET = 'dev_only_insecure_placeholder_do_not_deploy';
  }
}

module.exports = {
  PORT: process.env.PORT || 5000,
  NODE_ENV,
  CLIENT_URL: process.env.CLIENT_URL || 'http://localhost:5173',
  MONGO_URI: process.env.MONGO_URI || 'mongodb://localhost:27017/nexus_sih',
  REDIS_URL: process.env.REDIS_URL || 'redis://127.0.0.1:6379',
  ML_SERVICE_URL: process.env.ML_SERVICE_URL || 'http://127.0.0.1:8000',
  JWT_SECRET,
  JWT_EXPIRES_IN: process.env.JWT_EXPIRES_IN || '7d',
  UPLOAD_PATH: path.resolve(__dirname, '../../data/raw/sentinel2'),
  OUTPUT_PATH: path.resolve(__dirname, '../../data/outputs'),
  MAX_FILE_SIZE_MB: parseInt(process.env.MAX_FILE_SIZE_MB || '150', 10)
};

