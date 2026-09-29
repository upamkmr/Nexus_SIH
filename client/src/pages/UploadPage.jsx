import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  UploadCloud, 
  Settings2, 
  Sparkles, 
  Layers, 
  CheckCircle2, 
  AlertCircle,
  FileText,
  Sliders,
  Database,
  ArrowRight,
  Cpu
} from 'lucide-react';
import imageService from '../services/imageService';
import { io as socketIO } from 'socket.io-client';

// Single shared socket instance (lazy singleton)
let _socket = null;
const getSocket = () => {
  if (!_socket) {
    _socket = socketIO('http://localhost:5000', { transports: ['websocket'] });
  }
  return _socket;
};


export default function UploadPage() {
  const navigate = useNavigate();
  const [file, setFile] = useState(null);
  const [selectedSample, setSelectedSample] = useState('TCI.tif');
  const [useSample, setUseSample] = useState(false);
  const [modelType, setModelType] = useState('srgan');
  const [scaleFactor, setScaleFactor] = useState(4);
  const [estimateUncertainty, setEstimateUncertainty] = useState(true);
  const [runWaldValidation, setRunWaldValidation] = useState(false);
  const [selectedBands, setSelectedBands] = useState('rgb_nir');
  const [baselineOffset, setBaselineOffset] = useState('');
  const [processing, setProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [statusMessage, setStatusMessage] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  const [samples, setSamples] = useState([]);

  // True only when the user has actively provided input
  const isValid = !!file || useSample;

  useEffect(() => {
    // Load available Copernicus samples from backend
    imageService.getSamples()
      .then(res => {
        if (res.samples && res.samples.length > 0) {
          setSamples(res.samples);
        }
      })
      .catch(() => {
        // Fallback default samples if server is cold
        setSamples([
          { filename: 'TCI.tif', label: 'Sentinel-2 True Color Image (TCI)', description: 'Natural Color RGB 10m Granule from Copernicus CDSE', sizeMb: '0.97' },
          { filename: 'B04.tif', label: 'Sentinel-2 Band 04 (Red)', description: '10m Surface Reflectance Red Band', sizeMb: '0.68' },
          { filename: 'B08.tif', label: 'Sentinel-2 Band 08 (NIR)', description: '10m Surface Reflectance Near-Infrared Band', sizeMb: '0.68' },
          { filename: 'B02.tif', label: 'Sentinel-2 Band 02 (Blue)', description: '10m Surface Reflectance Blue Band', sizeMb: '0.69' }
        ]);
      });
  }, []);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setUseSample(false);
      setErrorMessage('');
    }
  };

  const handleSelectSample = (sampleFilename) => {
    setSelectedSample(sampleFilename);
    setUseSample(true);
    setFile(null);
    setErrorMessage('');
  };

  const handleProcess = async (e) => {
    e.preventDefault();
    if (!isValid) {
      setErrorMessage('Please upload a Sentinel-2 GeoTIFF or click one of the Copernicus samples below.');
      return;
    }

    setProcessing(true);
    setErrorMessage('');
    setProgress(15);
    setStatusMessage('Uploading raster & extracting radiometric BOA metadata...');

    // Smooth status progression while server runs inference
    const pTimer1 = setTimeout(() => {
      setProgress(40);
      setStatusMessage('Decomposing 10m raster into 256x256 tiles with 32px overlap...');
    }, 800);

    const pTimer2 = setTimeout(() => {
      setProgress(68);
      setStatusMessage(`Executing ${modelType.toUpperCase()} neural enhancement & Monte-Carlo uncertainty estimation...`);
    }, 1800);

    const pTimer3 = setTimeout(() => {
      setProgress(88);
      setStatusMessage('Blending overlapping tiles & generating 2.5m GeoTIFF output with georeferencing...');
    }, 3200);

    const jobId = crypto.randomUUID();
    getSocket().emit('join_job', jobId);

    try {
      let response;
      if (file) {
        const formData = new FormData();
        formData.append('satellite_image', file);
        formData.append('model_type', modelType);
        formData.append('scale_factor', scaleFactor);
        formData.append('estimate_uncertainty', estimateUncertainty);
        formData.append('run_wald_validation', runWaldValidation);
        const bandsToSend = selectedBands === 'rgb_nir' ? ['B04', 'B03', 'B02', 'B08'] : ['B04', 'B03', 'B02'];
        bandsToSend.forEach(b => formData.append('bands', b));
        formData.append('jobId', jobId);
        if (baselineOffset !== '') formData.append('baseline_offset', baselineOffset);
        response = await imageService.uploadAndProcess(formData);
      } else {
        response = await imageService.processSample({
          sample_file: selectedSample || 'TCI.tif',
          model_type: modelType,
          scale_factor: scaleFactor,
          estimate_uncertainty: estimateUncertainty,
          run_wald_validation: runWaldValidation,
          bands: selectedBands === 'rgb_nir' ? ['B04', 'B03', 'B02', 'B08'] : ['B04', 'B03', 'B02'],
          jobId: jobId,
          ...(baselineOffset !== '' ? { baseline_offset: baselineOffset } : {})
        });
      }

      clearTimeout(pTimer1);
      clearTimeout(pTimer2);
      clearTimeout(pTimer3);

      setProgress(100);
      setStatusMessage('Super-Resolution synthesis complete! Preserving spatial coordinates...');

      setTimeout(() => {
        // Navigate to comparison viewer with the generated result
        const resultData = response.data || response;
        navigate('/comparison', { state: { result: resultData } });
      }, 600);

    } catch (err) {
      clearTimeout(pTimer1);
      clearTimeout(pTimer2);
      clearTimeout(pTimer3);
      setProcessing(false);
      setErrorMessage(err.message || 'Error occurred during super-resolution execution. Ensure FastAPI service is running.');
      setStatusMessage('');
    }
  };

  return (
    <div style={{ maxWidth: '980px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      <div>
        <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem', backgroundColor: 'rgba(71, 85, 105, 0.08)', border: '1px solid rgba(71, 85, 105, 0.15)', padding: '0.35rem 0.85rem', borderRadius: '20px', color: '#334155', fontSize: '0.82rem', fontWeight: 600, marginBottom: '0.75rem' }}>
          SIH 2024 Remote Sensing Pipeline • 10m to &lt;4m GSD
        </div>
        <h1 style={{ fontSize: '2.1rem', marginBottom: '0.5rem', color: '#0f172a' }}>Enhance Sentinel-2 Imagery</h1>
        <p style={{ color: '#475569', lineHeight: 1.6 }}>
          Upload standard 10m Level-2A GeoTIFF rasters or choose an imported Copernicus Data Space scene to synthesize sub-4m high-fidelity spatial details with spectral consistency.
        </p>
      </div>

      {errorMessage && (
        <div style={{
          backgroundColor: 'rgba(185, 28, 28, 0.08)',
          border: '1px solid rgba(185, 28, 28, 0.25)',
          borderRadius: 'var(--radius-md)',
          padding: '1rem 1.25rem',
          display: 'flex',
          alignItems: 'center',
          gap: '0.75rem',
          color: '#7f1d1d'
        }}>
          <AlertCircle size={20} color="#991b1b" />
          <span>{errorMessage}</span>
        </div>
      )}

      <form onSubmit={handleProcess} style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem' }}>
        {/* Upload Zone */}
        <div 
          className="glass-panel"
          style={{
            border: file ? '1px solid rgba(51, 65, 85, 0.5)' : '1px dashed rgba(51, 65, 85, 0.45)',
            padding: '2.5rem 2rem',
            textAlign: 'center',
            cursor: 'pointer',
            borderRadius: 'var(--radius-lg)',
            backgroundColor: file ? '#f8fafc' : '#f2f5f8',
            transition: 'all 0.2s ease',
            boxShadow: 'inset 0 0 0 1px rgba(148, 163, 184, 0.12)',
            backgroundImage: 'linear-gradient(180deg, rgba(255,255,255,0.6), rgba(241,245,249,0.9))'
          }}
          onClick={() => document.getElementById('file-upload-input').click()}
        >
          <input 
            type="file" 
            id="file-upload-input" 
            style={{ display: 'none' }} 
            onChange={handleFileChange}
            accept=".tif,.tiff,.jp2,.png,.jpg,.jpeg"
          />
          <div style={{
            width: '64px',
            height: '64px',
            borderRadius: '50%',
            backgroundColor: file ? 'rgba(71, 85, 105, 0.08)' : 'rgba(148, 163, 184, 0.08)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            margin: '0 auto 1.25rem',
            color: '#334155'
          }}>
            <UploadCloud size={32} />
          </div>
          {file ? (
            <div>
              <div style={{ fontSize: '1.15rem', fontWeight: 600, color: '#0f172a', marginBottom: '0.25rem' }}>
                {file.name}
              </div>
              <div style={{ fontSize: '0.85rem', color: '#475569' }}>
                {(file.size / (1024 * 1024)).toFixed(2)} MB • File selected & ready for model execution
              </div>
            </div>
          ) : (
            <div>
              <div style={{ fontSize: '1.1rem', fontWeight: 600, color: '#0f172a', marginBottom: '0.5rem' }}>
                Drop Sentinel-2 GeoTIFF or image here, or click to browse
              </div>
              <div style={{ fontSize: '0.85rem', color: '#64748b' }}>
                Supports .TIF (GeoTIFF), .JP2 (JPEG 2000), .PNG, .JPG (Max 150MB)
              </div>
            </div>
          )}
        </div>

        {/* Quick-Select Copernicus Samples */}
        <div className="glass-panel" style={{ padding: '1.25rem 1.5rem', background: '#ffffff', border: '1px solid rgba(148,163,184,0.15)' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
            <span style={{ fontSize: '0.95rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#0f172a' }}>
              
              Or Test Instantly with Copernicus Data Space Ecosystem (CDSE) Scenes:
            </span>
            <span style={{ fontSize: '0.75rem', color: '#64748b' }}>Real 10m Sentinel-2 Data</span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))', gap: '0.85rem' }}>
            {samples.map((s) => {
              const isSelected = useSample && selectedSample === s.filename;
              return (
                <div
                  key={s.filename}
                  onClick={() => handleSelectSample(s.filename)}
                  style={{
                    padding: '0.85rem 1rem',
                    borderRadius: 'var(--radius-md)',
                    backgroundColor: isSelected ? 'rgba(71, 85, 105, 0.06)' : '#f8fafc',
                    border: isSelected ? '1px solid rgba(71, 85, 105, 0.3)' : '1px solid rgba(148, 163, 184, 0.2)',
                    cursor: 'pointer',
                    transition: 'all 0.15s ease'
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
                    <span style={{ fontWeight: 600, fontSize: '0.88rem', color: isSelected ? '#0f172a' : '#0f172a' }}>
                      {s.filename}
                    </span>
                    {isSelected && <CheckCircle2 size={16} color="#475569" />}
                  </div>
                  <div style={{ fontSize: '0.75rem', color: '#475569', marginBottom: '0.35rem' }}>
                    {s.label}
                  </div>
                  <div style={{ fontSize: '0.7rem', color: '#64748b' }}>
                    {s.sizeMb} MB
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Configuration Grid */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
          gap: '1.25rem'
        }}>
          {/* Model Selector */}
          <div className="glass-panel" style={{ padding: '1.5rem', background: '#ffffff', border: '1px solid rgba(148,163,184,0.15)' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 600, marginBottom: '0.75rem', fontSize: '0.95rem', color: '#0f172a' }}>
              <Settings2 size={16} color="#475569" />
              Generative Model Architecture
            </label>
            <select
              value={modelType}
              onChange={(e) => setModelType(e.target.value)}
              style={{
                width: '100%',
                padding: '0.75rem 1rem',
                borderRadius: 'var(--radius-md)',
                backgroundColor: '#f8fafc',
                border: '1px solid rgba(148, 163, 184, 0.25)',
                color: '#0f172a',
                fontFamily: 'var(--font-body)',
                fontSize: '0.9rem',
                outline: 'none'
              }}
            >
              <option value="srgan">Sentinel-2 SRGAN (PyTorch Deep ResNet - Default)</option>
              <option value="swin_ir">High-Frequency Spline Refiner (Classical Baseline)</option>
              <option value="bicubic">Bicubic Interpolation Baseline (Reference Standard)</option>
            </select>
          </div>

          {/* Scale Factor */}
          <div className="glass-panel" style={{ padding: '1.5rem', background: '#ffffff', border: '1px solid rgba(148,163,184,0.15)' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 600, marginBottom: '0.75rem', fontSize: '0.95rem', color: '#0f172a' }}>
              <Sliders size={16} color="#475569" />
              Target Resolution Scale
            </label>
            <select
              value={scaleFactor}
              onChange={(e) => setScaleFactor(Number(e.target.value))}
              style={{
                width: '100%',
                padding: '0.75rem 1rem',
                borderRadius: 'var(--radius-md)',
                backgroundColor: '#f8fafc',
                border: '1px solid rgba(148, 163, 184, 0.25)',
                color: '#0f172a',
                fontFamily: 'var(--font-body)',
                fontSize: '0.9rem',
                outline: 'none'
              }}
            >
              <option value={4}>4x Scale (10m → 2.5m GSD - SIH Requirement)</option>
              <option value={2}>2x Scale (10m → 5.0m GSD)</option>
            </select>
          </div>

          {/* Spectral Bands */}
          <div className="glass-panel" style={{ padding: '1.5rem', background: '#ffffff', border: '1px solid rgba(148,163,184,0.15)' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 600, marginBottom: '0.75rem', fontSize: '0.95rem', color: '#0f172a' }}>
              <Layers size={16} color="#475569" />
              Band Combination
            </label>
            <select
              value={selectedBands}
              onChange={(e) => setSelectedBands(e.target.value)}
              style={{
                width: '100%',
                padding: '0.75rem 1rem',
                borderRadius: 'var(--radius-md)',
                backgroundColor: '#f8fafc',
                border: '1px solid rgba(148, 163, 184, 0.25)',
                color: '#0f172a',
                fontFamily: 'var(--font-body)',
                fontSize: '0.9rem',
                outline: 'none'
              }}
            >
              <option value="rgb_nir">4 Bands: B04 (R), B03 (G), B02 (B), B08 (NIR)</option>
              <option value="rgb">3 Bands: True Color RGB (B04, B03, B02)</option>
            </select>
          </div>
          
          {/* Baseline Offset */}
          <div className="glass-panel" style={{ padding: '1.5rem', background: '#ffffff', border: '1px solid rgba(148,163,184,0.15)' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 600, marginBottom: '0.75rem', fontSize: '0.95rem', color: '#0f172a' }}>
              <Sliders size={16} color="#475569" />
              BOA Baseline Offset (Untagged uint16)
            </label>
            <select
              value={baselineOffset}
              onChange={(e) => setBaselineOffset(e.target.value)}
              style={{
                width: '100%',
                padding: '0.75rem 1rem',
                borderRadius: 'var(--radius-md)',
                backgroundColor: '#f8fafc',
                border: '1px solid rgba(148, 163, 184, 0.25)',
                color: '#0f172a',
                fontFamily: 'var(--font-body)',
                fontSize: '0.9rem',
                outline: 'none'
              }}
            >
              <option value="">Auto (Extract from Tag or Default 1000)</option>
              <option value="0">0 (Old Baseline &lt; 04.00)</option>
              <option value="1000">1000 (Baseline &gt;= 04.00)</option>
            </select>
          </div>
        </div>

        {/* Uncertainty Checkbox */}
        <div className="glass-panel" style={{ padding: '1.25rem 1.5rem', display: 'flex', alignItems: 'center', gap: '1rem', background: '#ffffff', border: '1px solid rgba(148,163,184,0.15)' }}>
          <input
            type="checkbox"
            id="uncertainty-toggle"
            checked={estimateUncertainty}
            onChange={(e) => setEstimateUncertainty(e.target.checked)}
            style={{ width: '18px', height: '18px', accentColor: '#475569', cursor: 'pointer' }}
          />
          <label htmlFor="uncertainty-toggle" style={{ cursor: 'pointer', fontSize: '0.92rem', color: '#0f172a' }}>
            <strong>Spatial Uncertainty Quantification:</strong> Runs Test-Time Augmentation (TTA) geometric symmetry passes to map sub-pixel inferred detail variance.
          </label>
        </div>

        {/* Wald Protocol Benchmark Checkbox */}
        <div className="glass-panel" style={{ padding: '1.25rem 1.5rem', display: 'flex', alignItems: 'center', gap: '1rem', background: '#ffffff', border: '1px solid rgba(148,163,184,0.15)' }}>
          <input
            type="checkbox"
            id="wald-toggle"
            checked={runWaldValidation}
            onChange={(e) => setRunWaldValidation(e.target.checked)}
            style={{ width: '18px', height: '18px', accentColor: '#475569', cursor: 'pointer' }}
          />
          <label htmlFor="wald-toggle" style={{ cursor: 'pointer', fontSize: '0.92rem', color: '#0f172a' }}>
            <strong>Wald Protocol Validation:</strong> Degrades input 4x to 40m, reconstructs to 10m, and benchmarks against original 10m Sentinel-2 with Bicubic baseline comparison (+Δ dB PSNR).
          </label>
        </div>

        {/* Progress Display */}
        {processing && (
          <div className="glass-panel" style={{ padding: '1.5rem', background: '#ffffff', border: '1px solid rgba(148,163,184,0.15)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem', fontSize: '0.88rem', color: '#0f172a' }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Cpu size={16} className="animate-spin" color="#475569" />
                {statusMessage}
              </span>
              <span style={{ fontWeight: 700, color: '#334155' }}>{progress}%</span>
            </div>
            <div style={{ height: '8px', backgroundColor: '#e2e8f0', borderRadius: '4px', overflow: 'hidden' }}>
              <div style={{
                height: '100%',
                width: `${progress}%`,
                background: 'linear-gradient(90deg, #475569 0%, #94a3b8 100%)',
                transition: 'width 0.4s ease'
              }} />
            </div>
          </div>
        )}

        {/* Submit Button */}
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', width: '100%', marginTop: '0.5rem' }}>
          <button
            type="submit"
            className="btn btn-primary"
            id="btn-start-super-resolution"
            disabled={processing || !isValid}
            style={{
              padding: '1rem 2.4rem',
              fontSize: '1.05rem',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '0.75rem',
              background: (!isValid || processing) ? '#475569' : '#172033',
              color: '#f8fafc',
              boxShadow: '0 10px 20px rgba(15, 23, 42, 0.12)',
              border: '1px solid rgba(15, 23, 42, 0.18)',
              minWidth: '320px',
              borderRadius: '12px',
              fontWeight: 700,
              letterSpacing: '0.01em',
              cursor: (!isValid || processing) ? 'not-allowed' : 'pointer',
              opacity: (!isValid || processing) ? 0.6 : 1
            }}
          >
            <Sparkles size={20} />
            {processing ? 'Synthesizing 2.5m Super-Resolution...' : 'Start Generative Super-Resolution'}
          </button>
        </div>

        {!isValid && (
          <div style={{ display: 'flex', justifyContent: 'center', width: '100%' }}>
            <span style={{ fontSize: '0.85rem', color: '#64748b' }}>
              (Click a Copernicus sample or upload a file to begin)
            </span>
          </div>
        )}
      </form>
    </div>
  );
}
