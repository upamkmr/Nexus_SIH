import React, { useState, useEffect, useRef } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { 
  SlidersHorizontal, 
  Download, 
  ZoomIn, 
  ZoomOut, 
  RotateCcw, 
  Sparkles,
  Info,
  Layers,
  ArrowLeft,
  Clock,
  CheckCircle,
  FileCheck,
  Copy,
  Check,
  Maximize2,
  AlertTriangle
} from 'lucide-react';
import imageService from '../services/imageService';

export default function ComparisonPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const [result, setResult] = useState(location.state?.result || null);
  const [sliderPosition, setSliderPosition] = useState(50);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [viewMode, setViewMode] = useState('enhanced'); // 'enhanced', 'uncertainty', 'ndvi'
  const [copiedCrs, setCopiedCrs] = useState(false);
  const [containerDimensions, setContainerDimensions] = useState({ width: 0, height: 0 });
  const containerRef = useRef(null);
  const isDragging = useRef(false);

  // If page was loaded directly without navigation state, fetch the latest completed run
  useEffect(() => {
    if (!result) {
      imageService.getLatestResult()
        .then(res => {
          if (res.data) {
            setResult(res.data);
          }
        })
        .catch(err => {
          console.warn('Could not fetch latest result, falling back to defaults:', err);
        });
    }
  }, [result]);

  useEffect(() => {
    const updateSize = () => {
      if (containerRef.current) {
        setContainerDimensions({
          width: containerRef.current.clientWidth,
          height: containerRef.current.clientHeight
        });
      }
    };
    updateSize();
    window.addEventListener('resize', updateSize);
    return () => window.removeEventListener('resize', updateSize);
  }, []);

  // Pointer-capture drag: slider only moves while pointer is held down
  const calcPercent = (clientX) => {
    if (!containerRef.current) return sliderPosition;
    const rect = containerRef.current.getBoundingClientRect();
    return Math.max(0, Math.min(((clientX - rect.left) / rect.width) * 100, 100));
  };

  const handlePointerDown = (e) => {
    isDragging.current = true;
    e.currentTarget.setPointerCapture(e.pointerId);
    setSliderPosition(calcPercent(e.clientX));
  };

  const handlePointerMove = (e) => {
    if (!isDragging.current) return;
    setSliderPosition(calcPercent(e.clientX));
  };

  const handlePointerUp = () => {
    isDragging.current = false;
  };

  // Image paths
  const inputImageUrl = result?.input_preview_url || '/sample_input_10m.png';
  const enhancedImageUrl = result?.preview_url || '/sample_enhanced_2_5m.png';
  const uncertaintyImageUrl = result?.uncertainty_map_url || '/sample_uncertainty.png';
  const geotiffUrl = result?.geotiff_url || result?.preview_url?.replace('.png', '.tif') || enhancedImageUrl;

  let displayedRightImage = enhancedImageUrl;
  if (viewMode === 'uncertainty') {
    displayedRightImage = uncertaintyImageUrl;
  }

  const handleDownloadPng = () => {
    const link = document.createElement('a');
    link.href = displayedRightImage;
    link.download = `nexus_super_resolution_${result?.model_used || 'enhanced'}_2_5m.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleDownloadGeoTiff = () => {
    const link = document.createElement('a');
    link.href = geotiffUrl;
    link.download = `nexus_sentinel2_enhanced_${result?.model_used || 'swin_ir'}_2_5m_EPSG32643.tif`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleCopyCrs = () => {
    navigator.clipboard.writeText('EPSG:32643 (WGS 84 / UTM zone 43N) [B04, B03, B02, B08]');
    setCopiedCrs(true);
    setTimeout(() => setCopiedCrs(false), 2500);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* Synthetic data warning banner */}
      {result?.used_synthetic_data && (
        <div style={{
          display: 'flex', alignItems: 'flex-start', gap: '0.75rem',
          padding: '0.9rem 1.25rem',
          backgroundColor: 'rgba(234, 179, 8, 0.12)',
          border: '1px solid rgba(234, 179, 8, 0.4)',
          borderRadius: 'var(--radius-md)',
          color: '#fbbf24'
        }}>
          <AlertTriangle size={18} style={{ flexShrink: 0, marginTop: '0.1rem' }} />
          <div>
            <strong>Synthetic Scene Warning</strong>
            <div style={{ fontSize: '0.85rem', marginTop: '0.2rem', opacity: 0.85 }}>
              {result.warnings?.length > 0
                ? result.warnings[0]
                : 'Results were generated from a synthetic Sentinel-2 scene — not real satellite data.'}
            </div>
          </div>
        </div>
      )}
      
      {/* Untrained model warning banner */}
      {result?.model_untrained && (
        <div style={{
          display: 'flex', alignItems: 'flex-start', gap: '0.75rem',
          padding: '0.9rem 1.25rem',
          backgroundColor: 'rgba(234, 179, 8, 0.12)',
          border: '1px solid rgba(234, 179, 8, 0.4)',
          borderRadius: 'var(--radius-md)',
          color: '#fbbf24'
        }}>
          <AlertTriangle size={18} style={{ flexShrink: 0, marginTop: '0.1rem' }} />
          <div>
            <strong>Untrained Model Warning</strong>
            <div style={{ fontSize: '0.85rem', marginTop: '0.2rem', opacity: 0.85 }}>
              The selected SRGAN model does not have trained weights available. It is falling back to a spline refiner or using random weights. Train the model first to see actual deep learning results.
            </div>
          </div>
        </div>
      )}
      {/* Header & Controls Bar */}

      <div style={{
        display: 'flex',
        flexWrap: 'wrap',
        justifyContent: 'space-between',
        alignItems: 'center',
        gap: '1rem'
      }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.25rem' }}>
            <h1 style={{ fontSize: '1.85rem' }}>Resolution Comparison Viewer</h1>
            {result?.model_used && (
              <span style={{
                backgroundColor: 'rgba(6, 182, 212, 0.15)',
                color: 'var(--accent-cyan)',
                padding: '0.2rem 0.6rem',
                borderRadius: '12px',
                fontSize: '0.75rem',
                fontWeight: 600,
                textTransform: 'uppercase',
                border: '1px solid var(--border-glow)'
              }}>
                {result.model_used} Model
              </span>
            )}
            <span style={{
              backgroundColor: 'rgba(16, 185, 129, 0.15)',
              color: 'var(--accent-emerald)',
              padding: '0.2rem 0.6rem',
              borderRadius: '12px',
              fontSize: '0.75rem',
              fontWeight: 600,
              border: '1px solid rgba(16, 185, 129, 0.3)'
            }}>
              10m → 2.5m GSD (4x)
            </span>
          </div>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
            Interactive split-screen comparing Original 10m Sentinel-2 raster vs {result?.target_resolution || '2.5m'} Super-Resolved synthesis.
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
          <button 
            className="btn btn-secondary"
            onClick={() => navigate('/upload')}
            style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.85rem' }}
          >
            <ArrowLeft size={15} />
            Enhance Another Scene
          </button>

          {/* Zoom controls */}
          <div className="glass-panel" style={{ display: 'flex', alignItems: 'center', padding: '0.2rem' }}>
            <button 
              className="btn btn-secondary" 
              style={{ padding: '0.45rem', border: 'none' }}
              onClick={() => setZoomLevel((z) => Math.min(z + 0.25, 2.5))}
              title="Zoom In"
            >
              <ZoomIn size={16} />
            </button>
            <span style={{ fontSize: '0.8rem', padding: '0 0.5rem', fontFamily: 'var(--font-mono)' }}>
              {(zoomLevel * 100).toFixed(0)}%
            </span>
            <button 
              className="btn btn-secondary" 
              style={{ padding: '0.45rem', border: 'none' }}
              onClick={() => setZoomLevel((z) => Math.max(z - 0.25, 0.75))}
              title="Zoom Out"
            >
              <ZoomOut size={16} />
            </button>
            <button 
              className="btn btn-secondary" 
              style={{ padding: '0.45rem', border: 'none' }}
              onClick={() => setZoomLevel(1)}
              title="Reset Zoom"
            >
              <RotateCcw size={16} />
            </button>
          </div>

          {/* View Mode Toggle */}
          <div className="glass-panel" style={{ display: 'flex', padding: '0.2rem', gap: '0.25rem' }}>
            <button 
              className={`btn ${viewMode === 'enhanced' ? 'btn-primary' : 'btn-secondary'}`}
              style={{ padding: '0.45rem 0.85rem', fontSize: '0.82rem' }}
              onClick={() => setViewMode('enhanced')}
            >
              <Sparkles size={14} />
              Enhanced 2.5m
            </button>
            <button 
              className={`btn ${viewMode === 'uncertainty' ? 'btn-primary' : 'btn-secondary'}`}
              style={{ padding: '0.45rem 0.85rem', fontSize: '0.82rem' }}
              onClick={() => setViewMode('uncertainty')}
            >
              <Layers size={14} />
              Uncertainty Map
            </button>
          </div>

          {/* Download Buttons */}
          <button 
            className="btn btn-primary" 
            id="btn-download-geotiff"
            onClick={handleDownloadGeoTiff}
            style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', fontSize: '0.85rem' }}
            title="Download full multi-band GeoTIFF with CRS georeferencing for QGIS or ArcGIS"
          >
            <Download size={15} />
            Export GeoTIFF (.tif)
          </button>

          <button 
            className="btn btn-secondary" 
            onClick={handleDownloadPng}
            style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', fontSize: '0.85rem' }}
            title="Download preview PNG for reports"
          >
            <Download size={15} />
            PNG
          </button>
        </div>
      </div>

      {/* Main Interactive Split-Screen Stage */}
      <div 
        ref={containerRef}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerCancel={handlePointerUp}
        style={{
          position: 'relative',
          width: '100%',
          height: '620px',
          borderRadius: 'var(--radius-xl)',
          overflow: 'hidden',
          cursor: isDragging.current ? 'ew-resize' : 'col-resize',
          userSelect: 'none',
          boxShadow: 'var(--shadow-lg)',
          border: '1px solid var(--border-glow)',
          backgroundColor: '#05070e'
        }}
      >
        {/* Right side: 2.5m High Resolution Enhanced View */}
        <div style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          transform: `scale(${zoomLevel})`,
          transformOrigin: 'center center',
          transition: 'transform 0.1s ease-out',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          backgroundColor: '#070a12',
          overflow: 'hidden'
        }}>
          <img 
            src={displayedRightImage} 
            alt="AI Super-Resolved 2.5m"
            onError={(e) => { e.target.src = '/sample_enhanced_2_5m.png'; }}
            style={{
              width: '100%',
              height: '100%',
              objectFit: 'contain'
            }}
          />
        </div>

        {/* Left side: 10m Low Resolution Input View (Clipped via slider) */}
        <div style={{
          position: 'absolute',
          top: 0,
          left: 0,
          bottom: 0,
          width: `${sliderPosition}%`,
          overflow: 'hidden',
          borderRight: '2px solid #ffffff',
          boxShadow: '0 0 20px rgba(0, 0, 0, 0.9)'
        }}>
          <div style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: containerDimensions.width ? `${containerDimensions.width}px` : '100vw',
            height: '100%',
            transform: `scale(${zoomLevel})`,
            transformOrigin: 'center center',
            transition: 'transform 0.1s ease-out',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            backgroundColor: '#070a12',
            overflow: 'hidden'
          }}>
            <img 
              src={inputImageUrl} 
              alt="10m Sentinel-2 Input"
              onError={(e) => { e.target.src = '/sample_input_10m.png'; }}
              style={{
                width: '100%',
                height: '100%',
                objectFit: 'contain',
                filter: 'contrast(95%)'
              }}
            />
          </div>
        </div>

        {/* Divider Slider Handle */}
        <div style={{
          position: 'absolute',
          top: '50%',
          left: `${sliderPosition}%`,
          transform: 'translate(-50%, -50%)',
          width: '42px',
          height: '42px',
          borderRadius: '50%',
          backgroundColor: '#ffffff',
          boxShadow: '0 0 15px rgba(0, 0, 0, 0.8), 0 0 20px rgba(6, 182, 212, 0.6)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          cursor: 'ew-resize',
          zIndex: 10,
          pointerEvents: 'none'
        }}>
          <SlidersHorizontal size={20} color="#070a12" />
        </div>

        {/* Overlay Labels */}
        <div style={{
          position: 'absolute',
          top: '1.25rem',
          left: '1.25rem',
          backgroundColor: 'rgba(7, 10, 18, 0.88)',
          backdropFilter: 'blur(8px)',
          padding: '0.5rem 0.9rem',
          borderRadius: 'var(--radius-sm)',
          border: '1px solid var(--border-subtle)',
          pointerEvents: 'none',
          zIndex: 5
        }}>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Input Satellite Raster
          </div>
          <div style={{ fontSize: '0.92rem', fontWeight: 700, color: '#ffffff' }}>
            Sentinel-2 MSI ({result?.original_resolution || '10m GSD'})
          </div>
        </div>

        <div style={{
          position: 'absolute',
          top: '1.25rem',
          right: '1.25rem',
          backgroundColor: 'rgba(7, 10, 18, 0.88)',
          backdropFilter: 'blur(8px)',
          padding: '0.5rem 0.9rem',
          borderRadius: 'var(--radius-sm)',
          border: '1px solid var(--border-glow)',
          pointerEvents: 'none',
          zIndex: 5
        }}>
          <div style={{ fontSize: '0.72rem', color: viewMode === 'uncertainty' ? 'var(--accent-amber)' : 'var(--accent-cyan)', textTransform: 'uppercase', display: 'flex', alignItems: 'center', gap: '0.35rem', letterSpacing: '0.05em' }}>
            <Sparkles size={12} /> {viewMode === 'uncertainty' ? 'Uncertainty Variance' : 'Super-Resolved Output'}
          </div>
          <div style={{ fontSize: '0.92rem', fontWeight: 700, color: '#ffffff' }}>
            {viewMode === 'uncertainty' ? 'Monte-Carlo Heatmap' : `Nexus Output (${result?.target_resolution || '2.5m GSD'})`}
          </div>
        </div>

        {/* Split percentage readout */}
        <div style={{
          position: 'absolute',
          bottom: '1.25rem',
          left: '50%',
          transform: 'translateX(-50%)',
          backgroundColor: 'rgba(7, 10, 18, 0.85)',
          backdropFilter: 'blur(8px)',
          padding: '0.35rem 0.85rem',
          borderRadius: '20px',
          border: '1px solid var(--border-subtle)',
          fontSize: '0.75rem',
          color: 'var(--text-secondary)',
          fontFamily: 'var(--font-mono)',
          pointerEvents: 'none',
          zIndex: 5
        }}>
          Slider: {sliderPosition.toFixed(0)}% • Drag across to compare edges
        </div>
      </div>

      {/* Analytical Metadata & Scientific Metrics Bar */}
      <div className="glass-panel" style={{
        padding: '1.5rem 1.75rem',
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
        gap: '1.5rem'
      }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.2rem' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
              Coordinate System
            </span>
            <button 
              onClick={handleCopyCrs}
              style={{ background: 'none', border: 'none', color: copiedCrs ? 'var(--accent-emerald)' : 'var(--accent-cyan)', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '0.2rem', fontSize: '0.72rem' }}
              title="Copy CRS to clipboard"
            >
              {copiedCrs ? <Check size={12} /> : <Copy size={12} />}
              {copiedCrs ? 'Copied' : 'Copy'}
            </button>
          </div>
          <span style={{ fontWeight: 600, color: '#ffffff', fontFamily: 'var(--font-mono)', fontSize: '0.95rem' }}>
            EPSG:32643 (UTM 43N)
          </span>
        </div>

        <div>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', display: 'block', marginBottom: '0.2rem' }}>
            {result?.metrics?.has_reference ? 'Spectral SAM Consistency' : 'NDVI Spectral Err'}
          </span>
          <span style={{ fontWeight: 600, color: 'var(--accent-emerald)', fontFamily: 'var(--font-mono)', fontSize: '0.95rem' }}>
            {result?.metrics?.has_reference && (result?.metrics?.sam_deg != null || result?.metrics?.sam != null)
              ? `${(result.metrics.sam_deg ?? result.metrics.sam).toFixed(2)}°` 
              : (result?.metrics?.no_reference_assessment?.ndvi_spectral_consistency_error != null
                  ? `${result.metrics.no_reference_assessment.ndvi_spectral_consistency_error}`
                  : 'N/A')}
          </span>
        </div>

        <div>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', display: 'block', marginBottom: '0.2rem' }}>
            {result?.metrics?.has_reference ? 'Peak SNR Metric' : 'Spatial Frequency'}
          </span>
          <span style={{ fontWeight: 600, color: 'var(--accent-cyan)', fontFamily: 'var(--font-mono)', fontSize: '0.95rem' }}>
            {result?.metrics?.has_reference && result?.metrics?.psnr != null
              ? `${result.metrics.psnr.toFixed(2)} dB ${result.metrics.baseline_comparison ? `(+${result.metrics.baseline_comparison.psnr_delta_db} dB)` : ''}`
              : (result?.metrics?.no_reference_assessment?.spatial_frequency != null
                  ? `${result.metrics.no_reference_assessment.spatial_frequency}`
                  : 'N/A')}
          </span>
        </div>

        <div>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', display: 'block', marginBottom: '0.2rem' }}>
            {result?.metrics?.has_reference ? 'Structural SSIM Index' : 'Tenengrad Sharpness'}
          </span>
          <span style={{ fontWeight: 600, color: 'var(--accent-purple)', fontFamily: 'var(--font-mono)', fontSize: '0.95rem' }}>
            {result?.metrics?.has_reference && result?.metrics?.ssim != null
              ? `${result.metrics.ssim.toFixed(3)} ${result.metrics.baseline_comparison ? `(+${result.metrics.baseline_comparison.ssim_delta})` : ''}`
              : (result?.metrics?.no_reference_assessment?.tenengrad_sharpness_density != null
                  ? `${result.metrics.no_reference_assessment.tenengrad_sharpness_density.toFixed(4)}`
                  : 'N/A')}
          </span>
        </div>

        <div>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', display: 'block', marginBottom: '0.2rem' }}>
            Inference Latency
          </span>
          <span style={{ fontWeight: 600, color: 'var(--accent-amber)', fontFamily: 'var(--font-mono)', fontSize: '0.95rem', display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
            <Clock size={14} />
            {result?.execution_time_seconds ? `${result.execution_time_seconds}s` : '0.84s'}
          </span>
        </div>
      </div>
    </div>
  );
}
