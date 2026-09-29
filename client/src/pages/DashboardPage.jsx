import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { 
  UploadCloud, 
  SlidersHorizontal, 
  Layers, 
  Cpu, 
  TrendingUp, 
  Sparkles, 
  ShieldCheck, 
  ArrowUpRight 
} from 'lucide-react';
import api from '../services/api';

export default function DashboardPage() {
  const [stats, setStats] = useState(null);
  const [statsLoading, setStatsLoading] = useState(true);
  const [statsError, setStatsError] = useState(null);

  useEffect(() => {
    api.get('/dashboard/stats')
      .then((data) => { setStats(data); setStatsLoading(false); })
      .catch((err) => { setStatsError(err.message); setStatsLoading(false); });
  }, []);

  const fmt = (v, suffix = '') => (v == null ? '—' : `${v}${suffix}`);

  const statCards = [
    { title: 'Scenes Enhanced',      value: fmt(stats?.totalImagesProcessed), icon: Layers,    color: 'var(--accent-cyan)',    change: 'Real processed count' },
    { title: 'Mean PSNR Gain',       value: fmt(stats?.avgPsnr, ' dB'),       icon: TrendingUp, color: 'var(--accent-emerald)', change: 'Benchmark: >30 dB' },
    { title: 'Structural Similarity', value: fmt(stats?.avgSsim),             icon: Sparkles,  color: 'var(--accent-purple)', change: 'Target: >0.85 SSIM' },
    { title: 'Spatial Target GSD',   value: '2.5m',                           icon: Cpu,       color: 'var(--accent-amber)',  change: 'Down from 10m S2' }
  ];


  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      {/* Hero Banner */}
      <div className="glass-panel" style={{
        padding: '2.5rem',
        borderRadius: 'var(--radius-xl)',
        background: 'linear-gradient(135deg, rgba(19, 28, 49, 0.9) 0%, rgba(13, 19, 34, 0.95) 100%)',
        position: 'relative',
        overflow: 'hidden'
      }}>
        <div style={{
          position: 'absolute',
          top: '-30%',
          right: '-10%',
          width: '400px',
          height: '400px',
          background: 'radial-gradient(circle, rgba(6, 182, 212, 0.15) 0%, transparent 70%)',
          pointerEvents: 'none'
        }} />

        <div style={{ maxWidth: '780px', position: 'relative', zIndex: 1 }}>
          <div className="badge badge-cyan" style={{ marginBottom: '1rem' }}>
            EARTH OBSERVATION SUPER-RESOLUTION
          </div>
          <h1 style={{ fontSize: '2.5rem', lineHeight: 1.15, marginBottom: '1rem' }}>
            Transform 10m Sentinel-2 Imagery into <span className="gradient-text">&lt;4m Geospatial Products</span>
          </h1>
          <p style={{ color: 'var(--text-secondary)', fontSize: '1.05rem', marginBottom: '1.75rem', lineHeight: 1.6 }}>
            Harness deep generative models (SRGAN, GeoDiffusion, SwinIR) to reconstruct fine roads, field boundaries, and urban structures with certified spectral consistency and pixel uncertainty quantification.
          </p>
          <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
            <Link to="/upload" className="btn btn-primary" id="btn-hero-enhance">
              <UploadCloud size={18} />
              Enhance Sentinel-2 Scene
            </Link>
            <Link to="/comparison" className="btn btn-secondary" id="btn-hero-compare">
              <SlidersHorizontal size={18} />
              Open Comparison Viewer
            </Link>
          </div>
        </div>
      </div>

      {/* Metric Cards Grid */}
      {statsError && (
        <div style={{ padding: '0.75rem 1rem', backgroundColor: 'rgba(185,28,28,0.1)', border: '1px solid rgba(185,28,28,0.3)', borderRadius: 'var(--radius-md)', color: '#fca5a5', fontSize: '0.85rem' }}>
          Could not load live stats: {statsError}
        </div>
      )}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
        gap: '1.25rem',
        opacity: statsLoading ? 0.45 : 1,
        transition: 'opacity 0.3s'
      }}>
        {statCards.map((card, i) => {
          const Icon = card.icon;
          return (
            <div key={i} className="glass-panel" style={{ padding: '1.5rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                <span style={{ color: 'var(--text-muted)', fontSize: '0.85rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                  {card.title}
                </span>
                <div style={{
                  padding: '0.45rem',
                  borderRadius: '10px',
                  backgroundColor: 'rgba(255,255,255,0.04)',
                  color: card.color
                }}>
                  <Icon size={20} />
                </div>
              </div>
              <div style={{ fontSize: '1.85rem', fontWeight: 700, color: '#ffffff', marginBottom: '0.35rem' }}>
                {statsLoading ? '…' : card.value}
              </div>
              <div style={{ fontSize: '0.78rem', color: card.color, fontWeight: 500 }}>
                {card.change}
              </div>
            </div>
          );
        })}
      </div>


      {/* Model Architectures Showcase */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        <h2 style={{ fontSize: '1.35rem' }}>Available Super-Resolution Pipelines</h2>
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))',
          gap: '1.5rem'
        }}>
          {/* Card 1: SRGAN */}
          <div className="glass-panel" style={{ padding: '1.75rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <h3 style={{ fontSize: '1.15rem', color: '#ffffff' }}>Sentinel-2 SRGAN</h3>
              <span className="badge badge-cyan">4x Scale (2.5m)</span>
            </div>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', flex: 1 }}>
              Optimized deep adversarial network using perceptual VGG/ResNet loss for fast edge synthesis along roads and urban structures.
            </p>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              <span>Latency: <strong>~0.8s / tile</strong></span>
              <span>Memory: <strong>Lightweight</strong></span>
            </div>
          </div>

          {/* Card 2: Diffusion */}
          <div className="glass-panel" style={{ padding: '1.75rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <h3 style={{ fontSize: '1.15rem', color: '#ffffff' }}>GeoDiffusion-SR</h3>
              <span className="badge badge-purple">Generative Diffusion</span>
            </div>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', flex: 1 }}>
              Score-based conditional diffusion pipeline with iterative denoising to recover natural agricultural field textures without over-smoothing.
            </p>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              <span>Latency: <strong>~2.4s / tile</strong></span>
              <span>Detail: <strong>Ultra-High</strong></span>
            </div>
          </div>

          {/* Card 3: SwinIR */}
          <div className="glass-panel" style={{ padding: '1.75rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <h3 style={{ fontSize: '1.15rem', color: '#ffffff' }}>SwinIR Transformer</h3>
              <span className="badge badge-emerald">Spectral Transformer</span>
            </div>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', flex: 1 }}>
              Shifted window self-attention blocks tailored for multi-spectral remote sensing. Excels at preserving NDVI spectral fidelity.
            </p>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              <span>Latency: <strong>~1.1s / tile</strong></span>
              <span>Spectral SAM: <strong>&lt; 2.2°</strong></span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
