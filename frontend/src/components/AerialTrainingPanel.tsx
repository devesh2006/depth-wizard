import React, { useState, useEffect } from 'react';
import {
  Cpu,
  Activity,
  AlertTriangle,
  Play,
  Square,
  RefreshCw,
  Database,
  Lock,
  Download,
  Layers,
  Info
} from 'lucide-react';
import type { MultiTaskTrainingStatusResponse } from '../types/depth';

export const AerialTrainingPanel: React.FC = () => {
  const [statusData, setStatusData] = useState<MultiTaskTrainingStatusResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [isStarting, setIsStarting] = useState<boolean>(false);
  const [selectedMode, setSelectedMode] = useState<'ZERO_SHOT' | 'AERIAL_DOMAIN_ADAPTATION' | 'AERIAL_METRIC_FINE_TUNING'>('AERIAL_DOMAIN_ADAPTATION');

  const fetchTrainingStatus = async () => {
    setIsLoading(true);
    setErrorMsg(null);
    try {
      const resp = await fetch('/api/training/status');
      if (!resp.ok) {
        throw new Error('Failed to fetch training pipeline status.');
      }
      const data: MultiTaskTrainingStatusResponse = await resp.json();
      setStatusData(data);
    } catch (err: any) {
      setErrorMsg(err.message || 'Failed to load status.');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchTrainingStatus();
  }, []);

  const handleStartTraining = async () => {
    setIsStarting(true);
    setErrorMsg(null);
    try {
      const resp = await fetch('/api/training/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config_path: 'configs/aerial_depth_training.yaml', mode: selectedMode })
      });

      if (!resp.ok) {
        const errJson = await resp.json().catch(() => ({ detail: 'Failed to start training.' }));
        throw new Error(errJson.detail || 'Start training request failed.');
      }

      await fetchTrainingStatus();
    } catch (err: any) {
      setErrorMsg(err.message);
    } finally {
      setIsStarting(false);
    }
  };

  const handleStopTraining = async () => {
    try {
      await fetch('/api/training/stop', { method: 'POST' });
      await fetchTrainingStatus();
    } catch (err: any) {
      setErrorMsg(err.message);
    }
  };

  const handleExportArtifacts = async () => {
    try {
      const resp = await fetch('/api/training/status');
      const data = await resp.json();
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'depthwizard_phase2.1_training_report.json';
      a.click();
    } catch (err: any) {
      setErrorMsg('Failed to export training report.');
    }
  };

  const currentModeInfo = statusData?.modes_availability?.[selectedMode];
  const isBlocked = currentModeInfo?.status === 'BLOCKED';
  const isRunning = statusData?.training_state.is_running ?? false;

  const gamusStatus = statusData?.datasets?.gamus?.status ?? 'MISSING';
  const metricStatus = statusData?.datasets?.metric_depth?.status ?? 'MISSING';
  const gamusDetails = statusData?.datasets?.gamus?.details;

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 text-slate-100 shadow-2xl space-y-8">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between border-b border-slate-800 pb-4 gap-4">
        <div>
          <div className="flex items-center gap-2">
            <Cpu className="w-6 h-6 text-cyan-400" />
            <h2 className="text-xl font-bold text-white tracking-wide">
              Phase 2.1: GAMUS Aerial-Domain Adaptation & Multi-Task Fine-Tuning
            </h2>
            <span className="px-2.5 py-0.5 text-xs font-semibold bg-cyan-500/20 text-cyan-300 rounded-full border border-cyan-500/30">
              Multi-Task PyTorch Pipeline
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Explicit separation of camera-surface depth, AGL height, and semantic terrain classes. Frozen Stage 3 geometry engine.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={fetchTrainingStatus}
            disabled={isLoading}
            className="flex items-center gap-1.5 px-3 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-medium rounded-lg border border-slate-700 transition-all"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin' : ''}`} />
            <span>Refresh</span>
          </button>

          <button
            onClick={handleExportArtifacts}
            className="flex items-center gap-1.5 px-3 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-medium rounded-lg border border-slate-700 transition-all"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Export Report</span>
          </button>

          {isRunning ? (
            <button
              onClick={handleStopTraining}
              className="flex items-center gap-2 px-4 py-2 bg-rose-600 hover:bg-rose-500 text-white font-semibold text-xs rounded-lg shadow-lg shadow-rose-600/20 transition-all"
            >
              <Square className="w-3.5 h-3.5 fill-current" />
              <span>STOP TRAINING</span>
            </button>
          ) : (
            <button
              onClick={handleStartTraining}
              disabled={isBlocked || isStarting}
              title={isBlocked ? currentModeInfo?.blocked_reason || 'Mode blocked' : 'Start training'}
              className="flex items-center gap-2 px-5 py-2 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 disabled:opacity-40 disabled:cursor-not-allowed text-white font-semibold text-xs rounded-lg shadow-lg shadow-cyan-500/20 transition-all"
            >
              {isStarting ? (
                <Activity className="w-4 h-4 animate-spin" />
              ) : isBlocked ? (
                <Lock className="w-4 h-4" />
              ) : (
                <Play className="w-4 h-4 fill-current" />
              )}
              <span>START TRAINING</span>
            </button>
          )}
        </div>
      </div>

      {/* Error Banner */}
      {errorMsg && (
        <div className="p-4 bg-rose-950/50 border border-rose-800/50 rounded-lg text-rose-300 text-xs flex items-center gap-3">
          <AlertTriangle className="w-5 h-5 flex-shrink-0" />
          <span>{errorMsg}</span>
        </div>
      )}

      {/* SECTION 1: DATASETS MATRIX */}
      <div className="space-y-4">
        <div className="flex items-center gap-2 text-xs font-semibold text-slate-300 uppercase tracking-wider">
          <Database className="w-4 h-4 text-cyan-400" />
          <span>Registered Datasets Matrix</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* GAMUS Dataset Card */}
          <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-3">
            <div className="flex items-center justify-between">
              <span className="font-bold text-sm text-white">1. GAMUS Remote Sensing Dataset</span>
              <span className={`px-2.5 py-0.5 text-xs font-bold rounded-full ${gamusStatus === 'AVAILABLE' ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30' : 'bg-amber-500/20 text-amber-300 border border-amber-500/30'}`}>
                {gamusStatus}
              </span>
            </div>
            <p className="text-xs text-slate-400">
              Aerial RGB + AGL Height Maps + 7 Semantic Classes (others, ground, low_veg, buildings, water, road, tree).
            </p>
            {gamusDetails && (
              <div className="bg-slate-900/80 p-3 rounded text-[11px] font-mono text-slate-300 space-y-1">
                <div>Samples: {gamusDetails.sample_counts?.total ?? 0} total ({gamusDetails.sample_counts?.train ?? 0} train / {gamusDetails.sample_counts?.val ?? 0} val)</div>
                <div>AGL Range: {gamusDetails.agl_stats?.min ?? 0}m to {gamusDetails.agl_stats?.max ?? 0}m (valid: {((gamusDetails.agl_stats?.valid_ratio ?? 0) * 100).toFixed(0)}%)</div>
                <div>Semantic Classes: {Object.keys(gamusDetails.semantic_classes || {}).length} detected</div>
                <div className="text-emerald-400 font-semibold">Camera Surface Depth: NONE (Preserved as AGL only)</div>
              </div>
            )}
          </div>

          {/* True Metric Depth Dataset Card */}
          <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-3">
            <div className="flex items-center justify-between">
              <span className="font-bold text-sm text-white">2. True Camera-Surface Metric Depth</span>
              <span className={`px-2.5 py-0.5 text-xs font-bold rounded-full ${metricStatus === 'AVAILABLE' ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30' : 'bg-rose-500/20 text-rose-300 border border-rose-500/30'}`}>
                {metricStatus}
              </span>
            </div>
            <p className="text-xs text-slate-400">
              Paired Aerial RGB + Genuine Camera-to-Surface Distance Ground Truth (Airborne LiDAR / Photogrammetry).
            </p>
            <div className="bg-slate-900/80 p-3 rounded text-[11px] font-mono text-slate-300 space-y-1">
              <div>Target Folder: <span className="text-cyan-300">dataset/metric_depth/</span></div>
              {metricStatus === 'MISSING' ? (
                <div className="text-rose-400 font-semibold">
                  Status: No camera-surface depth ground truth dataset registered. Metric fine-tuning blocked.
                </div>
              ) : (
                <div className="text-emerald-400 font-semibold">
                  Status: True metric depth dataset ready for supervision.
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* SECTION 2: TRAINING MODES SELECTOR */}
      <div className="space-y-4">
        <div className="flex items-center gap-2 text-xs font-semibold text-slate-300 uppercase tracking-wider">
          <Layers className="w-4 h-4 text-indigo-400" />
          <span>Select Training Mode</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Mode 1: Zero-Shot */}
          <div
            onClick={() => setSelectedMode('ZERO_SHOT')}
            className={`cursor-pointer p-4 rounded-xl border transition-all ${
              selectedMode === 'ZERO_SHOT'
                ? 'bg-cyan-950/40 border-cyan-500 text-white shadow-lg shadow-cyan-500/10'
                : 'bg-slate-950 border-slate-800 text-slate-300 hover:border-slate-700'
            }`}
          >
            <div className="flex items-center justify-between mb-2">
              <span className="font-bold text-sm">Mode 1: Zero-Shot</span>
              <span className="text-[10px] font-bold px-2 py-0.5 bg-emerald-500/20 text-emerald-300 rounded">AVAILABLE</span>
            </div>
            <p className="text-xs text-slate-400 mb-3">
              Evaluates baseline pretrained Depth Anything V2 Outdoor Large model without fine-tuning.
            </p>
            <div className="text-[11px] font-mono text-slate-400 space-y-1">
              <div>✓ Pretrained DA-V2 Backbone</div>
            </div>
          </div>

          {/* Mode 2: Aerial Domain Adaptation */}
          <div
            onClick={() => setSelectedMode('AERIAL_DOMAIN_ADAPTATION')}
            className={`cursor-pointer p-4 rounded-xl border transition-all ${
              selectedMode === 'AERIAL_DOMAIN_ADAPTATION'
                ? 'bg-cyan-950/40 border-cyan-500 text-white shadow-lg shadow-cyan-500/10'
                : 'bg-slate-950 border-slate-800 text-slate-300 hover:border-slate-700'
            }`}
          >
            <div className="flex items-center justify-between mb-2">
              <span className="font-bold text-sm">Mode 2: Aerial Domain Adaptation</span>
              <span className={`text-[10px] font-bold px-2 py-0.5 rounded ${gamusStatus === 'AVAILABLE' ? 'bg-emerald-500/20 text-emerald-300' : 'bg-amber-500/20 text-amber-300'}`}>
                {gamusStatus === 'AVAILABLE' ? 'AVAILABLE' : 'BLOCKED'}
              </span>
            </div>
            <p className="text-xs text-slate-400 mb-3">
              Multi-task auxiliary AGL + Semantic supervision using GAMUS remote-sensing dataset.
            </p>
            <div className="text-[11px] font-mono text-slate-300 space-y-1">
              <div className="text-emerald-400">✓ GAMUS Dataset</div>
              <div className="text-emerald-400">✓ Aerial RGB</div>
              <div className="text-emerald-400">✓ AGL Height Map</div>
              <div className="text-emerald-400">✓ 7 Semantic Classes</div>
            </div>
          </div>

          {/* Mode 3: Full Metric Fine-Tuning */}
          <div
            onClick={() => setSelectedMode('AERIAL_METRIC_FINE_TUNING')}
            className={`cursor-pointer p-4 rounded-xl border transition-all ${
              selectedMode === 'AERIAL_METRIC_FINE_TUNING'
                ? 'bg-cyan-950/40 border-cyan-500 text-white shadow-lg shadow-cyan-500/10'
                : 'bg-slate-950 border-slate-800 text-slate-300 hover:border-slate-700'
            }`}
          >
            <div className="flex items-center justify-between mb-2">
              <span className="font-bold text-sm">Mode 3: Full Metric Fine-Tuning</span>
              <span className={`text-[10px] font-bold px-2 py-0.5 rounded ${metricStatus === 'AVAILABLE' ? 'bg-emerald-500/20 text-emerald-300' : 'bg-rose-500/20 text-rose-300'}`}>
                {metricStatus === 'AVAILABLE' ? 'AVAILABLE' : 'BLOCKED'}
              </span>
            </div>
            <p className="text-xs text-slate-400 mb-3">
              Direct supervision of camera-surface metric depth requiring genuine paired metric dataset.
            </p>
            <div className="text-[11px] font-mono text-slate-300 space-y-1">
              <div className="text-emerald-400">✓ GAMUS Aerial Adaptation</div>
              <div className={metricStatus === 'AVAILABLE' ? 'text-emerald-400' : 'text-rose-400 font-bold'}>
                {metricStatus === 'AVAILABLE' ? '✓ True Camera-Surface Depth' : '✕ Metric Camera Depth Dataset (Missing)'}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* SECTION 3: MODE STATUS & SCIENTIFIC DISCLAIMER */}
      {currentModeInfo && (
        <div className={`p-4 rounded-xl border text-xs flex items-start gap-3 ${
          isBlocked
            ? 'bg-amber-950/40 border-amber-500/40 text-amber-200'
            : 'bg-cyan-950/40 border-cyan-500/40 text-cyan-200'
        }`}>
          <Info className="w-5 h-5 flex-shrink-0 mt-0.5" />
          <div className="space-y-1">
            <div className="font-bold uppercase tracking-wider">
              {selectedMode} — {currentModeInfo.status}
            </div>
            <p className="text-slate-300 leading-relaxed">
              {currentModeInfo.description}
            </p>
            {isBlocked && (
              <p className="text-rose-400 font-semibold pt-1">
                {currentModeInfo.blocked_reason}
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
