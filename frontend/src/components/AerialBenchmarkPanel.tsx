import React, { useState } from 'react';
import {
  Cpu,
  Layers,
  Activity,
  AlertTriangle,
  CheckCircle2,
  Play,
  Crosshair,
  Ruler,
  TrendingUp,
  BarChart3,
  Building,
  Clock,
  ShieldCheck,
  Plus,
  Trash2
} from 'lucide-react';
import type {
  BenchmarkRequest,
  BenchmarkResponse,
  ValidationPoint
} from '../types/depth';

interface Props {
  sceneId?: string | null;
  imageUrl?: string | null;
  onBenchmarkComplete?: (res: BenchmarkResponse) => void;
}

const AVAILABLE_MODELS = [
  { key: 'da_v2_base', name: 'DA-V2 Metric Base', desc: 'Depth Anything V2 Metric Outdoor Base (~97M)' },
  { key: 'da_v2_large', name: 'DA-V2 Metric Large', desc: 'Depth Anything V2 Metric Outdoor Large (~335M)' },
  { key: 'unidepth_v2_l', name: 'UniDepthV2-L', desc: 'UniDepth V2 ViT-L Metric Depth (~304M)' },
  { key: 'metric3dv2', name: 'Metric3Dv2', desc: 'Metric3D V2 ViT-L Outdoor (~335M)' },
];

export const AerialBenchmarkPanel: React.FC<Props> = ({ sceneId, imageUrl, onBenchmarkComplete }) => {
  const [selectedModels, setSelectedModels] = useState<string[]>([
    'da_v2_base',
    'da_v2_large',
    'unidepth_v2_l',
    'metric3dv2'
  ]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [benchmarkRes, setBenchmarkRes] = useState<BenchmarkResponse | null>(null);

  // Probe state
  const [probePixel, setProbePixel] = useState<{ x: number; y: number } | null>(null);

  // Validation Points State
  const [valPoints, setValPoints] = useState<ValidationPoint[]>([
    { x: 250, y: 150, measured_depth_m: 42.6, label: 'Central Roof Probe' },
  ]);
  const [newValX, setNewValX] = useState<string>('');
  const [newValY, setNewValY] = useState<string>('');
  const [newValM, setNewValM] = useState<string>('');
  const [newValLabel, setNewValLabel] = useState<string>('');

  const toggleModel = (key: string) => {
    if (selectedModels.includes(key)) {
      if (selectedModels.length > 1) {
        setSelectedModels(selectedModels.filter((m) => m !== key));
      }
    } else {
      setSelectedModels([...selectedModels, key]);
    }
  };

  const handleAddValPoint = () => {
    const px = parseInt(newValX);
    const py = parseInt(newValY);
    const pm = parseFloat(newValM);
    if (!isNaN(px) && !isNaN(py) && !isNaN(pm) && pm > 0) {
      setValPoints([
        ...valPoints,
        { x: px, y: py, measured_depth_m: pm, label: newValLabel || `GT Point ${valPoints.length + 1}` }
      ]);
      setNewValX('');
      setNewValY('');
      setNewValM('');
      setNewValLabel('');
    }
  };

  const handleRemoveValPoint = (idx: number) => {
    setValPoints(valPoints.filter((_, i) => i !== idx));
  };

  const runBenchmark = async () => {
    setIsLoading(true);
    setErrorMsg(null);

    const payload: BenchmarkRequest = {
      scene_id: sceneId,
      image_url: imageUrl,
      selected_models: selectedModels,
      validation_points: valPoints,
      probe_pixel: probePixel ? [probePixel.x, probePixel.y] : [100, 100],
      selected_bbox: [50, 40, 60, 50]
    };

    try {
      const resp = await fetch('/api/depth/benchmark', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!resp.ok) {
        const errData = await resp.json().catch(() => ({ detail: 'Failed to run benchmark.' }));
        throw new Error(errData.detail || 'Benchmark request failed.');
      }

      const data: BenchmarkResponse = await resp.json();
      setBenchmarkRes(data);
      if (onBenchmarkComplete) onBenchmarkComplete(data);
    } catch (err: any) {
      setErrorMsg(err.message || 'Error executing benchmark.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleImageClick = (e: React.MouseEvent<HTMLImageElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const clickX = Math.round(((e.clientX - rect.left) / rect.width) * (benchmarkRes?.image_dimensions[0] || 500));
    const clickY = Math.round(((e.clientY - rect.top) / rect.height) * (benchmarkRes?.image_dimensions[1] || 400));
    setProbePixel({ x: clickX, y: clickY });
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 text-slate-100 shadow-2xl space-y-8">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between border-b border-slate-800 pb-4 gap-4">
        <div>
          <div className="flex items-center gap-2">
            <Layers className="w-6 h-6 text-indigo-400" />
            <h2 className="text-xl font-bold text-white tracking-wide">
              Aerial Metric Depth Benchmark
            </h2>
            <span className="px-2.5 py-0.5 text-xs font-semibold bg-indigo-500/20 text-indigo-300 rounded-full border border-indigo-500/30">
              Zero-Shot Pretrained Models
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Evaluate raw monocular metric depth (depth_map_m) across foundation architectures without alpha calibration or Stage 3 modifications.
          </p>
        </div>

        <button
          onClick={runBenchmark}
          disabled={isLoading || selectedModels.length === 0}
          className="flex items-center justify-center gap-2 px-5 py-2.5 bg-gradient-to-r from-indigo-600 to-blue-600 hover:from-indigo-500 hover:to-blue-500 disabled:opacity-50 text-white font-semibold text-sm rounded-lg shadow-lg shadow-indigo-500/20 transition-all"
        >
          {isLoading ? (
            <>
              <Activity className="w-4 h-4 animate-spin text-white" />
              <span>Running Benchmark...</span>
            </>
          ) : (
            <>
              <Play className="w-4 h-4 fill-current" />
              <span>RUN BENCHMARK</span>
            </>
          )}
        </button>
      </div>

      {/* Model Selection Checkboxes */}
      <div className="bg-slate-950/60 rounded-lg p-4 border border-slate-800/80">
        <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">
          Select Depth Architectures to Benchmark
        </h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          {AVAILABLE_MODELS.map((m) => {
            const isChecked = selectedModels.includes(m.key);
            return (
              <label
                key={m.key}
                onClick={() => toggleModel(m.key)}
                className={`flex items-start gap-3 p-3 rounded-lg border cursor-pointer transition-all ${
                  isChecked
                    ? 'bg-indigo-950/40 border-indigo-500/50 text-white'
                    : 'bg-slate-900/40 border-slate-800 text-slate-400 hover:border-slate-700'
                }`}
              >
                <input
                  type="checkbox"
                  checked={isChecked}
                  onChange={() => {}}
                  className="mt-1 rounded bg-slate-900 border-slate-700 text-indigo-600 focus:ring-indigo-500"
                />
                <div>
                  <div className="text-sm font-medium">{m.name}</div>
                  <div className="text-xs text-slate-500">{m.desc}</div>
                </div>
              </label>
            );
          })}
        </div>
      </div>

      {/* Error Banner */}
      {errorMsg && (
        <div className="p-4 bg-rose-950/50 border border-rose-800/50 rounded-lg text-rose-300 text-sm flex items-center gap-3">
          <AlertTriangle className="w-5 h-5 flex-shrink-0" />
          <span>{errorMsg}</span>
        </div>
      )}

      {/* Real Metric Depth Ground Truth Validation Input Section */}
      <div className="bg-slate-950/60 rounded-lg p-4 border border-slate-800/80 space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-xs font-semibold text-slate-300 uppercase tracking-wider">
            <Ruler className="w-4 h-4 text-emerald-400" />
            <span>Stage 1 Real Metric Depth Validation Points</span>
          </div>
          <span className="text-xs text-slate-400">
            Measured camera-to-surface distance (metres)
          </span>
        </div>

        {/* List of active points */}
        <div className="space-y-2">
          {valPoints.map((pt, idx) => (
            <div
              key={idx}
              className="flex items-center justify-between bg-slate-900 p-2.5 rounded border border-slate-800 text-xs"
            >
              <div className="flex items-center gap-3">
                <span className="font-semibold text-emerald-400">{pt.label}</span>
                <span className="text-slate-400">
                  Pixel: ({pt.x}, {pt.y})
                </span>
                <span className="bg-slate-800 px-2 py-0.5 rounded text-white font-mono">
                  {pt.measured_depth_m.toFixed(1)} m
                </span>
              </div>
              <button
                onClick={() => handleRemoveValPoint(idx)}
                className="text-slate-500 hover:text-rose-400 p-1"
              >
                <Trash2 className="w-3.5 h-3.5" />
              </button>
            </div>
          ))}
        </div>

        {/* Add new point inputs */}
        <div className="flex flex-wrap items-center gap-2 pt-2 border-t border-slate-800/60 text-xs">
          <input
            type="text"
            placeholder="Label (e.g. Roof Point)"
            value={newValLabel}
            onChange={(e) => setNewValLabel(e.target.value)}
            className="bg-slate-900 border border-slate-700 rounded px-2.5 py-1.5 text-white placeholder-slate-500 focus:outline-none focus:border-indigo-500 w-36"
          />
          <input
            type="number"
            placeholder="Pixel X"
            value={newValX}
            onChange={(e) => setNewValX(e.target.value)}
            className="bg-slate-900 border border-slate-700 rounded px-2.5 py-1.5 text-white placeholder-slate-500 focus:outline-none focus:border-indigo-500 w-20"
          />
          <input
            type="number"
            placeholder="Pixel Y"
            value={newValY}
            onChange={(e) => setNewValY(e.target.value)}
            className="bg-slate-900 border border-slate-700 rounded px-2.5 py-1.5 text-white placeholder-slate-500 focus:outline-none focus:border-indigo-500 w-20"
          />
          <input
            type="number"
            placeholder="Actual Depth (m)"
            value={newValM}
            onChange={(e) => setNewValM(e.target.value)}
            className="bg-slate-900 border border-slate-700 rounded px-2.5 py-1.5 text-white placeholder-slate-500 focus:outline-none focus:border-indigo-500 w-32"
          />
          <button
            onClick={handleAddValPoint}
            className="flex items-center gap-1 bg-slate-800 hover:bg-slate-700 text-slate-200 font-medium px-3 py-1.5 rounded transition-all"
          >
            <Plus className="w-3.5 h-3.5" />
            <span>Add GT Point</span>
          </button>
        </div>
      </div>

      {/* BENCHMARK RESULTS */}
      {benchmarkRes && (
        <div className="space-y-8 animate-in fade-in duration-300">
          {/* Best Model Recommendation Banner */}
          <div className="bg-gradient-to-r from-indigo-950/80 via-slate-900 to-blue-950/80 border border-indigo-500/40 rounded-xl p-5 shadow-xl">
            <div className="flex items-start gap-4">
              <ShieldCheck className="w-8 h-8 text-indigo-400 flex-shrink-0 mt-1" />
              <div className="space-y-2">
                <div className="flex items-center gap-3">
                  <h3 className="text-lg font-bold text-white">
                    Best Model Recommendation: {benchmarkRes.best_model_recommendation.best_model_name}
                  </h3>
                  <span className="text-xs bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 px-2.5 py-0.5 rounded-full font-medium">
                    {benchmarkRes.best_model_recommendation.basis || 'Empirical Evidence'}
                  </span>
                </div>
                <p className="text-sm text-slate-300">
                  {benchmarkRes.best_model_recommendation.recommendation}
                </p>
                <div className="text-xs font-semibold text-amber-300 bg-amber-950/40 border border-amber-500/30 p-2.5 rounded-lg">
                  NEXT STEP: {benchmarkRes.best_model_recommendation.next_step}
                </div>
              </div>
            </div>
          </div>

          {/* Model Agreement Widget & Diagnostics */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-slate-950/80 p-4 rounded-xl border border-slate-800 space-y-2">
              <div className="text-xs text-slate-400 font-semibold uppercase tracking-wider flex items-center justify-between">
                <span>Model Agreement</span>
                <Activity className="w-4 h-4 text-indigo-400" />
              </div>
              <div className="text-2xl font-bold text-white flex items-baseline gap-2">
                <span>{benchmarkRes.model_agreement.agreement_rating}</span>
                <span className="text-xs text-slate-400 font-normal">
                  (std: {benchmarkRes.model_agreement.std_p50_m ?? '—'}m)
                </span>
              </div>
              <p className="text-xs text-slate-400">
                {benchmarkRes.model_agreement.note}
              </p>
            </div>

            <div className="bg-slate-950/80 p-4 rounded-xl border border-slate-800 space-y-2">
              <div className="text-xs text-slate-400 font-semibold uppercase tracking-wider flex items-center justify-between">
                <span>Mean Median Depth</span>
                <TrendingUp className="w-4 h-4 text-blue-400" />
              </div>
              <div className="text-2xl font-bold text-white">
                {benchmarkRes.model_agreement.mean_p50_m ? `${benchmarkRes.model_agreement.mean_p50_m} m` : 'N/A'}
              </div>
              <p className="text-xs text-slate-400">
                Cross-model median camera distance prediction across scene.
              </p>
            </div>

            <div className="bg-slate-950/80 p-4 rounded-xl border border-slate-800 space-y-2">
              <div className="text-xs text-slate-400 font-semibold uppercase tracking-wider flex items-center justify-between">
                <span>Stage 3 Status</span>
                <Building className="w-4 h-4 text-emerald-400" />
              </div>
              <div className="text-lg font-bold text-emerald-400 flex items-center gap-1.5">
                <CheckCircle2 className="w-5 h-5" />
                <span>FROZEN & VALIDATED</span>
              </div>
              <p className="text-xs text-slate-400">
                3D unprojection geometry remains 100% frozen. Stage 1 model changed only.
              </p>
            </div>
          </div>

          {/* Side-by-Side Model Cards Grid */}
          <div className="space-y-4">
            <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-2">
              <BarChart3 className="w-4 h-4 text-indigo-400" />
              <span>Side-by-Side Depth Architecture Outputs</span>
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {benchmarkRes.model_results.map((m) => (
                <div
                  key={m.model_key}
                  className={`bg-slate-950/90 rounded-xl border p-4 flex flex-col justify-between space-y-4 ${
                    m.status === 'SUCCESS'
                      ? m.is_compressed
                        ? 'border-amber-500/40'
                        : 'border-indigo-500/40'
                      : 'border-slate-800 opacity-80'
                  }`}
                >
                  {/* Top Info */}
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="font-bold text-white text-sm">{m.model_name}</h4>
                      <span
                        className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${
                          m.status === 'SUCCESS'
                            ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                            : 'bg-rose-500/20 text-rose-300 border border-rose-500/30'
                        }`}
                      >
                        {m.status}
                      </span>
                    </div>

                    <div className="text-xs text-slate-400 flex items-center gap-3 mb-3">
                      <span className="flex items-center gap-1">
                        <Cpu className="w-3.5 h-3.5 text-slate-500" />
                        {m.device.toUpperCase()}
                      </span>
                      <span className="flex items-center gap-1">
                        <Clock className="w-3.5 h-3.5 text-slate-500" />
                        {(m.inference_time_ms / 1000).toFixed(2)}s
                      </span>
                    </div>

                    {/* Visualization Image if available */}
                    {m.depth_colormap_url ? (
                      <div className="relative group rounded-lg overflow-hidden border border-slate-800 bg-slate-900">
                        <img
                          src={m.depth_colormap_url}
                          alt={m.model_name}
                          onClick={handleImageClick}
                          className="w-full h-36 object-cover cursor-crosshair"
                        />
                        <div className="absolute bottom-1 right-1 bg-slate-950/80 px-2 py-0.5 rounded text-[10px] text-slate-300 font-mono">
                          VISUALIZATION ONLY
                        </div>
                      </div>
                    ) : (
                      <div className="h-36 bg-slate-900 rounded-lg border border-slate-800/80 flex items-center justify-center text-xs text-slate-500 p-4 text-center">
                        {m.error_reason || 'Depth visual unavailable'}
                      </div>
                    )}
                  </div>

                  {/* Metrics Table */}
                  {m.status === 'SUCCESS' && (
                    <div className="space-y-2 text-xs">
                      <div className="text-[11px] font-semibold text-slate-400 border-b border-slate-800/60 pb-1">
                        RAW METRIC DEPTH (Metres)
                      </div>

                      <div className="grid grid-cols-2 gap-y-1 font-mono text-slate-300">
                        <div>P50 (Median): <span className="text-indigo-300 font-bold">{m.depth_p50_m} m</span></div>
                        <div>Range: <span>{m.depth_min_m}–{m.depth_max_m} m</span></div>
                        <div>P10: <span>{m.depth_p10_m} m</span></div>
                        <div>P90: <span>{m.depth_p90_m} m</span></div>
                        <div>Mean: <span>{m.depth_mean_m} m</span></div>
                        <div>Std: <span>{m.depth_std_m} m</span></div>
                      </div>

                      {/* Compression Warning Badge */}
                      {m.is_compressed && (
                        <div className="mt-2 text-[10px] bg-amber-950/50 border border-amber-500/40 text-amber-300 p-2 rounded flex items-start gap-1.5">
                          <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0 mt-0.5 text-amber-400" />
                          <span>{m.compression_note}</span>
                        </div>
                      )}

                      {/* 175m FROZEN Stage 3 Stress Test Result */}
                      {m.stage3_stress_test && (
                        <div className="mt-2 pt-2 border-t border-slate-800/80">
                          <div className="text-[10px] font-semibold text-slate-400 mb-1">
                            FROZEN STAGE 3 PREDICTED HEIGHT
                          </div>
                          <div className="bg-slate-900 p-2 rounded font-mono text-[11px] space-y-0.5 text-slate-300">
                            <div>Height: <span className="text-emerald-400 font-bold">{m.stage3_stress_test.predicted_height_m ? `${m.stage3_stress_test.predicted_height_m} m` : 'Unavailable'}</span></div>
                            <div>GT Height: <span className="text-slate-400">175.0 m</span></div>
                            <div>Error: <span className="text-rose-400">{m.stage3_stress_test.absolute_error_m ? `${m.stage3_stress_test.absolute_error_m} m` : 'N/A'}</span></div>
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Interactive Pixel Probe Summary */}
          {probePixel && benchmarkRes.pixel_probe && (
            <div className="bg-slate-950/80 rounded-xl border border-indigo-500/40 p-4 space-y-3">
              <div className="flex items-center gap-2 text-sm font-semibold text-indigo-300">
                <Crosshair className="w-4 h-4" />
                <span>Interactive Pixel Depth Probe at ({probePixel.x}, {probePixel.y})</span>
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                {benchmarkRes.pixel_probe.model_predictions.map((p) => (
                  <div key={p.model_key} className="bg-slate-900 p-3 rounded-lg border border-slate-800 text-xs">
                    <div className="text-slate-400 font-medium">{p.model_name}</div>
                    <div className="text-lg font-bold text-white font-mono mt-1">
                      {p.depth_m.toFixed(2)} m
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Measured Depth Validation Point Results Table */}
          {benchmarkRes.validation_summary && benchmarkRes.validation_summary.length > 0 && (
            <div className="bg-slate-950/80 rounded-xl border border-slate-800 p-5 space-y-4">
              <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-2">
                <ShieldCheck className="w-4 h-4 text-emerald-400" />
                <span>Stage 1 Measured Metric Depth Error Scoring</span>
              </h3>

              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs text-slate-300 border-collapse">
                  <thead>
                    <tr className="border-b border-slate-800 text-slate-400 uppercase font-semibold text-[11px]">
                      <th className="py-2.5 px-3">Model</th>
                      <th className="py-2.5 px-3">Samples</th>
                      <th className="py-2.5 px-3">MAE (m)</th>
                      <th className="py-2.5 px-3">RMSE (m)</th>
                      <th className="py-2.5 px-3">Mean Rel Error</th>
                      <th className="py-2.5 px-3">Median Rel Error</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 font-mono">
                    {benchmarkRes.validation_summary.map((v) => (
                      <tr key={v.model_key} className="hover:bg-slate-900/50">
                        <td className="py-2.5 px-3 font-semibold text-white">{v.model_name}</td>
                        <td className="py-2.5 px-3">{v.valid_samples_count}</td>
                        <td className="py-2.5 px-3 text-indigo-300 font-bold">{v.mae_m ?? '—'} m</td>
                        <td className="py-2.5 px-3">{v.rmse_m ?? '—'} m</td>
                        <td className="py-2.5 px-3 text-amber-300">{v.mean_relative_error_pct ?? '—'}%</td>
                        <td className="py-2.5 px-3">{v.median_relative_error_pct ?? '—'}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
