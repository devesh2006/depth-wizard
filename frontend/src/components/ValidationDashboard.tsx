import React, { useEffect, useState } from "react";
import type { ValidationOverview } from "@/types/depth";
import { apiGet } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ShieldCheck, AlertTriangle } from "lucide-react";
import { 
  ResponsiveContainer, 
  ScatterChart, 
  Scatter, 
  XAxis, 
  YAxis, 
  Tooltip, 
  CartesianGrid
} from "recharts";

export const ValidationDashboard: React.FC = () => {
  const [data, setData] = useState<ValidationOverview | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [selectedSceneFilter, setSelectedSceneFilter] = useState<string>("all");

  useEffect(() => {
    async function loadValidation() {
      try {
        const res = await apiGet<ValidationOverview>("/depth/validation");
        setData(res);
      } catch (err) {
        console.error("Failed to load validation data", err);
      } finally {
        setIsLoading(false);
      }
    }
    loadValidation();
  }, []);

  if (isLoading || !data) {
    return (
      <div className="flex flex-col items-center justify-center p-16 border border-slate-800 rounded-2xl bg-[#0F172A]/40 text-center">
        <div className="w-10 h-10 rounded-full border-2 border-cyan-500 border-t-transparent animate-spin mb-4" />
        <h3 className="text-sm font-bold text-slate-200 font-mono-data">Loading LiDAR Benchmark Validations...</h3>
      </div>
    );
  }

  // Honest empty state: no verified reference dataset registered (SIH26175 S18).
  if (!data.data_available) {
    return (
      <div className="space-y-5" data-testid="validation-unavailable-panel">
        <div className="p-5 bg-amber-500/10 border border-amber-500/30 rounded-xl space-y-3">
          <div className="flex items-center gap-2.5">
            <AlertTriangle className="w-5 h-5 text-amber-400 shrink-0" />
            <h3
              className="text-sm font-bold text-amber-200 font-mono-data uppercase tracking-wide"
              data-testid="validation-unavailable-heading"
            >
              Validation data not available
            </h3>
          </div>
          <p className="text-xs text-amber-100/90 leading-relaxed max-w-4xl" data-testid="validation-unavailable-reason">
            {data.unavailable_reason}
          </p>
          <p className="text-[11px] text-slate-400 font-mono-data leading-relaxed max-w-4xl">
            No MAE, RMSE or error value is displayed here, because publishing accuracy figures without
            an independent reference dataset would be scientifically indefensible. Depth estimation,
            3D reconstruction and reference-based calibration are unaffected and fully demonstrable.
          </p>
        </div>

        <Card className="bg-[#0F172A]/90 border-slate-800 shadow-xl">
          <CardHeader className="p-4 pb-2 border-b border-slate-800">
            <CardTitle className="text-xs font-bold text-slate-100 font-mono-data tracking-wide uppercase">
              Required to populate real validation metrics
            </CardTitle>
            <CardDescription className="text-[11px] text-slate-400 font-mono-data">
              Once a reference dataset meeting these criteria is registered, this dashboard computes
              MAE, RMSE, median and relative error from live pipeline predictions automatically.
            </CardDescription>
          </CardHeader>
          <CardContent className="p-4">
            <ol className="space-y-2" data-testid="validation-requirements-list">
              {data.requirements.map((r, i) => (
                <li key={i} className="flex items-start gap-2.5 text-xs text-slate-300 font-mono-data">
                  <span className="mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center rounded bg-cyan-500/15 border border-cyan-500/40 text-[9px] font-bold text-cyan-300">
                    {i + 1}
                  </span>
                  <span className="leading-relaxed">{r}</span>
                </li>
              ))}
            </ol>
          </CardContent>
        </Card>
      </div>
    );
  }

  // Filtered scatter points
  const filteredScatter = selectedSceneFilter === "all"
    ? data.scatter_points
    : data.scatter_points.filter((p) => p.scene_id === selectedSceneFilter);

  return (
    <div className="space-y-6">
      {/* Top Header & Scientific Disclaimer */}
      <div className="p-4 bg-gradient-to-r from-slate-900 via-[#0F172A] to-cyan-950/40 border border-slate-800 rounded-xl flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-lg bg-cyan-500/10 border border-cyan-500/40 text-cyan-400">
            <ShieldCheck className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-sm font-bold text-slate-100 font-mono-data uppercase tracking-wide">
                SIH26175 Scientific Validation Framework
              </h3>
              <Badge className="bg-emerald-500/20 text-emerald-300 border-emerald-500/40 text-[10px] font-mono-data">
                Reference-Scored
              </Badge>
            </div>
            <p className="text-xs text-slate-400 mt-0.5 max-w-3xl">
              Error metrics are computed from live pipeline predictions against registered reference heights. Zero fabricated values.
            </p>
          </div>
        </div>

        <div className="text-right font-mono-data text-xs text-slate-400">
          <span>Datasets: <strong className="text-slate-200">4 Benchmark Scenes</strong></span>
          <span className="mx-2">|</span>
          <span>Tested Structures: <strong className="text-cyan-300">{data.total_structures_measured}</strong></span>
        </div>
      </div>

      {/* Primary 4 Metric Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 font-mono-data">
        {/* MAE */}
        <Card className="bg-[#0F172A]/90 border-slate-800 shadow-md">
          <CardContent className="p-4">
            <span className="text-[11px] text-slate-400 block uppercase font-bold">MAE (Mean Absolute Error)</span>
            <div className="text-3xl font-extrabold text-cyan-300 mt-1">
              {data.overall_mae_m} <span className="text-sm text-slate-400 font-normal">meters</span>
            </div>
            <span className="text-[10px] text-emerald-400 mt-1 block">
              Benchmark Target: &lt; 1.5m
            </span>
          </CardContent>
        </Card>

        {/* RMSE */}
        <Card className="bg-[#0F172A]/90 border-slate-800 shadow-md">
          <CardContent className="p-4">
            <span className="text-[11px] text-slate-400 block uppercase font-bold">RMSE (Root Mean Square)</span>
            <div className="text-3xl font-extrabold text-slate-100 mt-1">
              {data.overall_rmse_m} <span className="text-sm text-slate-400 font-normal">meters</span>
            </div>
            <span className="text-[10px] text-slate-400 mt-1 block">
              Tolerates heavy outlier penalties
            </span>
          </CardContent>
        </Card>

        {/* Median Error */}
        <Card className="bg-[#0F172A]/90 border-slate-800 shadow-md">
          <CardContent className="p-4">
            <span className="text-[11px] text-slate-400 block uppercase font-bold">Median Absolute Error</span>
            <div className="text-3xl font-extrabold text-emerald-400 mt-1">
              {data.overall_median_err_m} <span className="text-sm text-slate-400 font-normal">meters</span>
            </div>
            <span className="text-[10px] text-emerald-400 mt-1 block">
              50% of buildings within {data.overall_median_err_m}m
            </span>
          </CardContent>
        </Card>

        {/* MRE % */}
        <Card className="bg-[#0F172A]/90 border-slate-800 shadow-md">
          <CardContent className="p-4">
            <span className="text-[11px] text-slate-400 block uppercase font-bold">Mean Relative Error</span>
            <div className="text-3xl font-extrabold text-amber-300 mt-1">
              {data.overall_mre_pct}%
            </div>
            <span className="text-[10px] text-slate-400 mt-1 block">
              Average proportional error
            </span>
          </CardContent>
        </Card>
      </div>

      {/* Charts Row: Scatter Plot + Per-Scene Benchmarks */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        {/* Scatter Plot: Ground Truth vs Prediction (7 cols) */}
        <Card className="lg:col-span-7 bg-[#0F172A]/90 border-slate-800 shadow-xl flex flex-col">
          <CardHeader className="p-4 pb-2 border-b border-slate-800 flex flex-row items-center justify-between">
            <div>
              <CardTitle className="text-xs font-bold text-slate-100 font-mono-data tracking-wide uppercase">
                Reference Height vs DepthWizard Estimated Height
              </CardTitle>
              <CardDescription className="text-[11px] text-slate-400 font-mono-data">
                Points closer to the diagonal line indicate higher accuracy
              </CardDescription>
            </div>
            {/* Filter */}
            <Select value={selectedSceneFilter} onValueChange={setSelectedSceneFilter}>
              <SelectTrigger
                data-testid="validation-scene-filter"
                className="h-8 w-[210px] bg-slate-900 border-slate-700 text-xs text-slate-300 font-mono-data"
              >
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="bg-slate-900 border-slate-700 text-slate-200">
                <SelectItem value="all" className="text-xs font-mono-data">
                  All Scenes ({data.scatter_points.length})
                </SelectItem>
                {data.benchmarks.map((b) => (
                  <SelectItem key={b.scene_id} value={b.scene_id} className="text-xs font-mono-data">
                    {b.scene_name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </CardHeader>

          <CardContent className="p-4 flex-1">
            <div className="h-[280px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis 
                    type="number" 
                    dataKey="gt_height" 
                    name="Ground Truth" 
                    unit="m" 
                    stroke="#64748b"
                    fontSize={11}
                    fontFamily="JetBrains Mono"
                    domain={[0, 95]}
                  />
                  <YAxis 
                    type="number" 
                    dataKey="pred_height" 
                    name="DepthWizard" 
                    unit="m" 
                    stroke="#64748b"
                    fontSize={11}
                    fontFamily="JetBrains Mono"
                    domain={[0, 95]}
                  />
                  <Tooltip
                    content={({ payload }) => {
                      if (!payload || !payload.length) return null;
                      const pt = payload[0].payload;
                      return (
                        <div className="p-2.5 bg-slate-900/95 border border-cyan-500/50 rounded-lg text-xs font-mono-data text-slate-200 shadow-xl space-y-1">
                          <p className="text-cyan-400 font-bold">{pt.building}</p>
                          <p className="text-[11px] text-slate-400">{pt.scene}</p>
                          <div className="pt-1 border-t border-slate-800 space-y-0.5">
                            <p>Reference: <strong>{pt.gt_height} m</strong></p>
                            <p>DepthWizard Estimated: <strong className="text-cyan-300">{pt.pred_height} m</strong></p>
                            <p>Error: <strong className="text-emerald-400">{pt.error} m</strong></p>
                          </div>
                        </div>
                      );
                    }}
                  />
                  <Scatter name="Buildings" data={filteredScatter} fill="#00E5FF" />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Per-Scene MAE Breakdown (5 cols) */}
        <Card className="lg:col-span-5 bg-[#0F172A]/90 border-slate-800 shadow-xl flex flex-col">
          <CardHeader className="p-4 pb-2 border-b border-slate-800">
            <CardTitle className="text-xs font-bold text-slate-100 font-mono-data tracking-wide uppercase">
              Per-Scene MAE Accuracy Benchmarks
            </CardTitle>
            <CardDescription className="text-[11px] text-slate-400 font-mono-data">
              Per-scene error, computed live from registered reference heights
            </CardDescription>
          </CardHeader>

          <CardContent className="p-4 flex-1 space-y-3">
            {data.benchmarks.map((b) => (
              <div 
                key={b.scene_id} 
                className="p-3 bg-black/40 rounded-xl border border-slate-800 font-mono-data text-xs flex items-center justify-between gap-3"
              >
                <div>
                  <h4 className="text-xs font-bold text-slate-100">{b.scene_name}</h4>
                  <span className="text-[10px] text-slate-400 block">{b.sensor_type}</span>
                </div>
                <div className="text-right">
                  <span className="text-sm font-extrabold text-cyan-300 block">
                    MAE: {b.mae_m} m
                  </span>
                  <span className="text-[10px] text-slate-400 block">
                    RMSE: {b.rmse_m}m | MRE: {b.mean_relative_error_pct}%
                  </span>
                </div>
              </div>
            ))}
          </CardContent>
        </Card>
      </div>

      {/* Per-Building Validation Table */}
      <Card className="bg-[#0F172A]/90 border-slate-800 shadow-xl overflow-hidden">
        <CardHeader className="p-4 pb-2 border-b border-slate-800">
          <CardTitle className="text-xs font-bold text-slate-100 font-mono-data tracking-wide uppercase">
            Comprehensive Ground-Truth Validation Ledger
          </CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          <Table>
            <TableHeader className="bg-black/40 font-mono-data text-xs">
              <TableRow className="border-slate-800 hover:bg-transparent">
                <TableHead className="text-slate-300">Scene / Dataset</TableHead>
                <TableHead className="text-slate-300">Structure Name</TableHead>
                <TableHead className="text-slate-300 text-right">Reference Height</TableHead>
                <TableHead className="text-slate-300 text-right">DepthWizard Prediction</TableHead>
                <TableHead className="text-slate-300 text-right">Absolute Error</TableHead>
                <TableHead className="text-slate-300 text-right">Relative Error (%)</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody className="font-mono-data text-xs">
              {data.benchmarks.flatMap((b) =>
                b.sample_measurements.map((m) => (
                  <TableRow key={m.building_id} className="border-slate-800/60 hover:bg-slate-900/40">
                    <TableCell className="text-slate-400">{b.scene_name}</TableCell>
                    <TableCell className="font-semibold text-slate-200">{m.building_name}</TableCell>
                    <TableCell className="text-right text-emerald-400 font-bold">{m.gt_height_m} m</TableCell>
                    <TableCell className="text-right text-cyan-300 font-bold">{m.pred_height_m} m</TableCell>
                    <TableCell className="text-right text-slate-200">{m.error_m} m</TableCell>
                    <TableCell className="text-right text-amber-300">{m.rel_error_pct}%</TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
};
