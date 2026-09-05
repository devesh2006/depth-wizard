import React from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { ShieldCheck, AlertTriangle } from "lucide-react";

interface MethodologyModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export const MethodologyModal: React.FC<MethodologyModalProps> = ({
  open,
  onOpenChange,
}) => {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-3xl bg-[#0F172A] border-slate-700 text-slate-100 max-h-[85vh] overflow-y-auto">
        <DialogHeader className="border-b border-slate-800 pb-3">
          <div className="flex items-center gap-2">
            <div className="p-2 rounded bg-cyan-500/10 border border-cyan-500/30 text-cyan-400">
              <ShieldCheck className="w-5 h-5" />
            </div>
            <div>
              <DialogTitle className="text-base font-bold font-mono-data text-slate-100">
                DepthWizard Scientific Methodology & Theoretical Foundations
              </DialogTitle>
              <DialogDescription className="text-xs text-slate-400 font-mono-data">
                SIH Problem Statement SIH26175 — Monocular Depth vs Photogrammetry vs LiDAR
              </DialogDescription>
            </div>
          </div>
        </DialogHeader>

        <div className="space-y-5 text-xs text-slate-300 leading-relaxed pt-2">
          {/* Core Disclaimer Box */}
          <div className="p-3.5 bg-amber-500/10 border border-amber-500/30 rounded-xl space-y-1 text-amber-200">
            <div className="flex items-center gap-2 text-xs font-bold text-amber-300 font-mono-data">
              <AlertTriangle className="w-4 h-4" />
              Scientific & Engineering Principles
            </div>
            <p className="text-[11px] leading-relaxed">
              DepthWizard does not claim that single-view RGB imagery replaces LiDAR or multi-view photogrammetry. Monocular depth inherently contains scale and shift ambiguity (Z_metric = s * Z_rel + t). Without reference ground landmarks, GSD metadata, or terrain datum, metric heights cannot be mathematically derived.
            </p>
          </div>

          {/* Distinction Matrix */}
          <div>
            <h4 className="text-xs font-bold text-slate-100 uppercase font-mono-data mb-2">
              Three Fundamental Elevation Concepts
            </h4>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-2.5 font-mono-data text-xs">
              <div className="p-3 bg-black/40 rounded-lg border border-slate-800 space-y-1">
                <span className="text-cyan-400 font-bold block">A. Relative Depth</span>
                <p className="text-[11px] text-slate-400">
                  Unitless values [0.0 ... 1.0] representing which visual features are closer or farther from the camera sensor plane.
                </p>
              </div>
              <div className="p-3 bg-black/40 rounded-lg border border-slate-800 space-y-1">
                <span className="text-emerald-400 font-bold block">B. Metric Depth</span>
                <p className="text-[11px] text-slate-400">
                  True physical distance from the camera optical center to the target surface in meters (Z_c).
                </p>
              </div>
              <div className="p-3 bg-black/40 rounded-lg border border-slate-800 space-y-1">
                <span className="text-amber-400 font-bold block">C. Building Height</span>
                <p className="text-[11px] text-slate-400">
                  Vertical elevation difference between the rooftop peak and the local surrounding ground plane datum (ΔH = Z_roof - Z_0).
                </p>
              </div>
            </div>
          </div>

          {/* Mathematical Back-Projection Formulation */}
          <div className="p-3.5 bg-black/50 rounded-xl border border-slate-800 space-y-2 font-mono-data">
            <h4 className="text-xs font-bold text-slate-200 uppercase">
              Mathematical Back-Projection & Calibration Formulas
            </h4>
            <div className="p-2.5 bg-slate-900 rounded border border-slate-800 text-[11px] space-y-1 text-cyan-300">
              <p>• <strong>Pinhole Back-Projection:</strong> X = (u - cx) * Z / fx,  Y = (v - cy) * Z / fy</p>
              <p>• <strong>Reference Scale Factor:</strong> α = H_ref / ΔZ_ref (meters / unit)</p>
              <p>• <strong>Calibrated Height:</strong> H = α * (Z_roof_90% - Z_ground_10%)</p>
              <p>• <strong>Uncertainty Margin:</strong> σ_H = H * sqrt((σ_ref / H_ref)^2 + (σ_patch / ΔZ)^2)</p>
            </div>
          </div>

          {/* Sensor Comparison Matrix */}
          <div>
            <h4 className="text-xs font-bold text-slate-100 uppercase font-mono-data mb-2">
              Technology Comparison Matrix
            </h4>
            <Table>
              <TableHeader className="bg-black/40 font-mono-data text-xs">
                <TableRow className="border-slate-800">
                  <TableHead className="text-slate-300">Approach</TableHead>
                  <TableHead className="text-slate-300">Data Input</TableHead>
                  <TableHead className="text-slate-300">Vertical Accuracy</TableHead>
                  <TableHead className="text-slate-300">Deployment Speed</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody className="font-mono-data text-xs">
                <TableRow className="border-slate-800">
                  <TableCell className="text-cyan-400 font-bold">DepthWizard (Ours)</TableCell>
                  <TableCell>Single RGB Image + Reference</TableCell>
                  <TableCell className="text-amber-300 font-bold">Not yet quantified</TableCell>
                  <TableCell className="text-cyan-300 font-bold">~3 s measured (CPU)</TableCell>
                </TableRow>
                <TableRow className="border-slate-800">
                  <TableCell className="text-slate-300">Airborne LiDAR</TableCell>
                  <TableCell>Active Laser Scanning</TableCell>
                  <TableCell className="text-emerald-400 font-bold">±0.05m - ±0.15m</TableCell>
                  <TableCell className="text-slate-400">Hours / Flight Mission</TableCell>
                </TableRow>
                <TableRow className="border-slate-800">
                  <TableCell className="text-slate-300">Stereo Photogrammetry</TableCell>
                  <TableCell>60-80% Overlapping Pairs</TableCell>
                  <TableCell className="text-emerald-400 font-bold">±0.2m - ±0.5m</TableCell>
                  <TableCell className="text-slate-400">Minutes / Bundle Adjust</TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};
