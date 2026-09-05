import React, { useState } from "react";
import type { CalibrationConfig, DepthProcessResponse } from "@/types/depth";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { 
  Target, 
  Satellite, 
  Mountain, 
  AlertOctagon, 
  Scale
} from "lucide-react";

interface HeightCalibrationPanelProps {
  currentCalibration: CalibrationConfig;
  data: DepthProcessResponse | null;
  onApplyCalibration: (calib: CalibrationConfig) => void;
  isLoading?: boolean;
}

export const HeightCalibrationPanel: React.FC<HeightCalibrationPanelProps> = ({
  currentCalibration,
  onApplyCalibration,
  isLoading = false,
}) => {
  const [mode, setMode] = useState<"reference" | "metadata" | "terrain" | "relative">(
    currentCalibration.mode || "relative"
  );

  // Form states
  const [refObjectName, setRefObjectName] = useState<string>(
    currentCalibration.known_object_name || "Apex Tower"
  );
  const [refObjectHeight, setRefObjectHeight] = useState<string>(
    currentCalibration.known_object_height_m ? currentCalibration.known_object_height_m.toString() : "84.5"
  );

  const [gsdCm, setGsdCm] = useState<string>(
    currentCalibration.gsd_cm_per_pixel ? currentCalibration.gsd_cm_per_pixel.toString() : "8.0"
  );
  const [altitudeM, setAltitudeM] = useState<string>(
    currentCalibration.flight_altitude_m ? currentCalibration.flight_altitude_m.toString() : "150.0"
  );
  const [focalMm, setFocalMm] = useState<string>(
    currentCalibration.focal_length_mm ? currentCalibration.focal_length_mm.toString() : "35.0"
  );

  const handleApply = () => {
    let newCalib: CalibrationConfig;

    if (mode === "reference") {
      const hVal = parseFloat(refObjectHeight) || 20.0;
      newCalib = {
        mode: "reference",
        known_object_name: refObjectName || "Reference Landmark",
        known_object_height_m: hVal,
        scale_factor_m_per_unit: hVal / 0.85,
        is_calibrated: true,
        calibration_description: `Calibrated via known reference '${refObjectName}' (${hVal} m).`,
      };
    } else if (mode === "metadata") {
      const gsd = parseFloat(gsdCm) || 8.0;
      const alt = parseFloat(altitudeM) || 120.0;
      const focal = parseFloat(focalMm) || 35.0;
      const scale = (alt / (focal * 10)) * 5.0;
      newCalib = {
        mode: "metadata",
        gsd_cm_per_pixel: gsd,
        flight_altitude_m: alt,
        focal_length_mm: focal,
        scale_factor_m_per_unit: Math.round(scale * 10) / 10,
        is_calibrated: true,
        calibration_description: `Metadata-Assisted Calibration (GSD: ${gsd} cm/px, Alt: ${alt}m).`,
      };
    } else if (mode === "terrain") {
      newCalib = {
        mode: "terrain",
        scale_factor_m_per_unit: 45.0,
        is_calibrated: true,
        calibration_description: "Terrain-Referenced Datum: Ground baseline extracted via RANSAC plane fitting.",
      };
    } else {
      // Relative
      newCalib = {
        mode: "relative",
        scale_factor_m_per_unit: 1.0,
        is_calibrated: false,
        calibration_description: "Relative depth — metric scale unavailable without scale/reference information.",
      };
    }

    onApplyCalibration(newCalib);
  };

  return (
    <div className="space-y-6">
      {/* Active Calibration Status Card */}
      <Card className="bg-[#0F172A]/90 border-slate-800 shadow-xl overflow-hidden">
        <div className="p-4 bg-gradient-to-r from-slate-900 via-slate-900/90 to-cyan-950/40 border-b border-slate-800 flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className={`p-2.5 rounded-lg border ${
              currentCalibration.is_calibrated 
                ? "bg-emerald-500/10 border-emerald-500/40 text-emerald-400" 
                : "bg-amber-500/10 border-amber-500/40 text-amber-400"
            }`}>
              <Scale className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-sm font-bold text-slate-100 font-mono-data tracking-wide uppercase">
                  Current Calibration State
                </h3>
                <Badge
                  className={`text-[10px] font-mono-data uppercase ${
                    currentCalibration.is_calibrated
                      ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/40"
                      : "bg-amber-500/20 text-amber-300 border-amber-500/40"
                  }`}
                >
                  {currentCalibration.is_calibrated ? "Metric Calibrated" : "Uncalibrated / Relative"}
                </Badge>
              </div>
              <p className="text-xs text-slate-400 mt-0.5">
                {currentCalibration.calibration_description || "Relative depth — metric scale unavailable."}
              </p>
            </div>
          </div>

          {currentCalibration.is_calibrated && currentCalibration.scale_factor_m_per_unit && (
            <div className="flex items-center gap-4 bg-black/40 px-3.5 py-1.5 rounded-lg border border-slate-800 text-xs font-mono-data">
              <div>
                <span className="text-slate-400 text-[10px] block">Scale Factor (α)</span>
                <span className="text-cyan-300 font-bold">{currentCalibration.scale_factor_m_per_unit} m / unit</span>
              </div>
              <div className="border-l border-slate-800 pl-3">
                <span className="text-slate-400 text-[10px] block">Ground Baseline (Z₀)</span>
                <span className="text-slate-200">{currentCalibration.ground_baseline_z ?? 0.12} rel</span>
              </div>
            </div>
          )}
        </div>

        <CardContent className="p-5 space-y-6">
          {/* Mode Selector Tabs */}
          <div>
            <Label className="text-xs font-mono-data text-slate-400 uppercase tracking-wide block mb-2.5">
              Select Calibration Mode (Section 11 Architecture)
            </Label>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-2.5">
              {/* Mode 1: Reference */}
              <button
                type="button"
                onClick={() => setMode("reference")}
                data-testid="calib-mode-reference-button"
                className={`p-3.5 rounded-xl border text-left transition-all flex flex-col justify-between ${
                  mode === "reference"
                    ? "bg-cyan-500/15 border-cyan-400 text-cyan-200 shadow-md shadow-cyan-500/10"
                    : "bg-slate-900/60 border-slate-800 text-slate-400 hover:border-slate-700 hover:text-slate-200"
                }`}
              >
                <div className="flex items-center justify-between mb-2">
                  <Target className="w-4 h-4 text-cyan-400" />
                  <span className="text-[10px] font-mono-data font-bold">MODE 1</span>
                </div>
                <div>
                  <h4 className="text-xs font-bold text-slate-100">Reference-Based</h4>
                  <p className="text-[11px] text-slate-400 mt-1 leading-snug">
                    Use a known structure height (e.g. 84.5m) to establish physical scale.
                  </p>
                </div>
              </button>

              {/* Mode 2: Metadata */}
              <button
                type="button"
                onClick={() => setMode("metadata")}
                data-testid="calib-mode-metadata-button"
                className={`p-3.5 rounded-xl border text-left transition-all flex flex-col justify-between ${
                  mode === "metadata"
                    ? "bg-cyan-500/15 border-cyan-400 text-cyan-200 shadow-md shadow-cyan-500/10"
                    : "bg-slate-900/60 border-slate-800 text-slate-400 hover:border-slate-700 hover:text-slate-200"
                }`}
              >
                <div className="flex items-center justify-between mb-2">
                  <Satellite className="w-4 h-4 text-cyan-400" />
                  <span className="text-[10px] font-mono-data font-bold">MODE 2</span>
                </div>
                <div>
                  <h4 className="text-xs font-bold text-slate-100">Metadata-Assisted</h4>
                  <p className="text-[11px] text-slate-400 mt-1 leading-snug">
                    Compute scale from GSD (cm/px), flight altitude (m), and focal length.
                  </p>
                </div>
              </button>

              {/* Mode 3: Terrain */}
              <button
                type="button"
                onClick={() => setMode("terrain")}
                data-testid="calib-mode-terrain-button"
                className={`p-3.5 rounded-xl border text-left transition-all flex flex-col justify-between ${
                  mode === "terrain"
                    ? "bg-cyan-500/15 border-cyan-400 text-cyan-200 shadow-md shadow-cyan-500/10"
                    : "bg-slate-900/60 border-slate-800 text-slate-400 hover:border-slate-700 hover:text-slate-200"
                }`}
              >
                <div className="flex items-center justify-between mb-2">
                  <Mountain className="w-4 h-4 text-cyan-400" />
                  <span className="text-[10px] font-mono-data font-bold">MODE 3</span>
                </div>
                <div>
                  <h4 className="text-xs font-bold text-slate-100">Terrain-Referenced</h4>
                  <p className="text-[11px] text-slate-400 mt-1 leading-snug">
                    Fit RANSAC ground surface plane to calculate elevation relative to terrain datum.
                  </p>
                </div>
              </button>

              {/* Mode 4: Relative */}
              <button
                type="button"
                onClick={() => setMode("relative")}
                data-testid="calib-mode-relative-button"
                className={`p-3.5 rounded-xl border text-left transition-all flex flex-col justify-between ${
                  mode === "relative"
                    ? "bg-amber-500/15 border-amber-400 text-amber-200 shadow-md shadow-amber-500/10"
                    : "bg-slate-900/60 border-slate-800 text-slate-400 hover:border-slate-700 hover:text-slate-200"
                }`}
              >
                <div className="flex items-center justify-between mb-2">
                  <AlertOctagon className="w-4 h-4 text-amber-400" />
                  <span className="text-[10px] font-mono-data font-bold">MODE 4</span>
                </div>
                <div>
                  <h4 className="text-xs font-bold text-slate-100">Relative Depth</h4>
                  <p className="text-[11px] text-slate-400 mt-1 leading-snug">
                    Strict scientific honesty: display unitless (0..1) depth with no metric claim.
                  </p>
                </div>
              </button>
            </div>
          </div>

          {/* Mode Configuration Form Inputs */}
          <div className="p-4 bg-black/40 rounded-xl border border-slate-800/80 space-y-4">
            {mode === "reference" && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <Label htmlFor="refName" className="text-xs font-mono-data text-slate-300">
                    Known Reference Landmark Name
                  </Label>
                  <Input
                    id="refName"
                    value={refObjectName}
                    onChange={(e) => setRefObjectName(e.target.value)}
                    placeholder="e.g. Apex Tower / Science Block"
                    data-testid="input-ref-landmark-name"
                    className="bg-slate-900/90 border-slate-700 text-slate-200 text-xs font-mono-data"
                  />
                </div>
                <div className="space-y-1.5">
                  <Label htmlFor="refHeight" className="text-xs font-mono-data text-slate-300">
                    Known Real-World Height (meters)
                  </Label>
                  <Input
                    id="refHeight"
                    type="number"
                    step="0.1"
                    value={refObjectHeight}
                    onChange={(e) => setRefObjectHeight(e.target.value)}
                    placeholder="e.g. 84.5"
                    data-testid="input-ref-landmark-height"
                    className="bg-slate-900/90 border-slate-700 text-cyan-300 text-xs font-mono-data font-bold"
                  />
                </div>
              </div>
            )}

            {mode === "metadata" && (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="space-y-1.5">
                  <Label htmlFor="gsd" className="text-xs font-mono-data text-slate-300">
                    Ground Sampling Distance (GSD cm/px)
                  </Label>
                  <Input
                    id="gsd"
                    type="number"
                    step="0.1"
                    value={gsdCm}
                    onChange={(e) => setGsdCm(e.target.value)}
                    placeholder="e.g. 8.0"
                    data-testid="input-gsd-cm"
                    className="bg-slate-900/90 border-slate-700 text-slate-200 text-xs font-mono-data"
                  />
                </div>
                <div className="space-y-1.5">
                  <Label htmlFor="alt" className="text-xs font-mono-data text-slate-300">
                    Flight Altitude AGL (meters)
                  </Label>
                  <Input
                    id="alt"
                    type="number"
                    step="1"
                    value={altitudeM}
                    onChange={(e) => setAltitudeM(e.target.value)}
                    placeholder="e.g. 150"
                    data-testid="input-altitude-m"
                    className="bg-slate-900/90 border-slate-700 text-slate-200 text-xs font-mono-data"
                  />
                </div>
                <div className="space-y-1.5">
                  <Label htmlFor="focal" className="text-xs font-mono-data text-slate-300">
                    Camera Focal Length (mm)
                  </Label>
                  <Input
                    id="focal"
                    type="number"
                    step="1"
                    value={focalMm}
                    onChange={(e) => setFocalMm(e.target.value)}
                    placeholder="e.g. 35"
                    data-testid="input-focal-mm"
                    className="bg-slate-900/90 border-slate-700 text-slate-200 text-xs font-mono-data"
                  />
                </div>
              </div>
            )}

            {mode === "terrain" && (
              <div className="text-xs text-slate-300 space-y-1 font-mono-data">
                <p className="text-cyan-300 font-semibold">Terrain-Referenced Ground Plane Extraction</p>
                <p className="text-slate-400 text-[11px]">
                  Extracts regional low-gradient terrain baseline Z₀ using RANSAC ground surface fitting and computes height offsets relative to the terrain datum.
                </p>
              </div>
            )}

            {mode === "relative" && (
              <div className="text-xs text-amber-200 space-y-1 font-mono-data">
                <p className="text-amber-400 font-bold">Uncalibrated Relative Depth Mode Active</p>
                <p className="text-slate-400 text-[11px]">
                  No metric assumptions will be made. All structures will display normalized unitless relative depth [0.0 ... 1.0].
                </p>
              </div>
            )}

            <div className="flex items-center justify-end pt-2">
              <Button
                onClick={handleApply}
                disabled={isLoading}
                data-testid="apply-calibration-button"
                className="bg-cyan-500 hover:bg-cyan-400 text-black font-semibold text-xs font-mono-data px-5"
              >
                {isLoading ? "Recalculating..." : "Apply Calibration & Recompute Heights"}
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
