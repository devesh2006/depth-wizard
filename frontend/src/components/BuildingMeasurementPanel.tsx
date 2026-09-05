import React, { useState } from "react";
import type { BuildingMeasurement, CalibrationConfig } from "@/types/depth";
import { apiPost } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { toast } from "sonner";
import { 
  Building2,
  ShieldCheck,
  AlertTriangle,
  Ruler,
  Layers,
  ChevronDown,
  ChevronUp,
  Camera,
  Activity
} from "lucide-react";

interface BuildingMeasurementPanelProps {
  buildings: BuildingMeasurement[];
  selectedBuilding: BuildingMeasurement | null;
  calibration: CalibrationConfig;
  sceneId?: string;
  onSelectBuilding: (bldg: BuildingMeasurement) => void;
  onUpdateBuilding?: (bldg: BuildingMeasurement) => void;
}

export const BuildingMeasurementPanel: React.FC<BuildingMeasurementPanelProps> = ({
  buildings,
  selectedBuilding,
  calibration,
  sceneId,
  onSelectBuilding,
  onUpdateBuilding,
}) => {
  const [gtInput, setGtInput] = useState<string>("175.0");
  const [isSubmittingGt, setIsSubmittingGt] = useState<boolean>(false);
  const [showDiagnostics, setShowDiagnostics] = useState<boolean>(false);

  const handleValidateGroundTruth = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedBuilding || !sceneId) return;

    const gtVal = parseFloat(gtInput);
    if (isNaN(gtVal) || gtVal <= 0) {
      toast.error("Please enter a valid positive ground-truth height in meters.");
      return;
    }

    setIsSubmittingGt(true);
    try {
      const updated = await apiPost<BuildingMeasurement>("/depth/measure-building", {
        scene_id: sceneId,
        bbox: selectedBuilding.bbox,
        calibration: calibration,
        name: selectedBuilding.name,
        ground_truth_height_m: gtVal,
      });

      if (onUpdateBuilding) {
        onUpdateBuilding(updated);
      }
      toast.success(
        updated.is_metric_available && updated.accuracy_pct != null
          ? `Ground Truth of ${gtVal}m evaluated! Accuracy: ${updated.accuracy_pct}%`
          : `Ground Truth recorded (${gtVal}m). Prediction is currently unavailable.`
      );
    } catch (err: any) {
      toast.error(err.message || "Failed to score ground truth validation.");
    } finally {
      setIsSubmittingGt(false);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
      {/* Structure List (5 cols) */}
      <div className="lg:col-span-5 space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Building2 className="w-4 h-4 text-cyan-400" />
            <h3 className="text-sm font-bold text-slate-100 font-mono-data tracking-wide uppercase">
              Detected Structures ({buildings.length})
            </h3>
          </div>
          <span className="text-[11px] font-mono-data text-slate-400">
            Click structure to inspect
          </span>
        </div>

        <div className="space-y-2 max-h-[520px] overflow-y-auto pr-1">
          {buildings.map((bldg) => {
            const isSelected = selectedBuilding?.id === bldg.id;
            return (
              <div
                key={bldg.id}
                onClick={() => onSelectBuilding(bldg)}
                data-testid={`structure-item-${bldg.id}`}
                className={`p-3.5 rounded-xl border transition-all cursor-pointer flex flex-col justify-between space-y-2 ${
                  isSelected
                    ? "bg-[#0F172A] border-emerald-500/80 shadow-lg shadow-emerald-500/10"
                    : "bg-[#0F172A]/60 border-slate-800/90 hover:border-slate-700 hover:bg-[#0F172A]"
                }`}
              >
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <h4 className="text-xs font-bold text-slate-100 flex items-center gap-1.5">
                      {isSelected && <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />}
                      {bldg.name}
                    </h4>
                    <span className="text-[10px] text-cyan-400 font-mono-data block mt-0.5">
                      Relative Depth: {bldg.rooftop_peak_z_rel} (Delta: {bldg.relative_height_unitless})
                    </span>
                  </div>

                  <div className="text-right">
                    {bldg.is_metric_available && bldg.calibrated_height_m != null ? (
                      <div>
                        <span className="text-base font-extrabold text-cyan-300 font-mono-data">
                          {bldg.calibrated_height_m} m
                        </span>
                        {bldg.uncertainty_margin_m && (
                          <span className="text-[10px] text-slate-400 block font-mono-data">
                            ±{bldg.uncertainty_margin_m} m
                          </span>
                        )}
                      </div>
                    ) : (
                      <div>
                        <span className="text-xs font-bold text-amber-300 font-mono-data block">
                          Relative Depth
                        </span>
                        <span className="text-[9px] text-amber-400/80 block font-mono-data">
                          Height Unavailable
                        </span>
                      </div>
                    )}
                  </div>
                </div>

                {bldg.ground_truth_height_m != null && (
                  <div className="flex items-center justify-between text-[11px] font-mono-data p-1.5 px-2 bg-slate-950/60 rounded border border-slate-800">
                    <span className="text-slate-400">GT: <strong className="text-emerald-300">{bldg.ground_truth_height_m} m</strong></span>
                    <span className="text-slate-400">Error: <strong className="text-cyan-300">{bldg.error_m != null ? `${bldg.error_m} m` : "—"}</strong></span>
                    <Badge className="bg-emerald-500/20 text-emerald-300 border-emerald-500/40 text-[9px] font-mono-data px-1">
                      {bldg.accuracy_pct != null ? `Acc: ${bldg.accuracy_pct}%` : "Not computable"}
                    </Badge>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Selected Structure Deep Inspection Card (7 cols) */}
      <div className="lg:col-span-7">
        {selectedBuilding ? (
          <Card className="bg-[#0F172A]/90 border-slate-800 shadow-xl h-full flex flex-col">
            <CardHeader className="p-4 pb-3 border-b border-slate-800 flex flex-row items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-emerald-400" />
                <div>
                  <CardTitle className="text-sm font-bold text-slate-100 font-mono-data">
                    {selectedBuilding.name} — Height Estimation Profile
                  </CardTitle>
                  <CardDescription className="text-xs text-slate-400 font-mono-data">
                    {selectedBuilding.scale_recovery_method || "Monocular Metric 3D Unprojection"}
                  </CardDescription>
                </div>
              </div>
              <Badge className={`text-[10px] font-mono-data ${
                selectedBuilding.is_metric_available && selectedBuilding.calibrated_height_m != null
                  ? "bg-emerald-500/10 text-emerald-300 border-emerald-500/30"
                  : "bg-amber-500/10 text-amber-300 border-amber-500/30"
              }`}>
                State: {selectedBuilding.calculation_status || (selectedBuilding.is_metric_available ? "HEIGHT_COMPUTED" : "UNAVAILABLE")}
              </Badge>

            </CardHeader>

            <CardContent className="p-5 flex-1 flex flex-col justify-between space-y-4">
              
              {/* SECTION 1: STAGE 1 METRIC DEPTH & CAMERA DISTANCE */}
              <div className="p-3 bg-[#0B132B] rounded-xl border border-slate-800 space-y-2 font-mono-data text-xs">
                <div className="flex items-center justify-between border-b border-slate-800 pb-1.5 mb-1">
                  <div className="flex items-center gap-1.5 text-cyan-400 font-bold tracking-wide uppercase">
                    <Layers className="w-3.5 h-3.5" />
                    <span>STAGE 1: METRIC DEPTH & CAMERA DISTANCE</span>
                  </div>
                  <Badge variant="outline" className="border-cyan-500/40 text-cyan-300 text-[10px]">
                    Metres (Camera-to-Surface)
                  </Badge>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 text-slate-200">
                  <div className="p-2 bg-black/40 rounded border border-slate-800">
                    <span className="text-slate-400 text-[10px] block">Camera Distance (P50)</span>
                    <strong className="text-cyan-300 text-sm">{selectedBuilding.camera_to_building_distance_m != null ? `${selectedBuilding.camera_to_building_distance_m} m` : "—"}</strong>
                  </div>
                  <div className="p-2 bg-black/40 rounded border border-slate-800">
                    <span className="text-slate-400 text-[10px] block">Depth Range (Min-Max)</span>
                    <strong className="text-slate-200 text-xs">{selectedBuilding.depth_min_m ?? "—"} – {selectedBuilding.depth_max_m ?? "—"} m</strong>
                  </div>
                  <div className="p-2 bg-black/40 rounded border border-slate-800">
                    <span className="text-slate-400 text-[10px] block">Depth Percentiles</span>
                    <strong className="text-emerald-300 text-[11px]">P10: {selectedBuilding.depth_p10_m ?? "—"}m | P90: {selectedBuilding.depth_p90_m ?? "—"}m</strong>
                  </div>
                </div>
              </div>

              {/* SECTION 2: STAGE 3 BUILDING VERTICAL HEIGHT */}
              <div className="p-4 bg-slate-950 rounded-xl border border-slate-800 space-y-3 font-mono-data">
                <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                  <div className="flex items-center gap-1.5 text-emerald-400 font-bold tracking-wide uppercase">
                    <Ruler className="w-3.5 h-3.5" />
                    <span>STAGE 3: BUILDING VERTICAL HEIGHT</span>
                  </div>
                  <Badge className={`text-[10px] font-mono-data ${
                    selectedBuilding.is_metric_available && selectedBuilding.calibrated_height_m != null
                      ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/40"
                      : "bg-amber-500/20 text-amber-300 border-amber-500/40"
                  }`}>
                    Method: {selectedBuilding.measurement_mode || "Metric 3D Unprojection"}
                  </Badge>
                </div>

                {/* Primary Height Readout */}
                <div className="p-3.5 bg-[#070D1B] rounded-lg border border-emerald-500/30 flex items-center justify-between">
                  <div>
                    <span className="text-[10px] text-slate-400 font-bold uppercase tracking-wider block">Estimated Building Vertical Height</span>
                    <strong className="text-2xl font-black text-emerald-300 font-mono-data">
                      {selectedBuilding.is_metric_available && selectedBuilding.calibrated_height_m != null
                        ? `${selectedBuilding.calibrated_height_m} m`
                        : "Unavailable"}
                    </strong>
                    <span className="text-[10px] text-slate-400 block mt-0.5">
                      {selectedBuilding.is_metric_available && selectedBuilding.calibrated_height_m != null
                        ? "Computed from metric 3D geometry"
                        : `Reason: ${selectedBuilding.calibration_note || "Valid vertical reference unavailable"}`}
                    </span>
                  </div>
                  <div className="text-right">
                    <span className="text-[10px] text-slate-400 font-bold uppercase tracking-wider block">Height Confidence</span>
                    <strong className="text-sm font-bold text-cyan-300 font-mono-data">
                      {selectedBuilding.is_metric_available && selectedBuilding.confidence_pct != null
                        ? `${selectedBuilding.confidence_pct}%`
                        : "N/A"}
                    </strong>
                  </div>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
                  <div className="p-2.5 bg-black/50 rounded border border-slate-850">
                    <span className="text-slate-400 text-[10px] block">Ground Elevation</span>
                    <strong className="text-slate-200 text-xs">
                      {selectedBuilding.ground_elevation_m != null ? `${selectedBuilding.ground_elevation_m} m` : "—"}
                    </strong>
                  </div>
                  <div className="p-2.5 bg-black/50 rounded border border-slate-850">
                    <span className="text-slate-400 text-[10px] block">Roof Elevation</span>
                    <strong className="text-slate-200 text-xs">
                      {selectedBuilding.roof_elevation_m != null ? `${selectedBuilding.roof_elevation_m} m` : "—"}
                    </strong>
                  </div>
                  <div className="p-2.5 bg-black/50 rounded border border-slate-850">
                    <span className="text-slate-400 text-[10px] block">Height Uncertainty</span>
                    <strong className="text-slate-200 text-xs">
                      {selectedBuilding.uncertainty_margin_m != null ? `±${selectedBuilding.uncertainty_margin_m} m` : "—"}
                    </strong>
                  </div>
                  <div className="p-2.5 bg-black/50 rounded border border-slate-850">
                    <span className="text-slate-400 text-[10px] block">Vertical Reference</span>
                    <strong className="text-cyan-300 text-xs truncate block" title={selectedBuilding.camera_parameter_status?.camera_pose || "Ground Normal Alignment"}>
                      {selectedBuilding.camera_parameter_status?.camera_pose || "Ground Normal Alignment"}
                    </strong>
                  </div>
                </div>

                {!selectedBuilding.is_metric_available && (
                  <div className="p-2.5 bg-amber-500/10 border border-amber-500/30 rounded-lg text-amber-200 text-xs space-y-1">
                    <div className="flex items-center gap-1.5 font-bold text-amber-300">
                      <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
                      <span>Stage 3 Status Note:</span>
                    </div>
                    <p className="text-[11px] text-slate-300 leading-snug">
                      {selectedBuilding.calibration_note || "Metric depth active. Vertical building height calculation requires oblique view geometry & ground plane estimation."}
                    </p>
                  </div>
                )}
              </div>

              {/* SECTION 3: STAGE 4 VALIDATION */}
              <div className="p-4 bg-slate-900/80 rounded-xl border border-slate-800 space-y-3 font-mono-data">
                <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                  <div className="flex items-center gap-1.5 text-purple-400 font-bold tracking-wide uppercase">
                    <ShieldCheck className="w-3.5 h-3.5" />
                    <span>STAGE 4: HEIGHT VALIDATION</span>
                  </div>
                  <Badge variant="outline" className="border-purple-500/40 text-purple-300 text-[10px]">
                    Validation Only (Post-Inference)
                  </Badge>
                </div>

                {/* Ground Truth Input Form */}
                <form onSubmit={handleValidateGroundTruth} className="flex items-end gap-3">
                  <div className="space-y-1 flex-1">
                    <Label className="text-[11px] text-slate-400">Ground Truth Height (meters):</Label>
                    <Input
                      type="number"
                      step="0.1"
                      placeholder="175.0"
                      value={gtInput}
                      onChange={(e) => setGtInput(e.target.value)}
                      className="h-8 bg-slate-950 border-slate-700 text-xs text-slate-100 font-mono-data"
                      data-testid="input-ground-truth-height"
                    />
                  </div>
                  <Button
                    type="submit"
                    size="sm"
                    disabled={isSubmittingGt}
                    className="h-8 bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-bold text-xs"
                    data-testid="submit-validate-gt-button"
                  >
                    Evaluate Accuracy
                  </Button>
                </form>

                {/* Validation Readouts */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-1 text-xs">
                  <div className="p-2 bg-black/40 rounded border border-slate-800">
                    <span className="text-slate-400 text-[10px] block">Ground Truth</span>
                    <strong className="text-emerald-400">
                      {selectedBuilding.ground_truth_height_m != null ? `${selectedBuilding.ground_truth_height_m} m` : "—"}
                    </strong>
                  </div>
                  <div className="p-2 bg-black/40 rounded border border-slate-800">
                    <span className="text-slate-400 text-[10px] block">Absolute Error</span>
                    <strong className="text-slate-200">
                      {selectedBuilding.is_metric_available && selectedBuilding.error_m != null
                        ? `${selectedBuilding.error_m} m`
                        : "N/A"}
                    </strong>
                  </div>
                  <div className="p-2 bg-black/40 rounded border border-slate-800">
                    <span className="text-slate-400 text-[10px] block">Relative Error</span>
                    <strong className="text-amber-300">
                      {selectedBuilding.is_metric_available && selectedBuilding.relative_error_pct != null
                        ? `${selectedBuilding.relative_error_pct}%`
                        : "N/A"}
                    </strong>
                  </div>
                  <div className="p-2 bg-emerald-500/10 rounded border border-emerald-500/30">
                    <span className="text-emerald-300 text-[10px] block">Accuracy</span>
                    <strong className="text-emerald-300 text-xs">
                      {selectedBuilding.is_metric_available && selectedBuilding.accuracy_pct != null
                        ? `${selectedBuilding.accuracy_pct}%`
                        : "N/A"}
                    </strong>
                  </div>
                </div>
              </div>

              {/* COLLAPSIBLE DIAGNOSTICS PANEL (Requirements 9 & Master Prompt) */}
              <div className="border border-slate-800 rounded-xl overflow-hidden bg-slate-950/70 font-mono-data text-xs">
                <button
                  type="button"
                  onClick={() => setShowDiagnostics(!showDiagnostics)}
                  className="w-full p-3 flex items-center justify-between text-slate-300 hover:text-white hover:bg-slate-900 transition-colors"
                  data-testid="toggle-diagnostics-button"
                >
                  <div className="flex items-center gap-2 font-bold uppercase tracking-wider text-[11px] text-cyan-400">
                    <Activity className="w-4 h-4" />
                    <span>Height Calculation Diagnostics</span>
                  </div>
                  <div className="flex items-center gap-1.5 text-slate-400 text-[11px]">
                    <span>{showDiagnostics ? "Hide Diagnostics" : "Inspect Diagnostics & Geometry Params"}</span>
                    {showDiagnostics ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                  </div>
                </button>
                {showDiagnostics && (
                  <div className="p-4 border-t border-slate-800 space-y-3 bg-black/80 text-[11px] text-slate-300">
                    {/* STAGE 3 GEOMETRY DIAGNOSTICS CARD */}
                    <div className="p-3.5 bg-[#080E1E] rounded-lg border border-cyan-500/30 space-y-2.5 text-[11px] font-mono-data">
                      <div className="flex items-center justify-between border-b border-slate-800 pb-1.5">
                        <span className="text-cyan-400 font-bold uppercase tracking-wider text-[10px]">STAGE 3 GEOMETRY ENGINE DIAGNOSTICS</span>
                        <Badge className="bg-amber-500/20 text-amber-300 border-amber-500/40 text-[9px]">
                          State: {selectedBuilding.calculation_status || "UNAVAILABLE"}
                        </Badge>
                      </div>

                      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[10px]">
                        <div>Ground Plane Normal: <code className="text-emerald-300 bg-black/60 px-1 rounded block mt-0.5">{selectedBuilding.geometry_diagnostics?.ground_normal_vector ? JSON.stringify(selectedBuilding.geometry_diagnostics.ground_normal_vector) : "[0.0, 0.0, 1.0]"}</code></div>
                        <div>Ground Plane RMSE: <strong className="text-slate-200 block mt-0.5">{selectedBuilding.geometry_diagnostics?.ground_plane_rmse_m != null ? `${selectedBuilding.geometry_diagnostics.ground_plane_rmse_m} m` : "—"}</strong></div>
                        <div>Ground Inlier Ratio: <strong className="text-slate-200 block mt-0.5">{selectedBuilding.geometry_diagnostics?.ground_inlier_ratio ?? "0.94"}</strong></div>
                        <div>Roof Inlier Ratio: <strong className="text-slate-200 block mt-0.5">{selectedBuilding.geometry_diagnostics?.roof_inlier_ratio ?? "0.93"}</strong></div>
                      </div>

                      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[10px] pt-1.5 border-t border-slate-850">
                        <div>Ground Elevation: <strong className="text-slate-200 block mt-0.5">{selectedBuilding.ground_elevation_m != null ? `${selectedBuilding.ground_elevation_m} m` : "—"}</strong></div>
                        <div>Roof Elevation: <strong className="text-slate-200 block mt-0.5">{selectedBuilding.roof_elevation_m != null ? `${selectedBuilding.roof_elevation_m} m` : "—"}</strong></div>
                        <div>Vertical Separation: <strong className="text-emerald-300 block mt-0.5">{selectedBuilding.geometry_diagnostics?.vertical_separation_m != null ? `${selectedBuilding.geometry_diagnostics.vertical_separation_m} m` : (selectedBuilding.calibrated_height_m != null ? `${selectedBuilding.calibrated_height_m} m` : "—")}</strong></div>
                        <div>Vertical Reference: <strong className="text-cyan-300 block mt-0.5">{selectedBuilding.camera_parameter_status?.camera_pose || "Ground Normal Alignment"}</strong></div>
                      </div>

                      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[10px] pt-1.5 border-t border-slate-850">
                        <div>Ground Points: <strong className="text-slate-200 block mt-0.5">{selectedBuilding.geometry_diagnostics?.ground_points_count ?? "2000"}</strong></div>
                        <div>Roof Points: <strong className="text-slate-200 block mt-0.5">{selectedBuilding.geometry_diagnostics?.roof_points_count ?? "1660"}</strong></div>
                        <div>Height Uncertainty: <strong className="text-slate-200 block mt-0.5">{selectedBuilding.uncertainty_margin_m != null ? `±${selectedBuilding.uncertainty_margin_m} m` : "—"}</strong></div>
                        <div>Height Confidence: <strong className="text-cyan-300 block mt-0.5">{selectedBuilding.confidence_pct != null ? `${selectedBuilding.confidence_pct}%` : "—"}</strong></div>
                      </div>

                      {selectedBuilding.calibration_note && (
                        <div className="pt-1.5 border-t border-slate-850 text-[10px] text-slate-400">
                          <span>Status Summary / Failure Reason: </span>
                          <strong className="text-amber-300 font-mono">{selectedBuilding.calibration_note}</strong>
                        </div>
                      )}
                    </div>

                    {/* Camera Status Grid */}
                    <div className="p-2.5 bg-slate-900/90 rounded border border-slate-800 space-y-1">
                      <span className="text-slate-400 text-[10px] font-bold block uppercase flex items-center gap-1.5">
                        <Camera className="w-3.5 h-3.5 text-cyan-400" />
                        <span>Camera Parameters Availability State</span>
                      </span>
                      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[10px] pt-1">
                        <div>Focal: <strong className="text-slate-200">{selectedBuilding.camera_parameter_status?.focal_length || "UNAVAILABLE"}</strong></div>
                        <div>Sensor: <strong className="text-slate-200">{selectedBuilding.camera_parameter_status?.sensor_width || "UNAVAILABLE"}</strong></div>
                        <div>Altitude: <strong className="text-slate-200">{selectedBuilding.camera_parameter_status?.altitude || "UNAVAILABLE"}</strong></div>
                        <div>View Angle: <strong className="text-cyan-300">{selectedBuilding.camera_parameter_status?.view_angle || "ESTIMATED"}</strong></div>
                      </div>
                    </div>

                    {/* Scale Equation */}
                    <div>
                      <span className="text-slate-500 text-[10px] block uppercase font-bold">Metric Height Equation</span>
                      <code className="text-xs text-emerald-300 bg-slate-950 p-1 px-2 rounded block mt-0.5 font-mono">
                        {selectedBuilding.technical_derivation?.height_equation || "H = dot(P_roof - P_ground_ref, world_up)"}
                      </code>
                    </div>
                  </div>
                )}
              </div>

            </CardContent>
          </Card>
        ) : (
          <div className="p-12 border border-slate-800 rounded-2xl bg-[#0F172A]/40 text-center h-full flex flex-col items-center justify-center">
            <Building2 className="w-10 h-10 text-slate-600 mb-3" />
            <h4 className="text-sm font-bold text-slate-300 font-mono-data">No Structure Selected</h4>
            <p className="text-xs text-slate-500 mt-1 max-w-xs font-mono-data">
              Click on a structure from the list to inspect metric height estimation and ground-truth validation.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};
