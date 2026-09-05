import React, { useState, useEffect, useRef } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiGet, apiPost } from "@/lib/api";
import type { 
  DepthProcessResponse, 
  CalibrationConfig, 
  BuildingMeasurement, 
  HeroSceneSummary 
} from "@/types/depth";
import { TriPanelInspector } from "@/components/TriPanelInspector";
import { HeightCalibrationPanel } from "@/components/HeightCalibrationPanel";
import { BuildingMeasurementPanel } from "@/components/BuildingMeasurementPanel";
import { ValidationDashboard } from "@/components/ValidationDashboard";
import { PresentationMode } from "@/components/PresentationMode";
import { MethodologyModal } from "@/components/MethodologyModal";
import { AerialBenchmarkPanel } from "@/components/AerialBenchmarkPanel";
import { AerialTrainingPanel } from "@/components/AerialTrainingPanel";
import { GeospatialParticleBackground } from "@/components/GeospatialParticleBackground";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { toast } from "sonner";
import { 
  Scan, 
  Upload, 
  Sparkles, 
  Scale, 
  BarChart2, 
  Layers, 
  BookOpen, 
  Radio, 
  Building2
} from "lucide-react";

export default function Home() {
  const queryClient = useQueryClient();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [activeTab, setActiveTab] = useState<string>("inspector");
  const [selectedHeroId, setSelectedHeroId] = useState<string>("downtown_highrise");
  const [selectedBuilding, setSelectedBuilding] = useState<BuildingMeasurement | null>(null);
  const [isMethodologyOpen, setIsMethodologyOpen] = useState<boolean>(false);
  const [uploadImageType] = useState<"aerial" | "drone" | "satellite">("aerial");

  // Fetch Hero Scenes list
  const { data: heroScenes } = useQuery<HeroSceneSummary[]>({
    queryKey: ["hero-scenes"],
    queryFn: () => apiGet<HeroSceneSummary[]>("/depth/hero-scenes"),
  });

  // Fetch / Process Active Scene
  const { 
    data: sceneData
  } = useQuery<DepthProcessResponse>({
    queryKey: ["depth-scene", selectedHeroId],
    queryFn: () => apiGet<DepthProcessResponse>(`/depth/hero-scene/${selectedHeroId}`),
    staleTime: 1000 * 60 * 10,
  });

  // Set default selected building when scene data arrives
  useEffect(() => {
    if (sceneData && sceneData.sample_buildings && sceneData.sample_buildings.length > 0) {
      setSelectedBuilding(sceneData.sample_buildings[0]);
    }
  }, [sceneData]);

  // Calibration Mutation
  const calibrationMutation = useMutation({
    mutationFn: async (calib: CalibrationConfig) => {
      if (!sceneData) throw new Error("No active scene");
      return await apiPost<DepthProcessResponse>(`/depth/calibrate?scene_id=${sceneData.id}`, calib);
    },
    onSuccess: (updated) => {
      queryClient.setQueryData(["depth-scene", selectedHeroId], updated);
      if (updated.sample_buildings && updated.sample_buildings.length > 0) {
        setSelectedBuilding(updated.sample_buildings[0]);
      }
      toast.success("Calibration applied successfully!");
    },
    onError: (err: any) => {
      toast.error(`Calibration error: ${err.message || "Failed to calibrate"}`);
    }
  });

  // Custom Bounding Box Measurement Mutation
  const measureMutation = useMutation({
    mutationFn: async (bbox: [number, number, number, number]) => {
      if (!sceneData) throw new Error("No active scene");
      return await apiPost<BuildingMeasurement>("/depth/measure-building", {
        scene_id: sceneData.id,
        bbox,
        name: `Custom Structure ${sceneData.sample_buildings.length + 1}`
      });
    },
    onSuccess: (newMeasurement) => {
      if (sceneData) {
        const updated = {
          ...sceneData,
          sample_buildings: [newMeasurement, ...sceneData.sample_buildings]
        };
        queryClient.setQueryData(["depth-scene", selectedHeroId], updated);
        setSelectedBuilding(newMeasurement);
        toast.success(`Measured: ${newMeasurement.calibrated_height_m ? `${newMeasurement.calibrated_height_m}m` : 'Relative elevation extracted'}`);
      }
    },
    onError: (err: any) => {
      toast.error(`Measurement failed: ${err.message}`);
    }
  });

  // Handle Custom File Upload
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);
    formData.append("image_type", uploadImageType);
    formData.append("calibration_mode", "relative");

    toast.info("Uploading and running Depth Anything V2 inference...");
    try {
      const response = await fetch("/api/depth/upload", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Upload failed with status ${response.status}`);
      }

      const resData = (await response.json()) as DepthProcessResponse;
      queryClient.setQueryData(["depth-scene", "custom_upload"], resData);
      setSelectedHeroId("custom_upload");
      if (resData.sample_buildings && resData.sample_buildings.length > 0) {
        setSelectedBuilding(resData.sample_buildings[0]);
      }
      setActiveTab("inspector");
      toast.success("Image processed and 3D reconstructed!");
    } catch (err: any) {
      toast.error(err.message || "Failed to process uploaded image");
    }
  };

  const currentCalibration = sceneData?.calibration || {
    mode: "relative",
    is_calibrated: false,
    calibration_description: "Relative depth — metric scale unavailable.",
  };

  // Resolve the human-readable label for the active scene.
  const activeScene = (heroScenes ?? []).find((h) => h.id === selectedHeroId);
  const activeSceneLabel = activeScene
    ? `${activeScene.title} — ${activeScene.view_angle}`
    : sceneData?.scene_name ?? "Loading scene…";

  return (
    <div className="min-h-screen bg-[#070A0F] text-slate-100 flex flex-col font-sans relative">
      {/* Live Animated Particle & Tech-Geometry Background */}
      <GeospatialParticleBackground />

      {/* Top Sticky Header */}
      <header className="sticky top-0 z-50 bg-[#0F172A]/85 backdrop-blur-md border-b border-slate-800/80 px-4 lg:px-6 py-2.5">
        <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-3">
          {/* Logo Branding */}
          <div className="flex items-center gap-3">
            <div className="relative flex items-center justify-center w-8 h-8 rounded-lg bg-cyan-500/10 border border-cyan-500/40 text-cyan-400">
              <Scan className="w-4 h-4" />
              <span className="absolute -top-0.5 -right-0.5 w-2 h-2 rounded-full bg-cyan-400 animate-ping" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-base font-extrabold tracking-tight text-white font-mono-data">
                  DEPTH<span className="text-cyan-400">WIZARD</span>
                </h1>
                <Badge className="bg-cyan-500/10 text-cyan-300 border-cyan-500/30 text-[9px] font-mono-data py-0 px-1.5 h-4">
                  SIH26175
                </Badge>
              </div>
              <p className="text-[10px] text-slate-400 font-mono-data">
                Single-Image 3D Reconstruction & Metric Height Estimation
              </p>
            </div>
          </div>

          {/* Center: Hero Scene Dropdown Selector */}
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono-data text-slate-400 hidden sm:inline">Hero Dataset:</span>
            <Select
              value={selectedHeroId}
              onValueChange={(v) => {
                setSelectedHeroId(v);
                setSelectedBuilding(null);
              }}
            >
              <SelectTrigger
                data-testid="hero-scene-select"
                className="h-9 min-w-[300px] bg-slate-900 border-slate-700 text-xs text-cyan-300 font-mono-data focus:border-cyan-400"
              >
                {/* Explicit label so the trigger never falls back to the raw scene id
                    while the scene list is still loading. */}
                <SelectValue placeholder="Select a benchmark scene">
                  {activeSceneLabel}
                </SelectValue>
              </SelectTrigger>
              <SelectContent className="bg-slate-900 border-slate-700 text-slate-200">
                {(heroScenes ?? []).map((hs) => (
                  <SelectItem
                    key={hs.id}
                    value={hs.id}
                    data-testid={`hero-scene-option-${hs.id}`}
                    className="text-xs font-mono-data focus:bg-cyan-500/15 focus:text-cyan-200"
                  >
                    {hs.title} — {hs.view_angle}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Right Actions: Upload, Presentation Mode, Methodology */}
          <div className="flex items-center gap-2">
            {/* Upload Button */}
            <input
              ref={fileInputRef}
              type="file"
              accept="image/jpeg,image/png,image/webp,image/tiff"
              onChange={handleFileUpload}
              className="hidden"
            />
            <Button
              size="sm"
              variant="outline"
              onClick={() => fileInputRef.current?.click()}
              data-testid="upload-image-button"
              className="h-8 text-xs font-mono-data border-slate-700 text-slate-300 hover:text-white hover:border-cyan-400"
            >
              <Upload className="w-3.5 h-3.5 mr-1 text-cyan-400" />
              Upload Image
            </Button>

            {/* 3-Min SIH Presentation Launcher */}
            <Button
              size="sm"
              onClick={() => setActiveTab("presentation")}
              data-testid="launch-presentation-mode-button"
              className="h-8 text-xs font-mono-data bg-cyan-500 hover:bg-cyan-400 text-black font-bold shadow-md shadow-cyan-500/20"
            >
              <Sparkles className="w-3.5 h-3.5 mr-1" />
              3-Min Demo
            </Button>

            {/* Methodology Modal Trigger */}
            <Button
              size="icon-xs"
              variant="ghost"
              onClick={() => setIsMethodologyOpen(true)}
              title="Scientific Methodology & Limitations"
              data-testid="open-methodology-button"
              className="h-8 w-8 text-slate-400 hover:text-cyan-400"
            >
              <BookOpen className="w-4 h-4" />
            </Button>
          </div>
        </div>
      </header>

      {/* Main App Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 lg:p-6 space-y-6 relative z-10">
        {/* Navigation Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full space-y-6">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-3">
            <TabsList className="bg-[#0F172A] border border-slate-800 p-1 rounded-xl">
              <TabsTrigger
                value="inspector"
                data-testid="tab-tri-panel-inspector"
                className="text-xs font-mono-data data-[state=active]:bg-cyan-500 data-[state=active]:text-black data-[state=active]:font-bold"
              >
                <Layers className="w-3.5 h-3.5 mr-1.5" />
                Tri-Panel 3D Studio
              </TabsTrigger>
              <TabsTrigger
                value="calibration"
                data-testid="tab-height-calibration"
                className="text-xs font-mono-data data-[state=active]:bg-cyan-500 data-[state=active]:text-black data-[state=active]:font-bold"
              >
                <Scale className="w-3.5 h-3.5 mr-1.5" />
                Height Calibration & Structures
              </TabsTrigger>
              <TabsTrigger
                value="validation"
                data-testid="tab-validation-dashboard"
                className="text-xs font-mono-data data-[state=active]:bg-cyan-500 data-[state=active]:text-black data-[state=active]:font-bold"
              >
                <BarChart2 className="w-3.5 h-3.5 mr-1.5" />
                Validation Framework
              </TabsTrigger>
              <TabsTrigger
                value="benchmark"
                data-testid="tab-aerial-benchmark"
                className="text-xs font-mono-data data-[state=active]:bg-indigo-600 data-[state=active]:text-white data-[state=active]:font-bold"
              >
                <Layers className="w-3.5 h-3.5 mr-1.5" />
                Aerial Depth Benchmark
              </TabsTrigger>
              <TabsTrigger
                value="training"
                data-testid="tab-aerial-training"
                className="text-xs font-mono-data data-[state=active]:bg-cyan-600 data-[state=active]:text-white data-[state=active]:font-bold"
              >
                <Radio className="w-3.5 h-3.5 mr-1.5" />
                Aerial Model Training
              </TabsTrigger>
              <TabsTrigger
                value="presentation"
                data-testid="tab-presentation-walkthrough"
                className="text-xs font-mono-data data-[state=active]:bg-cyan-500 data-[state=active]:text-black data-[state=active]:font-bold"
              >
                <Sparkles className="w-3.5 h-3.5 mr-1.5" />
                SIH 3-Min Walkthrough
              </TabsTrigger>
            </TabsList>

            {/* Calibration Status pill */}
            <div className="flex items-center gap-2">
              <Badge
                className={`text-[10px] font-mono-data ${
                  currentCalibration.is_calibrated
                    ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/40"
                    : "bg-amber-500/20 text-amber-300 border-amber-500/40"
                }`}
              >
                {currentCalibration.is_calibrated ? "● Calibrated Scale" : "○ Relative Depth Only"}
              </Badge>
              {sceneData?.quality_report && (
                <Badge className="bg-slate-800 text-slate-300 border-slate-700 text-[10px] font-mono-data">
                  {sceneData.quality_report.view_geometry}
                </Badge>
              )}
              {sceneData?.depth_backend && (
                <Badge
                  data-testid="depth-backend-badge"
                  title={sceneData.depth_backend}
                  className={`text-[10px] font-mono-data ${
                    sceneData.depth_backend_is_neural
                      ? "bg-cyan-500/15 text-cyan-300 border-cyan-500/40"
                      : "bg-amber-500/15 text-amber-300 border-amber-500/40"
                  }`}
                >
                  {sceneData.depth_backend_is_neural
                    ? "Neural: Depth Anything V2"
                    : "Heuristic baseline (non-neural)"}
                </Badge>
              )}
            </div>
          </div>

          {/* TAB 1: Tri-Panel 3D Inspector */}
          <TabsContent value="inspector" className="m-0 space-y-4 outline-none">
            <TriPanelInspector
              data={sceneData || null}
              selectedBuilding={selectedBuilding}
              onSelectBuilding={setSelectedBuilding}
              onCustomMeasure={(bbox) => measureMutation.mutate(bbox)}
            />

            {/* Quick Structure Elevation Bar */}
            {sceneData && sceneData.sample_buildings.length > 0 && (
              <div className="p-3 bg-[#0F172A]/80 border border-slate-800 rounded-xl flex flex-wrap items-center justify-between gap-3 text-xs font-mono-data">
                <div className="flex items-center gap-2">
                  <Building2 className="w-4 h-4 text-cyan-400" />
                  <span className="text-slate-300 font-bold">Scene Structures ({sceneData.sample_buildings.length}):</span>
                </div>
                <div className="flex flex-wrap items-center gap-2">
                  {sceneData.sample_buildings.map((b) => (
                    <button
                      key={b.id}
                      onClick={() => setSelectedBuilding(b)}
                      data-testid={`quick-bldg-btn-${b.id}`}
                      className={`px-2.5 py-1 rounded-lg border text-[11px] transition-all ${
                        selectedBuilding?.id === b.id
                          ? "bg-emerald-500/20 border-emerald-400 text-emerald-300 font-bold shadow"
                          : "bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200"
                      }`}
                    >
                      {b.name}: {b.calibrated_height_m ? `${b.calibrated_height_m}m` : `Relative Depth: ${b.rooftop_peak_z_rel}`}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </TabsContent>

          {/* TAB 2: Height Calibration & Building Measurements */}
          <TabsContent value="calibration" className="m-0 space-y-6 outline-none">
            <HeightCalibrationPanel
              currentCalibration={currentCalibration}
              data={sceneData || null}
              onApplyCalibration={(calib) => calibrationMutation.mutate(calib)}
              isLoading={calibrationMutation.isPending}
            />

            {sceneData && (
              <BuildingMeasurementPanel
                buildings={sceneData.sample_buildings}
                selectedBuilding={selectedBuilding}
                calibration={currentCalibration}
                sceneId={sceneData.id}
                onSelectBuilding={setSelectedBuilding}
                onUpdateBuilding={(updatedBldg) => {
                  setSelectedBuilding(updatedBldg);
                  if (sceneData) {
                    const newBldgs = sceneData.sample_buildings.map((b) =>
                      b.id === updatedBldg.id ? updatedBldg : b
                    );
                    queryClient.setQueryData(["depth-scene", selectedHeroId], {
                      ...sceneData,
                      sample_buildings: newBldgs,
                    });
                  }
                }}
              />
            )}
          </TabsContent>

          {/* TAB 3: Scientific Validation Benchmark */}
          <TabsContent value="validation" className="m-0 space-y-6 outline-none">
            <ValidationDashboard />
          </TabsContent>

          {/* TAB 4: Zero-Shot Aerial Metric Depth Benchmark */}
          <TabsContent value="benchmark" className="m-0 space-y-6 outline-none">
            <AerialBenchmarkPanel
              sceneId={sceneData?.id}
              imageUrl={sceneData?.image_url}
            />
          </TabsContent>

          {/* TAB 5: Phase 2 Aerial Model Training */}
          <TabsContent value="training" className="m-0 space-y-6 outline-none">
            <AerialTrainingPanel />
          </TabsContent>

          {/* TAB 4: SIH 3-Minute Walkthrough Presentation */}
          <TabsContent value="presentation" className="m-0 space-y-6 outline-none">
            <PresentationMode
              data={sceneData || null}
              onSelectHeroScene={(id) => {
                setSelectedHeroId(id);
                setSelectedBuilding(null);
              }}
              onNavigateTab={setActiveTab}
            />

            {/* Also show Tri-Panel below presentation for interactive judge inspection */}
            <div className="pt-2">
              <TriPanelInspector
                data={sceneData || null}
                selectedBuilding={selectedBuilding}
                onSelectBuilding={setSelectedBuilding}
                onCustomMeasure={(bbox) => measureMutation.mutate(bbox)}
              />
            </div>
          </TabsContent>
        </Tabs>
      </main>

      {/* Process Timing Breakdown Footer */}
      <footer className="mt-auto border-t border-slate-800 bg-[#070A0F]/80 backdrop-blur-sm py-3 px-4 lg:px-6 font-mono-data text-[11px] text-slate-400 relative z-10">
        <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-4">
            <span className="flex items-center gap-1.5 text-cyan-400 font-semibold">
              <Radio className="w-3.5 h-3.5 animate-pulse" />
              Pipeline Telemetry
            </span>
            {sceneData?.timing_ms && (
              <>
                <span>Preprocess: <strong className="text-slate-200">{sceneData.timing_ms.preprocessing_ms} ms</strong></span>
                <span>Depth Inference: <strong className="text-slate-200">{sceneData.timing_ms.depth_inference_ms} ms</strong></span>
                <span>3D Reconstruction: <strong className="text-slate-200">{sceneData.timing_ms.geometry_reconstruction_ms} ms</strong></span>
                <span>Total Latency: <strong className="text-emerald-400">{sceneData.timing_ms.total_ms} ms</strong></span>
              </>
            )}
          </div>

          <div className="text-slate-500 text-[10px] max-w-md text-right">
            SIH26175 DepthWizard | Rigorous Elevation & Metric Calibration Engine
          </div>
        </div>
      </footer>

      {/* Methodology & Limitations Modal */}
      <MethodologyModal
        open={isMethodologyOpen}
        onOpenChange={setIsMethodologyOpen}
      />
    </div>
  );
}
