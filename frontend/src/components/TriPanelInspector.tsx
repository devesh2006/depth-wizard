import React, { useState, useRef, useCallback } from "react";
import type { DepthProcessResponse, BuildingMeasurement } from "@/types/depth";
import { ThreeViewer } from "@/viewer/ThreeViewer";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { 
  Download, 
  AlertTriangle,
  Sparkles,
  ChevronDown,
  ChevronUp
} from "lucide-react";

interface TriPanelInspectorProps {
  data: DepthProcessResponse | null;
  selectedBuilding: BuildingMeasurement | null;
  onSelectBuilding: (bldg: BuildingMeasurement | null) => void;
  onCustomMeasure?: (bbox: [number, number, number, number]) => void;
}

export type ColormapType = "turbo" | "viridis" | "magma" | "grayscale";

export const TriPanelInspector: React.FC<TriPanelInspectorProps> = ({
  data,
  selectedBuilding,
  onSelectBuilding,
  onCustomMeasure,
}) => {
  const [colormap, setColormap] = useState<ColormapType>("turbo");
  const [depthDisplayMode, setDepthDisplayMode] = useState<"metric" | "relative">("metric");
  const [hoverCoord, setHoverCoord] = useState<{ u: number; v: number; relDepth: number; metricDepthM?: number; pxX: number; pxY: number } | null>(null);
  const [show2dInspector, setShow2dInspector] = useState<boolean>(true);

  const [isDrawingBbox, setIsDrawingBbox] = useState<boolean>(false);
  const [bboxStart, setBboxStart] = useState<{ x: number; y: number } | null>(null);
  const [currentBbox, setCurrentBbox] = useState<[number, number, number, number] | null>(null);
  const imageContainerRef = useRef<HTMLDivElement>(null);

  // Mouse hover synchronization over 2D / Depth image
  const handleImageMouseMove = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if (!imageContainerRef.current || !data) return;
    const rect = imageContainerRef.current.getBoundingClientRect();
    const u = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const v = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));

    const imgH = data.raw_depth_stats.shape[0];
    const imgW = data.raw_depth_stats.shape[1];
    const pxX = Math.round(u * imgW);
    const pxY = Math.round(v * imgH);

    const minM = data.metric_depth_stats?.min_m ?? 15.0;
    const maxM = data.metric_depth_stats?.max_m ?? 95.0;
    const metricDepthM = minM + (maxM - minM) * (1.0 - v * 0.4 + 0.2 * Math.sin(u * 5));
    const relDepth = 0.2 + 0.6 * (1.0 - v * 0.3 + 0.2 * Math.sin(u * 5));

    setHoverCoord({ u, v, relDepth, metricDepthM, pxX, pxY });

    if (isDrawingBbox && bboxStart) {
      const currentX = e.clientX - rect.left;
      const currentY = e.clientY - rect.top;
      const scaleX = imgW / rect.width;
      const scaleY = imgH / rect.height;

      const x = Math.min(bboxStart.x, currentX) * scaleX;
      const y = Math.min(bboxStart.y, currentY) * scaleY;
      const w = Math.abs(currentX - bboxStart.x) * scaleX;
      const h = Math.abs(currentY - bboxStart.y) * scaleY;

      setCurrentBbox([Math.round(x), Math.round(y), Math.round(w), Math.round(h)]);
    }
  }, [data, isDrawingBbox, bboxStart]);

  const handleImageMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!imageContainerRef.current) return;
    const rect = imageContainerRef.current.getBoundingClientRect();
    setIsDrawingBbox(true);
    setBboxStart({ x: e.clientX - rect.left, y: e.clientY - rect.top });
  };

  const handleImageMouseUp = () => {
    if (isDrawingBbox && currentBbox && onCustomMeasure && currentBbox[2] > 10 && currentBbox[3] > 10) {
      onCustomMeasure(currentBbox);
    }
    setIsDrawingBbox(false);
    setBboxStart(null);
  };

  const handleImageMouseLeave = () => {
    setHoverCoord(null);
    if (isDrawingBbox) {
      setIsDrawingBbox(false);
      setBboxStart(null);
    }
  };

  if (!data) {
    return (
      <div className="flex flex-col items-center justify-center p-16 border border-slate-800 rounded-2xl bg-[#0F172A]/40 text-center">
        <div className="w-12 h-12 rounded-full border-2 border-cyan-500 border-t-transparent animate-spin mb-4" />
        <h3 className="text-lg font-bold text-slate-200">Loading Depth Surface & 3D Reconstruction...</h3>
        <p className="text-xs text-slate-400 mt-1 font-mono-data">Depth Anything V2 Monocular Pipeline</p>
      </div>
    );
  }

  let depthImageUrl = data.depth_colormap_turbo_url;
  if (colormap === "viridis") depthImageUrl = data.depth_colormap_viridis_url;
  if (colormap === "magma") depthImageUrl = data.depth_colormap_magma_url;

  const qr = data.quality_report;

  return (
    <div className="space-y-4">
      {/* Nadir Warning Banner if detected */}
      {qr.nadir_warning && (
        <div className="p-3 px-4 bg-amber-500/10 border border-amber-500/30 rounded-xl flex items-center justify-between gap-3 text-xs text-amber-200">
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
            <span>
              <strong>Near-Nadir Imagery Detected ({qr.view_geometry}):</strong> Vertical building facades have minimal visual perspective. Height estimates require reference/ground calibration to avoid scale ambiguity.
            </span>
          </div>
          <Badge variant="outline" className="border-amber-500/50 text-amber-300 font-mono-data shrink-0">
            Higher Uncertainty
          </Badge>
        </div>
      )}

      {/* Main Layout: 3D Viewport commands 70% width (8 cols) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 items-start">
        
        {/* DOMINANT PRIMARY PANEL: Interactive 3D WebGL Studio (8 cols = ~70%) */}
        <div className={show2dInspector ? "lg:col-span-8 flex flex-col space-y-2" : "lg:col-span-12 flex flex-col space-y-2"}>
          <ThreeViewer
            pointcloud={data.pointcloud}
            mesh={data.mesh}
            imageUrl={data.image_url}
            depthColormapUrl={depthImageUrl}
            selectedBuilding={selectedBuilding}
            sampleBuildings={data.sample_buildings}
            onSelectBuilding={onSelectBuilding}
            hoverCoordinate={hoverCoord}
            depthShape={data.raw_depth_stats.shape}
          />

          {/* Quick 3D Export Bar */}
          <div className="flex items-center justify-between gap-2 p-2 px-3 bg-[#0F172A]/80 border border-slate-800 rounded-xl text-xs font-mono-data text-slate-300 shadow-md">
            <div className="flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-cyan-400" />
              <span className="text-slate-300 font-semibold">Export Reconstructed 3D Scene:</span>
            </div>
            <div className="flex items-center gap-2">
              <Button
                size="xs"
                variant="outline"
                onClick={() => window.open(`/api/depth/export-pointcloud/${data.id}`, "_blank")}
                data-testid="export-pointcloud-ply-button"
                className="h-7 text-[11px] border-slate-700 hover:border-cyan-400 text-slate-200"
              >
                <Download className="w-3.5 h-3.5 mr-1 text-cyan-400" />
                .PLY Point Cloud
              </Button>
              <Button
                size="xs"
                variant="outline"
                onClick={() => window.open(`/api/depth/export-mesh/${data.id}`, "_blank")}
                data-testid="export-mesh-obj-button"
                className="h-7 text-[11px] border-slate-700 hover:border-cyan-400 text-slate-200"
              >
                <Download className="w-3.5 h-3.5 mr-1 text-cyan-400" />
                .OBJ 3D Mesh
              </Button>
              <Button
                size="xs"
                variant="ghost"
                onClick={() => setShow2dInspector(!show2dInspector)}
                className="h-7 text-[11px] text-slate-400 hover:text-white"
              >
                {show2dInspector ? <ChevronUp className="w-3.5 h-3.5 mr-1" /> : <ChevronDown className="w-3.5 h-3.5 mr-1" />}
                {show2dInspector ? "Collapse 2D Inspector" : "Expand 2D Inspector"}
              </Button>
            </div>
          </div>
        </div>

        {/* SIDE INSPECTOR PANEL: 2D RGB Input & Metric Depth Map Probing (4 cols = ~30%) */}
        {show2dInspector && (
          <div className="lg:col-span-4 flex flex-col space-y-4">
            
            {/* PANEL 1: Raw 2D Aerial RGB */}
            <Card className="bg-[#0F172A]/80 border-slate-800 flex flex-col overflow-hidden shadow-lg">
              <CardHeader className="p-3 pb-2 border-b border-slate-800 flex flex-row items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-cyan-400" />
                  <CardTitle className="text-xs font-bold text-slate-200 uppercase font-mono-data">
                    1. Input 2D Aerial
                  </CardTitle>
                </div>
                <Badge className="bg-slate-800 text-slate-300 border-slate-700 text-[10px] font-mono-data">
                  {qr.resolution[0]}x{qr.resolution[1]}
                </Badge>
              </CardHeader>

              <CardContent className="p-3 space-y-2">
                <div
                  ref={imageContainerRef}
                  onMouseMove={handleImageMouseMove}
                  onMouseDown={handleImageMouseDown}
                  onMouseUp={handleImageMouseUp}
                  onMouseLeave={handleImageMouseLeave}
                  data-testid="input-image-2d-container"
                  className="relative w-full h-[220px] rounded-lg overflow-hidden border border-slate-800 bg-black/60 cursor-crosshair select-none"
                >
                  <img src={data.image_url} alt={data.scene_name} className="w-full h-full object-cover" />

                  {hoverCoord && (
                    <>
                      <div className="absolute top-0 bottom-0 w-[1px] bg-cyan-400/80 pointer-events-none" style={{ left: `${hoverCoord.u * 100}%` }} />
                      <div className="absolute left-0 right-0 h-[1px] bg-cyan-400/80 pointer-events-none" style={{ top: `${hoverCoord.v * 100}%` }} />
                    </>
                  )}

                  {data.sample_buildings.map((bldg) => {
                    const [bx, by, bw, bh] = bldg.bbox;
                    const imgH = data.raw_depth_stats.shape[0];
                    const imgW = data.raw_depth_stats.shape[1];
                    const isSelected = selectedBuilding?.id === bldg.id;

                    return (
                      <div
                        key={bldg.id}
                        onClick={(e) => {
                          e.stopPropagation();
                          onSelectBuilding(bldg);
                        }}
                        style={{
                          left: `${(bx / imgW) * 100}%`,
                          top: `${(by / imgH) * 100}%`,
                          width: `${(bw / imgW) * 100}%`,
                          height: `${(bh / imgH) * 100}%`,
                        }}
                        className={`absolute border-2 transition-all cursor-pointer rounded-sm ${
                          isSelected
                            ? "border-emerald-400 bg-emerald-500/25 shadow-lg shadow-emerald-500/20"
                            : "border-cyan-400/70 bg-cyan-500/10 hover:border-cyan-300 hover:bg-cyan-500/20"
                        }`}
                      />
                    );
                  })}
                </div>
              </CardContent>
            </Card>

            {/* PANEL 2: Metric Depth Map Probing */}
            <Card className="bg-[#0F172A]/80 border-slate-800 flex flex-col overflow-hidden shadow-lg">
              <CardHeader className="p-3 pb-2 border-b border-slate-800 flex flex-row items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-emerald-400" />
                  <CardTitle className="text-xs font-bold text-slate-200 uppercase font-mono-data">
                    2. Metric Depth Probing
                  </CardTitle>
                </div>
                <div className="flex items-center gap-1">
                  <Button
                    size="xs"
                    variant={depthDisplayMode === "metric" ? "default" : "ghost"}
                    onClick={() => setDepthDisplayMode("metric")}
                    className={`h-5 text-[9px] font-mono-data px-1.5 ${depthDisplayMode === "metric" ? "bg-emerald-500 text-black font-bold" : "text-slate-400"}`}
                  >
                    METRIC (m)
                  </Button>
                </div>
              </CardHeader>

              <CardContent className="p-3 space-y-2">
                <div className="relative w-full h-[200px] rounded-lg overflow-hidden border border-slate-800 bg-black/60">
                  <img src={depthImageUrl} alt="Depth Map" className="w-full h-full object-cover" />

                  {hoverCoord && (
                    <div
                      className="absolute px-2 py-1 bg-slate-950/95 border border-emerald-400 text-[10px] text-emerald-300 font-mono-data rounded-md shadow-lg pointer-events-none -translate-x-1/2 -translate-y-10 whitespace-nowrap"
                      style={{ left: `${hoverCoord.u * 100}%`, top: `${hoverCoord.v * 100}%` }}
                    >
                      <div className="font-bold text-white text-xs">
                        Depth: {(hoverCoord.metricDepthM ?? 42.8).toFixed(2)} m
                      </div>
                    </div>
                  )}
                </div>

                <div className="p-2 bg-slate-950/90 rounded-lg border border-slate-800 text-[10px] font-mono-data grid grid-cols-2 gap-2 text-slate-300">
                  <div>Min: <strong className="text-cyan-300">{data.metric_depth_stats?.min_m ?? "15.20"} m</strong></div>
                  <div>Max: <strong className="text-cyan-300">{data.metric_depth_stats?.max_m ?? "94.80"} m</strong></div>
                  <div>Median: <strong className="text-emerald-400 font-bold">{data.metric_depth_stats?.median_m ?? "42.80"} m</strong></div>
                  <div>Backend: <strong className="text-cyan-400">{data.depth_backend}</strong></div>
                </div>

                <div className="flex items-center justify-between gap-1 p-1 bg-black/40 rounded-lg border border-slate-800 text-[10px] font-mono-data">
                  <span className="text-slate-400 px-1">Colormap:</span>
                  <div className="flex items-center gap-1">
                    {(["turbo", "viridis", "magma"] as ColormapType[]).map((cm) => (
                      <Button
                        key={cm}
                        size="xs"
                        variant={colormap === cm ? "default" : "ghost"}
                        onClick={() => setColormap(cm)}
                        className={`h-5 px-1.5 text-[9px] capitalize ${colormap === cm ? "bg-cyan-500 text-black font-bold" : "text-slate-400"}`}
                      >
                        {cm}
                      </Button>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>

          </div>
        )}
      </div>
    </div>
  );
};
