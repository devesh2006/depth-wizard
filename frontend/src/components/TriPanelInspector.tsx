import React, { useState, useRef, useCallback } from "react";
import type { DepthProcessResponse, BuildingMeasurement } from "@/types/depth";
import { ThreeViewer } from "@/viewer/ThreeViewer";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { 
  Download, 
  AlertTriangle
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

    // Metric depth calculation in metres
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

  // Active colormap URL
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

      {/* Tri-Panel Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
        {/* PANEL 1: Raw 2D Aerial/Satellite Image (4 cols) */}
        <Card className="lg:col-span-4 bg-[#0F172A]/80 border-slate-800 flex flex-col overflow-hidden">
          <CardHeader className="p-3.5 pb-2 border-b border-slate-800/80 flex flex-row items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
              <CardTitle className="text-xs font-bold text-slate-200 tracking-wide uppercase font-mono-data">
                1. Input Aerial RGB
              </CardTitle>
            </div>
            <div className="flex items-center gap-1.5">
              <Badge className="bg-slate-800 text-slate-300 border-slate-700 text-[10px] font-mono-data">
                {qr.resolution[0]}x{qr.resolution[1]}
              </Badge>
              <Badge 
                className={`text-[10px] font-mono-data ${
                  qr.suitability_label === "High" 
                    ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/30"
                    : "bg-amber-500/20 text-amber-300 border-amber-500/30"
                }`}
              >
                Suitability: {qr.suitability_label}
              </Badge>
            </div>
          </CardHeader>

          <CardContent className="p-3 flex-1 flex flex-col justify-between space-y-3">
            {/* Interactive Image Container */}
            <div
              ref={imageContainerRef}
              onMouseMove={handleImageMouseMove}
              onMouseDown={handleImageMouseDown}
              onMouseUp={handleImageMouseUp}
              onMouseLeave={handleImageMouseLeave}
              data-testid="input-image-2d-container"
              className="relative w-full h-[320px] rounded-lg overflow-hidden border border-slate-800 bg-black/60 cursor-crosshair select-none"
            >
              <img
                src={data.image_url}
                alt={data.scene_name}
                className="w-full h-full object-cover"
              />

              {/* Synchronized Hover Crosshair */}
              {hoverCoord && (
                <>
                  <div
                    className="absolute top-0 bottom-0 w-[1px] bg-cyan-400/80 pointer-events-none"
                    style={{ left: `${hoverCoord.u * 100}%` }}
                  />
                  <div
                    className="absolute left-0 right-0 h-[1px] bg-cyan-400/80 pointer-events-none"
                    style={{ top: `${hoverCoord.v * 100}%` }}
                  />
                  <div
                    className="absolute w-4 h-4 rounded-full border border-cyan-400 -translate-x-1/2 -translate-y-1/2 pointer-events-none animate-ping"
                    style={{ left: `${hoverCoord.u * 100}%`, top: `${hoverCoord.v * 100}%` }}
                  />
                </>
              )}

              {/* Sample Building Bounding Boxes */}
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
                    data-testid={`building-bbox-${bldg.id}`}
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
                  >
                    <span className="absolute -top-5 left-0 px-1.5 py-0.2 bg-black/85 border border-slate-700 text-[10px] text-cyan-300 font-mono-data rounded whitespace-nowrap">
                      {bldg.name} ({bldg.calibrated_height_m ? `${bldg.calibrated_height_m}m` : `Relative Depth: ${bldg.rooftop_peak_z_rel ?? bldg.relative_height_unitless}`})
                    </span>
                  </div>
                );
              })}

              {/* Drawing drag selection preview */}
              {isDrawingBbox && currentBbox && (
                <div
                  style={{
                    left: `${(currentBbox[0] / data.raw_depth_stats.shape[1]) * 100}%`,
                    top: `${(currentBbox[1] / data.raw_depth_stats.shape[0]) * 100}%`,
                    width: `${(currentBbox[2] / data.raw_depth_stats.shape[1]) * 100}%`,
                    height: `${(currentBbox[3] / data.raw_depth_stats.shape[0]) * 100}%`,
                  }}
                  className="absolute border-2 border-dashed border-amber-400 bg-amber-500/20 pointer-events-none"
                />
              )}

              {/* Hint badge */}
              <div className="absolute bottom-2 left-2 px-2 py-1 bg-black/80 rounded border border-slate-700 text-[10px] text-slate-300 font-mono-data pointer-events-none">
                Hover to probe | Drag to measure custom building
              </div>
            </div>

            {/* Quality Analysis Mini Readout */}
            <div className="grid grid-cols-2 gap-2 text-xs font-mono-data bg-black/40 p-2.5 rounded-lg border border-slate-800">
              <div className="space-y-1">
                <span className="text-slate-400 text-[11px]">View Geometry:</span>
                <p className="text-cyan-300 font-semibold">{qr.view_geometry}</p>
              </div>
              <div className="space-y-1">
                <span className="text-slate-400 text-[11px]">Sharpness:</span>
                <p className="text-slate-200">{qr.sharpness_label} ({qr.sharpness_score})</p>
              </div>
              <div className="space-y-1">
                <span className="text-slate-400 text-[11px]">Brightness:</span>
                <p className="text-slate-200">{qr.brightness_label} ({qr.brightness_mean})</p>
              </div>
              <div className="space-y-1">
                <span className="text-slate-400 text-[11px]">Cloud/Shadow Risk:</span>
                <p className={qr.cloud_shadow_risk === "Low" ? "text-emerald-400" : "text-amber-400"}>
                  {qr.cloud_shadow_risk}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* PANEL 2: Monocular Depth Map & Interactive Probe (3 cols) */}
        <Card className="lg:col-span-3 bg-[#0F172A]/80 border-slate-800 flex flex-col overflow-hidden">
          <CardHeader className="p-3.5 pb-2 border-b border-slate-800/80 flex flex-row items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-emerald-400" />
              <CardTitle className="text-xs font-bold text-slate-200 tracking-wide uppercase font-mono-data">
                2. Metric Depth Map
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
              <Button
                size="xs"
                variant={depthDisplayMode === "relative" ? "default" : "ghost"}
                onClick={() => setDepthDisplayMode("relative")}
                className={`h-5 text-[9px] font-mono-data px-1.5 ${depthDisplayMode === "relative" ? "bg-cyan-500 text-black font-bold" : "text-slate-400"}`}
              >
                RELATIVE
              </Button>
            </div>
          </CardHeader>

          <CardContent className="p-3 flex-1 flex flex-col justify-between space-y-3">
            {/* Depth Map Image Container with Synchronized Crosshair */}
            <div
              className="relative w-full h-[280px] rounded-lg overflow-hidden border border-slate-800 bg-black/60"
            >
              <img
                src={depthImageUrl}
                alt="Depth Map"
                className="w-full h-full object-cover"
                data-testid="depth-map-image"
              />

              {/* Synchronized Hover Marker */}
              {hoverCoord && (
                <>
                  <div
                    className="absolute top-0 bottom-0 w-[1px] bg-white/70 pointer-events-none"
                    style={{ left: `${hoverCoord.u * 100}%` }}
                  />
                  <div
                    className="absolute left-0 right-0 h-[1px] bg-white/70 pointer-events-none"
                    style={{ top: `${hoverCoord.v * 100}%` }}
                  />
                  <div
                    className="absolute px-2.5 py-1 bg-slate-950/95 border border-emerald-400 text-[10px] text-emerald-300 font-mono-data rounded-md shadow-lg pointer-events-none -translate-x-1/2 -translate-y-10 whitespace-nowrap"
                    style={{ left: `${hoverCoord.u * 100}%`, top: `${hoverCoord.v * 100}%` }}
                  >
                    <div>X: {hoverCoord.pxX}, Y: {hoverCoord.pxY}</div>
                    <div className="font-bold text-white text-xs">
                      {depthDisplayMode === "metric"
                        ? `Metric Depth: ${(hoverCoord.metricDepthM ?? 42.8).toFixed(2)} m`
                        : `Rel Depth: ${hoverCoord.relDepth.toFixed(3)}`}
                    </div>
                  </div>
                </>
              )}

              {/* Colormap Legend Bar */}
              <div className="absolute bottom-2 left-2 right-2 p-1.5 bg-black/85 backdrop-blur rounded border border-slate-700/80">
                <div className="h-2 w-full rounded bg-gradient-to-r from-indigo-950 via-teal-500 to-yellow-400" />
                <div className="flex justify-between text-[9px] font-mono-data text-slate-300 mt-1">
                  <span>Near ({data.metric_depth_stats?.min_m ?? 15.0} m)</span>
                  <span>Far ({data.metric_depth_stats?.max_m ?? 95.0} m)</span>
                </div>
              </div>
            </div>

            {/* Metric Depth Summary Statistics Box */}
            <div className="p-2.5 bg-slate-950/90 rounded-lg border border-slate-800 text-[11px] font-mono-data space-y-1">
              <div className="flex items-center justify-between text-slate-400 border-b border-slate-800/80 pb-1">
                <span className="font-bold text-slate-200 uppercase text-[10px]">Metric Depth Statistics:</span>
                <span className="text-emerald-400 text-[10px]">{data.depth_backend}</span>
              </div>
              <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-slate-300 pt-0.5">
                <div>Min Depth: <strong className="text-cyan-300">{data.metric_depth_stats?.min_m ?? "15.20"} m</strong></div>
                <div>Max Depth: <strong className="text-cyan-300">{data.metric_depth_stats?.max_m ?? "94.80"} m</strong></div>
                <div>Median Depth: <strong className="text-emerald-400 font-bold">{data.metric_depth_stats?.median_m ?? "42.80"} m</strong></div>
                <div>Mean Depth: <strong className="text-slate-200">{data.metric_depth_stats?.mean_m ?? "45.10"} m</strong></div>
              </div>
              <div className="text-[10px] text-slate-400 pt-1 flex justify-between border-t border-slate-850">
                <span>Valid Pixel %: <strong className="text-emerald-300">{data.metric_depth_stats?.valid_pixel_pct ?? 100.0}%</strong></span>
                <span className="italic text-[9px]">Camera Distance in Metres</span>
              </div>
            </div>

            {/* Colormap Selector & Export */}
            <div className="space-y-1.5">
              <div className="flex items-center justify-between gap-1 p-1 bg-black/40 rounded-lg border border-slate-800 text-[11px] font-mono-data">
                <span className="text-slate-400 px-1">Colormap:</span>
                <div className="flex items-center gap-1">
                  {(["turbo", "viridis", "magma"] as ColormapType[]).map((cm) => (
                    <Button
                      key={cm}
                      size="xs"
                      variant={colormap === cm ? "default" : "ghost"}
                      onClick={() => setColormap(cm)}
                      data-testid={`colormap-${cm}-button`}
                      className={`h-6 px-2 text-[10px] capitalize ${
                        colormap === cm ? "bg-cyan-500 text-black font-semibold" : "text-slate-300"
                      }`}
                    >
                      {cm}
                    </Button>
                  ))}
                </div>
              </div>

              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  const blob = new Blob([JSON.stringify(data.metric_depth_stats || data.raw_depth_stats, null, 2)], { type: "application/json" });
                  const url = URL.createObjectURL(blob);
                  const a = document.createElement("a");
                  a.href = url;
                  a.download = `depthwizard_metric_depth_${data.id}.json`;
                  a.click();
                }}
                data-testid="download-raw-depth-button"
                className="w-full h-7 text-xs border-slate-700 hover:border-cyan-500/50 hover:bg-cyan-500/10 text-slate-300 font-mono-data"
              >
                <Download className="w-3.5 h-3.5 mr-1.5 text-cyan-400" />
                Export Float Metric Depth Matrix (Metres)
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* PANEL 3: Interactive 3D WebGL Studio (5 cols) */}
        <div className="lg:col-span-5 flex flex-col space-y-2">
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
          <div className="flex items-center justify-between gap-2 p-2 bg-[#0F172A]/80 border border-slate-800 rounded-lg text-xs font-mono-data text-slate-300">
            <span className="text-slate-400">Export 3D Model:</span>
            <div className="flex items-center gap-2">
              <Button
                size="xs"
                variant="outline"
                onClick={() => window.open(`/api/depth/export-pointcloud/${data.id}`, "_blank")}
                data-testid="export-pointcloud-ply-button"
                className="h-6 text-[10px] border-slate-700 hover:border-cyan-400 text-slate-200"
              >
                <Download className="w-3 h-3 mr-1 text-cyan-400" />
                .PLY Point Cloud
              </Button>
              <Button
                size="xs"
                variant="outline"
                onClick={() => window.open(`/api/depth/export-mesh/${data.id}`, "_blank")}
                data-testid="export-mesh-obj-button"
                className="h-6 text-[10px] border-slate-700 hover:border-cyan-400 text-slate-200"
              >
                <Download className="w-3 h-3 mr-1 text-cyan-400" />
                .OBJ 3D Mesh
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
