import React, { useEffect, useRef, useState, useCallback } from "react";
import * as THREE from "three";
import type { PointCloudData, MeshHeightfieldData, BuildingMeasurement } from "@/types/depth";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { 
  RotateCcw, 
  Grid3X3,
  Boxes,
  Maximize2,
  Minimize2,
  Image as ImageIcon,
  Sparkles,
  Building2,
  Layers,
  Sliders,
  ShieldCheck,
  Ruler,
  Navigation,
  Crosshair,
  SlidersHorizontal,
  ChevronDown,
  Check,
  Flame,
  Box
} from "lucide-react";

interface ThreeViewerProps {
  pointcloud?: PointCloudData | null;
  mesh?: MeshHeightfieldData | null;
  imageUrl?: string;
  depthColormapUrl?: string;
  selectedBuilding?: BuildingMeasurement | null;
  sampleBuildings?: BuildingMeasurement[];
  onSelectBuilding?: (building: BuildingMeasurement | null) => void;
  hoverCoordinate?: { u: number; v: number; relDepth: number } | null;
  heightScale?: number;
  depthShape?: [number, number];
  isLoading?: boolean;
}

export type RenderMode = "mesh" | "depth" | "heatmap" | "wireframe" | "pointcloud" | "voxel";
export type ScaleMode = "scientific" | "exploration";
export type CameraPreset = "isometric" | "top" | "orbit" | "street" | "focus";

export const ThreeViewer: React.FC<ThreeViewerProps> = ({
  mesh,
  imageUrl,
  depthColormapUrl,
  selectedBuilding,
  sampleBuildings = [],
  onSelectBuilding,
  hoverCoordinate,
  depthShape,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  
  const [renderMode, setRenderMode] = useState<RenderMode>("mesh");
  const [scaleMode, setScaleMode] = useState<ScaleMode>("scientific");
  const [verticalExaggeration, setVerticalExaggeration] = useState<number>(1.0);
  const [cameraPreset, setCameraPresetState] = useState<CameraPreset>("orbit");

  const [isFlyModeActive, setIsFlyModeActive] = useState<boolean>(false);
  const [isMeasurementMode, setIsMeasurementMode] = useState<boolean>(false);
  const [showAerialThumbnail, setShowAerialThumbnail] = useState<boolean>(true);
  const [showDebugHud, setShowDebugHud] = useState<boolean>(false);

  const [showGrid, setShowGrid] = useState<boolean>(true);
  const [showAxes, setShowAxes] = useState<boolean>(false);
  const [showBoundingBox, setShowBoundingBox] = useState<boolean>(false);
  const [showAnalysisMenu, setShowAnalysisMenu] = useState<boolean>(false);

  const [measurePoints, setMeasurePoints] = useState<THREE.Vector3[]>([]);
  const [measureDistanceM, setMeasureDistanceM] = useState<number | null>(null);

  const [debugStats, setDebugStats] = useState<{ fps: number; drawCalls: number; triangles: number; points: number }>({
    fps: 60,
    drawCalls: 0,
    triangles: 0,
    points: 0,
  });

  const [hoveredBuilding, setHoveredBuilding] = useState<BuildingMeasurement | null>(null);
  const [hoverTooltip, setHoverTooltip] = useState<{ 
    x: number; 
    y: number; 
    info: string; 
    height?: string;
    depthM?: string;
    surfaceType?: string;
  } | null>(null);

  const [headingDeg, setHeadingDeg] = useState<number>(45);

  const sceneRef = useRef<THREE.Scene | null>(null);
  const sceneRootRef = useRef<THREE.Group | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);

  const terrainMeshRef = useRef<THREE.Mesh | null>(null);
  const depthMeshRef = useRef<THREE.Mesh | null>(null);
  const wireframeMeshRef = useRef<THREE.Mesh | null>(null);
  const heatmapMeshRef = useRef<THREE.Mesh | null>(null);
  const pointCloudPointsRef = useRef<THREE.Points | null>(null);
  const voxelGroupRef = useRef<THREE.Group | null>(null);
  const buildingsGroupRef = useRef<THREE.Group | null>(null);
  const treesGroupRef = useRef<THREE.Group | null>(null);
  const cursorMarkerRef = useRef<THREE.Group | null>(null);
  const buildingBoxRef = useRef<THREE.LineSegments | null>(null);

  const gridHelperRef = useRef<THREE.GridHelper | null>(null);
  const axesHelperRef = useRef<THREE.AxesHelper | null>(null);
  const boxHelperRef = useRef<THREE.BoxHelper | null>(null);
  const measureLineRef = useRef<THREE.Line | null>(null);
  const measureMarkersGroupRef = useRef<THREE.Group | null>(null);

  const loadedRgbTextureRef = useRef<THREE.Texture | null>(null);
  const loadedDepthTextureRef = useRef<THREE.Texture | null>(null);

  const cameraTarget = useRef<THREE.Vector3>(new THREE.Vector3(0, 0, 0));
  const desiredCameraTarget = useRef<THREE.Vector3>(new THREE.Vector3(0, 0, 0));
  const cameraAngle = useRef<{ theta: number; phi: number; radius: number }>({
    theta: Math.PI / 4,
    phi: Math.PI / 3.4,
    radius: 140,
  });
  const desiredCameraAngle = useRef<{ theta: number; phi: number; radius: number }>({
    theta: Math.PI / 4,
    phi: Math.PI / 3.4,
    radius: 140,
  });

  const animFrameId = useRef<number | null>(null);
  const isMouseDown = useRef<boolean>(false);
  const mouseButton = useRef<number>(0);
  const mousePrev = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
  const keysPressed = useRef<{ [key: string]: boolean }>({});
  const lastRaycastTime = useRef<number>(0);
  const frameCountRef = useRef<number>(0);
  const lastFpsCalcTimeRef = useRef<number>(performance.now());

  const activeExaggeration = scaleMode === "scientific" ? 1.0 : verticalExaggeration;

  const fitCameraToScene = useCallback(() => {
    if (!sceneRootRef.current || !cameraRef.current) return;

    const box = new THREE.Box3().setFromObject(sceneRootRef.current);
    if (box.isEmpty()) return;

    const sphere = new THREE.Sphere();
    box.getBoundingSphere(sphere);

    desiredCameraTarget.current.copy(sphere.center);
    desiredCameraAngle.current.radius = Math.max(35, Math.min(300, sphere.radius * 2.1));
    desiredCameraAngle.current.theta = Math.PI / 3.8;
    desiredCameraAngle.current.phi = Math.PI / 3.4;

    cameraRef.current.near = Math.max(0.1, sphere.radius * 0.01);
    cameraRef.current.far = Math.max(1000, sphere.radius * 12);
    cameraRef.current.updateProjectionMatrix();

    setCameraPresetState("orbit");
  }, []);

  const updateCameraPosition = useCallback(() => {
    if (!cameraRef.current) return;

    const dt = 0.15;
    cameraAngle.current.theta += (desiredCameraAngle.current.theta - cameraAngle.current.theta) * dt;
    cameraAngle.current.phi += (desiredCameraAngle.current.phi - cameraAngle.current.phi) * dt;
    cameraAngle.current.radius += (desiredCameraAngle.current.radius - cameraAngle.current.radius) * dt;
    cameraTarget.current.lerp(desiredCameraTarget.current, dt);

    const { theta, phi, radius } = cameraAngle.current;
    const target = cameraTarget.current;

    const x = target.x + radius * Math.sin(phi) * Math.sin(theta);
    const y = target.y + radius * Math.cos(phi);
    const z = target.z + radius * Math.sin(phi) * Math.cos(theta);

    cameraRef.current.position.set(x, y, z);
    cameraRef.current.lookAt(target);

    const deg = Math.round((((theta * 180) / Math.PI) % 360 + 360) % 360);
    setHeadingDeg(deg);
  }, []);

  useEffect(() => {
    if (!imageUrl) return;
    const loader = new THREE.TextureLoader();
    loader.crossOrigin = "anonymous";
    loader.load(imageUrl, (tex) => {
      tex.colorSpace = THREE.SRGBColorSpace;
      loadedRgbTextureRef.current = tex;
      if (terrainMeshRef.current) {
        const mat = terrainMeshRef.current.material as THREE.MeshStandardMaterial;
        mat.map = tex;
        mat.vertexColors = false;
        mat.needsUpdate = true;
      }
    });
  }, [imageUrl]);

  useEffect(() => {
    if (!depthColormapUrl) return;
    const loader = new THREE.TextureLoader();
    loader.crossOrigin = "anonymous";
    loader.load(depthColormapUrl, (tex) => {
      tex.colorSpace = THREE.SRGBColorSpace;
      loadedDepthTextureRef.current = tex;
      if (depthMeshRef.current) {
        const mat = depthMeshRef.current.material as THREE.MeshStandardMaterial;
        mat.map = tex;
        mat.vertexColors = false;
        mat.needsUpdate = true;
      }
    });
  }, [depthColormapUrl]);

  useEffect(() => {
    if (!containerRef.current) return;
    const container = containerRef.current;
    const width = container.clientWidth || 800;
    const height = container.clientHeight || 580;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x070a0f);
    scene.fog = new THREE.FogExp2(0x070a0f, 0.0015);
    sceneRef.current = scene;

    const sceneRoot = new THREE.Group();
    scene.add(sceneRoot);
    sceneRootRef.current = sceneRoot;

    const camera = new THREE.PerspectiveCamera(42, width / height, 0.5, 2000);
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.75));
    renderer.shadowMap.enabled = true;

    container.innerHTML = "";
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.85);
    scene.add(ambientLight);

    const sunLight = new THREE.DirectionalLight(0xfff7ed, 1.6);
    sunLight.position.set(90, 150, 80);
    sunLight.castShadow = true;
    scene.add(sunLight);

    const skyFill = new THREE.DirectionalLight(0x38bdf8, 0.4);
    skyFill.position.set(-90, 70, -90);
    scene.add(skyFill);

    const trayGeo = new THREE.BoxGeometry(112, 4, 112);
    const trayMat = new THREE.MeshStandardMaterial({ color: 0x0f172a, roughness: 0.85, metalness: 0.15 });
    const trayMesh = new THREE.Mesh(trayGeo, trayMat);
    trayMesh.position.y = -2.1;
    trayMesh.receiveShadow = true;
    scene.add(trayMesh);

    const gridHelper = new THREE.GridHelper(112, 36, 0x06b6d4, 0x1e293b);
    gridHelper.position.y = -0.05;
    scene.add(gridHelper);
    gridHelperRef.current = gridHelper;

    const axesHelper = new THREE.AxesHelper(35);
    axesHelper.position.y = 0.1;
    axesHelper.visible = false;
    scene.add(axesHelper);
    axesHelperRef.current = axesHelper;

    const boxHelper = new THREE.BoxHelper(sceneRoot, 0x06b6d4);
    boxHelper.visible = false;
    scene.add(boxHelper);
    boxHelperRef.current = boxHelper;

    const measureGroup = new THREE.Group();
    scene.add(measureGroup);
    measureMarkersGroupRef.current = measureGroup;

    const markerGroup = new THREE.Group();
    const pinGeo = new THREE.CylinderGeometry(0.2, 0.8, 12, 12);
    const pinMat = new THREE.MeshBasicMaterial({ color: 0x06b6d4 });
    const pinMesh = new THREE.Mesh(pinGeo, pinMat);
    pinMesh.position.y = 6;
    markerGroup.add(pinMesh);

    const ringGeo = new THREE.RingGeometry(1.5, 2.5, 32);
    const ringMat = new THREE.MeshBasicMaterial({ color: 0x06b6d4, side: THREE.DoubleSide });
    const ringMesh = new THREE.Mesh(ringGeo, ringMat);
    ringMesh.rotation.x = -Math.PI / 2;
    ringMesh.position.y = 0.2;
    markerGroup.add(ringMesh);

    markerGroup.visible = false;
    scene.add(markerGroup);
    cursorMarkerRef.current = markerGroup;

    const handleKeyDown = (e: KeyboardEvent) => {
      keysPressed.current[e.key.toLowerCase()] = true;
      if (e.key.toLowerCase() === "r") fitCameraToScene();
      else if (e.key === "Escape") {
        onSelectBuilding?.(null);
        clearMeasurement();
      }
    };

    const handleKeyUp = (e: KeyboardEvent) => {
      keysPressed.current[e.key.toLowerCase()] = false;
    };

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);

    const animate = () => {
      animFrameId.current = requestAnimationFrame(animate);
      if (isFlyModeActive) {
        const moveSpeed = keysPressed.current["shift"] ? 1.8 : 0.8;
        const { theta } = cameraAngle.current;
        const right = new THREE.Vector3(1, 0, 0).applyAxisAngle(new THREE.Vector3(0, 1, 0), theta);
        const forward = new THREE.Vector3(0, 0, -1).applyAxisAngle(new THREE.Vector3(0, 1, 0), theta);
        if (keysPressed.current["w"]) desiredCameraTarget.current.addScaledVector(forward, moveSpeed);
        if (keysPressed.current["s"]) desiredCameraTarget.current.addScaledVector(forward, -moveSpeed);
        if (keysPressed.current["a"]) desiredCameraTarget.current.addScaledVector(right, -moveSpeed);
        if (keysPressed.current["d"]) desiredCameraTarget.current.addScaledVector(right, moveSpeed);
        if (keysPressed.current[" "]) desiredCameraTarget.current.y += moveSpeed;
        if (keysPressed.current["control"]) desiredCameraTarget.current.y = Math.max(-5, desiredCameraTarget.current.y - moveSpeed);
      }
      updateCameraPosition();
      if (rendererRef.current && sceneRef.current && cameraRef.current) {
        rendererRef.current.render(sceneRef.current, cameraRef.current);
        frameCountRef.current++;
        const now = performance.now();
        if (now - lastFpsCalcTimeRef.current >= 1000) {
          setDebugStats({ fps: Math.round((frameCountRef.current * 1000) / (now - lastFpsCalcTimeRef.current)), drawCalls: rendererRef.current.info.render.calls, triangles: rendererRef.current.info.render.triangles, points: rendererRef.current.info.render.points });
          frameCountRef.current = 0;
          lastFpsCalcTimeRef.current = now;
        }
      }
    };
    animate();

    const handleResize = () => {
      const w = container.clientWidth;
      const h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
      if (animFrameId.current) cancelAnimationFrame(animFrameId.current);
      renderer.dispose();
    };
  }, [updateCameraPosition, isFlyModeActive, fitCameraToScene, onSelectBuilding]);

  useEffect(() => {
    if (gridHelperRef.current) gridHelperRef.current.visible = showGrid;
    if (axesHelperRef.current) axesHelperRef.current.visible = showAxes;
    if (boxHelperRef.current) boxHelperRef.current.visible = showBoundingBox;
  }, [showGrid, showAxes, showBoundingBox]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const onMouseDown = (e: MouseEvent) => {
      isMouseDown.current = true;
      mouseButton.current = e.button;
      mousePrev.current = { x: e.clientX, y: e.clientY };

      if (isMeasurementMode && e.button === 0 && cameraRef.current) {
        const rect = container.getBoundingClientRect();
        const mouseX = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        const mouseY = -((e.clientY - rect.top) / rect.height) * 2 + 1;
        const raycaster = new THREE.Raycaster();
        raycaster.setFromCamera(new THREE.Vector2(mouseX, mouseY), cameraRef.current);
        const intersects = raycaster.intersectObjects(sceneRootRef.current ? [sceneRootRef.current] : [], true);
        if (intersects.length > 0) {
          const hitPoint = intersects[0].point.clone();
          setMeasurePoints((prev) => {
            if (prev.length >= 2) return [hitPoint];
            const next = [...prev, hitPoint];
            if (next.length === 2) {
              const dist3d = next[0].distanceTo(next[1]);
              const distM = (dist3d / 100.0) * (depthShape?.[1] ? 42.5 : 40.0);
              setMeasureDistanceM(distM);
              renderMeasurementLine(next[0], next[1]);
            }
            return next;
          });
        }
      }
    };

    const onMouseMove = (e: MouseEvent) => {
      if (!container || !cameraRef.current) return;
      if (isMouseDown.current && !isMeasurementMode) {
        const dx = e.clientX - mousePrev.current.x;
        const dy = e.clientY - mousePrev.current.y;
        mousePrev.current = { x: e.clientX, y: e.clientY };
        if (mouseButton.current === 0 && !e.shiftKey) {
          desiredCameraAngle.current.theta -= dx * 0.0075;
          desiredCameraAngle.current.phi = Math.max(0.08, Math.min(Math.PI / 2 - 0.02, desiredCameraAngle.current.phi + dy * 0.0075));
        } else {
          const right = new THREE.Vector3(1, 0, 0).applyAxisAngle(new THREE.Vector3(0, 1, 0), cameraAngle.current.theta);
          const forward = new THREE.Vector3(0, 0, 1).applyAxisAngle(new THREE.Vector3(0, 1, 0), cameraAngle.current.theta);
          const panSpeed = cameraAngle.current.radius * 0.0012;
          desiredCameraTarget.current.addScaledVector(right, -dx * panSpeed);
          desiredCameraTarget.current.addScaledVector(forward, -dy * panSpeed);
        }
        updateCameraPosition();
      } else {
        const now = performance.now();
        if (now - lastRaycastTime.current < 35) return;
        lastRaycastTime.current = now;
        const rect = container.getBoundingClientRect();
        const mouseX = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        const mouseY = -((e.clientY - rect.top) / rect.height) * 2 + 1;
        const raycaster = new THREE.Raycaster();
        raycaster.setFromCamera(new THREE.Vector2(mouseX, mouseY), cameraRef.current);
        if (buildingsGroupRef.current) {
          const intersects = raycaster.intersectObjects(buildingsGroupRef.current.children, true);
          if (intersects.length > 0) {
            const hitObj = intersects[0].object;
            const bId = hitObj.userData?.buildingId;
            if (bId) {
              const matchedBldg = sampleBuildings.find((b) => b.id === bId);
              if (matchedBldg) {
                setHoveredBuilding(matchedBldg);
                setHoverTooltip({
                  x: e.clientX - rect.left,
                  y: e.clientY - rect.top,
                  info: matchedBldg.name,
                  height: matchedBldg.calibrated_height_m ? `${matchedBldg.calibrated_height_m} m` : `Peak: ${matchedBldg.rooftop_peak_z_rel.toFixed(2)} rel`,
                  depthM: matchedBldg.depth_p50_m ? `${matchedBldg.depth_p50_m.toFixed(1)} m` : undefined,
                  surfaceType: "Building Roof Plate"
                });
                return;
              }
            } else {
              setHoveredBuilding(null);
              setHoverTooltip({ x: e.clientX - rect.left, y: e.clientY - rect.top, info: "Secondary Structure", height: "Extruded Volume", surfaceType: "Building Facade Wall" });
              return;
            }
          }
        }
        if (terrainMeshRef.current) {
          const intersects = raycaster.intersectObject(terrainMeshRef.current);
          if (intersects.length > 0) {
            const hitPoint = intersects[0].point;
            const relDepth = Math.max(0, Math.min(1, hitPoint.y / 28.0));
            const depthM = (1.0 - relDepth) * 45.0 + 15.0;
            setHoveredBuilding(null);
            setHoverTooltip({ x: e.clientX - rect.left, y: e.clientY - rect.top, info: "Terrain Surface", depthM: `${depthM.toFixed(1)} m`, surfaceType: hitPoint.y > 0.5 ? "Vegetation / Ground" : "Asphalt / Road Surface" });
            return;
          }
        }
        setHoveredBuilding(null);
        setHoverTooltip(null);
      }
    };

    const onMouseUp = (e: MouseEvent) => {
      isMouseDown.current = false;
      if (hoveredBuilding && e.button === 0 && !isMeasurementMode) {
        onSelectBuilding?.(hoveredBuilding);
        focusBuilding(hoveredBuilding);
      }
    };

    const onDoubleClick = (e: MouseEvent) => {
      if (hoveredBuilding) {
        onSelectBuilding?.(hoveredBuilding);
        focusBuilding(hoveredBuilding);
      }
    };

    const onContextMenu = (e: MouseEvent) => e.preventDefault();
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      desiredCameraAngle.current.radius = Math.max(15, Math.min(320, desiredCameraAngle.current.radius + e.deltaY * 0.14));
      updateCameraPosition();
    };

    container.addEventListener("mousedown", onMouseDown);
    container.addEventListener("contextmenu", onContextMenu);
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
    container.addEventListener("dblclick", onDoubleClick);
    container.addEventListener("wheel", onWheel, { passive: false });

    return () => {
      container.removeEventListener("mousedown", onMouseDown);
      container.removeEventListener("contextmenu", onContextMenu);
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);
      container.removeEventListener("dblclick", onDoubleClick);
      container.removeEventListener("wheel", onWheel);
    };
  }, [updateCameraPosition, hoveredBuilding, sampleBuildings, onSelectBuilding, isMeasurementMode, depthShape]);

  const renderMeasurementLine = (p1: THREE.Vector3, p2: THREE.Vector3) => {
    if (!measureMarkersGroupRef.current) return;
    const grp = measureMarkersGroupRef.current;
    while (grp.children.length > 0) {
      const c = grp.children[0];
      grp.remove(c);
      if ((c as THREE.Mesh).geometry) (c as THREE.Mesh).geometry.dispose();
    }
    const lineGeo = new THREE.BufferGeometry().setFromPoints([p1, p2]);
    const lineMat = new THREE.LineBasicMaterial({ color: 0x10b981, linewidth: 3 });
    const lineMesh = new THREE.Line(lineGeo, lineMat);
    grp.add(lineMesh);
    measureLineRef.current = lineMesh;
    const mGeo = new THREE.SphereGeometry(1.2, 16, 16);
    const mMat = new THREE.MeshBasicMaterial({ color: 0x10b981 });
    const m1 = new THREE.Mesh(mGeo, mMat);
    m1.position.copy(p1);
    const m2 = new THREE.Mesh(mGeo, mMat);
    m2.position.copy(p2);
    grp.add(m1);
    grp.add(m2);
  };

  const clearMeasurement = () => {
    setMeasurePoints([]);
    setMeasureDistanceM(null);
    if (measureMarkersGroupRef.current) {
      const grp = measureMarkersGroupRef.current;
      while (grp.children.length > 0) {
        const c = grp.children[0];
        grp.remove(c);
        if ((c as THREE.Mesh).geometry) (c as THREE.Mesh).geometry.dispose();
      }
    }
  };

  const focusBuilding = useCallback((bldg: BuildingMeasurement) => {
    const [bx, by, bw, bh] = bldg.bbox;
    const refH = depthShape?.[0] ?? 1024;
    const refW = depthShape?.[1] ?? 1024;
    const cx = (bx + bw / 2) / refW;
    const cy = (by + bh / 2) / refH;
    const x3d = (cx - 0.5) * 100;
    const z3d = (cy - 0.5) * 100;
    const roofZ = bldg.rooftop_peak_z_rel * 28 * activeExaggeration;
    desiredCameraTarget.current.set(x3d, roofZ * 0.5, z3d);
    desiredCameraAngle.current = { theta: Math.PI / 4, phi: Math.PI / 3.5, radius: 45 };
    setCameraPresetState("focus");
    updateCameraPosition();
  }, [depthShape, activeExaggeration, updateCameraPosition]);

  const getEdgePreservedHeights = useCallback((meshData: MeshHeightfieldData): Float32Array => {
    const gw = meshData.grid_width;
    const gh = meshData.grid_height;
    const raw = meshData.heights;
    const out = new Float32Array(raw.length);
    const thresh = 0.06;
    for (let r = 0; r < gh; r++) {
      for (let c = 0; c < gw; c++) {
        const idx = r * gw + c;
        const val = raw[idx] || 0;
        let maxNeighbor = val, minNeighbor = val, edgeCount = 0;
        for (let dr = -1; dr <= 1; dr++) {
          for (let dc = -1; dc <= 1; dc++) {
            if (dr === 0 && dc === 0) continue;
            const nr = Math.min(gh - 1, Math.max(0, r + dr));
            const nc = Math.min(gw - 1, Math.max(0, c + dc));
            const nval = raw[nr * gw + nc] || 0;
            if (Math.abs(nval - val) > thresh) edgeCount++;
            maxNeighbor = Math.max(maxNeighbor, nval);
            minNeighbor = Math.min(minNeighbor, nval);
          }
        }
        out[idx] = edgeCount >= 3 ? (val > (maxNeighbor + minNeighbor) / 2 ? maxNeighbor : minNeighbor) : val;
      }
    }
    return out;
  }, []);

  const getTurboColor = (x: number) => {
    const r = 0.1357 + x * (4.61539 - x * (42.6603 - x * (132.131 - x * (161.073 - x * 65.402))));
    const g = 0.0914 + x * (2.19418 + x * (16.4218 - x * (57.4583 - x * (71.3094 - x * 31.764))));
    const b = 0.1067 + x * (12.5925 - x * (60.1097 - x * (109.0745 - x * (88.5061 - x * 26.818))));
    return { r: Math.min(1, Math.max(0, r)), g: Math.min(1, Math.max(0, g)), b: Math.min(1, Math.max(0, b)) };
  };

  const handleCameraPresetChange = (preset: CameraPreset) => {
    setCameraPresetState(preset);
    if (preset === "isometric") {
      desiredCameraTarget.current.set(0, 0, 0);
      desiredCameraAngle.current = { theta: Math.PI / 4, phi: Math.PI / 3.2, radius: 135 };
    } else if (preset === "top") {
      desiredCameraTarget.current.set(0, 0, 0);
      desiredCameraAngle.current = { theta: 0, phi: 0.05, radius: 155 };
    } else if (preset === "orbit") {
      desiredCameraTarget.current.set(0, 0, 0);
      desiredCameraAngle.current = { theta: Math.PI / 4, phi: Math.PI / 3.4, radius: 140 };
    } else if (preset === "street") {
      desiredCameraTarget.current.set(0, 3, 0);
      desiredCameraAngle.current = { theta: Math.PI / 4, phi: Math.PI / 2.2, radius: 75 };
    } else if (preset === "focus" && selectedBuilding) {
      focusBuilding(selectedBuilding);
    }
    updateCameraPosition();
  };

  useEffect(() => {
    const sceneRoot = sceneRootRef.current;
    if (!sceneRoot || !mesh || mesh.heights.length === 0) return;

    while (sceneRoot.children.length > 0) {
      const child = sceneRoot.children[0];
      sceneRoot.remove(child);
      if ((child as THREE.Mesh).geometry) (child as THREE.Mesh).geometry.dispose();
      if ((child as THREE.Mesh).material) {
        const mat = (child as THREE.Mesh).material;
        if (Array.isArray(mat)) mat.forEach((m) => m.dispose());
        else mat.dispose();
      }
    }

    const gw = mesh.grid_width;
    const gh = mesh.grid_height;
    const planeSize = 100.0;
    const refH = depthShape?.[0] ?? 1024;
    const refW = depthShape?.[1] ?? 1024;
    const heightsFiltered = getEdgePreservedHeights(mesh);

    const isBuildingGrid = new Uint8Array(gw * gh);
    const terrainHeights = new Float32Array(heightsFiltered);

    sampleBuildings.forEach((bldg) => {
      const [bx, by, bw, bh] = bldg.bbox;
      const cMin = Math.max(0, Math.floor((bx / refW) * gw));
      const cMax = Math.min(gw - 1, Math.ceil(((bx + bw) / refW) * gw));
      const rMin = Math.max(0, Math.floor((by / refH) * gh));
      const rMax = Math.min(gh - 1, Math.ceil(((by + bh) / refH) * gh));
      const groundH = bldg.ground_base_z_rel ?? 0.05;

      for (let r = rMin; r <= rMax; r++) {
        for (let c = cMin; c <= cMax; c++) {
          const idx = r * gw + c;
          isBuildingGrid[idx] = 1;
          terrainHeights[idx] = groundH;
        }
      }
    });

    const surroundingBuildings: Array<{ centerU: number; centerV: number; widthU: number; heightV: number; groundRel: number; roofRel: number; avgColor: THREE.Color }> = [];
    const clusterSize = 3;
    for (let r = clusterSize; r < gh - clusterSize; r += clusterSize) {
      for (let c = clusterSize; c < gw - clusterSize; c += clusterSize) {
        const idx = r * gw + c;
        if (isBuildingGrid[idx]) continue;
        const h = heightsFiltered[idx] || 0;
        const red = mesh.colors[idx * 3] ?? 0.5;
        const green = mesh.colors[idx * 3 + 1] ?? 0.5;
        const blue = mesh.colors[idx * 3 + 2] ?? 0.5;
        const isGreen = green > red * 1.1 && green > blue * 1.05;
        if (!isGreen && h > 0.28) {
          for (let dr = -1; dr <= 1; dr++) {
            for (let dc = -1; dc <= 1; dc++) {
              const nidx = (r + dr) * gw + (c + dc);
              isBuildingGrid[nidx] = 1;
              terrainHeights[nidx] = 0.05;
            }
          }
          surroundingBuildings.push({ centerU: (c + 0.5) / gw, centerV: (r + 0.5) / gh, widthU: 3.5 / gw, heightV: 3.5 / gh, groundRel: 0.05, roofRel: h, avgColor: new THREE.Color(red * 0.9, green * 0.9, blue * 0.9) });
        }
      }
    }

    for (let r = 1; r < gh - 1; r++) {
      for (let c = 1; c < gw - 1; c++) {
        const idx = r * gw + c;
        if (isBuildingGrid[idx]) continue;
        let sum = 0, count = 0;
        for (let dr = -1; dr <= 1; dr++) {
          for (let dc = -1; dc <= 1; dc++) {
            const nidx = (r + dr) * gw + (c + dc);
            if (!isBuildingGrid[nidx]) { sum += heightsFiltered[nidx] || 0; count++; }
          }
        }
        if (count > 0) terrainHeights[idx] = sum / count;
      }
    }

    const buildingsGroup = new THREE.Group();
    const facadeWallMat = new THREE.MeshStandardMaterial({ color: 0x2e3b4e, roughness: 0.35, metalness: 0.25 });
    const roofAerialMat = new THREE.MeshStandardMaterial({ map: loadedRgbTextureRef.current || null, color: loadedRgbTextureRef.current ? 0xffffff : 0x94a3b8, roughness: 0.45, metalness: 0.1 });
    const bldgMultiMaterial = [facadeWallMat, facadeWallMat, roofAerialMat, facadeWallMat, facadeWallMat, facadeWallMat];

    sampleBuildings.forEach((bldg) => {
      const [bx, by, bw, bh] = bldg.bbox;
      const cx = (bx + bw / 2) / refW;
      const cy = (by + bh / 2) / refH;
      const x3d = (cx - 0.5) * planeSize;
      const z3d = (cy - 0.5) * planeSize;
      const w3d = Math.max(2.8, (bw / refW) * planeSize);
      const d3d = Math.max(2.8, (bh / refH) * planeSize);
      const baseZ = Math.max(0.1, (bldg.ground_base_z_rel ?? 0.05) * 28 * activeExaggeration);
      const roofZ = Math.max(baseZ + 2.5, (bldg.rooftop_peak_z_rel ?? 0.4) * 28 * activeExaggeration);
      const wallHeight = roofZ - baseZ;
      const bldgGeo = new THREE.BoxGeometry(w3d * 0.98, wallHeight, d3d * 0.98);

      const uvAttr = bldgGeo.attributes.uv;
      if (uvAttr) {
        const uMin = bx / refW; const uMax = (bx + bw) / refW;
        const vMin = 1.0 - (by + bh) / refH; const vMax = 1.0 - by / refH;
        uvAttr.setXY(16, uMin, vMax); uvAttr.setXY(17, uMax, vMax);
        uvAttr.setXY(18, uMin, vMin); uvAttr.setXY(19, uMax, vMin);
        uvAttr.needsUpdate = true;
      }

      const bldgMesh = new THREE.Mesh(bldgGeo, bldgMultiMaterial);
      bldgMesh.position.set(x3d, baseZ + wallHeight / 2, z3d);
      bldgMesh.castShadow = true; bldgMesh.receiveShadow = true;
      bldgMesh.userData = { buildingId: bldg.id, buildingName: bldg.name };

      const edgesGeo = new THREE.EdgesGeometry(bldgGeo);
      const lineMat = new THREE.LineBasicMaterial({ color: bldg.id === selectedBuilding?.id ? 0x10b981 : 0x38bdf8, linewidth: 2 });
      const edgesMesh = new THREE.LineSegments(edgesGeo, lineMat);
      bldgMesh.add(edgesMesh);

      if (wallHeight > 6) {
        const roofDetailGeo = new THREE.BoxGeometry(w3d * 0.35, 1.4, d3d * 0.35);
        const roofDetailMat = new THREE.MeshStandardMaterial({ color: 0x64748b, roughness: 0.3, metalness: 0.3 });
        const roofDetailMesh = new THREE.Mesh(roofDetailGeo, roofDetailMat);
        roofDetailMesh.position.set(0, wallHeight / 2 + 0.7, 0);
        roofDetailMesh.castShadow = true;
        bldgMesh.add(roofDetailMesh);
      }
      buildingsGroup.add(bldgMesh);
    });

    surroundingBuildings.forEach((sb) => {
      const x3d = (sb.centerU - 0.5) * planeSize; const z3d = (sb.centerV - 0.5) * planeSize;
      const w3d = Math.max(2.2, sb.widthU * planeSize); const d3d = Math.max(2.2, sb.heightV * planeSize);
      const baseZ = Math.max(0.1, sb.groundRel * 28 * activeExaggeration);
      const roofZ = Math.max(baseZ + 2.0, sb.roofRel * 28 * activeExaggeration);
      const wallHeight = roofZ - baseZ;
      const sGeo = new THREE.BoxGeometry(w3d, wallHeight, d3d);
      const sMat = new THREE.MeshStandardMaterial({ color: sb.avgColor.getHex() || 0x334155, roughness: 0.45, metalness: 0.15 });
      const sMesh = new THREE.Mesh(sGeo, sMat);
      sMesh.position.set(x3d, baseZ + wallHeight / 2, z3d);
      sMesh.castShadow = true; sMesh.receiveShadow = true;
      const sEdgesGeo = new THREE.EdgesGeometry(sGeo);
      const sEdgesMesh = new THREE.LineSegments(sEdgesGeo, new THREE.LineBasicMaterial({ color: 0x475569 }));
      sMesh.add(sEdgesMesh);
      buildingsGroup.add(sMesh);
    });

    sceneRoot.add(buildingsGroup);
    buildingsGroupRef.current = buildingsGroup;

    const planeGeo = new THREE.PlaneGeometry(planeSize, (planeSize * gh) / gw, gw - 1, gh - 1);
    planeGeo.rotateX(-Math.PI / 2);
    const posAttr = planeGeo.attributes.position;
    const meshColors = new Float32Array(posAttr.count * 3);
    const heatmapColors = new Float32Array(posAttr.count * 3);

    for (let i = 0; i < posAttr.count; i++) {
      const hTerrain = terrainHeights[i] || 0;
      const hRaw = heightsFiltered[i] || 0;
      posAttr.setY(i, hTerrain * 28.0 * activeExaggeration);
      const r = mesh.colors[i * 3] ?? 0.5; const g = mesh.colors[i * 3 + 1] ?? 0.5; const b = mesh.colors[i * 3 + 2] ?? 0.5;
      meshColors[i * 3] = r; meshColors[i * 3 + 1] = g; meshColors[i * 3 + 2] = b;
      const hm = getTurboColor(Math.min(1.0, Math.max(0.0, hRaw)));
      heatmapColors[i * 3] = hm.r; heatmapColors[i * 3 + 1] = hm.g; heatmapColors[i * 3 + 2] = hm.b;
    }
    planeGeo.setAttribute("color", new THREE.BufferAttribute(meshColors, 3));
    planeGeo.computeVertexNormals();

    const meshMat = new THREE.MeshStandardMaterial({ map: loadedRgbTextureRef.current || null, vertexColors: !loadedRgbTextureRef.current, roughness: 0.55, metalness: 0.05, side: THREE.DoubleSide });
    const terrainMesh = new THREE.Mesh(planeGeo, meshMat);
    terrainMesh.receiveShadow = true; terrainMesh.castShadow = true;
    sceneRoot.add(terrainMesh);
    terrainMeshRef.current = terrainMesh;

    const rawPlaneGeo = planeGeo.clone();
    const rawPosAttr = rawPlaneGeo.attributes.position;
    for (let i = 0; i < rawPosAttr.count; i++) rawPosAttr.setY(i, (heightsFiltered[i] || 0) * 28.0 * activeExaggeration);
    rawPlaneGeo.computeVertexNormals();

    const depthMat = new THREE.MeshStandardMaterial({ map: loadedDepthTextureRef.current || null, vertexColors: !loadedDepthTextureRef.current, roughness: 0.5, side: THREE.DoubleSide });
    const depthMesh = new THREE.Mesh(rawPlaneGeo, depthMat);
    depthMesh.visible = false;
    sceneRoot.add(depthMesh);
    depthMeshRef.current = depthMesh;

    const wireMat = new THREE.MeshBasicMaterial({ color: 0x00e5ff, wireframe: true, transparent: true, opacity: 0.85, polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -1 });
    const wireMesh = new THREE.Mesh(rawPlaneGeo, wireMat);
    wireMesh.visible = false;
    sceneRoot.add(wireMesh);
    wireframeMeshRef.current = wireMesh;

    const hmGeo = rawPlaneGeo.clone();
    hmGeo.setAttribute("color", new THREE.BufferAttribute(heatmapColors, 3));
    const hmMat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.4, metalness: 0.2 });
    const hmMesh = new THREE.Mesh(hmGeo, hmMat);
    hmMesh.visible = false;
    sceneRoot.add(hmMesh);
    heatmapMeshRef.current = hmMesh;

    const treesGroup = new THREE.Group();
    const treePositions: THREE.Vector3[] = [];
    for (let r = 0; r < gh; r += 4) {
      for (let c = 0; c < gw; c += 4) {
        const idx = r * gw + c;
        if (isBuildingGrid[idx]) continue;
        const red = mesh.colors[idx * 3] ?? 0.5; const green = mesh.colors[idx * 3 + 1] ?? 0.5; const blue = mesh.colors[idx * 3 + 2] ?? 0.5;
        const rawH = terrainHeights[idx] || 0.0;
        if (green > red * 1.12 && green > blue * 1.08 && rawH < 0.35 && (r * c) % 7 === 0) {
          treePositions.push(new THREE.Vector3((c / gw - 0.5) * planeSize, rawH * 28.0 * activeExaggeration, (r / gh - 0.5) * planeSize));
        }
      }
    }

    if (treePositions.length > 0) {
      const treeTrunkGeo = new THREE.CylinderGeometry(0.3, 0.45, 2.4, 6); treeTrunkGeo.translate(0, 1.2, 0);
      const treeLeavesGeo = new THREE.ConeGeometry(1.6, 3.2, 6); treeLeavesGeo.translate(0, 3.6, 0);
      const instancedTrunks = new THREE.InstancedMesh(treeTrunkGeo, new THREE.MeshStandardMaterial({ color: 0x78350f, roughness: 0.9 }), treePositions.length);
      const instancedLeaves = new THREE.InstancedMesh(treeLeavesGeo, new THREE.MeshStandardMaterial({ color: 0x22c55e, roughness: 0.5 }), treePositions.length);
      const dummyMatrix = new THREE.Matrix4();
      treePositions.forEach((pos, i) => {
        dummyMatrix.setPosition(pos.x, pos.y, pos.z);
        instancedTrunks.setMatrixAt(i, dummyMatrix);
        instancedLeaves.setMatrixAt(i, dummyMatrix);
      });
      instancedTrunks.instanceMatrix.needsUpdate = true;
      instancedLeaves.instanceMatrix.needsUpdate = true;
      treesGroup.add(instancedTrunks);
      treesGroup.add(instancedLeaves);
    }
    sceneRoot.add(treesGroup);
    treesGroupRef.current = treesGroup;

    const voxelGroup = new THREE.Group();
    const vxCols = Math.min(56, gw); const vxRows = Math.min(56, gh);
    const blockWidth = planeSize / vxCols; const blockDepth = planeSize / vxRows;
    const instancedVoxels = new THREE.InstancedMesh(new THREE.BoxGeometry(blockWidth * 0.94, 1.0, blockDepth * 0.94).translate(0, 0.5, 0), new THREE.MeshStandardMaterial({ roughness: 0.55, metalness: 0.12 }), vxCols * vxRows);
    const dummyMatrix = new THREE.Matrix4(); const colorObj = new THREE.Color();
    let voxelIndex = 0;
    for (let r = 0; r < vxRows; r++) {
      for (let c = 0; c < vxCols; c++) {
        const srcIdx = Math.floor((r / vxRows) * gh) * gw + Math.floor((c / vxCols) * gw);
        const rawH = heightsFiltered[srcIdx] || 0.0;
        const red = mesh.colors[srcIdx * 3] ?? 0.5; const green = mesh.colors[srcIdx * 3 + 1] ?? 0.5; const blue = mesh.colors[srcIdx * 3 + 2] ?? 0.5;
        const quantizedH = Math.max(0.5, Math.round(rawH * 28) / 28) * 32.0 * activeExaggeration;
        if (green > red * 1.12 && green > blue * 1.1 && rawH < 0.35) colorObj.setRGB(0.18 + green * 0.4, 0.65 + green * 0.3, 0.18 + blue * 0.2);
        else if (red < 0.35 && green < 0.35 && blue < 0.35 && rawH < 0.18) colorObj.setHex(0x334155);
        else colorObj.setRGB(red * 0.95, green * 0.95, blue * 0.95);
        dummyMatrix.makeScale(1.0, quantizedH, 1.0);
        dummyMatrix.setPosition((c / vxCols - 0.5) * planeSize + blockWidth / 2, 0.0, (r / vxRows - 0.5) * planeSize + blockDepth / 2);
        instancedVoxels.setMatrixAt(voxelIndex, dummyMatrix);
        instancedVoxels.setColorAt(voxelIndex, colorObj);
        voxelIndex++;
      }
    }
    instancedVoxels.instanceMatrix.needsUpdate = true;
    if (instancedVoxels.instanceColor) instancedVoxels.instanceColor.needsUpdate = true;
    voxelGroup.add(instancedVoxels);
    voxelGroup.visible = false;
    sceneRoot.add(voxelGroup);
    voxelGroupRef.current = voxelGroup;

    const pGeo = new THREE.BufferGeometry();
    const pCount = mesh.heights.length;
    const posArray = new Float32Array(pCount * 3);
    const colArray = new Float32Array(pCount * 3);
    for (let i = 0; i < pCount; i++) {
      const r = Math.floor(i / gw); const c = i % gw; const h = heightsFiltered[i] || 0;
      posArray[i * 3] = (c / gw - 0.5) * planeSize;
      posArray[i * 3 + 1] = h * 28.0 * activeExaggeration;
      posArray[i * 3 + 2] = (r / gh - 0.5) * planeSize;
      colArray[i * 3] = mesh.colors[i * 3] ?? 0.5;
      colArray[i * 3 + 1] = mesh.colors[i * 3 + 1] ?? 0.5;
      colArray[i * 3 + 2] = mesh.colors[i * 3 + 2] ?? 0.5;
    }
    pGeo.setAttribute("position", new THREE.BufferAttribute(posArray, 3));
    pGeo.setAttribute("color", new THREE.BufferAttribute(colArray, 3));
    const pts = new THREE.Points(pGeo, new THREE.PointsMaterial({ size: 1.6, vertexColors: true, sizeAttenuation: true }));
    pts.visible = false;
    sceneRoot.add(pts);
    pointCloudPointsRef.current = pts;

    fitCameraToScene();
  }, [mesh, sampleBuildings, activeExaggeration, getEdgePreservedHeights, depthShape, fitCameraToScene]);

  useEffect(() => {
    if (terrainMeshRef.current) terrainMeshRef.current.visible = renderMode === "mesh";
    if (depthMeshRef.current) depthMeshRef.current.visible = renderMode === "depth";
    if (wireframeMeshRef.current) wireframeMeshRef.current.visible = renderMode === "wireframe";
    if (heatmapMeshRef.current) heatmapMeshRef.current.visible = renderMode === "heatmap";
    if (pointCloudPointsRef.current) pointCloudPointsRef.current.visible = renderMode === "pointcloud";
    if (voxelGroupRef.current) voxelGroupRef.current.visible = renderMode === "voxel";
    if (buildingsGroupRef.current) buildingsGroupRef.current.visible = renderMode === "mesh" || renderMode === "depth";
    if (treesGroupRef.current) treesGroupRef.current.visible = renderMode === "mesh";
  }, [renderMode]);

  useEffect(() => {
    if (!cursorMarkerRef.current) return;
    if (hoverCoordinate) {
      cursorMarkerRef.current.position.set((hoverCoordinate.u - 0.5) * 100.0, hoverCoordinate.relDepth * 28.0 * activeExaggeration, (hoverCoordinate.v - 0.5) * 100.0);
      cursorMarkerRef.current.visible = true;
    } else {
      cursorMarkerRef.current.visible = false;
    }
  }, [hoverCoordinate, activeExaggeration]);

  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) return;
    if (buildingBoxRef.current) {
      scene.remove(buildingBoxRef.current);
      buildingBoxRef.current.geometry.dispose();
      (buildingBoxRef.current.material as THREE.Material).dispose();
      buildingBoxRef.current = null;
    }
    if (selectedBuilding) {
      const [bx, by, bw, bh] = selectedBuilding.bbox;
      const refH = depthShape?.[0] ?? 1024; const refW = depthShape?.[1] ?? 1024;
      const cx = (bx + bw / 2) / refW; const cy = (by + bh / 2) / refH;
      const x3d = (cx - 0.5) * 100; const z3d = (cy - 0.5) * 100;
      const w3d = (bw / refW) * 100; const d3d = (bh / refH) * 100;
      const baseZ = selectedBuilding.ground_base_z_rel * 28 * activeExaggeration;
      const roofZ = selectedBuilding.rooftop_peak_z_rel * 28 * activeExaggeration;
      const h3d = Math.max(2.0, roofZ - baseZ);
      const boxGeo = new THREE.BoxGeometry(w3d, h3d, d3d);
      const edges = new THREE.EdgesGeometry(boxGeo);
      const lineMat = new THREE.LineBasicMaterial({ color: 0x10b981, linewidth: 3 });
      const boxLines = new THREE.LineSegments(edges, lineMat);
      boxLines.position.set(x3d, baseZ + h3d / 2, z3d);
      scene.add(boxLines);
      buildingBoxRef.current = boxLines;
    }
  }, [selectedBuilding, activeExaggeration, depthShape]);

  return (
    <div className="relative w-full h-[620px] rounded-2xl overflow-hidden border border-slate-800 bg-[#070A0F] flex flex-col select-none shadow-2xl">
      <div className="absolute top-3 left-3 z-20 flex items-center gap-2.5 p-2 px-3 bg-[#0F172A]/90 backdrop-blur-md rounded-xl border border-slate-700/80 shadow-xl pointer-events-auto">
        <div className="w-8 h-8 rounded-lg bg-cyan-500/20 border border-cyan-400/40 flex items-center justify-center text-cyan-400">
          <Sparkles className="w-5 h-5 text-cyan-300" />
        </div>
        <div>
          <h3 className="text-xs font-extrabold text-slate-100 tracking-wide uppercase font-mono-data flex items-center gap-1.5">
            DepthWizard 3D
            <Badge variant="outline" className="bg-cyan-500/20 text-cyan-300 border-cyan-500/40 text-[9px] px-1 py-0 font-mono-data">Digital Twin</Badge>
          </h3>
          <p className="text-[10px] text-slate-400 font-mono-data">Single-Image Geospatial Reconstruction</p>
        </div>

        <div className="flex items-center gap-1 bg-slate-900/90 p-1 rounded-lg border border-slate-800 ml-2">
          <Button size="xs" variant={scaleMode === "scientific" ? "default" : "ghost"} onClick={() => setScaleMode("scientific")} className={`h-6 text-[10px] ${scaleMode === "scientific" ? "bg-emerald-500 text-black font-bold" : "text-slate-400"}`}>
            <ShieldCheck className="w-3 h-3 mr-1" />Scientific (1.0x)
          </Button>
          <Button size="xs" variant={scaleMode === "exploration" ? "default" : "ghost"} onClick={() => setScaleMode("exploration")} className={`h-6 text-[10px] ${scaleMode === "exploration" ? "bg-cyan-500 text-black font-bold" : "text-slate-400"}`}>
            <Sliders className="w-3 h-3 mr-1" />Exploration
          </Button>
        </div>
      </div>

      <div className="absolute top-3 right-3 z-20 flex items-center gap-2 pointer-events-none">
        <div className="flex items-center gap-1 p-1 bg-[#0F172A]/90 backdrop-blur-md rounded-xl border border-slate-700 pointer-events-auto shadow-xl">
          <Button size="sm" variant={renderMode === "mesh" ? "default" : "ghost"} onClick={() => setRenderMode("mesh")} className={`h-7 px-2.5 text-xs font-mono-data ${renderMode === "mesh" ? "bg-cyan-500 text-black font-bold shadow-md" : "text-slate-300"}`}><ImageIcon className="w-3.5 h-3.5 mr-1" />REALISTIC</Button>
          <Button size="sm" variant={renderMode === "depth" ? "default" : "ghost"} onClick={() => setRenderMode("depth")} className={`h-7 px-2.5 text-xs font-mono-data ${renderMode === "depth" ? "bg-cyan-500 text-black font-bold" : "text-slate-300"}`}><Layers className="w-3.5 h-3.5 mr-1" />DEPTH</Button>
          <Button size="sm" variant={renderMode === "heatmap" ? "default" : "ghost"} onClick={() => setRenderMode("heatmap")} className={`h-7 px-2.5 text-xs font-mono-data ${renderMode === "heatmap" ? "bg-cyan-500 text-black font-semibold" : "text-slate-300"}`}><Flame className="w-3.5 h-3.5 mr-1" />HEIGHT</Button>
          <Button size="sm" variant={renderMode === "pointcloud" ? "default" : "ghost"} onClick={() => setRenderMode("pointcloud")} className={`h-7 px-2.5 text-xs font-mono-data ${renderMode === "pointcloud" ? "bg-cyan-500 text-black font-semibold" : "text-slate-300"}`}><Box className="w-3.5 h-3.5 mr-1" />POINT CLOUD</Button>
          <Button size="sm" variant={renderMode === "wireframe" ? "default" : "ghost"} onClick={() => setRenderMode("wireframe")} className={`h-7 px-2.5 text-xs font-mono-data ${renderMode === "wireframe" ? "bg-cyan-500 text-black font-semibold" : "text-slate-300"}`}><Grid3X3 className="w-3.5 h-3.5 mr-1" />WIREFRAME</Button>
          <div className="w-[1px] h-4 bg-slate-700 mx-0.5" />
          <Button size="sm" variant={renderMode === "voxel" ? "default" : "ghost"} onClick={() => setRenderMode("voxel")} className={`h-7 px-2 text-xs font-mono-data ${renderMode === "voxel" ? "bg-cyan-500 text-black font-bold" : "text-slate-400"}`}><Boxes className="w-3.5 h-3.5 mr-1" />Voxel</Button>
        </div>
        <div title={`Heading: ${headingDeg}°`} className="w-9 h-9 p-1 bg-[#0F172A]/90 backdrop-blur-md rounded-xl border border-slate-700/80 shadow-xl pointer-events-auto flex items-center justify-center cursor-pointer hover:border-cyan-400 text-cyan-400 relative">
          <div style={{ transform: `rotate(${-headingDeg}deg)` }} className="transition-transform duration-100 flex flex-col items-center justify-center text-[10px] font-bold font-mono-data">
            <span className="text-emerald-400 font-extrabold text-[9px] -mb-1">N</span>
            <Navigation className="w-4 h-4 fill-cyan-400/30 text-cyan-400" />
          </div>
        </div>
      </div>

      <div className="absolute top-14 right-3 z-20 flex items-center gap-1.5 p-1 bg-[#0F172A]/90 backdrop-blur-md rounded-xl border border-slate-700/80 shadow-xl pointer-events-auto text-xs font-mono-data">
        <span className="text-[10px] text-slate-400 px-1 font-bold">Preset:</span>
        {["isometric", "top", "orbit", "street"].map(p => (
            <Button key={p} size="xs" variant={cameraPreset === p ? "default" : "ghost"} onClick={() => handleCameraPresetChange(p as CameraPreset)} className={`h-6 text-[10px] px-2 ${cameraPreset === p ? "bg-cyan-500 text-black font-bold" : "text-slate-300"}`}>{p.toUpperCase()}</Button>
        ))}
        {selectedBuilding && <Button size="xs" variant={cameraPreset === "focus" ? "default" : "ghost"} onClick={() => handleCameraPresetChange("focus")} className={`h-6 text-[10px] px-2 ${cameraPreset === "focus" ? "bg-emerald-500 text-black font-bold" : "text-emerald-400"}`}>FOCUS</Button>}
        <div className="w-[1px] h-4 bg-slate-800 mx-0.5" />
        <Button size="xs" variant="outline" onClick={fitCameraToScene} className="h-6 text-[10px] px-2 border-cyan-500/40 text-cyan-300 hover:bg-cyan-500/20 font-bold"><RotateCcw className="w-3 h-3 mr-1" />FIT VIEW [R]</Button>
        <div className="w-[1px] h-4 bg-slate-800 mx-0.5" />
        <Button size="xs" variant={isMeasurementMode ? "default" : "ghost"} onClick={() => { setIsMeasurementMode(!isMeasurementMode); if (isMeasurementMode) clearMeasurement(); }} className={`h-6 text-[10px] px-2 ${isMeasurementMode ? "bg-emerald-500 text-black font-bold" : "text-slate-300"}`}><Ruler className="w-3 h-3 mr-1" />{isMeasurementMode ? "Measuring..." : "Measure"}</Button>
        <Button size="xs" variant={isFlyModeActive ? "default" : "ghost"} onClick={() => setIsFlyModeActive(!isFlyModeActive)} className={`h-6 text-[10px] px-2 ${isFlyModeActive ? "bg-indigo-500 text-white font-bold" : "text-slate-400"}`}><Crosshair className="w-3 h-3 mr-1" />Fly Mode</Button>
        <div className="relative">
          <Button size="xs" variant={showAnalysisMenu ? "default" : "ghost"} onClick={() => setShowAnalysisMenu(!showAnalysisMenu)} className="h-6 text-[10px] px-1.5 text-slate-300"><SlidersHorizontal className="w-3 h-3 mr-1" />Analysis<ChevronDown className="w-3 h-3 ml-0.5" /></Button>
          {showAnalysisMenu && (
            <div className="absolute right-0 top-7 z-30 p-2 bg-[#0F172A]/95 border border-slate-700 rounded-xl shadow-2xl w-44 space-y-1 text-[11px] font-mono-data text-slate-200">
              <div className="font-bold border-b border-slate-800 pb-1 text-cyan-300 text-[10px]">Analysis Helpers</div>
              {[{label: "Ground Grid", state: showGrid, setter: setShowGrid}, {label: "3D Spatial Axes", state: showAxes, setter: setShowAxes}, {label: "Scene Bounding Box", state: showBoundingBox, setter: setShowBoundingBox}, {label: "Performance Telemetry", state: showDebugHud, setter: setShowDebugHud}].map(item => (
                <button key={item.label} onClick={() => item.setter(!item.state)} className="w-full flex items-center justify-between p-1 hover:bg-slate-800 rounded text-left">
                  <span>{item.label}</span>
                  {item.state && <Check className="w-3.5 h-3.5 text-cyan-400" />}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {isMeasurementMode && (
        <div className="absolute top-24 left-3 z-20 p-2.5 px-3 bg-[#0F172A]/95 border border-emerald-500/50 rounded-xl shadow-2xl backdrop-blur-md font-mono-data text-xs space-y-1">
          <div className="flex items-center justify-between gap-2 text-emerald-400 font-bold border-b border-slate-800 pb-1">
            <span className="flex items-center gap-1"><Ruler className="w-3.5 h-3.5" />3D Metric Measurement</span>
            <button onClick={clearMeasurement} className="text-slate-400 hover:text-white text-xs">Clear</button>
          </div>
          <div className="text-[11px] text-slate-300">
            {measurePoints.length === 0 && <span>Click Point A on 3D surface...</span>}
            {measurePoints.length === 1 && <span>Click Point B to measure distance...</span>}
            {measurePoints.length === 2 && measureDistanceM !== null && <div className="text-emerald-300 font-extrabold text-sm">Distance: {measureDistanceM.toFixed(2)} m</div>}
          </div>
        </div>
      )}

      {hoverTooltip && !isMeasurementMode && (
        <div style={{ left: hoverTooltip.x + 15, top: hoverTooltip.y - 15 }} className="absolute z-30 pointer-events-none p-2 px-3 bg-[#0F172A]/95 border border-cyan-500/40 rounded-lg shadow-2xl backdrop-blur-md font-mono-data text-xs space-y-0.5">
          <div className="flex items-center gap-1.5 text-cyan-300 font-bold"><Building2 className="w-3.5 h-3.5" /><span>{hoverTooltip.info}</span></div>
          {hoverTooltip.surfaceType && <div className="text-[9px] text-slate-400 uppercase font-semibold">{hoverTooltip.surfaceType}</div>}
          {hoverTooltip.depthM && <div className="text-[10px] text-slate-300">Depth: <span className="text-cyan-400 font-semibold">{hoverTooltip.depthM}</span></div>}
          {hoverTooltip.height && <div className="text-[10px] text-slate-300">Height: <span className="text-emerald-400 font-semibold">{hoverTooltip.height}</span></div>}
        </div>
      )}

      {selectedBuilding && (
        <div className="absolute bottom-14 left-3 z-20 pointer-events-auto p-3.5 bg-[#0F172A]/95 backdrop-blur-md rounded-xl border border-emerald-500/40 shadow-2xl w-64 space-y-2 font-mono-data text-xs animate-in fade-in slide-in-from-left-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-2">
            <div className="flex items-center gap-2 text-emerald-400 font-bold"><Building2 className="w-4 h-4" /><span className="truncate">{selectedBuilding.name}</span></div>
            <button onClick={() => onSelectBuilding?.(null)} className="text-slate-400 hover:text-white">✕</button>
          </div>
          <div className="grid grid-cols-2 gap-2 text-[11px]">
            <div className="p-2 bg-slate-900/90 rounded-lg border border-slate-800"><span className="text-slate-400 block text-[9px]">ESTIMATED HEIGHT</span><span className="text-emerald-300 font-bold text-sm">{selectedBuilding.calibrated_height_m ? `${selectedBuilding.calibrated_height_m} m` : `${selectedBuilding.relative_height_unitless.toFixed(2)} rel`}</span></div>
            <div className="p-2 bg-slate-900/90 rounded-lg border border-slate-800"><span className="text-slate-400 block text-[9px]">CONFIDENCE</span><span className="text-cyan-300 font-bold text-sm">{selectedBuilding.confidence_pct}%</span></div>
          </div>
          <div className="text-[10px] text-slate-300 space-y-1">
            <div className="flex justify-between"><span className="text-slate-400">Surface:</span><span className="text-slate-200 font-semibold">Building / Roof</span></div>
            <div className="flex justify-between"><span className="text-slate-400">Footprint Box:</span><span className="text-slate-200">{selectedBuilding.bbox[2]}×{selectedBuilding.bbox[3]} px</span></div>
          </div>
          <div className="pt-1 flex gap-1.5 border-t border-slate-800">
            <Button size="xs" variant="outline" onClick={() => focusBuilding(selectedBuilding)} className="w-full h-6 text-[10px] border-emerald-500/40 text-emerald-300 hover:bg-emerald-500/20">Focus</Button>
            <Button size="xs" variant="outline" onClick={() => setIsMeasurementMode(true)} className="w-full h-6 text-[10px] border-cyan-500/40 text-cyan-300 hover:bg-cyan-500/20">Measure</Button>
          </div>
        </div>
      )}

      {imageUrl && (
        <div className="absolute bottom-3 left-3 z-20 pointer-events-auto">
          {showAerialThumbnail ? (
            <div className="p-2 bg-[#0F172A]/95 backdrop-blur-md rounded-xl border border-slate-700/90 shadow-2xl space-y-1.5 w-44">
              <div className="flex items-center justify-between text-[10px] font-bold text-slate-200 font-mono-data"><span>2D Aerial Reference</span><button onClick={() => setShowAerialThumbnail(false)} className="text-slate-400 hover:text-white p-0.5"><Minimize2 className="w-3 h-3" /></button></div>
              <div className="w-full h-24 rounded-lg overflow-hidden border border-slate-800 bg-black relative"><img src={imageUrl} alt="Input Aerial" className="w-full h-full object-cover" /></div>
            </div>
          ) : (
            <Button size="sm" variant="outline" onClick={() => setShowAerialThumbnail(true)} className="h-7 px-2.5 bg-[#0F172A]/90 border-slate-700 text-xs font-mono-data text-slate-300 hover:text-white"><Maximize2 className="w-3 h-3 mr-1 text-cyan-400" />2D Reference</Button>
          )}
        </div>
      )}

      <div className="absolute bottom-3 right-3 z-20 p-2.5 px-3 bg-[#0F172A]/90 backdrop-blur-md rounded-xl border border-slate-700/80 shadow-xl pointer-events-auto text-xs text-slate-300 font-mono-data space-y-1.5">
        <div className="flex items-center gap-2 text-[10px] text-slate-300"><span className="text-slate-400 font-bold">Scale:</span><div className="flex items-center gap-1 font-mono-data"><span className="text-cyan-400 font-bold">0m</span><div className="w-16 h-[2px] bg-cyan-400 relative flex items-center justify-between"><div className="w-[1px] h-2 bg-cyan-400" /><div className="w-[1px] h-2 bg-cyan-400" /></div><span className="text-cyan-400 font-bold">20m</span></div><Badge variant="outline" className="bg-slate-900 border-slate-700 text-[9px] px-1 text-cyan-300">SCALE: {scaleMode === "scientific" ? "1.0×" : `${verticalExaggeration.toFixed(1)}×`}</Badge></div>
        {scaleMode === "exploration" && (
          <div className="pt-1 flex items-center justify-between gap-2 border-t border-slate-800"><span className="text-[10px] text-slate-400">Exaggeration:</span><input type="range" min="0.5" max="3.0" step="0.1" value={verticalExaggeration} onChange={(e) => setVerticalExaggeration(parseFloat(e.target.value))} className="w-20 h-1 bg-slate-700 rounded appearance-none cursor-pointer accent-cyan-400" /><span className="text-cyan-300 text-[10px] font-bold">{verticalExaggeration.toFixed(1)}x</span></div>
        )}
      </div>

      {showDebugHud && (
        <div className="absolute top-24 right-3 z-30 p-2.5 bg-[#0F172A]/95 backdrop-blur-md rounded-xl border border-cyan-500/40 shadow-2xl pointer-events-auto text-[10px] font-mono-data space-y-1 text-slate-300 w-48">
          <div className="flex justify-between font-bold border-b border-slate-800 pb-1 text-cyan-300"><span>Performance Telemetry</span><span className={debugStats.fps >= 50 ? "text-emerald-400" : "text-amber-400"}>{debugStats.fps} FPS</span></div>
          <div className="flex justify-between"><span>Draw Calls:</span><span className="text-slate-100 font-bold">{debugStats.drawCalls}</span></div>
          <div className="flex justify-between"><span>Triangles:</span><span className="text-slate-100">{debugStats.triangles.toLocaleString()}</span></div>
          <div className="flex justify-between"><span>Points:</span><span className="text-slate-100">{debugStats.points.toLocaleString()}</span></div>
        </div>
      )}

      <div ref={containerRef} data-testid="three-canvas-container" className="w-full h-full cursor-grab active:cursor-grabbing flex-1" />
    </div>
  );
};
