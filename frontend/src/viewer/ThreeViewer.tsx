import React, { useEffect, useRef, useState, useCallback } from "react";
import * as THREE from "three";
import type { PointCloudData, MeshHeightfieldData, BuildingMeasurement } from "@/types/depth";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { 
  RotateCcw, 
  Eye, 
  Compass, 
  Box,
  Flame,
  Grid3X3,
  Boxes,
  Maximize2,
  Minimize2,
  Image as ImageIcon,
  Sparkles,
  Building2,
  Camera,
  Layers,
  Activity,
  Sliders,
  ShieldCheck
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
export type PerformanceTier = "auto" | "low" | "medium" | "high";

export const ThreeViewer: React.FC<ThreeViewerProps> = ({
  pointcloud,
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
  
  // Controls & Display Options
  const [renderMode, setRenderMode] = useState<RenderMode>("mesh");
  const [scaleMode, setScaleMode] = useState<ScaleMode>("scientific");
  const [verticalExaggeration, setVerticalExaggeration] = useState<number>(1.0);
  const [isFlythroughActive, setIsFlythroughActive] = useState<boolean>(false);
  const [showAerialThumbnail, setShowAerialThumbnail] = useState<boolean>(true);
  const [showDebugHud, setShowDebugHud] = useState<boolean>(false);

  // Telemetry & Debug Stats
  const [debugStats, setDebugStats] = useState<{ fps: number; drawCalls: number; triangles: number; points: number }>({
    fps: 60,
    drawCalls: 0,
    triangles: 0,
    points: 0,
  });

  // Hover & Raycast State
  const [hoveredBuilding, setHoveredBuilding] = useState<BuildingMeasurement | null>(null);
  const [hoverTooltip, setHoverTooltip] = useState<{ x: number; y: number; info: string; height?: string } | null>(null);

  // Three.js Core Instances (using refs for zero React re-render overhead)
  const sceneRef = useRef<THREE.Scene | null>(null);
  const sceneRootRef = useRef<THREE.Group | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);

  // Reusable Object References inside sceneRoot
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

  // Cached Materials & Textures
  const loadedRgbTextureRef = useRef<THREE.Texture | null>(null);
  const loadedDepthTextureRef = useRef<THREE.Texture | null>(null);

  // Camera Motion Lerp State
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

  // Flythrough & Animation refs
  const flythroughCurveRef = useRef<THREE.CatmullRomCurve3 | null>(null);
  const flythroughT = useRef<number>(0);
  const animFrameId = useRef<number | null>(null);

  // Interaction state refs
  const isMouseDown = useRef<boolean>(false);
  const mouseButton = useRef<number>(0);
  const mousePrev = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
  const keysPressed = useRef<{ [key: string]: boolean }>({});
  const lastRaycastTime = useRef<number>(0);
  const frameCountRef = useRef<number>(0);
  const lastFpsCalcTimeRef = useRef<number>(performance.now());

  // Active Effective Scale Factor
  const activeExaggeration = scaleMode === "scientific" ? 1.0 : verticalExaggeration;

  // -----------------------------------------------------------------
  // AUTO CAMERA FRAMING FUNCTION
  // -----------------------------------------------------------------
  const fitCameraToScene = useCallback(() => {
    if (!sceneRootRef.current || !cameraRef.current) return;

    const box = new THREE.Box3().setFromObject(sceneRootRef.current);
    if (box.isEmpty()) return;

    const sphere = new THREE.Sphere();
    box.getBoundingSphere(sphere);

    desiredCameraTarget.current.copy(sphere.center);
    desiredCameraAngle.current.radius = Math.max(30, Math.min(300, sphere.radius * 2.1));
    desiredCameraAngle.current.theta = Math.PI / 3.8;
    desiredCameraAngle.current.phi = Math.PI / 3.4;

    cameraRef.current.near = Math.max(0.1, sphere.radius * 0.01);
    cameraRef.current.far = Math.max(1000, sphere.radius * 12);
    cameraRef.current.updateProjectionMatrix();
  }, []);

  // Update Camera Position smoothly
  const updateCameraPosition = useCallback(() => {
    if (!cameraRef.current || isFlythroughActive) return;

    const dt = 0.15; // smooth damping
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
  }, [isFlythroughActive]);

  // Pre-load RGB Aerial Texture
  useEffect(() => {
    if (!imageUrl) return;
    const loader = new THREE.TextureLoader();
    loader.crossOrigin = "anonymous";

    loader.load(
      imageUrl,
      (tex) => {
        tex.colorSpace = THREE.SRGBColorSpace;
        tex.generateMipmaps = true;
        tex.minFilter = THREE.LinearMipmapLinearFilter;
        tex.magFilter = THREE.LinearFilter;
        tex.wrapS = THREE.ClampToEdgeWrapping;
        tex.wrapT = THREE.ClampToEdgeWrapping;

        if (rendererRef.current) {
          tex.anisotropy = Math.min(8, rendererRef.current.capabilities.getMaxAnisotropy());
        }
        loadedRgbTextureRef.current = tex;

        if (terrainMeshRef.current) {
          const mat = terrainMeshRef.current.material as THREE.MeshStandardMaterial;
          mat.map = tex;
          mat.vertexColors = false;
          mat.needsUpdate = true;
        }
      },
      undefined,
      (err) => console.warn("RGB texture load warning:", err)
    );
  }, [imageUrl]);

  // Pre-load Depth Colormap Texture
  useEffect(() => {
    if (!depthColormapUrl) return;
    const loader = new THREE.TextureLoader();
    loader.crossOrigin = "anonymous";

    loader.load(
      depthColormapUrl,
      (tex) => {
        tex.colorSpace = THREE.SRGBColorSpace;
        tex.generateMipmaps = true;
        tex.minFilter = THREE.LinearMipmapLinearFilter;
        tex.magFilter = THREE.LinearFilter;
        loadedDepthTextureRef.current = tex;

        if (depthMeshRef.current) {
          const mat = depthMeshRef.current.material as THREE.MeshStandardMaterial;
          mat.map = tex;
          mat.vertexColors = false;
          mat.needsUpdate = true;
        }
      },
      undefined,
      (err) => console.warn("Depth texture load warning:", err)
    );
  }, [depthColormapUrl]);

  // Initialize Core WebGL Scene & Persistent Animation Loop
  useEffect(() => {
    if (!containerRef.current) return;
    const container = containerRef.current;
    const width = container.clientWidth || 800;
    const height = container.clientHeight || 580;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x070a0f);
    scene.fog = new THREE.FogExp2(0x070a0f, 0.0016);
    sceneRef.current = scene;

    // Canonical Reconstruction Root
    const sceneRoot = new THREE.Group();
    scene.add(sceneRoot);
    sceneRootRef.current = sceneRoot;

    // Perspective Camera
    const camera = new THREE.PerspectiveCamera(42, width / height, 0.5, 2000);
    cameraRef.current = camera;

    // WebGL Renderer
    const renderer = new THREE.WebGLRenderer({ 
      antialias: true, 
      alpha: true, 
      powerPreference: "high-performance",
      preserveDrawingBuffer: true
    });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.75));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.2;

    container.innerHTML = "";
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Lighting (Sunlight + Sky Fill + Ground Bounce)
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.9);
    scene.add(ambientLight);

    const sunLight = new THREE.DirectionalLight(0xfff7ed, 1.6);
    sunLight.position.set(90, 150, 80);
    sunLight.castShadow = true;
    sunLight.shadow.mapSize.width = 1024;
    sunLight.shadow.mapSize.height = 1024;
    sunLight.shadow.bias = -0.0003;
    scene.add(sunLight);

    const skyFill = new THREE.DirectionalLight(0x38bdf8, 0.4);
    skyFill.position.set(-90, 70, -90);
    scene.add(skyFill);

    // Floor Base Tray & Grid Helper
    const trayGeo = new THREE.BoxGeometry(112, 4, 112);
    const trayMat = new THREE.MeshStandardMaterial({ color: 0x1e293b, roughness: 0.85, metalness: 0.15 });
    const trayMesh = new THREE.Mesh(trayGeo, trayMat);
    trayMesh.position.y = -2.1;
    trayMesh.receiveShadow = true;
    scene.add(trayMesh);

    const gridHelper = new THREE.GridHelper(112, 36, 0x00e5ff, 0x334155);
    gridHelper.position.y = -0.05;
    scene.add(gridHelper);

    // Cursor Hover Pin
    const markerGroup = new THREE.Group();
    const pinGeo = new THREE.CylinderGeometry(0.2, 0.8, 12, 12);
    const pinMat = new THREE.MeshBasicMaterial({ color: 0x00e5ff });
    const pinMesh = new THREE.Mesh(pinGeo, pinMat);
    pinMesh.position.y = 6;
    markerGroup.add(pinMesh);

    const ringGeo = new THREE.RingGeometry(1.5, 2.5, 32);
    const ringMat = new THREE.MeshBasicMaterial({ color: 0x00e5ff, side: THREE.DoubleSide });
    const ringMesh = new THREE.Mesh(ringGeo, ringMat);
    ringMesh.rotation.x = -Math.PI / 2;
    ringMesh.position.y = 0.2;
    markerGroup.add(ringMesh);

    markerGroup.visible = false;
    scene.add(markerGroup);
    cursorMarkerRef.current = markerGroup;

    // Flythrough Spline Path
    const splinePoints = [
      new THREE.Vector3(110, 90, 110),
      new THREE.Vector3(40, 45, 90),
      new THREE.Vector3(-60, 35, 40),
      new THREE.Vector3(-80, 55, -60),
      new THREE.Vector3(20, 65, -90),
      new THREE.Vector3(90, 75, -20),
      new THREE.Vector3(110, 90, 110),
    ];
    flythroughCurveRef.current = new THREE.CatmullRomCurve3(splinePoints, true);

    // Keyboard handlers
    const handleKeyDown = (e: KeyboardEvent) => {
      keysPressed.current[e.key.toLowerCase()] = true;
      if (e.key.toLowerCase() === "r") resetCamera();
      else if (e.key === "Escape") onSelectBuilding?.(null);
    };

    const handleKeyUp = (e: KeyboardEvent) => {
      keysPressed.current[e.key.toLowerCase()] = false;
    };

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);

    // SINGLE PERSISTENT RENDER LOOP
    const animate = () => {
      animFrameId.current = requestAnimationFrame(animate);

      // WASD Flight Navigation
      const moveSpeed = keysPressed.current["shift"] ? 1.8 : 0.8;
      const { theta } = cameraAngle.current;
      const right = new THREE.Vector3(1, 0, 0).applyAxisAngle(new THREE.Vector3(0, 1, 0), theta);
      const forward = new THREE.Vector3(0, 0, -1).applyAxisAngle(new THREE.Vector3(0, 1, 0), theta);

      if (keysPressed.current["w"]) { desiredCameraTarget.current.addScaledVector(forward, moveSpeed); }
      if (keysPressed.current["s"]) { desiredCameraTarget.current.addScaledVector(forward, -moveSpeed); }
      if (keysPressed.current["a"]) { desiredCameraTarget.current.addScaledVector(right, -moveSpeed); }
      if (keysPressed.current["d"]) { desiredCameraTarget.current.addScaledVector(right, moveSpeed); }
      if (keysPressed.current[" "]) { desiredCameraTarget.current.y += moveSpeed; }
      if (keysPressed.current["control"]) { desiredCameraTarget.current.y = Math.max(-5, desiredCameraTarget.current.y - moveSpeed); }

      updateCameraPosition();

      if (isFlythroughActive && flythroughCurveRef.current && cameraRef.current) {
        flythroughT.current = (flythroughT.current + 0.0015) % 1.0;
        const pos = flythroughCurveRef.current.getPointAt(flythroughT.current);
        cameraRef.current.position.copy(pos);
        cameraRef.current.lookAt(0, 5, 0);
      }

      if (rendererRef.current && sceneRef.current && cameraRef.current) {
        rendererRef.current.render(sceneRef.current, cameraRef.current);

        // Debug FPS calculation
        frameCountRef.current++;
        const now = performance.now();
        if (now - lastFpsCalcTimeRef.current >= 1000) {
          const fps = Math.round((frameCountRef.current * 1000) / (now - lastFpsCalcTimeRef.current));
          const info = rendererRef.current.info;
          setDebugStats({
            fps,
            drawCalls: info.render.calls,
            triangles: info.render.triangles,
            points: info.render.points,
          });
          frameCountRef.current = 0;
          lastFpsCalcTimeRef.current = now;
        }
      }
    };
    animate();

    const handleResize = () => {
      if (!container || !rendererRef.current || !cameraRef.current) return;
      const w = container.clientWidth;
      const h = container.clientHeight;
      cameraRef.current.aspect = w / h;
      cameraRef.current.updateProjectionMatrix();
      rendererRef.current.setSize(w, h);
    };
    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
      if (animFrameId.current) cancelAnimationFrame(animFrameId.current);
      renderer.dispose();
    };
  }, [updateCameraPosition, isFlythroughActive]);

  // Mouse Interaction (Orbit, Pan, Zoom, Double-click Focus, Raycast Hover)
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const onMouseDown = (e: MouseEvent) => {
      isMouseDown.current = true;
      mouseButton.current = e.button;
      mousePrev.current = { x: e.clientX, y: e.clientY };
    };

    const onMouseMove = (e: MouseEvent) => {
      if (!container || !cameraRef.current) return;

      if (isMouseDown.current && !isFlythroughActive) {
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
        // Throttled Raycasting against building meshes only
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
            const bId = intersects[0].object.userData?.buildingId;
            if (bId) {
              const matchedBldg = sampleBuildings.find((b) => b.id === bId);
              if (matchedBldg) {
                setHoveredBuilding(matchedBldg);
                setHoverTooltip({
                  x: e.clientX - rect.left,
                  y: e.clientY - rect.top,
                  info: matchedBldg.name,
                  height: matchedBldg.calibrated_height_m 
                    ? `${matchedBldg.calibrated_height_m}m` 
                    : `Peak: ${matchedBldg.rooftop_peak_z_rel.toFixed(2)} rel`
                });
                return;
              }
            }
          }
        }
        setHoveredBuilding(null);
        setHoverTooltip(null);
      }
    };

    const onMouseUp = (e: MouseEvent) => {
      isMouseDown.current = false;
      if (hoveredBuilding && e.button === 0) {
        onSelectBuilding?.(hoveredBuilding);
        focusBuilding(hoveredBuilding);
      }
    };

    const onDoubleClick = (e: MouseEvent) => {
      e.preventDefault();
      if (hoveredBuilding) {
        onSelectBuilding?.(hoveredBuilding);
        focusBuilding(hoveredBuilding);
      } else if (container && cameraRef.current) {
        const rect = container.getBoundingClientRect();
        const mouseX = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        const mouseY = -((e.clientY - rect.top) / rect.height) * 2 + 1;

        const raycaster = new THREE.Raycaster();
        raycaster.setFromCamera(new THREE.Vector2(mouseX, mouseY), cameraRef.current);

        if (terrainMeshRef.current) {
          const intersects = raycaster.intersectObject(terrainMeshRef.current);
          if (intersects.length > 0) {
            desiredCameraTarget.current.copy(intersects[0].point);
            desiredCameraAngle.current.radius = Math.max(30, cameraAngle.current.radius * 0.6);
            updateCameraPosition();
          }
        }
      }
    };

    const onContextMenu = (e: MouseEvent) => e.preventDefault();

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      if (isFlythroughActive) return;
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
  }, [updateCameraPosition, isFlythroughActive, hoveredBuilding, sampleBuildings, onSelectBuilding]);

  // Focus camera smoothly on selected building
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
    updateCameraPosition();
  }, [depthShape, activeExaggeration, updateCameraPosition]);

  // Edge-preserving Bilateral Filter
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

        let maxNeighbor = val;
        let minNeighbor = val;
        let edgeCount = 0;

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

  // -----------------------------------------------------------------
  // 3D GEOMETRY GENERATION (Runs ONCE when mesh/dataset changes)
  // -----------------------------------------------------------------
  useEffect(() => {
    const sceneRoot = sceneRootRef.current;
    if (!sceneRoot || !mesh || mesh.heights.length === 0) return;

    // Clear Previous Scene Children
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
    const heightsFiltered = getEdgePreservedHeights(mesh);

    // 1. TERRAIN BASE GEOMETRY
    const planeGeo = new THREE.PlaneGeometry(planeSize, (planeSize * gh) / gw, gw - 1, gh - 1);
    planeGeo.rotateX(-Math.PI / 2);

    const posAttr = planeGeo.attributes.position;
    const meshColors = new Float32Array(posAttr.count * 3);
    const heatmapColors = new Float32Array(posAttr.count * 3);

    for (let i = 0; i < posAttr.count; i++) {
      const h = heightsFiltered[i] || 0;
      posAttr.setY(i, h * 28.0 * activeExaggeration);

      const r = mesh.colors[i * 3] ?? 0.5;
      const g = mesh.colors[i * 3 + 1] ?? 0.5;
      const b = mesh.colors[i * 3 + 2] ?? 0.5;
      meshColors[i * 3] = r; meshColors[i * 3 + 1] = g; meshColors[i * 3 + 2] = b;

      const hm = getTurboColor(Math.min(1.0, Math.max(0.0, h)));
      heatmapColors[i * 3] = hm.r; heatmapColors[i * 3 + 1] = hm.g; heatmapColors[i * 3 + 2] = hm.b;
    }

    planeGeo.setAttribute("color", new THREE.BufferAttribute(meshColors, 3));
    planeGeo.computeVertexNormals();

    // 1A. PBR REALISTIC TERRAIN MESH
    const meshMat = new THREE.MeshStandardMaterial({
      map: loadedRgbTextureRef.current || null,
      vertexColors: !loadedRgbTextureRef.current,
      roughness: 0.45,
      metalness: 0.08,
      side: THREE.DoubleSide,
    });
    const terrainMesh = new THREE.Mesh(planeGeo, meshMat);
    terrainMesh.receiveShadow = true;
    terrainMesh.castShadow = true;
    sceneRoot.add(terrainMesh);
    terrainMeshRef.current = terrainMesh;

    // 1B. DEPTH COLORMAP MESH
    const depthMat = new THREE.MeshStandardMaterial({
      map: loadedDepthTextureRef.current || null,
      vertexColors: !loadedDepthTextureRef.current,
      roughness: 0.5,
      side: THREE.DoubleSide,
    });
    const depthMesh = new THREE.Mesh(planeGeo, depthMat);
    depthMesh.visible = false;
    sceneRoot.add(depthMesh);
    depthMeshRef.current = depthMesh;

    // 1C. WIREFRAME MESH (Shares planeGeo with polygon offset to prevent z-fighting)
    const wireMat = new THREE.MeshBasicMaterial({
      color: 0x00e5ff,
      wireframe: true,
      transparent: true,
      opacity: 0.85,
      polygonOffset: true,
      polygonOffsetFactor: -1,
      polygonOffsetUnits: -1,
    });
    const wireMesh = new THREE.Mesh(planeGeo, wireMat);
    wireMesh.visible = false;
    sceneRoot.add(wireMesh);
    wireframeMeshRef.current = wireMesh;

    // 1D. HEATMAP MESH
    const hmGeo = planeGeo.clone();
    hmGeo.setAttribute("color", new THREE.BufferAttribute(heatmapColors, 3));
    const hmMat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.4, metalness: 0.2 });
    const hmMesh = new THREE.Mesh(hmGeo, hmMat);
    hmMesh.visible = false;
    sceneRoot.add(hmMesh);
    heatmapMeshRef.current = hmMesh;

    // 2. EXTRUDED 3D BUILDING STRUCTURES
    const buildingsGroup = new THREE.Group();
    const refH = depthShape?.[0] ?? 1024;
    const refW = depthShape?.[1] ?? 1024;

    sampleBuildings.forEach((bldg) => {
      const [bx, by, bw, bh] = bldg.bbox;
      const cx = (bx + bw / 2) / refW;
      const cy = (by + bh / 2) / refH;

      const x3d = (cx - 0.5) * planeSize;
      const z3d = (cy - 0.5) * planeSize;
      const w3d = Math.max(2.5, (bw / refW) * planeSize);
      const d3d = Math.max(2.5, (bh / refH) * planeSize);

      const baseZ = Math.max(0.1, bldg.ground_base_z_rel * 28 * activeExaggeration);
      const roofZ = Math.max(baseZ + 2.0, bldg.rooftop_peak_z_rel * 28 * activeExaggeration);
      const wallHeight = roofZ - baseZ;

      const bldgGeo = new THREE.BoxGeometry(w3d * 0.98, wallHeight, d3d * 0.98);
      const bldgMat = new THREE.MeshStandardMaterial({
        map: loadedRgbTextureRef.current || null,
        color: loadedRgbTextureRef.current ? 0xffffff : 0x00e5ff,
        roughness: 0.35,
        metalness: 0.2,
      });

      const bldgMesh = new THREE.Mesh(bldgGeo, bldgMat);
      bldgMesh.position.set(x3d, baseZ + wallHeight / 2, z3d);
      bldgMesh.castShadow = true;
      bldgMesh.receiveShadow = true;
      bldgMesh.userData = { buildingId: bldg.id, buildingName: bldg.name };

      const edgesGeo = new THREE.EdgesGeometry(bldgGeo);
      const lineMat = new THREE.LineBasicMaterial({
        color: bldg.id === selectedBuilding?.id ? 0x10b981 : 0x38bdf8,
        linewidth: 2,
      });
      const edgesMesh = new THREE.LineSegments(edgesGeo, lineMat);
      bldgMesh.add(edgesMesh);

      if (wallHeight > 6) {
        const roofDetailGeo = new THREE.BoxGeometry(w3d * 0.35, 1.4, d3d * 0.35);
        const roofDetailMat = new THREE.MeshStandardMaterial({ color: 0x94a3b8, roughness: 0.3 });
        const roofDetailMesh = new THREE.Mesh(roofDetailGeo, roofDetailMat);
        roofDetailMesh.position.set(0, wallHeight / 2 + 0.7, 0);
        roofDetailMesh.castShadow = true;
        bldgMesh.add(roofDetailMesh);
      }

      buildingsGroup.add(bldgMesh);
    });
    sceneRoot.add(buildingsGroup);
    buildingsGroupRef.current = buildingsGroup;

    // 3. GPU INSTANCED VEGETATION TREES
    const treesGroup = new THREE.Group();
    const treePositions: THREE.Vector3[] = [];

    for (let r = 0; r < gh; r += 4) {
      for (let c = 0; c < gw; c += 4) {
        const idx = r * gw + c;
        const red = mesh.colors[idx * 3] ?? 0.5;
        const green = mesh.colors[idx * 3 + 1] ?? 0.5;
        const blue = mesh.colors[idx * 3 + 2] ?? 0.5;
        const rawH = heightsFiltered[idx] || 0.0;

        const isGreenVegetation = green > red * 1.12 && green > blue * 1.08 && rawH < 0.35;
        if (isGreenVegetation && (r * c) % 7 === 0) {
          const posX = (c / gw - 0.5) * planeSize;
          const posZ = (r / gh - 0.5) * planeSize;
          const posY = rawH * 28.0 * activeExaggeration;
          treePositions.push(new THREE.Vector3(posX, posY, posZ));
        }
      }
    }

    if (treePositions.length > 0) {
      const treeTrunkGeo = new THREE.CylinderGeometry(0.3, 0.45, 2.4, 6);
      treeTrunkGeo.translate(0, 1.2, 0);
      const treeLeavesGeo = new THREE.ConeGeometry(1.6, 3.2, 6);
      treeLeavesGeo.translate(0, 3.6, 0);

      const trunkMat = new THREE.MeshStandardMaterial({ color: 0x78350f, roughness: 0.9 });
      const leavesMat = new THREE.MeshStandardMaterial({ color: 0x22c55e, roughness: 0.5 });

      const instancedTrunks = new THREE.InstancedMesh(treeTrunkGeo, trunkMat, treePositions.length);
      const instancedLeaves = new THREE.InstancedMesh(treeLeavesGeo, leavesMat, treePositions.length);

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

    // 4. VOXEL WORLD (ROBLOX / MINECRAFT STYLE BLOCK CITY)
    const voxelGroup = new THREE.Group();
    const vxCols = Math.min(56, gw);
    const vxRows = Math.min(56, gh);
    const blockWidth = planeSize / vxCols;
    const blockDepth = planeSize / vxRows;
    const totalVoxels = vxCols * vxRows;

    const blockGeo = new THREE.BoxGeometry(blockWidth * 0.94, 1.0, blockDepth * 0.94);
    blockGeo.translate(0, 0.5, 0);
    const blockMat = new THREE.MeshStandardMaterial({ roughness: 0.55, metalness: 0.12 });
    const instancedVoxels = new THREE.InstancedMesh(blockGeo, blockMat, totalVoxels);

    const dummyMatrix = new THREE.Matrix4();
    const colorObj = new THREE.Color();
    let voxelIndex = 0;

    for (let r = 0; r < vxRows; r++) {
      for (let c = 0; c < vxCols; c++) {
        const srcX = Math.floor((c / vxCols) * gw);
        const srcY = Math.floor((r / vxRows) * gh);
        const srcIdx = srcY * gw + srcX;

        const rawH = heightsFiltered[srcIdx] || 0.0;
        const red = mesh.colors[srcIdx * 3] ?? 0.5;
        const green = mesh.colors[srcIdx * 3 + 1] ?? 0.5;
        const blue = mesh.colors[srcIdx * 3 + 2] ?? 0.5;

        const maxBlockHeight = 32.0 * activeExaggeration;
        const numSteps = 28;
        const quantizedH = Math.max(0.5, Math.round(rawH * numSteps) / numSteps) * maxBlockHeight;

        const posX = (c / vxCols - 0.5) * planeSize + blockWidth / 2;
        const posZ = (r / vxRows - 0.5) * planeSize + blockDepth / 2;

        const isGreenVegetation = green > red * 1.12 && green > blue * 1.1 && rawH < 0.35;
        const isDarkRoad = red < 0.35 && green < 0.35 && blue < 0.35 && rawH < 0.18;

        if (isGreenVegetation) colorObj.setRGB(0.18 + green * 0.4, 0.65 + green * 0.3, 0.18 + blue * 0.2);
        else if (isDarkRoad) colorObj.setHex(0x334155);
        else colorObj.setRGB(red * 0.95, green * 0.95, blue * 0.95);

        dummyMatrix.makeScale(1.0, quantizedH, 1.0);
        dummyMatrix.setPosition(posX, 0.0, posZ);
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

    // 5. POINT CLOUD (USES EXACT SAME WORLD TRANSFORM AS TERRAIN)
    const pGeo = new THREE.BufferGeometry();
    const pCount = mesh.heights.length;
    const posArray = new Float32Array(pCount * 3);
    const colArray = new Float32Array(pCount * 3);

    for (let i = 0; i < pCount; i++) {
      const r = Math.floor(i / gw);
      const c = i % gw;
      const h = heightsFiltered[i] || 0;

      posArray[i * 3] = (c / gw - 0.5) * planeSize;
      posArray[i * 3 + 1] = h * 28.0 * activeExaggeration;
      posArray[i * 3 + 2] = (r / gh - 0.5) * planeSize;

      colArray[i * 3] = mesh.colors[i * 3] ?? 0.5;
      colArray[i * 3 + 1] = mesh.colors[i * 3 + 1] ?? 0.5;
      colArray[i * 3 + 2] = mesh.colors[i * 3 + 2] ?? 0.5;
    }

    pGeo.setAttribute("position", new THREE.BufferAttribute(posArray, 3));
    pGeo.setAttribute("color", new THREE.BufferAttribute(colArray, 3));

    const pMat = new THREE.PointsMaterial({
      size: 1.6,
      vertexColors: true,
      sizeAttenuation: true,
    });
    const pts = new THREE.Points(pGeo, pMat);
    pts.visible = false;
    sceneRoot.add(pts);
    pointCloudPointsRef.current = pts;

    // AUTO CAMERA FRAMING
    fitCameraToScene();
  }, [mesh, sampleBuildings, activeExaggeration, getEdgePreservedHeights, depthShape, fitCameraToScene]);

  // -----------------------------------------------------------------
  // INSTANT ZERO-ALLOCATION MODE SWITCHING
  // -----------------------------------------------------------------
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

  // Synchronized Hover Cursor Pin
  useEffect(() => {
    if (!cursorMarkerRef.current) return;
    if (hoverCoordinate) {
      const x = (hoverCoordinate.u - 0.5) * 100.0;
      const z = (hoverCoordinate.v - 0.5) * 100.0;
      const y = hoverCoordinate.relDepth * 28.0 * activeExaggeration;
      cursorMarkerRef.current.position.set(x, y, z);
      cursorMarkerRef.current.visible = true;
    } else {
      cursorMarkerRef.current.visible = false;
    }
  }, [hoverCoordinate, activeExaggeration]);

  // Highlight Selected Building 3D Bounding Box
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
      const refH = depthShape?.[0] ?? 1024;
      const refW = depthShape?.[1] ?? 1024;
      const cx = (bx + bw / 2) / refW;
      const cy = (by + bh / 2) / refH;

      const x3d = (cx - 0.5) * 100;
      const z3d = (cy - 0.5) * 100;
      const w3d = (bw / refW) * 100;
      const d3d = (bh / refH) * 100;

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

  // Turbo Colormap helper
  function getTurboColor(x: number) {
    const r = 0.1357 + x * (4.61539 - x * (42.6603 - x * (132.131 - x * (161.073 - x * 65.402))));
    const g = 0.0914 + x * (2.19418 + x * (16.4218 - x * (57.4583 - x * (71.3094 - x * 31.764))));
    const b = 0.1067 + x * (12.5925 - x * (60.1097 - x * (109.0745 - x * (88.5061 - x * 26.818))));
    return { r: Math.min(1, Math.max(0, r)), g: Math.min(1, Math.max(0, g)), b: Math.min(1, Math.max(0, b)) };
  }

  // Camera Presets
  const resetCamera = () => {
    setIsFlythroughActive(false);
    fitCameraToScene();
  };

  const setIsometricView = () => {
    setIsFlythroughActive(false);
    desiredCameraTarget.current.set(0, 0, 0);
    desiredCameraAngle.current = { theta: Math.PI / 4, phi: Math.PI / 3.2, radius: 135 };
    updateCameraPosition();
  };

  const setTopView = () => {
    setIsFlythroughActive(false);
    desiredCameraTarget.current.set(0, 0, 0);
    desiredCameraAngle.current = { theta: 0, phi: 0.05, radius: 155 };
    updateCameraPosition();
  };

  const setStreetView = () => {
    setIsFlythroughActive(false);
    desiredCameraTarget.current.set(0, 3, 0);
    desiredCameraAngle.current = { theta: Math.PI / 4, phi: Math.PI / 2.2, radius: 75 };
    updateCameraPosition();
  };

  return (
    <div className="relative w-full h-[620px] rounded-2xl overflow-hidden border border-slate-800 bg-[#070A0F] flex flex-col select-none shadow-2xl">
      
      {/* ------------------------------------------------------------- */}
      {/* TOP-LEFT OVERLAY: Title & Scale Mode Selector */}
      {/* ------------------------------------------------------------- */}
      <div className="absolute top-3 left-3 z-20 flex items-center gap-2.5 p-2 px-3 bg-[#0F172A]/90 backdrop-blur-md rounded-xl border border-slate-700/80 shadow-xl pointer-events-auto">
        <div className="w-8 h-8 rounded-lg bg-cyan-500/20 border border-cyan-400/40 flex items-center justify-center text-cyan-400">
          <Sparkles className="w-5 h-5 animate-pulse text-cyan-300" />
        </div>
        <div>
          <h3 className="text-xs font-extrabold text-slate-100 tracking-wide uppercase font-mono-data flex items-center gap-1.5">
            DepthWizard 3D
            <Badge variant="outline" className="bg-cyan-500/20 text-cyan-300 border-cyan-500/40 text-[9px] px-1 py-0 font-mono-data">
              {scaleMode === "scientific" ? "Scientific 1.0x" : "Exploration Mode"}
            </Badge>
          </h3>
          <p className="text-[10px] text-slate-400 font-mono-data">Canonical WebGL 3D Reconstruction</p>
        </div>

        {/* Scale Mode Switcher */}
        <div className="flex items-center gap-1 bg-slate-900/90 p-1 rounded-lg border border-slate-800 ml-2">
          <Button
            size="xs"
            variant={scaleMode === "scientific" ? "default" : "ghost"}
            onClick={() => setScaleMode("scientific")}
            title="Strict 1.0x Metric Scale (Zero Distortion)"
            className={`h-6 text-[10px] font-mono-data px-2 ${scaleMode === "scientific" ? "bg-emerald-500 text-black font-bold" : "text-slate-400 hover:text-white"}`}
          >
            <ShieldCheck className="w-3 h-3 mr-1" />
            Scientific (1.0x)
          </Button>
          <Button
            size="xs"
            variant={scaleMode === "exploration" ? "default" : "ghost"}
            onClick={() => setScaleMode("exploration")}
            title="Enable Visual Elevation Exaggeration Slider"
            className={`h-6 text-[10px] font-mono-data px-2 ${scaleMode === "exploration" ? "bg-cyan-500 text-black font-bold" : "text-slate-400 hover:text-white"}`}
          >
            <Sliders className="w-3 h-3 mr-1" />
            Exploration
          </Button>
        </div>
      </div>

      {/* ------------------------------------------------------------- */}
      {/* TOP-RIGHT OVERLAY: Instant Mode Switcher & Presets */}
      {/* ------------------------------------------------------------- */}
      <div className="absolute top-3 right-3 z-20 flex items-center gap-2 pointer-events-none">
        <div className="flex items-center gap-1 p-1 bg-[#0F172A]/90 backdrop-blur-md rounded-xl border border-slate-700 pointer-events-auto shadow-xl">
          <Button
            size="sm"
            variant={renderMode === "mesh" ? "default" : "ghost"}
            onClick={() => setRenderMode("mesh")}
            data-testid="viewer-mode-mesh-button"
            className={`h-7 px-2.5 text-xs font-mono-data ${
              renderMode === "mesh" ? "bg-gradient-to-r from-cyan-500 to-sky-500 text-slate-950 font-bold shadow-md shadow-cyan-500/20" : "text-slate-300 hover:text-white"
            }`}
          >
            <ImageIcon className="w-3.5 h-3.5 mr-1" />
            Realistic PBR
          </Button>

          <Button
            size="sm"
            variant={renderMode === "depth" ? "default" : "ghost"}
            onClick={() => setRenderMode("depth")}
            data-testid="viewer-mode-depth-button"
            className={`h-7 px-2.5 text-xs font-mono-data ${
              renderMode === "depth" ? "bg-cyan-500 text-black font-bold" : "text-slate-300 hover:text-white"
            }`}
          >
            <Layers className="w-3.5 h-3.5 mr-1" />
            Depth Map
          </Button>

          <Button
            size="sm"
            variant={renderMode === "wireframe" ? "default" : "ghost"}
            onClick={() => setRenderMode("wireframe")}
            data-testid="viewer-mode-wireframe-button"
            className={`h-7 px-2.5 text-xs font-mono-data ${
              renderMode === "wireframe" ? "bg-cyan-500 text-black font-semibold" : "text-slate-300 hover:text-white"
            }`}
          >
            <Grid3X3 className="w-3.5 h-3.5 mr-1" />
            Wireframe
          </Button>

          <Button
            size="sm"
            variant={renderMode === "pointcloud" ? "default" : "ghost"}
            onClick={() => setRenderMode("pointcloud")}
            data-testid="viewer-mode-pointcloud-button"
            className={`h-7 px-2.5 text-xs font-mono-data ${
              renderMode === "pointcloud" ? "bg-cyan-500 text-black font-semibold" : "text-slate-300 hover:text-white"
            }`}
          >
            <Box className="w-3.5 h-3.5 mr-1" />
            Point Cloud
          </Button>

          <Button
            size="sm"
            variant={renderMode === "voxel" ? "default" : "ghost"}
            onClick={() => setRenderMode("voxel")}
            data-testid="viewer-mode-voxel-button"
            className={`h-7 px-2.5 text-xs font-mono-data ${
              renderMode === "voxel" ? "bg-cyan-500 text-black font-bold" : "text-slate-300 hover:text-white"
            }`}
          >
            <Boxes className="w-3.5 h-3.5 mr-1" />
            Voxel World
          </Button>

          <Button
            size="sm"
            variant={renderMode === "heatmap" ? "default" : "ghost"}
            onClick={() => setRenderMode("heatmap")}
            data-testid="viewer-mode-heatmap-button"
            className={`h-7 px-2.5 text-xs font-mono-data ${
              renderMode === "heatmap" ? "bg-cyan-500 text-black font-semibold" : "text-slate-300 hover:text-white"
            }`}
          >
            <Flame className="w-3.5 h-3.5 mr-1" />
            Height
          </Button>

          <div className="w-[1px] h-4 bg-slate-700 mx-0.5" />

          {/* Camera View Presets */}
          <Button
            size="icon-xs"
            variant="ghost"
            onClick={setIsometricView}
            title="Isometric 3D View"
            className="text-slate-300 hover:text-cyan-400 h-7 w-7"
          >
            <Compass className="w-3.5 h-3.5" />
          </Button>

          <Button
            size="icon-xs"
            variant="ghost"
            onClick={setTopView}
            title="Top-Down Ortho View"
            className="text-slate-300 hover:text-cyan-400 h-7 w-7"
          >
            <Eye className="w-3.5 h-3.5" />
          </Button>

          <Button
            size="icon-xs"
            variant="ghost"
            onClick={setStreetView}
            title="Street-Level View"
            className="text-slate-300 hover:text-cyan-400 h-7 w-7"
          >
            <Camera className="w-3.5 h-3.5" />
          </Button>
        </div>
      </div>

      {/* ------------------------------------------------------------- */}
      {/* RIGHT SIDEBAR: Controls & Performance Options */}
      {/* ------------------------------------------------------------- */}
      <div className="absolute top-16 right-3 z-20 flex flex-col gap-2.5 pointer-events-none w-48">
        
        {/* Controls Panel */}
        <div className="p-3 bg-[#0F172A]/90 backdrop-blur-md rounded-xl border border-slate-700/80 shadow-xl pointer-events-auto text-xs font-mono-data space-y-2">
          <div className="flex items-center justify-between text-slate-200 font-bold border-b border-slate-800 pb-1.5">
            <span>Spatial Controls</span>
            <Button
              size="icon-xs"
              variant="ghost"
              onClick={resetCamera}
              title="Fit Scene [R]"
              className="text-cyan-400 hover:bg-cyan-500/20 h-5 w-5"
            >
              <RotateCcw className="w-3 h-3" />
            </Button>
          </div>

          <div className="space-y-1 text-[10px] text-slate-300">
            <div className="flex items-center gap-1.5">
              <span className="font-bold text-cyan-400">WASD</span>
              <span>Fly Camera</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="font-bold text-emerald-400">Left Drag</span>
              <span>Orbit / Rotate</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="font-bold text-amber-400">Right Drag</span>
              <span>Pan Camera</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="font-bold text-indigo-400">Dbl Click</span>
              <span>Focus Object</span>
            </div>
            <div className="pt-1 flex gap-1">
              <Button
                size="sm"
                variant="outline"
                onClick={resetCamera}
                className="w-full h-6 text-[10px] bg-slate-800/80 border-slate-700 text-cyan-300 hover:bg-cyan-500/20"
              >
                [R] Fit View
              </Button>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => setShowDebugHud(!showDebugHud)}
                title="Toggle Performance Telemetry"
                className={`h-6 text-[10px] px-1.5 ${showDebugHud ? "bg-cyan-500/20 text-cyan-300" : "text-slate-400"}`}
              >
                <Activity className="w-3 h-3" />
              </Button>
            </div>
          </div>
        </div>

        {/* Debug HUD Overlay */}
        {showDebugHud && (
          <div className="p-2.5 bg-[#0F172A]/95 backdrop-blur-md rounded-xl border border-cyan-500/40 shadow-2xl pointer-events-auto text-[10px] font-mono-data space-y-1 text-slate-300 animate-in fade-in">
            <div className="flex justify-between font-bold border-b border-slate-800 pb-1 text-cyan-300">
              <span>Performance Telemetry</span>
              <span className={debugStats.fps >= 50 ? "text-emerald-400" : "text-amber-400"}>{debugStats.fps} FPS</span>
            </div>
            <div className="flex justify-between">
              <span>Draw Calls:</span>
              <span className="text-slate-100 font-bold">{debugStats.drawCalls}</span>
            </div>
            <div className="flex justify-between">
              <span>Triangles:</span>
              <span className="text-slate-100">{debugStats.triangles.toLocaleString()}</span>
            </div>
            <div className="flex justify-between">
              <span>Points:</span>
              <span className="text-slate-100">{debugStats.points.toLocaleString()}</span>
            </div>
          </div>
        )}

      </div>

      {/* ------------------------------------------------------------- */}
      {/* HOVER TOOLTIP CARD */}
      {/* ------------------------------------------------------------- */}
      {hoverTooltip && (
        <div
          style={{ left: hoverTooltip.x + 15, top: hoverTooltip.y - 15 }}
          className="absolute z-30 pointer-events-none p-2 px-3 bg-[#0F172A]/95 border border-cyan-500/40 rounded-lg shadow-2xl backdrop-blur-md font-mono-data text-xs space-y-0.5"
        >
          <div className="flex items-center gap-1.5 text-cyan-300 font-bold">
            <Building2 className="w-3.5 h-3.5" />
            <span>{hoverTooltip.info}</span>
          </div>
          <div className="text-[10px] text-slate-300">
            Height: <span className="text-emerald-400 font-semibold">{hoverTooltip.height}</span>
          </div>
        </div>
      )}

      {/* ------------------------------------------------------------- */}
      {/* FLOATING SELECTED BUILDING INFO CARD */}
      {/* ------------------------------------------------------------- */}
      {selectedBuilding && (
        <div className="absolute bottom-16 left-3 z-20 pointer-events-auto p-3.5 bg-[#0F172A]/95 backdrop-blur-md rounded-xl border border-emerald-500/40 shadow-2xl w-64 space-y-2 font-mono-data text-xs animate-in fade-in slide-in-from-left-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-2">
            <div className="flex items-center gap-2 text-emerald-400 font-bold">
              <Building2 className="w-4 h-4" />
              <span className="truncate">{selectedBuilding.name}</span>
            </div>
            <button onClick={() => onSelectBuilding?.(null)} className="text-slate-400 hover:text-white text-xs px-1">✕</button>
          </div>

          <div className="grid grid-cols-2 gap-2 text-[11px]">
            <div className="p-2 bg-slate-900/90 rounded-lg border border-slate-800">
              <span className="text-slate-400 block text-[9px]">ESTIMATED HEIGHT</span>
              <span className="text-emerald-300 font-bold text-sm">
                {selectedBuilding.calibrated_height_m 
                  ? `${selectedBuilding.calibrated_height_m} m` 
                  : `${selectedBuilding.relative_height_unitless.toFixed(2)} rel`}
              </span>
            </div>

            <div className="p-2 bg-slate-900/90 rounded-lg border border-slate-800">
              <span className="text-slate-400 block text-[9px]">CONFIDENCE</span>
              <span className="text-cyan-300 font-bold text-sm">
                {selectedBuilding.confidence_pct}%
              </span>
            </div>
          </div>

          <div className="text-[10px] text-slate-300 space-y-1">
            <div className="flex justify-between">
              <span className="text-slate-400">Surface Type:</span>
              <span className="text-slate-200 font-semibold">Building Roof Plate</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Footprint Box:</span>
              <span className="text-slate-200">{selectedBuilding.bbox[2]}×{selectedBuilding.bbox[3]} px</span>
            </div>
          </div>
        </div>
      )}

      {/* ------------------------------------------------------------- */}
      {/* BOTTOM-LEFT OVERLAY: Input Aerial Image Thumbnail */}
      {/* ------------------------------------------------------------- */}
      {imageUrl && (
        <div className="absolute bottom-3 left-3 z-20 pointer-events-auto">
          {showAerialThumbnail ? (
            <div className="p-2 bg-[#0F172A]/95 backdrop-blur-md rounded-xl border border-slate-700/90 shadow-2xl space-y-1.5 w-48">
              <div className="flex items-center justify-between text-[11px] font-bold text-slate-200 font-mono-data">
                <span>Input Aerial Image</span>
                <button onClick={() => setShowAerialThumbnail(false)} className="text-slate-400 hover:text-white p-0.5">
                  <Minimize2 className="w-3 h-3" />
                </button>
              </div>
              <div className="w-full h-28 rounded-lg overflow-hidden border border-slate-800 bg-black relative">
                <img src={imageUrl} alt="Input Aerial" className="w-full h-full object-cover" />
                {selectedBuilding && (
                  <div
                    style={{
                      left: `${(selectedBuilding.bbox[0] / (depthShape?.[1] || 1024)) * 100}%`,
                      top: `${(selectedBuilding.bbox[1] / (depthShape?.[0] || 1024)) * 100}%`,
                      width: `${(selectedBuilding.bbox[2] / (depthShape?.[1] || 1024)) * 100}%`,
                      height: `${(selectedBuilding.bbox[3] / (depthShape?.[0] || 1024)) * 100}%`,
                    }}
                    className="absolute border-2 border-emerald-400 bg-emerald-400/20 shadow-md animate-pulse pointer-events-none"
                  />
                )}
              </div>
            </div>
          ) : (
            <Button
              size="sm"
              variant="outline"
              onClick={() => setShowAerialThumbnail(true)}
              className="h-7 px-2.5 bg-[#0F172A]/90 border-slate-700 text-xs font-mono-data text-slate-300 hover:text-white"
            >
              <Maximize2 className="w-3 h-3 mr-1 text-cyan-400" />
              Show 2D Image
            </Button>
          )}
        </div>
      )}

      {/* ------------------------------------------------------------- */}
      {/* BOTTOM-RIGHT OVERLAY: Telemetry & Controls */}
      {/* ------------------------------------------------------------- */}
      <div className="absolute bottom-3 right-3 z-20 max-w-xs p-3 bg-[#0F172A]/90 backdrop-blur-md rounded-xl border border-slate-700/80 shadow-xl pointer-events-auto text-xs text-slate-300 font-mono-data space-y-1">
        <div className="flex items-center justify-between">
          <h4 className="font-bold text-slate-100 flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
            3D Spatial Engine
          </h4>
          <span className="text-[10px] text-cyan-400 font-bold uppercase">{scaleMode}</span>
        </div>

        {scaleMode === "exploration" && (
          <div className="pt-1 flex items-center justify-between gap-2 border-t border-slate-800">
            <span className="text-[10px] text-slate-400">Exaggeration:</span>
            <input
              type="range"
              min="0.5"
              max="3.0"
              step="0.1"
              value={verticalExaggeration}
              onChange={(e) => setVerticalExaggeration(parseFloat(e.target.value))}
              className="w-20 h-1 bg-slate-700 rounded appearance-none cursor-pointer accent-cyan-400"
            />
            <span className="text-cyan-300 text-[10px] font-bold">{verticalExaggeration.toFixed(1)}x</span>
          </div>
        )}
      </div>

      {/* WEBGL CANVAS */}
      <div
        ref={containerRef}
        data-testid="three-canvas-container"
        className="w-full h-full cursor-grab active:cursor-grabbing flex-1"
      />

    </div>
  );
};
