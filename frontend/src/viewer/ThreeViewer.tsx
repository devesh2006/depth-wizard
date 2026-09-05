import React, { useEffect, useRef, useState, useCallback } from "react";
import * as THREE from "three";
import type { PointCloudData, MeshHeightfieldData, BuildingMeasurement } from "@/types/depth";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { 
  Play, 
  Pause, 
  RotateCcw, 
  Eye, 
  Compass, 
  Box,
  Flame,
  Grid3X3,
  Boxes,
  Maximize2,
  Minimize2,
  MousePointer,
  Move,
  ZoomIn,
  Image as ImageIcon,
  Sparkles
} from "lucide-react";

interface ThreeViewerProps {
  pointcloud?: PointCloudData | null;
  mesh?: MeshHeightfieldData | null;
  imageUrl?: string;
  selectedBuilding?: BuildingMeasurement | null;
  hoverCoordinate?: { u: number; v: number; relDepth: number } | null;
  heightScale?: number;
  /** [height, width] of the depth grid: the coordinate space bboxes are measured in. */
  depthShape?: [number, number];
}

export type RenderMode = "mesh" | "voxel" | "pointcloud" | "wireframe" | "heatmap";

export const ThreeViewer: React.FC<ThreeViewerProps> = ({
  pointcloud,
  mesh,
  imageUrl,
  selectedBuilding,
  hoverCoordinate,
  depthShape,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [renderMode, setRenderMode] = useState<RenderMode>("mesh");
  const [isFlythroughActive, setIsFlythroughActive] = useState<boolean>(false);
  const [flythroughProgress, setFlythroughProgress] = useState<number>(0);
  const [verticalExaggeration, setVerticalExaggeration] = useState<number>(1.3);
  const [pointSize] = useState<number>(1.6);
  const [showAerialThumbnail, setShowAerialThumbnail] = useState<boolean>(true);
  const [maxSceneHeightMeters, setMaxSceneHeightMeters] = useState<number>(50);

  // Three.js instances refs
  const sceneRef = useRef<THREE.Scene | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const pointsObjectRef = useRef<THREE.Points | null>(null);
  const meshObjectRef = useRef<THREE.Mesh | null>(null);
  const wireframeObjectRef = useRef<THREE.Mesh | null>(null);
  const heatmapObjectRef = useRef<THREE.Mesh | null>(null);
  const voxelGroupRef = useRef<THREE.Group | null>(null);
  const cursorMarkerRef = useRef<THREE.Group | null>(null);
  const buildingBoxRef = useRef<THREE.LineSegments | null>(null);
  const flythroughCurveRef = useRef<THREE.CatmullRomCurve3 | null>(null);
  const loadedTextureRef = useRef<THREE.Texture | null>(null);
  const flythroughT = useRef<number>(0);
  const animFrameId = useRef<number | null>(null);

  // Interaction refs
  const isMouseDown = useRef<boolean>(false);
  const mouseButton = useRef<number>(0); // 0: left (rotate), 2: right (pan)
  const mousePrev = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
  const cameraTarget = useRef<THREE.Vector3>(new THREE.Vector3(0, 0, 0));
  const cameraAngle = useRef<{ theta: number; phi: number; radius: number }>({
    theta: Math.PI / 4,
    phi: Math.PI / 3.5,
    radius: 140,
  });

  // Update max scene height in meters when selected building or scene changes
  useEffect(() => {
    if (selectedBuilding && selectedBuilding.calibrated_height_m) {
      setMaxSceneHeightMeters(Math.max(40, Math.ceil(selectedBuilding.calibrated_height_m * 1.3)));
    } else {
      setMaxSceneHeightMeters(50);
    }
  }, [selectedBuilding]);

  // Update Camera Position & Target
  const updateCameraPosition = useCallback(() => {
    if (!cameraRef.current || isFlythroughActive) return;
    const { theta, phi, radius } = cameraAngle.current;
    const target = cameraTarget.current;

    const x = target.x + radius * Math.sin(phi) * Math.sin(theta);
    const y = target.y + radius * Math.cos(phi);
    const z = target.z + radius * Math.sin(phi) * Math.cos(theta);

    cameraRef.current.position.set(x, y, z);
    cameraRef.current.lookAt(target);
  }, [isFlythroughActive]);

  // Pre-load HD Aerial Texture from imageUrl
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
        loadedTextureRef.current = tex;

        // Apply texture to 3D mesh if already constructed
        if (meshObjectRef.current) {
          const mat = meshObjectRef.current.material as THREE.MeshStandardMaterial;
          mat.map = tex;
          mat.vertexColors = false;
          mat.needsUpdate = true;
        }
      },
      undefined,
      (err) => console.warn("Aerial texture load warning:", err)
    );
  }, [imageUrl]);

  // Initialize Three.js Scene
  useEffect(() => {
    if (!containerRef.current) return;
    const container = containerRef.current;
    const width = container.clientWidth || 700;
    const height = container.clientHeight || 540;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x070a0f);
    scene.fog = new THREE.FogExp2(0x070a0f, 0.0022);
    sceneRef.current = scene;

    // Perspective Camera
    const camera = new THREE.PerspectiveCamera(40, width / height, 0.5, 1200);
    cameraRef.current = camera;

    // WebGL Renderer with High Precision & ACES Filmic Tone Mapping
    const renderer = new THREE.WebGLRenderer({ 
      antialias: true, 
      alpha: true, 
      powerPreference: "high-performance",
      preserveDrawingBuffer: true
    });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.15;

    container.innerHTML = "";
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Realistic Lighting System (Sunlight + Ambient Occlusion Fill)
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.85);
    scene.add(ambientLight);

    const sunLight = new THREE.DirectionalLight(0xfff7ed, 1.5);
    sunLight.position.set(85, 135, 75);
    sunLight.castShadow = true;
    sunLight.shadow.mapSize.width = 2048;
    sunLight.shadow.mapSize.height = 2048;
    sunLight.shadow.bias = -0.0004;
    scene.add(sunLight);

    const skyFill = new THREE.DirectionalLight(0x38bdf8, 0.4);
    skyFill.position.set(-80, 60, -80);
    scene.add(skyFill);

    const groundBounce = new THREE.DirectionalLight(0xa3e635, 0.2);
    groundBounce.position.set(0, -40, 0);
    scene.add(groundBounce);

    // Floor Base Slab
    const trayGeo = new THREE.BoxGeometry(108, 4, 108);
    const trayMat = new THREE.MeshStandardMaterial({
      color: 0x1e293b,
      roughness: 0.85,
      metalness: 0.15,
    });
    const trayMesh = new THREE.Mesh(trayGeo, trayMat);
    trayMesh.position.y = -2.1;
    trayMesh.receiveShadow = true;
    scene.add(trayMesh);

    // Subtle Grid Overlay
    const gridHelper = new THREE.GridHelper(108, 36, 0x00e5ff, 0x334155);
    gridHelper.position.y = -0.05;
    scene.add(gridHelper);

    // Synchronized Cursor Pin
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

    // Flythrough Path
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

    // Camera initial angle
    cameraAngle.current = { theta: Math.PI / 3.8, phi: Math.PI / 3.4, radius: 145 };
    updateCameraPosition();

    // Render Loop
    const animate = () => {
      animFrameId.current = requestAnimationFrame(animate);

      if (isFlythroughActive && flythroughCurveRef.current && cameraRef.current) {
        flythroughT.current = (flythroughT.current + 0.0015) % 1.0;
        setFlythroughProgress(Math.round(flythroughT.current * 100));
        const pos = flythroughCurveRef.current.getPointAt(flythroughT.current);
        const lookTarget = new THREE.Vector3(0, 5 * verticalExaggeration, 0);
        cameraRef.current.position.copy(pos);
        cameraRef.current.lookAt(lookTarget);
      }

      if (rendererRef.current && sceneRef.current && cameraRef.current) {
        rendererRef.current.render(sceneRef.current, cameraRef.current);
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
      if (animFrameId.current) cancelAnimationFrame(animFrameId.current);
      renderer.dispose();
    };
  }, [updateCameraPosition]);

  // Mouse Controls
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const onMouseDown = (e: MouseEvent) => {
      isMouseDown.current = true;
      mouseButton.current = e.button;
      mousePrev.current = { x: e.clientX, y: e.clientY };
    };

    const onMouseMove = (e: MouseEvent) => {
      if (!isMouseDown.current || isFlythroughActive) return;
      const dx = e.clientX - mousePrev.current.x;
      const dy = e.clientY - mousePrev.current.y;
      mousePrev.current = { x: e.clientX, y: e.clientY };

      if (mouseButton.current === 0 && !e.shiftKey) {
        cameraAngle.current.theta -= dx * 0.0075;
        cameraAngle.current.phi = Math.max(0.08, Math.min(Math.PI / 2 - 0.02, cameraAngle.current.phi + dy * 0.0075));
      } else {
        const right = new THREE.Vector3(1, 0, 0).applyAxisAngle(new THREE.Vector3(0, 1, 0), cameraAngle.current.theta);
        const forward = new THREE.Vector3(0, 0, 1).applyAxisAngle(new THREE.Vector3(0, 1, 0), cameraAngle.current.theta);
        const panSpeed = cameraAngle.current.radius * 0.0012;
        cameraTarget.current.addScaledVector(right, -dx * panSpeed);
        cameraTarget.current.addScaledVector(forward, -dy * panSpeed);
      }
      updateCameraPosition();
    };

    const onMouseUp = () => {
      isMouseDown.current = false;
    };

    const onContextMenu = (e: MouseEvent) => e.preventDefault();

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      if (isFlythroughActive) return;
      cameraAngle.current.radius = Math.max(15, Math.min(320, cameraAngle.current.radius + e.deltaY * 0.14));
      updateCameraPosition();
    };

    container.addEventListener("mousedown", onMouseDown);
    container.addEventListener("contextmenu", onContextMenu);
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
    container.addEventListener("wheel", onWheel, { passive: false });

    return () => {
      container.removeEventListener("mousedown", onMouseDown);
      container.removeEventListener("contextmenu", onContextMenu);
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);
      container.removeEventListener("wheel", onWheel);
    };
  }, [updateCameraPosition, isFlythroughActive]);

  // Edge-preserving Bilateral Filter to make building facades vertical and roofs sharp
  const getEdgePreservedHeights = useCallback((meshData: MeshHeightfieldData): Float32Array => {
    const gw = meshData.grid_width;
    const gh = meshData.grid_height;
    const raw = meshData.heights;
    const out = new Float32Array(raw.length);
    const thresh = 0.07; // step boundary threshold

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

            if (Math.abs(nval - val) > thresh) {
              edgeCount++;
            }
            maxNeighbor = Math.max(maxNeighbor, nval);
            minNeighbor = Math.min(minNeighbor, nval);
          }
        }

        if (edgeCount >= 3) {
          // Sharpen building edge to prevent diagonal slope melting
          out[idx] = val > (maxNeighbor + minNeighbor) / 2 ? maxNeighbor : minNeighbor;
        } else {
          out[idx] = val;
        }
      }
    }
    return out;
  }, []);

  // Build 3D Objects (Realistic Textured Mesh, Roblox/Minecraft Voxel World, Point Cloud, Wireframe, Heatmap)
  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) return;

    // Disposal
    if (voxelGroupRef.current) {
      scene.remove(voxelGroupRef.current);
      voxelGroupRef.current.traverse((child) => {
        if ((child as THREE.Mesh).geometry) (child as THREE.Mesh).geometry.dispose();
        if ((child as THREE.Mesh).material) {
          const mat = (child as THREE.Mesh).material;
          if (Array.isArray(mat)) mat.forEach((m) => m.dispose());
          else mat.dispose();
        }
      });
      voxelGroupRef.current = null;
    }
    if (pointsObjectRef.current) {
      scene.remove(pointsObjectRef.current);
      pointsObjectRef.current.geometry.dispose();
      (pointsObjectRef.current.material as THREE.Material).dispose();
      pointsObjectRef.current = null;
    }
    if (meshObjectRef.current) {
      scene.remove(meshObjectRef.current);
      meshObjectRef.current.geometry.dispose();
      (meshObjectRef.current.material as THREE.Material).dispose();
      meshObjectRef.current = null;
    }
    if (wireframeObjectRef.current) {
      scene.remove(wireframeObjectRef.current);
      wireframeObjectRef.current.geometry.dispose();
      (wireframeObjectRef.current.material as THREE.Material).dispose();
      wireframeObjectRef.current = null;
    }
    if (heatmapObjectRef.current) {
      scene.remove(heatmapObjectRef.current);
      heatmapObjectRef.current.geometry.dispose();
      (heatmapObjectRef.current.material as THREE.Material).dispose();
      heatmapObjectRef.current = null;
    }

    if (!mesh || mesh.heights.length === 0) return;

    const gw = mesh.grid_width;
    const gh = mesh.grid_height;
    const planeSize = 100.0;
    const heightsFiltered = getEdgePreservedHeights(mesh);

    // -------------------------------------------------------------
    // 1. REALISTIC SMOOTH TEXTURED 3D MESH (UV IMAGE MAPPED)
    // -------------------------------------------------------------
    const planeGeo = new THREE.PlaneGeometry(planeSize, (planeSize * gh) / gw, gw - 1, gh - 1);
    planeGeo.rotateX(-Math.PI / 2);

    const posAttr = planeGeo.attributes.position;
    const meshColors = new Float32Array(posAttr.count * 3);
    const heatmapColors = new Float32Array(posAttr.count * 3);

    for (let i = 0; i < posAttr.count; i++) {
      const h = heightsFiltered[i] || 0;
      posAttr.setY(i, h * 28.0 * verticalExaggeration);

      const r = mesh.colors[i * 3] ?? 0.5;
      const g = mesh.colors[i * 3 + 1] ?? 0.5;
      const b = mesh.colors[i * 3 + 2] ?? 0.5;
      meshColors[i * 3] = r;
      meshColors[i * 3 + 1] = g;
      meshColors[i * 3 + 2] = b;

      const normH = Math.min(1.0, Math.max(0.0, h));
      const hm = getTurboColor(normH);
      heatmapColors[i * 3] = hm.r;
      heatmapColors[i * 3 + 1] = hm.g;
      heatmapColors[i * 3 + 2] = hm.b;
    }

    planeGeo.setAttribute("color", new THREE.BufferAttribute(meshColors, 3));
    planeGeo.computeVertexNormals();

    // Create Realistic PBR Material
    const meshMat = new THREE.MeshStandardMaterial({
      map: loadedTextureRef.current || null,
      vertexColors: !loadedTextureRef.current,
      roughness: 0.5,
      metalness: 0.08,
      side: THREE.DoubleSide,
    });

    const meshObj = new THREE.Mesh(planeGeo.clone(), meshMat);
    meshObj.receiveShadow = true;
    meshObj.castShadow = true;
    meshObj.visible = renderMode === "mesh";
    scene.add(meshObj);
    meshObjectRef.current = meshObj;

    // -------------------------------------------------------------
    // 2. VOXEL WORLD (ROBLOX / MINECRAFT STYLE BLOCK CITY)
    // -------------------------------------------------------------
    const voxelGroup = new THREE.Group();
    const vxCols = Math.min(56, gw);
    const vxRows = Math.min(56, gh);
    const blockWidth = planeSize / vxCols;
    const blockDepth = planeSize / vxRows;
    const totalVoxels = vxCols * vxRows;

    const blockGeo = new THREE.BoxGeometry(blockWidth * 0.94, 1.0, blockDepth * 0.94);
    blockGeo.translate(0, 0.5, 0);

    const blockMat = new THREE.MeshStandardMaterial({
      roughness: 0.55,
      metalness: 0.12,
      shadowSide: THREE.DoubleSide,
    });

    const instancedVoxels = new THREE.InstancedMesh(blockGeo, blockMat, totalVoxels);
    instancedVoxels.castShadow = true;
    instancedVoxels.receiveShadow = true;

    const roofDetailsGeo = new THREE.BoxGeometry(blockWidth * 0.45, 1.2, blockDepth * 0.45);
    roofDetailsGeo.translate(0, 0.6, 0);
    const roofDetailsMat = new THREE.MeshStandardMaterial({ color: 0x94a3b8, roughness: 0.4 });
    const roofDetailsGroup = new THREE.Group();

    const treeTrunkGeo = new THREE.BoxGeometry(blockWidth * 0.3, 2.5, blockDepth * 0.3);
    const treeLeavesGeo = new THREE.BoxGeometry(blockWidth * 1.1, 2.2, blockDepth * 1.1);
    const treeTrunkMat = new THREE.MeshStandardMaterial({ color: 0x78350f, roughness: 0.9 });
    const treeLeavesMat = new THREE.MeshStandardMaterial({ color: 0x22c55e, roughness: 0.6 });
    const treeGroup = new THREE.Group();

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

        const maxBlockHeight = 32.0 * verticalExaggeration;
        const numSteps = 28;
        const quantizedH = Math.max(0.5, Math.round(rawH * numSteps) / numSteps) * maxBlockHeight;

        const posX = (c / vxCols - 0.5) * planeSize + blockWidth / 2;
        const posZ = (r / vxRows - 0.5) * planeSize + blockDepth / 2;

        const isGreenVegetation = green > red * 1.12 && green > blue * 1.1 && rawH < 0.35;
        const isDarkRoad = red < 0.35 && green < 0.35 && blue < 0.35 && rawH < 0.18;
        const isBuilding = rawH >= 0.22 && !isGreenVegetation;

        if (isGreenVegetation) {
          colorObj.setRGB(0.18 + green * 0.4, 0.65 + green * 0.3, 0.18 + blue * 0.2);
          if ((c + r) % 3 === 0 && (c * r) % 5 === 0) {
            const trunkMesh = new THREE.Mesh(treeTrunkGeo, treeTrunkMat);
            trunkMesh.position.set(posX, 0.5, posZ);
            trunkMesh.castShadow = true;
            treeGroup.add(trunkMesh);

            const leavesMesh = new THREE.Mesh(treeLeavesGeo, treeLeavesMat);
            leavesMesh.position.set(posX, 2.5, posZ);
            leavesMesh.castShadow = true;
            treeGroup.add(leavesMesh);
          }
        } else if (isDarkRoad) {
          const isCrosswalkLine = (c % 8 === 0 && r % 2 === 0) || (r % 8 === 0 && c % 2 === 0);
          colorObj.setHex(isCrosswalkLine ? 0xe2e8f0 : 0x334155);
        } else if (isBuilding) {
          if (red > green && red > blue * 1.1) {
            colorObj.setRGB(0.85, 0.55 + red * 0.2, 0.42);
          } else if (red > 0.6 && green > 0.6 && blue > 0.5) {
            colorObj.setRGB(0.92, 0.86, 0.74);
          } else {
            colorObj.setRGB(0.48 + red * 0.3, 0.52 + green * 0.3, 0.58 + blue * 0.3);
          }

          if (rawH > 0.45 && (c * r) % 4 === 0) {
            const hvacMesh = new THREE.Mesh(roofDetailsGeo, roofDetailsMat);
            hvacMesh.position.set(posX, quantizedH, posZ);
            hvacMesh.castShadow = true;
            roofDetailsGroup.add(hvacMesh);
          }
        } else {
          colorObj.setRGB(red * 0.95, green * 0.95, blue * 0.95);
        }

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
    voxelGroup.add(roofDetailsGroup);
    voxelGroup.add(treeGroup);
    voxelGroup.visible = renderMode === "voxel";
    scene.add(voxelGroup);
    voxelGroupRef.current = voxelGroup;

    // -------------------------------------------------------------
    // 3. BUILD WIREFRAME MESH
    // -------------------------------------------------------------
    const wireMat = new THREE.MeshBasicMaterial({
      color: 0x00e5ff,
      wireframe: true,
      transparent: true,
      opacity: 0.75,
    });
    const wireObj = new THREE.Mesh(planeGeo.clone(), wireMat);
    wireObj.visible = renderMode === "wireframe";
    scene.add(wireObj);
    wireframeObjectRef.current = wireObj;

    // -------------------------------------------------------------
    // 4. BUILD ELEVATION HEATMAP MESH
    // -------------------------------------------------------------
    const hmGeo = planeGeo.clone();
    hmGeo.setAttribute("color", new THREE.BufferAttribute(heatmapColors, 3));
    const hmMat = new THREE.MeshStandardMaterial({
      vertexColors: true,
      roughness: 0.4,
      metalness: 0.2,
    });
    const hmObj = new THREE.Mesh(hmGeo, hmMat);
    hmObj.visible = renderMode === "heatmap";
    scene.add(hmObj);
    heatmapObjectRef.current = hmObj;

    // -------------------------------------------------------------
    // 5. BUILD POINT CLOUD
    // -------------------------------------------------------------
    if (pointcloud && pointcloud.positions.length > 0) {
      const pGeo = new THREE.BufferGeometry();
      const posArray = new Float32Array(pointcloud.positions.length);
      const colArray = new Float32Array(pointcloud.colors.length);

      for (let i = 0; i < pointcloud.positions.length; i += 3) {
        posArray[i] = pointcloud.positions[i];
        posArray[i + 1] = pointcloud.positions[i + 1] * verticalExaggeration;
        posArray[i + 2] = pointcloud.positions[i + 2];

        colArray[i] = pointcloud.colors[i];
        colArray[i + 1] = pointcloud.colors[i + 1];
        colArray[i + 2] = pointcloud.colors[i + 2];
      }

      pGeo.setAttribute("position", new THREE.BufferAttribute(posArray, 3));
      pGeo.setAttribute("color", new THREE.BufferAttribute(colArray, 3));

      const pMat = new THREE.PointsMaterial({
        size: pointSize,
        vertexColors: true,
        sizeAttenuation: true,
      });

      const pts = new THREE.Points(pGeo, pMat);
      pts.visible = renderMode === "pointcloud";
      scene.add(pts);
      pointsObjectRef.current = pts;
    }
  }, [pointcloud, mesh, verticalExaggeration, pointSize, renderMode, getEdgePreservedHeights]);

  // Visibility Switcher
  useEffect(() => {
    if (meshObjectRef.current) meshObjectRef.current.visible = renderMode === "mesh";
    if (voxelGroupRef.current) voxelGroupRef.current.visible = renderMode === "voxel";
    if (pointsObjectRef.current) pointsObjectRef.current.visible = renderMode === "pointcloud";
    if (wireframeObjectRef.current) wireframeObjectRef.current.visible = renderMode === "wireframe";
    if (heatmapObjectRef.current) heatmapObjectRef.current.visible = renderMode === "heatmap";
  }, [renderMode]);

  // Synchronized Hover Cursor Pin Update
  useEffect(() => {
    if (!cursorMarkerRef.current) return;
    if (hoverCoordinate) {
      const x = (hoverCoordinate.u - 0.5) * 100.0;
      const z = (hoverCoordinate.v - 0.5) * 100.0;
      const y = hoverCoordinate.relDepth * 28.0 * verticalExaggeration;

      cursorMarkerRef.current.position.set(x, y, z);
      cursorMarkerRef.current.visible = true;
    } else {
      cursorMarkerRef.current.visible = false;
    }
  }, [hoverCoordinate, verticalExaggeration]);

  // Highlight Selected Building 3D Extruded Box
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

      const baseZ = selectedBuilding.ground_base_z_rel * 28 * verticalExaggeration;
      const roofZ = selectedBuilding.rooftop_peak_z_rel * 28 * verticalExaggeration;
      const h3d = Math.max(1.8, roofZ - baseZ);

      const boxGeo = new THREE.BoxGeometry(w3d, h3d, d3d);
      const edges = new THREE.EdgesGeometry(boxGeo);
      const lineMat = new THREE.LineBasicMaterial({ color: 0x10b981, linewidth: 3 });
      const boxLines = new THREE.LineSegments(edges, lineMat);
      boxLines.position.set(x3d, baseZ + h3d / 2, z3d);

      scene.add(boxLines);
      buildingBoxRef.current = boxLines;
    }
  }, [selectedBuilding, verticalExaggeration, depthShape]);

  // Turbo Color helper
  function getTurboColor(x: number) {
    const r = 0.1357 + x * (4.61539 - x * (42.6603 - x * (132.131 - x * (161.073 - x * 65.402))));
    const g = 0.0914 + x * (2.19418 + x * (16.4218 - x * (57.4583 - x * (71.3094 - x * 31.764))));
    const b = 0.1067 + x * (12.5925 - x * (60.1097 - x * (109.0745 - x * (88.5061 - x * 26.818))));
    return {
      r: Math.min(1, Math.max(0, r)),
      g: Math.min(1, Math.max(0, g)),
      b: Math.min(1, Math.max(0, b)),
    };
  }

  const resetCamera = () => {
    setIsFlythroughActive(false);
    cameraTarget.current.set(0, 0, 0);
    cameraAngle.current = { theta: Math.PI / 3.8, phi: Math.PI / 3.4, radius: 145 };
    updateCameraPosition();
  };

  const setIsometricView = () => {
    setIsFlythroughActive(false);
    cameraTarget.current.set(0, 0, 0);
    cameraAngle.current = { theta: Math.PI / 4, phi: Math.PI / 3.2, radius: 135 };
    updateCameraPosition();
  };

  const setTopView = () => {
    setIsFlythroughActive(false);
    cameraTarget.current.set(0, 0, 0);
    cameraAngle.current = { theta: 0, phi: 0.05, radius: 155 };
    updateCameraPosition();
  };

  return (
    <div className="relative w-full h-[550px] rounded-xl overflow-hidden border border-slate-800 bg-[#070A0F] flex flex-col select-none">
      
      {/* ------------------------------------------------------------- */}
      {/* TOP-LEFT OVERLAY: DepthWizard 3D Header Badge */}
      {/* ------------------------------------------------------------- */}
      <div className="absolute top-3 left-3 z-20 flex items-center gap-2.5 p-2 px-3 bg-[#0F172A]/90 backdrop-blur-md rounded-xl border border-slate-700/80 shadow-xl pointer-events-auto">
        <div className="w-8 h-8 rounded-lg bg-cyan-500/20 border border-cyan-400/40 flex items-center justify-center text-cyan-400">
          <Sparkles className="w-5 h-5 animate-pulse text-cyan-300" />
        </div>
        <div>
          <h3 className="text-xs font-extrabold text-slate-100 tracking-wide uppercase font-mono-data flex items-center gap-1.5">
            DepthWizard 3D
            <Badge variant="outline" className="bg-cyan-500/20 text-cyan-300 border-cyan-500/40 text-[9px] px-1 py-0 font-mono-data">
              HD Photogrammetry
            </Badge>
          </h3>
          <p className="text-[10px] text-slate-400 font-mono-data">Real 2D UV Texture → Crisp 3D Elevation</p>
        </div>
      </div>

      {/* ------------------------------------------------------------- */}
      {/* TOP-RIGHT OVERLAY: Mode Switcher & Camera Views */}
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
            Textured 3D Mesh
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
            variant={renderMode === "heatmap" ? "default" : "ghost"}
            onClick={() => setRenderMode("heatmap")}
            data-testid="viewer-mode-heatmap-button"
            className={`h-7 px-2.5 text-xs font-mono-data ${
              renderMode === "heatmap" ? "bg-cyan-500 text-black font-semibold" : "text-slate-300 hover:text-white"
            }`}
          >
            <Flame className="w-3.5 h-3.5 mr-1" />
            Heatmap
          </Button>

          <div className="w-[1px] h-4 bg-slate-700 mx-0.5" />

          <Button
            size="icon-xs"
            variant="ghost"
            onClick={setIsometricView}
            title="Isometric 3D View"
            data-testid="viewer-isometric-button"
            className="text-slate-300 hover:text-cyan-400 h-7 w-7"
          >
            <Compass className="w-3.5 h-3.5" />
          </Button>

          <Button
            size="icon-xs"
            variant="ghost"
            onClick={setTopView}
            title="Top-Down Ortho View"
            data-testid="viewer-topview-button"
            className="text-slate-300 hover:text-cyan-400 h-7 w-7"
          >
            <Eye className="w-3.5 h-3.5" />
          </Button>
        </div>
      </div>

      {/* ------------------------------------------------------------- */}
      {/* RIGHT SIDEBAR: Height Legend & Interactive Controls Panel */}
      {/* ------------------------------------------------------------- */}
      <div className="absolute top-16 right-3 z-20 flex flex-col gap-2.5 pointer-events-none w-44">
        
        {/* Height Legend Panel */}
        <div className="p-3 bg-[#0F172A]/90 backdrop-blur-md rounded-xl border border-slate-700/80 shadow-xl pointer-events-auto text-xs font-mono-data space-y-2">
          <div className="flex items-center justify-between text-slate-200 font-bold border-b border-slate-800 pb-1.5">
            <span>Height (meters)</span>
            <span className="text-[10px] text-cyan-400">{selectedBuilding?.calibrated_height_m ? 'Metric' : 'Relative'}</span>
          </div>

          <div className="flex items-center gap-2.5">
            <div className="w-3.5 h-28 rounded-md bg-gradient-to-t from-red-600 via-yellow-400 via-emerald-400 via-sky-400 to-slate-700 border border-slate-700 shrink-0" />

            <div className="flex flex-col justify-between h-28 text-[10px] text-slate-300 font-mono-data">
              <span className="font-semibold text-red-400">{maxSceneHeightMeters}+ m</span>
              <span className="text-yellow-300">{Math.round(maxSceneHeightMeters * 0.8)} m</span>
              <span className="text-emerald-300">{Math.round(maxSceneHeightMeters * 0.6)} m</span>
              <span className="text-sky-300">{Math.round(maxSceneHeightMeters * 0.2)} m</span>
              <span className="text-slate-400">0 m</span>
            </div>
          </div>
        </div>

        {/* Controls Panel */}
        <div className="p-3 bg-[#0F172A]/90 backdrop-blur-md rounded-xl border border-slate-700/80 shadow-xl pointer-events-auto text-xs font-mono-data space-y-2">
          <div className="flex items-center justify-between text-slate-200 font-bold border-b border-slate-800 pb-1.5">
            <span>Controls</span>
            <Button
              size="icon-xs"
              variant="ghost"
              onClick={resetCamera}
              title="Reset View [R]"
              className="text-cyan-400 hover:bg-cyan-500/20 h-5 w-5"
            >
              <RotateCcw className="w-3 h-3" />
            </Button>
          </div>

          <div className="space-y-1.5 text-[11px] text-slate-300">
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded bg-slate-800 border border-slate-700 flex items-center justify-center text-cyan-400">
                <MousePointer className="w-2.5 h-2.5" />
              </div>
              <span>Left Drag: Rotate</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded bg-slate-800 border border-slate-700 flex items-center justify-center text-emerald-400">
                <Move className="w-2.5 h-2.5" />
              </div>
              <span>Right Drag: Pan</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded bg-slate-800 border border-slate-700 flex items-center justify-center text-amber-400">
                <ZoomIn className="w-2.5 h-2.5" />
              </div>
              <span>Scroll: Zoom</span>
            </div>
            <div className="pt-1 flex items-center gap-1.5">
              <Button
                size="sm"
                variant="outline"
                onClick={resetCamera}
                className="w-full h-6 text-[10px] bg-slate-800/80 border-slate-700 text-cyan-300 hover:bg-cyan-500/20"
              >
                [R] Reset View
              </Button>
            </div>
          </div>
        </div>

      </div>

      {/* ------------------------------------------------------------- */}
      {/* BOTTOM-LEFT OVERLAY: Input Aerial Image Thumbnail Overlay */}
      {/* ------------------------------------------------------------- */}
      {imageUrl && (
        <div className="absolute bottom-3 left-3 z-20 pointer-events-auto">
          {showAerialThumbnail ? (
            <div className="p-2 bg-[#0F172A]/95 backdrop-blur-md rounded-xl border border-slate-700/90 shadow-2xl space-y-1.5 w-48">
              <div className="flex items-center justify-between text-[11px] font-bold text-slate-200 font-mono-data">
                <span>Input Aerial Image</span>
                <button
                  onClick={() => setShowAerialThumbnail(false)}
                  className="text-slate-400 hover:text-white p-0.5"
                >
                  <Minimize2 className="w-3 h-3" />
                </button>
              </div>
              <div className="w-full h-28 rounded-lg overflow-hidden border border-slate-800 bg-black">
                <img src={imageUrl} alt="Input Aerial" className="w-full h-full object-cover" />
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
      {/* BOTTOM-RIGHT OVERLAY: Description & Controls */}
      {/* ------------------------------------------------------------- */}
      <div className="absolute bottom-3 right-3 z-20 max-w-xs p-3 bg-[#0F172A]/90 backdrop-blur-md rounded-xl border border-slate-700/80 shadow-xl pointer-events-auto text-xs text-slate-300 font-mono-data space-y-1">
        <h4 className="font-bold text-slate-100 flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
          Realistic 2D-to-3D UV Photogrammetry
        </h4>
        <p className="text-[11px] text-slate-400 leading-tight">
          High-definition 2D texture mapped onto edge-preserved neural depth geometry.
        </p>
        <div className="pt-1 flex items-center justify-between">
          <Button
            size="sm"
            variant="ghost"
            onClick={() => setIsFlythroughActive(!isFlythroughActive)}
            className={`h-6 text-[10px] px-2 font-mono-data ${
              isFlythroughActive ? "bg-cyan-500/20 text-cyan-300 animate-pulse" : "text-cyan-400 hover:bg-cyan-500/20"
            }`}
          >
            {isFlythroughActive ? <Pause className="w-3 h-3 mr-1" /> : <Play className="w-3 h-3 mr-1" />}
            {isFlythroughActive ? `Flythrough ${flythroughProgress}%` : "Cinematic Flythrough"}
          </Button>
          <div className="flex items-center gap-1 text-[10px] text-slate-400 font-mono-data">
            <span>Scale:</span>
            <input
              type="range"
              min="0.4"
              max="3.0"
              step="0.1"
              value={verticalExaggeration}
              onChange={(e) => setVerticalExaggeration(parseFloat(e.target.value))}
              className="w-14 h-1 bg-slate-700 rounded appearance-none cursor-pointer accent-cyan-400"
            />
            <span className="text-cyan-300">{verticalExaggeration.toFixed(1)}x</span>
          </div>
        </div>
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
