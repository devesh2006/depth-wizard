# SIH26175 — DepthWizard
## Final Consolidated Specification (MVP + Visual Upgrade, merged)
### Single-Image Depth-Assisted 3D Reconstruction and Calibrated Height Estimation

This document merges the three pasted prompts (the original MVP spec and the two identical Visual-Upgrade prompts) into **one final, de-duplicated specification**, plus the configuration decisions selected below. Where the two source prompts overlapped, the content was merged once. Nothing scientific was changed — this is an organizational consolidation, not a new set of rules.

---

## 0. Configuration Decisions (locked for this build)

These four choices were made explicitly and now govern every downstream section of this spec:

| Decision Point | Selection |
|---|---|
| **Depth estimation backend** | **Depth Anything V2**, with a robust CPU/GPU tensor-inference path and a fallback geometry engine. Must support both an instant preview from a **pre-computed hero cache** and true **live inference** — the UI should make clear which one the user is looking at. |
| **Hero validation dataset (SIH judging)** | **Urban Buildings Focus** — high-density downtown building scenes, validated against **LiDAR-derived ground-truth heights**. This becomes the default demo dataset and the default validation benchmark. |
| **3D viewer & measurement modes** | Not explicitly chosen — **defaults applied** (see §14.1). |
| **Calibration handling for uncertain/uncalibrated scenes** | Not explicitly chosen — **defaults applied** (see §11.5). |

Everything below reflects these four decisions wherever relevant.

---

## 1. Goal Statement

The goal is **not** to claim that a single RGB image can replace LiDAR or stereo photogrammetry.

The goal is a technically credible pipeline: a single aerial/satellite/drone image → depth/relative elevation → a visually meaningful 3D representation → **calibrated approximate height estimates when sufficient reference information is available.**

Concept: **Single Image → Depth/Height Map → 3D Model → Interactive Flythrough → Measurement.**

Principal technical challenges: metric-scale calibration and aerial-view generalization.

---

## 2. Core Engineering Principle — Data Integrity Rule

This rule governs the entire application, front end and back end alike. It appeared in all three source prompts and is treated here as the single non-negotiable constraint.

**Never:**
- hard-code building heights
- fabricate accuracy values
- pretend relative depth is absolute elevation
- claim satellite-grade photogrammetric accuracy
- claim LiDAR replacement
- use random scale factors without explaining them
- display a number like "23.4 m" without a defensible calibration/reference basis
- invent validation metrics
- label simulated geometry as measured geometry

**If metric scale cannot be established, the UI must explicitly display:**
> **"Relative depth — metric scale unavailable."**

Monocular depth estimation and geospatial elevation estimation are not the same problem, and the application must never blur that line.

---

## 3. Three Concepts That Must Never Be Conflated

| Concept | Definition |
|---|---|
| **A. Relative Depth** | What is closer/farther in the image. Unitless. |
| **B. Metric Depth** | Depth expressed in meters — only valid once calibrated. |
| **C. Height / Elevation** | Vertical height relative to a defined ground/reference surface. |

---

## 4. End-to-End Pipeline

```
INPUT IMAGE
   ↓
Image Quality / Scene Analysis
   ↓
Scene / View-Geometry Classification (oblique vs. near-nadir)
   ↓
Depth Anything V2  ← [chosen backend, §5]
   ↓
Raw Floating-Point Depth
   ↓
Depth Quality / Confidence Assessment
   ↓
Geometric Reconstruction (camera intrinsics + back-projection)
   ↓
3D Point Cloud
   ↓
3D Mesh / Height Surface
   ↓
Optional Metric Calibration (4 modes, §11)
   ↓
Building / Object Height Estimation
   ↓
Interactive 3D Viewer (dark grayscale heat-map, §14)
   ↓
Measurement + Validation (§16, LiDAR-benchmarked, §17.1)
```

---

## 5. Model Strategy (Backend: Depth Anything V2, per Configuration Decision)

Use **Depth Anything V2** as the monocular-depth backbone, wrapped behind a clean interface so it can later be swapped for an aerial/satellite-fine-tuned model without rewriting the application:

```
DepthEstimator
    ├── DepthAnythingV2          (current)
    └── Future/AerialFineTunedModel
```

Notes:
- The official repo's *metric*-depth models are trained on indoor Hypersim and outdoor Virtual KITTI 2 — they are **not** satellite-specific, and the UI/README must say so plainly.
- Start with **Depth Anything V2 Small or Base**, sized to available GPU memory. Do not default to the largest checkpoint. Benchmark inference time, GPU memory, visual quality, and stability, and pick accordingly.

**Per the configuration decision, the backend must support two execution paths side by side:**
1. **Live inference** — real Depth Anything V2 forward pass on CPU or GPU, with a CPU fallback if no GPU/CUDA is present.
2. **Hero cache preview** — instant load of a previously generated, legitimately-computed output for the bundled Urban Buildings Focus scenes (§17), used when live inference is slow, unavailable, or during judging if live inference fails.

The UI must always label which path produced the currently displayed result (`LIVE` vs `CACHED — precomputed`), so nothing is presented as live when it isn't.

A "fallback geometry engine" activates if depth inference itself fails or produces degenerate output (e.g. all-flat depth): it falls back to an approximate planar/terrain assumption, clearly labeled as **low-confidence geometric approximation**, never silently substituted for a real depth result.

---

## 6. Input Types & Metadata

UI selector:
```
Image Type:
[ Satellite ]  [ Aerial ]  [ Drone ]
```

Optional camera/geometry fields (never required for the visual 3D reconstruction, but required before any metric-height claim):
```
GSD:                    Optional
Altitude:               Optional
Focal Length:           Optional
Sensor Information:     Optional
Known Reference Height: Optional
```

---

## 7. Image Quality Analysis

Run before inference. Calculate: resolution, blur/sharpness, brightness, contrast, saturation, potential cloud/occlusion, extreme shadows, aspect ratio.

```
Image Quality
----------------
Resolution: 3840 × 2160
Sharpness: Good
Brightness: Good
Potential Occlusion: Low
Depth Suitability: Medium
```

If the image is unsuitable, warn the user rather than silently producing a misleading result.

---

## 8. Depth Estimation

```
RGB Image → Preprocessing → Depth Anything V2 → Raw Depth → Normalization → Depth Visualization
```

Persist artifacts for every run:
```
raw_depth.npy
depth_visualization.png
normalized_depth.png
```
Raw floating-point depth is never discarded, even after visualization.

---

## 9. Depth Quality / Confidence

Since one monocular image has no ground-truth depth, confidence is **not** shown as a fake probability. Instead compute real quality signals: local depth consistency, gradient consistency, invalid-pixel percentage, extreme-depth concentration, flat-depth detection, plus the image-quality indicators from §7.

```
Depth Quality: GOOD
```
or
```
Depth Quality: LOW
Reason: Input is close to nadir and contains limited geometric cues.
```

---

## 10. Near-Nadir vs. Oblique Handling

Near-nadir satellite imagery is fundamentally harder for a monocular depth model than oblique/perspective imagery: building facades may be invisible, roof geometry dominates, perspective cues are weak, and apparent depth may not correspond to building height.

```
VIEW GEOMETRY
  Oblique / Perspective  → better geometric cues
  Near-Nadir             → limited height cues → requires stronger calibration
```

If near-nadir is detected, show, unhidden:
> **"Near-nadir imagery detected. Height estimates may have higher uncertainty."**

---

## 11. Height Estimation — Calibration Architecture

Never compute `height = arbitrary_depth × constant`. Use one of four explicit modes.

### 11.1 Mode 1 — Reference-Based Calibration
User supplies a known reference height for a visible building/object; that reference establishes the depth→meters scale for the scene.

### 11.2 Mode 2 — Metadata-Assisted Calibration
Uses GSD, camera altitude, sensor/camera parameters, and/or terrain reference when supplied.

### 11.3 Mode 3 — Terrain-Referenced Height
```
Estimated Surface Elevation − Ground Elevation = Approximate Object Height
```
Requires an external DEM.

### 11.4 Mode 4 — Relative Height (default fallback)
No metric reference → show **Relative Height / Relative Depth** and state:
> **"Metric height unavailable without scale/reference information."**

### 11.5 Default Behavior for Uncertain / Uncalibrated Scenes *(Configuration Decision — default applied)*

Since no explicit policy was chosen, the system defaults to the **most conservative, most defensible behavior**, consistent with §2:

- If **no calibration input exists at all** → always fall back to **Mode 4 (Relative Height)**. Never guess a metric value.
- If a calibration input exists but is **low-confidence** (e.g. a single weak reference, no cross-check, near-nadir geometry with no metadata) → still display a metric estimate, but:
  - force **Confidence: Low**
  - widen the displayed **uncertainty band** rather than hiding it
  - show the calibration mode used and *why* confidence is low (e.g. "near-nadir geometry, single reference point")
- If calibration inputs **conflict** (e.g. metadata-derived scale and reference-based scale disagree by more than a defined threshold) → do not silently average them. Show **both** results side by side with their individual confidence, and flag the disagreement explicitly rather than resolving it invisibly.
- Uncertainty (`± X.X m`) is always computed from the actual calibration/validation methodology (e.g. variance across repeated/perturbed calibration attempts) — never a placeholder value.

---

## 12. Building Height Measurement

```
RGB Image → Building Segmentation → Building Footprints → Depth/3D Surface → Ground Reference → Building Height
```

MVP: simple segmentation approach. Later: SAM/SAM2, semantic segmentation, OSM footprints, or a custom detector. Keep this component modular so the segmentation method can be swapped independently of the depth/calibration stack.

### 12.1 Building Analysis UI

Calibrated:
```
BUILDING ANALYSIS
Building ID:        B-024
Estimated Height:   18.7 m
Measurement Mode:   Reference Calibrated
Confidence:         Medium
Reference:          Known Building Height
Uncertainty:        ± X.X m
```

Uncalibrated:
```
BUILDING ANALYSIS
Metric Height:      Not available
Relative Height:    0.73
Reason:             No metric reference available
```

On hover (pre-click): subtly illuminate the building's mesh, show a bounding outline, building ID, relative/metric height, uncertainty, and reconstruction quality.

---

## 13. 3D Reconstruction

```
Depth + Camera Intrinsics (fx, fy, cx, cy)
   ↓
Back Projection
   ↓
3D Point Cloud
   ↓
Outlier Removal → Downsampling → Surface Reconstruction
   ↓
3D Mesh
```

If camera parameters are unavailable, estimate approximate intrinsics and **clearly mark the reconstruction as approximate** — never claim geospatial accuracy from guessed intrinsics. Use **Open3D**.

### 13.1 Mesh Quality Honesty
Single-view reconstruction is inherently incomplete and may contain holes, stretched textures, missing surfaces, distorted building sides, or floating points. Apply statistical outlier removal, voxel downsampling, optional smoothing, optional Poisson reconstruction, depth clipping, and mesh simplification — but always **retain the raw point-cloud view** so a degraded mesh never hides the underlying (more honest) data.

---

## 14. 3D Viewer & Visual Design

Stack: **React + Three.js.**

### 14.1 Default Viewer & Measurement Modes *(Configuration Decision — defaults applied)*

Since no explicit selection was made, enable the full baseline set from the spec as the default, with the **hybrid mesh + wireframe** view as the initial mode shown on load (it best communicates "this is a reconstruction, not a photo," matching the transparency principle in §2):

**Visualization modes (toggle group, hybrid selected by default):**
```
● Hybrid (Mesh + Wireframe)   ← default on load
○ Solid Mesh
○ Point Cloud
○ Depth Heatmap
○ Elevation Heatmap
○ Confidence / Quality Overlay
```

**Camera controls (all enabled by default):** orbit, zoom, pan, reset, top view, perspective view, flythrough.

**Measurement tools (all enabled by default):** vertical height ruler, horizontal distance ruler, point-to-point measurement, ground reference line, building footprint outline, elevation marker, contour measurement, coordinate readout.

### 14.2 Visual Language — Dark Grayscale 3D

Premium black/charcoal/graphite/grayscale-only palette, in the spirit of satellite-intelligence software, a scientific visualization platform, a CAD/digital-twin tool, and a modern AI SaaS dashboard — deliberately **not** colorful cyberpunk.

Core palette: `#050505, #0A0A0A, #111111, #171717, #222222, #303030, #444444, #666666`, plus white/near-white for critical information. Use luminance differences, not colorful gradients.

### 14.3 Height-Based Grayscale Heat-Map
```
Depth/Elevation → Normalize → Map value → grayscale intensity → Apply to 3D geometry
```
Lowest surface → near-black … Highest structures → near-white.

Legend:
```
ELEVATION
HIGH   ░░░░░░  35 m
       ▒▒▒▒▒▒
       ▓▓▓▓▓▓
LOW    ██████   0 m
```
If metric calibration is unavailable, the legend must read **RELATIVE DEPTH**, never implied meters.

### 14.4 Mesh / Voxel Aesthetic
Clean low-poly/triangulated mesh; visible mesh topology, building surfaces, roof structures, terrain, elevation contours, subtle wireframe. Stylized **voxel/low-poly geospatial digital twin**, "Minecraft-readable, GIS-accurate" — never literal Minecraft/Roblox assets, logos, or branding.

### 14.5 Live Particles & Data-Flow Animation
Subtle floating/pulsing particles along depth/elevation contours, opacity ~30–50%, never overlapping geometry, measurement lines, labels, or UI. During processing, animate the 2D→depth→points→mesh→heatmap transformation with scanning-line/particle effects — cinematic, but never simulating analysis that hasn't actually run.

### 14.6 Uncertainty Visualization
```
18.7 m
│
│  uncertainty region
│
17.3 m ─────
```
A transparent vertical band represents the uncertainty range from §11.5, not a decorative flourish.

### 14.7 Confidence / Quality Overlay
A separate grayscale overlay: higher-quality regions lighter, lower-quality regions darker/transparent/hatched, driven by the indicators in §9 — never presented as a fabricated probability.

### 14.8 Three-Panel Synchronized Interface
```
┌──────────────┬──────────────┬────────────────┐
│  ORIGINAL    │ DEPTH MAP    │ 3D DIGITAL     │
│  IMAGE       │ HEATMAP      │  TWIN          │
└──────────────┴──────────────┴────────────────┘
```
Selecting a building in one panel highlights it in the other two.

### 14.9 Background & Performance
Slow-moving grayscale particles, thin mesh connections, faint contour lines, subtle grid, faint volumetric fog — no bright neon, no colorful particles. Use instancing, LOD, progressive mesh loading, point-cloud downsampling, and frustum culling so the aesthetic never compromises the underlying geometry's fidelity or frame rate.

---

## 15. Model Validation, Improvement Loop & Accuracy Target

Do not claim "85% accurate" because a document says so. Treat **≥85% validation performance** as a **measurable research target**, only reportable once the metric and dataset genuinely justify it.

```
Collect aerial data → Ground truth → Clean dataset → Train → Validate
   → Evaluate unseen regions → Analyze errors → Improve model → Repeat
```
Track baseline vs. fine-tuned model: MAE, RMSE, Median Error, Relative Error, Inference Time, GPU Memory — and make model-version comparison easy in the UI.

**Data split:** 70% training / 15% validation / 15% test, with **geographic separation** — training cities, a distinct validation city, and a fully unseen test city. Never train and test on visually near-identical neighboring tiles.

---

## 16. Validation Framework & Dashboard

Per test scene, store: Scene ID, Image, Reference Source, Reference Height, Predicted Height, Absolute Error, Relative Error. Compute MAE, RMSE, Median Absolute Error, Mean Relative Error.

For terrain: compare against a DEM only where resolution/coordinate systems make the comparison meaningful (never compare a 1–2 m building estimate against a 30 m DEM and call it building-level ground truth).

```
MODEL VALIDATION
Scenes Evaluated:      25
MAE:                   X.XX m
RMSE:                  X.XX m
Median Error:          X.XX m
Mean Relative Error:   X.XX %
Validation Score:      XX.X %
```
```
Scene | Reference | Prediction | Error
---------------------------------------
A01   | 18.0 m    | 17.2 m     | 0.8 m
A02   | 24.0 m    | 26.1 m     | 2.1 m
```
If there is no validation data yet: display **"Validation data not available."** — never populate the table with placeholder numbers.

---

## 17. Hero Demo Dataset — Urban Buildings Focus *(Configuration Decision)*

The bundled SIH hero dataset is **high-density downtown buildings, benchmarked against LiDAR-derived ground-truth heights.** This is both the default demo scenario and the default validation benchmark referenced in §16.

### 17.1 Why LiDAR benchmarking here
LiDAR-derived heights are the strongest available ground truth for dense urban buildings (versus, e.g., a coarse DEM), which makes this dataset suitable for genuinely computing MAE/RMSE rather than "reference not available." Every hero scene must store its LiDAR reference height alongside the predicted height so the validation dashboard (§16) is populated from real data for at least this dataset.

### 17.2 Hero scene contents (3–4 scenes)
Each hero scene bundles:
```
Original Image
Raw Depth
Depth Visualization
Point Cloud
Mesh
Calibration (LiDAR-referenced)
Building Measurements
Reference Data (LiDAR ground truth)
Validation Results (MAE/RMSE for this scene)
```
All hero outputs must have been generated by the real pipeline beforehand — never fabricated to look plausible.

### 17.3 Demo Flow (~3 minutes)
```
1. Select Hero Image (Urban Buildings Focus)
2. Show Original
3. Click Generate
4. Show Depth Map
5. Show 3D Reconstruction
6. Start Flythrough
7. Select Building
8. Show Height Estimate (LiDAR-calibrated, with uncertainty)
9. Open Validation
10. Show Actual Experimental Metrics (MAE/RMSE vs. LiDAR)
```

---

## 18. LIVE vs. DEMO Mode

**LIVE MODE** — runs Depth Anything V2 for real, on the selected input image.
**DEMO MODE** — loads the precomputed, legitimately-generated Urban Buildings Focus hero outputs (§17), so judging can continue smoothly if live GPU inference fails. Demo Mode outputs are never fabricated; they are cached results of a real prior pipeline run.

---

## 19. Performance Targets

Never promise a fixed number (e.g. "3 seconds") without benchmarking. Measure and display actual timing:
```
Inference: 2.8 s
3D reconstruction: 1.4 s
Total: 4.2 s
```
Hardware-dependent performance is acceptable and should be stated as such.

---

## 20. Backend API

```
POST /api/upload
POST /api/depth
POST /api/reconstruct
POST /api/calibrate
POST /api/measure
POST /api/validate

GET  /api/result/{id}
GET  /api/depth/{id}
GET  /api/pointcloud/{id}
GET  /api/mesh/{id}
```
Asynchronous/background processing for inference; job status returned as `queued | processing | completed | failed`.

---

## 21. Frontend Screens

- **Dashboard** — upload + recent results
- **Analysis** — original image + depth map + quality info
- **3D Explorer** — interactive 3D model (§14)
- **Measurement** — building selection + height estimation (§12)
- **Validation** — accuracy metrics (§16, LiDAR-benchmarked by default)
- **About / Methodology** — explains monocular depth, calibration, 3D reconstruction, and limitations plainly

---

## 22. Geospatial Coordinate Handling

Support CRS, latitude/longitude, GSD, and bounding box. Preserve spatial metadata for georeferenced imagery using `rasterio`, `pyproj`, `geopandas` as needed. Never pretend a generic JPG has geographic coordinates it doesn't have.

---

## 23. Engineering Requirements & Project Structure

Code must be modular, typed where practical, documented, error-tolerant, reproducible, and easy to install. Provide `README.md`, `requirements.txt`, `.env.example`, `Dockerfile`, `docker-compose.yml`, and setup instructions for Windows, Linux, CUDA GPU, and CPU fallback (the last of which is required per the backend decision in §5).

```
depthwizard/
├── frontend/  src/{components, pages, viewer, api}
├── backend/   app/{api, core, depth, geometry, calibration, segmentation, validation, models}
├── data/      {samples, references, outputs}
├── models/
├── scripts/
├── tests/
├── README.md
├── requirements.txt
└── docker-compose.yml
```

---

## 24. Testing

- **Unit:** image preprocessing, calibration, coordinate conversion, depth normalization, height calculation
- **Integration:** image→depth, image→point cloud, image→3D mesh
- **UI:** upload, processing status, viewer loading, measurement interaction

---

## 25. Security / Robustness

Validate MIME type, extension, file size, image dimensions on every upload. Prevent arbitrary file execution, path traversal, and uncontrolled memory consumption.

---

## 26. Scientific Disclaimer (must be visible in-app)

> "Depth and height estimates are dependent on image geometry, model generalization, calibration and reference data. Results should not be treated as survey-grade measurements without independent validation."

Stating this openly is a strength in front of an ISRO-caliber technical jury, not a weakness.

---

## 27. What the Final System Must Prove

1. **Feasibility** — a single image can be processed into a useful depth representation
2. **Visualization** — depth can become an understandable 3D scene
3. **Interaction** — users can explore the reconstructed environment
4. **Measurement** — metric height can be estimated when calibration/reference information exists
5. **Validation** — the system quantifies its errors instead of only looking impressive

---

## 28. Build Order (what NOT to build first)

Skip initially: complex auth, cloud deployment, payment systems, a mobile app, unnecessary dashboards, elaborate animation, or training a huge model from scratch.

**Build in this order:**
```
IMAGE → DEPTH → POINT CLOUD → 3D → VIEWER
   then  CALIBRATION → HEIGHT → VALIDATION
   then  UI polish
```

### Development Phases
1. **Proof of Concept** — Image → Depth
2. **3D** — Depth → Point Cloud → 3D (Three.js viewer)
3. **Geometric Calibration** — camera parameters, reference scale, uncertainty
4. **Building Measurement** — selection, height estimation, confidence
5. **Validation** — test dataset, MAE/RMSE, error visualization
6. **SIH Demo** — polished UI, hero scenes, flythrough, validation dashboard, offline fallback, presentation mode

---

## 29. Judge-Ready Explanation

One-sentence framing:
> **"DepthWizard converts a single aerial or satellite image into a depth-assisted 3D representation and, when metric reference information is available, estimates object heights without requiring a stereo image pair or LiDAR survey."**

**Never say:** "We get exact 3D from one image." / "We replace LiDAR." / "Our AI knows the exact height of every building."

**Say instead:** "We provide rapid, depth-assisted 3D reconstruction and calibrated approximate height estimation from a single image, with explicit uncertainty and validation." *(Note: LiDAR is used only as the ground-truth reference for the Urban Buildings Focus validation benchmark, §17 — not as an input the live pipeline depends on.)*

---

## 30. Final Acceptance Criteria (merged, single checklist)

- [ ] User can upload one image
- [ ] Real Depth Anything V2 inference works (live path, §5)
- [ ] Hero-cache preview path works and is clearly labeled as cached (§5)
- [ ] Raw floating-point depth is preserved
- [ ] Depth quality is calculated
- [ ] Near-nadir scenes are detected and flagged
- [ ] Point cloud generated from depth + camera geometry
- [ ] 3D mesh generated from actual reconstruction
- [ ] Grayscale elevation/depth heat-map derived from actual values (§14.3)
- [ ] Hybrid mesh+wireframe default view + point cloud/heatmap/quality toggles all functional (§14.1)
- [ ] Live particles/animations subtle and performant, never obscuring data
- [ ] Building selection and hover states work
- [ ] All four calibration modes implemented, with the §11.5 default behavior for uncertain/conflicting inputs
- [ ] Metric height is never shown without a valid scale basis
- [ ] Uncertainty is always displayed alongside any metric value
- [ ] Validation uses genuine reference data, with Urban Buildings Focus benchmarked against real LiDAR heights (§17)
- [ ] MAE/RMSE calculated automatically, never hard-coded
- [ ] ≥85% accuracy is treated strictly as a research target, not an asserted fact
- [ ] Geographic train/test separation respected
- [ ] Offline Demo Mode works using legitimately precomputed Urban Buildings Focus scenes
- [ ] 3–4 reliable hero scenes are prepared
- [ ] Three.js flythrough works
- [ ] 2D → Depth → 3D three-panel synchronization works
- [ ] No fake measurements, no fabricated accuracy values anywhere
- [ ] Scientific disclaimer visible in-app
- [ ] Full workflow demonstrable end-to-end in ~3 minutes
- [ ] README explains installation, model setup, dataset setup, and known limitations

---

## 31. Final Instruction to the Coding Agent

Do not generate a large volume of code for its own sake. Work incrementally: implement one stage, run it, test it, verify the output, fix errors, and only then move to the next stage. If a required dataset/model/API (e.g. a real LiDAR-referenced Urban Buildings dataset) is not actually available, do not invent one — build the correct interface and state plainly what is required to complete it.

At completion, deliver: (1) full project structure, (2) all source files, (3) installation instructions, (4) model setup instructions, (5) dataset setup instructions, (6) backend run commands, (7) frontend run commands, (8) demo instructions, (9) testing instructions, (10) known limitations, (11) recommended next steps for improving aerial/satellite accuracy.

**Priority order, always:** technical credibility + working demonstration + measurable validation, over raw feature count.
