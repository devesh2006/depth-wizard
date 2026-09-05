# SIH26175 DepthWizard — Technical Specification

## Overview
DepthWizard converts a **single** aerial / drone / satellite RGB image into a depth-assisted
3D representation and, when metric reference information is available, estimates structure
heights with explicit uncertainty. It deliberately distinguishes **Relative Depth**,
**Metric Depth** and **Height above ground**, and never presents one as another.

Public URL: https://depth-wizard.preview.emergentagent.com
No authentication — every screen is open (see `memory/test_credentials.md`).

## Scientific integrity rules enforced in code
These are hard constraints, not documentation aspirations:

1. **Real model.** Depth inference runs genuine Depth Anything V2 Small transformer
   weights (`depth-anything/Depth-Anything-V2-Small-hf`) on CPU via `transformers`.
   Every response carries `depth_backend` + `depth_backend_is_neural`, surfaced as a
   badge in the header. If torch/weights are missing, `create_depth_estimator()` falls
   back to `StructuralGradientEstimator`, which is labelled
   *"Structural Gradient Relief Baseline (heuristic, non-neural)"* — a heuristic result
   is never dressed up as a neural prediction.
2. **No invented scale factors.** Mode 1 derives `alpha = H_known / dZ_measured`, where
   `dZ_measured` is measured from the current depth map at the reference footprint. The
   reference structure therefore re-measures back to its known height (84.5 m known →
   **84.44 m** measured). Mode 2 derives a ground footprint from the pinhole identity
   `footprint = altitude × sensor_mm / focal_mm`. Mode 3 fits a ground datum and derives
   scale from measured above-ground relief. If a scale cannot be derived, the scene
   degrades to Mode 4 rather than guessing.
3. **No metres without calibration.** Mode 4 returns `calibrated_height_m = null` and
   states *"Relative depth — metric scale unavailable."*
4. **No fabricated accuracy.** No prediction, error or accuracy value is stored anywhere.
   `GET /api/depth/validation` returns `data_available: false` plus the concrete dataset
   requirements, because no verified reference-height dataset is registered for the
   bundled stock imagery. MAE/RMSE/median/MRE are `null`, not `0`.
5. **Limitations shown, not hidden.** Near-nadir imagery raises an explicit
   elevated-uncertainty warning; the methodology modal states the scale/shift ambiguity
   `Z_metric = s·Z_rel + t` and that LiDAR is not replaced.

## Pipeline
```
RGB image
  -> Image quality + view geometry (Laplacian sharpness, brightness, contrast,
     saturation, cloud/shadow risk, Near-Nadir vs Oblique vs Drone classification)
  -> Depth Anything V2 (bounded to MAX_WORK_SIDE=1024, robust 1/99 percentile normalise)
  -> Depth confidence (local consistency, gradient alignment, flat-depth, invalid %)
  -> Reference footprint probed FIRST -> calibration scale derived
  -> Back-projection: X=(u-cx)Z/fx, Y=(v-cy)Z/fy  (intrinsics in DEPTH-GRID space)
     -> point cloud (outlier removal, subsampled to 20k) + heightfield mesh
  -> Structure measurement: rooftop peak (90th pct) vs ground ring (12th pct), ±sigma
  -> Validation scoring (only against registered reference heights)
```

## Data model notes
- Footprints are stored as **fractional** bboxes (`bbox_frac`), resolved against the
  actual depth-grid resolution — so they align at any image size.
- `bbox` values returned by the API are in **depth-grid pixel space**; the frontend
  normalises overlays against `raw_depth_stats.shape`, and `ThreeViewer` receives
  `depthShape` for the 3D selection box.
- **Display geometry is normalised (0..1)**; metric scale lives only in the measurement
  layer. Multiplying vertices by the metric scale (≈158 m/unit) previously pushed the
  surface outside the camera frustum.
- Memory is bounded: depth working resolution capped at 1024 px (**44.4 MB → 2.25 MB**
  per cached scene) and `MAX_CACHED_SCENES = 6`. Unbounded caching previously
  OOM-killed the uvicorn worker and produced 502s on `/api/depth/calibrate`.

## API (all on `api_router`, prefix `/api`)
| Endpoint | Purpose |
|---|---|
| `GET /api/depth/hero-scenes` | Benchmark scene list |
| `GET /api/depth/hero-scene/{id}` | Full pipeline run for a scene (404 on unknown id) |
| `POST /api/depth/process` | Process a remote/base64 image |
| `POST /api/depth/upload` | Multipart upload (MIME allow-list, 25 MB cap) |
| `POST /api/depth/calibrate?scene_id=` | Re-derive scale, re-measure structures |
| `POST /api/depth/measure-building` | Measure a user-drawn bbox |
| `GET /api/depth/validation` | Live-scored metrics, or honest unavailability |
| `GET /api/depth/export-pointcloud/{id}` | ASCII PLY |
| `GET /api/depth/export-mesh/{id}` | Wavefront OBJ |

## Frontend
- **Tri-Panel 3D Studio** — input RGB with clickable/draggable footprints, depth colormap
  (Turbo/Viridis/Magma), and the Three.js studio; synchronised hover crosshair.
- **3D Studio** — orbit/zoom, top-down + perspective, Textured Mesh / Point Cloud /
  Wireframe / Elevation Heatmap, Z-exaggeration slider, Catmull-Rom flythrough.
- **Height Calibration & Structures** — 4 calibration modes, peak-vs-baseline breakdown,
  ±uncertainty, confidence.
- **Validation Framework** — live metrics when reference data exists, otherwise the
  "Validation data not available" panel plus the 5 dataset requirements.
- **SIH 3-Min Walkthrough** — 5-step guided pitch with countdown timer.
- **Methodology modal** — concept distinctions, back-projection maths, disclaimer.
- Animated particle/tech-geometry canvas background (`GeospatialParticleBackground`).

## Known limitations
- No verified ground-truth dataset ⇒ **no accuracy figure is published**. See
  `REFERENCE_DATASET_REQUIREMENTS` in `backend/services/hero_dataset.py`.
- Bundled imagery is stock photography with no CRS; georeferencing fields exist in the
  design but a plain JPEG is never treated as georeferenced.
- Non-reference structures on stock imagery have no surveyed heights, so their metre
  values are calibrated estimates only, not validated measurements.
- CPU-only inference (~1.2–2.6 s per image); no CUDA device in this pod.
