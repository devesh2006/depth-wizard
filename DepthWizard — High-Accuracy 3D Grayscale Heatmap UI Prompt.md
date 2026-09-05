You are a senior **computer-vision, geospatial-AI, 3D graphics, UX/UI and full-stack engineer** building a competition-grade MVP for **SIH26175 — DepthWizard**.

Upgrade the existing DepthWizard application into a **highly polished, futuristic 3D geospatial visualization system** while preserving the project's core scientific integrity.

The application must visually communicate:

**SINGLE IMAGE → AI DEPTH → HEIGHT/DEPTH HEATMAP → 3D MESH → BUILDING ANALYSIS → MEASUREMENT → VALIDATION**

Do not create a fake visual demo. Every analytical value must originate from the actual processing pipeline or clearly be labelled as simulated/demo data.

---

# 1. VISUAL DIRECTION — DARK GRAYSCALE 3D

Create a premium **black / charcoal / graphite / grayscale-only** visual language.

The entire application should feel like a combination of:

- advanced satellite intelligence software
- scientific visualization platform
- futuristic geospatial command center
- Minecraft/Roblox-inspired 3D environment
- professional CAD / digital-twin software
- modern AI SaaS dashboard

Avoid colorful cyberpunk aesthetics.

Use primarily:

- #050505
- #0A0A0A
- #111111
- #171717
- #222222
- #303030
- #444444
- #666666
- white / near-white for important information

Use **soft grayscale gradients and luminance differences** rather than colorful gradients.

---

# 2. 3D HEAT-MAP STYLE

The primary 3D visualization should look like a **grayscale elevation/depth heat map represented as a 3D environment**.

Transform buildings and terrain into:

- 3D extruded structures
- voxel-like blocks
- low-poly surfaces
- triangulated meshes
- point clouds
- elevation surfaces
- wireframe overlays

The visual inspiration can resemble **Minecraft or Roblox-style 3D environments**, but make it more sophisticated and scientific.

Do NOT literally copy Minecraft or Roblox assets, logos, UI, characters, textures, or branding.

Instead create:

**“stylized voxel/low-poly geospatial digital twin.”**

Buildings should appear as recognizable 3D masses with accurate geometry derived from the reconstruction pipeline.

---

# 3. HEIGHT-BASED GRAYSCALE

Use luminance to represent depth/elevation.

Example:

Lowest surface
→ near-black

Low elevation
→ dark graphite

Medium elevation
→ gray

High elevation
→ light gray

Highest reconstructed structures
→ near-white

This should create a **3D grayscale heat-map effect**.

The height visualization must be based on actual depth/elevation values rather than random colors or random heights.

For example:

```text
Depth / Elevation
        ↓
Normalize
        ↓
Map value → grayscale intensity
        ↓
Apply to 3D geometry
```

The UI should include a small dynamic legend:

```text
ELEVATION

HIGH       ░░░░░░  35 m
           ▒▒▒▒▒▒
           ▓▓▓▓▓▓
LOW        ██████    0 m
```

When metric calibration is unavailable, replace meter labels with:

**RELATIVE DEPTH**

and do not imply that the values are meters.

---

# 4. 3D MESH

The reconstructed scene should have a **clean low-poly / triangulated mesh aesthetic**.

Show:

- triangular mesh topology
- building surfaces
- roof structures
- terrain
- elevation contours
- subtle wireframe
- point-cloud mode
- solid mesh mode
- heatmap mode
- hybrid mesh + wireframe mode

The mesh should not look like a generic smooth blob.

Preserve meaningful geometric structure from the actual depth map.

Apply:

- outlier removal
- depth clipping
- voxel downsampling
- optional smoothing
- mesh simplification
- surface reconstruction

while preserving the original point-cloud representation for inspection.

This follows the existing project requirement that single-image reconstruction may contain holes, distorted surfaces and missing geometry, so the system should expose the reconstruction rather than hide imperfections.

---

# 5. LIVE PARTICLES

Add subtle **live animated particles** in the 3D environment.

Particles should:

- slowly float
- move along depth/elevation contours
- occasionally pulse
- react subtly to camera movement
- create a sense of spatial depth
- remain extremely subtle

Opacity:

**approximately 30–50%**

Never allow particles to interfere with:

- building geometry
- measurement lines
- labels
- buttons
- data
- important UI

Particles should look like **scientific data points**, not stars.

---

# 6. DATA FLOW ANIMATION

When processing an image, visually animate the transformation:

```text
2D IMAGE
   ↓
DEPTH EXTRACTION
   ↓
DEPTH FIELD
   ↓
POINT CLOUD
   ↓
MESH
   ↓
HEIGHT MAP
   ↓
3D DIGITAL TWIN
```

Use animated scanning lines, particles and mesh construction effects.

For example:

1. Image appears.
2. A grayscale scan sweeps across it.
3. Depth points emerge.
4. Points rise into 3D.
5. Triangles connect.
6. Buildings become 3D structures.
7. Heatmap luminance is applied.
8. Measurement interface becomes available.

The animation should be smooth and cinematic but must never fake analytical processing.

---

# 7. ROBLOX / MINECRAFT-LIKE 3D EXPERIENCE

Create a **stylized block-based urban environment**.

Buildings should have:

- simplified geometric forms
- block-like masses
- flat or subtly faceted surfaces
- clean edges
- realistic relative proportions
- grayscale elevation shading

Terrain should appear as a continuous 3D surface.

Add subtle:

- grid lines
- contour lines
- coordinate markers
- elevation labels
- measurement guides

The result should visually feel like:

**“Minecraft-like 3D readability + scientific GIS accuracy + AI reconstruction.”**

Do NOT sacrifice geometry for aesthetics.

---

# 8. ACCURACY-FIRST AI PIPELINE

The visual design must never override the scientific pipeline.

Use:

**Depth Anything V2**

as the initial monocular depth backbone.

Architecture:

```text
Input Image
    ↓
Image Quality Analysis
    ↓
Scene/View Geometry Classification
    ↓
Depth Anything V2
    ↓
Raw Floating-Point Depth
    ↓
Depth Quality Assessment
    ↓
Geometric Reconstruction
    ↓
Point Cloud
    ↓
3D Mesh
    ↓
Calibration
    ↓
Building/Height Estimation
    ↓
Validation
```

The existing project specification explicitly distinguishes relative depth, metric depth and height/elevation. Preserve these distinctions throughout the UI and backend.

---

# 9. TRAINING / MODEL IMPROVEMENT

Do NOT claim that the system is “85% accurate” simply because the prompt says so.

Instead implement an **accuracy optimization target**:

**Target: ≥85% validation performance where the chosen metric and dataset justify that target.**

Train and evaluate using genuine aerial/satellite data.

Dataset structure:

```text
RGB Image
+
Depth / Height Ground Truth
+
Camera Metadata
+
GSD
+
Geospatial Metadata
```

Use:

```text
70% Training
15% Validation
15% Test
```

with geographic separation between train and test regions.

Prefer:

```text
Training Cities
      ↓
Validation City
      ↓
Unseen Test City
```

Never train and test on nearly identical neighboring tiles.

The model should optimize for:

- depth accuracy
- building-height accuracy
- robustness to different camera angles
- robustness to different image resolutions
- robustness to different urban environments
- generalization to unseen geographic areas

---

# 10. ACCURACY DASHBOARD

Create a highly visual **AI Accuracy / Validation panel**.

Show:

```text
MODEL VALIDATION

Scenes Evaluated       25
MAE                    X.XX m
RMSE                   X.XX m
Median Error           X.XX m
Mean Relative Error    X.XX %
Validation Score       XX.X %
```

The values must be calculated from actual validation data.

Never hard-code:

```text
85%
0.74 m
0.96 m
```

unless those numbers actually come from the current validation dataset.

The original project specifically requires actual measurements and states that fabricated accuracy values must not be used.

---

# 11. BUILDING SELECTION

When the user hovers over a reconstructed building:

- subtly illuminate its mesh
- show bounding outline
- display building ID
- show relative/metric height
- display uncertainty
- show reconstruction quality

On click:

```text
BUILDING ANALYSIS

Building ID: B-024

Estimated Height
18.7 m

Measurement Mode
Reference Calibrated

Confidence
Medium

Uncertainty
± X.X m

Reference
Known Building Height
```

If no metric calibration exists:

```text
BUILDING ANALYSIS

Metric Height
Unavailable

Relative Height
0.73

Reason
No metric reference available
```

This behavior matches the project's required calibration-safe measurement model.

---

# 12. CALIBRATION

Support four modes:

### Reference-Based

User enters a known reference height.

### Metadata-Assisted

Use:

- GSD
- camera altitude
- focal length
- sensor information
- terrain reference

### Terrain-Referenced

Use:

```text
Surface Elevation
-
Ground Elevation
=
Approximate Object Height
```

### Relative

If no metric reference exists:

```text
Metric Height Unavailable
Relative Depth Only
```

Never silently convert relative depth into meters.

---

# 13. UNCERTAINTY VISUALIZATION

Do not show a single height value as absolute truth.

Show:

```text
18.7 m
± 1.4 m
```

Visually represent uncertainty using a subtle transparent vertical measurement band.

For example:

```text
18.7 m
│
│  uncertainty region
│
17.3 m ─────
```

The uncertainty must be derived from the actual calibration/validation methodology where possible.

---

# 14. 3D MEASUREMENT TOOLS

Add professional measurement tools:

- vertical height ruler
- horizontal distance ruler
- point-to-point measurement
- ground reference line
- building footprint
- elevation marker
- contour measurement
- coordinate readout

Use thin grayscale lines with subtle glow.

Example:

```text
       ▲
       │
       │ 18.7 m
       │
       ▼
──────────────
 GROUND
```

---

# 15. LIVE HEATMAP

Make the heatmap dynamic.

When the user changes:

- camera angle
- depth threshold
- height range
- selected building
- visualization mode

the 3D grayscale heatmap should update smoothly.

Add controls:

```text
VISUALIZATION

● Depth
○ Elevation
○ Height
○ Confidence
○ Wireframe
○ Point Cloud
```

---

# 16. CONFIDENCE / QUALITY OVERLAY

Create a separate grayscale quality visualization.

High-quality regions:

lighter

Low-quality regions:

darker / transparent / hatched

Possible quality indicators:

- local depth consistency
- gradient consistency
- invalid pixels
- extreme depth concentration
- flat-depth detection
- image quality

Do not present this as fake probability.

The project specification specifically calls for a quality indicator rather than pretending monocular depth provides direct ground-truth probability.

---

# 17. NEAR-NADIR DETECTION

Detect whether the image is:

```text
OBLIQUE / PERSPECTIVE
```

or

```text
NEAR-NADIR
```

If near-nadir:

show a subtle warning:

**“Near-nadir imagery detected. Height estimates may have higher uncertainty.”**

Do not hide this limitation.

Near-nadir imagery has weaker geometric cues and building facades may not be visible.

---

# 18. THREE-PANEL INTERFACE

Create a synchronized:

```text
┌──────────────┬──────────────┬────────────────┐
│  ORIGINAL    │ DEPTH MAP    │ 3D DIGITAL     │
│  AERIAL      │ HEATMAP      │  TWIN          │
│              │              │                │
└──────────────┴──────────────┴────────────────┘
```

Use the same grayscale visual language across all three.

When a building is selected:

- highlight it in the RGB image
- highlight the corresponding depth region
- highlight the 3D building

This directly supports the existing Tri-Panel concept.

---

# 19. HERO 3D PRESENTATION

Create a cinematic hero scene for SIH judging.

Start from a top-down grayscale aerial image.

Then:

```text
IMAGE
 ↓
DEPTH
 ↓
POINTS
 ↓
MESH
 ↓
3D CITY
```

Slowly rotate the camera.

Fly between buildings.

Highlight one building.

Display:

```text
HEIGHT
18.7 m

UNCERTAINTY
± X.X m
```

Then transition to:

```text
VALIDATION

MAE
X.XX m

RMSE
X.XX m
```

Keep the entire sequence under approximately **3 minutes** for the live SIH demonstration.

---

# 20. UI DESIGN

Use a premium dark interface.

Avoid excessive cards.

Prefer:

- floating glass-like dark panels
- thin borders
- subtle shadows
- compact typography
- small technical labels
- monospaced numbers for measurements
- clean icons
- restrained animation

Opacity of decorative background elements:

**approximately 30–50%.**

The actual analytical UI should remain highly readable.

---

# 21. BACKGROUND

Create a subtle animated background consisting of:

- grayscale particles
- 3D points
- thin mesh connections
- topographic contour lines
- slow-moving data streams
- faint grid
- subtle volumetric fog

Everything should move slowly.

No bright neon.

No colorful particles.

No distracting animation.

The background should look like a **live 3D geospatial data field**.

---

# 22. PERFORMANCE

Use GPU acceleration where available.

Do not render millions of particles unnecessarily.

Use:

- instancing
- level of detail
- progressive mesh loading
- point-cloud downsampling
- frustum culling
- efficient Three.js buffers
- WebGL/WebGPU where appropriate

Maintain smooth interaction without compromising analytical geometry.

---

# 23. DATA INTEGRITY RULE

This is critical.

NEVER:

- invent building heights
- invent validation metrics
- hard-code accuracy
- use random scale factors
- convert relative depth directly into meters
- label simulated geometry as measured geometry
- claim LiDAR replacement
- claim survey-grade accuracy

If metric scale cannot be established, explicitly display:

**“Relative depth — metric scale unavailable.”**

This requirement is fundamental to the project's technical credibility.

---

# 24. DEMO DATA

Bundle 3–4 legitimate preprocessed hero scenes for offline SIH presentation.

Each scene should contain:

```text
Original Image
Raw Depth
Depth Visualization
Point Cloud
Mesh
Calibration
Building Measurements
Reference Data
Validation Results
```

All hero outputs must have been generated by the real pipeline beforehand.

The application must support:

```text
LIVE MODE
```

and

```text
DEMO MODE
```

If live inference fails during judging, Demo Mode should load legitimate precomputed outputs.

---

# 25. ACCURACY IMPROVEMENT LOOP

Implement a continuous research pipeline:

```text
Collect aerial data
       ↓
Generate / obtain ground truth
       ↓
Clean dataset
       ↓
Train
       ↓
Validate
       ↓
Evaluate unseen geographic regions
       ↓
Analyze errors
       ↓
Improve model
       ↓
Repeat
```

Track:

```text
Baseline Model
Fine-Tuned Model
MAE
RMSE
Median Error
Relative Error
Inference Time
GPU Memory
```

The application should make it easy to compare model versions.

---

# 26. FINAL EXPERIENCE

The finished application should feel like:

**“Google Earth + Minecraft-like 3D city + AI depth estimation + scientific heatmap + CAD measurement software.”**

But the underlying results must remain scientifically grounded.

The visual experience should be spectacular.

The analytical claims should be conservative.

The UI should make the technical pipeline immediately understandable to an SIH judge.

The primary message should be:

**“From one image to an interpretable depth-assisted 3D world — with calibration, uncertainty and measurable validation.”**

Do not optimize for flashy visuals at the expense of accuracy.

Optimize for:

**TECHNICAL CREDIBILITY + VISUAL IMPACT + REAL DATA + MEASURABLE VALIDATION + SMOOTH 3D INTERACTION.**

---

# 27. FINAL ACCEPTANCE CHECK

Before declaring the implementation complete, verify:

[ ] Real Depth Anything V2 inference works

[ ] Raw floating-point depth is preserved

[ ] Depth quality is calculated

[ ] Near-nadir scenes are detected

[ ] Point cloud is generated from depth + camera geometry

[ ] 3D mesh is generated from actual reconstruction

[ ] Grayscale elevation/depth heatmap is derived from actual values

[ ] 3D voxel/low-poly visualization works

[ ] Live particles and mesh animations are subtle and performant

[ ] Building selection works

[ ] Calibration works with legitimate references

[ ] Metric height is NOT shown without a valid scale

[ ] Uncertainty is displayed

[ ] Validation uses genuine reference data

[ ] MAE/RMSE are calculated automatically

[ ] Accuracy target of ≥85% is treated as a measurable research target, NOT a fabricated claim

[ ] Geographic train/test separation is respected

[ ] Offline hero demo works

[ ] 3–4 legitimate hero scenes are available

[ ] Three.js flythrough works

[ ] 2D → Depth → 3D synchronization works

[ ] No fake measurements exist

[ ] No fabricated accuracy values exist

[ ] Scientific disclaimer is visible

[ ] Complete workflow can be demonstrated in approximately 3 minutes

Finally, run the complete system, test every major feature, inspect the actual outputs, fix errors, and only then declare the MVP complete.