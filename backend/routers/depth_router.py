import time
import io
import uuid
import base64
import logging
import httpx
import numpy as np
from PIL import Image
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Response
from typing import List, Optional, Dict, Any

from models.depth import (
    ProcessImageRequest,
    DepthProcessResponse,
    CalibrationConfig,
    MeasureBuildingRequest,
    BuildingMeasurement,
    ValidationOverview,
    HeroSceneSummary,
    HeightEstimationRequest,
    HeightEstimationResponse,
    HeightValidationRequest,
    HeightValidationResponse,
    BenchmarkRequest,
    BenchmarkResponse,
    ModelBenchmarkResult
)
from services.depth_benchmark_service import run_zero_shot_benchmark
from services.image_analysis import analyze_image_quality
from services.depth_estimator import (
    create_depth_estimator,
    compute_depth_confidence,
    generate_depth_colormaps,
    compare_depth_models,
)
from services.geometry import reconstruct_3d_geometry, export_pointcloud_ply, export_mesh_obj
from services.calibration_service import compute_calibration_scale
from services.segmentation_service import measure_building_height
from services.stage3_height_service import estimate_building_vertical_height
from services.hero_dataset import (
    HERO_SCENES,
    get_hero_scene_summaries,
    get_reference_bbox_frac,
    frac_bbox_to_pixels,
)
from services.validation_service import build_validation_overview

router = APIRouter(prefix="/depth", tags=["Depth & 3D Reconstruction"])

logger = logging.getLogger(__name__)

# In-memory session cache for processed depth arrays and images
DEPTH_CACHE: Dict[str, Dict[str, Any]] = {}
# Hard ceiling on resident scenes so repeated requests cannot grow without bound.
MAX_CACHED_SCENES = 6
depth_model = create_depth_estimator()

SCIENTIFIC_DISCLAIMER = (
    "Depth and height estimates are dependent on image geometry, model generalization, calibration and "
    "reference data. Results should not be treated as survey-grade measurements without independent validation."
)

@router.get("/hero-scenes", response_model=List[HeroSceneSummary])
async def list_hero_scenes():
    """Returns list of preloaded Hero datasets for 3-minute SIH live judging."""
    return get_hero_scene_summaries()

@router.get("/hero-scene/{scene_id}", response_model=DepthProcessResponse)
async def get_hero_scene(scene_id: str):
    """Loads pre-cached or on-demand processed hero scene."""
    if scene_id not in HERO_SCENES:
        raise HTTPException(status_code=404, detail=f"Hero scene '{scene_id}' not found.")
        
    scene_info = HERO_SCENES[scene_id]
    
    # Process or load from cache
    req = ProcessImageRequest(
        image_url=scene_info["image_url"],
        image_type=scene_info["type"],
        scene_preset_id=scene_id,
        calibration=CalibrationConfig(**scene_info["default_calibration"])
    )
    return await process_image_pipeline(req)

@router.post("/process", response_model=DepthProcessResponse)
async def process_image_endpoint(request: ProcessImageRequest):
    """Executes the complete DepthWizard end-to-end computer vision & 3D pipeline."""
    return await process_image_pipeline(request)

@router.post("/upload", response_model=DepthProcessResponse)
async def upload_and_process(
    file: UploadFile = File(...),
    image_type: str = Form("aerial"),
    calibration_mode: str = Form("relative"),
    known_object_height: Optional[float] = Form(None)
):
    """Uploads an aerial/satellite/drone image and processes depth and 3D reconstruction."""
    # 1. Security & MIME validation
    if file.content_type not in ["image/jpeg", "image/png", "image/webp", "image/tiff"]:
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload JPEG, PNG, WEBP, or TIFF.")
        
    contents = await file.read()
    if len(contents) > 25 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Maximum allowed size is 25MB.")
        
    try:
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
        
    # Convert image to base64
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=90)
    b64_str = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
    
    calib = CalibrationConfig(
        mode=calibration_mode,
        known_object_height_m=known_object_height
    )
    
    req = ProcessImageRequest(
        image_base64=b64_str,
        image_type=image_type,
        calibration=calib
    )
    return await _execute_pipeline(pil_img, req, scene_name=file.filename or "Uploaded Aerial Survey")

@router.post("/calibrate", response_model=DepthProcessResponse)
async def calibrate_scene(
    scene_id: str,
    calibration: CalibrationConfig
):
    """Re-calibrates active scene with updated scale factor and re-evaluates building heights."""
    if scene_id not in DEPTH_CACHE:
        raise HTTPException(status_code=404, detail="Scene not found or expired. Please reprocess.")
        
    cache_item = DEPTH_CACHE[scene_id]
    depth = cache_item["depth"]
    pil_img = cache_item["image"]
    req = cache_item["request"]
    req.calibration = calibration
    
    return await _execute_pipeline(pil_img, req, scene_id=scene_id, precomputed_depth=depth, scene_name=cache_item["scene_name"])

@router.post("/measure-building", response_model=BuildingMeasurement)
async def measure_custom_building(request: MeasureBuildingRequest):
    """Calculates height for custom user-drawn bounding box on the image."""
    scene_id = request.scene_id
    if not scene_id or scene_id not in DEPTH_CACHE:
        raise HTTPException(status_code=404, detail="Active scene depth matrix not found. Please process an image first.")
        
    cache_item = DEPTH_CACHE[scene_id]
    depth = cache_item["depth"]
    calib = request.calibration or cache_item["calibration"]
    intrinsics = cache_item.get("intrinsics")
    view_geom = "Oblique"
    if "quality_report" in cache_item and hasattr(cache_item["quality_report"], "view_geometry"):
        view_geom = cache_item["quality_report"].view_geometry

    measurement = measure_building_height(
        depth_map=depth,
        bbox=request.bbox,
        calibration=calib,
        building_name=request.name or "Selected Structure",
        ground_truth_height_m=request.ground_truth_height_m,
        view_geometry=view_geom,
        intrinsics=intrinsics
    )
    return measurement

@router.get("/validation", response_model=ValidationOverview)
async def get_validation_data():
    """
    Scores LIVE pipeline predictions against registered reference heights.

    If no scene carries verified reference heights, returns data_available=False
    with the concrete dataset requirements - never fabricated error metrics.
    """
    scenes_with_reference = [
        (sid, s) for sid, s in HERO_SCENES.items()
        if any(b.get("ground_truth_height_m") is not None for b in s.get("sample_buildings", []))
    ]

    if not scenes_with_reference:
        # Short-circuit: nothing to score, so skip the expensive inference entirely.
        return build_validation_overview([])

    scene_results = []
    for scene_id, scene_info in scenes_with_reference:
        try:
            req = ProcessImageRequest(
                image_url=scene_info["image_url"],
                image_type=scene_info["type"],
                scene_preset_id=scene_id,
                calibration=CalibrationConfig(**scene_info["default_calibration"]),
            )
            result = await process_image_pipeline(req)
            scene_results.append((scene_id, result))
        except Exception as exc:
            logger.warning("Validation scene '%s' could not be processed: %s", scene_id, exc)
            continue

    return build_validation_overview(scene_results)

@router.get("/export-pointcloud/{scene_id}")
async def download_pointcloud(scene_id: str):
    """Exports ASCII PLY 3D point cloud file."""
    if scene_id not in DEPTH_CACHE:
        raise HTTPException(status_code=404, detail="Scene not found in cache.")
        
    cache_item = DEPTH_CACHE[scene_id]
    pc_data = cache_item["pointcloud"]
    
    pos = np.array(pc_data.positions).reshape((-1, 3))
    col = np.array(pc_data.colors).reshape((-1, 3))
    ply_str = export_pointcloud_ply(pos, col)
    
    return Response(
        content=ply_str,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename=depthwizard_{scene_id}_pointcloud.ply"}
    )

@router.get("/export-mesh/{scene_id}")
async def download_mesh(scene_id: str):
    """Exports Wavefront OBJ 3D height mesh file."""
    if scene_id not in DEPTH_CACHE:
        raise HTTPException(status_code=404, detail="Scene not found in cache.")
        
    cache_item = DEPTH_CACHE[scene_id]
    mesh_data = cache_item["mesh"]
    obj_str = export_mesh_obj(mesh_data)
    
    return Response(
        content=obj_str,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename=depthwizard_{scene_id}_mesh.obj"}
    )


@router.get("/probe-pixel")
async def probe_pixel(scene_id: str, x: int, y: int):
    """
    Returns pixel-level metric camera depth in metres at (x, y) coordinates.
    """
    if scene_id not in DEPTH_CACHE:
        raise HTTPException(status_code=404, detail="Scene depth map not found or expired.")

    cache = DEPTH_CACHE[scene_id]
    depth_m = cache["depth_m"]
    depth_rel = cache["depth_rel"]
    dh, dw = depth_m.shape

    px = max(0, min(dw - 1, x))
    py = max(0, min(dh - 1, y))

    val_m = round(float(depth_m[py, px]), 2)
    val_rel = round(float(depth_rel[py, px]), 3)

    return {
        "scene_id": scene_id,
        "x": px,
        "y": py,
        "metric_depth_m": val_m,
        "relative_depth": val_rel,
        "unit": "metres"
    }


@router.post("/benchmark-models")
async def benchmark_depth_models(scene_id: Optional[str] = None):
    """
    Executes depth model comparison benchmark comparing:
      1. Depth Anything V2 Metric Base
      2. Depth Anything V2 Metric Small
      3. Depth Anything V2 Relative Small
      4. Structural Geometric Baseline
    """
    target_img = None
    if scene_id and scene_id in DEPTH_CACHE:
        target_img = DEPTH_CACHE[scene_id]["image"]
    else:
        hero = HERO_SCENES["downtown_highrise"]
        async with httpx.AsyncClient(timeout=15.0) as client:
            res = await client.get(hero["image_url"])
            if res.status_code == 200:
                target_img = Image.open(io.BytesIO(res.content)).convert("RGB")

    if target_img is None:
        raise HTTPException(status_code=400, detail="Could not load benchmark image.")

    results = compare_depth_models(target_img, image_type="aerial")
    return {"benchmark_results": results}


@router.post("/benchmark", response_model=BenchmarkResponse)
async def run_aerial_metric_depth_benchmark(request: BenchmarkRequest):
    """
    Executes ZERO-SHOT AERIAL METRIC DEPTH MODEL BENCHMARK across selected pretrained models.
    Stage 3 geometry remains FROZEN and UNCHANGED.
    """
    pil_img = None
    if request.scene_id and request.scene_id in DEPTH_CACHE:
        pil_img = DEPTH_CACHE[request.scene_id]["image"]

    if pil_img is None and request.image_base64:
        try:
            b64_data = request.image_base64
            if "," in b64_data:
                b64_data = b64_data.split(",", 1)[1]
            raw_bytes = base64.b64decode(b64_data)
            pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")

    if pil_img is None and request.image_url:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                res = await client.get(request.image_url)
                if res.status_code == 200:
                    pil_img = Image.open(io.BytesIO(res.content)).convert("RGB")
        except Exception:
            pass

    if pil_img is None:
        hero = HERO_SCENES["downtown_highrise"]
        async with httpx.AsyncClient(timeout=15.0) as client:
            res = await client.get(hero["image_url"])
            if res.status_code == 200:
                pil_img = Image.open(io.BytesIO(res.content)).convert("RGB")

    if pil_img is None:
        raise HTTPException(status_code=400, detail="Could not load target image for depth benchmark.")

    val_points = [pt.model_dump() for pt in request.validation_points] if request.validation_points else None

    benchmark_data = run_zero_shot_benchmark(
        img=pil_img,
        selected_model_keys=request.selected_models,
        validation_points=val_points,
        probe_pixel=request.probe_pixel,
        selected_bbox=request.selected_bbox,
        image_type=request.image_type
    )

    return BenchmarkResponse(
        scene_id=request.scene_id,
        image_dimensions=benchmark_data["image_dimensions"],
        selected_models=benchmark_data["selected_models"],
        model_results=[ModelBenchmarkResult(**mr) for mr in benchmark_data["model_results"]],
        model_agreement=benchmark_data["model_agreement"],
        pixel_probe=benchmark_data["pixel_probe"],
        region_analysis=benchmark_data["region_analysis"],
        validation_summary=benchmark_data["validation_summary"],
        best_model_recommendation=benchmark_data["best_model_recommendation"]
    )


@router.post("/validate-depth")
async def validate_metric_depth(
    predicted_depth_m: float,
    ground_truth_depth_m: float
):
    """
    Evaluates depth prediction against ground truth entered by user AFTER inference.
    Formula:
      Absolute Error = |Predicted - GroundTruth|
      Relative Error = (Absolute Error / GroundTruth) * 100
      Accuracy = max(0, 100 - Relative Error)
    Ground truth is strictly isolated and NEVER affects model inference.
    """
    if ground_truth_depth_m <= 0:
        raise HTTPException(status_code=400, detail="Ground truth depth must be greater than zero.")

    abs_err = round(abs(predicted_depth_m - ground_truth_depth_m), 2)
    rel_err_pct = round((abs_err / ground_truth_depth_m) * 100.0, 2)
    accuracy_pct = round(max(0.0, 100.0 - rel_err_pct), 2)

    return {
        "predicted_depth_m": round(predicted_depth_m, 2),
        "ground_truth_depth_m": round(ground_truth_depth_m, 2),
        "absolute_error_m": abs_err,
        "relative_error_pct": rel_err_pct,
        "accuracy_pct": accuracy_pct,
        "evaluation_note": "Ground truth evaluated strictly after model inference."
    }


@router.post("/estimate-height", response_model=HeightEstimationResponse)
async def estimate_height_endpoint(request: HeightEstimationRequest):
    """
    Stage 3: Estimates building vertical height in metres using verified metric depth,
    pinhole 3D unprojection, and RANSAC ground-plane / roof-plane fitting.

    Ground truth is strictly isolated and NEVER influences estimation.
    """
    scene_id = request.scene_id
    if not scene_id or scene_id not in DEPTH_CACHE:
        raise HTTPException(status_code=404, detail="Active scene depth matrix not found. Please process an image first.")

    cache = DEPTH_CACHE[scene_id]
    depth_m = cache["depth_m"]
    intrinsics = cache.get("intrinsics")

    res = estimate_building_vertical_height(
        depth_m=depth_m,
        bbox=request.bbox,
        intrinsics=intrinsics,
        camera_pose=request.camera_pose,
        ground_reference_bbox=request.ground_reference_bbox if request.use_ground_reference else None,
        roof_reference_bbox=request.roof_reference_bbox if request.use_roof_reference else None
    )

    return HeightEstimationResponse(**res)


@router.post("/validate-height", response_model=HeightValidationResponse)
async def validate_height_endpoint(request: HeightValidationRequest):
    """
    Stage 4: Evaluates building vertical height prediction against independent ground truth.

    Ground truth is provided AFTER inference and has ZERO influence on estimation.
    """
    gt_m = request.ground_truth_height_m
    est_m = request.estimated_height_m

    if gt_m <= 0:
        raise HTTPException(status_code=400, detail="Ground truth height must be greater than zero.")

    if est_m is None or est_m <= 0:
        return HeightValidationResponse(
            ground_truth_height_m=gt_m,
            estimated_height_m=est_m,
            absolute_error_m=None,
            relative_error_percent=None,
            accuracy_percent=None,
            evaluation_note="Prediction unavailable. Accuracy is N/A."
        )

    abs_err = round(abs(est_m - gt_m), 2)
    rel_err_pct = round((abs_err / gt_m) * 100.0, 2)
    acc_pct = round(max(0.0, 100.0 - rel_err_pct), 2)

    return HeightValidationResponse(
        ground_truth_height_m=gt_m,
        estimated_height_m=est_m,
        absolute_error_m=abs_err,
        relative_error_percent=rel_err_pct,
        accuracy_percent=acc_pct,
        evaluation_note="Ground truth scored strictly after independent height estimation."
    )


async def process_image_pipeline(req: ProcessImageRequest) -> DepthProcessResponse:
    # 1. Fetch or decode image
    pil_img = None
    scene_name = "Aerial Scene"
    
    if req.scene_preset_id and req.scene_preset_id in HERO_SCENES:
        hero = HERO_SCENES[req.scene_preset_id]
        scene_name = hero["title"]
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                res = await client.get(hero["image_url"])
                if res.status_code == 200:
                    pil_img = Image.open(io.BytesIO(res.content)).convert("RGB")
        except Exception:
            pass
            
    if pil_img is None and req.image_url:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                res = await client.get(req.image_url)
                if res.status_code == 200:
                    pil_img = Image.open(io.BytesIO(res.content)).convert("RGB")
                    scene_name = "Remote Image"
        except Exception:
            pass
            
    if pil_img is None and req.image_base64:
        try:
            b64_data = req.image_base64
            if "," in b64_data:
                b64_data = b64_data.split(",", 1)[1]
            raw_bytes = base64.b64decode(b64_data)
            pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            scene_name = "User Uploaded Image"
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to decode base64 image: {str(e)}")
            
    if pil_img is None:
        hero = HERO_SCENES["downtown_highrise"]
        scene_name = hero["title"]
        async with httpx.AsyncClient(timeout=15.0) as client:
            res = await client.get(hero["image_url"])
            pil_img = Image.open(io.BytesIO(res.content)).convert("RGB")
            
    return await _execute_pipeline(pil_img, req, scene_name=scene_name)


async def _execute_pipeline(
    pil_img: Image.Image,
    req: ProcessImageRequest,
    scene_id: Optional[str] = None,
    precomputed_depth: Optional[np.ndarray] = None,
    scene_name: str = "Aerial Scene"
) -> DepthProcessResponse:
    t0 = time.time()
    
    # 1. Image Quality & View Geometry Analysis
    quality_report = analyze_image_quality(pil_img, image_type=req.image_type)
    t_qual = time.time()
    
    # 2. Metric & Relative Monocular Depth Estimation
    selected_estimator = create_depth_estimator(req.model_variant or "metric_base")
    if precomputed_depth is not None:
        depth_m = precomputed_depth
        lo, hi = float(np.percentile(depth_m, 1)), float(np.percentile(depth_m, 99))
        depth_rel = np.clip((depth_m - lo) / max(1e-6, hi - lo), 0.0, 1.0)
    else:
        depth_m, depth_rel = selected_estimator.estimate_depth(pil_img, image_type=req.image_type)
    t_depth = time.time()
    
    # 3. Depth Confidence Assessment
    confidence_report = compute_depth_confidence(depth_m, quality_report)
    
    # 4. Calibration Config
    raw_calib = req.calibration or CalibrationConfig(mode="relative")
    ref_rel_height: Optional[float] = None

    if raw_calib.mode == "reference" and req.scene_preset_id:
        ref_frac = get_reference_bbox_frac(req.scene_preset_id)
        if ref_frac:
            dh, dw = depth_m.shape
            ref_bbox = frac_bbox_to_pixels(ref_frac, dw, dh)
            ref_probe = measure_building_height(
                depth_map=depth_m,
                bbox=ref_bbox,
                calibration=CalibrationConfig(mode="relative"),
                building_name="Calibration Reference Probe",
            )
            ref_rel_height = ref_probe.relative_height_unitless

    calibration = compute_calibration_scale(raw_calib, depth_rel, ref_rel_height=ref_rel_height)
    
    # 5. Colormaps (generated from depth_rel for UI display)
    colormaps = generate_depth_colormaps(depth_rel)
    
    # 6. Camera Intrinsics
    dh, dw = depth_m.shape
    if req.camera_intrinsics:
        intrinsics = {
            "fx": float(req.camera_intrinsics.get("fx", max(dw, dh) * 1.2)),
            "fy": float(req.camera_intrinsics.get("fy", max(dw, dh) * 1.2)),
            "cx": float(req.camera_intrinsics.get("cx", dw / 2.0)),
            "cy": float(req.camera_intrinsics.get("cy", dh / 2.0)),
            "status": "KNOWN (EXIF)"
        }
    else:
        intrinsics = {
            "fx": round(max(dw, dh) * 1.2, 2),
            "fy": round(max(dw, dh) * 1.2, 2),
            "cx": round(dw / 2.0, 2),
            "cy": round(dh / 2.0, 2),
            "status": "ESTIMATED (Pinhole Model Assumption)"
        }

    # 7. 3D Geometric Reconstruction (Point Cloud & Mesh)
    pointcloud, mesh = reconstruct_3d_geometry(
        img=pil_img,
        depth=depth_m,
        scale_factor=1.0,
        intrinsics=intrinsics
    )
    t_geom = time.time()
    
    # 8. Building Metric Depth Extraction
    sample_buildings: List[BuildingMeasurement] = []
    dh, dw = depth_m.shape
    logger.info(f"[LIVE TRACE] _execute_pipeline invoked for scene '{scene_name}', shape=({dh}, {dw})")
    if req.scene_preset_id and req.scene_preset_id in HERO_SCENES:
        hero_bldgs = HERO_SCENES[req.scene_preset_id].get("sample_buildings", [])
        for hb in hero_bldgs:
            bbox_px = frac_bbox_to_pixels(hb["bbox_frac"], dw, dh)
            bm = measure_building_height(
                depth_map=depth_m,
                bbox=bbox_px,
                calibration=calibration,
                building_name=hb["name"],
                ground_truth_height_m=hb.get("ground_truth_height_m"),
                view_geometry=quality_report.view_geometry,
                intrinsics=intrinsics
            )
            bm.id = hb["id"]
            sample_buildings.append(bm)
    else:
        default_bboxes = [
            ([int(dw * 0.35), int(dh * 0.3), int(dw * 0.25), int(dh * 0.25)], "Central Structure"),
            ([int(dw * 0.15), int(dh * 0.4), int(dw * 0.2), int(dw * 0.2)], "West Complex"),
            ([int(dw * 0.65), int(dh * 0.35), int(dw * 0.2), int(dw * 0.22)], "East Wing")
        ]
        for bbox, name in default_bboxes:
            bm = measure_building_height(
                depth_map=depth_m,
                bbox=bbox,
                calibration=calibration,
                building_name=name,
                view_geometry=quality_report.view_geometry,
                intrinsics=intrinsics
            )
            sample_buildings.append(bm)
    logger.info(f"[LIVE TRACE] _execute_pipeline finished. Generated {len(sample_buildings)} building measurements.")
            
    # Statistics
    metric_stats = {
        "min_m": round(float(np.min(depth_m)), 2),
        "max_m": round(float(np.max(depth_m)), 2),
        "mean_m": round(float(np.mean(depth_m)), 2),
        "median_m": round(float(np.median(depth_m)), 2),
        "valid_pixel_pct": round(float(np.mean((depth_m > 0.1) & (~np.isnan(depth_m))) * 100.0), 1),
        "unit": "metres"
    }

    raw_stats = {
        "min": round(float(np.min(depth_rel)), 4),
        "max": round(float(np.max(depth_rel)), 4),
        "mean": round(float(np.mean(depth_rel)), 4),
        "shape": list(depth_m.shape)
    }
    
    # Prepare image data URL if not remote URL
    buf = io.BytesIO()
    prev_img = pil_img.copy()
    if max(prev_img.size) > 1024:
        prev_img.thumbnail((1024, 1024), Image.Resampling.BILINEAR)
    prev_img.save(buf, format="JPEG", quality=85)
    image_data_url = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
    
    sid = scene_id or (req.scene_preset_id or str(uuid.uuid4())[:8])
    
    cache_img = pil_img
    if cache_img.size != (depth_m.shape[1], depth_m.shape[0]):
        cache_img = pil_img.resize((depth_m.shape[1], depth_m.shape[0]), Image.Resampling.BILINEAR)

    DEPTH_CACHE[sid] = {
        "scene_name": scene_name,
        "image": cache_img,
        "depth": depth_m,
        "depth_m": depth_m,
        "depth_rel": depth_rel,
        "request": req,
        "calibration": calibration,
        "pointcloud": pointcloud,
        "mesh": mesh
    }

    if len(DEPTH_CACHE) > MAX_CACHED_SCENES:
        for stale_key in list(DEPTH_CACHE.keys())[:-MAX_CACHED_SCENES]:
            DEPTH_CACHE.pop(stale_key, None)
    
    timing = {
        "preprocessing_ms": round((t_qual - t0) * 1000.0, 1),
        "depth_inference_ms": round((t_depth - t_qual) * 1000.0, 1),
        "geometry_reconstruction_ms": round((t_geom - t_depth) * 1000.0, 1),
        "total_ms": round((time.time() - t0) * 1000.0, 1)
    }
    
    return DepthProcessResponse(
        id=sid,
        scene_name=scene_name,
        image_type=req.image_type,
        depth_backend=getattr(selected_estimator, "backend_name", "unknown"),
        depth_backend_is_neural=bool(getattr(selected_estimator, "is_neural", False)),
        image_url=req.image_url or image_data_url,
        depth_map_url=colormaps["turbo"],
        depth_colormap_turbo_url=colormaps["turbo"],
        depth_colormap_viridis_url=colormaps["viridis"],
        depth_colormap_magma_url=colormaps["magma"],
        raw_depth_stats=raw_stats,
        metric_depth_stats=metric_stats,
        camera_intrinsics=intrinsics,
        depth_disclaimer="Metric depth produced by pretrained outdoor metric-depth model.",
        quality_report=quality_report,
        confidence_report=confidence_report,
        pointcloud=pointcloud,
        mesh=mesh,
        calibration=calibration,
        sample_buildings=sample_buildings,
        timing_ms=timing,
        scientific_disclaimer=SCIENTIFIC_DISCLAIMER
    )
