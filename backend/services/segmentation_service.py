import numpy as np
import uuid
from typing import List, Dict, Any, Optional
from models.depth import BuildingMeasurement, CalibrationConfig
from services.stage3_height_service import estimate_building_vertical_height

def measure_building_height(
    depth_map: np.ndarray,
    bbox: List[int], # [x, y, width, height]
    calibration: CalibrationConfig,
    building_name: str = "Structure",
    ground_truth_height_m: Optional[float] = None,
    view_geometry: str = "Oblique",
    intrinsics: Optional[Dict[str, float]] = None,
    camera_pose: Optional[Dict[str, float]] = None,
    ground_reference_bbox: Optional[List[int]] = None,
    roof_reference_bbox: Optional[List[int]] = None
) -> BuildingMeasurement:
    """
    Measures Stage 1 Camera Distance and delegates Stage 3 Building Vertical Height
    estimation to 3D metric unprojection & RANSAC ground plane fitting.

    Ground truth is strictly isolated and NEVER affects prediction.
    """
    img_h, img_w = depth_map.shape
    x, y, w, h = bbox

    # Clip coordinates to image boundaries
    x1 = max(0, min(img_w - 1, int(x)))
    y1 = max(0, min(img_h - 1, int(y)))
    x2 = max(0, min(img_w, int(x + w)))
    y2 = max(0, min(img_h, int(y + h)))

    if x2 <= x1 or y2 <= y1:
        x1, y1, x2, y2 = 0, 0, min(50, img_w), min(50, img_h)

    building_patch = depth_map[y1:y2, x1:x2]

    # Stage 1: Building Camera Depth Statistics in Metres
    d_p10 = round(float(np.percentile(building_patch, 10)), 2)
    d_p50 = round(float(np.percentile(building_patch, 50)), 2)
    d_p90 = round(float(np.percentile(building_patch, 90)), 2)
    d_min = round(float(np.min(building_patch)), 2)
    d_max = round(float(np.max(building_patch)), 2)
    cam_dist_m = d_p50

    rooftop_peak_rel = float(np.percentile(building_patch, 92))
    relative_depth_val = float(np.mean(building_patch))

    # Perimeter ring for relative height delta tracking
    pad_x = max(10, int(w * 0.35))
    pad_y = max(10, int(h * 0.35))

    ring_x1 = max(0, x1 - pad_x)
    ring_y1 = max(0, y1 - pad_y)
    ring_x2 = min(img_w, x2 + pad_x)
    ring_y2 = min(img_h, y2 + pad_y)

    surrounding_patch = depth_map[ring_y1:ring_y2, ring_x1:ring_x2]
    ground_base_rel = float(np.percentile(surrounding_patch, 10))
    rel_height = float(max(0.01, rooftop_peak_rel - ground_base_rel))

    # Stage 3: Delegate to True 3D Metric Building Vertical Height Estimation
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"[LIVE TRACE] measure_building_height invoked for structure '{building_name}', bbox={[x1, y1, x2 - x1, y2 - y1]}, view_geometry='{view_geometry}'")

    stage3_res = estimate_building_vertical_height(
        depth_m=depth_map,
        bbox=[x1, y1, x2 - x1, y2 - y1],
        intrinsics=intrinsics,
        camera_pose=camera_pose,
        ground_reference_bbox=ground_reference_bbox,
        roof_reference_bbox=roof_reference_bbox,
        view_geometry=view_geometry
    )

    logger.info(f"[LIVE TRACE] stage3_res for '{building_name}': height={stage3_res.get('estimated_height_m')} ground={stage3_res.get('ground_elevation_m')} roof={stage3_res.get('roof_elevation_m')} status={stage3_res.get('calculation_status')}")

    is_metric_available = stage3_res["height_available"]
    calibrated_height_m = stage3_res["estimated_height_m"]
    ground_elevation_m = stage3_res["ground_elevation_m"]
    roof_elevation_m = stage3_res["roof_elevation_m"]
    uncertainty_margin_m = stage3_res["height_uncertainty_m"]
    confidence_pct = stage3_res["confidence_pct"]
    measurement_mode = stage3_res["method"]
    calibration_note = stage3_res["reason_if_unavailable"] or f"Vertical height: {calibrated_height_m} m (Ground: {ground_elevation_m} m, Roof: {roof_elevation_m} m)."

    metric_status = (
        f"Estimated Vertical Height: {calibrated_height_m:.1f} m"
        if is_metric_available and calibrated_height_m is not None
        else f"Building Height: {stage3_res['reason_if_unavailable'] or 'Unavailable'}"
    )

    # Post-inference Ground Truth Validation Scoring (Validation ONLY)
    error_m = None
    relative_error_pct = None
    accuracy_pct = None

    if ground_truth_height_m and ground_truth_height_m > 0 and calibrated_height_m is not None:
        error_m = round(abs(calibrated_height_m - ground_truth_height_m), 2)
        relative_error_pct = round((error_m / ground_truth_height_m) * 100.0, 2)
        accuracy_pct = round(max(0.0, 100.0 - relative_error_pct), 2)

    calc_status = stage3_res.get("calculation_status", "UNAVAILABLE")
    if ground_truth_height_m and ground_truth_height_m > 0 and calibrated_height_m is not None:
        calc_status = "VALIDATION_AVAILABLE"

    pose_display = (
        "KNOWN"
        if stage3_res.get("camera_pose_available")
        else ("ESTIMATED (Ground Normal Alignment)" if stage3_res.get("camera_pose_status") != "Unavailable" else "UNAVAILABLE")
    )

    camera_param_status = {
        "focal_length": "KNOWN" if intrinsics and "fx" in intrinsics else "ESTIMATED",
        "sensor_width": "KNOWN" if calibration.sensor_width_mm else "UNAVAILABLE",
        "altitude": "KNOWN" if calibration.flight_altitude_m else "UNAVAILABLE",
        "view_angle": view_geometry.upper(),
        "camera_pose": pose_display
    }

    technical_derivation = stage3_res["technical_derivation"]
    technical_derivation.update({
        "camera_to_building_distance_m": cam_dist_m,
        "depth_p10_m": d_p10,
        "depth_p50_m": d_p50,
        "depth_p90_m": d_p90,
    })

    if is_metric_available and calibrated_height_m is not None:
        if stage3_res.get("camera_pose_available"):
            scale_recovery_method = "Metric 3D Geometry"
        else:
            scale_recovery_method = "Metric 3D Geometry · Estimated Vertical"
    else:
        scale_recovery_method = "Stage 3 Unavailable"

    return BuildingMeasurement(
        id=str(uuid.uuid4())[:8],
        name=building_name,
        bbox=[x1, y1, x2 - x1, y2 - y1],
        camera_to_building_distance_m=cam_dist_m,
        depth_p10_m=d_p10,
        depth_p50_m=d_p50,
        depth_p90_m=d_p90,
        depth_min_m=d_min,
        depth_max_m=d_max,
        relative_depth=round(relative_depth_val, 3),
        rooftop_peak_z_rel=round(rooftop_peak_rel, 3),
        ground_base_z_rel=round(ground_base_rel, 3),
        relative_height_unitless=round(rel_height, 3),
        calibrated_height_m=calibrated_height_m,
        ground_elevation_m=ground_elevation_m,
        roof_elevation_m=roof_elevation_m,
        height_fused_m=calibrated_height_m,
        ground_truth_height_m=ground_truth_height_m,
        error_m=error_m,
        relative_error_pct=relative_error_pct,
        accuracy_pct=accuracy_pct,
        uncertainty_margin_m=uncertainty_margin_m,
        confidence_pct=confidence_pct,
        is_metric_available=is_metric_available,
        measurement_mode=measurement_mode,
        calibration_note=calibration_note,
        scale_recovery_method=scale_recovery_method,
        ground_plane_method="RANSAC Ground Plane Fitting",
        metric_status=metric_status,
        stage_status=stage3_res.get("stage_status", "Stage 3 Under Geometric Validation"),
        calculation_status=calc_status,
        geometry_diagnostics=stage3_res.get("geometry_diagnostics", {}),
        camera_parameter_status=camera_param_status,
        technical_derivation=technical_derivation
    )
