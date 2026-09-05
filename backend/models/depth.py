from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime

class ImageQualityReport(BaseModel):
    resolution: List[int] = Field(description="[Width, Height] in pixels")
    aspect_ratio: float
    sharpness_score: float = Field(description="Laplacian variance score")
    sharpness_label: str = Field(description="Excellent, Good, Fair, or Blurry")
    brightness_mean: float = Field(description="Mean grayscale brightness 0-255")
    brightness_label: str = Field(description="Optimal, Overexposed, or Underexposed")
    contrast_score: float = Field(description="RMS contrast")
    contrast_label: str = Field(description="High, Optimal, or Low")
    saturation_mean: float = Field(description="Mean HSV saturation 0-1")
    cloud_shadow_risk: str = Field(description="Low, Moderate, or High")
    view_geometry: str = Field(description="Oblique (25-45°), Near-Nadir (75-90°), or Low-Altitude Drone")
    nadir_warning: bool = Field(description="True if near-nadir angle with limited vertical facade cues")
    image_quality_score: float = Field(default=80.0, description="Visual image quality score 0-100")
    depth_quality_score: float = Field(default=80.0, description="Monocular depth quality score 0-100")
    geometry_quality_score: float = Field(default=75.0, description="Geometric structure & view angle quality 0-100")
    metric_calibration_quality_score: float = Field(default=20.0, description="Metric calibration readiness score 0-100")
    height_suitability_score: float = Field(default=30.0, description="Suitability score specifically for vertical height estimation 0-100")
    height_suitability_label: str = Field(default="Low", description="High, Medium, or Low")
    height_suitability_reason: str = Field(default="Low observability for metric height — oblique imagery recommended.", description="Detailed explanation of height observability")
    suitability_score: float = Field(description="Overall depth suitability 0-100")
    suitability_label: str = Field(description="High, Medium, or Low")


class DepthConfidenceReport(BaseModel):
    overall_confidence: str = Field(description="HIGH, MEDIUM, or LOW")
    confidence_score: float = Field(description="0-100 quality confidence score")
    local_gradient_consistency: float
    edge_alignment_score: float
    flat_depth_detected: bool
    invalid_pixel_pct: float
    reasons: List[str] = Field(default_factory=list)

class CalibrationConfig(BaseModel):
    mode: str = Field(default="relative", description="reference, metadata, terrain, or relative")
    known_object_name: Optional[str] = None
    known_object_height_m: Optional[float] = None
    ref_pixel_coords: Optional[List[int]] = None  # [x, y, w, h]
    gsd_cm_per_pixel: Optional[float] = None
    flight_altitude_m: Optional[float] = None
    focal_length_mm: Optional[float] = None
    sensor_width_mm: Optional[float] = None
    scale_factor_m_per_unit: Optional[float] = 1.0
    ground_baseline_z: Optional[float] = 0.0
    is_calibrated: bool = False
    calibration_description: Optional[str] = "Relative depth — metric scale unavailable."

class BuildingMeasurement(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    bbox: List[int] = Field(description="[x, y, width, height] in image coordinates")
    polygon: Optional[List[List[int]]] = None
    camera_to_building_distance_m: Optional[float] = Field(default=None, description="Estimated camera-to-building metric distance in metres")
    depth_p10_m: Optional[float] = Field(default=None, description="10th percentile building depth in metres")
    depth_p50_m: Optional[float] = Field(default=None, description="50th percentile (median) building depth in metres")
    depth_p90_m: Optional[float] = Field(default=None, description="90th percentile building depth in metres")
    depth_min_m: Optional[float] = Field(default=None, description="Minimum building depth in metres")
    depth_max_m: Optional[float] = Field(default=None, description="Maximum building depth in metres")
    relative_depth: float = Field(default=0.0, description="Raw relative depth value at building top")
    rooftop_peak_z_rel: float
    ground_base_z_rel: float
    relative_height_unitless: float
    calibrated_height_m: Optional[float] = Field(default=None, description="Building vertical height in metres (Stage 3 3D unprojection)")
    ground_elevation_m: Optional[float] = Field(default=None, description="Terrain base ground elevation in metres")
    roof_elevation_m: Optional[float] = Field(default=None, description="Building roof peak elevation in metres")
    height_fused_m: Optional[float] = None
    height_depth_cue_m: Optional[float] = None
    height_geom_cue_m: Optional[float] = None
    height_shadow_cue_m: Optional[float] = None
    height_ref_cue_m: Optional[float] = None
    ground_truth_height_m: Optional[float] = None
    error_m: Optional[float] = None
    relative_error_pct: Optional[float] = None
    accuracy_pct: Optional[float] = None
    uncertainty_margin_m: Optional[float] = None
    confidence_pct: float
    is_metric_available: bool
    measurement_mode: str
    calibration_note: str
    scale_recovery_method: Optional[str] = "Stage 3 Metric 3D Unprojection"
    ground_plane_method: Optional[str] = "RANSAC / Perimeter Ring Datum"
    metric_status: str = "Relative depth — metric scale unavailable."
    stage_status: str = "Stage 3 Under Geometric Validation"
    calculation_status: str = Field(default="UNAVAILABLE", description="UNAVAILABLE, GEOMETRY_READY, VERTICAL_REFERENCE_READY, HEIGHT_COMPUTED, or VALIDATION_AVAILABLE")
    geometry_diagnostics: Dict[str, Any] = Field(default_factory=dict)
    camera_parameter_status: Dict[str, str] = Field(
        default_factory=lambda: {
            "focal_length": "UNAVAILABLE",
            "sensor_width": "UNAVAILABLE",
            "altitude": "UNAVAILABLE",
            "view_angle": "ESTIMATED"
        }
    )
    technical_derivation: Dict[str, Any] = Field(default_factory=dict)

class HeightEstimationRequest(BaseModel):
    scene_id: Optional[str] = None
    bbox: List[int] = Field(description="[x, y, width, height]")
    camera_pose: Optional[Dict[str, float]] = Field(default=None, description="pitch_deg, roll_deg, yaw_deg")
    ground_reference_bbox: Optional[List[int]] = None
    roof_reference_bbox: Optional[List[int]] = None
    use_ground_reference: bool = False
    use_roof_reference: bool = False

class HeightEstimationResponse(BaseModel):
    height_available: bool
    estimated_height_m: Optional[float] = None
    ground_elevation_m: Optional[float] = None
    roof_elevation_m: Optional[float] = None
    height_uncertainty_m: Optional[float] = None
    confidence: float
    confidence_pct: float
    method: str
    camera_pose_available: bool
    camera_pose_status: str = "Unavailable"
    vertical_reference: str = "Estimated Ground-Plane Normal"
    intrinsics_source: str
    ground_plane_inlier_ratio: float
    roof_plane_inlier_ratio: float
    stage_status: str = "Stage 3 Under Geometric Validation"
    calculation_status: str = Field(default="UNAVAILABLE", description="UNAVAILABLE, GEOMETRY_READY, VERTICAL_REFERENCE_READY, HEIGHT_COMPUTED, or VALIDATION_AVAILABLE")
    reason_if_unavailable: Optional[str] = None
    geometry_diagnostics: Dict[str, Any] = Field(default_factory=dict)
    technical_derivation: Dict[str, Any] = Field(default_factory=dict)

class HeightValidationRequest(BaseModel):
    estimated_height_m: Optional[float] = Field(default=None, description="Estimated building vertical height in metres")
    ground_truth_height_m: float = Field(description="Independent surveyed ground truth height in metres")

class HeightValidationResponse(BaseModel):
    ground_truth_height_m: float
    estimated_height_m: Optional[float] = None
    absolute_error_m: Optional[float] = None
    relative_error_percent: Optional[float] = None
    accuracy_percent: Optional[float] = None
    evaluation_note: str

class PointCloudData(BaseModel):
    points_count: int
    positions: List[float] = Field(description="Flattened camera-space [x, y, z, ...]")
    colors: List[float] = Field(description="Flattened [r, g, b, ...]")
    bounds: Dict[str, float]

class MeshHeightfieldData(BaseModel):
    grid_width: int
    grid_height: int
    heights: List[float] = Field(description="Flattened 2D grid height values")
    colors: List[float] = Field(description="Flattened 2D grid RGB color values")

class ProcessImageRequest(BaseModel):
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    image_type: str = Field(default="aerial", description="satellite, aerial, or drone")
    scene_preset_id: Optional[str] = None
    calibration: Optional[CalibrationConfig] = None
    camera_intrinsics: Optional[Dict[str, float]] = None
    model_variant: Optional[str] = Field(default="metric_base", description="metric_base, metric_small, relative_small, or structural")

class DepthProcessResponse(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    scene_name: str
    image_type: str
    depth_backend: str = Field(description="Name of the depth engine that actually produced this result")
    depth_backend_is_neural: bool = Field(description="True only if real learned weights performed the inference")
    image_url: str
    depth_map_url: str
    depth_colormap_turbo_url: str
    depth_colormap_viridis_url: str
    depth_colormap_magma_url: str
    raw_depth_stats: Dict[str, Any]
    metric_depth_stats: Dict[str, Any] = Field(default_factory=dict, description="Min, Max, Mean, Median depth in metres")
    camera_intrinsics: Dict[str, Any] = Field(default_factory=dict, description="fx, fy, cx, cy, status")
    depth_disclaimer: str = Field(default="Metric depth produced by pretrained outdoor metric-depth model.")
    quality_report: ImageQualityReport
    confidence_report: DepthConfidenceReport
    pointcloud: PointCloudData
    mesh: MeshHeightfieldData
    calibration: CalibrationConfig
    sample_buildings: List[BuildingMeasurement]
    timing_ms: Dict[str, float]
    scientific_disclaimer: str

class MeasureBuildingRequest(BaseModel):
    scene_id: Optional[str] = None
    bbox: List[int] = Field(description="[x, y, width, height]")
    calibration: Optional[CalibrationConfig] = None
    depth_matrix_id: Optional[str] = None
    name: Optional[str] = "Target Structure"
    ground_truth_height_m: Optional[float] = Field(default=None, description="Optional surveyed ground-truth height for error scoring")

class ValidationBenchmark(BaseModel):
    scene_id: str
    scene_name: str
    sensor_type: str
    lidar_source: str
    buildings_tested: int
    mae_m: float
    rmse_m: float
    median_err_m: float
    mean_relative_error_pct: float
    pct_within_2m: float = 0.0
    pct_within_5m: float = 0.0
    pct_within_10m: float = 0.0
    sample_measurements: List[Dict[str, Any]]

class ValidationOverview(BaseModel):
    data_available: bool = Field(
        description="False when no verified reference dataset is registered; metrics are then meaningless"
    )
    unavailable_reason: Optional[str] = Field(
        default=None, description="Why validation cannot be computed"
    )
    requirements: List[str] = Field(
        default_factory=list, description="What is required to populate real validation"
    )
    total_scenes_evaluated: int
    total_structures_measured: int
    overall_mae_m: Optional[float] = None
    overall_rmse_m: Optional[float] = None
    overall_median_err_m: Optional[float] = None
    overall_mre_pct: Optional[float] = None
    overall_pct_within_2m: Optional[float] = None
    overall_pct_within_5m: Optional[float] = None
    overall_pct_within_10m: Optional[float] = None
    benchmarks: List[ValidationBenchmark]
    scatter_points: List[Dict[str, Any]]

class HeroSceneSummary(BaseModel):
    id: str
    title: str
    type: str
    description: str
    gt_benchmark_mae: str
    image_url: str
    view_angle: str
    sensor: str


class ValidationPoint(BaseModel):
    x: int = Field(description="Pixel X coordinate")
    y: int = Field(description="Pixel Y coordinate")
    measured_depth_m: float = Field(description="Actual measured camera-to-surface distance in metres")
    label: Optional[str] = Field(default=None, description="Optional point label (e.g. Ground Control 1)")


class ModelBenchmarkResult(BaseModel):
    model_key: str
    model_name: str
    status: str = Field(description="SUCCESS, UNAVAILABLE, or INFERENCE_FAILED")
    device: str = "cpu"
    device_info: Dict[str, Any] = Field(default_factory=dict)
    inference_time_ms: float
    preprocessing_time_ms: float = 0.0
    postprocessing_time_ms: float = 0.0
    total_time_ms: float = 0.0
    depth_min_m: Optional[float] = None
    depth_max_m: Optional[float] = None
    depth_mean_m: Optional[float] = None
    depth_median_m: Optional[float] = None
    depth_p10_m: Optional[float] = None
    depth_p25_m: Optional[float] = None
    depth_p50_m: Optional[float] = None
    depth_p75_m: Optional[float] = None
    depth_p90_m: Optional[float] = None
    depth_std_m: Optional[float] = None
    depth_spread_m: Optional[float] = None
    p10_p90_spread_m: Optional[float] = None
    valid_pixel_ratio: Optional[float] = None
    is_compressed: bool = False
    compression_note: Optional[str] = None
    error_reason: Optional[str] = None
    depth_colormap_url: Optional[str] = None
    building_stats: List[Dict[str, Any]] = Field(default_factory=list)
    stage1_validation_metrics: Optional[Dict[str, Any]] = None
    stage3_stress_test: Optional[Dict[str, Any]] = Field(default=None, description="FROZEN Stage 3 predicted height + Stage 4 validation against 175m")


class BenchmarkRequest(BaseModel):
    scene_id: Optional[str] = None
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    image_type: str = Field(default="aerial", description="aerial, drone, or satellite")
    selected_models: List[str] = Field(
        default=["da_v2_base", "da_v2_large", "unidepth_v2_l", "metric3dv2"],
        description="Keys of models to benchmark"
    )
    validation_points: Optional[List[ValidationPoint]] = Field(default_factory=list)
    probe_pixel: Optional[List[int]] = Field(default=None, description="[x, y] pixel coordinates to probe across models")
    selected_bbox: Optional[List[int]] = Field(default=None, description="[x, y, w, h] region to analyze across models")


class BenchmarkResponse(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    scene_id: Optional[str] = None
    image_dimensions: List[int]
    selected_models: List[str]
    model_results: List[ModelBenchmarkResult]
    model_agreement: Dict[str, Any] = Field(default_factory=dict)
    pixel_probe: Optional[Dict[str, Any]] = None
    region_analysis: Optional[Dict[str, Any]] = None
    validation_summary: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    best_model_recommendation: Dict[str, Any] = Field(default_factory=dict)

