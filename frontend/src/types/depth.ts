export interface ImageQualityReport {
  resolution: [number, number];
  aspect_ratio: number;
  sharpness_score: number;
  sharpness_label: "Excellent" | "Good" | "Fair" | "Blurry";
  brightness_mean: number;
  brightness_label: "Optimal" | "Overexposed" | "Underexposed";
  contrast_score: number;
  contrast_label: "High" | "Optimal" | "Low";
  saturation_mean: number;
  cloud_shadow_risk: "Low" | "Moderate" | "High";
  view_geometry: string;
  nadir_warning: boolean;
  suitability_score: number;
  suitability_label: "High" | "Medium" | "Low";
}

export interface DepthConfidenceReport {
  overall_confidence: "HIGH" | "MEDIUM" | "LOW";
  confidence_score: number;
  local_gradient_consistency: number;
  edge_alignment_score: number;
  flat_depth_detected: boolean;
  invalid_pixel_pct: number;
  reasons: string[];
}

export interface CalibrationConfig {
  mode: "reference" | "metadata" | "terrain" | "relative";
  known_object_name?: string | null;
  known_object_height_m?: number | null;
  ref_pixel_coords?: number[] | null;
  gsd_cm_per_pixel?: number | null;
  flight_altitude_m?: number | null;
  focal_length_mm?: number | null;
  sensor_width_mm?: number | null;
  scale_factor_m_per_unit?: number | null;
  ground_baseline_z?: number | null;
  is_calibrated: boolean;
  calibration_description?: string | null;
}

export interface BuildingMeasurement {
  id: string;
  name: string;
  bbox: [number, number, number, number]; // [x, y, width, height]
  polygon?: [number, number][] | null;
  camera_to_building_distance_m?: number | null;
  depth_p10_m?: number | null;
  depth_p50_m?: number | null;
  depth_p90_m?: number | null;
  depth_min_m?: number | null;
  depth_max_m?: number | null;
  relative_depth?: number;
  rooftop_peak_z_rel: number;
  ground_base_z_rel: number;
  relative_height_unitless: number;
  calibrated_height_m?: number | null;
  ground_elevation_m?: number | null;
  roof_elevation_m?: number | null;
  height_fused_m?: number | null;
  height_depth_cue_m?: number | null;
  height_geom_cue_m?: number | null;
  height_shadow_cue_m?: number | null;
  height_ref_cue_m?: number | null;
  ground_truth_height_m?: number | null;
  error_m?: number | null;
  relative_error_pct?: number | null;
  accuracy_pct?: number | null;
  uncertainty_margin_m?: number | null;
  confidence_pct: number;
  is_metric_available: boolean;
  measurement_mode: string;
  calibration_note: string;
  scale_recovery_method?: string;
  ground_plane_method?: string;
  metric_status?: string;
  stage_status?: string;
  calculation_status?: string;
  geometry_diagnostics?: Record<string, any>;
  camera_parameter_status?: Record<string, string>;
  technical_derivation?: Record<string, any>;
}

export interface PointCloudData {
  points_count: number;
  positions: number[]; // flattened [x, y, z, ...]
  colors: number[];    // flattened [r, g, b, ...]
  bounds: {
    min_x: number;
    max_x: number;
    min_y: number;
    max_y: number;
    min_z: number;
    max_z: number;
  };
}

export interface MeshHeightfieldData {
  grid_width: number;
  grid_height: number;
  heights: number[];
  colors: number[];
}

export interface DepthProcessResponse {
  id: string;
  timestamp: string;
  scene_name: string;
  image_type: string;
  depth_backend: string;
  depth_backend_is_neural: boolean;
  image_url: string;
  depth_map_url: string;
  depth_colormap_turbo_url: string;
  depth_colormap_viridis_url: string;
  depth_colormap_magma_url: string;

  raw_depth_stats: {
    min: number;
    max: number;
    mean: number;
    std?: number;
    shape: [number, number];
  };
  metric_depth_stats?: {
    min_m: number;
    max_m: number;
    mean_m: number;
    median_m: number;
    valid_pixel_pct: number;
    unit: string;
  };
  camera_intrinsics?: {
    fx: number;
    fy: number;
    cx: number;
    cy: number;
    status: string;
  };
  depth_disclaimer?: string;
  quality_report: ImageQualityReport;
  confidence_report: DepthConfidenceReport;
  pointcloud: PointCloudData;
  mesh: MeshHeightfieldData;
  calibration: CalibrationConfig;
  sample_buildings: BuildingMeasurement[];
  timing_ms: {
    preprocessing_ms: number;
    depth_inference_ms: number;
    geometry_reconstruction_ms: number;
    total_ms: number;
  };
  scientific_disclaimer: string;
}

export interface ProcessImageRequest {
  image_url?: string | null;
  image_base64?: string | null;
  image_type: "satellite" | "aerial" | "drone";
  scene_preset_id?: string | null;
  calibration?: CalibrationConfig | null;
  camera_intrinsics?: Record<string, number> | null;
  model_variant?: "metric_base" | "metric_small" | "relative_small" | "structural" | null;
}

export interface MeasureBuildingRequest {
  scene_id?: string | null;
  bbox: [number, number, number, number];
  calibration?: CalibrationConfig | null;
  depth_matrix_id?: string | null;
  name?: string | null;
  ground_truth_height_m?: number | null;
}

export interface ValidationMeasurement {
  building_id: string;
  building_name: string;
  gt_height_m: number;
  pred_height_m: number;
  error_m: number;
  rel_error_pct: number;
  confidence_pct: number;
}

export interface ValidationBenchmark {
  scene_id: string;
  scene_name: string;
  sensor_type: string;
  lidar_source: string;
  buildings_tested: number;
  mae_m: number;
  rmse_m: number;
  median_err_m: number;
  mean_relative_error_pct: number;
  pct_within_2m?: number;
  pct_within_5m?: number;
  pct_within_10m?: number;
  sample_measurements: ValidationMeasurement[];
}

export interface ValidationOverview {
  data_available: boolean;
  unavailable_reason?: string | null;
  requirements: string[];
  total_scenes_evaluated: number;
  total_structures_measured: number;
  overall_mae_m?: number | null;
  overall_rmse_m?: number | null;
  overall_median_err_m?: number | null;
  overall_mre_pct?: number | null;
  overall_pct_within_2m?: number | null;
  overall_pct_within_5m?: number | null;
  overall_pct_within_10m?: number | null;
  benchmarks: ValidationBenchmark[];
  scatter_points: {
    scene: string;
    scene_id: string;
    building: string;
    gt_height: number;
    pred_height: number;
    error: number;
    sensor: string;
  }[];
}


export interface HeroSceneSummary {
  id: string;
  title: string;
  type: string;
  description: string;
  gt_benchmark_mae: string;
  image_url: string;
  view_angle: string;
  sensor: string;
}

export interface ValidationPoint {
  x: number;
  y: number;
  measured_depth_m: number;
  label?: string | null;
}

export interface ModelBenchmarkResult {
  model_key: string;
  model_name: string;
  status: "SUCCESS" | "UNAVAILABLE" | "INFERENCE_FAILED";
  device: string;
  device_info: {
    device?: string;
    param_count?: string;
    precision?: string;
    [key: string]: any;
  };
  inference_time_ms: number;
  preprocessing_time_ms: number;
  postprocessing_time_ms: number;
  total_time_ms: number;
  depth_min_m?: number | null;
  depth_max_m?: number | null;
  depth_mean_m?: number | null;
  depth_median_m?: number | null;
  depth_p10_m?: number | null;
  depth_p25_m?: number | null;
  depth_p50_m?: number | null;
  depth_p75_m?: number | null;
  depth_p90_m?: number | null;
  depth_std_m?: number | null;
  depth_spread_m?: number | null;
  p10_p90_spread_m?: number | null;
  valid_pixel_ratio?: number | null;
  is_compressed?: boolean;
  compression_note?: string | null;
  error_reason?: string | null;
  depth_colormap_url?: string | null;
  building_stats?: {
    building_name: string;
    bbox: number[];
    depth_p10_m: number;
    depth_p50_m: number;
    depth_p90_m: number;
    mean_depth_m: number;
    std_depth_m: number;
    stage3_predicted_height_m?: number | null;
    stage3_ground_elevation_m?: number | null;
    stage3_roof_elevation_m?: number | null;
    stage3_status?: string;
  }[];
  stage1_validation_metrics?: Record<string, any> | null;
  stage3_stress_test?: {
    building_name: string;
    metric_depth_p50_m: number;
    predicted_height_m?: number | null;
    ground_truth_height_m: number;
    absolute_error_m?: number | null;
    relative_error_pct?: number | null;
    accuracy_pct?: number | null;
    stage3_status?: string;
    vertical_reference?: string;
  } | null;
}

export interface BenchmarkRequest {
  scene_id?: string | null;
  image_url?: string | null;
  image_base64?: string | null;
  image_type?: string;
  selected_models: string[];
  validation_points?: ValidationPoint[];
  probe_pixel?: [number, number] | null;
  selected_bbox?: [number, number, number, number] | null;
}

export interface BenchmarkResponse {
  timestamp: string;
  scene_id?: string | null;
  image_dimensions: [number, number];
  selected_models: string[];
  model_results: ModelBenchmarkResult[];
  model_agreement: {
    agreement_rating: string;
    mean_p50_m?: number | null;
    std_p50_m?: number | null;
    spread_p50_m?: number | null;
    relative_std_pct?: number | null;
    models_compared_count?: number;
    note?: string;
  };
  pixel_probe?: {
    x: number;
    y: number;
    model_predictions: {
      model_key: string;
      model_name: string;
      depth_m: number;
    }[];
  } | null;
  region_analysis?: {
    bbox: number[];
    model_predictions: {
      model_key: string;
      model_name: string;
      min_m: number;
      max_m: number;
      mean_m: number;
      median_m: number;
      p10_m: number;
      p25_m: number;
      p50_m: number;
      p75_m: number;
      p90_m: number;
      std_m: number;
    }[];
  } | null;
  validation_summary?: {
    model_key: string;
    model_name: string;
    valid_samples_count: number;
    mae_m?: number | null;
    rmse_m?: number | null;
    mean_relative_error_pct?: number | null;
    median_relative_error_pct?: number | null;
    point_evaluations: {
      x: number;
      y: number;
      label: string;
      predicted_depth_m: number;
      measured_depth_m: number;
      abs_error_m: number;
      rel_error_pct: number;
    }[];
  }[];
  best_model_recommendation: {
    best_model_key?: string | null;
    best_model_name: string;
    basis?: string;
    recommendation: string;
    next_step: string;
    mae_m?: number;
    rmse_m?: number;
  };
}

export interface QualityGateReport {
  is_passed: boolean;
  status_label: string;
  dataset_sufficiency: "LOW" | "MEDIUM" | "GOOD" | "NONE";
  sufficiency_reason: string;
  total_samples: number;
  total_scenes: number;
  train_samples: number;
  val_samples: number;
  test_split_samples: number;
  detected_depth_units: string[];
  depth_min_observed_m?: number | null;
  depth_max_observed_m?: number | null;
  valid_pixel_ratio_mean?: number | null;
  leakage_detected: boolean;
  warnings: string[];
  blockers: string[];
  required_dataset_schema: Record<string, any>;
}

export interface TrainingState {
  status: string;
  mode?: string;
  is_running: boolean;
  current_epoch: number;
  total_epochs: number;
  progress_pct: number;
  last_run_summary?: Record<string, any> | null;
  error_message?: string | null;
}

export interface ModelRangeAnalysis {
  model_name: string;
  model_max_depth_m: number;
  dataset_max_depth_m: number;
  compatibility_status: string;
  is_compatible: boolean;
}

export interface TrainingStatusResponse {
  training_state: TrainingState;
  quality_gate_report: QualityGateReport;
  model_range_analysis: ModelRangeAnalysis;
  checkpoint_available: boolean;
  checkpoint_path?: string | null;
  training_summary?: Record<string, any> | null;
  required_dataset_specifications: Record<string, any>;
}

export interface GAMUSDatasetReport {
  status: "VALID" | "INVALID" | "CORRUPTED" | "EMPTY";
  data_dir: string;
  sample_counts: {
    total: number;
    train: number;
    val: number;
    test: number;
  };
  rgb_dimensions: [number, number];
  semantic_classes: Record<string, number>;
  agl_stats: {
    min?: number;
    max?: number;
    mean?: number;
    median?: number;
    valid_ratio?: number;
  };
  warnings: string[];
  errors: string[];
  corrupted_files: string[];
}

export interface TrainingModeInfo {
  status: "AVAILABLE" | "BLOCKED";
  requirements: string[];
  description: string;
  blocked_reason?: string | null;
}

export interface MultiTaskTrainingStatusResponse {
  training_state: TrainingState;
  datasets: {
    gamus: {
      status: "AVAILABLE" | "MISSING";
      details: GAMUSDatasetReport;
    };
    metric_depth: {
      status: "AVAILABLE" | "MISSING";
      details: QualityGateReport;
    };
  };
  modes_availability: {
    ZERO_SHOT: TrainingModeInfo;
    AERIAL_DOMAIN_ADAPTATION: TrainingModeInfo;
    AERIAL_METRIC_FINE_TUNING: TrainingModeInfo;
  };
  last_report?: Record<string, any> | null;
}
