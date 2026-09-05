import sys
import os
import pytest
import numpy as np
from PIL import Image

# Add backend directory to path
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
sys.path.insert(0, backend_path)

from models.depth import CalibrationConfig
from services.segmentation_service import measure_building_height
from services.shadow_service import analyze_building_shadow
from services.fusion_engine import fuse_height_estimates
from services.validation_service import (
    calculate_mae,
    calculate_rmse,
    calculate_mean_relative_error,
    calculate_median_relative_error,
    calculate_pct_within_threshold
)

def create_synthetic_scene(seed=42):
    np.random.seed(seed)
    # Metric camera depth map (values in metres 30m - 50m)
    depth = np.ones((200, 200), dtype=np.float32) * 45.0
    # Peak rooftop region (closer to camera, e.g. 32m vs 45m)
    depth[60:110, 60:110] = 32.0
    bbox = [60, 60, 50, 50]
    img = Image.new("RGB", (200, 200), color=(128, 128, 128))
    return depth, bbox, img

# ---------------------------------------------------------------------------
# 16 AUTOMATED UNIT TESTS FOR STAGE 3 METRIC PIPELINE
# ---------------------------------------------------------------------------

def test_1_no_metadata():
    depth, bbox, img = create_synthetic_scene()
    calib = CalibrationConfig(mode="metadata", is_calibrated=False)
    m = measure_building_height(depth, bbox, calib, "Test 1", None, "Oblique (30°)")
    assert m.camera_parameter_status["focal_length"] in ["ESTIMATED", "UNAVAILABLE"]

def test_2_no_calibration():
    depth, bbox, img = create_synthetic_scene()
    calib = CalibrationConfig(mode="relative", is_calibrated=False)
    m = measure_building_height(depth, bbox, calib, "Test 2", None, "Oblique (30°)")
    assert m.calibrated_height_m is not None
    assert m.accuracy_pct is None

def test_3_near_nadir_image():
    depth, bbox, img = create_synthetic_scene()
    calib = CalibrationConfig(mode="relative", is_calibrated=False)
    m = measure_building_height(depth, bbox, calib, "Test 3", 179.0, "Near-Nadir (85°)")
    assert m.is_metric_available is False
    assert "nadir" in m.metric_status.lower() or "near-nadir" in m.calibration_note.lower()

def test_4_oblique_image():
    depth, bbox, img = create_synthetic_scene()
    calib = CalibrationConfig(mode="metadata", is_calibrated=True)
    m = measure_building_height(depth, bbox, calib, "Test 4", 179.0, "Oblique (35°)")
    assert m.is_metric_available is True
    assert m.calibrated_height_m is not None
    assert m.calibrated_height_m > 0

def test_5_ground_truth_absent():
    depth, bbox, img = create_synthetic_scene()
    calib = CalibrationConfig(mode="metadata", is_calibrated=True)
    m = measure_building_height(depth, bbox, calib, "Test 5", None, "Oblique (35°)")
    assert m.ground_truth_height_m is None
    assert m.error_m is None
    assert m.accuracy_pct is None

def test_6_ground_truth_present():
    depth, bbox, img = create_synthetic_scene()
    calib = CalibrationConfig(mode="metadata", is_calibrated=True)
    m = measure_building_height(depth, bbox, calib, "Test 6", 179.0, "Oblique (35°)")
    assert m.ground_truth_height_m == 179.0
    assert m.error_m is not None
    assert m.accuracy_pct is not None

def test_7_gt_changed_isolation():
    depth, bbox, img = create_synthetic_scene()
    calib = CalibrationConfig(mode="metadata", is_calibrated=True)
    m_179 = measure_building_height(depth, bbox, calib, "Test 7", 179.0, "Oblique (35°)")
    m_175 = measure_building_height(depth, bbox, calib, "Test 7", 175.0, "Oblique (35°)")
    assert m_179.calibrated_height_m == m_175.calibrated_height_m
    assert m_179.error_m != m_175.error_m

def test_8_perfect_prediction():
    depth, bbox, img = create_synthetic_scene()
    m_probe = measure_building_height(depth, bbox, CalibrationConfig(mode="relative"), "Probe", None, "Oblique")
    pred_h = m_probe.calibrated_height_m
    m = measure_building_height(depth, bbox, CalibrationConfig(mode="relative"), "Test 8", pred_h, "Oblique")
    assert abs(m.error_m) <= 0.2
    assert m.accuracy_pct >= 99.8

def test_9_known_prediction_error():
    depth, bbox, img = create_synthetic_scene()
    m_probe = measure_building_height(depth, bbox, CalibrationConfig(mode="relative"), "Probe", None, "Oblique")
    pred_h = m_probe.calibrated_height_m
    gt_h = pred_h + 9.0
    m = measure_building_height(depth, bbox, CalibrationConfig(mode="relative"), "Test 9", gt_h, "Oblique")
    assert abs(m.error_m - 9.0) <= 0.2

def test_10_invalid_prediction():
    depth, bbox, img = create_synthetic_scene()
    calib = CalibrationConfig(mode="relative", is_calibrated=False)
    m = measure_building_height(depth, bbox, calib, "Test 10", 179.0, "Near-Nadir")
    assert m.calibrated_height_m is None
    assert m.is_metric_available is False

def test_11_missing_camera_geometry():
    depth, bbox, img = create_synthetic_scene()
    calib = CalibrationConfig(mode="metadata", focal_length_mm=None, flight_altitude_m=None)
    m = measure_building_height(depth, bbox, calib, "Test 11", 179.0, "Oblique")
    assert m.camera_parameter_status["focal_length"] in ["ESTIMATED", "UNAVAILABLE"]

def test_12_missing_ground_plane():
    flat_depth = np.ones((100, 100), dtype=np.float32) * 30.0
    bbox = [20, 20, 30, 30]
    calib = CalibrationConfig(mode="terrain", is_calibrated=False)
    m = measure_building_height(flat_depth, bbox, calib, "Test 12", None, "Oblique")
    assert m.relative_height_unitless <= 0.02

def test_13_shadow_available():
    depth, bbox, img = create_synthetic_scene()
    res = analyze_building_shadow(img, depth, bbox, gsd_m_per_pixel=0.1, sun_elevation_deg=45.0)
    assert "shadow_detected" in res
    assert "confidence" in res

def test_14_shadow_unavailable():
    depth, bbox, img = create_synthetic_scene()
    res = analyze_building_shadow(img, depth, bbox, gsd_m_per_pixel=None, sun_elevation_deg=45.0)
    assert res["height_shadow_cue_m"] is None

def test_15_calibration_available():
    depth, bbox, img = create_synthetic_scene()
    calib = CalibrationConfig(mode="reference", known_object_height_m=50.0, is_calibrated=True)
    m = measure_building_height(depth, bbox, calib, "Test 15", None, "Oblique")
    assert "Metric 3D" in m.measurement_mode or "RANSAC" in m.measurement_mode
    assert m.is_metric_available is True

def test_16_calibration_unavailable():
    depth, bbox, img = create_synthetic_scene()
    calib = CalibrationConfig(mode="relative", is_calibrated=False)
    m = measure_building_height(depth, bbox, calib, "Test 16", None, "Near-Nadir")
    assert m.is_metric_available is False

if __name__ == "__main__":
    pytest.main([__file__])
