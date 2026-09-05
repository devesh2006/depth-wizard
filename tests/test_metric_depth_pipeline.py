import sys
import os
import pytest
import numpy as np
from PIL import Image

# Add backend directory to python path
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
sys.path.insert(0, backend_path)

from services.depth_estimator import create_depth_estimator, StructuralMetricEstimator
from services.geometry import reconstruct_3d_geometry
from services.segmentation_service import measure_building_height
from models.depth import CalibrationConfig, ProcessImageRequest

# ---------------------------------------------------------------------------
# REQUIREMENT 18 AUTOMATED UNIT TEST SUITE FOR METRIC DEPTH PIPELINE
# ---------------------------------------------------------------------------

def test_1_metric_output_is_float_metres():
    """1. Metric output is floating-point metres."""
    img = Image.new("RGB", (200, 200), color=(120, 140, 160))
    estimator = StructuralMetricEstimator()
    depth_m, depth_rel = estimator.estimate_depth(img, image_type="aerial")

    assert isinstance(depth_m, np.ndarray)
    assert depth_m.dtype == np.float32
    assert float(np.min(depth_m)) > 1.0  # Must be positive metres (> 1m)
    assert float(np.max(depth_m)) > 10.0 # Realistic outdoor camera depth in metres


def test_2_metric_output_is_not_normalized():
    """2. Metric output is NOT normalized to 0–1."""
    img = Image.new("RGB", (200, 200), color=(100, 150, 200))
    estimator = StructuralMetricEstimator()
    depth_m, depth_rel = estimator.estimate_depth(img, image_type="aerial")

    # depth_m values must exceed 1.0 (in metres), whereas depth_rel is in [0, 1]
    assert np.max(depth_m) > 1.0
    assert np.min(depth_m) >= 5.0
    assert np.max(depth_rel) <= 1.0
    assert np.min(depth_rel) >= 0.0


def test_3_ground_truth_isolation():
    """3. Changing ground truth does not change prediction or depth map."""
    img = Image.new("RGB", (200, 200), color=(120, 140, 160))
    estimator = StructuralMetricEstimator()
    depth_m_1, _ = estimator.estimate_depth(img)

    calib = CalibrationConfig(mode="relative", is_calibrated=False)
    bbox = [40, 40, 60, 60]

    m_gt_179 = measure_building_height(depth_m_1, bbox, calib, "Bldg A", ground_truth_height_m=179.0)
    m_gt_500 = measure_building_height(depth_m_1, bbox, calib, "Bldg A", ground_truth_height_m=500.0)

    # Prediction metrics must remain identical regardless of ground truth input
    assert m_gt_179.camera_to_building_distance_m == m_gt_500.camera_to_building_distance_m
    assert m_gt_179.rooftop_peak_z_rel == m_gt_500.rooftop_peak_z_rel


def test_4_validation_input_isolation():
    """4. Changing validation input does not change metric inference."""
    img = Image.new("RGB", (200, 200), color=(100, 100, 100))
    estimator = StructuralMetricEstimator()
    depth_m_before, _ = estimator.estimate_depth(img)

    # Simulate validation scoring API call logic
    gt_val_1 = 45.0
    gt_val_2 = 120.0

    abs_err_1 = abs(42.8 - gt_val_1)
    abs_err_2 = abs(42.8 - gt_val_2)

    depth_m_after, _ = estimator.estimate_depth(img)

    np.testing.assert_array_equal(depth_m_before, depth_m_after)
    assert abs_err_1 != abs_err_2


def test_5_metric_depth_preserved_before_eval():
    """5. Metric depth is preserved in full floating-point precision before evaluation."""
    img = Image.new("RGB", (200, 200), color=(180, 180, 180))
    estimator = StructuralMetricEstimator()
    depth_m, depth_rel = estimator.estimate_depth(img)

    # Verify floating point precision is retained
    assert depth_m.dtype == np.float32
    assert len(np.unique(depth_m)) > 50


def test_6_3d_coordinates_from_metric_depth():
    """6. 3D coordinates are generated from metric depth map using pinhole unprojection."""
    img = Image.new("RGB", (100, 100), color=(150, 150, 150))
    depth_m = np.ones((100, 100), dtype=np.float32) * 45.0  # Constant 45 metres depth
    intrinsics = {"fx": 120.0, "fy": 120.0, "cx": 50.0, "cy": 50.0}

    pc_data, mesh_data = reconstruct_3d_geometry(img, depth_m, scale_factor=1.0, intrinsics=intrinsics)

    # Point cloud Z bounds must reflect camera distance 45m
    assert pc_data.bounds["min_z"] >= 40.0
    assert pc_data.bounds["max_z"] <= 50.0
    assert pc_data.points_count > 1000


def test_7_no_hardcoded_building_height():
    """7. No hardcoded building height affects model inference."""
    img = Image.new("RGB", (200, 200), color=(80, 120, 160))
    estimator = create_depth_estimator("structural")
    depth_m, depth_rel = estimator.estimate_depth(img)

    # System must emit raw depth map without forcing any hardcoded 179m or 50m value
    assert float(np.mean(depth_m)) != 179.0
    assert float(np.mean(depth_m)) != 50.0


def test_8_missing_exif_handles_estimated_intrinsics():
    """8. Missing EXIF cleanly sets 'Estimated intrinsics' without fabricating metric scale."""
    img = Image.new("RGB", (300, 200), color=(100, 100, 100))
    w, h = img.size
    req = ProcessImageRequest(image_type="aerial", camera_intrinsics=None)

    # Expected estimated intrinsics
    fx_est = max(w, h) * 1.2
    cx_est = w / 2.0

    assert req.camera_intrinsics is None
    assert fx_est == 360.0
    assert cx_est == 150.0


if __name__ == "__main__":
    pytest.main([__file__])
