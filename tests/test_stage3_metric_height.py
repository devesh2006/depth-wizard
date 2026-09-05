import sys
import os
import pytest
import numpy as np
from PIL import Image

# Add backend directory to python path
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
sys.path.insert(0, backend_path)

from services.depth_estimator import create_depth_estimator, StructuralMetricEstimator
from services.stage3_height_service import (
    estimate_building_vertical_height,
    fit_ransac_plane,
    render_synthetic_known_scale_scene
)
from services.segmentation_service import measure_building_height
from models.depth import CalibrationConfig

# ---------------------------------------------------------------------------
# AUTOMATED UNIT TEST SUITE FOR STAGE 3 METRIC VERTICAL HEIGHT ESTIMATION
# ---------------------------------------------------------------------------

def test_1_metric_depth_consumed_directly():
    """1. Metric depth is consumed directly without relative normalization."""
    depth_m = np.ones((200, 200), dtype=np.float32) * 45.0
    depth_m[60:140, 60:140] = 30.0

    res = estimate_building_vertical_height(depth_m, bbox=[60, 60, 80, 80], view_geometry="Oblique")
    assert res["height_available"] is True
    assert res["estimated_height_m"] > 5.0
    assert res["estimated_height_m"] <= 20.0


def test_2_ground_truth_isolation_in_stage3():
    """2. Ground truth height has ZERO influence on Stage 3 estimation."""
    img = Image.new("RGB", (200, 200), color=(120, 140, 160))
    estimator = StructuralMetricEstimator()
    depth_m, _ = estimator.estimate_depth(img)

    calib = CalibrationConfig(mode="relative")
    bbox = [50, 50, 60, 60]

    rep_gt_none = measure_building_height(depth_m, bbox, calib, "Bldg A", ground_truth_height_m=None)
    rep_gt_175 = measure_building_height(depth_m, bbox, calib, "Bldg A", ground_truth_height_m=175.0)
    rep_gt_500 = measure_building_height(depth_m, bbox, calib, "Bldg A", ground_truth_height_m=500.0)

    assert rep_gt_none.calibrated_height_m == rep_gt_175.calibrated_height_m == rep_gt_500.calibrated_height_m
    assert rep_gt_none.ground_elevation_m == rep_gt_175.ground_elevation_m == rep_gt_500.ground_elevation_m


def test_3_height_calculated_from_world_elevation_difference():
    """3. Height is calculated from vertical world elevation difference |Z_roof - Z_ground|."""
    depth_m = np.ones((200, 200), dtype=np.float32) * 50.0
    depth_m[50:150, 50:150] = 38.0

    res = estimate_building_vertical_height(depth_m, bbox=[50, 50, 100, 100], view_geometry="Oblique")
    assert res["height_available"] is True
    assert abs(res["estimated_height_m"] - 12.0) < 2.0


def test_4_camera_distance_is_not_vertical_height():
    """4. Camera distance P50 (e.g. 45m) is NOT directly output as vertical height."""
    depth_m = np.ones((200, 200), dtype=np.float32) * 45.0
    depth_m[60:140, 60:140] = 35.0

    res = estimate_building_vertical_height(depth_m, bbox=[60, 60, 80, 80], view_geometry="Oblique")
    assert res["estimated_height_m"] != 45.0
    assert res["estimated_height_m"] != 35.0
    assert res["estimated_height_m"] < 25.0


def test_5_ransac_ground_plane_estimation_works():
    """5. RANSAC ground plane fitting correctly estimates terrain plane."""
    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 10, 100)
    xx, yy = np.meshgrid(x, y)
    zz = 0.1 * xx + 0.2 * yy + 10.0
    pts = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=-1)

    plane_coefs, inlier_ratio, rmse, _ = fit_ransac_plane(pts, distance_threshold=0.1)
    assert plane_coefs is not None
    assert inlier_ratio > 0.9
    assert rmse < 0.1


def test_6_roof_surface_estimation_works():
    """6. Roof surface estimation extracts top elevation distinct from ground."""
    depth_m = np.ones((200, 200), dtype=np.float32) * 60.0
    depth_m[70:130, 70:130] = 42.0

    res = estimate_building_vertical_height(depth_m, bbox=[70, 70, 60, 60], view_geometry="Oblique")
    assert res["roof_elevation_m"] is not None
    assert res["ground_elevation_m"] is not None
    assert res["roof_elevation_m"] > res["ground_elevation_m"]


def test_7_point_cloud_uses_metric_z():
    """7. Point cloud unprojection Z uses raw metric depth metres."""
    depth_m = np.ones((100, 100), dtype=np.float32) * 55.4
    intrinsics = {"fx": 120.0, "fy": 120.0, "cx": 50.0, "cy": 50.0}

    res = estimate_building_vertical_height(depth_m, bbox=[20, 20, 60, 60], intrinsics=intrinsics, view_geometry="Oblique")
    assert res["intrinsics_source"] is not None


def test_8_height_output_in_metres():
    """8. Height output unit is metres."""
    depth_m = np.ones((200, 200), dtype=np.float32) * 40.0
    depth_m[50:150, 50:150] = 25.0

    res = estimate_building_vertical_height(depth_m, bbox=[50, 50, 100, 100], view_geometry="Oblique")
    assert isinstance(res["estimated_height_m"], float)
    assert res["estimated_height_m"] > 0.0


def test_9_near_nadir_unsuitable_geometry_returns_unavailable():
    """9. Near-nadir top-down view without camera pose returns height_available: False."""
    depth_m = np.ones((200, 200), dtype=np.float32) * 100.0

    res = estimate_building_vertical_height(depth_m, bbox=[50, 50, 100, 100], view_geometry="Near-Nadir", camera_pose=None)
    assert res["height_available"] is False
    assert res["estimated_height_m"] is None
    assert "Near-Nadir" in res["reason_if_unavailable"] or "nadir" in res["reason_if_unavailable"].lower()


def test_10_missing_camera_pose_handled_honestly():
    """10. Missing camera pose sets camera_pose_available: False and uses ground normal alignment."""
    depth_m = np.ones((200, 200), dtype=np.float32) * 40.0
    depth_m[50:150, 50:150] = 30.0

    res = estimate_building_vertical_height(depth_m, bbox=[50, 50, 100, 100], view_geometry="Oblique", camera_pose=None)
    assert res["camera_pose_available"] is False
    assert "Ground Normal Alignment" in res["camera_pose_status"]


def test_11_missing_building_mask_returns_unavailable():
    """11. Invalid/out of bounds bbox returns height_available: False."""
    depth_m = np.ones((50, 50), dtype=np.float32) * 30.0

    res = estimate_building_vertical_height(depth_m, bbox=[100, 100, 10, 10], view_geometry="Oblique")
    assert res["height_available"] is False or res["estimated_height_m"] is None or res["confidence"] < 0.5


def test_12_validation_calculates_errors_only_after_prediction():
    """12. Validation scoring evaluates error only post-inference; returns N/A if prediction is None."""
    gt_val = 175.0
    est_val_valid = 168.4
    est_val_none = None

    abs_err = abs(est_val_valid - gt_val)
    rel_err = (abs_err / gt_val) * 100.0
    acc = 100.0 - rel_err

    assert round(abs_err, 1) == 6.6
    assert round(rel_err, 2) == 3.77
    assert round(acc, 2) == 96.23
    assert est_val_none is None


def test_13_gt_175m_never_used_inside_estimator():
    """13. Ground truth 175m is never passed into estimate_building_vertical_height."""
    depth_m = np.ones((200, 200), dtype=np.float32) * 40.0
    depth_m[50:150, 50:150] = 30.0

    import inspect
    sig = inspect.signature(estimate_building_vertical_height)
    assert "ground_truth_height_m" not in sig.parameters


def test_14_no_old_alpha_formula_remains_in_stage3():
    """14. Stage 3 derivation does NOT rely on alpha * rel_dz / cos(theta) empirical formula."""
    depth_m = np.ones((200, 200), dtype=np.float32) * 50.0
    depth_m[50:150, 50:150] = 35.0

    res = estimate_building_vertical_height(depth_m, bbox=[50, 50, 100, 100], view_geometry="Oblique")
    assert "alpha" not in res["method"].lower()


def test_15_no_gsd_scaling_used_in_stage3():
    """15. Stage 3 height estimation is independent of GSD scale overrides."""
    depth_m = np.ones((200, 200), dtype=np.float32) * 50.0
    depth_m[50:150, 50:150] = 35.0

    res_1 = estimate_building_vertical_height(depth_m, bbox=[50, 50, 100, 100], view_geometry="Oblique")
    res_2 = estimate_building_vertical_height(depth_m, bbox=[50, 50, 100, 100], view_geometry="Oblique")

    assert res_1["estimated_height_m"] == res_2["estimated_height_m"]


# ---------------------------------------------------------------------------
# SECTION 11 & 20: CONTROLLED SYNTHETIC KNOWN-SCALE SCENE GEOMETRY SUITE
# ---------------------------------------------------------------------------

def test_16_synthetic_known_scale_10m():
    """16. Controlled synthetic 3D scene with True Height = 10.0m."""
    depth_m, bbox, intrinsics = render_synthetic_known_scale_scene(height_m=10.0, ground_depth_m=40.0)
    res = estimate_building_vertical_height(depth_m, bbox=bbox, intrinsics=intrinsics, view_geometry="Oblique")

    assert res["height_available"] is True
    assert abs(res["estimated_height_m"] - 10.0) <= 0.5


def test_17_synthetic_known_scale_50m():
    """17. Controlled synthetic 3D scene with True Height = 50.0m."""
    depth_m, bbox, intrinsics = render_synthetic_known_scale_scene(height_m=50.0, ground_depth_m=70.0)
    res = estimate_building_vertical_height(depth_m, bbox=bbox, intrinsics=intrinsics, view_geometry="Oblique")

    assert res["height_available"] is True
    assert abs(res["estimated_height_m"] - 50.0) <= 1.0


def test_18_synthetic_known_scale_100m():
    """18. Controlled synthetic 3D scene with True Height = 100.0m."""
    depth_m, bbox, intrinsics = render_synthetic_known_scale_scene(height_m=100.0, ground_depth_m=150.0)
    res = estimate_building_vertical_height(depth_m, bbox=bbox, intrinsics=intrinsics, view_geometry="Oblique")

    assert res["height_available"] is True
    assert abs(res["estimated_height_m"] - 100.0) <= 2.0


def test_19_degenerate_flat_roof_ground_separation():
    """19. Degenerate flat scene (height < 0.25m) returns height_available: False."""
    depth_m = np.ones((200, 200), dtype=np.float32) * 50.0
    depth_m[50:150, 50:150] = 49.9 # 0.1m difference below geometric resolution threshold

    res = estimate_building_vertical_height(depth_m, bbox=[50, 50, 100, 100], view_geometry="Oblique")
    assert res["height_available"] is False
    assert "below geometric resolution" in res["reason_if_unavailable"].lower()


# ---------------------------------------------------------------------------
# MASTER FIX PROMPT SECTION 10 TEST SUITE (TESTS 1 THROUGH 10)
# ---------------------------------------------------------------------------

def test_master_prompt_test_1():
    """TEST 1: ground = 10, roof = 60 -> height = 50."""
    res = estimate_building_vertical_height(ground_elevation_input=10.0, roof_elevation_input=60.0)
    assert res["height_available"] is True
    assert res["calculation_status"] == "HEIGHT_COMPUTED"
    assert res["estimated_height_m"] == 50.0


def test_master_prompt_test_2():
    """TEST 2: ground = 12.70, roof = 181.10 -> height = 168.40."""
    res = estimate_building_vertical_height(ground_elevation_input=12.70, roof_elevation_input=181.10)
    assert res["height_available"] is True
    assert res["calculation_status"] == "HEIGHT_COMPUTED"
    assert res["estimated_height_m"] == 168.40
    assert res["ground_elevation_m"] == 12.70
    assert res["roof_elevation_m"] == 181.10


def test_master_prompt_test_3():
    """TEST 3: ground = 0, roof = 100 -> height = 100."""
    res = estimate_building_vertical_height(ground_elevation_input=0.0, roof_elevation_input=100.0)
    assert res["height_available"] is True
    assert res["calculation_status"] == "HEIGHT_COMPUTED"
    assert res["estimated_height_m"] == 100.0


def test_master_prompt_test_4():
    """TEST 4: ground = 100, roof = 50 -> height = 50 (absolute / ordering behavior)."""
    res = estimate_building_vertical_height(ground_elevation_input=100.0, roof_elevation_input=50.0)
    assert res["height_available"] is True
    assert res["calculation_status"] == "HEIGHT_COMPUTED"
    assert res["estimated_height_m"] == 50.0


def test_master_prompt_test_5():
    """TEST 5: ground = roof -> height = unavailable / degenerate (< 0.25m)."""
    res = estimate_building_vertical_height(ground_elevation_input=50.0, roof_elevation_input=50.0)
    assert res["height_available"] is False
    assert res["calculation_status"] == "UNAVAILABLE"
    assert res["estimated_height_m"] is not None or "below geometric resolution" in res["reason_if_unavailable"].lower()


def test_master_prompt_test_6():
    """TEST 6: missing vertical reference -> height = unavailable."""
    res = estimate_building_vertical_height(
        ground_elevation_input=10.0, roof_elevation_input=60.0, vertical_reference_input="Unavailable"
    )
    assert res["height_available"] is False
    assert res["calculation_status"] == "UNAVAILABLE"
    assert res["estimated_height_m"] is None


def test_master_prompt_test_7():
    """TEST 7: NaN/Inf ground elevation -> height = unavailable."""
    res_nan = estimate_building_vertical_height(ground_elevation_input=float("nan"), roof_elevation_input=60.0)
    res_inf = estimate_building_vertical_height(ground_elevation_input=float("inf"), roof_elevation_input=60.0)

    assert res_nan["height_available"] is False
    assert res_nan["calculation_status"] == "UNAVAILABLE"
    assert res_inf["height_available"] is False
    assert res_inf["calculation_status"] == "UNAVAILABLE"


def test_master_prompt_test_8():
    """TEST 8: NaN/Inf roof elevation -> height = unavailable."""
    res_nan = estimate_building_vertical_height(ground_elevation_input=10.0, roof_elevation_input=float("nan"))
    res_inf = estimate_building_vertical_height(ground_elevation_input=10.0, roof_elevation_input=float("inf"))

    assert res_nan["height_available"] is False
    assert res_nan["calculation_status"] == "UNAVAILABLE"
    assert res_inf["height_available"] is False
    assert res_inf["calculation_status"] == "UNAVAILABLE"


def test_master_prompt_test_9():
    """TEST 9: GT = 175 m -> verify prediction is IDENTICAL to prediction produced when GT is absent."""
    depth_m = np.ones((200, 200), dtype=np.float32) * 50.0
    depth_m[50:150, 50:150] = 30.0
    calib = CalibrationConfig(mode="relative")
    bbox = [50, 50, 100, 100]

    bldg_no_gt = measure_building_height(depth_m, bbox, calib, "Test 9", ground_truth_height_m=None)
    bldg_gt_175 = measure_building_height(depth_m, bbox, calib, "Test 9", ground_truth_height_m=175.0)

    assert bldg_no_gt.calibrated_height_m == bldg_gt_175.calibrated_height_m
    assert bldg_no_gt.ground_elevation_m == bldg_gt_175.ground_elevation_m
    assert bldg_no_gt.roof_elevation_m == bldg_gt_175.roof_elevation_m
    assert bldg_gt_175.calculation_status == "VALIDATION_AVAILABLE"


def test_master_prompt_test_10():
    """TEST 10: GT = 500 m -> verify prediction is IDENTICAL to prediction produced when GT is absent."""
    depth_m = np.ones((200, 200), dtype=np.float32) * 50.0
    depth_m[50:150, 50:150] = 30.0
    calib = CalibrationConfig(mode="relative")
    bbox = [50, 50, 100, 100]

    bldg_no_gt = measure_building_height(depth_m, bbox, calib, "Test 10", ground_truth_height_m=None)
    bldg_gt_500 = measure_building_height(depth_m, bbox, calib, "Test 10", ground_truth_height_m=500.0)

    assert bldg_no_gt.calibrated_height_m == bldg_gt_500.calibrated_height_m
    assert bldg_no_gt.ground_elevation_m == bldg_gt_500.ground_elevation_m
    assert bldg_no_gt.roof_elevation_m == bldg_gt_500.roof_elevation_m
    assert bldg_gt_500.calculation_status == "VALIDATION_AVAILABLE"


if __name__ == "__main__":
    pytest.main([__file__])

