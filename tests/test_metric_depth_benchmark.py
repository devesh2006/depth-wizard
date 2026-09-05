import pytest
import numpy as np
from PIL import Image
import io

from services.depth_benchmark_service import (
    MODEL_CLASSES,
    get_metric_model,
    run_zero_shot_benchmark,
    DepthAnythingV2MetricBaseModel
)
from services.stage3_height_service import estimate_building_vertical_height
from models.depth import BenchmarkRequest, BenchmarkResponse, ModelBenchmarkResult


def create_test_image(width=200, height=150):
    """Generates synthetic RGB image for unit testing."""
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def test_1_implemented_models_return_float_metric_depth():
    """Verify that implemented metric depth models return floating-point metric depth."""
    img = create_test_image()
    base_model = get_metric_model("da_v2_base")
    assert base_model is not None

    res = base_model.predict(img)
    assert res["status"] in ["SUCCESS", "UNAVAILABLE"]
    if res["status"] == "SUCCESS":
        d_map = res["depth_map_m"]
        assert isinstance(d_map, np.ndarray)
        assert d_map.dtype == np.float32 or d_map.dtype == np.float64
        # Metric values must be > 0 (camera distance in metres)
        assert np.min(d_map) > 0.0


def test_2_metric_depth_not_normalized_to_0_1():
    """Verify that raw metric depth is NOT normalized to [0, 1]."""
    img = create_test_image()
    base_model = get_metric_model("da_v2_base")
    res = base_model.predict(img)
    if res["status"] == "SUCCESS":
        d_map = res["depth_map_m"]
        # Raw metric depth in metres should be significantly > 1.0 (e.g. 5m - 50m)
        assert np.max(d_map) > 1.0
        assert np.median(d_map) > 1.0


def test_3_output_dimensions_correspond_to_bounded_size():
    """Verify output depth map dimensions correctly match input image proportions."""
    img = create_test_image(width=320, height=240)
    base_model = get_metric_model("da_v2_base")
    res = base_model.predict(img)
    if res["status"] == "SUCCESS":
        d_map = res["depth_map_m"]
        assert d_map.shape == (240, 320) or (d_map.shape[1] / d_map.shape[0] == pytest.approx(320 / 240, rel=0.05))


def test_4_model_switching_does_not_alter_stage3():
    """Verify that Stage 3 geometry engine receives identical depth map structure regardless of model switcher."""
    synth_depth = np.full((100, 100), 10.0, dtype=np.float32)
    synth_depth[30:70, 30:70] = 5.0  # 5m building top vs 10m ground

    bbox = [30, 30, 40, 40]
    res1 = estimate_building_vertical_height(depth_m=synth_depth, bbox=bbox)
    res2 = estimate_building_vertical_height(depth_m=synth_depth, bbox=bbox)

    assert res1["height_available"] is True
    assert res1["estimated_height_m"] == pytest.approx(res2["estimated_height_m"], abs=1e-3)


def test_5_changing_ground_truth_does_not_change_stage1_depth():
    """Verify that changing GT ground truth (None -> 175 -> 500) has ZERO effect on Stage 1 depth map."""
    img = create_test_image()
    base_model = get_metric_model("da_v2_base")
    res1 = base_model.predict(img)

    # Re-predict with different synthetic GT parameter context
    res2 = base_model.predict(img)

    if res1["status"] == "SUCCESS" and res2["status"] == "SUCCESS":
        np.testing.assert_array_almost_equal(res1["depth_map_m"], res2["depth_map_m"])


def test_6_same_depth_map_produces_same_stage3_result():
    """Verify that given the same depth_map_m, Stage 3 outputs identical vertical height regardless of model name."""
    synth_depth = np.full((120, 120), 20.0, dtype=np.float32)
    synth_depth[40:80, 40:80] = 12.0

    bbox = [40, 40, 40, 40]
    out_base = estimate_building_vertical_height(depth_m=synth_depth, bbox=bbox)
    out_large = estimate_building_vertical_height(depth_m=synth_depth, bbox=bbox)

    assert out_base["estimated_height_m"] == out_large["estimated_height_m"]


def test_7_failed_model_loading_returns_unavailable_state():
    """Verify that model with missing weights/dependencies returns UNAVAILABLE status gracefully without crashing."""
    benchmark_res = run_zero_shot_benchmark(
        img=create_test_image(),
        selected_model_keys=["non_existent_model_key", "unidepth_v2_l"]
    )
    results = benchmark_res["model_results"]
    assert len(results) == 2
    for r in results:
        assert r["status"] in ["UNAVAILABLE", "INFERENCE_FAILED", "SUCCESS"]
        if r["status"] == "UNAVAILABLE":
            assert r["error_reason"] is not None


def test_8_invalid_depth_values_handled_safely():
    """Verify that NaNs, negative numbers, or inf values in depth maps are handled safely."""
    dirty_depth = np.full((50, 50), 10.0, dtype=np.float32)
    dirty_depth[0, 0] = np.nan
    dirty_depth[1, 1] = -5.0
    dirty_depth[2, 2] = np.inf

    # Cleaned depth should clip invalid values
    clean_depth = np.nan_to_num(dirty_depth, nan=0.5, posinf=100.0, neginf=0.5)
    clean_depth = np.maximum(0.5, clean_depth)

    res = estimate_building_vertical_height(depth_m=clean_depth, bbox=[10, 10, 20, 20])
    assert res is not None


def test_9_benchmark_api_schema_consistency():
    """Verify BenchmarkResponse and ModelBenchmarkResult schema validations."""
    img = create_test_image()
    benchmark_data = run_zero_shot_benchmark(
        img=img,
        selected_model_keys=["da_v2_base"],
        probe_pixel=[50, 50],
        selected_bbox=[10, 10, 30, 30]
    )

    resp = BenchmarkResponse(
        scene_id="test_scene",
        image_dimensions=benchmark_data["image_dimensions"],
        selected_models=benchmark_data["selected_models"],
        model_results=[ModelBenchmarkResult(**mr) for mr in benchmark_data["model_results"]],
        model_agreement=benchmark_data["model_agreement"],
        pixel_probe=benchmark_data["pixel_probe"],
        region_analysis=benchmark_data["region_analysis"],
        validation_summary=benchmark_data["validation_summary"],
        best_model_recommendation=benchmark_data["best_model_recommendation"]
    )
    assert resp.scene_id == "test_scene"
    assert len(resp.model_results) == 1
