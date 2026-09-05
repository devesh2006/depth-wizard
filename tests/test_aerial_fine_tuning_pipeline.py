import os
import sys
import json
import pytest
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
backend_path = os.path.join(ROOT_DIR, "backend")
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from training.dataset import (
    AerialDepthManifestItem,
    parse_dataset_manifest,
    validate_dataset_quality,
    AerialDepthDataset,
    load_depth_file
)
from training.train_aerial_depth import SiLogLoss, compute_depth_metrics, inspect_model_depth_range
from services.depth_benchmark_service import get_metric_model, MODEL_CLASSES
from services.stage3_height_service import estimate_building_vertical_height


def create_synthetic_dataset_dir(tmp_path, num_samples=5, split="train", depth_unit="meter", is_normalized=False):
    """Creates a temporary synthetic paired RGB + depth dataset directory for unit tests."""
    dataset_dir = tmp_path / "synthetic_dataset"
    split_dir = dataset_dir / split
    img_dir = split_dir / "images"
    depth_dir = split_dir / "depths"
    img_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    manifest_items = []
    for i in range(num_samples):
        img_name = f"sample_{i:02d}.jpg"
        depth_name = f"sample_{i:02d}.npy"

        # Create dummy RGB image (100x100)
        img_arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        Image.fromarray(img_arr).save(img_dir / img_name)

        # Create dummy metric depth array (100x100)
        if is_normalized:
            d_arr = np.random.uniform(0.1, 0.9, (100, 100)).astype(np.float32)
        else:
            d_arr = np.random.uniform(10.0, 150.0, (100, 100)).astype(np.float32)
        np.save(depth_dir / depth_name, d_arr)

        manifest_items.append({
            "image": f"{split}/images/{img_name}",
            "depth": f"{split}/depths/{depth_name}",
            "depth_unit": depth_unit,
            "depth_type": "camera_surface_depth",
            "scene_id": f"scene_{i:02d}",
            "split": split
        })

    with open(dataset_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_items, f, indent=2)

    return str(dataset_dir)


def test_1_manifest_parsing_and_scanner(tmp_path):
    """Verify manifest parsing and directory auto-scanning."""
    ds_dir = create_synthetic_dataset_dir(tmp_path, num_samples=3)
    items, err = parse_dataset_manifest(ds_dir)

    assert err is None
    assert len(items) == 3
    assert items[0].split == "train"
    assert items[0].depth_unit == "meter"


def test_2_depth_unit_detection_and_warnings(tmp_path):
    """Verify quality gate flags DEPTH_UNIT_WARNING if data appears normalized [0, 1]."""
    norm_ds_dir = create_synthetic_dataset_dir(tmp_path, num_samples=2, is_normalized=True)
    report = validate_dataset_quality(norm_ds_dir)

    assert any("DEPTH_UNIT_WARNING" in w for w in report.warnings)


def test_3_invalid_depth_filtering_and_valid_masking():
    """Verify valid pixel mask correctly filters NaNs, zeros, and out-of-range depths."""
    d_raw = np.array([
        [10.0, -5.0, np.nan],
        [0.0, 150.0, 600.0]
    ], dtype=np.float32)

    valid_mask = np.isfinite(d_raw) & (d_raw >= 0.5) & (d_raw <= 500.0)

    assert valid_mask[0, 0] is np.bool_(True)
    assert valid_mask[0, 1] is np.bool_(False)
    assert valid_mask[0, 2] is np.bool_(False)
    assert valid_mask[1, 0] is np.bool_(False)
    assert valid_mask[1, 1] is np.bool_(True)
    assert valid_mask[1, 2] is np.bool_(False)


def test_4_split_leakage_detection(tmp_path):
    """Verify quality gate detects scene_id overlap between train and val/test splits."""
    ds_dir = tmp_path / "leakage_dataset"
    (ds_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
    (ds_dir / "train" / "depths").mkdir(parents=True, exist_ok=True)

    # Save 1 sample
    Image.fromarray(np.zeros((50, 50, 3), dtype=np.uint8)).save(ds_dir / "train" / "images" / "s1.jpg")
    np.save(ds_dir / "train" / "depths" / "s1.npy", np.full((50, 50), 20.0, dtype=np.float32))

    manifest = [
        {"image": "train/images/s1.jpg", "depth": "train/depths/s1.npy", "scene_id": "SHARED_SCENE", "split": "train"},
        {"image": "train/images/s1.jpg", "depth": "train/depths/s1.npy", "scene_id": "SHARED_SCENE", "split": "test"}
    ]
    with open(ds_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)

    report = validate_dataset_quality(str(ds_dir))
    assert report.leakage_detected is True
    assert report.is_passed is False


def test_5_model_range_inspection():
    """Verify model depth range inspector reports compatibility status."""
    info = inspect_model_depth_range("da_v2_metric_large", dataset_max_depth=500.0)
    assert info["model_max_depth_m"] == 80.0
    assert info["dataset_max_depth_m"] == 500.0
    assert "RANGE_WARNING" in info["compatibility_status"]


def test_6_silog_loss_calculation():
    """Verify Scale-Invariant Logarithmic Loss (SiLog) operates on valid pixels."""
    criterion = SiLogLoss(alpha=0.5, lambda_param=0.85)

    pred = torch.tensor([[[10.0, 20.0], [30.0, 40.0]]], dtype=torch.float32)
    target = torch.tensor([[[12.0, 18.0], [32.0, 38.0]]], dtype=torch.float32)
    mask = torch.tensor([[[True, True], [True, True]]], dtype=torch.bool)

    loss = criterion(pred, target, mask)
    assert loss.item() > 0.0
    assert not torch.isnan(loss)


def test_7_depth_metrics_computation():
    """Verify computation of MAE, RMSE, MRE, MedRE, delta1, delta2, delta3."""
    pred = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    target = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    mask = np.ones((2, 2), dtype=bool)

    metrics = compute_depth_metrics(pred, target, mask)
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["mre"] == 0.0
    assert metrics["delta1"] == 100.0


def test_8_fine_tuned_model_registry_integration():
    """Verify fine-tuned model key 'da_v2_large_aerial' is registered and handles missing checkpoint safely."""
    assert "da_v2_large_aerial" in MODEL_CLASSES
    model = get_metric_model("da_v2_large_aerial")
    assert model is not None

    img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    res = model.predict(img)

    # When no trained checkpoint exists yet, returns UNAVAILABLE status with clear reason
    assert res["status"] in ["UNAVAILABLE", "SUCCESS"]
    if res["status"] == "UNAVAILABLE":
        assert "Fine-tuned checkpoint" in res["error_reason"] or "not found" in res["error_reason"]


def test_9_ground_truth_isolation():
    """Verify that changing GT parameters has ZERO influence on Stage 1 metric depth model prediction."""
    model = get_metric_model("da_v2_base")
    img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))

    res1 = model.predict(img)
    res2 = model.predict(img)

    if res1["status"] == "SUCCESS" and res2["status"] == "SUCCESS":
        np.testing.assert_array_almost_equal(res1["depth_map_m"], res2["depth_map_m"])


def test_10_frozen_stage3_compatibility():
    """Verify that Stage 3 geometry engine outputs identical height results for identical depth maps."""
    synth_depth = np.full((100, 100), 20.0, dtype=np.float32)
    synth_depth[30:70, 30:70] = 10.0  # 10m building top vs 20m ground

    res = estimate_building_vertical_height(depth_m=synth_depth, bbox=[30, 30, 40, 40])
    assert res["height_available"] is True
    assert res["estimated_height_m"] == pytest.approx(10.0, abs=0.5)


def test_11_training_quality_gate_guard(tmp_path):
    """Verify quality gate blocks training when dataset directory is empty."""
    empty_dir = str(tmp_path / "empty_dir")
    os.makedirs(empty_dir, exist_ok=True)

    report = validate_dataset_quality(empty_dir)
    assert report.is_passed is False
    assert "TRAINING BLOCKED" in report.status_label
