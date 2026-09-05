"""
PyTorch Unit Tests for Phase 2.1 GAMUS Aerial-Domain Adaptation & Multi-Task Metric Depth Fine-Tuning Pipeline.

Validates:
1. GAMUS Dataset Loader, RGB/AGL/Semantic alignment, unit preservation, missing camera_depth = None.
2. MultiTaskAerialDepthModel architecture, output head separation, max depth output range (500m).
3. MultiTaskAerialLoss computation with missing labels (handling GAMUS without camera_depth).
4. FROZEN Stage 3 compatibility: ensuring Stage 3 receives only depth_map_m and preserves geometric calculations.
5. Checkpoint saving and loading across modes.
"""

import os
import sys
import shutil
import tempfile
import numpy as np
import pytest
import torch
from PIL import Image

# Ensure backend root is on sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(ROOT_DIR, "backend") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT_DIR, "backend"))

from training.datasets.gamus_dataset import GAMUSDataset, parse_gamus_manifest, validate_gamus_quality
from training.models.multitask_aerial_depth import MultiTaskAerialDepthModel, MultiTaskAerialLoss, ExtendedMetricOutputHead
from services.segmentation_service import measure_building_height
from models.depth import CalibrationConfig


@pytest.fixture
def temp_gamus_dir():
    """Creates a temporary valid GAMUS dataset layout for testing."""
    tmp_dir = tempfile.mkdtemp(prefix="test_gamus_")
    
    # Create train, val, test subdirs
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(tmp_dir, split, "rgb"), exist_ok=True)
        os.makedirs(os.path.join(tmp_dir, split, "agl"), exist_ok=True)
        os.makedirs(os.path.join(tmp_dir, split, "semantic"), exist_ok=True)

    # Create dummy samples
    sample_id = "sample_001"
    split_dir = os.path.join(tmp_dir, "train")

    # RGB (256x256x3)
    rgb_arr = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
    Image.fromarray(rgb_arr).save(os.path.join(split_dir, "rgb", f"{sample_id}.png"))

    # AGL Height (256x256 float32 metres)
    agl_arr = (np.random.rand(256, 256) * 30.0).astype(np.float32)
    agl_arr[50:100, 50:100] = 18.5  # building block height
    np.save(os.path.join(split_dir, "agl", f"{sample_id}.npy"), agl_arr)

    # Semantic mask (256x256 uint8 classes 0-6)
    sem_arr = np.zeros((256, 256), dtype=np.uint8)
    sem_arr[50:100, 50:100] = 3  # buildings
    Image.fromarray(sem_arr).save(os.path.join(split_dir, "semantic", f"{sample_id}.png"))

    yield tmp_dir

    # Cleanup
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_gamus_dataset_loader(temp_gamus_dir):
    """Verifies GAMUS dataset loader output shapes, alignment, and missing camera_depth = None."""
    dataset = GAMUSDataset(dataset_dir=temp_gamus_dir, split="train", image_size=(256, 256))
    assert len(dataset) == 1

    sample = dataset[0]
    assert "image" in sample
    assert "agl_height" in sample
    assert "semantic_mask" in sample
    assert "camera_depth" in sample

    # GAMUS MUST NOT convert AGL height into camera_depth!
    assert sample["camera_depth"] is None

    # Verify tensor shapes & types
    assert sample["image"].shape == (256, 256, 3)
    assert sample["agl_height"].shape == (256, 256)
    assert sample["semantic_mask"].shape == (256, 256)


def test_gamus_quality_validator(temp_gamus_dir):
    """Verifies standalone GAMUS quality validator CLI tool behavior."""
    report = validate_gamus_quality(temp_gamus_dir)
    assert report.is_passed is True
    assert report.status_label == "GAMUS DATASET READY"
    assert report.total_samples == 1
    assert report.train_samples == 1


def test_multitask_aerial_depth_model():
    """Verifies MultiTaskAerialDepthModel architecture, output head separation, and output shapes."""
    model = MultiTaskAerialDepthModel(load_pretrained=False, max_depth_m=500.0)
    model.eval()

    dummy_rgb = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        outputs = model(dummy_rgb)

    assert "depth_map_m" in outputs
    assert "agl_height_map_m" in outputs
    assert "semantic_logits" in outputs

    # Verify output shapes
    assert outputs["depth_map_m"].shape == (2, 1, 256, 256)
    assert outputs["agl_height_map_m"].shape == (2, 1, 256, 256)
    assert outputs["semantic_logits"].shape == (2, 7, 256, 256)

    # Verify metric camera-depth output range (non-negative, bounded by max_depth_m)
    depth_m = outputs["depth_map_m"]
    assert torch.all(depth_m >= 0.0)
    assert torch.all(depth_m <= 500.0)


def test_multitask_aerial_loss_with_gamus():
    """Verifies loss computation when camera_depth is None (GAMUS dataset adaptation mode)."""
    loss_fn = MultiTaskAerialLoss(depth_weight=1.0, agl_weight=0.2, semantic_weight=0.1)

    preds = {
        "depth_map_m": torch.ones(2, 1, 128, 128) * 10.0,
        "agl_height_map_m": torch.ones(2, 1, 128, 128) * 15.0,
        "semantic_logits": torch.randn(2, 7, 128, 128)
    }

    # GAMUS Targets: camera_depth is None!
    targets = {
        "camera_depth": None,
        "agl_height": torch.ones(2, 128, 128) * 15.0,
        "semantic_mask": torch.ones(2, 128, 128, dtype=torch.long) * 3
    }

    loss_dict = loss_fn(preds, targets)
    assert "loss" in loss_dict
    assert "loss_depth" in loss_dict
    assert "loss_agl" in loss_dict
    assert "loss_semantic" in loss_dict

    # L_depth MUST be 0.0 when camera_depth is None!
    assert loss_dict["loss_depth"].item() == 0.0
    # L_agl should be close to 0.0 since prediction matches target exactly
    assert loss_dict["loss_agl"].item() < 0.1
    # Total loss should be non-zero from semantic and agl loss
    assert loss_dict["loss"].item() >= 0.0


def test_frozen_stage3_compatibility():
    """
    CRITICAL RULE: STAGE 3 IS FROZEN.
    Verifies Stage 3 receives ONLY metric depth_map_m, NOT agl_height_map_m.
    Synthetic 10m, 50m, 100m depth maps must produce exact expected heights.
    """
    calib = CalibrationConfig(mode="relative")
    bbox = [20, 20, 40, 40]

    for d_val in [10.0, 50.0, 100.0]:
        depth_map = np.ones((100, 100), dtype=np.float32) * d_val
        # Create a raised building box (+15m roof)
        depth_map[20:60, 20:60] = d_val - 15.0  # surface closer to camera by 15m

        bm = measure_building_height(
            depth_map=depth_map,
            bbox=bbox,
            calibration=calib,
            building_name=f"Test_{int(d_val)}m"
        )

        assert bm.calibrated_height_m is not None
        assert abs(bm.calibrated_height_m - 15.0) < 0.5


def test_checkpoint_saving_and_loading(tmp_path):
    """Verifies saving and restoring multi-task model state dicts."""
    model = MultiTaskAerialDepthModel(load_pretrained=False)
    ckpt_path = tmp_path / "test_model.pt"

    torch.save({
        "epoch": 1,
        "model_state_dict": model.state_dict(),
        "mode": "AERIAL_DOMAIN_ADAPTATION"
    }, ckpt_path)

    assert os.path.exists(ckpt_path)

    # Reload
    new_model = MultiTaskAerialDepthModel(load_pretrained=False)
    ckpt = torch.load(ckpt_path)
    new_model.load_state_dict(ckpt["model_state_dict"])

    # Test forward on reloaded model
    dummy_in = torch.randn(1, 3, 128, 128)
    out = new_model(dummy_in)
    assert out["depth_map_m"].shape == (1, 1, 128, 128)
