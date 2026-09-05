import os
import json
import logging
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AerialDepthManifestItem(BaseModel):
    image: str = Field(description="Relative path to RGB aerial image")
    depth: str = Field(description="Relative path to metric depth map (.tif, .npy, .npz, .png)")
    mask: Optional[str] = Field(default=None, description="Optional building or valid region mask")
    depth_unit: str = Field(default="meter", description="meter, millimeter, or relative")
    depth_type: str = Field(default="camera_surface_depth", description="camera_surface_depth, dsm, dtm, or building_height")
    camera_altitude_m: Optional[float] = None
    focal_length_px: Optional[float] = None
    sensor_width_mm: Optional[float] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    source: Optional[str] = None
    scene_id: Optional[str] = None
    building_id: Optional[str] = None
    split: str = Field(default="train", description="train, val, or test")


class QualityGateReport(BaseModel):
    is_passed: bool
    status_label: str
    dataset_sufficiency: str = Field(description="LOW, MEDIUM, or GOOD")
    sufficiency_reason: str
    total_samples: int = 0
    total_scenes: int = 0
    train_samples: int = 0
    val_samples: int = 0
    test_split_samples: int = 0
    detected_depth_units: List[str] = Field(default_factory=list)
    depth_min_observed_m: Optional[float] = None
    depth_max_observed_m: Optional[float] = None
    valid_pixel_ratio_mean: Optional[float] = None
    leakage_detected: bool = False
    warnings: List[str] = Field(default_factory=list)
    blockers: List[str] = Field(default_factory=list)
    required_dataset_schema: Dict[str, Any] = Field(default_factory=dict)


def parse_dataset_manifest(dataset_dir: str) -> Tuple[List[AerialDepthManifestItem], Optional[str]]:
    """
    Parses manifest.json or scans dataset directory for paired aerial RGB and metric depth images.
    Returns (manifest_items, error_reason).
    """
    if not os.path.exists(dataset_dir):
        return [], f"Dataset directory '{dataset_dir}' does not exist."

    manifest_path = os.path.join(dataset_dir, "manifest.json")
    items: List[AerialDepthManifestItem] = []

    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            if isinstance(raw_data, list):
                for d in raw_data:
                    items.append(AerialDepthManifestItem(**d))
            elif isinstance(raw_data, dict) and "samples" in raw_data:
                for d in raw_data["samples"]:
                    items.append(AerialDepthManifestItem(**d))
            return items, None
        except Exception as exc:
            return [], f"Failed to parse manifest.json: {str(exc)}"

    # Directory auto-scan for train/val/test splits if manifest.json is absent
    found_any = False
    for split in ["train", "val", "test"]:
        img_dir = os.path.join(dataset_dir, split, "images")
        depth_dir = os.path.join(dataset_dir, split, "depths")
        if os.path.exists(img_dir) and os.path.exists(depth_dir):
            found_any = True
            for fname in os.listdir(img_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
                    base_name = os.path.splitext(fname)[0]
                    # Look for matching depth file with same basename
                    matched_depth = None
                    for ext in [".npy", ".npz", ".tif", ".tiff", ".png"]:
                        cand = os.path.join(depth_dir, base_name + ext)
                        if os.path.exists(cand):
                            matched_depth = os.path.relpath(cand, dataset_dir)
                            break
                    if matched_depth:
                        items.append(AerialDepthManifestItem(
                            image=os.path.relpath(os.path.join(img_dir, fname), dataset_dir),
                            depth=matched_depth,
                            scene_id=f"{split}_{base_name}",
                            split=split
                        ))

    if not items:
        if not found_any:
            return [], f"No paired 'images' and 'depths' subdirectories or manifest.json found in '{dataset_dir}'."
        return [], f"No matching RGB image and depth file pairs found in '{dataset_dir}'."

    return items, None


def load_depth_file(full_depth_path: str) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Loads raw depth array from .npy, .npz, .tif, .tiff, or .png file."""
    if not os.path.exists(full_depth_path):
        return None, f"File not found: {full_depth_path}"

    ext = os.path.splitext(full_depth_path)[1].lower()
    try:
        if ext == ".npy":
            arr = np.load(full_depth_path).astype(np.float32)
            return arr, None
        elif ext == ".npz":
            npz = np.load(full_depth_path)
            key = npz.files[0]
            return npz[key].astype(np.float32), None
        elif ext in [".tif", ".tiff"]:
            img = Image.open(full_depth_path)
            arr = np.array(img, dtype=np.float32)
            return arr, None
        elif ext == ".png":
            img = Image.open(full_depth_path)
            arr = np.array(img, dtype=np.float32)
            return arr, None
        else:
            return None, f"Unsupported depth file format '{ext}'."
    except Exception as exc:
        return None, f"Failed to load depth map '{full_depth_path}': {str(exc)}"


def validate_dataset_quality(dataset_dir: str) -> QualityGateReport:
    """
    Executes research-grade dataset quality gates.
    Validates RGB/depth pairing, depth units, alignment, split leakage, and sufficiency.
    """
    schema_reqs = {
        "required_structure": {
            "dataset_dir": dataset_dir,
            "manifest_file": "dataset/manifest.json (or dataset/{train,val,test}/{images,depths})",
            "image_formats": [".jpg", ".jpeg", ".png", ".tif"],
            "depth_formats": [".npy", ".npz", ".tif", ".tiff"],
            "depth_units": "meters (float32 camera-to-surface depth)"
        },
        "manifest_schema": {
            "image": "relative path to RGB image",
            "depth": "relative path to metric depth map",
            "depth_unit": "meter",
            "depth_type": "camera_surface_depth",
            "scene_id": "unique scene/location ID for leak-free splitting",
            "split": "train | val | test"
        }
    }

    items, parse_err = parse_dataset_manifest(dataset_dir)

    if parse_err or not items:
        return QualityGateReport(
            is_passed=False,
            status_label="TRAINING BLOCKED: NO VALID PAIRED AERIAL METRIC-DEPTH DATASET AVAILABLE",
            dataset_sufficiency="NONE",
            sufficiency_reason=parse_err or "No paired aerial RGB and metric depth images found in workspace.",
            blockers=[parse_err or f"No paired aerial RGB and metric depth samples available in '{dataset_dir}'."],
            warnings=["Please populate 'dataset/manifest.json' or 'dataset/train/images' + 'dataset/train/depths' with paired LiDAR/photogrammetry aerial depth maps."],
            required_dataset_schema=schema_reqs
        )

    blockers = []
    warnings = []
    train_scenes = set()
    val_scenes = set()
    test_scenes = set()
    detected_units = set()
    observed_mins = []
    observed_maxs = []
    valid_ratios = []

    train_count = 0
    val_count = 0
    test_count = 0

    for idx, item in enumerate(items):
        full_img_path = os.path.join(dataset_dir, item.image)
        full_depth_path = os.path.join(dataset_dir, item.depth)

        if not os.path.exists(full_img_path):
            blockers.append(f"Sample {idx}: RGB image file missing at '{item.image}'.")
            continue
        if not os.path.exists(full_depth_path):
            blockers.append(f"Sample {idx}: Depth file missing at '{item.depth}'.")
            continue

        # Check depth map content
        d_arr, err_msg = load_depth_file(full_depth_path)
        if err_msg or d_arr is None:
            blockers.append(f"Sample {idx}: {err_msg}")
            continue

        d_min = float(np.nanmin(d_arr))
        d_max = float(np.nanmax(d_arr))
        observed_mins.append(d_min)
        observed_maxs.append(d_max)

        valid_mask = np.isfinite(d_arr) & (d_arr > 0.1)
        v_ratio = float(np.mean(valid_mask))
        valid_ratios.append(v_ratio)

        # Depth unit checks
        if d_min >= 0.0 and d_max <= 1.05 and item.depth_unit != "relative":
            warnings.append(f"DEPTH_UNIT_WARNING: Sample {idx} ({item.depth}) appears normalized [0, 1]. Metric depth in metres required.")
            detected_units.add("normalized_0_1")
        elif d_max > 5000.0 and item.depth_unit != "millimeter":
            warnings.append(f"POSSIBLE_MILLIMETER_UNIT: Sample {idx} ({item.depth}) has max depth {d_max:.1f} > 5000. Values may be in millimeters.")
            detected_units.add("millimeter")
        else:
            detected_units.add(item.depth_unit or "meter")

        # Spatial alignment check
        try:
            pil_img = Image.open(full_img_path)
            w_img, h_img = pil_img.size
            if (d_arr.shape[1], d_arr.shape[0]) != (w_img, h_img):
                warnings.append(f"Sample {idx}: RGB size ({w_img}x{h_img}) differs from depth size ({d_arr.shape[1]}x{d_arr.shape[0]}). Resampling will be applied.")
        except Exception:
            pass

        # Split scene tracking
        scene_key = item.scene_id or item.building_id or item.image
        if item.split == "train":
            train_count += 1
            train_scenes.add(scene_key)
        elif item.split == "val":
            val_count += 1
            val_scenes.add(scene_key)
        elif item.split == "test":
            test_count += 1
            test_scenes.add(scene_key)

    # Check split leakage
    overlap_train_val = train_scenes.intersection(val_scenes)
    overlap_train_test = train_scenes.intersection(test_scenes)
    leakage = False
    if overlap_train_val or overlap_train_test:
        leakage = True
        blockers.append(f"DATASET LEAKAGE DETECTED: Scenes {overlap_train_val | overlap_train_test} appear in both train and val/test splits.")

    total_scenes = len(train_scenes | val_scenes | test_scenes)
    total_samples = len(items)

    if total_scenes < 50:
        sufficiency = "LOW"
        suff_reason = f"Dataset contains {total_scenes} unique scenes (< 50). Generalization to diverse aerial perspective may be limited."
    elif total_scenes < 200:
        sufficiency = "MEDIUM"
        suff_reason = f"Dataset contains {total_scenes} unique scenes (50–200). Moderate aerial scene diversity."
    else:
        sufficiency = "GOOD"
        suff_reason = f"Dataset contains {total_scenes} unique scenes (> 200). Strong aerial diversity."

    is_passed = len(blockers) == 0 and total_samples > 0

    return QualityGateReport(
        is_passed=is_passed,
        status_label="PASSED: DATASET READY FOR AERIAL FINE-TUNING" if is_passed else "TRAINING BLOCKED: DATASET QUALITY GATE FAILURES",
        dataset_sufficiency=sufficiency,
        sufficiency_reason=suff_reason,
        total_samples=total_samples,
        total_scenes=total_scenes,
        train_samples=train_count,
        val_samples=val_count,
        test_split_samples=test_count,
        detected_depth_units=list(detected_units),
        depth_min_observed_m=round(float(min(observed_mins)), 2) if observed_mins else None,
        depth_max_observed_m=round(float(max(observed_maxs)), 2) if observed_maxs else None,
        valid_pixel_ratio_mean=round(float(np.mean(valid_ratios)), 3) if valid_ratios else None,
        leakage_detected=leakage,
        warnings=warnings[:10],
        blockers=blockers,
        required_dataset_schema=schema_reqs
    )


class AerialDepthDataset:
    """
    PyTorch Dataset implementation for paired Aerial RGB and Metric Depth training.
    Applies valid pixel masking and synchronized spatial transformations.
    """
    def __init__(self, dataset_dir: str, split: str = "train", image_size: Tuple[int, int] = (518, 518), min_depth_m: float = 0.5, max_depth_m: float = 500.0):
        self.dataset_dir = dataset_dir
        self.split = split
        self.image_size = image_size
        self.min_depth_m = min_depth_m
        self.max_depth_m = max_depth_m

        items, _ = parse_dataset_manifest(dataset_dir)
        self.items = [it for it in items if it.split == split]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        full_img_path = os.path.join(self.dataset_dir, item.image)
        full_depth_path = os.path.join(self.dataset_dir, item.depth)

        pil_img = Image.open(full_img_path).convert("RGB")
        w_orig, h_orig = pil_img.size
        pil_img_resized = pil_img.resize(self.image_size, Image.Resampling.BILINEAR)
        img_np = np.array(pil_img_resized, dtype=np.float32) / 255.0

        d_raw, _ = load_depth_file(full_depth_path)
        if d_raw is None:
            d_raw = np.full((h_orig, w_orig), 10.0, dtype=np.float32)

        # Convert units if needed
        if item.depth_unit == "millimeter":
            d_raw = d_raw / 1000.0

        d_pil = Image.fromarray(d_raw)
        d_resized = np.array(d_pil.resize(self.image_size, Image.Resampling.BICUBIC), dtype=np.float32)

        valid_mask = np.isfinite(d_resized) & (d_resized >= self.min_depth_m) & (d_resized <= self.max_depth_m)

        return {
            "image": img_np,  # [H, W, 3] float32
            "depth": d_resized,  # [H, W] float32 metres
            "valid_mask": valid_mask,  # [H, W] bool
            "scene_id": item.scene_id or str(idx)
        }
