import os
import json
import logging
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

GAMUS_SEMANTIC_CLASSES = {
    0: "others",
    1: "ground",
    2: "low_vegetation",
    3: "buildings",
    4: "water",
    5: "road",
    6: "tree"
}


class GAMUSManifestItem(BaseModel):
    image: str = Field(description="Relative path to RGB aerial image")
    agl_height: str = Field(description="Relative path to AGL height map (.tif, .npy, .npz)")
    semantic_mask: Optional[str] = Field(default=None, description="Relative path to semantic class map")
    depth_unit: str = Field(default="meter", description="AGL height unit (meter)")
    scene_id: Optional[str] = None
    building_id: Optional[str] = None
    split: str = Field(default="train", description="train, val, or test")


class GAMUSQualityReport(BaseModel):
    is_passed: bool
    status_label: str
    dataset_sufficiency: str = Field(description="LOW, MEDIUM, GOOD, or NONE")
    sufficiency_reason: str
    total_samples: int = 0
    total_scenes: int = 0
    train_samples: int = 0
    val_samples: int = 0
    test_split_samples: int = 0
    observed_agl_min_m: Optional[float] = None
    observed_agl_max_m: Optional[float] = None
    observed_agl_mean_m: Optional[float] = None
    valid_agl_ratio_mean: Optional[float] = None
    detected_semantic_classes: List[int] = Field(default_factory=list)
    leakage_detected: bool = False
    warnings: List[str] = Field(default_factory=list)
    blockers: List[str] = Field(default_factory=list)
    required_schema: Dict[str, Any] = Field(default_factory=dict)


def parse_gamus_manifest(dataset_dir: str) -> Tuple[List[GAMUSManifestItem], Optional[str]]:
    """Parses manifest.json or scans GAMUS directory structure."""
    if not os.path.exists(dataset_dir):
        return [], f"GAMUS dataset directory '{dataset_dir}' does not exist."

    manifest_path = os.path.join(dataset_dir, "manifest.json")
    items: List[GAMUSManifestItem] = []

    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, list):
                for d in raw:
                    items.append(GAMUSManifestItem(**d))
            elif isinstance(raw, dict) and "samples" in raw:
                for d in raw["samples"]:
                    items.append(GAMUSManifestItem(**d))
            return items, None
        except Exception as exc:
            return [], f"Failed to parse GAMUS manifest.json: {str(exc)}"

    # Directory auto-scanner for GAMUS structure
    found_any = False
    for split in ["train", "val", "test"]:
        img_dir = None
        for cand in ["images", "rgb"]:
            p = os.path.join(dataset_dir, split, cand)
            if os.path.exists(p):
                img_dir = p
                break

        agl_dir = os.path.join(dataset_dir, split, "agl")

        sem_dir = None
        for cand in ["semantics", "semantic"]:
            p = os.path.join(dataset_dir, split, cand)
            if os.path.exists(p):
                sem_dir = p
                break

        if img_dir and os.path.exists(agl_dir):
            found_any = True
            for fname in os.listdir(img_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
                    base = os.path.splitext(fname)[0]
                    matched_agl = None
                    for ext in [".npy", ".npz", ".tif", ".tiff", ".png"]:
                        cand = os.path.join(agl_dir, base + ext)
                        if os.path.exists(cand):
                            matched_agl = os.path.relpath(cand, dataset_dir)
                            break

                    matched_sem = None
                    if sem_dir and os.path.exists(sem_dir):
                        for ext in [".png", ".tif", ".npy"]:
                            cand_sem = os.path.join(sem_dir, base + ext)
                            if os.path.exists(cand_sem):
                                matched_sem = os.path.relpath(cand_sem, dataset_dir)
                                break

                    if matched_agl:
                        items.append(GAMUSManifestItem(
                            image=os.path.relpath(os.path.join(img_dir, fname), dataset_dir),
                            agl_height=matched_agl,
                            semantic_mask=matched_sem,
                            scene_id=f"gamus_{split}_{base}",
                            split=split
                        ))

    if not items:
        if not found_any:
            return [], f"No paired GAMUS 'images' and 'agl' subdirectories or manifest.json found in '{dataset_dir}'."
        return [], f"No matching GAMUS RGB and AGL height map pairs found in '{dataset_dir}'."

    return items, None


def load_array_file(file_path: str) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Loads 2D numerical array from .npy, .npz, .tif, .tiff, or .png file."""
    if not os.path.exists(file_path):
        return None, f"File not found: {file_path}"
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".npy":
            return np.load(file_path).astype(np.float32), None
        elif ext == ".npz":
            npz = np.load(file_path)
            return npz[npz.files[0]].astype(np.float32), None
        elif ext in [".tif", ".tiff", ".png"]:
            img = Image.open(file_path)
            return np.array(img, dtype=np.float32), None
        else:
            return None, f"Unsupported array format '{ext}'."
    except Exception as exc:
        return None, f"Failed to load array file '{file_path}': {str(exc)}"


def validate_gamus_quality(dataset_dir: str = "dataset/gamus") -> GAMUSQualityReport:
    """Executes quality gate checks for GAMUS aerial domain adaptation dataset."""
    schema = {
        "dataset_name": "GAMUS Remote-Sensing Aerial Dataset",
        "structure": {
            "root_dir": dataset_dir,
            "manifest": "dataset/gamus/manifest.json (or dataset/gamus/{train,val,test}/{images,agl,semantics})",
            "image_format": "RGB (.jpg, .png, .tif)",
            "agl_format": "AGL height map in metres (.tif, .npy)",
            "semantics_format": "Semantic class integer map (0:others, 1:ground, 2:low_veg, 3:buildings, 4:water, 5:road, 6:tree)"
        }
    }

    items, parse_err = parse_gamus_manifest(dataset_dir)
    if parse_err or not items:
        return GAMUSQualityReport(
            is_passed=False,
            status_label="GAMUS DATASET MISSING / BLOCKED",
            dataset_sufficiency="NONE",
            sufficiency_reason=parse_err or f"GAMUS dataset directory '{dataset_dir}' not found.",
            blockers=[parse_err or f"No paired GAMUS RGB + AGL samples found in '{dataset_dir}'."],
            warnings=["Please populate 'dataset/gamus/manifest.json' or download GAMUS dataset."],
            required_schema=schema
        )

    blockers = []
    warnings = []
    train_scenes = set()
    val_scenes = set()
    test_scenes = set()
    found_classes = set()
    agl_mins = []
    agl_maxs = []
    agl_means = []
    valid_ratios = []

    train_c = val_c = test_c = 0

    for idx, item in enumerate(items):
        full_img = os.path.join(dataset_dir, item.image)
        full_agl = os.path.join(dataset_dir, item.agl_height)

        if not os.path.exists(full_img):
            blockers.append(f"Sample {idx}: RGB image file missing at '{item.image}'.")
            continue
        if not os.path.exists(full_agl):
            blockers.append(f"Sample {idx}: AGL height file missing at '{item.agl_height}'.")
            continue

        agl_arr, err = load_array_file(full_agl)
        if err or agl_arr is None:
            blockers.append(f"Sample {idx}: {err}")
            continue

        valid = np.isfinite(agl_arr) & (agl_arr >= 0.0)
        v_ratio = float(np.mean(valid))
        valid_ratios.append(v_ratio)

        if valid.any():
            agl_mins.append(float(np.min(agl_arr[valid])))
            agl_maxs.append(float(np.max(agl_arr[valid])))
            agl_means.append(float(np.mean(agl_arr[valid])))

        # Check semantic map
        if item.semantic_mask:
            full_sem = os.path.join(dataset_dir, item.semantic_mask)
            if os.path.exists(full_sem):
                sem_arr, s_err = load_array_file(full_sem)
                if sem_arr is not None:
                    u_cls = set(np.unique(sem_arr.astype(int)).tolist())
                    found_classes.update(u_cls)

        scene_k = item.scene_id or item.image
        if item.split == "train":
            train_c += 1
            train_scenes.add(scene_k)
        elif item.split == "val":
            val_c += 1
            val_scenes.add(scene_k)
        elif item.split == "test":
            test_c += 1
            test_scenes.add(scene_k)

    leakage = False
    overlap = train_scenes.intersection(val_scenes | test_scenes)
    if overlap:
        leakage = True
        blockers.append(f"SPLIT LEAKAGE: Scenes {overlap} present in multiple splits.")

    total_scenes = len(train_scenes | val_scenes | test_scenes)
    total_samples = len(items)

    if total_scenes < 50:
        sufficiency = "LOW"
        suff_reason = f"GAMUS contains {total_scenes} unique scenes (< 50)."
    elif total_scenes < 200:
        sufficiency = "MEDIUM"
        suff_reason = f"GAMUS contains {total_scenes} unique scenes (50–200)."
    else:
        sufficiency = "GOOD"
        suff_reason = f"GAMUS contains {total_scenes} unique scenes (> 200)."

    is_passed = len(blockers) == 0 and total_samples > 0

    return GAMUSQualityReport(
        is_passed=is_passed,
        status_label="GAMUS DATASET READY" if is_passed else "GAMUS QUALITY GATE FAILURES",
        dataset_sufficiency=sufficiency,
        sufficiency_reason=suff_reason,
        total_samples=total_samples,
        total_scenes=total_scenes,
        train_samples=train_c,
        val_samples=val_c,
        test_split_samples=test_c,
        observed_agl_min_m=round(float(min(agl_mins)), 2) if agl_mins else None,
        observed_agl_max_m=round(float(max(agl_maxs)), 2) if agl_maxs else None,
        observed_agl_mean_m=round(float(np.mean(agl_means)), 2) if agl_means else None,
        valid_agl_ratio_mean=round(float(np.mean(valid_ratios)), 3) if valid_ratios else None,
        detected_semantic_classes=list(found_classes),
        leakage_detected=leakage,
        warnings=warnings,
        blockers=blockers,
        required_schema=schema
    )


class GAMUSDataset:
    """
    PyTorch Dataset implementation for GAMUS Aerial RGB, AGL Height, and Semantic Labels.
    Explicitly outputs camera_depth = None to maintain semantic separation.
    """
    def __init__(self, dataset_dir: str = "dataset/gamus", split: str = "train", image_size: Tuple[int, int] = (518, 518)):
        self.dataset_dir = dataset_dir
        self.split = split
        self.image_size = image_size

        items, _ = parse_gamus_manifest(dataset_dir)
        self.items = [it for it in items if it.split == split]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        full_img = os.path.join(self.dataset_dir, item.image)
        full_agl = os.path.join(self.dataset_dir, item.agl_height)

        pil_img = Image.open(full_img).convert("RGB")
        w_orig, h_orig = pil_img.size
        pil_img_resized = pil_img.resize(self.image_size, Image.Resampling.BILINEAR)
        img_np = np.array(pil_img_resized, dtype=np.float32) / 255.0

        agl_raw, _ = load_array_file(full_agl)
        if agl_raw is None:
            agl_raw = np.zeros((h_orig, w_orig), dtype=np.float32)

        agl_pil = Image.fromarray(agl_raw)
        agl_resized = np.array(agl_pil.resize(self.image_size, Image.Resampling.BICUBIC), dtype=np.float32)
        valid_agl_mask = np.isfinite(agl_resized) & (agl_resized >= 0.0)

        sem_resized = None
        if item.semantic_mask:
            full_sem = os.path.join(self.dataset_dir, item.semantic_mask)
            if os.path.exists(full_sem):
                sem_raw, _ = load_array_file(full_sem)
                if sem_raw is not None:
                    sem_pil = Image.fromarray(sem_raw.astype(np.uint8))
                    sem_resized = np.array(sem_pil.resize(self.image_size, Image.Resampling.NEAREST), dtype=np.int64)

        if sem_resized is None:
            sem_resized = np.full(self.image_size, -1, dtype=np.int64)

        return {
            "image": img_np,  # [H, W, 3] float32
            "agl_height": agl_resized,  # [H, W] float32 metres AGL
            "valid_agl_mask": valid_agl_mask,  # [H, W] bool
            "semantic_mask": sem_resized,  # [H, W] int64 classes (0..6 or -1)
            "camera_depth": None,  # EXPLICITLY NONE for GAMUS samples!
            "scene_id": item.scene_id or str(idx)
        }
