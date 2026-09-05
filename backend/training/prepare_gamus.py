"""
GAMUS Dataset Preparation Script.

Handles controlled dataset directory initialization, disk space check,
manifest generation, and structure verification for GAMUS aerial data.
Does NOT automatically download massive external files without explicit confirmation.
"""

import sys
import os
import json
import shutil
import argparse
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from training.datasets.gamus_dataset import parse_gamus_manifest, GAMUS_SEMANTIC_CLASSES


def main():
    parser = argparse.ArgumentParser(description="Prepare and verify GAMUS dataset layout.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="dataset/gamus",
        help="Target GAMUS data directory (default: dataset/gamus)"
    )
    parser.add_argument(
        "--create-placeholders",
        action="store_true",
        help="Create empty structure and demo sample if dataset does not exist."
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    print("=" * 60)
    print("GAMUS DATASET PREPARATION & INITIALIZATION")
    print(f"Target Directory: {data_dir.resolve()}")
    print("=" * 60)

    # Check available disk space
    try:
        total, used, free = shutil.disk_usage(data_dir.parent if data_dir.parent.exists() else Path("."))
        free_gb = free / (1024 ** 3)
        print(f"Available Disk Space: {free_gb:.2f} GB")
        if free_gb < 10.0:
            print("WARNING: Disk space is below 10 GB. Large dataset downloads require sufficient storage.")
    except Exception as e:
        print(f"Disk space check skipped: {e}")

    # Ensure directories exist
    for split in ["train", "val", "test"]:
        (data_dir / split / "rgb").mkdir(parents=True, exist_ok=True)
        (data_dir / split / "agl").mkdir(parents=True, exist_ok=True)
        (data_dir / split / "semantic").mkdir(parents=True, exist_ok=True)

    manifest_path = data_dir / "manifest.json"
    manifest_data = []
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
        except Exception:
            manifest_data = []

    # Check existing samples
    items, err = parse_gamus_manifest(str(data_dir))
    total_samples = len(items)
    print(f"\nCurrent Dataset Samples Found: {total_samples}")

    if total_samples == 0 and args.create_placeholders:
        print("\nCreating synthetic placeholder sample for verification...")
        import numpy as np
        from PIL import Image

        sample_id = "gamus_demo_001"
        split_dir = data_dir / "train"

        # Create RGB image
        rgb_arr = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
        Image.fromarray(rgb_arr).save(split_dir / "rgb" / f"{sample_id}.png")

        # Create AGL height map (float32 metres, range 0 to 45m)
        agl_arr = (np.random.rand(256, 256) * 45.0).astype(np.float32)
        # Add a building block (25m)
        agl_arr[50:150, 50:150] = 25.0
        np.save(split_dir / "agl" / f"{sample_id}.npy", agl_arr)

        # Create Semantic mask (uint8 classes 0-6)
        sem_arr = np.ones((256, 256), dtype=np.uint8)  # ground
        sem_arr[50:150, 50:150] = 3  # buildings
        Image.fromarray(sem_arr).save(split_dir / "semantic" / f"{sample_id}.png")

        # Update manifest list
        new_entry = {
            "image": f"train/rgb/{sample_id}.png",
            "agl_height": f"train/agl/{sample_id}.npy",
            "semantic_mask": f"train/semantic/{sample_id}.png",
            "depth_unit": "meter",
            "scene_id": sample_id,
            "split": "train"
        }
        if isinstance(manifest_data, list):
            manifest_data.append(new_entry)
        else:
            manifest_data = [new_entry]

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        print(f"Created placeholder sample '{sample_id}' in train split.")

    print("\nRun 'python -m training.validate_gamus' to execute quality checks on the prepared dataset.")
    print("=" * 60)


if __name__ == "__main__":
    main()
