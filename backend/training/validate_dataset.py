import sys
import os
import argparse
import json

# Ensure backend root is on sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from training.dataset import validate_dataset_quality


def main():
    parser = argparse.ArgumentParser(description="DepthWizard Aerial Dataset Quality Gate Validator")
    parser.add_argument("--dataset_dir", type=str, default="dataset", help="Path to aerial dataset directory")
    parser.add_argument("--json", action="store_true", help="Output raw JSON quality report")
    args = parser.parse_args()

    report = validate_dataset_quality(args.dataset_dir)

    if args.json:
        print(json.dumps(report.model_dump(), indent=2))
        sys.exit(0 if report.is_passed else 1)

    print("\n==================================================================")
    print("      DEPTHWIZARD AERIAL METRIC DEPTH DATASET QUALITY GATE       ")
    print("==================================================================")
    print(f"Target Dataset Directory : {args.dataset_dir}")
    print(f"Quality Gate Status      : {report.status_label}")
    print(f"Dataset Sufficiency      : {report.dataset_sufficiency}")
    print(f"Sufficiency Explanation  : {report.sufficiency_reason}")
    print("------------------------------------------------------------------")
    print(f"Total Paired Samples     : {report.total_samples}")
    print(f"Total Unique Scenes      : {report.total_scenes}")
    print(f"Train Split Samples      : {report.train_samples}")
    print(f"Validation Split Samples : {report.val_samples}")
    print(f"Test Split Samples       : {report.test_split_samples}")
    print(f"Detected Depth Units     : {', '.join(report.detected_depth_units) if report.detected_depth_units else 'None'}")
    print(f"Observed Depth Range     : {report.depth_min_observed_m if report.depth_min_observed_m is not None else 'N/A'}m – {report.depth_max_observed_m if report.depth_max_observed_m is not None else 'N/A'}m")
    print(f"Mean Valid Pixel Ratio   : {report.valid_pixel_ratio_mean if report.valid_pixel_ratio_mean is not None else 'N/A'}")
    print(f"Data Leakage Detected    : {'YES (BLOCKER)' if report.leakage_detected else 'NO'}")
    print("==================================================================")

    if report.blockers:
        print("\n[CRITICAL BLOCKERS]")
        for b in report.blockers:
            print(f" [X] {b}")

    if report.warnings:
        print("\n[QUALITY WARNINGS]")
        for w in report.warnings:
            print(f" [!] {w}")

    if not report.is_passed:
        print("\n==================================================================")
        print("RESULT: TRAINING BLOCKED. NO VALID PAIRED AERIAL DATASET AVAILABLE.")
        print("Please format your aerial RGB images and LiDAR/photogrammetry depth maps")
        print("according to the schema below before attempting fine-tuning:")
        print(json.dumps(report.required_dataset_schema, indent=2))
        print("==================================================================\n")
        sys.exit(1)
    else:
        print("\n[SUCCESS] Dataset quality gates PASSED! Ready for aerial fine-tuning.\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
