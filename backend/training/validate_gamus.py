"""
GAMUS Dataset Quality Validator CLI.

Executes standalone validation of the GAMUS dataset structure, checking:
- Total, train, validation, and test sample counts
- RGB image dimensions and formats
- AGL height statistics (min, max, mean, median, valid ratio, unit check)
- Semantic class distribution and pixel counts
- Missing or corrupted files
- RGB/AGL/Semantic spatial alignment
- Dataset quality status output
"""

import sys
import json
import argparse
from pathlib import Path

# Add backend directory to sys.path if needed
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from training.datasets.gamus_dataset import validate_gamus_quality


def main():
    parser = argparse.ArgumentParser(description="Validate GAMUS dataset quality and statistics.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="dataset/gamus",
        help="Path to GAMUS dataset directory (default: dataset/gamus)"
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="reports/gamus_quality_report.json",
        help="Path to save JSON validation report (default: reports/gamus_quality_report.json)"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    print("=" * 60)
    print(f"GAMUS DATASET QUALITY VALIDATOR")
    print(f"Target Directory: {data_dir.resolve()}")
    print("=" * 60)

    report_obj = validate_gamus_quality(str(data_dir))
    report = report_obj.model_dump() if hasattr(report_obj, "model_dump") else dict(report_obj)

    # Pretty print summary
    print(f"\n--- VALIDATION SUMMARY ---")
    print(f"Status:             {report.get('status_label', 'UNKNOWN')}")
    print(f"Passed Quality Gate: {report.get('is_passed', False)}")
    print(f"Total Samples:      {report.get('total_samples', 0)}")
    print(f"  - Train:          {report.get('train_samples', 0)}")
    print(f"  - Val:            {report.get('val_samples', 0)}")
    print(f"  - Test:           {report.get('test_split_samples', 0)}")

    print(f"\nAGL Height Statistics:")
    print(f"  - Min:            {report.get('observed_agl_min_m', 'N/A')} m")
    print(f"  - Max:            {report.get('observed_agl_max_m', 'N/A')} m")
    print(f"  - Mean:           {report.get('observed_agl_mean_m', 'N/A')} m")
    print(f"  - Valid Ratio:    {(report.get('valid_agl_ratio_mean', 0.0) or 0.0) * 100:.2f}%")

    sem_classes = report.get("detected_semantic_classes", [])
    print(f"\nSemantic Classes Detected: {len(sem_classes)}")
    print(f"  - Class IDs:      {sem_classes}")

    if report.get("warnings"):
        print(f"\nWarnings ({len(report['warnings'])}):")
        for w in report["warnings"]:
            print(f"  ! {w}")

    if report.get("blockers"):
        print(f"\nBlockers ({len(report['blockers'])}):")
        for e in report["blockers"]:
            print(f"  X {e}")

    # Save output JSON
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved quality report to: {output_path.resolve()}")
    print("=" * 60)

    if report.get("is_passed", False):
        print("RESULT: GAMUS Dataset Quality Check PASSED")
        sys.exit(0)
    else:
        print(f"RESULT: GAMUS Dataset Quality Check FAILED ({report.get('status_label')})")
        sys.exit(1)


if __name__ == "__main__":
    main()
