import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from models.depth import ValidationOverview, ValidationBenchmark, DepthProcessResponse
from services.hero_dataset import HERO_SCENES, REFERENCE_DATASET_REQUIREMENTS

# Reusable accuracy evaluation functions (Requirement 9)
def calculate_mae(gt: List[float], pred: List[float]) -> float:
    if not gt or len(gt) != len(pred): return 0.0
    return float(np.mean(np.abs(np.array(pred) - np.array(gt))))

def calculate_rmse(gt: List[float], pred: List[float]) -> float:
    if not gt or len(gt) != len(pred): return 0.0
    return float(np.sqrt(np.mean((np.array(pred) - np.array(gt)) ** 2)))

def calculate_mean_relative_error(gt: List[float], pred: List[float]) -> float:
    if not gt or len(gt) != len(pred): return 0.0
    rel_errs = np.abs(np.array(pred) - np.array(gt)) / np.maximum(1e-6, np.array(gt)) * 100.0
    return float(np.mean(rel_errs))

def calculate_median_relative_error(gt: List[float], pred: List[float]) -> float:
    if not gt or len(gt) != len(pred): return 0.0
    rel_errs = np.abs(np.array(pred) - np.array(gt)) / np.maximum(1e-6, np.array(gt)) * 100.0
    return float(np.median(rel_errs))

def calculate_accuracy_score(gt: List[float], pred: List[float]) -> float:
    mre = calculate_mean_relative_error(gt, pred)
    return float(max(0.0, 100.0 - mre))

def calculate_pct_within_threshold(gt: List[float], pred: List[float], threshold_m: float) -> float:
    if not gt or len(gt) != len(pred): return 0.0
    errs = np.abs(np.array(pred) - np.array(gt))
    within = errs <= threshold_m
    return float(np.mean(within) * 100.0)

def build_validation_overview(
    scene_results: List[Tuple[str, DepthProcessResponse]]
) -> ValidationOverview:
    """
    Computes validation metrics ENTIRELY from live pipeline predictions.
    
    Metrics: MAE, RMSE, Median Error, Mean Relative Error, Accuracy, and % within 2m/5m/10m.
    """
    benchmarks: List[ValidationBenchmark] = []
    all_gt: List[float] = []
    all_pred: List[float] = []
    all_abs_err: List[float] = []
    total_structures = 0
    scatter_points: List[Dict[str, Any]] = []

    for scene_id, result in scene_results:
        scene_meta = HERO_SCENES.get(scene_id, {})
        scene_gt: List[float] = []
        scene_pred: List[float] = []
        scene_measurements: List[Dict[str, Any]] = []

        for b in result.sample_buildings:
            gt_h = b.ground_truth_height_m
            pred_h = b.calibrated_height_m

            if gt_h is None or pred_h is None or not b.is_metric_available or gt_h <= 0:
                continue

            err = abs(pred_h - gt_h)
            rel_err_pct = (err / gt_h) * 100.0
            acc_pct = max(0.0, 100.0 - rel_err_pct)

            scene_gt.append(gt_h)
            scene_pred.append(pred_h)
            all_gt.append(gt_h)
            all_pred.append(pred_h)
            all_abs_err.append(err)
            total_structures += 1

            scene_measurements.append({
                "building_id": b.id,
                "building_name": b.name,
                "relative_depth": b.relative_depth,
                "gt_height_m": round(gt_h, 2),
                "pred_height_m": round(pred_h, 2),
                "error_m": round(err, 2),
                "rel_error_pct": round(rel_err_pct, 2),
                "accuracy_pct": round(acc_pct, 2),
                "confidence_pct": b.confidence_pct,
            })

            scatter_points.append({
                "scene": scene_meta.get("title", scene_id),
                "scene_id": scene_id,
                "building": b.name,
                "gt_height": round(gt_h, 1),
                "pred_height": round(pred_h, 1),
                "error": round(err, 2),
                "rel_error_pct": round(rel_err_pct, 2),
                "accuracy_pct": round(acc_pct, 2),
                "sensor": scene_meta.get("sensor", "Unknown sensor"),
            })

        if scene_gt:
            mae = calculate_mae(scene_gt, scene_pred)
            rmse = calculate_rmse(scene_gt, scene_pred)
            med_err = float(np.median(np.abs(np.array(scene_pred) - np.array(scene_gt))))
            mre = calculate_mean_relative_error(scene_gt, scene_pred)
            p2m = calculate_pct_within_threshold(scene_gt, scene_pred, 2.0)
            p5m = calculate_pct_within_threshold(scene_gt, scene_pred, 5.0)
            p10m = calculate_pct_within_threshold(scene_gt, scene_pred, 10.0)
        else:
            mae = rmse = med_err = mre = p2m = p5m = p10m = 0.0

        benchmarks.append(ValidationBenchmark(
            scene_id=scene_id,
            scene_name=scene_meta.get("title", scene_id),
            sensor_type=scene_meta.get("view_angle", "Unknown geometry"),
            lidar_source=scene_meta.get("sensor", "Unknown sensor"),
            buildings_tested=len(scene_measurements),
            mae_m=round(mae, 2),
            rmse_m=round(rmse, 2),
            median_err_m=round(med_err, 2),
            mean_relative_error_pct=round(mre, 1),
            pct_within_2m=round(p2m, 1),
            pct_within_5m=round(p5m, 1),
            pct_within_10m=round(p10m, 1),
            sample_measurements=scene_measurements,
        ))

    if all_gt:
        mae = calculate_mae(all_gt, all_pred)
        rmse = calculate_rmse(all_gt, all_pred)
        med_err = float(np.median(all_abs_err))
        mre = calculate_mean_relative_error(all_gt, all_pred)
        op2m = calculate_pct_within_threshold(all_gt, all_pred, 2.0)
        op5m = calculate_pct_within_threshold(all_gt, all_pred, 5.0)
        op10m = calculate_pct_within_threshold(all_gt, all_pred, 10.0)

        return ValidationOverview(
            data_available=True,
            unavailable_reason=None,
            requirements=[],
            total_scenes_evaluated=len(benchmarks),
            total_structures_measured=total_structures,
            overall_mae_m=round(mae, 2),
            overall_rmse_m=round(rmse, 2),
            overall_median_err_m=round(med_err, 2),
            overall_mre_pct=round(mre, 1),
            overall_pct_within_2m=round(op2m, 1),
            overall_pct_within_5m=round(op5m, 1),
            overall_pct_within_10m=round(op10m, 1),
            benchmarks=benchmarks,
            scatter_points=scatter_points,
        )

    return ValidationOverview(
        data_available=False,
        unavailable_reason=(
            "Validation data not available. No verified reference-height dataset is "
            "registered for the current imagery, so absolute error against ground "
            "truth cannot be computed. Depth, 3D reconstruction and reference-based "
            "calibration remain fully functional."
        ),
        requirements=REFERENCE_DATASET_REQUIREMENTS,
        total_scenes_evaluated=0,
        total_structures_measured=0,
        overall_mae_m=None,
        overall_rmse_m=None,
        overall_median_err_m=None,
        overall_mre_pct=None,
        overall_pct_within_2m=None,
        overall_pct_within_5m=None,
        overall_pct_within_10m=None,
        benchmarks=[],
        scatter_points=[],
    )

