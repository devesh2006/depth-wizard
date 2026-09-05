import os
import sys
import json
import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from training.datasets.gamus_dataset import validate_gamus_quality
from training.dataset import validate_dataset_quality
from training.train_aerial_depth import train_aerial_depth

router = APIRouter(prefix="/training", tags=["Phase 2.1 Multi-Task Aerial Training"])

logger = logging.getLogger(__name__)

TRAINING_JOB_STATE: Dict[str, Any] = {
    "status": "TRAINING IDLE",
    "mode": "AERIAL_DOMAIN_ADAPTATION",
    "is_running": False,
    "current_epoch": 0,
    "total_epochs": 3,
    "progress_pct": 0.0,
    "last_run_summary": None,
    "error_message": None
}


class ValidateDatasetRequest(BaseModel):
    dataset_dir: str = Field(default="dataset/gamus", description="Path to dataset directory")


class StartTrainingRequest(BaseModel):
    config_path: str = Field(default="configs/aerial_depth_training.yaml", description="Path to training config YAML")
    mode: str = Field(default="AERIAL_DOMAIN_ADAPTATION", description="ZERO_SHOT, AERIAL_DOMAIN_ADAPTATION, or AERIAL_METRIC_FINE_TUNING")


@router.get("/status")
async def get_training_status():
    """
    Returns complete Phase 2.1 dataset matrix, mode availability, and training job state.
    """
    gamus_report = validate_gamus_quality("dataset/gamus")
    metric_report = validate_dataset_quality("dataset/metric_depth")

    gamus_available = gamus_report["status"] == "VALID"
    metric_available = metric_report.is_passed

    state = dict(TRAINING_JOB_STATE)

    modes_availability = {
        "ZERO_SHOT": {
            "status": "AVAILABLE",
            "requirements": ["Pretrained DA-V2 Large Backbone"],
            "description": "Baseline pretrained model evaluation without training."
        },
        "AERIAL_DOMAIN_ADAPTATION": {
            "status": "AVAILABLE" if gamus_available else "BLOCKED",
            "requirements": ["RGB Image", "AGL Height Map", "Semantic Mask (GAMUS)"],
            "description": "Adapts shared feature representations using GAMUS AGL and semantic supervision.",
            "blocked_reason": None if gamus_available else "GAMUS dataset missing or invalid."
        },
        "AERIAL_METRIC_FINE_TUNING": {
            "status": "AVAILABLE" if metric_available else "BLOCKED",
            "requirements": ["RGB Image", "AGL Height Map", "Semantic Mask", "True Camera-Surface Metric Depth"],
            "description": "Supervises camera-surface metric depth using genuine paired metric depth dataset.",
            "blocked_reason": None if metric_available else "No paired camera-surface metric depth dataset registered in 'dataset/metric_depth/'"
        }
    }

    report_path = "reports/aerial_training_report.json"
    last_report = None
    if os.path.exists(report_path):
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                last_report = json.load(f)
        except Exception:
            pass

    return {
        "training_state": state,
        "datasets": {
            "gamus": {
                "status": "AVAILABLE" if gamus_available else "MISSING",
                "details": gamus_report
            },
            "metric_depth": {
                "status": "AVAILABLE" if metric_available else "MISSING",
                "details": metric_report.model_dump()
            }
        },
        "modes_availability": modes_availability,
        "last_report": last_report
    }


@router.post("/validate-gamus")
async def validate_gamus_endpoint(request: ValidateDatasetRequest):
    """Runs quality validation on GAMUS dataset directory."""
    return validate_gamus_quality(request.dataset_dir)


def _run_training_background_task(config_path: str, mode: str):
    global TRAINING_JOB_STATE
    TRAINING_JOB_STATE["is_running"] = True
    TRAINING_JOB_STATE["status"] = "TRAINING_RUNNING"
    TRAINING_JOB_STATE["mode"] = mode
    TRAINING_JOB_STATE["error_message"] = None

    try:
        res = train_aerial_depth(config_path, mode_override=mode)
        TRAINING_JOB_STATE["last_run_summary"] = res
        if res.get("status") == "TRAINING_COMPLETED" or res.get("status") == "ZERO_SHOT_EVALUATION_COMPLETE":
            TRAINING_JOB_STATE["status"] = res["status"]
            TRAINING_JOB_STATE["progress_pct"] = 100.0
        else:
            TRAINING_JOB_STATE["status"] = res.get("status", "TRAINING_BLOCKED")
            TRAINING_JOB_STATE["error_message"] = res.get("message")
    except Exception as exc:
        logger.error(f"Background training failed: {exc}")
        TRAINING_JOB_STATE["status"] = "TRAINING_FAILED"
        TRAINING_JOB_STATE["error_message"] = str(exc)
    finally:
        TRAINING_JOB_STATE["is_running"] = False


@router.post("/start")
async def start_training_job(request: StartTrainingRequest, background_tasks: BackgroundTasks):
    """Starts requested training mode background job."""
    global TRAINING_JOB_STATE
    if TRAINING_JOB_STATE["is_running"]:
        raise HTTPException(status_code=400, detail="Training job is already running.")

    if request.mode == "AERIAL_METRIC_FINE_TUNING":
        metric_report = validate_dataset_quality("dataset/metric_depth")
        if not metric_report.is_passed:
            raise HTTPException(
                status_code=400,
                detail="AERIAL_METRIC_FINE_TUNING is BLOCKED: No true paired camera-surface metric depth ground truth dataset found in 'dataset/metric_depth/'"
            )

    background_tasks.add_task(_run_training_background_task, request.config_path, request.mode)

    TRAINING_JOB_STATE["is_running"] = True
    TRAINING_JOB_STATE["status"] = "TRAINING_RUNNING"
    TRAINING_JOB_STATE["mode"] = request.mode
    TRAINING_JOB_STATE["progress_pct"] = 0.0

    return {
        "status": "TRAINING_STARTED",
        "mode": request.mode,
        "message": f"Training job initiated in {request.mode} mode."
    }


@router.post("/stop")
async def stop_training_job():
    """Cancels running training job."""
    global TRAINING_JOB_STATE
    if not TRAINING_JOB_STATE["is_running"]:
        return {"status": "TRAINING IDLE", "message": "No active training job."}

    TRAINING_JOB_STATE["is_running"] = False
    TRAINING_JOB_STATE["status"] = "TRAINING_STOPPED"
    return {"status": "TRAINING_STOPPED", "message": "Training job cancelled."}
