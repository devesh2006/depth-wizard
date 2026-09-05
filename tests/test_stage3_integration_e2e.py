import sys
import os
import pytest
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

# Add backend directory to python path
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
sys.path.insert(0, backend_path)

from server import app
from services.stage3_height_service import estimate_building_vertical_height
from services.segmentation_service import measure_building_height
from models.depth import CalibrationConfig, MeasureBuildingRequest
from routers.depth_router import DEPTH_CACHE

client = TestClient(app)

def test_stage3_controlled_ground_roof_e2e_calc():
    """Verify ground=12.70m, roof=181.10m yields height=168.40m and status=HEIGHT_COMPUTED."""
    res = estimate_building_vertical_height(
        ground_elevation_input=12.70,
        roof_elevation_input=181.10,
        view_geometry="Oblique"
    )

    assert res["height_available"] is True
    assert abs(res["estimated_height_m"] - 168.40) < 1e-4
    assert res["ground_elevation_m"] == 12.70
    assert res["roof_elevation_m"] == 181.10
    assert res["calculation_status"] == "HEIGHT_COMPUTED"
    assert "Ground-Plane Normal" in res["vertical_reference"] or "Ground Normal Alignment" in res["vertical_reference"]

def test_stage3_controlled_ground_roof_via_segmentation_service():
    """Verify structure-level height calculation with synthesized depth."""
    calib = CalibrationConfig(mode="relative")
    depth_m = np.ones((100, 100), dtype=np.float32) * 50.0
    depth_m[20:60, 20:60] = 30.0 # building pixels at 30m depth

    measurement = measure_building_height(
        depth_map=depth_m,
        bbox=[20, 20, 40, 40],
        calibration=calib,
        building_name="Central Structure",
        view_geometry="Oblique"
    )

    assert measurement.is_metric_available is True
    assert measurement.calibrated_height_m is not None
    assert measurement.calibrated_height_m > 0
    assert measurement.ground_elevation_m is not None
    assert measurement.roof_elevation_m is not None
    assert measurement.calculation_status in ["HEIGHT_COMPUTED", "GEOMETRY_READY", "VALIDATION_AVAILABLE"]
    assert "Metric 3D Geometry" in measurement.scale_recovery_method

def test_stage3_gt_isolation_e2e():
    """Verify Ground Truth (None vs 175m vs 500m) has zero influence on predicted height/elevations."""
    calib = CalibrationConfig(mode="relative")
    depth_m = np.ones((100, 100), dtype=np.float32) * 50.0
    depth_m[20:60, 20:60] = 32.0

    m_none = measure_building_height(
        depth_map=depth_m,
        bbox=[20, 20, 40, 40],
        calibration=calib,
        building_name="Central Structure",
        ground_truth_height_m=None
    )

    m_175 = measure_building_height(
        depth_map=depth_m,
        bbox=[20, 20, 40, 40],
        calibration=calib,
        building_name="Central Structure",
        ground_truth_height_m=175.0
    )

    m_500 = measure_building_height(
        depth_map=depth_m,
        bbox=[20, 20, 40, 40],
        calibration=calib,
        building_name="Central Structure",
        ground_truth_height_m=500.0
    )

    # Identical physical outputs across all GT cases
    assert m_none.calibrated_height_m == m_175.calibrated_height_m == m_500.calibrated_height_m
    assert m_none.ground_elevation_m == m_175.ground_elevation_m == m_500.ground_elevation_m
    assert m_none.roof_elevation_m == m_175.roof_elevation_m == m_500.roof_elevation_m

    # GT only affects evaluation error metrics
    assert m_none.error_m is None
    assert m_175.error_m == abs(m_175.calibrated_height_m - 175.0)
    assert m_500.error_m == abs(m_500.calibrated_height_m - 500.0)

def test_stage3_api_measure_building_endpoint():
    """Verify POST /api/depth/measure-building API endpoint returns full Stage 3 structure."""
    scene_id = "test_scene_e2e"
    depth_m = np.ones((100, 100), dtype=np.float32) * 40.0
    depth_m[25:50, 25:50] = 25.0
    pil_img = Image.new("RGB", (100, 100), color=(100, 120, 140))
    calib = CalibrationConfig(mode="relative")

    DEPTH_CACHE[scene_id] = {
        "scene_name": "Test E2E Scene",
        "image": pil_img,
        "depth": depth_m,
        "depth_m": depth_m,
        "depth_rel": depth_m / 40.0,
        "request": None,
        "calibration": calib,
        "intrinsics": {"fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 50.0}
    }

    payload = {
        "scene_id": scene_id,
        "bbox": [25, 25, 25, 25],
        "name": "E2E Test Building"
    }

    response = client.post("/api/depth/measure-building", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["name"] == "E2E Test Building"
    assert data["calibrated_height_m"] is not None
    assert data["ground_elevation_m"] is not None
    assert data["roof_elevation_m"] == round(data["ground_elevation_m"] + data["calibrated_height_m"], 2)
    assert data["calculation_status"] in ["HEIGHT_COMPUTED", "GEOMETRY_READY", "VALIDATION_AVAILABLE"]
    assert "Metric 3D Geometry" in data["scale_recovery_method"]

def test_stage3_api_estimate_height_endpoint():
    """Verify POST /api/depth/estimate-height API endpoint executes Stage 3 height service directly."""
    scene_id = "test_scene_estimate"
    depth_m = np.ones((100, 100), dtype=np.float32) * 45.0
    depth_m[30:60, 30:60] = 30.0

    DEPTH_CACHE[scene_id] = {
        "depth_m": depth_m,
        "intrinsics": {"fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 50.0}
    }

    payload = {
        "scene_id": scene_id,
        "bbox": [30, 30, 30, 30]
    }

    response = client.post("/api/depth/estimate-height", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["height_available"] is True
    assert data["estimated_height_m"] is not None
    assert data["calculation_status"] == "HEIGHT_COMPUTED"
    assert "Ground-Plane Normal" in data["vertical_reference"] or "Ground Normal Alignment" in data["vertical_reference"]
