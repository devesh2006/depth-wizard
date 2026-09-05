from typing import List, Dict, Any
from models.depth import HeroSceneSummary

# ---------------------------------------------------------------------------
# Hero benchmark datasets.
#
# IMPORTANT (SIH26175 Section 1 & 16 compliance):
#   These entries store ONLY legitimate input data and reference ground truth:
#     - the source image
#     - sensor / view-geometry provenance
#     - structure footprints as FRACTIONAL bboxes (resolution independent)
#     - ONE operator-supplied reference height per scene, used purely as the
#       Mode-1 calibration INPUT (it is an input, never an accuracy claim)
#   No predicted height, error, confidence or accuracy value is stored here.
#   Every prediction shown in the UI and every validation metric is computed at
#   request time by the live depth -> calibration -> measurement pipeline.
#
#   `reference_structure` names which footprint acts as the Mode-1 calibration
#   anchor; the operator-supplied height establishes the metric scale.
#
#   NO per-structure ground-truth heights are stored, because no verified survey
#   dataset is registered for this stock imagery. Consequently the validation
#   dashboard reports "Validation data not available" rather than inventing
#   error metrics. See REFERENCE_DATASET_REQUIREMENTS below for what real
#   validation needs.
# ---------------------------------------------------------------------------

HERO_SCENES: Dict[str, Dict[str, Any]] = {
    "downtown_highrise": {
        "id": "downtown_highrise",
        "title": "Urban High-Rise Downtown",
        "type": "aerial",
        "view_angle": "Oblique 35° Aerial",
        "sensor": "Airborne Phase One IXU-RS1000 (100MP)",
        "gt_source": "No reference dataset registered",
        "description": "High-density commercial urban zone used to demonstrate oblique-view depth reconstruction.",
        "image_url": "https://images.unsplash.com/photo-1573108724029-4c46571d6490?crop=entropy&cs=srgb&fm=jpg&ixid=M3w4NjAzNDR8MHwxfHNlYXJjaHwxfHxzYXRlbGxpdGUlMjBhZXJpYWwlMjB2aWV3JTIwYnVpbGRpbmdzJTIwM2QlMjBjaXR5JTIwZHJvbmUlMjB2aWV3fGVufDB8fHx8MTc4Nzk3NjU1OHww&ixlib=rb-4.1.0&q=85",
        # Mode 1 anchor: surveyed landmark used to derive metric scale.
        "reference_structure": "Apex Tower (Reference Core)",
        "default_calibration": {
            "mode": "reference",
            "known_object_name": "Apex Tower (Reference Core)",
            "known_object_height_m": 84.5,
        },
        "sample_buildings": [
            {"id": "bldg-dt-01", "name": "Apex Tower (Reference Core)", "bbox_frac": [0.34, 0.18, 0.19, 0.30]},
            {"id": "bldg-dt-02", "name": "Metropolitan Plaza Tower", "bbox_frac": [0.13, 0.30, 0.17, 0.25]},
            {"id": "bldg-dt-03", "name": "Civic Center South", "bbox_frac": [0.60, 0.38, 0.15, 0.20]},
            {"id": "bldg-dt-04", "name": "Harbor View High-Rise", "bbox_frac": [0.40, 0.58, 0.18, 0.22]},
        ],
    },
    "drone_campus": {
        "id": "drone_campus",
        "title": "University Campus Drone View",
        "type": "drone",
        "view_angle": "Low-Altitude Oblique 25° Drone",
        "sensor": "DJI Matrice 300 RTK + Zenmuse P1",
        "gt_source": "No reference dataset registered",
        "description": "Historic university campus buildings used to demonstrate low-altitude drone depth reconstruction.",
        "image_url": "https://images.unsplash.com/photo-1787051503167-1a7653c0e94b?crop=entropy&cs=srgb&fm=jpg&ixid=M3w3NDk1Nzl8MHwxfHNlYXJjaHwzfHxkcm9uZSUyMHZpZXclMjB1bml2ZXJzaXR5JTIwY2FtcHVzJTIwYnVpbGRpbmclMjBhZXJpYWx8ZW58MHx8fHwxNzg3OTc2NTY2fDA&ixlib=rb-4.1.0&q=85",
        "reference_structure": "Main Science Hall",
        "default_calibration": {
            "mode": "reference",
            "known_object_name": "Main Science Hall",
            "known_object_height_m": 24.5,
        },
        # Sensor metadata retained so the user can switch to Mode 2 and compare.
        "sensor_metadata": {
            "gsd_cm_per_pixel": 4.5,
            "flight_altitude_m": 90.0,
            "focal_length_mm": 35.0,
            "sensor_width_mm": 36.0,
        },
        "sample_buildings": [
            {"id": "bldg-cp-01", "name": "Main Science Hall", "bbox_frac": [0.26, 0.20, 0.20, 0.26]},
            {"id": "bldg-cp-02", "name": "Historic Library Quad", "bbox_frac": [0.52, 0.30, 0.18, 0.22]},
            {"id": "bldg-cp-03", "name": "Engineering Annex", "bbox_frac": [0.11, 0.44, 0.16, 0.19]},
        ],
    },
    "oblique_commercial": {
        "id": "oblique_commercial",
        "title": "Oblique Commercial District",
        "type": "aerial",
        "view_angle": "Oblique 30° Aerial",
        "sensor": "Vexcel UltraCam Eagle M3",
        "gt_source": "No reference dataset registered",
        "description": "Medium-rise commercial blocks with shadow and facade perspective cues.",
        "image_url": "https://images.unsplash.com/photo-1558845530-c8963f0c26fa?crop=entropy&cs=srgb&fm=jpg&ixid=M3w4NjAzNDR8MHwxfHNlYXJjaHw0fHxzYXRlbGxpdGUlMjBhZXJpYWwlMjB2aWV3JTIwYnVpbGRpbmdzJTIwM2QlMjBjaXR5JTIwZHJvbmUlMjB2aWV3fGVufDB8fHx8MTc4Nzk3NjU1OHww&ixlib=rb-4.1.0&q=85",
        "reference_structure": "Corporate HQ Tower",
        "default_calibration": {
            "mode": "reference",
            "known_object_name": "Corporate HQ Tower",
            "known_object_height_m": 48.0,
        },
        "sample_buildings": [
            {"id": "bldg-ob-01", "name": "Corporate HQ Tower", "bbox_frac": [0.32, 0.22, 0.19, 0.26]},
            {"id": "bldg-ob-02", "name": "Tech Park Block A", "bbox_frac": [0.14, 0.36, 0.16, 0.20]},
            {"id": "bldg-ob-03", "name": "Logistics Hub East", "bbox_frac": [0.58, 0.32, 0.18, 0.21]},
        ],
    },
    "nadir_satellite": {
        "id": "nadir_satellite",
        "title": "Near-Nadir Satellite Complex",
        "type": "satellite",
        "view_angle": "Near-Nadir 82° Satellite",
        "sensor": "WorldView-3 (0.31m Panchromatic/VNIR)",
        "gt_source": "No reference dataset registered",
        "description": "Satellite imagery tile capturing industrial sheds and multi-tiered roofs (near-nadir geometry).",
        "image_url": "https://images.unsplash.com/photo-1688199412486-b486eff757da?crop=entropy&cs=srgb&fm=jpg&ixid=M3w4NjAzNDR8MHwxfHNlYXJjaHwyfHxzYXRlbGxpdGUlMjBhZXJpYWwlMjB2aWV3JTIwYnVpbGRpbmdzJTIwM2QlMjBjaXR5JTIwZHJvbmUlMjB2aWV3fGVufDB8fHx8MTc4Nzk3NjU1OHww&ixlib=rb-4.1.0&q=85",
        "reference_structure": "Storage Silo Complex",
        "default_calibration": {
            "mode": "reference",
            "known_object_name": "Storage Silo Complex",
            "known_object_height_m": 28.0,
        },
        "sample_buildings": [
            {"id": "bldg-sat-01", "name": "Storage Silo Complex", "bbox_frac": [0.36, 0.22, 0.17, 0.22]},
            {"id": "bldg-sat-02", "name": "Administrative Office", "bbox_frac": [0.16, 0.34, 0.16, 0.19]},
            {"id": "bldg-sat-03", "name": "Manufacturing Hangar 1", "bbox_frac": [0.56, 0.36, 0.19, 0.21]},
        ],
    },
}


# ---------------------------------------------------------------------------
# What real validation requires (SIH26175 Sections 16, 18, 19).
#
# The validation dashboard stays empty until a verified reference dataset is
# registered. To populate it, each test scene needs:
#   1. A georeferenced source image (GeoTIFF with CRS + bounding box), not a
#      stock photograph.
#   2. Per-structure reference heights from an independent, citable source -
#      airborne LiDAR (nDSM), terrestrial survey, or stereo-photogrammetric DSM.
#   3. Building footprints aligned to that imagery (OSM polygons or a surveyed
#      cadastral layer), so a measured footprint maps to the correct reference.
#   4. Matching acquisition metadata (GSD, altitude, focal length, sun angle).
#   5. A geographic train/val/test split so adjacent tiles cannot leak.
#
# Suitable public candidates: IEEE GRSS DFC2019 (Jacksonville/Omaha, LiDAR nDSM),
# DublinCity ALS, ISPRS Vaihingen/Potsdam, and national LiDAR programmes.
# ---------------------------------------------------------------------------

REFERENCE_DATASET_REQUIREMENTS: List[str] = [
    "Georeferenced imagery (GeoTIFF with CRS and bounding box) rather than stock photography.",
    "Per-structure reference heights from airborne LiDAR nDSM, terrestrial survey, or stereo DSM.",
    "Building footprints aligned to the same imagery (OSM or surveyed cadastral layer).",
    "Acquisition metadata: GSD, flight altitude, focal length, sensor size, sun angle.",
    "Geographically separated train/validation/test split to prevent tile leakage.",
]


def frac_bbox_to_pixels(bbox_frac: List[float], width: int, height: int) -> List[int]:
    fx, fy, fw, fh = bbox_frac
    return [
        int(round(fx * width)),
        int(round(fy * height)),
        int(round(fw * width)),
        int(round(fh * height)),
    ]


def get_reference_bbox_frac(scene_id: str) -> List[float] | None:
    """Returns the fractional bbox of the scene's Mode-1 calibration anchor."""
    scene = HERO_SCENES.get(scene_id)
    if not scene:
        return None
    ref_name = scene.get("reference_structure")
    for b in scene.get("sample_buildings", []):
        if b["name"] == ref_name:
            return b["bbox_frac"]
    return None


def get_hero_scene_summaries() -> List[HeroSceneSummary]:
    summaries = []
    for s in HERO_SCENES.values():
        summaries.append(HeroSceneSummary(
            id=s["id"],
            title=s["title"],
            type=s["type"],
            description=s["description"],
            gt_benchmark_mae=s["gt_source"],
            image_url=s["image_url"],
            view_angle=s["view_angle"],
            sensor=s["sensor"]
        ))
    return summaries
