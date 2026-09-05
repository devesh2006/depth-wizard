import numpy as np
from PIL import Image
from typing import Tuple, Dict, Any
from models.depth import ImageQualityReport

def extract_camera_exif_metadata(img: Image.Image) -> Dict[str, Any]:
    """
    Extracts EXIF camera metadata if present in the raw image:
    - Focal length (mm & 35mm equivalent)
    - Camera Make & Model
    - GPS Altitude AGL/MSL
    - Sensor Width (mm)
    - Field of View (FOV degrees)
    """
    metadata: Dict[str, Any] = {
        "has_exif": False,
        "focal_length_mm": None,
        "focal_length_35mm": None,
        "camera_make": None,
        "camera_model": None,
        "gps_altitude_m": None,
        "sensor_width_mm": None,
        "fov_deg": None
    }
    try:
        exif = img._getexif()
        if exif:
            metadata["has_exif"] = True
            for tag_id, value in exif.items():
                if tag_id == 37386: # FocalLength
                    try: metadata["focal_length_mm"] = float(value)
                    except: pass
                elif tag_id == 41989: # FocalLengthIn35mmFilm
                    try: metadata["focal_length_35mm"] = float(value)
                    except: pass
                elif tag_id == 271:
                    metadata["camera_make"] = str(value).strip()
                elif tag_id == 272:
                    metadata["camera_model"] = str(value).strip()
                elif tag_id == 34853 and isinstance(value, dict): # GPSInfo
                    if 6 in value: # GPSAltitude
                        try: metadata["gps_altitude_m"] = float(value[6])
                        except: pass

            fl = metadata["focal_length_mm"] or metadata["focal_length_35mm"]
            if fl and fl > 0:
                sensor_w = 36.0 if metadata["focal_length_35mm"] else 24.0
                metadata["sensor_width_mm"] = sensor_w
                metadata["fov_deg"] = round(2.0 * float(np.degrees(np.arctan((sensor_w / 2.0) / fl))), 1)
    except Exception:
        pass
    return metadata


def analyze_image_quality(img: Image.Image, image_type: str = "aerial") -> ImageQualityReport:
    """
    Analyzes input image quality and height estimation suitability:
    - Image Quality (Resolution, Sharpness, Contrast)
    - Depth Quality (Signal variance)
    - Geometry Quality (View angle, vertical foreshortening)
    - Height Suitability Score (Separated from visual image quality)
    """
    img_rgb = img.convert('RGB')
    width, height = img.size
    aspect_ratio = round(width / max(1, height), 2)
    
    proc_img = img_rgb.resize((min(1024, width), min(1024, height)), Image.Resampling.BILINEAR)
    arr_rgb = np.array(proc_img, dtype=np.float32) / 255.0
    arr_gray = 0.2989 * arr_rgb[:, :, 0] + 0.5870 * arr_rgb[:, :, 1] + 0.1140 * arr_rgb[:, :, 2]
    
    # 1. Image Quality (Sharpness & Contrast)
    pad_gray = np.pad(arr_gray, 1, mode='edge')
    laplacian = (
        pad_gray[:-2, 1:-1] +
        pad_gray[2:, 1:-1] +
        pad_gray[1:-1, :-2] +
        pad_gray[1:-1, 2:] -
        4.0 * arr_gray
    )
    sharpness_score = float(np.var(laplacian) * 10000.0)
    
    if sharpness_score > 35.0: sharpness_label = "Excellent"
    elif sharpness_score > 15.0: sharpness_label = "Good"
    elif sharpness_score > 5.0: sharpness_label = "Fair"
    else: sharpness_label = "Blurry"
        
    brightness_mean = float(np.mean(arr_gray) * 255.0)
    if 70.0 <= brightness_mean <= 190.0: brightness_label = "Optimal"
    elif brightness_mean > 190.0: brightness_label = "Overexposed"
    else: brightness_label = "Underexposed"
        
    contrast_score = float(np.std(arr_gray) * 255.0)
    if contrast_score > 50.0: contrast_label = "High"
    elif contrast_score > 30.0: contrast_label = "Optimal"
    else: contrast_label = "Low"
        
    max_c = np.maximum(np.maximum(arr_rgb[:, :, 0], arr_rgb[:, :, 1]), arr_rgb[:, :, 2])
    min_c = np.minimum(np.minimum(arr_rgb[:, :, 0], arr_rgb[:, :, 1]), arr_rgb[:, :, 2])
    delta = max_c - min_c
    sat = np.where(max_c > 0.001, delta / (max_c + 1e-6), 0.0)
    saturation_mean = float(np.mean(sat))
    
    cloud_mask = (arr_gray > 0.90) & (sat < 0.15)
    cloud_pct = float(np.mean(cloud_mask) * 100.0)
    shadow_mask = arr_gray < 0.08
    shadow_pct = float(np.mean(shadow_mask) * 100.0)
    
    if cloud_pct > 20.0 or shadow_pct > 25.0: cloud_shadow_risk = "High"
    elif cloud_pct > 8.0 or shadow_pct > 12.0: cloud_shadow_risk = "Moderate"
    else: cloud_shadow_risk = "Low"
        
    # 2. View Geometry & Height Observability Analysis
    dx = np.abs(arr_gray[:, 1:] - arr_gray[:, :-1])
    dy = np.abs(arr_gray[1:, :] - arr_gray[:-1, :])
    edge_ratio = float(np.mean(dy) / max(0.001, np.mean(dx)))
    
    if image_type == "satellite":
        view_geometry = "Near-Nadir (75-90°)"
        nadir_warning = True
    elif image_type == "drone":
        view_geometry = "Low-Altitude Drone (15-35°)"
        nadir_warning = False
    else:
        if 0.85 <= edge_ratio <= 1.15 and contrast_score < 40:
            view_geometry = "Near-Nadir (70-85°)"
            nadir_warning = True
        else:
            view_geometry = "Oblique (25-45°)"
            nadir_warning = False
            
    # 3. Independent Quality Metrics
    image_quality_score = float(np.clip(
        75.0 + (sharpness_score * 0.5) - (0.0 if brightness_label == "Optimal" else 15.0),
        20.0, 98.0
    ))
    
    depth_quality_score = float(np.clip(
        80.0 + (contrast_score * 0.2) - (20.0 if cloud_shadow_risk == "High" else 0.0),
        25.0, 95.0
    ))
    
    geometry_quality_score = 35.0 if nadir_warning else 85.0
    metric_calibration_quality_score = 10.0 # Uncalibrated baseline
    
    # 4. Height Suitability Score (Specifically for building height estimation)
    # CRITICAL: "Image suitability: High" MUST NOT imply "height accuracy: high"
    if nadir_warning:
        height_suitability_score = 25.0
        height_suitability_label = "Low"
        height_suitability_reason = "Low observability for metric height — oblique imagery recommended."
    else:
        height_suitability_score = float(np.clip((image_quality_score * 0.4 + geometry_quality_score * 0.6), 30.0, 95.0))
        height_suitability_label = "High" if height_suitability_score >= 75.0 else "Medium"
        height_suitability_reason = "Oblique perspective with observable building facade vertical cues."

    exif_data = extract_camera_exif_metadata(img)
    if exif_data["has_exif"]:
        metric_calibration_quality_score = 75.0

    return ImageQualityReport(
        resolution=[width, height],
        aspect_ratio=aspect_ratio,
        sharpness_score=round(sharpness_score, 2),
        sharpness_label=sharpness_label,
        brightness_mean=round(brightness_mean, 1),
        brightness_label=brightness_label,
        contrast_score=round(contrast_score, 2),
        contrast_label=contrast_label,
        saturation_mean=round(saturation_mean, 3),
        cloud_shadow_risk=cloud_shadow_risk,
        view_geometry=view_geometry,
        nadir_warning=nadir_warning,
        image_quality_score=round(image_quality_score, 1),
        depth_quality_score=round(depth_quality_score, 1),
        geometry_quality_score=round(geometry_quality_score, 1),
        metric_calibration_quality_score=round(metric_calibration_quality_score, 1),
        height_suitability_score=round(height_suitability_score, 1),
        height_suitability_label=height_suitability_label,
        height_suitability_reason=height_suitability_reason,
        suitability_score=round(height_suitability_score, 1),
        suitability_label=height_suitability_label
    )

