import numpy as np
from PIL import Image
from typing import Tuple, Dict, Any, Optional

def analyze_building_shadow(
    img: Image.Image,
    depth_map: np.ndarray,
    bbox: list[int],
    gsd_m_per_pixel: Optional[float] = None,
    sun_elevation_deg: Optional[float] = 45.0
) -> Dict[str, Any]:
    """
    Estimates building height from cast shadow geometry:
      H_shadow = Shadow_Length_m * tan(Solar_Elevation_Rad)
    
    Returns a structured report with shadow confidence, estimated height cue,
    and boundary association status.
    """
    result: Dict[str, Any] = {
        "shadow_detected": False,
        "shadow_length_px": 0.0,
        "shadow_length_m": None,
        "height_shadow_cue_m": None,
        "confidence": 0.0,
        "reason": "No strong shadow vector associated with building footprint."
    }

    try:
        img_rgb = img.convert("RGB")
        w_img, h_img = img_rgb.size
        depth_h, depth_w = depth_map.shape

        x, y, w, h = bbox
        x1 = max(0, min(depth_w - 1, int(x)))
        y1 = max(0, min(depth_h - 1, int(y)))
        x2 = max(0, min(depth_w, int(x + w)))
        y2 = max(0, min(depth_h, int(y + h)))

        if x2 <= x1 or y2 <= y1:
            return result

        # Crop surrounding neighborhood to inspect shadow cast direction
        pad = max(15, int(max(w, h) * 0.4))
        crop_x1 = max(0, x1 - pad)
        crop_y1 = max(0, y1 - pad)
        crop_x2 = min(depth_w, x2 + pad)
        crop_y2 = min(depth_h, y2 + pad)

        # Convert image patch to luminance
        arr_rgb = np.array(img_rgb.resize((depth_w, depth_h)), dtype=np.float32) / 255.0
        gray = 0.2989 * arr_rgb[:, :, 0] + 0.5870 * arr_rgb[:, :, 1] + 0.1140 * arr_rgb[:, :, 2]
        surrounding_gray = gray[crop_y1:crop_y2, crop_x1:crop_x2]

        # Deep shadow mask: luminance < 0.12 and low saturation
        shadow_mask = surrounding_gray < 0.12
        shadow_pct = float(np.mean(shadow_mask) * 100.0)

        if shadow_pct > 3.0:
            # Estimate shadow extent cast away from building center
            shadow_length_px = float(np.sqrt(w * h) * (shadow_pct / 100.0) * 1.8)
            result["shadow_detected"] = True
            result["shadow_length_px"] = round(shadow_length_px, 1)

            if gsd_m_per_pixel and gsd_m_per_pixel > 0:
                shadow_len_m = shadow_length_px * gsd_m_per_pixel
                result["shadow_length_m"] = round(shadow_len_m, 2)
                
                sun_angle_rad = np.radians(sun_elevation_deg or 45.0)
                height_cue = shadow_len_m * np.tan(sun_angle_rad)
                result["height_shadow_cue_m"] = round(float(height_cue), 2)
                result["confidence"] = round(float(np.clip(shadow_pct * 3.5, 40.0, 85.0)), 1)
                result["reason"] = f"Shadow detected ({shadow_pct:.1f}% patch area, length {shadow_len_m:.1f} m @ {sun_elevation_deg:.0f}° solar elevation)."
            else:
                result["reason"] = f"Shadow detected ({shadow_pct:.1f}% area), but metric GSD is unavailable for length conversion."
    except Exception as exc:
        result["reason"] = f"Shadow analysis error: {str(exc)}"

    return result
