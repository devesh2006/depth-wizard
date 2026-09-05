import numpy as np
from typing import Dict, Any, List, Optional, Tuple

def fuse_height_estimates(
    cues: Dict[str, Dict[str, Any]]
) -> Tuple[Optional[float], Optional[float], float, List[str]]:
    """
    Principled Inverse-Variance Multi-Cue Height Fusion Engine.
    
    Input cues:
      cues["depth"]     = {"height_m": 172.5, "uncertainty_m": 5.2, "weight": 1.0}
      cues["geometry"]  = {"height_m": 176.8, "uncertainty_m": 6.8, "weight": 0.8}
      cues["shadow"]    = {"height_m": 178.4, "uncertainty_m": 8.0, "weight": 0.5}
      cues["reference"] = {"height_m": 179.0, "uncertainty_m": 1.2, "weight": 2.0}

    Formula:
      w_i = 1 / (sigma_i^2)
      H_fused = sum(w_i * H_i) / sum(w_i)
      sigma_fused = sqrt(1 / sum(w_i))

    Returns:
      (fused_height_m, fused_uncertainty_m, overall_confidence_pct, contributing_methods)
    """
    valid_estimates: List[float] = []
    weights: List[float] = []
    uncertainties: List[float] = []
    contributing: List[str] = []

    for name, cue in cues.items():
        h = cue.get("height_m")
        u = cue.get("uncertainty_m")
        w_factor = cue.get("weight", 1.0)

        if h is not None and h > 0 and u is not None and u > 0:
            sigma = max(0.2, float(u))
            w = (1.0 / (sigma ** 2)) * w_factor
            valid_estimates.append(float(h))
            weights.append(w)
            uncertainties.append(sigma)
            contributing.append(f"{name.capitalize()} Cue ({h:.1f} m ±{sigma:.1f} m)")

    if not valid_estimates:
        return None, None, 0.0, ["No valid metric height cues available."]

    w_arr = np.array(weights, dtype=np.float32)
    h_arr = np.array(valid_estimates, dtype=np.float32)

    sum_w = float(np.sum(w_arr))
    fused_height = float(np.sum(w_arr * h_arr) / sum_w)
    fused_uncertainty = float(np.sqrt(1.0 / sum_w))

    # Overall fusion confidence score based on uncertainty spread and cue count
    std_spread = float(np.std(h_arr)) if len(h_arr) > 1 else 0.0
    base_conf = 92.0 if len(h_arr) >= 3 else (85.0 if len(h_arr) == 2 else 75.0)
    overall_confidence = float(np.clip(base_conf - std_spread * 2.5 - fused_uncertainty * 2.0, 30.0, 96.0))

    return round(fused_height, 2), round(fused_uncertainty, 2), round(overall_confidence, 1), contributing
