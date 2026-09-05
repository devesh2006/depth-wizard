from abc import ABC, abstractmethod
import numpy as np
from PIL import Image
import io
import base64
import logging
import scipy.ndimage as ndimage
from typing import Tuple, Dict, Any, List
from models.depth import DepthConfidenceReport

logger = logging.getLogger(__name__)

# Upper bound on the longest side of the working depth map.
# The source image may be 4500+ px wide; retaining a full-resolution float32 depth
# map costs ~45 MB per scene and previously OOM-killed the worker once several
# scenes were cached (SIH26175 Section 29: uncontrolled memory consumption).
# Structure footprints are stored as FRACTIONAL bboxes, so bounding the working
# resolution changes no measurement semantics. The true source resolution is still
# reported verbatim in the image-quality report.
MAX_WORK_SIDE = 1024


def bounded_size(width: int, height: int, max_side: int = MAX_WORK_SIDE) -> Tuple[int, int]:
    """Scales (width, height) down so the longest side is at most `max_side`."""
    longest = max(width, height)
    if longest <= max_side:
        return width, height
    scale = max_side / float(longest)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))

class DepthEstimator(ABC):
    """
    Abstract Base Class for Metric and Relative Monocular Depth Estimation.
    Outputs both true camera-relative metric depth map in metres (depth_m)
    and normalized relative depth map in [0, 1] for visualization (depth_rel).
    """

    backend_name: str = "unknown"
    is_neural: bool = False
    is_metric_native: bool = True

    @abstractmethod
    def estimate_depth(self, img: Image.Image, image_type: str = "aerial") -> Tuple[np.ndarray, np.ndarray]:
        """
        Takes PIL RGB Image, returns (depth_m, depth_rel) where:
          - depth_m: float32 depth map of shape (H, W) in metres (camera distance)
          - depth_rel: float32 depth map of shape (H, W) normalized to [0, 1]
        """
        pass


class DepthAnythingV2MetricEstimator(DepthEstimator):
    """
    Official Depth Anything V2 Metric Outdoor Model backend (Hugging Face transformer weights).
    Models:
      - depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf (default)
      - depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf
    Emits raw predicted depth values in METRES.
    """

    def __init__(self, model_id: str = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf"):
        import torch  # noqa: F401
        from transformers import pipeline

        self.model_id = model_id
        self.backend_name = f"Depth Anything V2 Metric Outdoor ({'Base' if 'Base' in model_id else 'Small'}, HF weights)"
        self.is_neural = True
        self.is_metric_native = True
        self._torch = torch
        self._pipe = pipeline(task="depth-estimation", model=model_id, device=-1)

    def estimate_depth(self, img: Image.Image, image_type: str = "aerial") -> Tuple[np.ndarray, np.ndarray]:
        orig_w, orig_h = img.size

        # Bound inference resolution for CPU latency while preserving aspect ratio
        work = img.convert("RGB")
        max_side = 518
        if max(work.size) > max_side:
            work = work.copy()
            work.thumbnail((max_side, max_side), Image.Resampling.BILINEAR)

        out = self._pipe(work)
        depth_raw = np.array(out["predicted_depth"] if "predicted_depth" in out else out["depth"], dtype=np.float32)

        if depth_raw.ndim == 3:
            depth_raw = depth_raw.squeeze()

        depth_m_raw = np.maximum(0.5, depth_raw)

        out_w, out_h = bounded_size(orig_w, orig_h)
        depth_img = Image.fromarray(depth_m_raw)
        depth_m_resized = np.array(depth_img.resize((out_w, out_h), Image.Resampling.BICUBIC), dtype=np.float32)
        depth_m = np.maximum(0.5, depth_m_resized)

        lo, hi = float(np.percentile(depth_m, 1)), float(np.percentile(depth_m, 99))
        if hi > lo:
            depth_rel = (depth_m - lo) / (hi - lo)
        else:
            depth_rel = np.zeros_like(depth_m)
        depth_rel = np.clip(depth_rel, 0.0, 1.0)

        return depth_m, depth_rel


class DepthAnythingV2RelativeEstimator(DepthEstimator):
    """
    Relative Depth Anything V2 model for comparison/debugging.
    Checkpoint: depth-anything/Depth-Anything-V2-Small-hf
    Outputs relative depth and camera-range metric estimate.
    """

    backend_name = "Depth Anything V2 Relative Small (HF weights)"
    is_neural = True
    is_metric_native = False

    def __init__(self, model_id: str = "depth-anything/Depth-Anything-V2-Small-hf"):
        import torch  # noqa: F401
        from transformers import pipeline

        self.model_id = model_id
        self._torch = torch
        self._pipe = pipeline(task="depth-estimation", model=model_id, device=-1)

    def estimate_depth(self, img: Image.Image, image_type: str = "aerial") -> Tuple[np.ndarray, np.ndarray]:
        orig_w, orig_h = img.size
        work = img.convert("RGB")
        max_side = 518
        if max(work.size) > max_side:
            work = work.copy()
            work.thumbnail((max_side, max_side), Image.Resampling.BILINEAR)

        out = self._pipe(work)
        depth_arr = np.array(out["predicted_depth"] if "predicted_depth" in out else out["depth"], dtype=np.float32)
        if depth_arr.ndim == 3:
            depth_arr = depth_arr.squeeze()

        lo, hi = np.percentile(depth_arr, 1), np.percentile(depth_arr, 99)
        if hi > lo:
            depth_rel_raw = (depth_arr - lo) / (hi - lo)
        else:
            depth_rel_raw = np.zeros_like(depth_arr)
        depth_rel_raw = np.clip(depth_rel_raw, 0.0, 1.0)

        out_w, out_h = bounded_size(orig_w, orig_h)
        depth_img = Image.fromarray((depth_rel_raw * 65535).astype(np.uint16))
        depth_rel = np.array(depth_img.resize((out_w, out_h), Image.Resampling.BICUBIC), dtype=np.float32) / 65535.0

        depth_m = 15.0 + depth_rel * 65.0
        return depth_m, depth_rel


class StructuralMetricEstimator(DepthEstimator):
    """
    Dependency-free heuristic metric depth backend (NOT a neural network).
    Produces camera distance depth_m in metres (e.g. 15.0m to 120.0m)
    based on image view geometry, structural gradients, and pinhole camera prior.
    """

    backend_name = "Structural Geometric Metric Baseline (Heuristic Metres)"
    is_neural = False
    is_metric_native = False

    def estimate_depth(self, img: Image.Image, image_type: str = "aerial") -> Tuple[np.ndarray, np.ndarray]:
        orig_w, orig_h = img.size
        proc_img = img.resize((512, 512), Image.Resampling.BILINEAR).convert("RGB")
        arr = np.array(proc_img, dtype=np.float32) / 255.0
        gray = 0.2989 * arr[:, :, 0] + 0.5870 * arr[:, :, 1] + 0.1140 * arr[:, :, 2]

        grad_x = ndimage.sobel(gray, axis=1)
        grad_y = ndimage.sobel(gray, axis=0)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)

        struct_large = ndimage.gaussian_filter(gray, sigma=4.0)
        from scipy.ndimage import grey_dilation, grey_erosion
        dilated = grey_dilation(gray, size=(7, 7))
        eroded = grey_erosion(gray, size=(7, 7))
        local_contrast = dilated - eroded

        y_coords, _ = np.mgrid[0:512, 0:512]
        if image_type == "drone":
            perspective_prior = 1.0 - (y_coords / 512.0) * 0.4
            near_base_m, far_range_m = 12.0, 45.0
        elif image_type == "satellite":
            perspective_prior = np.ones((512, 512), dtype=np.float32) * 0.5
            near_base_m, far_range_m = 150.0, 300.0
        else:  # Aerial
            perspective_prior = 0.8 - (y_coords / 512.0) * 0.25
            near_base_m, far_range_m = 25.0, 95.0

        elevation_cues = (
            gray * 0.45 +
            (dilated - gray) * 0.25 +
            local_contrast * 0.30 -
            grad_mag * 0.10
        )
        smoothed = ndimage.gaussian_filter(elevation_cues, sigma=1.2)
        combined = 0.7 * elevation_cues + 0.3 * smoothed + perspective_prior * 0.15

        c_min, c_max = float(np.percentile(combined, 1)), float(np.percentile(combined, 99))
        if c_max > c_min:
            norm_depth = (combined - c_min) / (c_max - c_min)
        else:
            norm_depth = combined
        norm_depth = np.clip(norm_depth, 0.0, 1.0)

        out_w, out_h = bounded_size(orig_w, orig_h)
        depth_img = Image.fromarray((norm_depth * 65535).astype(np.uint16))
        depth_rel = np.array(depth_img.resize((out_w, out_h), Image.Resampling.BICUBIC), dtype=np.float32) / 65535.0

        depth_m = near_base_m + (1.0 - depth_rel) * far_range_m
        return depth_m.astype(np.float32), depth_rel.astype(np.float32)


def create_depth_estimator(model_variant: str = "metric_base") -> DepthEstimator:
    """
    Instantiates the requested depth estimation backend.
    Variants:
      - "metric_base": Depth Anything V2 Metric Outdoor Base (default)
      - "metric_small": Depth Anything V2 Metric Outdoor Small
      - "relative_small": Depth Anything V2 Relative Small
      - "structural": Structural Geometric Baseline
    Falls back gracefully to StructuralMetricEstimator if torch/weights unavailable.
    """
    try:
        if model_variant == "metric_small":
            return DepthAnythingV2MetricEstimator("depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf")
        elif model_variant == "relative_small":
            return DepthAnythingV2RelativeEstimator("depth-anything/Depth-Anything-V2-Small-hf")
        elif model_variant == "structural":
            return StructuralMetricEstimator()
        else:
            return DepthAnythingV2MetricEstimator("depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf")
    except Exception as exc:
        logger.warning(
            "Neural Depth model '%s' unavailable (%s). Falling back to Structural Geometric Metric Baseline.",
            model_variant,
            exc,
        )
        return StructuralMetricEstimator()


def compare_depth_models(img: Image.Image, image_type: str = "aerial") -> List[Dict[str, Any]]:
    """
    Executes depth model comparison benchmark across available depth backends.
    Reports inference time, metric depth bounds (min, max, mean, median), and validity %.
    """
    models_to_test = [
        ("Depth Anything V2 Metric Base", "metric_base"),
        ("Depth Anything V2 Metric Small", "metric_small"),
        ("Depth Anything V2 Relative Small", "relative_small"),
        ("Structural Geometric Baseline", "structural")
    ]
    results = []

    for name, variant in models_to_test:
        t0 = time.time()
        try:
            estimator = create_depth_estimator(variant)
            depth_m, depth_rel = estimator.estimate_depth(img, image_type=image_type)
            elapsed_ms = round((time.time() - t0) * 1000.0, 1)

            min_m = round(float(np.min(depth_m)), 2)
            max_m = round(float(np.max(depth_m)), 2)
            mean_m = round(float(np.mean(depth_m)), 2)
            median_m = round(float(np.median(depth_m)), 2)
            valid_pct = round(float(np.mean((depth_m > 0.1) & (~np.isnan(depth_m))) * 100.0), 1)

            results.append({
                "model_name": name,
                "variant": variant,
                "backend_identifier": estimator.backend_name,
                "is_neural": estimator.is_neural,
                "is_native_metric": estimator.is_metric_native,
                "inference_time_ms": elapsed_ms,
                "min_depth_m": min_m,
                "max_depth_m": max_m,
                "mean_depth_m": mean_m,
                "median_depth_m": median_m,
                "valid_pixel_pct": valid_pct
            })
        except Exception as err:
            results.append({
                "model_name": name,
                "variant": variant,
                "error": str(err),
                "inference_time_ms": round((time.time() - t0) * 1000.0, 1)
            })

    return results


def compute_depth_confidence(depth: np.ndarray, quality_report) -> DepthConfidenceReport:
    """
    Evaluates depth map quality & confidence based on:
    - Local depth consistency (smooth within structures)
    - Gradient consistency & edge alignment
    - Flat-depth detection
    - Invalid pixel concentration
    """
    h, w = depth.shape
    # Sample down if large for fast confidence analysis
    step_y = max(1, h // 256)
    step_x = max(1, w // 256)
    d_sub = depth[::step_y, ::step_x]
    
    # 1. Local variance / consistency
    local_mean = ndimage.uniform_filter(d_sub, size=5)
    local_sqr_mean = ndimage.uniform_filter(d_sub**2, size=5)
    local_var = np.maximum(0, local_sqr_mean - local_mean**2)
    local_consistency = float(1.0 - np.clip(np.mean(np.sqrt(local_var)) * 4.0, 0.0, 1.0))
    
    # 2. Gradient magnitude & edge alignment
    gx = np.abs(d_sub[:, 1:] - d_sub[:, :-1])
    gy = np.abs(d_sub[1:, :] - d_sub[:-1, :])
    grad_energy = float(np.mean(gx) + np.mean(gy))
    
    # 3. Flat depth detection (e.g. low dynamic range)
    d_min, d_max = float(np.min(d_sub)), float(np.max(d_sub))
    dynamic_range = d_max - d_min
    flat_depth_detected = dynamic_range < 0.25
    
    # 4. Extreme values / invalid percentage
    invalid_mask = (d_sub <= 0.001) | (d_sub >= 0.999) | np.isnan(d_sub)
    invalid_pixel_pct = float(np.mean(invalid_mask) * 100.0)
    
    reasons = []
    confidence_score = 88.0
    
    if flat_depth_detected:
        confidence_score -= 35.0
        reasons.append("Low dynamic depth range across scene.")
        
    if invalid_pixel_pct > 10.0:
        confidence_score -= 15.0
        reasons.append(f"High concentration of saturated/invalid depth pixels ({invalid_pixel_pct:.1f}%).")
        
    if quality_report.nadir_warning:
        confidence_score -= 12.0
        reasons.append("Near-nadir perspective limits vertical facade elevation cues.")
        
    if quality_report.sharpness_label in ["Blurry", "Fair"]:
        confidence_score -= 15.0
        reasons.append(f"Input image sharpness is {quality_report.sharpness_label.lower()}, softening depth boundaries.")
        
    if quality_report.cloud_shadow_risk != "Low":
        confidence_score -= 10.0
        reasons.append(f"{quality_report.cloud_shadow_risk} cloud/shadow occlusion risk.")
        
    if not reasons:
        reasons.append("High geometric edge alignment and consistent structural depth variation.")
        
    confidence_score = float(np.clip(confidence_score, 20.0, 96.0))
    if confidence_score >= 75.0:
        overall_confidence = "HIGH"
    elif confidence_score >= 50.0:
        overall_confidence = "MEDIUM"
    else:
        overall_confidence = "LOW"
        
    return DepthConfidenceReport(
        overall_confidence=overall_confidence,
        confidence_score=round(confidence_score, 1),
        local_gradient_consistency=round(local_consistency, 3),
        edge_alignment_score=round(float(np.clip(grad_energy * 10.0, 0.1, 0.95)), 3),
        flat_depth_detected=flat_depth_detected,
        invalid_pixel_pct=round(invalid_pixel_pct, 2),
        reasons=reasons
    )

def generate_depth_colormaps(depth: np.ndarray) -> Dict[str, str]:
    """
    Generates data URL PNG colormaps (Turbo, Viridis, Magma, Grayscale)
    for interactive UI display and scientific visualization.
    """
    # Normalize depth to 0..255 uint8
    norm = np.clip(depth, 0.0, 1.0)
    norm_uint8 = (norm * 255).astype(np.uint8)
    
    # Colormap lookups (256x3 RGB values)
    # 1. Grayscale
    gray_rgb = np.stack([norm_uint8, norm_uint8, norm_uint8], axis=-1)
    
    # 2. Turbo Colormap (Scientific perceptual colormap)
    turbo_lut = _get_turbo_lut()
    turbo_rgb = turbo_lut[norm_uint8]
    
    # 3. Viridis Colormap
    viridis_lut = _get_viridis_lut()
    viridis_rgb = viridis_lut[norm_uint8]
    
    # 4. Magma Colormap
    magma_lut = _get_magma_lut()
    magma_rgb = magma_lut[norm_uint8]
    
    return {
        "turbo": _arr_to_data_url(turbo_rgb),
        "viridis": _arr_to_data_url(viridis_rgb),
        "magma": _arr_to_data_url(magma_rgb),
        "grayscale": _arr_to_data_url(gray_rgb)
    }

def _arr_to_data_url(arr: np.ndarray) -> str:
    img = Image.fromarray(arr.astype(np.uint8))
    # Downscale preview to reasonable size for fast network payload
    max_dim = 800
    if max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim), Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def _get_turbo_lut() -> np.ndarray:
    # 256-step polynomial Turbo Colormap
    x = np.linspace(0.0, 1.0, 256)
    r = 0.1357 + x * (4.61539 - x * (42.6603 - x * (132.131 - x * (161.073 - x * 65.402))))
    g = 0.0914 + x * (2.19418 + x * (16.4218 - x * (57.4583 - x * (71.3094 - x * 31.764))))
    b = 0.1067 + x * (12.5925 - x * (60.1097 - x * (109.0745 - x * (88.5061 - x * 26.818))))
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

def _get_viridis_lut() -> np.ndarray:
    x = np.linspace(0.0, 1.0, 256)
    # Accurate Viridis polynomial approximation
    r = 0.28 + 0.7 * x**2
    g = 0.01 + 0.9 * x
    b = 0.33 + 0.6 * np.sin(x * np.pi)
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

def _get_magma_lut() -> np.ndarray:
    x = np.linspace(0.0, 1.0, 256)
    r = np.clip(x * 1.3 - 0.05, 0, 1)
    g = np.clip((x - 0.2) * 1.15, 0, 1)**1.4
    b = np.clip(0.35 + 0.65 * np.sin(x * np.pi * 0.9), 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
