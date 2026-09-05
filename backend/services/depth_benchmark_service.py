from abc import ABC, abstractmethod
import time
import logging
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Optional, Tuple
import scipy.ndimage as ndimage

logger = logging.getLogger(__name__)

# Bounded working resolution for benchmarks to prevent OOM
MAX_BENCHMARK_SIDE = 1024

def bounded_benchmark_size(width: int, height: int, max_side: int = MAX_BENCHMARK_SIDE) -> Tuple[int, int]:
    longest = max(width, height)
    if longest <= max_side:
        return width, height
    scale = max_side / float(longest)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


class MetricDepthModel(ABC):
    """
    Modular Base Interface for Zero-Shot Metric Depth Models.
    Outputs raw floating-point camera distance depth in metres (depth_map_m).
    MUST NOT normalize metric depth to [0,1] for downstream calculations.
    """
    key: str
    name: str
    model_id: str
    device: str = "cpu"
    param_count_label: str = "Unknown"
    is_loaded: bool = False
    load_error: Optional[str] = None

    @abstractmethod
    def load(self) -> bool:
        """Loads model weights lazily. Returns True if loaded successfully, False if unavailable."""
        pass

    @abstractmethod
    def predict(self, img: Image.Image, image_type: str = "aerial") -> Dict[str, Any]:
        """
        Executes depth inference on PIL Image.
        Returns dictionary containing:
          - depth_map_m: np.ndarray [H, W] float32 in metres
          - model_name, status, timing, statistics, validity, compression diagnosis
        """
        pass


class DepthAnythingV2MetricBaseModel(MetricDepthModel):
    """
    Official Depth Anything V2 Metric Outdoor Base Model (518x518 input, ~97M params).
    Checkpoint: depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf
    """
    key = "da_v2_base"
    name = "DA-V2 Metric Base"
    model_id = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf"
    param_count_label = "97M"

    def __init__(self):
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._torch = torch
        self._pipe = None

    def load(self) -> bool:
        if self.is_loaded:
            return True
        try:
            from transformers import pipeline
            device_id = 0 if self.device == "cuda" else -1
            self._pipe = pipeline("depth-estimation", model=self.model_id, device=device_id)
            self.is_loaded = True
            self.load_error = None
            logger.info(f"Successfully loaded {self.name} on {self.device}")
            return True
        except Exception as exc:
            self.is_loaded = False
            self.load_error = str(exc)
            logger.warning(f"Failed to load {self.name}: {exc}")
            return False

    def predict(self, img: Image.Image, image_type: str = "aerial") -> Dict[str, Any]:
        t0 = time.time()
        if not self.is_loaded and not self.load():
            return {
                "model_key": self.key,
                "model_name": self.name,
                "status": "UNAVAILABLE",
                "device": self.device,
                "error_reason": self.load_error or "Model weights or pipeline unavailable.",
                "inference_time_ms": round((time.time() - t0) * 1000.0, 1),
                "depth_map_m": None
            }

        t_pre0 = time.time()
        orig_w, orig_h = img.size
        work_img = img.convert("RGB")
        max_side = 518
        if max(work_img.size) > max_side:
            work_img = work_img.copy()
            work_img.thumbnail((max_side, max_side), Image.Resampling.BILINEAR)
        t_pre = (time.time() - t_pre0) * 1000.0

        t_inf0 = time.time()
        try:
            out = self._pipe(work_img)
            t_inf = (time.time() - t_inf0) * 1000.0

            t_post0 = time.time()
            depth_raw = np.array(out["predicted_depth"] if "predicted_depth" in out else out["depth"], dtype=np.float32)
            if depth_raw.ndim == 3:
                depth_raw = depth_raw.squeeze()
            depth_raw = np.maximum(0.5, depth_raw)

            out_w, out_h = bounded_benchmark_size(orig_w, orig_h)
            depth_pil = Image.fromarray(depth_raw)
            depth_m = np.array(depth_pil.resize((out_w, out_h), Image.Resampling.BICUBIC), dtype=np.float32)
            depth_m = np.maximum(0.5, depth_m)
            t_post = (time.time() - t_post0) * 1000.0

            return build_model_result_dict(
                model_key=self.key,
                model_name=self.name,
                device=self.device,
                param_count=self.param_count_label,
                depth_map_m=depth_m,
                pre_ms=t_pre,
                inf_ms=t_inf,
                post_ms=t_post
            )
        except Exception as exc:
            logger.error(f"Inference error in {self.name}: {exc}")
            return {
                "model_key": self.key,
                "model_name": self.name,
                "status": "INFERENCE_FAILED",
                "device": self.device,
                "error_reason": str(exc),
                "inference_time_ms": round((time.time() - t0) * 1000.0, 1),
                "depth_map_m": None
            }


class DepthAnythingV2MetricLargeModel(MetricDepthModel):
    """
    Official Depth Anything V2 Metric Outdoor Large Model (~335M params).
    Checkpoint: depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf
    """
    key = "da_v2_large"
    name = "DA-V2 Metric Large"
    model_id = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"
    param_count_label = "335M"

    def __init__(self):
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._torch = torch
        self._pipe = None

    def load(self) -> bool:
        if self.is_loaded:
            return True
        try:
            from transformers import pipeline
            device_id = 0 if self.device == "cuda" else -1
            self._pipe = pipeline("depth-estimation", model=self.model_id, device=device_id)
            self.is_loaded = True
            self.load_error = None
            logger.info(f"Successfully loaded {self.name} on {self.device}")
            return True
        except Exception as exc:
            self.is_loaded = False
            self.load_error = str(exc)
            logger.warning(f"Failed to load {self.name}: {exc}")
            return False

    def predict(self, img: Image.Image, image_type: str = "aerial") -> Dict[str, Any]:
        t0 = time.time()
        if not self.is_loaded and not self.load():
            return {
                "model_key": self.key,
                "model_name": self.name,
                "status": "UNAVAILABLE",
                "device": self.device,
                "error_reason": self.load_error or "Model weights or pipeline unavailable.",
                "inference_time_ms": round((time.time() - t0) * 1000.0, 1),
                "depth_map_m": None
            }

        t_pre0 = time.time()
        orig_w, orig_h = img.size
        work_img = img.convert("RGB")
        max_side = 518
        if max(work_img.size) > max_side:
            work_img = work_img.copy()
            work_img.thumbnail((max_side, max_side), Image.Resampling.BILINEAR)
        t_pre = (time.time() - t_pre0) * 1000.0

        t_inf0 = time.time()
        try:
            out = self._pipe(work_img)
            t_inf = (time.time() - t_inf0) * 1000.0

            t_post0 = time.time()
            depth_raw = np.array(out["predicted_depth"] if "predicted_depth" in out else out["depth"], dtype=np.float32)
            if depth_raw.ndim == 3:
                depth_raw = depth_raw.squeeze()
            depth_raw = np.maximum(0.5, depth_raw)

            out_w, out_h = bounded_benchmark_size(orig_w, orig_h)
            depth_pil = Image.fromarray(depth_raw)
            depth_m = np.array(depth_pil.resize((out_w, out_h), Image.Resampling.BICUBIC), dtype=np.float32)
            depth_m = np.maximum(0.5, depth_m)
            t_post = (time.time() - t_post0) * 1000.0

            return build_model_result_dict(
                model_key=self.key,
                model_name=self.name,
                device=self.device,
                param_count=self.param_count_label,
                depth_map_m=depth_m,
                pre_ms=t_pre,
                inf_ms=t_inf,
                post_ms=t_post
            )
        except Exception as exc:
            logger.error(f"Inference error in {self.name}: {exc}")
            return {
                "model_key": self.key,
                "model_name": self.name,
                "status": "INFERENCE_FAILED",
                "device": self.device,
                "error_reason": str(exc),
                "inference_time_ms": round((time.time() - t0) * 1000.0, 1),
                "depth_map_m": None
            }


class UniDepthV2LargeModel(MetricDepthModel):
    """
    UniDepthV2-L Model (ViT-Large backbone).
    Tries standard HF transformers pipeline, AutoModel, or torch.hub.
    If custom repository requirements are missing, marks status as UNAVAILABLE gracefully.
    """
    key = "unidepth_v2_l"
    name = "UniDepthV2-L"
    model_id = "lmsys/unidepth-v2-vitl14"
    param_count_label = "304M"

    def __init__(self):
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._torch = torch
        self._model = None
        self._processor = None

    def load(self) -> bool:
        if self.is_loaded:
            return True
        try:
            from transformers import pipeline
            device_id = 0 if self.device == "cuda" else -1
            self._model = pipeline("depth-estimation", model=self.model_id, device=device_id, trust_remote_code=True)
            self.is_loaded = True
            self.load_error = None
            logger.info(f"Successfully loaded {self.name} on {self.device}")
            return True
        except Exception as e1:
            self.is_loaded = False
            self.load_error = f"UniDepthV2-L model checkpoint or custom architecture unavailable in current environment ({e1})."
            logger.warning(f"{self.name} unavailable: {self.load_error}")
            return False

    def predict(self, img: Image.Image, image_type: str = "aerial") -> Dict[str, Any]:
        t0 = time.time()
        if not self.is_loaded and not self.load():
            return {
                "model_key": self.key,
                "model_name": self.name,
                "status": "UNAVAILABLE",
                "device": self.device,
                "error_reason": self.load_error or "UniDepthV2-L checkpoint/architecture unavailable.",
                "inference_time_ms": round((time.time() - t0) * 1000.0, 1),
                "depth_map_m": None
            }

        t_pre0 = time.time()
        orig_w, orig_h = img.size
        work_img = img.convert("RGB")
        t_pre = (time.time() - t_pre0) * 1000.0

        t_inf0 = time.time()
        try:
            if callable(self._model):
                out = self._model(work_img)
                depth_raw = np.array(out["predicted_depth"] if "predicted_depth" in out else out["depth"], dtype=np.float32)
            else:
                # Custom torchhub model
                import torch
                with torch.no_grad():
                    inputs = self._torch.from_numpy(np.array(work_img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                    res = self._model(inputs.to(self.device))
                    depth_raw = res["depth"].squeeze().cpu().numpy()
            t_inf = (time.time() - t_inf0) * 1000.0

            t_post0 = time.time()
            if depth_raw.ndim == 3:
                depth_raw = depth_raw.squeeze()
            depth_raw = np.maximum(0.5, depth_raw)

            out_w, out_h = bounded_benchmark_size(orig_w, orig_h)
            depth_pil = Image.fromarray(depth_raw)
            depth_m = np.array(depth_pil.resize((out_w, out_h), Image.Resampling.BICUBIC), dtype=np.float32)
            depth_m = np.maximum(0.5, depth_m)
            t_post = (time.time() - t_post0) * 1000.0

            return build_model_result_dict(
                model_key=self.key,
                model_name=self.name,
                device=self.device,
                param_count=self.param_count_label,
                depth_map_m=depth_m,
                pre_ms=t_pre,
                inf_ms=t_inf,
                post_ms=t_post
            )
        except Exception as exc:
            logger.error(f"Inference error in {self.name}: {exc}")
            return {
                "model_key": self.key,
                "model_name": self.name,
                "status": "INFERENCE_FAILED",
                "device": self.device,
                "error_reason": str(exc),
                "inference_time_ms": round((time.time() - t0) * 1000.0, 1),
                "depth_map_m": None
            }


class Metric3Dv2Model(MetricDepthModel):
    """
    Metric3Dv2 Model (ViT-Large backbone).
    Tries standard HF transformers pipeline, AutoModel, or torch.hub.
    If custom mmcv/ops layers are missing, marks status as UNAVAILABLE gracefully.
    """
    key = "metric3dv2"
    name = "Metric3Dv2"
    model_id = "JianyuanGuo/Metric3D-ViT-Large"
    param_count_label = "335M"

    def __init__(self):
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._torch = torch
        self._model = None

    def load(self) -> bool:
        if self.is_loaded:
            return True
        try:
            from transformers import pipeline
            device_id = 0 if self.device == "cuda" else -1
            self._model = pipeline("depth-estimation", model=self.model_id, device=device_id, trust_remote_code=True)
            self.is_loaded = True
            self.load_error = None
            logger.info(f"Successfully loaded {self.name} on {self.device}")
            return True
        except Exception as e1:
            self.is_loaded = False
            self.load_error = f"Metric3Dv2 model or custom mmcv/ops layer unavailable in current environment ({e1})."
            logger.warning(f"{self.name} unavailable: {self.load_error}")
            return False

    def predict(self, img: Image.Image, image_type: str = "aerial") -> Dict[str, Any]:
        t0 = time.time()
        if not self.is_loaded and not self.load():
            return {
                "model_key": self.key,
                "model_name": self.name,
                "status": "UNAVAILABLE",
                "device": self.device,
                "error_reason": self.load_error or "Metric3Dv2 weights or custom mmcv/ops layer unavailable.",
                "inference_time_ms": round((time.time() - t0) * 1000.0, 1),
                "depth_map_m": None
            }

        t_pre0 = time.time()
        orig_w, orig_h = img.size
        work_img = img.convert("RGB")
        t_pre = (time.time() - t_pre0) * 1000.0

        t_inf0 = time.time()
        try:
            if callable(self._model):
                out = self._model(work_img)
                depth_raw = np.array(out["predicted_depth"] if "predicted_depth" in out else out["depth"], dtype=np.float32)
            else:
                import torch
                with torch.no_grad():
                    inputs = self._torch.from_numpy(np.array(work_img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                    res = self._model(inputs.to(self.device))
                    depth_raw = res["depth"].squeeze().cpu().numpy()
            t_inf = (time.time() - t_inf0) * 1000.0

            t_post0 = time.time()
            if depth_raw.ndim == 3:
                depth_raw = depth_raw.squeeze()
            depth_raw = np.maximum(0.5, depth_raw)

            out_w, out_h = bounded_benchmark_size(orig_w, orig_h)
            depth_pil = Image.fromarray(depth_raw)
            depth_m = np.array(depth_pil.resize((out_w, out_h), Image.Resampling.BICUBIC), dtype=np.float32)
            depth_m = np.maximum(0.5, depth_m)
            t_post = (time.time() - t_post0) * 1000.0

            return build_model_result_dict(
                model_key=self.key,
                model_name=self.name,
                device=self.device,
                param_count=self.param_count_label,
                depth_map_m=depth_m,
                pre_ms=t_pre,
                inf_ms=t_inf,
                post_ms=t_post
            )
        except Exception as exc:
            logger.error(f"Inference error in {self.name}: {exc}")
            return {
                "model_key": self.key,
                "model_name": self.name,
                "status": "INFERENCE_FAILED",
                "device": self.device,
                "error_reason": str(exc),
                "inference_time_ms": round((time.time() - t0) * 1000.0, 1),
                "depth_map_m": None
            }


class DepthAnythingV2MetricLargeAerialModel(MetricDepthModel):
    """
    Aerial-Specific Fine-Tuned Depth Anything V2 Metric Large Model (~335M params).
    Loads fine-tuned weights from checkpoints/best_aerial_depth_model.pt.
    If checkpoint file does not exist, returns status UNAVAILABLE gracefully.
    """
    key = "da_v2_large_aerial"
    name = "Aerial Fine-Tuned DA-V2 Large"
    model_id = "checkpoints/best_aerial_depth_model.pt"
    param_count_label = "335M"

    def __init__(self):
        import torch
        import os
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._torch = torch
        self._model = None
        self._os = os

    def load(self) -> bool:
        if self.is_loaded:
            return True
        ckpt_path = "checkpoints/best_aerial_depth_model.pt"
        if not self._os.path.exists(ckpt_path):
            self.is_loaded = False
            self.load_error = f"Fine-tuned checkpoint file not found at '{ckpt_path}'. Paired aerial dataset training required."
            logger.warning(f"{self.name} unavailable: {self.load_error}")
            return False
        try:
            from transformers import AutoModelForDepthEstimation
            base_model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf")
            ckpt = self._torch.load(ckpt_path, map_location=self.device)
            base_model.load_state_dict(ckpt["model_state_dict"])
            self._model = base_model.to(self.device).eval()
            self.is_loaded = True
            self.load_error = None
            logger.info(f"Successfully loaded fine-tuned model {self.name} from {ckpt_path}")
            return True
        except Exception as exc:
            self.is_loaded = False
            self.load_error = f"Failed to load fine-tuned model weights: {str(exc)}"
            return False

    def predict(self, img: Image.Image, image_type: str = "aerial") -> Dict[str, Any]:
        t0 = time.time()
        if not self.is_loaded and not self.load():
            return {
                "model_key": self.key,
                "model_name": self.name,
                "status": "UNAVAILABLE",
                "device": self.device,
                "error_reason": self.load_error or "Fine-tuned checkpoint unavailable.",
                "inference_time_ms": round((time.time() - t0) * 1000.0, 1),
                "depth_map_m": None
            }

        t_pre0 = time.time()
        orig_w, orig_h = img.size
        work_img = img.convert("RGB")
        max_side = 518
        if max(work_img.size) > max_side:
            work_img = work_img.copy()
            work_img.thumbnail((max_side, max_side), Image.Resampling.BILINEAR)
        t_pre = (time.time() - t_pre0) * 1000.0

        t_inf0 = time.time()
        try:
            with self._torch.no_grad():
                img_np = np.array(work_img, dtype=np.float32) / 255.0
                tensor_in = self._torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
                out = self._model(tensor_in)
                preds = out.predicted_depth if hasattr(out, "predicted_depth") else out.logits
                depth_raw = preds.squeeze().cpu().numpy()
            t_inf = (time.time() - t_inf0) * 1000.0

            t_post0 = time.time()
            if depth_raw.ndim == 3:
                depth_raw = depth_raw.squeeze()
            depth_raw = np.maximum(0.5, depth_raw)

            out_w, out_h = bounded_benchmark_size(orig_w, orig_h)
            depth_pil = Image.fromarray(depth_raw)
            depth_m = np.array(depth_pil.resize((out_w, out_h), Image.Resampling.BICUBIC), dtype=np.float32)
            depth_m = np.maximum(0.5, depth_m)
            t_post = (time.time() - t_post0) * 1000.0

            return build_model_result_dict(
                model_key=self.key,
                model_name=self.name,
                device=self.device,
                param_count=self.param_count_label,
                depth_map_m=depth_m,
                pre_ms=t_pre,
                inf_ms=t_inf,
                post_ms=t_post
            )
        except Exception as exc:
            logger.error(f"Inference error in {self.name}: {exc}")
            return {
                "model_key": self.key,
                "model_name": self.name,
                "status": "INFERENCE_FAILED",
                "device": self.device,
                "error_reason": str(exc),
                "inference_time_ms": round((time.time() - t0) * 1000.0, 1),
                "depth_map_m": None
            }


class DepthAnythingV2MetricLargeAerialAdaptedModel(MetricDepthModel):
    """
    GAMUS Aerial Domain-Adapted Depth Anything V2 Metric Large Model (~335M params).
    Loads domain-adapted weights from checkpoints/aerial_adaptation/best_model.pt.
    If checkpoint file does not exist, returns status UNAVAILABLE gracefully.
    """
    key = "da_v2_large_aerial_adapted"
    name = "Aerial Domain-Adapted DA-V2 Large"
    model_id = "checkpoints/aerial_adaptation/best_model.pt"
    param_count_label = "335M"

    def __init__(self):
        import torch
        import os
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._torch = torch
        self._model = None
        self._os = os

    def load(self) -> bool:
        if self.is_loaded:
            return True
        ckpt_path = "checkpoints/aerial_adaptation/best_model.pt"
        if not self._os.path.exists(ckpt_path):
            self.is_loaded = False
            self.load_error = f"Aerial domain-adapted checkpoint file not found at '{ckpt_path}'. GAMUS aerial adaptation training required."
            logger.warning(f"{self.name} unavailable: {self.load_error}")
            return False
        try:
            from training.models.multitask_aerial_depth import MultiTaskAerialDepthModel
            model = MultiTaskAerialDepthModel(load_pretrained=False)
            ckpt = self._torch.load(ckpt_path, map_location=self.device)
            model.load_state_dict(ckpt["model_state_dict"])
            self._model = model.to(self.device).eval()
            self.is_loaded = True
            self.load_error = None
            logger.info(f"Successfully loaded domain-adapted model {self.name} from {ckpt_path}")
            return True
        except Exception as exc:
            self.is_loaded = False
            self.load_error = f"Failed to load domain-adapted model weights: {str(exc)}"
            return False

    def predict(self, img: Image.Image, image_type: str = "aerial") -> Dict[str, Any]:
        t0 = time.time()
        if not self.is_loaded and not self.load():
            return {
                "model_key": self.key,
                "model_name": self.name,
                "status": "UNAVAILABLE",
                "device": self.device,
                "error_reason": self.load_error or "Domain-adapted checkpoint unavailable.",
                "inference_time_ms": round((time.time() - t0) * 1000.0, 1),
                "depth_map_m": None
            }

        t_pre0 = time.time()
        orig_w, orig_h = img.size
        work_img = img.convert("RGB")
        max_side = 518
        if max(work_img.size) > max_side:
            work_img = work_img.copy()
            work_img.thumbnail((max_side, max_side), Image.Resampling.BILINEAR)
        t_pre = (time.time() - t_pre0) * 1000.0

        t_inf0 = time.time()
        try:
            with self._torch.no_grad():
                img_np = np.array(work_img, dtype=np.float32) / 255.0
                tensor_in = self._torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
                out_dict = self._model(tensor_in)
                # STRICT RULE: ONLY depth_map_m is returned into depth pipeline!
                preds = out_dict["depth_map_m"]
                depth_raw = preds.squeeze().cpu().numpy()
            t_inf = (time.time() - t_inf0) * 1000.0

            t_post0 = time.time()
            if depth_raw.ndim == 3:
                depth_raw = depth_raw.squeeze()
            depth_raw = np.maximum(0.5, depth_raw)

            out_w, out_h = bounded_benchmark_size(orig_w, orig_h)
            depth_pil = Image.fromarray(depth_raw)
            depth_m = np.array(depth_pil.resize((out_w, out_h), Image.Resampling.BICUBIC), dtype=np.float32)
            depth_m = np.maximum(0.5, depth_m)
            t_post = (time.time() - t_post0) * 1000.0

            return build_model_result_dict(
                model_key=self.key,
                model_name=self.name,
                device=self.device,
                param_count=self.param_count_label,
                depth_map_m=depth_m,
                pre_ms=t_pre,
                inf_ms=t_inf,
                post_ms=t_post
            )
        except Exception as exc:
            logger.error(f"Inference error in {self.name}: {exc}")
            return {
                "model_key": self.key,
                "model_name": self.name,
                "status": "INFERENCE_FAILED",
                "device": self.device,
                "error_reason": str(exc),
                "inference_time_ms": round((time.time() - t0) * 1000.0, 1),
                "depth_map_m": None
            }


# ---------------------------------------------------------------------------
# MODEL REGISTRY
# ---------------------------------------------------------------------------
MODEL_CLASSES: Dict[str, type] = {
    "da_v2_base": DepthAnythingV2MetricBaseModel,
    "da_v2_large": DepthAnythingV2MetricLargeModel,
    "da_v2_large_aerial": DepthAnythingV2MetricLargeAerialModel,
    "da_v2_large_aerial_adapted": DepthAnythingV2MetricLargeAerialAdaptedModel,
    "unidepth_v2_l": UniDepthV2LargeModel,
    "metric3dv2": Metric3Dv2Model,
}

_LOADED_MODEL_INSTANCES: Dict[str, MetricDepthModel] = {}

def get_metric_model(model_key: str) -> Optional[MetricDepthModel]:
    """Retrieves or lazy-instantiates model by key."""
    if model_key not in MODEL_CLASSES:
        return None
    if model_key not in _LOADED_MODEL_INSTANCES:
        _LOADED_MODEL_INSTANCES[model_key] = MODEL_CLASSES[model_key]()
    return _LOADED_MODEL_INSTANCES[model_key]


# ---------------------------------------------------------------------------
# HELPER FUNCTIONS FOR STATS, DIAGNOSTICS, AGREEMENT & VALIDATION
# ---------------------------------------------------------------------------

def build_model_result_dict(
    model_key: str,
    model_name: str,
    device: str,
    param_count: str,
    depth_map_m: np.ndarray,
    pre_ms: float,
    inf_ms: float,
    post_ms: float
) -> Dict[str, Any]:
    """Builds comprehensive model output payload with depth statistics and compression diagnostics."""
    d_min = round(float(np.min(depth_map_m)), 2)
    d_max = round(float(np.max(depth_map_m)), 2)
    d_mean = round(float(np.mean(depth_map_m)), 2)
    d_median = round(float(np.median(depth_map_m)), 2)
    d_p10 = round(float(np.percentile(depth_map_m, 10)), 2)
    d_p25 = round(float(np.percentile(depth_map_m, 25)), 2)
    d_p50 = round(float(np.percentile(depth_map_m, 50)), 2)
    d_p75 = round(float(np.percentile(depth_map_m, 75)), 2)
    d_p90 = round(float(np.percentile(depth_map_m, 90)), 2)
    d_std = round(float(np.std(depth_map_m)), 2)
    spread = round(float(d_max - d_min), 2)
    p10_p90_spread = round(float(d_p90 - d_p10), 2)
    valid_ratio = round(float(np.mean((depth_map_m > 0.1) & (~np.isnan(depth_map_m)))), 3)

    is_compressed = p10_p90_spread < 3.0 or spread < 5.0
    compression_note = (
        "Depth distribution appears highly compressed. Further real-world validation is required."
        if is_compressed
        else "Normal depth dynamic distribution."
    )

    tot_ms = round(pre_ms + inf_ms + post_ms, 1)

    return {
        "model_key": model_key,
        "model_name": model_name,
        "status": "SUCCESS",
        "device": device,
        "device_info": {
            "device": device,
            "param_count": param_count,
            "precision": "fp32"
        },
        "inference_time_ms": round(inf_ms, 1),
        "preprocessing_time_ms": round(pre_ms, 1),
        "postprocessing_time_ms": round(post_ms, 1),
        "total_time_ms": tot_ms,
        "depth_min_m": d_min,
        "depth_max_m": d_max,
        "depth_mean_m": d_mean,
        "depth_median_m": d_median,
        "depth_p10_m": d_p10,
        "depth_p25_m": d_p25,
        "depth_p50_m": d_p50,
        "depth_p75_m": d_p75,
        "depth_p90_m": d_p90,
        "depth_std_m": d_std,
        "depth_spread_m": spread,
        "p10_p90_spread_m": p10_p90_spread,
        "valid_pixel_ratio": valid_ratio,
        "is_compressed": is_compressed,
        "compression_note": compression_note,
        "error_reason": None,
        "depth_map_m": depth_map_m
    }


def compute_model_agreement(successful_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculates pixel-wise/distributional agreement across multiple model predictions."""
    if not successful_results:
        return {
            "agreement_rating": "N/A",
            "mean_p50_m": None,
            "std_p50_m": None,
            "spread_p50_m": None,
            "note": "No successful model predictions available to score agreement."
        }

    p50_vals = [r["depth_p50_m"] for r in successful_results if r.get("depth_p50_m") is not None]
    if not p50_vals:
        return {"agreement_rating": "N/A", "note": "No P50 values."}

    mean_p50 = float(np.mean(p50_vals))
    std_p50 = float(np.std(p50_vals))
    spread_p50 = float(np.max(p50_vals) - np.min(p50_vals))
    rel_std = (std_p50 / max(0.1, mean_p50)) * 100.0

    if rel_std < 15.0:
        rating = "HIGH"
    elif rel_std < 35.0:
        rating = "MODERATE"
    else:
        rating = "LOW"

    return {
        "agreement_rating": rating,
        "mean_p50_m": round(mean_p50, 2),
        "std_p50_m": round(std_p50, 2),
        "spread_p50_m": round(spread_p50, 2),
        "relative_std_pct": round(rel_std, 1),
        "models_compared_count": len(p50_vals),
        "note": f"Model agreement is {rating} (relative std across models: {rel_std:.1f}%)."
    }


def compute_stage1_depth_validation(
    validation_points: List[Dict[str, Any]],
    successful_results: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Evaluates Stage 1 metric depth against real measured camera-to-surface distances.
    Calculates MAE, RMSE, Relative Error, and Median Relative Error.
    STRICTLY SEPARATED from Stage 4 building-height formulas.
    """
    validation_results = []
    for res in successful_results:
        depth_m = res.get("depth_map_m")
        if depth_m is None:
            continue

        h, w = depth_m.shape
        point_evals = []
        abs_errors = []
        rel_errors = []

        for pt in validation_points:
            px = max(0, min(w - 1, int(pt.get("x", 0))))
            py = max(0, min(h - 1, int(pt.get("y", 0))))
            actual_m = float(pt.get("measured_depth_m", 0.0))
            if actual_m <= 0:
                continue

            pred_m = float(depth_m[py, px])
            abs_err = abs(pred_m - actual_m)
            rel_err = abs_err / actual_m

            abs_errors.append(abs_err)
            rel_errors.append(rel_err)
            point_evals.append({
                "x": px,
                "y": py,
                "label": pt.get("label", f"Point ({px},{py})"),
                "predicted_depth_m": round(pred_m, 2),
                "measured_depth_m": round(actual_m, 2),
                "abs_error_m": round(abs_err, 2),
                "rel_error_pct": round(rel_err * 100.0, 2)
            })

        if abs_errors:
            mae = round(float(np.mean(abs_errors)), 2)
            rmse = round(float(np.sqrt(np.mean(np.square(abs_errors)))), 2)
            mean_rel_pct = round(float(np.mean(rel_errors)) * 100.0, 2)
            med_rel_pct = round(float(np.median(rel_errors)) * 100.0, 2)
        else:
            mae = rmse = mean_rel_pct = med_rel_pct = None

        validation_results.append({
            "model_key": res["model_key"],
            "model_name": res["model_name"],
            "valid_samples_count": len(point_evals),
            "mae_m": mae,
            "rmse_m": rmse,
            "mean_relative_error_pct": mean_rel_pct,
            "median_relative_error_pct": med_rel_pct,
            "point_evaluations": point_evals
        })

    return validation_results


def run_zero_shot_benchmark(
    img: Image.Image,
    selected_model_keys: List[str],
    validation_points: Optional[List[Dict[str, Any]]] = None,
    probe_pixel: Optional[List[int]] = None,
    selected_bbox: Optional[List[int]] = None,
    image_type: str = "aerial"
) -> Dict[str, Any]:
    """
    Executes the complete Zero-Shot Aerial Metric Depth Benchmark across requested models.
    Stage 3 geometry remains FROZEN.
    """
    from services.depth_estimator import generate_depth_colormaps
    from services.segmentation_service import measure_building_height
    from models.depth import CalibrationConfig

    orig_w, orig_h = img.size
    dw, dh = bounded_benchmark_size(orig_w, orig_h)

    model_results = []
    successful_results = []

    # Sample building bboxes (fractional coordinates converted to working resolution)
    sample_bldgs_def = [
        ([int(dw * 0.35), int(dh * 0.3), int(dw * 0.25), int(dh * 0.25)], "Central Structure"),
        ([int(dw * 0.15), int(dh * 0.4), int(dw * 0.2), int(dw * 0.2)], "West Complex"),
        ([int(dw * 0.65), int(dh * 0.35), int(dw * 0.2), int(dw * 0.22)], "East Wing")
    ]

    for key in selected_model_keys:
        model_instance = get_metric_model(key)
        if model_instance is None:
            model_results.append({
                "model_key": key,
                "model_name": key.upper(),
                "status": "UNAVAILABLE",
                "device": "cpu",
                "error_reason": f"Unknown model key '{key}'",
                "inference_time_ms": 0.0,
                "building_stats": []
            })
            continue

        res = model_instance.predict(img, image_type=image_type)
        depth_m = res.get("depth_map_m")

        if res["status"] == "SUCCESS" and depth_m is not None:
            # 1. Visualization (normalized relative depth for display ONLY)
            lo, hi = float(np.percentile(depth_m, 1)), float(np.percentile(depth_m, 99))
            d_rel = (depth_m - lo) / max(1e-4, hi - lo)
            d_rel = np.clip(d_rel, 0.0, 1.0)
            colormaps = generate_depth_colormaps(d_rel)
            res["depth_colormap_url"] = colormaps["turbo"]

            # 2. Per-Building Metric Depth & FROZEN Stage 3 Height Stress Test
            building_stats = []
            central_stage3_test = None

            for bbox, bldg_name in sample_bldgs_def:
                # Stage 1 building depth stats
                bx1, by1, bw, bh = bbox
                bx2, by2 = min(dw, bx1 + bw), min(dh, by1 + bh)
                patch = depth_m[by1:by2, bx1:bx2]

                p10 = round(float(np.percentile(patch, 10)), 2)
                p50 = round(float(np.percentile(patch, 50)), 2)
                p90 = round(float(np.percentile(patch, 90)), 2)
                b_mean = round(float(np.mean(patch)), 2)
                b_std = round(float(np.std(patch)), 2)

                # FROZEN Stage 3 building height evaluation
                calib = CalibrationConfig(mode="relative")
                stage3_bm = measure_building_height(
                    depth_map=depth_m,
                    bbox=bbox,
                    calibration=calib,
                    building_name=bldg_name,
                    ground_truth_height_m=175.0 if bldg_name == "Central Structure" else None
                )

                bldg_info = {
                    "building_name": bldg_name,
                    "bbox": bbox,
                    "depth_p10_m": p10,
                    "depth_p50_m": p50,
                    "depth_p90_m": p90,
                    "mean_depth_m": b_mean,
                    "std_depth_m": b_std,
                    "stage3_predicted_height_m": stage3_bm.calibrated_height_m,
                    "stage3_ground_elevation_m": stage3_bm.ground_elevation_m,
                    "stage3_roof_elevation_m": stage3_bm.roof_elevation_m,
                    "stage3_status": stage3_bm.calculation_status,
                }
                building_stats.append(bldg_info)

                if bldg_name == "Central Structure":
                    central_stage3_test = {
                        "building_name": bldg_name,
                        "metric_depth_p50_m": p50,
                        "predicted_height_m": stage3_bm.calibrated_height_m,
                        "ground_truth_height_m": 175.0,
                        "absolute_error_m": stage3_bm.error_m,
                        "relative_error_pct": stage3_bm.relative_error_pct,
                        "accuracy_pct": stage3_bm.accuracy_pct,
                        "stage3_status": stage3_bm.calculation_status,
                        "vertical_reference": stage3_bm.camera_parameter_status.get("camera_pose", "Ground Normal Alignment")
                    }

            res["building_stats"] = building_stats
            res["stage3_stress_test"] = central_stage3_test
            successful_results.append(res)

        # Sanitize depth_map_m array before serialization
        res_copy = dict(res)
        res_copy.pop("depth_map_m", None)
        model_results.append(res_copy)

    # 3. Model Agreement Analysis
    agreement = compute_model_agreement(successful_results)

    # 4. Pixel Probe across models
    pixel_probe_res = None
    if probe_pixel and len(probe_pixel) == 2:
        px, py = probe_pixel
        probe_models = []
        for s in successful_results:
            d_map = s.get("depth_map_m")
            if d_map is not None:
                h_m, w_m = d_map.shape
                c_px = max(0, min(w_m - 1, int(px)))
                c_py = max(0, min(h_m - 1, int(py)))
                probe_models.append({
                    "model_key": s["model_key"],
                    "model_name": s["model_name"],
                    "depth_m": round(float(d_map[c_py, c_px]), 2)
                })
        pixel_probe_res = {
            "x": px,
            "y": py,
            "model_predictions": probe_models
        }

    # 5. Region Analysis across models
    region_res = None
    if selected_bbox and len(selected_bbox) == 4:
        rx, ry, rw, rh = selected_bbox
        region_models = []
        for s in successful_results:
            d_map = s.get("depth_map_m")
            if d_map is not None:
                h_m, w_m = d_map.shape
                rx1, ry1 = max(0, min(w_m - 1, int(rx))), max(0, min(h_m - 1, int(ry)))
                rx2, ry2 = min(w_m, rx1 + max(1, int(rw))), min(h_m, ry1 + max(1, int(rh)))
                r_patch = d_map[ry1:ry2, rx1:rx2]

                region_models.append({
                    "model_key": s["model_key"],
                    "model_name": s["model_name"],
                    "min_m": round(float(np.min(r_patch)), 2),
                    "max_m": round(float(np.max(r_patch)), 2),
                    "mean_m": round(float(np.mean(r_patch)), 2),
                    "median_m": round(float(np.median(r_patch)), 2),
                    "p10_m": round(float(np.percentile(r_patch, 10)), 2),
                    "p25_m": round(float(np.percentile(r_patch, 25)), 2),
                    "p50_m": round(float(np.percentile(r_patch, 50)), 2),
                    "p75_m": round(float(np.percentile(r_patch, 75)), 2),
                    "p90_m": round(float(np.percentile(r_patch, 90)), 2),
                    "std_m": round(float(np.std(r_patch)), 2)
                })
        region_res = {
            "bbox": selected_bbox,
            "model_predictions": region_models
        }

    # 6. Stage 1 Metric Depth Validation
    validation_summary = []
    if validation_points:
        validation_summary = compute_stage1_depth_validation(validation_points, successful_results)

    # 7. Best Model Recommendation Logic based on empirical evidence
    best_recommendation = determine_best_model_recommendation(successful_results, validation_summary)

    return {
        "timestamp": str(time.time()),
        "image_dimensions": [orig_w, orig_h],
        "selected_models": selected_model_keys,
        "model_results": model_results,
        "model_agreement": agreement,
        "pixel_probe": pixel_probe_res,
        "region_analysis": region_res,
        "validation_summary": validation_summary,
        "best_model_recommendation": best_recommendation
    }


def determine_best_model_recommendation(
    successful_results: List[Dict[str, Any]],
    validation_summary: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Determines the best zero-shot metric depth model based on empirical benchmark evidence."""
    if not successful_results:
        return {
            "best_model_key": None,
            "best_model_name": "None",
            "recommendation": "No depth models produced successful predictions.",
            "next_step": "Investigate environment or model loading issues."
        }

    # If Stage 1 Ground-Truth depth points exist, rank by lowest MAE / RMSE
    valid_with_mae = [v for v in validation_summary if v.get("mae_m") is not None]
    if valid_with_mae:
        valid_with_mae.sort(key=lambda x: (x["mae_m"], x["rmse_m"]))
        best = valid_with_mae[0]
        return {
            "best_model_key": best["model_key"],
            "best_model_name": best["model_name"],
            "basis": "Stage 1 Measured Metric Depth MAE/RMSE Validation",
            "mae_m": best["mae_m"],
            "rmse_m": best["rmse_m"],
            "mean_rel_error_pct": best["mean_relative_error_pct"],
            "recommendation": f"Model '{best['model_name']}' produced lowest Stage 1 metric depth error (MAE: {best['mae_m']}m, RMSE: {best['rmse_m']}m).",
            "next_step": "Adopt as primary zero-shot metric depth backend for Stage 1."
        }

    # If zero-shot comparison only (no measured camera depth ground truth)
    # Check for uncompressed models with highest P10-P90 spread and metric consistency
    uncompressed = [r for r in successful_results if not r.get("is_compressed", True)]
    if uncompressed:
        # Prefer DA-V2 Large if available, otherwise DA-V2 Base
        uncompressed.sort(key=lambda x: (x.get("p10_p90_spread_m", 0), x.get("total_time_ms", 9999)), reverse=True)
        best = uncompressed[0]
        return {
            "best_model_key": best["model_key"],
            "best_model_name": best["model_name"],
            "basis": "Zero-Shot Dynamic Range & Non-Compressed Depth Spread",
            "p10_p90_spread_m": best.get("p10_p90_spread_m"),
            "recommendation": f"Model '{best['model_name']}' exhibits healthiest dynamic range (P10-P90 spread: {best.get('p10_p90_spread_m')}m).",
            "next_step": "Recommend for Stage 1. If metric depth on aerial imagery remains compressed overall, proceed to Aerial-Specific Fine-Tuning phase."
        }

    # If all models are compressed or default
    best = successful_results[0]
    return {
        "best_model_key": best["model_key"],
        "best_model_name": best["model_name"],
        "basis": "Default Available Pretrained Baseline",
        "recommendation": "All zero-shot terrestrial metric depth models exhibit compressed metric range on aerial perspective.",
        "next_step": "PRETRAINED MODEL BENCHMARK COMPLETE: Evidence supports moving to Aerial-Specific Fine-Tuning of a Pretrained Foundation Model."
    }

