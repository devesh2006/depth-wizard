"""
Multi-Task Aerial Depth & Domain Adaptation Architecture.

Combines a shared pretrained Depth Anything V2 backbone with:
1. Metric Depth Head -> outputs camera-surface distance (depth_map_m) in metres.
2. Auxiliary AGL Height Head -> outputs above-ground-level height (agl_height_map_m) in metres.
3. Auxiliary Semantic Segmentation Head -> outputs semantic logits (semantic_logits) for aerial terrain classes.

CRITICAL ARCHITECTURAL CONSTRAINTS:
- Only depth_map_m is passed into Stage 3 geometry reconstruction.
- agl_height_map_m and semantic_logits provide auxiliary domain supervision during training.
- No artificial post-inference depth scale multipliers (depth * alpha, depth * GSD, etc.).
- Output head range parameterization supports extended aerial ranges (up to 500m) while retaining pretrained features.
"""

import math
import logging
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ExtendedMetricOutputHead(nn.Module):
    """
    Re-parameterized Metric Output Head for Aerial Range (up to max_depth=500m).
    
    Original DA-V2 Metric Outdoor max depth: 80.0m.
    New aerial max depth: 500.0m.
    
    Mathematical Formulation:
        d(x) = max_depth * sigmoid(linear_proj(f(x)))
    
    Retains pretrained backbone feature extraction while allowing gradient flow
    across the expanded 0-500m depth continuum without post-hoc scaling factors.
    """
    def __init__(self, in_channels: int, max_depth: float = 500.0):
        super().__init__()
        self.max_depth = float(max_depth)
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels // 2, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        # Smooth activation mapping to [0, max_depth]
        depth_m = torch.sigmoid(out) * self.max_depth
        return depth_m


class AGLHeightHead(nn.Module):
    """Auxiliary AGL (Above-Ground-Level) Height Prediction Head."""
    def __init__(self, in_channels: int, max_agl: float = 200.0):
        super().__init__()
        self.max_agl = float(max_agl)
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels // 2, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        # AGL height is non-negative, bounded by max_agl
        agl_m = torch.sigmoid(out) * self.max_agl
        return agl_m


class AerialSemanticHead(nn.Module):
    """Auxiliary Semantic Segmentation Head for GAMUS terrain classes."""
    def __init__(self, in_channels: int, num_classes: int = 7):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels // 2, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.relu(out)
        logits = self.conv2(out)
        return logits


class LightweightBackbone(nn.Module):
    """Fallback lightweight feature extraction backbone when HF weights are absent or in tests."""
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)


class MultiTaskAerialDepthModel(nn.Module):
    """
    Multi-Task Model for Aerial Depth, AGL Height, and Semantic Segmentation.
    
    Inputs:
        x: RGB image tensor [B, 3, H, W]
    
    Outputs:
        dict with:
            - "depth_map_m": Camera-surface depth prediction tensor [B, 1, H, W] in metres
            - "agl_height_map_m": AGL height prediction tensor [B, 1, H, W] in metres
            - "semantic_logits": Class logits tensor [B, num_classes, H, W]
    """
    def __init__(
        self,
        pretrained_backbone_name: str = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
        num_semantic_classes: int = 7,
        max_depth_m: float = 500.0,
        max_agl_m: float = 200.0,
        load_pretrained: bool = True
    ):
        super().__init__()
        self.pretrained_name = pretrained_backbone_name
        self.num_semantic_classes = num_semantic_classes
        self.max_depth_m = max_depth_m
        self.max_agl_m = max_agl_m
        self.using_hf_backbone = False

        in_channels = 256
        if load_pretrained:
            try:
                from transformers import AutoModelForDepthEstimation
                hf_model = AutoModelForDepthEstimation.from_pretrained(pretrained_backbone_name)
                # Use HF backbone feature extractor
                self.backbone = hf_model.backbone if hasattr(hf_model, "backbone") else hf_model
                self.using_hf_backbone = True
                in_channels = getattr(self.backbone.config, "hidden_size", 256)
                logger.info(f"Loaded HF backbone from {pretrained_backbone_name}")
            except Exception as e:
                logger.warning(f"Could not load HF backbone '{pretrained_backbone_name}': {e}. Using LightweightBackbone.")
                self.backbone = LightweightBackbone()
                in_channels = 256
        else:
            self.backbone = LightweightBackbone()
            in_channels = 256

        # Modular Heads
        self.depth_head = ExtendedMetricOutputHead(in_channels=in_channels, max_depth=max_depth_m)
        self.agl_head = AGLHeightHead(in_channels=in_channels, max_agl=max_agl_m)
        self.semantic_head = AerialSemanticHead(in_channels=in_channels, num_classes=num_semantic_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, _, orig_h, orig_w = x.shape

        if self.using_hf_backbone:
            # HF backbone outputs feature dict or tuple
            features = self.backbone(x)
            if hasattr(features, "last_hidden_state"):
                feat = features.last_hidden_state
            elif isinstance(features, (list, tuple)):
                feat = features[-1]
            elif isinstance(features, dict):
                feat = list(features.values())[-1]
            else:
                feat = features
        else:
            feat = self.backbone(x)

        # Predict tasks
        depth_out = self.depth_head(feat)
        agl_out = self.agl_head(feat)
        sem_out = self.semantic_head(feat)

        # Upsample outputs to match original image spatial dimensions
        if depth_out.shape[-2:] != (orig_h, orig_w):
            depth_out = F.interpolate(depth_out, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
            agl_out = F.interpolate(agl_out, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
            sem_out = F.interpolate(sem_out, size=(orig_h, orig_w), mode="bilinear", align_corners=False)

        return {
            "depth_map_m": depth_out,
            "agl_height_map_m": agl_out,
            "semantic_logits": sem_out
        }


class MultiTaskAerialLoss(nn.Module):
    """
    Multi-Task Loss Module for Aerial Adaptation & Fine-Tuning.
    
    Loss = lambda_depth * L_depth + lambda_agl * L_agl + lambda_semantic * L_semantic
    
    Handling Missing Ground Truth Labels:
    - If camera_depth is None (e.g. GAMUS dataset), L_depth is NOT computed (0.0).
    - If agl_height is None, L_agl is NOT computed (0.0).
    - If semantic_mask is None, L_semantic is NOT computed (0.0).
    """
    def __init__(
        self,
        depth_weight: float = 1.0,
        agl_weight: float = 0.2,
        semantic_weight: float = 0.1,
        silog_variance_weight: float = 0.85,
        ignore_index: int = 255
    ):
        super().__init__()
        self.depth_weight = depth_weight
        self.agl_weight = agl_weight
        self.semantic_weight = semantic_weight
        self.silog_variance_weight = silog_variance_weight
        self.ignore_index = ignore_index

    def compute_silog_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Scale-Invariant Logarithmic (SiLog) Loss for camera depth."""
        if not mask.any():
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        
        pred_valid = pred[mask]
        target_valid = target[mask]
        
        # Clamp to avoid log(0)
        pred_valid = torch.clamp(pred_valid, min=0.1)
        target_valid = torch.clamp(target_valid, min=0.1)
        
        diff = torch.log(pred_valid) - torch.log(target_valid)
        loss = torch.mean(diff ** 2) - self.silog_variance_weight * (torch.mean(diff) ** 2)
        return torch.sqrt(torch.clamp(loss, min=1e-8))

    def compute_agl_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Smooth L1 loss for AGL height."""
        if not mask.any():
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        return F.smooth_l1_loss(pred[mask], target[mask], beta=1.0)

    def compute_semantic_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Cross-Entropy loss for semantic classes."""
        # target shape: [B, H, W] long
        return F.cross_entropy(logits, target, ignore_index=self.ignore_index)

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, Optional[torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        device = predictions["depth_map_m"].device
        zero_tensor = torch.tensor(0.0, device=device)

        loss_depth = zero_tensor
        loss_agl = zero_tensor
        loss_semantic = zero_tensor

        # 1. Camera Depth Loss (if true metric depth gt exists)
        target_depth = targets.get("camera_depth")
        if target_depth is not None:
            pred_depth = predictions["depth_map_m"].squeeze(1)
            valid_mask = (target_depth > 0.1) & (~torch.isnan(target_depth)) & (~torch.isinf(target_depth))
            if valid_mask.any():
                loss_depth = self.compute_silog_loss(pred_depth, target_depth, valid_mask)

        # 2. AGL Height Loss (if AGL ground truth exists, e.g. GAMUS)
        target_agl = targets.get("agl_height")
        if target_agl is not None:
            pred_agl = predictions["agl_height_map_m"].squeeze(1)
            valid_mask = (target_agl >= 0.0) & (~torch.isnan(target_agl)) & (~torch.isinf(target_agl))
            if valid_mask.any():
                loss_agl = self.compute_agl_loss(pred_agl, target_agl, valid_mask)

        # 3. Semantic Loss (if semantic mask ground truth exists, e.g. GAMUS)
        target_sem = targets.get("semantic_mask")
        if target_sem is not None:
            pred_sem = predictions["semantic_logits"]
            loss_semantic = self.compute_semantic_loss(pred_sem, target_sem.long())

        # Total Loss
        total_loss = (
            self.depth_weight * loss_depth +
            self.agl_weight * loss_agl +
            self.semantic_weight * loss_semantic
        )

        return {
            "loss": total_loss,
            "loss_depth": loss_depth,
            "loss_agl": loss_agl,
            "loss_semantic": loss_semantic
        }
