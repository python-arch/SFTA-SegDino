from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

from .encoder import SmallMaskEncoder


class SmallImageEncoder(nn.Module):
    """
    Lightweight CNN that maps a masked RGB image region to an embedding.
    Input: (B,3,H,W) float in [0,1] (or normalized; consistency matters more than scale).
    Output: (B,D) normalized embedding.
    """

    def __init__(self, embed_dim: int = 64, width: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(width * 2, width * 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(width * 2, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        z = self.head(z)
        return F.normalize(z, dim=-1)


class SmallImageBackbone(nn.Module):
    """
    Lightweight CNN feature extractor for RGB images.
    Returns: (B,C,h,w)
    """

    def __init__(self, width: int = 32) -> None:
        super().__init__()
        self.out_dim = int(width) * 2
        self.net = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(width * 2, width * 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def imagenet_normalize_rgb01(x: torch.Tensor) -> torch.Tensor:
    """
    x: (B,3,H,W) float in [0,1]
    """
    mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (x - mean) / std


def _soft_region_pool(feat: torch.Tensor, mask01: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    feat: (B,C,h,w)
    mask01: (B,1,H,W) in [0,1]
    returns: (B,C)
    """
    if mask01.ndim != 4 or mask01.shape[1] != 1:
        raise ValueError(f"Expected mask01 shape (B,1,H,W), got {tuple(mask01.shape)}")
    m = F.interpolate(mask01, size=feat.shape[-2:], mode="bilinear", align_corners=False).clamp(0.0, 1.0)
    num = (feat * m).sum(dim=(2, 3))
    den = m.sum(dim=(2, 3)).clamp_min(eps)
    return num / den


def _global_pool(feat: torch.Tensor) -> torch.Tensor:
    return feat.mean(dim=(2, 3))


def _dilate_mask(mask01: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask01
    k = 2 * int(radius) + 1
    return F.max_pool2d(mask01, kernel_size=k, stride=1, padding=int(radius))


def _mask_uncertainty_stats(p: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    p: (B,1,H,W) float in [0,1]
    Returns:
      stats_vec: (B,4) = [conf_mean, entropy_mean, boundary_sharpness, area_mean]
      stats_dict: named tensors (B,)
    """
    p = p.clamp(eps, 1.0 - eps)
    conf = torch.maximum(p, 1.0 - p).mean(dim=(1, 2, 3))
    ent = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p)).mean(dim=(1, 2, 3))
    dx = (p[:, :, :, 1:] - p[:, :, :, :-1]).abs()
    dy = (p[:, :, 1:, :] - p[:, :, :-1, :]).abs()
    sharp = (dx.mean(dim=(1, 2, 3)) + dy.mean(dim=(1, 2, 3))) * 0.5
    area = p.mean(dim=(1, 2, 3))
    stats_vec = torch.stack([conf, ent, sharp, area], dim=-1)
    return stats_vec, {"conf_mean": conf, "entropy_mean": ent, "boundary_sharpness": sharp, "area_mean": area}


def _get_torchvision_weights_enum(backbone: str):
    try:
        import torchvision.models as models
    except ModuleNotFoundError:
        return None

    mapping = {
        "resnet18": getattr(models, "ResNet18_Weights", None),
        "resnet34": getattr(models, "ResNet34_Weights", None),
        "mobilenet_v3_small": getattr(models, "MobileNet_V3_Small_Weights", None),
    }
    return mapping.get(backbone)


def _download_or_cache_weights(backbone: str) -> Optional[Dict[str, torch.Tensor]]:
    weights_enum_cls = _get_torchvision_weights_enum(backbone)
    if weights_enum_cls is None:
        logging.warning("Torchvision weights for backbone=%s are unavailable (missing torchvision).", backbone)
        return None

    weights_enum = getattr(weights_enum_cls, "DEFAULT", None)
    if weights_enum is None:
        logging.warning("Torchvision backbone %s does not expose DEFAULT weights; using random init.", backbone)
        return None

    cache_dir = Path(os.environ.get("SYALIGN_TORCHVISION_CACHE", Path.home() / ".cache" / "symalign_torchvision"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    url_obj = urlparse(getattr(weights_enum, "url", ""))
    file_name = Path(url_obj.path).name or f"{backbone}_weights.pth"

    try:
        state_dict = load_state_dict_from_url(
            getattr(weights_enum, "url", ""),
            model_dir=str(cache_dir),
            file_name=file_name,
            map_location="cpu",
            check_hash=False,
            progress=True,
        )
        return state_dict
    except Exception as exc:  # pragma: no cover - network/device errors
        logging.warning("Failed to download pretrained weights for %s (%s); falling back to random init.", backbone, exc)
        return None


class TorchvisionFeatureBackbone(nn.Module):
    """
    Torchvision backbone returning a feature map (B,C,h,w).
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        weights: Any = None,
        weights_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        import torchvision.models as models

        backbone = backbone.lower()
        init_weights = weights if weights_state_dict is None else None
        if backbone == "resnet18":
            model = models.resnet18(weights=init_weights)
            self.out_dim = 512
            self.features = nn.Sequential(*list(model.children())[:-2])  # up to layer4
        elif backbone == "resnet34":
            model = models.resnet34(weights=init_weights)
            self.out_dim = 512
            self.features = nn.Sequential(*list(model.children())[:-2])
        elif backbone == "mobilenet_v3_small":
            model = models.mobilenet_v3_small(weights=init_weights)
            self.out_dim = int(model.classifier[0].in_features)
            self.features = model.features
        else:
            raise ValueError(f"Unsupported torchvision backbone: {backbone}")

        if weights_state_dict is not None:
            missing, unexpected = model.load_state_dict(weights_state_dict, strict=False)
            if missing or unexpected:
                logging.warning(
                    "Loaded pretrained weights for %s with missing=%s unexpected=%s",
                    backbone,
                    missing,
                    unexpected,
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class TorchvisionImageEncoder(nn.Module):
    """
    Torchvision backbone that returns a pooled embedding.

    Supports offline-safe initialization:
    - weights=None (default) never triggers downloads.
    - weights="imagenet" attempts torchvision default weights; if unavailable, caller should catch and retry.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        embed_dim: int = 64,
        weights: Any = None,
        weights_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        import torchvision.models as models

        backbone = backbone.lower()
        init_weights = weights if weights_state_dict is None else None
        if backbone == "resnet18":
            model = models.resnet18(weights=init_weights)
            feat_dim = 512
            self.features = nn.Sequential(*list(model.children())[:-1])  # (B,512,1,1)
        elif backbone == "resnet34":
            model = models.resnet34(weights=init_weights)
            feat_dim = 512
            self.features = nn.Sequential(*list(model.children())[:-1])
        elif backbone == "mobilenet_v3_small":
            model = models.mobilenet_v3_small(weights=init_weights)
            feat_dim = model.classifier[0].in_features
            self.features = nn.Sequential(model.features, nn.AdaptiveAvgPool2d(1))  # (B,C,1,1)
        else:
            raise ValueError(f"Unsupported torchvision backbone: {backbone}")

        if weights_state_dict is not None:
            missing, unexpected = model.load_state_dict(weights_state_dict, strict=False)
            if missing or unexpected:
                logging.warning(
                    "Loaded pretrained weights for %s with missing=%s unexpected=%s",
                    backbone,
                    missing,
                    unexpected,
                )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)
        z = self.head(z)
        return F.normalize(z, dim=-1)


class FusionMLP(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, z_mask: torch.Tensor, z_img: torch.Tensor) -> torch.Tensor:
        z = torch.cat([z_mask, z_img], dim=-1)
        return F.normalize(self.proj(z), dim=-1)


class FusionAttention(nn.Module):
    """
    Computes a soft weight over (mask,image) embeddings then projects to embed_dim.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, 2),
        )
        self.proj = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, z_mask: torch.Tensor, z_img: torch.Tensor) -> torch.Tensor:
        concat = torch.cat([z_mask, z_img], dim=-1)
        w = torch.softmax(self.attn(concat), dim=-1)  # (B,2)
        zm = w[:, 0:1] * z_mask
        zi = w[:, 1:2] * z_img
        fused = torch.cat([zm, zi], dim=-1)
        return F.normalize(self.proj(fused), dim=-1)

class UncertaintyGatedFusion(nn.Module):
    """
    Reliability-aware fusion that uses uncertainty statistics to gate (mask, region, global) streams.
    """

    def __init__(self, embed_dim: int, stats_dim: int = 4, hidden: int = 64) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 3 + stats_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 3),
        )
        self.proj = nn.Linear(embed_dim * 3, embed_dim)

    def forward(
        self, z_mask: torch.Tensor, z_region: torch.Tensor, z_global: torch.Tensor, stats: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gate_in = torch.cat([z_mask, z_region, z_global, stats], dim=-1)
        w = torch.softmax(self.gate(gate_in), dim=-1)  # (B,3)
        concat = torch.cat([w[:, 0:1] * z_mask, w[:, 1:2] * z_region, w[:, 2:3] * z_global], dim=-1)
        z = F.normalize(self.proj(concat), dim=-1)
        return z, w


@dataclass(frozen=True)
class MultiModalConfig:
    embed_dim: int = 64
    mask_width: int = 32
    image_encoder: str = "small_cnn"  # small_cnn|resnet18|resnet34|mobilenet_v3_small
    image_width: int = 32  # used only for small_cnn
    image_weights: str = "none"  # none|imagenet|auto
    image_pool: str = "features"  # pixels|features
    pool_dilate_px: int = 0  # used for region pooling
    use_imagenet_norm: bool = True
    fusion: str = "mlp"  # mlp|attn|uncertainty
    gate_hidden: int = 64


class MultiModalSymbolicEncoder(nn.Module):
    """
    Produces (z_fused, z_mask, z_img).
    The mask encoder is typically loaded from a pretrained `E_theta` checkpoint and frozen.
    """

    def __init__(self, mask_encoder: SmallMaskEncoder, cfg: MultiModalConfig) -> None:
        super().__init__()
        self.mask_encoder = mask_encoder
        self.cfg = cfg

        # Image branch: either legacy pixel-masked encoder, or feature-based region pooling.
        self._image_weights_state: Optional[Dict[str, torch.Tensor]] = None
        self._image_weights_obj: Any = None
        weights_mode = (cfg.image_weights or "none").lower()
        if weights_mode in {"imagenet", "auto"}:
            self._image_weights_state = _download_or_cache_weights(cfg.image_encoder)
            if self._image_weights_state is None:
                enum_cls = _get_torchvision_weights_enum(cfg.image_encoder)
                self._image_weights_obj = getattr(enum_cls, "DEFAULT", None) if enum_cls is not None else None
        elif weights_mode not in {"none"}:
            raise ValueError(f"Unsupported image_weights mode: {cfg.image_weights}")

        if cfg.image_pool == "pixels":
            if cfg.image_encoder == "small_cnn":
                self.image_encoder = SmallImageEncoder(embed_dim=cfg.embed_dim, width=cfg.image_width)
            else:
                self.image_encoder = TorchvisionImageEncoder(
                    backbone=cfg.image_encoder,
                    embed_dim=cfg.embed_dim,
                    weights=self._image_weights_obj,
                    weights_state_dict=self._image_weights_state,
                )
            self.image_backbone = None
            self.image_head = None
        elif cfg.image_pool == "features":
            if cfg.image_encoder == "small_cnn":
                self.image_backbone = SmallImageBackbone(width=cfg.image_width)
                feat_dim = int(self.image_backbone.out_dim)
            else:
                self.image_backbone = TorchvisionFeatureBackbone(
                    backbone=cfg.image_encoder,
                    weights=self._image_weights_obj,
                    weights_state_dict=self._image_weights_state,
                )
                feat_dim = int(self.image_backbone.out_dim)
            self.image_head = nn.Sequential(
                nn.Linear(feat_dim, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, cfg.embed_dim),
            )
            self.image_encoder = None
        else:
            raise ValueError(f"Unsupported image_pool mode: {cfg.image_pool}")

        if cfg.fusion == "attn":
            self.fusion = FusionAttention(embed_dim=cfg.embed_dim)
        elif cfg.fusion == "uncertainty":
            self.fusion = UncertaintyGatedFusion(embed_dim=cfg.embed_dim, hidden=cfg.gate_hidden)
        else:
            self.fusion = FusionMLP(embed_dim=cfg.embed_dim)

    def forward(
        self,
        image: torch.Tensor,
        mask_pair: torch.Tensor,
        *,
        pool_mask: Optional[torch.Tensor] = None,
        drop_mask: Optional[torch.Tensor] = None,
        drop_image: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        image: (B,3,H,W)
        mask_pair: (B,2,H,W) where channels are [mask01, boundary_band01] or [prob, boundary]
        pool_mask: optional (B,1,H,W) used for appearance pooling (decouples image pooling from the mask_pair view).
        """
        z_mask = self.mask_encoder(mask_pair)
        pm = pool_mask if pool_mask is not None else mask_pair[:, 0:1]
        pm = pm.clamp(0.0, 1.0)

        stats_vec, stats_named = _mask_uncertainty_stats(pm)

        if self.cfg.image_pool == "pixels":
            assert self.image_encoder is not None
            masked_image = image * pm
            if self.cfg.use_imagenet_norm and (self.cfg.image_weights or "none").lower() in {"imagenet", "auto"}:
                masked_image = imagenet_normalize_rgb01(masked_image)
            z_region = self.image_encoder(masked_image)
            z_global = z_region
        else:
            assert self.image_backbone is not None and self.image_head is not None
            x = image
            if self.cfg.use_imagenet_norm and (self.cfg.image_weights or "none").lower() in {"imagenet", "auto"}:
                x = imagenet_normalize_rgb01(x)
            feat = self.image_backbone(x)  # (B,C,h,w)
            pm_d = _dilate_mask(pm, radius=int(self.cfg.pool_dilate_px))
            v_region = _soft_region_pool(feat, pm_d)
            v_global = _global_pool(feat)
            z_region = F.normalize(self.image_head(v_region), dim=-1)
            z_global = F.normalize(self.image_head(v_global), dim=-1)

        if drop_mask is not None:
            dm = drop_mask.to(device=z_mask.device, dtype=torch.bool).view(-1, 1)
            z_mask = torch.where(dm, torch.zeros_like(z_mask), z_mask)
        if drop_image is not None:
            di = drop_image.to(device=z_region.device, dtype=torch.bool).view(-1, 1)
            z_region = torch.where(di, torch.zeros_like(z_region), z_region)
            z_global = torch.where(di, torch.zeros_like(z_global), z_global)

        aux: Dict[str, torch.Tensor] = {"stats_vec": stats_vec, **stats_named}
        if isinstance(self.fusion, UncertaintyGatedFusion):
            z_fused, w = self.fusion(z_mask, z_region, z_global, stats_vec)
            aux["w_mask"] = w[:, 0]
            aux["w_region"] = w[:, 1]
            aux["w_global"] = w[:, 2]
        elif isinstance(self.fusion, FusionAttention):
            z_fused = self.fusion(z_mask, z_region)
        else:
            z_fused = self.fusion(z_mask, z_region)

        aux["z_img_global_norm"] = z_global.norm(dim=-1)
        if return_aux:
            return z_fused, z_mask, z_region, aux
        return z_fused, z_mask, z_region
