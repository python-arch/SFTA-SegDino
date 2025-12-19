from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

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

class TorchvisionImageEncoder(nn.Module):
    """
    Torchvision backbone that returns a pooled embedding.

    Supports offline-safe initialization:
    - weights=None (default) never triggers downloads.
    - weights="imagenet" attempts torchvision default weights; if unavailable, caller should catch and retry.
    """

    def __init__(self, backbone: str = "resnet18", embed_dim: int = 64, weights: Any = None) -> None:
        super().__init__()
        import torchvision.models as models

        backbone = backbone.lower()
        if backbone == "resnet18":
            model = models.resnet18(weights=weights)
            feat_dim = 512
            self.features = nn.Sequential(*list(model.children())[:-1])  # (B,512,1,1)
        elif backbone == "resnet34":
            model = models.resnet34(weights=weights)
            feat_dim = 512
            self.features = nn.Sequential(*list(model.children())[:-1])
        elif backbone == "mobilenet_v3_small":
            model = models.mobilenet_v3_small(weights=weights)
            feat_dim = model.classifier[0].in_features
            self.features = nn.Sequential(model.features, nn.AdaptiveAvgPool2d(1))  # (B,C,1,1)
        else:
            raise ValueError(f"Unsupported torchvision backbone: {backbone}")

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


@dataclass(frozen=True)
class MultiModalConfig:
    embed_dim: int = 64
    mask_width: int = 32
    image_encoder: str = "small_cnn"  # small_cnn|resnet18|resnet34|mobilenet_v3_small
    image_width: int = 32  # used only for small_cnn
    image_weights: str = "none"  # none|imagenet
    fusion: str = "mlp"  # mlp|attn


class MultiModalSymbolicEncoder(nn.Module):
    """
    Produces (z_fused, z_mask, z_img).
    The mask encoder is typically loaded from a pretrained `E_theta` checkpoint and frozen.
    """

    def __init__(self, mask_encoder: SmallMaskEncoder, cfg: MultiModalConfig) -> None:
        super().__init__()
        self.mask_encoder = mask_encoder
        if cfg.image_encoder == "small_cnn":
            self.image_encoder = SmallImageEncoder(embed_dim=cfg.embed_dim, width=cfg.image_width)
        else:
            weights_obj = None
            if cfg.image_weights == "imagenet":
                # Avoid forcing downloads in restricted environments; fall back to random init if weights are unavailable.
                try:
                    import torchvision.models as models

                    if cfg.image_encoder == "resnet18":
                        weights_obj = models.ResNet18_Weights.DEFAULT
                    elif cfg.image_encoder == "resnet34":
                        weights_obj = models.ResNet34_Weights.DEFAULT
                    elif cfg.image_encoder == "mobilenet_v3_small":
                        weights_obj = models.MobileNet_V3_Small_Weights.DEFAULT
                except Exception:
                    weights_obj = None

            self.image_encoder = TorchvisionImageEncoder(
                backbone=cfg.image_encoder,
                embed_dim=cfg.embed_dim,
                weights=weights_obj,
            )
        if cfg.fusion == "attn":
            self.fusion = FusionAttention(embed_dim=cfg.embed_dim)
        else:
            self.fusion = FusionMLP(embed_dim=cfg.embed_dim)
        self.cfg = cfg

    def forward(self, image: torch.Tensor, mask_pair: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        image: (B,3,H,W)
        mask_pair: (B,2,H,W) where channels are [mask01, boundary_band01] or [prob, boundary]
        """
        z_mask = self.mask_encoder(mask_pair)
        mask_ch = mask_pair[:, 0:1].clamp(0.0, 1.0)
        masked_image = image * mask_ch
        z_img = self.image_encoder(masked_image)
        z_fused = self.fusion(z_mask, z_img)
        return z_fused, z_mask, z_img
