from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PairAugmentConfig:
    out_size: Tuple[int, int] = (256, 256)  # (H,W)
    max_rotate_deg: float = 20.0
    hflip_p: float = 0.5
    vflip_p: float = 0.0
    color_jitter_strength: float = 0.2
    gaussian_noise_std: float = 0.02


def _maybe_hflip(x: torch.Tensor, p: float, rng: random.Random) -> torch.Tensor:
    if rng.random() < p:
        return torch.flip(x, dims=[-1])
    return x


def _maybe_vflip(x: torch.Tensor, p: float, rng: random.Random) -> torch.Tensor:
    if rng.random() < p:
        return torch.flip(x, dims=[-2])
    return x


def _rotate_bilinear(x: torch.Tensor, angle_deg: float) -> torch.Tensor:
    """
    x: (C,H,W) or (B,C,H,W)
    """
    if x.ndim == 3:
        x = x.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    b, c, h, w = x.shape
    theta = torch.zeros((b, 2, 3), device=x.device, dtype=x.dtype)
    angle = torch.tensor(angle_deg * 3.141592653589793 / 180.0, device=x.device, dtype=x.dtype)
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    theta[:, 0, 0] = cos_a
    theta[:, 0, 1] = -sin_a
    theta[:, 1, 0] = sin_a
    theta[:, 1, 1] = cos_a

    grid = F.affine_grid(theta, size=x.size(), align_corners=False)
    y = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    if squeeze:
        y = y.squeeze(0)
    return y


def _color_jitter(img: torch.Tensor, strength: float, rng: random.Random) -> torch.Tensor:
    """
    img: (3,H,W), float in [0,1]
    """
    if strength <= 0:
        return img
    # brightness and contrast (very lightweight, deterministic via rng)
    b = rng.uniform(1.0 - strength, 1.0 + strength)
    c = rng.uniform(1.0 - strength, 1.0 + strength)
    mean = img.mean(dim=(1, 2), keepdim=True)
    out = (img - mean) * c + mean
    out = out * b
    return out.clamp(0.0, 1.0)


def _add_gaussian_noise(img: torch.Tensor, std: float, rng: random.Random) -> torch.Tensor:
    if std <= 0:
        return img
    # Uses global torch RNG; determinism controlled by the training script's seeding.
    noise = torch.randn_like(img) * std
    return (img + noise).clamp(0.0, 1.0)


def augment_pair(
    image: torch.Tensor,
    mask_pair: torch.Tensor,
    cfg: PairAugmentConfig,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply the same geometric augmentation to image and mask_pair, and photometric to image only.

    image: (3,H,W) in [0,1]
    mask_pair: (2,H,W) in {0,1}
    """
    image = _maybe_hflip(image, cfg.hflip_p, rng)
    mask_pair = _maybe_hflip(mask_pair, cfg.hflip_p, rng)

    image = _maybe_vflip(image, cfg.vflip_p, rng)
    mask_pair = _maybe_vflip(mask_pair, cfg.vflip_p, rng)

    angle = rng.uniform(-cfg.max_rotate_deg, cfg.max_rotate_deg)
    image = _rotate_bilinear(image, angle_deg=angle)
    mask_pair = _rotate_bilinear(mask_pair, angle_deg=angle)
    mask_pair = (mask_pair > 0.5).float()

    image = _color_jitter(image, strength=cfg.color_jitter_strength, rng=rng)
    image = _add_gaussian_noise(image, std=cfg.gaussian_noise_std, rng=rng)

    return image, mask_pair
