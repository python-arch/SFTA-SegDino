from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
import torch


@dataclass(frozen=True)
class MaskAugmentConfig:
    out_size: Tuple[int, int] = (256, 256)  # (H,W)
    hflip_p: float = 0.5
    vflip_p: float = 0.0
    rot90_p: float = 0.5
    max_morph_radius: int = 2  # conservative; 0 disables


def _resize_nearest(x: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    h, w = out_hw
    # x: (C,H,W)
    c = x.shape[0]
    out = np.zeros((c, h, w), dtype=x.dtype)
    for i in range(c):
        out[i] = cv2.resize(x[i], (w, h), interpolation=cv2.INTER_NEAREST)
    return out


def _random_morph(mask01: np.ndarray, rng: random.Random, max_radius: int) -> np.ndarray:
    if max_radius <= 0:
        return mask01
    r = rng.randint(0, max_radius)
    if r == 0:
        return mask01
    k = 2 * r + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m_u8 = (mask01 * 255).astype(np.uint8)
    # randomly pick erode/dilate
    if rng.random() < 0.5:
        out = cv2.erode(m_u8, kernel, iterations=1)
    else:
        out = cv2.dilate(m_u8, kernel, iterations=1)
    return (out > 0).astype(np.float32)


def augment_mask_pair(x: torch.Tensor, cfg: MaskAugmentConfig, rng: random.Random) -> torch.Tensor:
    """
    x: (2,H,W) float, channels = [mask, boundary]
    Applies structure-preserving transforms consistently across channels.
    """
    x_np = x.detach().cpu().numpy().astype(np.float32)
    # flips
    if rng.random() < cfg.hflip_p:
        x_np = x_np[..., ::-1]
    if cfg.vflip_p > 0 and rng.random() < cfg.vflip_p:
        x_np = x_np[..., ::-1, :]
    # 90-degree rotation
    if rng.random() < cfg.rot90_p:
        k = rng.choice([1, 2, 3])
        x_np = np.rot90(x_np, k=k, axes=(1, 2)).copy()

    # resize to fixed size
    x_np = _resize_nearest(x_np, cfg.out_size)

    # optional conservative morph on the mask channel; recompute boundary channel from morphed mask
    if cfg.max_morph_radius > 0:
        m = x_np[0]
        m2 = _random_morph(m, rng=rng, max_radius=cfg.max_morph_radius)
        x_np[0] = m2
        # boundary channel stays as-is; training also sees this mild mismatch as augmentation.

    return torch.from_numpy(x_np).float()

