from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def list_mask_files(mask_dir: Path) -> List[Path]:
    paths: List[Path] = []
    for p in mask_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            paths.append(p)
    paths.sort()
    if not paths:
        raise RuntimeError(f"No mask files found under: {mask_dir}")
    return paths


def read_mask01(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise RuntimeError(f"Failed to read mask: {path}")
    m01 = (m.astype(np.float32) / 255.0)
    m01 = (m01 > 0.5).astype(np.float32)
    return m01


def boundary_band(mask01: np.ndarray, width: int = 2) -> np.ndarray:
    """
    Returns a 0/1 boundary band via morphological gradient.
    """
    if width <= 0:
        return np.zeros_like(mask01, dtype=np.float32)
    k = 2 * int(width) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m_u8 = (mask01 * 255).astype(np.uint8)
    grad = cv2.morphologyEx(m_u8, cv2.MORPH_GRADIENT, kernel)
    return (grad > 0).astype(np.float32)


class MaskPairDataset(Dataset):
    """
    Loads masks from `dataset_root/train/masks` and produces 2-channel inputs:
      channel0 = binary mask
      channel1 = boundary band
    Augmentations are applied in the training script to generate two views.
    """

    def __init__(
        self,
        mask_dir: str | Path,
        boundary_width: int = 2,
        out_size: Tuple[int, int] = (256, 256),
    ) -> None:
        self.mask_dir = Path(mask_dir)
        self.boundary_width = int(boundary_width)
        self.out_size = (int(out_size[0]), int(out_size[1]))  # (H,W)
        self.paths = list_mask_files(self.mask_dir)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        p = self.paths[idx]
        m01 = read_mask01(p)
        b01 = boundary_band(m01, width=self.boundary_width)
        x = np.stack([m01, b01], axis=0).astype(np.float32)  # (2,H,W)

        # Resize to a fixed size so DataLoader can stack batches.
        out_h, out_w = self.out_size
        if x.shape[1] != out_h or x.shape[2] != out_w:
            x_resized = np.zeros((2, out_h, out_w), dtype=np.float32)
            x_resized[0] = cv2.resize(x[0], (out_w, out_h), interpolation=cv2.INTER_NEAREST)
            x_resized[1] = cv2.resize(x[1], (out_w, out_h), interpolation=cv2.INTER_NEAREST)
            x = x_resized

        xt = torch.from_numpy(x).float()
        return xt, p.stem
