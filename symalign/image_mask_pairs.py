from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .masks import SUPPORTED_EXTS, boundary_band, read_mask01


def _list_files(dir_path: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for p in dir_path.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in SUPPORTED_EXTS:
            continue
        out[p.stem] = p
    return out


def read_image_rgb01(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return (img.astype(np.float32) / 255.0)


class ImageMaskPairDataset(Dataset):
    """
    Loads paired (image, mask) from:
      <dataset_root>/train/images
      <dataset_root>/train/masks

    Returns:
      image: (3,H,W) float32 in [0,1]
      mask_pair: (2,H,W) float32 where channels are [mask01, boundary_band01]
      image_id: string (stem)
    """

    def __init__(
        self,
        dataset_root: str | Path,
        split: str = "train",
        img_dir_name: str = "images",
        mask_dir_name: str = "masks",
        out_size: Tuple[int, int] = (256, 256),
        boundary_width: int = 2,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.img_dir = self.dataset_root / split / img_dir_name
        self.mask_dir = self.dataset_root / split / mask_dir_name
        self.out_size = (int(out_size[0]), int(out_size[1]))  # (H,W)
        self.boundary_width = int(boundary_width)

        if not self.img_dir.exists():
            raise RuntimeError(f"Image dir not found: {self.img_dir}")
        if not self.mask_dir.exists():
            raise RuntimeError(f"Mask dir not found: {self.mask_dir}")

        imgs = _list_files(self.img_dir)
        masks = _list_files(self.mask_dir)
        common = sorted(set(imgs.keys()) & set(masks.keys()))
        if not common:
            raise RuntimeError(f"No paired image/mask stems found under: {self.img_dir} and {self.mask_dir}")

        self._pairs: List[tuple[str, Path, Path]] = [(k, imgs[k], masks[k]) for k in common]

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        image_id, img_path, mask_path = self._pairs[idx]

        img = read_image_rgb01(img_path)  # (H,W,3)
        mask01 = read_mask01(mask_path)  # (H,W)
        bnd01 = boundary_band(mask01, width=self.boundary_width)

        out_h, out_w = self.out_size
        if img.shape[0] != out_h or img.shape[1] != out_w:
            img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        if mask01.shape[0] != out_h or mask01.shape[1] != out_w:
            mask01 = cv2.resize(mask01, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
            bnd01 = cv2.resize(bnd01, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

        img_t = torch.from_numpy(img.transpose(2, 0, 1)).float()  # (3,H,W)
        mask_pair = torch.from_numpy(np.stack([mask01, bnd01], axis=0)).float()  # (2,H,W)
        return img_t, mask_pair, image_id


def collate_image_mask_pairs(batch: List[tuple[torch.Tensor, torch.Tensor, str]]) -> tuple[torch.Tensor, torch.Tensor, List[str]]:
    images, masks, ids = zip(*batch)
    return torch.stack(list(images), dim=0), torch.stack(list(masks), dim=0), list(ids)
