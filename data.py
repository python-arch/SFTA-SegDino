from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

try:
    import cv2  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    cv2 = None

try:
    import numpy as np  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    np = None
import torch
import torch.utils.data as data
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _require_cv2() -> None:
    if cv2 is None:  # pragma: no cover
        raise ModuleNotFoundError(
            "OpenCV (cv2) is required. Install with `pip install opencv-python-headless` "
            "(or `pip install opencv-python`)."
        )

def _require_numpy() -> None:
    if np is None:  # pragma: no cover
        raise ModuleNotFoundError("NumPy is required. Install with `pip install numpy`.")


def read_manifest(manifest_path: str | os.PathLike) -> List[str]:
    p = Path(manifest_path)
    if not p.is_file():
        raise FileNotFoundError(f"Manifest not found: {p}")
    lines: List[str] = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        lines.append(s)
    if not lines:
        raise RuntimeError(f"Manifest is empty (after filtering comments): {p}")
    return lines


def list_images(images_dir: Path) -> List[Path]:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")
    paths: List[Path] = []
    for p in images_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            paths.append(p)
    paths.sort()
    if not paths:
        raise RuntimeError(f"No images found under: {images_dir}")
    return paths


def default_mask_path(img_rel: str, img_dir_name: str = "images", mask_dir_name: str = "masks") -> str:
    parts = img_rel.replace("\\", "/").split("/")
    try:
        i = parts.index(img_dir_name)
    except ValueError as e:
        raise RuntimeError(f"Expected '{img_dir_name}' in manifest path: {img_rel}") from e
    parts[i] = mask_dir_name
    base, ext = os.path.splitext("/".join(parts))
    return base + ext


@dataclass(frozen=True)
class SegSample:
    image: torch.Tensor
    mask: Optional[torch.Tensor]
    meta: Dict[str, str]

@dataclass(frozen=True)
class SegViewsSample:
    weak: torch.Tensor
    strong: torch.Tensor
    mask: Optional[torch.Tensor]
    meta: Dict[str, str]

def collate_seg_samples(batch: List[SegSample]) -> Dict[str, object]:
    images = torch.stack([b.image for b in batch], dim=0)
    masks: Optional[torch.Tensor]
    if batch[0].mask is None:
        masks = None
    else:
        masks = torch.stack([b.mask for b in batch if b.mask is not None], dim=0)
    meta: Dict[str, List[str]] = {}
    for k in batch[0].meta.keys():
        meta[k] = [b.meta.get(k, "") for b in batch]
    return {"image": images, "mask": masks, "meta": meta}


def collate_seg_views_samples(batch: List[SegViewsSample]) -> Dict[str, object]:
    weak = torch.stack([b.weak for b in batch], dim=0)
    strong = torch.stack([b.strong for b in batch], dim=0)
    masks: Optional[torch.Tensor]
    if batch[0].mask is None:
        masks = None
    else:
        masks = torch.stack([b.mask for b in batch if b.mask is not None], dim=0)
    meta: Dict[str, List[str]] = {}
    for k in batch[0].meta.keys():
        meta[k] = [b.meta.get(k, "") for b in batch]
    return {"weak": weak, "strong": strong, "mask": masks, "meta": meta}


class ResizeAndNormalize:
    def __init__(
        self,
        size: Tuple[int, int] = (256, 256),
        mean: Tuple[float, float, float] = IMAGENET_MEAN,
        std: Tuple[float, float, float] = IMAGENET_STD,
        mask_threshold: float = 0.5,
    ) -> None:
        self.size = size  # (H, W)
        self.mean = mean
        self.std = std
        self.mask_threshold = mask_threshold

    def __call__(self, img_bgr: np.ndarray, mask_hw: Optional[np.ndarray]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        _require_cv2()
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_resized = TF.resize(img_pil, self.size, interpolation=InterpolationMode.BICUBIC, antialias=True)
        img_t = TF.to_tensor(img_resized)
        img_t = TF.normalize(img_t, self.mean, self.std)

        if mask_hw is None:
            return img_t, None

        mask_pil = Image.fromarray(mask_hw)
        mask_resized = TF.resize(mask_pil, self.size, interpolation=InterpolationMode.NEAREST)
        mask_t = TF.to_tensor(mask_resized)[0:1].float()
        mask_t = (mask_t > self.mask_threshold).float()
        return img_t, mask_t


class ManifestSegmentationDataset(data.Dataset):
    """
    Manifest-driven dataset for canonical structure:

      dataset_root/
        train/images, train/masks
        test/images,  test/masks

    Each manifest line is a relative path from dataset_root,
    e.g. `test/images/0001.png`.
    """

    def __init__(
        self,
        dataset_root: str | os.PathLike,
        split: str,
        *,
        manifest_path: Optional[str | os.PathLike] = None,
        img_dir_name: str = "images",
        mask_dir_name: str = "masks",
        return_mask: bool = True,
        transform: Optional[Callable[[np.ndarray, Optional[np.ndarray]], Tuple[torch.Tensor, Optional[torch.Tensor]]]] = None,
        image_pre_transform: Optional[Callable[[np.ndarray, str], np.ndarray]] = None,
        strict_pair: bool = True,
    ) -> None:
        super().__init__()
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.img_dir_name = img_dir_name
        self.mask_dir_name = mask_dir_name
        self.return_mask = return_mask
        self.transform = transform
        self.image_pre_transform = image_pre_transform
        self.strict_pair = strict_pair

        if manifest_path is not None:
            rels = read_manifest(manifest_path)
            self.img_rel_paths = [r.replace("\\", "/") for r in rels]
        else:
            images_dir = self.dataset_root / split / img_dir_name
            img_paths = list_images(images_dir)
            self.img_rel_paths = [p.relative_to(self.dataset_root).as_posix() for p in img_paths]

        if not self.img_rel_paths:
            raise RuntimeError("No images found/selected.")

        if self.strict_pair:
            missing: List[str] = []
            for img_rel in self.img_rel_paths:
                mask_rel = default_mask_path(img_rel, img_dir_name=self.img_dir_name, mask_dir_name=self.mask_dir_name)
                if not (self.dataset_root / mask_rel).is_file():
                    missing.append(mask_rel)
            if missing:
                examples = "\n".join(missing[:20])
                raise FileNotFoundError(f"Missing {len(missing)} masks. First examples:\n{examples}")

    def __len__(self) -> int:
        return len(self.img_rel_paths)

    def __getitem__(self, idx: int) -> SegSample:
        _require_cv2()
        img_rel = self.img_rel_paths[idx]
        img_path = self.dataset_root / img_rel
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        if self.image_pre_transform is not None:
            _require_numpy()
            img = self.image_pre_transform(img, img_rel)

        mask_t: Optional[torch.Tensor] = None
        mask_hw: Optional[np.ndarray] = None
        mask_rel = default_mask_path(img_rel, img_dir_name=self.img_dir_name, mask_dir_name=self.mask_dir_name)
        mask_path = self.dataset_root / mask_rel
        if self.return_mask:
            mask_hw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_hw is None:
                if self.strict_pair:
                    raise RuntimeError(f"Failed to read mask: {mask_path}")
                mask_hw = None

        if self.transform is not None:
            img_t, mask_t = self.transform(img, mask_hw)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
            if mask_hw is None:
                mask_t = None
            else:
                m = torch.from_numpy(mask_hw).unsqueeze(0).float() / 255.0
                mask_t = (m > 0.5).float()

        sample_id = Path(img_rel).stem
        meta = {
            "id": sample_id,
            "image_rel": img_rel,
            "mask_rel": mask_rel,
            "split": self.split,
        }
        return SegSample(image=img_t, mask=mask_t, meta=meta)


class ManifestConsistencyDataset(data.Dataset):
    """
    Manifest-driven dataset that produces weak/strong image views for consistency training.

    - Images: weak/strong are normalized tensors
    - Masks: optional, resized to match the view size (for debugging/analysis only)
    """

    def __init__(
        self,
        dataset_root: str | os.PathLike,
        split: str,
        *,
        manifest_path: Optional[str | os.PathLike] = None,
        img_dir_name: str = "images",
        mask_dir_name: str = "masks",
        return_mask: bool = False,
        view_transform: Optional[Callable[[np.ndarray], object]] = None,
        image_pre_transform: Optional[Callable[[np.ndarray, str], np.ndarray]] = None,
        mask_size: Tuple[int, int] = (256, 256),
        mask_threshold: float = 0.5,
        strict_pair: bool = True,
    ) -> None:
        super().__init__()
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.img_dir_name = img_dir_name
        self.mask_dir_name = mask_dir_name
        self.return_mask = return_mask
        self.view_transform = view_transform
        self.image_pre_transform = image_pre_transform
        self.mask_size = mask_size
        self.mask_threshold = mask_threshold
        self.strict_pair = strict_pair

        if self.view_transform is None:
            raise ValueError("view_transform is required (e.g., WeakStrongViewTransform).")

        if manifest_path is not None:
            rels = read_manifest(manifest_path)
            self.img_rel_paths = [r.replace("\\", "/") for r in rels]
        else:
            images_dir = self.dataset_root / split / img_dir_name
            img_paths = list_images(images_dir)
            self.img_rel_paths = [p.relative_to(self.dataset_root).as_posix() for p in img_paths]

        if not self.img_rel_paths:
            raise RuntimeError("No images found/selected.")

        if self.strict_pair and self.return_mask:
            missing: List[str] = []
            for img_rel in self.img_rel_paths:
                mask_rel = default_mask_path(img_rel, img_dir_name=self.img_dir_name, mask_dir_name=self.mask_dir_name)
                if not (self.dataset_root / mask_rel).is_file():
                    missing.append(mask_rel)
            if missing:
                examples = "\n".join(missing[:20])
                raise FileNotFoundError(f"Missing {len(missing)} masks. First examples:\n{examples}")

    def __len__(self) -> int:
        return len(self.img_rel_paths)

    def _load_mask_tensor(self, mask_path: Path) -> Optional[torch.Tensor]:
        _require_cv2()
        mask_hw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_hw is None:
            if self.strict_pair:
                raise RuntimeError(f"Failed to read mask: {mask_path}")
            return None
        mask_pil = Image.fromarray(mask_hw)
        mask_resized = TF.resize(mask_pil, self.mask_size, interpolation=InterpolationMode.NEAREST)
        mask_t = TF.to_tensor(mask_resized)[0:1].float()
        mask_t = (mask_t > self.mask_threshold).float()
        return mask_t

    def __getitem__(self, idx: int) -> SegViewsSample:
        _require_cv2()
        img_rel = self.img_rel_paths[idx]
        img_path = self.dataset_root / img_rel
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        if self.image_pre_transform is not None:
            _require_numpy()
            img = self.image_pre_transform(img, img_rel)

        view_pair = self.view_transform(img)
        weak = view_pair.weak
        strong = view_pair.strong

        mask_t: Optional[torch.Tensor] = None
        mask_rel = default_mask_path(img_rel, img_dir_name=self.img_dir_name, mask_dir_name=self.mask_dir_name)
        mask_path = self.dataset_root / mask_rel
        if self.return_mask:
            mask_t = self._load_mask_tensor(mask_path)

        sample_id = Path(img_rel).stem
        meta = {
            "id": sample_id,
            "image_rel": img_rel,
            "mask_rel": mask_rel,
            "split": self.split,
        }
        return SegViewsSample(weak=weak, strong=strong, mask=mask_t, meta=meta)
