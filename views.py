from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

try:
    import cv2  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    cv2 = None
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from .data import IMAGENET_MEAN, IMAGENET_STD


@dataclass(frozen=True)
class ViewPair:
    weak: torch.Tensor
    strong: torch.Tensor


class WeakStrongViewTransform:
    """
    Produces weak/strong augmented *image tensors* suitable for consistency training.

    This transform is image-only (no mask). It assumes input is a BGR uint8 OpenCV image.
    """

    def __init__(
        self,
        size: Tuple[int, int] = (256, 256),
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    ) -> None:
        self.size = size
        self.mean = mean
        self.std = std

        self.weak_aug = T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply([T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02)], p=0.5),
            ]
        )

        self.strong_aug = T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.05)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([T.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))], p=0.5),
            ]
        )

    def _to_tensor_norm(self, img_pil: Image.Image) -> torch.Tensor:
        img_resized = TF.resize(img_pil, self.size, interpolation=InterpolationMode.BICUBIC, antialias=True)
        img_t = TF.to_tensor(img_resized)
        img_t = TF.normalize(img_t, self.mean, self.std)
        return img_t

    def __call__(self, img_bgr: np.ndarray) -> ViewPair:
        if cv2 is None:  # pragma: no cover
            raise ModuleNotFoundError(
                "OpenCV (cv2) is required. Install with `pip install opencv-python-headless` "
                "(or `pip install opencv-python`)."
            )
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        weak = self._to_tensor_norm(self.weak_aug(img_pil))
        strong = self._to_tensor_norm(self.strong_aug(img_pil))
        return ViewPair(weak=weak, strong=strong)
