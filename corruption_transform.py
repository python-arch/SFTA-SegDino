from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

from .corruptions import CorruptionSpec, MixedCorruptionSpec, apply_corruption_bgr, apply_mixed_corruption_bgr


SpecType = Union[CorruptionSpec, MixedCorruptionSpec]


@dataclass(frozen=True)
class CorruptionTransform:
    """
    Callable hook for `image_pre_transform` in datasets.

    It takes `(img_bgr, image_id)` and returns a corrupted image.
    """

    spec: SpecType

    def __call__(self, img_bgr, image_id: str):
        if isinstance(self.spec, CorruptionSpec):
            return apply_corruption_bgr(img_bgr, image_id=image_id, spec=self.spec)
        return apply_mixed_corruption_bgr(img_bgr, image_id=image_id, spec=self.spec)
