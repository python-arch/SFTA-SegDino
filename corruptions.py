from __future__ import annotations

import hashlib
import io
import math
import random
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

try:
    import numpy as np  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    np = None

try:
    import cv2  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    cv2 = None

from PIL import Image, ImageEnhance, ImageFilter


def _require_numpy() -> None:
    if np is None:  # pragma: no cover
        raise ModuleNotFoundError("NumPy is required. Install with `pip install numpy`.")


def _seed_from_parts(*parts: str) -> int:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\0")
    # use 32 bits for compatibility with NumPy RandomState, etc.
    return int.from_bytes(h.digest()[:4], "big", signed=False)


@dataclass(frozen=True)
class CorruptionSpec:
    """
    Single-family corruption specification.

    - `family`: one of {"none","blur","noise","jpeg","illumination"}
    - `severity`: integer in [0,4] where 0 means identity
    """

    family: str
    severity: int
    corruption_id: str = "default"

    def validate(self) -> None:
        if self.family not in {"none", "blur", "noise", "jpeg", "illumination"}:
            raise ValueError(f"Unknown corruption family: {self.family}")
        if not (0 <= self.severity <= 8):
            raise ValueError("severity must be in [0,8]")


@dataclass(frozen=True)
class MixedCorruptionSpec:
    """
    Mixed corruption: deterministically apply 1–2 families per image.
    """

    families: Tuple[str, ...] = ("blur", "noise", "jpeg", "illumination")
    severity: int = 3
    num_ops: int = 2
    corruption_id: str = "mixed_v1"

    def validate(self) -> None:
        if not (0 <= self.severity <= 8):
            raise ValueError("severity must be in [0,8]")
        if not (1 <= self.num_ops <= len(self.families)):
            raise ValueError(f"num_ops must be in [1, {len(self.families)}]")
        bad = [f for f in self.families if f not in {"blur", "noise", "jpeg", "illumination"}]
        if bad:
            raise ValueError(f"Invalid families: {bad}")


def _bgr_to_pil(img_bgr) -> Image.Image:
    if cv2 is not None:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)

    _require_numpy()
    arr = img_bgr[..., ::-1]  # BGR->RGB
    return Image.fromarray(arr.astype("uint8"))


def _pil_to_bgr(img_pil: Image.Image):
    _require_numpy()
    arr = np.asarray(img_pil.convert("RGB"))
    bgr = arr[..., ::-1].copy()
    return bgr


def _jpeg_compress(img_pil: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img_pil.save(buf, format="JPEG", quality=int(quality), optimize=True)
    buf.seek(0)
    out = Image.open(buf)
    return out.convert("RGB")


def _apply_blur(img_pil: Image.Image, severity: int) -> Image.Image:
    if severity <= 0:
        return img_pil
    sigma_map = {
        1: 0.7,
        2: 1.2,
        3: 2.0,
        4: 3.0,
        5: 4.0,
        6: 5.5,
        7: 7.0,
        8: 9.0,
    }
    sigma = sigma_map.get(severity, 3.0)
    return img_pil.filter(ImageFilter.GaussianBlur(radius=float(sigma)))


def _apply_noise(img_pil: Image.Image, severity: int, rng: random.Random) -> Image.Image:
    if severity <= 0:
        return img_pil
    _require_numpy()
    std_map = {
        1: 5.0,
        2: 10.0,
        3: 20.0,
        4: 35.0,
        5: 50.0,
        6: 70.0,
        7: 90.0,
        8: 120.0,
    }
    std = std_map.get(severity, 35.0)
    arr = np.asarray(img_pil.convert("RGB")).astype("float32")
    # deterministic noise seeded via rng -> 32-bit seed for NumPy
    rs = np.random.RandomState(rng.getrandbits(32))
    noise = rs.normal(loc=0.0, scale=std, size=arr.shape).astype("float32")
    out = np.clip(arr + noise, 0.0, 255.0).astype("uint8")
    return Image.fromarray(out, mode="RGB")


def _apply_jpeg(img_pil: Image.Image, severity: int) -> Image.Image:
    if severity <= 0:
        return img_pil
    q_map = {
        1: 80,
        2: 60,
        3: 35,
        4: 15,
        5: 10,
        6: 7,
        7: 5,
        8: 3,
    }
    quality = q_map.get(severity, 15)
    return _jpeg_compress(img_pil, quality=quality)


def _apply_illumination(img_pil: Image.Image, severity: int, rng: random.Random) -> Image.Image:
    if severity <= 0:
        return img_pil
    # deterministically sample parameters within severity-dependent ranges
    # ranges chosen to be noticeable but not fully destructive
    sev = float(severity)
    # brightness factor in [1-d, 1+d]
    d_b = 0.08 + 0.10 * sev
    # contrast factor in [1-d, 1+d]
    d_c = 0.10 + 0.12 * sev
    # gamma in [1-d, 1+d] but asymmetrically allow stronger darkening
    d_g = 0.08 + 0.10 * sev

    b = rng.uniform(1.0 - d_b, 1.0 + d_b)
    c = rng.uniform(1.0 - d_c, 1.0 + d_c)
    g = rng.uniform(1.0 - d_g, 1.0 + d_g)
    b = max(0.1, min(2.0, b))
    c = max(0.1, min(2.0, c))
    g = max(0.2, min(2.5, g))

    out = ImageEnhance.Brightness(img_pil).enhance(float(b))
    out = ImageEnhance.Contrast(out).enhance(float(c))

    # gamma correction: out = out^(1/g)
    _require_numpy()
    arr = np.asarray(out.convert("RGB")).astype("float32") / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    arr = np.power(arr, 1.0 / float(g))
    arr = (arr * 255.0).round().clip(0.0, 255.0).astype("uint8")
    return Image.fromarray(arr, mode="RGB")


def apply_corruption_bgr(
    img_bgr,
    *,
    image_id: str,
    spec: CorruptionSpec,
) -> "np.ndarray":
    """
    Apply a deterministic corruption to a BGR uint8 image.

    Determinism is controlled by hashing `(image_id, spec.family, spec.severity, spec.corruption_id)`.
    """
    spec.validate()
    if spec.family == "none" or spec.severity == 0:
        return img_bgr

    seed = _seed_from_parts(image_id, spec.family, str(spec.severity), spec.corruption_id)
    rng = random.Random(seed)

    img_pil = _bgr_to_pil(img_bgr)
    if spec.family == "blur":
        out = _apply_blur(img_pil, spec.severity)
    elif spec.family == "noise":
        out = _apply_noise(img_pil, spec.severity, rng=rng)
    elif spec.family == "jpeg":
        out = _apply_jpeg(img_pil, spec.severity)
    elif spec.family == "illumination":
        out = _apply_illumination(img_pil, spec.severity, rng=rng)
    else:
        raise ValueError(f"Unhandled family: {spec.family}")

    return _pil_to_bgr(out)


def apply_mixed_corruption_bgr(
    img_bgr,
    *,
    image_id: str,
    spec: MixedCorruptionSpec,
) -> "np.ndarray":
    spec.validate()
    if spec.severity == 0 or spec.num_ops == 0:
        return img_bgr

    seed = _seed_from_parts(image_id, str(spec.severity), str(spec.num_ops), spec.corruption_id)
    rng = random.Random(seed)
    families = list(spec.families)
    rng.shuffle(families)
    ops = families[: spec.num_ops]

    out = img_bgr
    for i, fam in enumerate(ops):
        out = apply_corruption_bgr(
            out,
            image_id=f"{image_id}|op{i}",
            spec=CorruptionSpec(family=fam, severity=spec.severity, corruption_id=spec.corruption_id),
        )
    return out
