from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

try:
    import numpy as np  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    np = None

try:
    import cv2  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    cv2 = None


def _require_numpy() -> None:
    if np is None:  # pragma: no cover
        raise ModuleNotFoundError("NumPy is required. Install with `pip install numpy`.")


def _require_cv2() -> None:
    if cv2 is None:  # pragma: no cover
        raise ModuleNotFoundError(
            "OpenCV (cv2) is required. Install with `pip install opencv-python-headless`."
        )


def dice_iou_binary(pred: "np.ndarray", target: "np.ndarray", eps: float = 1e-6) -> Tuple[float, float]:
    _require_numpy()
    pred = pred.astype(bool)
    target = target.astype(bool)
    inter = np.logical_and(pred, target).sum(dtype=np.float64)
    union = np.logical_or(pred, target).sum(dtype=np.float64)
    s = pred.sum(dtype=np.float64) + target.sum(dtype=np.float64)
    dice = float((2.0 * inter + eps) / (s + eps))
    iou = float((inter + eps) / (union + eps))
    return dice, iou


def _binary_boundary(mask_bin: "np.ndarray") -> "np.ndarray":
    _require_numpy()
    _require_cv2()
    mask_u8 = (mask_bin.astype(np.uint8) * 255)
    if mask_u8.max() == 0:
        return np.zeros_like(mask_bin, dtype=np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    boundary = np.zeros_like(mask_u8)
    for cnt in contours:
        cv2.drawContours(boundary, [cnt], -1, color=1, thickness=1)
    return boundary.astype(np.uint8)


def hd95_binary(pred_bin: "np.ndarray", tgt_bin: "np.ndarray") -> float:
    """
    HD95 in pixel units, computed via OpenCV distanceTransform.

    Conventions:
    - If both masks have no foreground: 0
    - If only one has foreground: +inf
    """
    _require_numpy()
    _require_cv2()

    pred_has = bool(pred_bin.any())
    tgt_has = bool(tgt_bin.any())
    if (not pred_has) and (not tgt_has):
        return 0.0
    if pred_has != tgt_has:
        return float("inf")

    Pb = _binary_boundary(pred_bin)
    Tb = _binary_boundary(tgt_bin)

    if Pb.max() == 0:
        Pb = pred_bin.astype(np.uint8)
    if Tb.max() == 0:
        Tb = tgt_bin.astype(np.uint8)

    def dist_to_border(border01: "np.ndarray") -> "np.ndarray":
        inv = np.where(border01 > 0, 0, 1).astype(np.uint8)
        dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
        return dist

    dist_to_T = dist_to_border(Tb)
    dist_to_P = dist_to_border(Pb)

    Py, Px = np.where(Pb > 0)
    Ty, Tx = np.where(Tb > 0)
    if len(Py) == 0 or len(Ty) == 0:
        return float("inf")

    d_PT = dist_to_T[Py, Px]
    d_TP = dist_to_P[Ty, Tx]
    d_all = np.concatenate([d_PT, d_TP], axis=0)
    if d_all.size == 0:
        return float("inf")
    return float(np.percentile(d_all, 95))


def boundary_fscore(pred_bin: "np.ndarray", tgt_bin: "np.ndarray", tolerance_px: int = 2) -> float:
    """
    Boundary F-score with a pixel tolerance (common for segmentation boundary evaluation).
    """
    _require_numpy()
    _require_cv2()

    Pb = _binary_boundary(pred_bin).astype(np.uint8)
    Tb = _binary_boundary(tgt_bin).astype(np.uint8)

    pred_has = bool(Pb.any())
    tgt_has = bool(Tb.any())
    if (not pred_has) and (not tgt_has):
        return 1.0
    if pred_has != tgt_has:
        return 0.0

    k = max(1, int(tolerance_px))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))

    Pb_d = cv2.dilate(Pb, kernel, iterations=1)
    Tb_d = cv2.dilate(Tb, kernel, iterations=1)

    # precision: how much of pred boundary matches target boundary (within tol)
    tp_p = (Pb > 0) & (Tb_d > 0)
    prec = float(tp_p.sum(dtype=np.float64) / max(1.0, float((Pb > 0).sum(dtype=np.float64))))

    # recall: how much of target boundary is recovered by pred boundary (within tol)
    tp_r = (Tb > 0) & (Pb_d > 0)
    rec = float(tp_r.sum(dtype=np.float64) / max(1.0, float((Tb > 0).sum(dtype=np.float64))))

    if prec + rec == 0.0:
        return 0.0
    return float(2.0 * prec * rec / (prec + rec))


@dataclass
class RunningStats:
    n: int = 0
    sum_dice: float = 0.0
    sum_iou: float = 0.0
    sum_bf: float = 0.0
    sum_hd95_finite: float = 0.0
    n_hd95_finite: int = 0
    n_empty_pred: int = 0
    n_full_pred: int = 0

    def update(self, *, dice: float, iou: float, bf: float, hd95: float, empty_pred: bool, full_pred: bool) -> None:
        self.n += 1
        self.sum_dice += float(dice)
        self.sum_iou += float(iou)
        self.sum_bf += float(bf)
        if math.isfinite(hd95):
            self.sum_hd95_finite += float(hd95)
            self.n_hd95_finite += 1
        if empty_pred:
            self.n_empty_pred += 1
        if full_pred:
            self.n_full_pred += 1

    def means(self) -> Dict[str, float]:
        denom = max(1, self.n)
        mean_hd95 = float("inf") if self.n_hd95_finite == 0 else float(self.sum_hd95_finite / max(1, self.n_hd95_finite))
        return {
            "n": float(self.n),
            "dice": float(self.sum_dice / denom),
            "iou": float(self.sum_iou / denom),
            "boundary_f": float(self.sum_bf / denom),
            "hd95": mean_hd95,
            "empty_rate": float(self.n_empty_pred / denom),
            "full_rate": float(self.n_full_pred / denom),
        }
