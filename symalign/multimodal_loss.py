from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn

from .encoder import nt_xent


@dataclass(frozen=True)
class MultiModalLossWeights:
    mask: float = 1.0
    image: float = 1.0
    cross: float = 0.5
    fused: float = 1.0


class MultiModalContrastiveLoss(nn.Module):
    """
    Multi-objective contrastive training:
    - intra-mask: z_mask(view1) vs z_mask(view2)
    - intra-image: z_img(view1) vs z_img(view2)
    - cross-modal: z_mask(view1) vs z_img(view1)
    - fused: z_fused(view1) vs z_fused(view2)
    """

    def __init__(self, temperature: float = 0.1, w: MultiModalLossWeights | None = None) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.w = w or MultiModalLossWeights()

    def forward(
        self,
        z_fused_1: torch.Tensor,
        z_mask_1: torch.Tensor,
        z_img_1: torch.Tensor,
        z_fused_2: torch.Tensor,
        z_mask_2: torch.Tensor,
        z_img_2: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_mask = nt_xent(z_mask_1, z_mask_2, temperature=self.temperature)
        loss_img = nt_xent(z_img_1, z_img_2, temperature=self.temperature)
        loss_cross = nt_xent(z_mask_1, z_img_1, temperature=self.temperature)
        loss_fused = nt_xent(z_fused_1, z_fused_2, temperature=self.temperature)

        total = (
            self.w.mask * loss_mask
            + self.w.image * loss_img
            + self.w.cross * loss_cross
            + self.w.fused * loss_fused
        )
        logs = {
            "loss_total": float(total.detach().cpu().item()),
            "loss_mask": float(loss_mask.detach().cpu().item()),
            "loss_img": float(loss_img.detach().cpu().item()),
            "loss_cross": float(loss_cross.detach().cpu().item()),
            "loss_fused": float(loss_fused.detach().cpu().item()),
        }
        return total, logs
