from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

try:
    from segdino.symalign.encoder import SmallMaskEncoder
    from segdino.symalign.masks import boundary_band
    from segdino.symalign.prior import EMAStats, robust_huber
except ModuleNotFoundError:
    from symalign.encoder import SmallMaskEncoder
    from symalign.masks import boundary_band
    from symalign.prior import EMAStats, robust_huber


def boundary_from_prob(p: torch.Tensor, width: int = 2) -> torch.Tensor:
    """
    p: (B,1,H,W) float in [0,1]
    returns: (B,1,H,W) float in {0,1} boundary band computed per-sample on CPU via OpenCV.
    """
    bs = p.shape[0]
    outs = []
    for i in range(bs):
        pi = p[i, 0].detach().cpu().numpy().astype(np.float32)
        bi = boundary_band((pi > 0.5).astype(np.float32), width=width)
        outs.append(torch.from_numpy(bi).unsqueeze(0))
    return torch.stack(outs, dim=0).to(device=p.device, dtype=p.dtype)


@dataclass
class SymbolicAlignment:
    encoder: SmallMaskEncoder
    ema_global: EMAStats
    ema_boundary: EMAStats
    boundary_width: int = 2
    huber_delta: float = 1.0

    def compute_embeddings(self, p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        p: (B,1,H,W) probabilities
        returns: (z_g, z_b) each (B,D)
        """
        bnd = boundary_from_prob(p, width=self.boundary_width)
        x_g = torch.cat([p, bnd], dim=1)  # (B,2,H,W)
        # boundary-only view: duplicate boundary band in both channels
        x_b = torch.cat([bnd, bnd], dim=1)
        z_g = self.encoder(x_g)
        z_b = self.encoder(x_b)
        return z_g, z_b

    def update_priors(self, z_g: torch.Tensor, z_b: torch.Tensor, mask: torch.Tensor) -> None:
        """
        Update EMA priors using only samples selected by mask (B,).
        """
        if mask.numel() == 0:
            return
        if mask.dtype != torch.bool:
            mask = mask.bool()
        if mask.any():
            self.ema_global.update(z_g[mask].detach())
            self.ema_boundary.update(z_b[mask].detach())

    def loss(self, z_g: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        zg, _, _ = self.ema_global.zscore(z_g)
        zb, _, _ = self.ema_boundary.zscore(z_b)
        return robust_huber(zg, delta=self.huber_delta) + robust_huber(zb, delta=self.huber_delta)
