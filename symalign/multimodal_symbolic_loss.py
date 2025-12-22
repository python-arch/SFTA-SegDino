from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch

from .masks import boundary_band
from .multimodal_encoder import MultiModalSymbolicEncoder
from .prior import EMAStats, MemoryBank, robust_huber


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
class MultiModalSymbolicAlignment:
    encoder: MultiModalSymbolicEncoder
    ema_global: EMAStats
    ema_boundary: EMAStats
    boundary_width: int = 2
    huber_delta: float = 1.0
    output: str = "fused"  # fused|mask|image

    def compute_embeddings(self, image: torch.Tensor, p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        image: (B,3,H,W)
        p: (B,1,H,W) probabilities
        returns: (z_g, z_b) each (B,D) for the selected output channel.
        """
        bnd = boundary_from_prob(p, width=self.boundary_width)
        x_g = torch.cat([p, bnd], dim=1)  # (B,2,H,W)
        x_b = torch.cat([bnd, bnd], dim=1)  # boundary-only view

        # Decouple appearance pooling from the view: always pool with p (not boundary-only).
        zf_g, zm_g, zi_g = self.encoder(image, x_g, pool_mask=p)
        zf_b, zm_b, zi_b = self.encoder(image, x_b, pool_mask=p)

        if self.output == "mask":
            return zm_g, zm_b
        if self.output == "image":
            return zi_g, zi_b
        return zf_g, zf_b

    def update_priors(self, z_g: torch.Tensor, z_b: torch.Tensor, mask: torch.Tensor) -> None:
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


@dataclass
class MultiModalSymbolicAlignmentMemory:
    """
    Guarded memory-bank priors (global+boundary) for a selected output stream.
    """

    encoder: MultiModalSymbolicEncoder
    mem_global: MemoryBank
    mem_boundary: MemoryBank
    boundary_width: int = 2
    huber_delta: float = 1.0
    output: str = "fused"  # fused|mask|image

    min_size: int = 32
    min_cos_sim: float = 0.0

    def compute_embeddings(self, image: torch.Tensor, p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bnd = boundary_from_prob(p, width=self.boundary_width)
        x_g = torch.cat([p, bnd], dim=1)  # (B,2,H,W)
        x_b = torch.cat([bnd, bnd], dim=1)  # boundary-only view

        zf_g, zm_g, zi_g = self.encoder(image, x_g, pool_mask=p)
        zf_b, zm_b, zi_b = self.encoder(image, x_b, pool_mask=p)

        if self.output == "mask":
            return zm_g, zm_b
        if self.output == "image":
            return zi_g, zi_b
        return zf_g, zf_b

    def priors_ready(self) -> bool:
        return self.mem_global.size() >= 1 and self.mem_boundary.size() >= 1

    def update_priors(self, z_g: torch.Tensor, z_b: torch.Tensor, ok: torch.Tensor) -> torch.Tensor:
        """
        ok: (B,) bool gate from confidence/shape sanity.
        Returns: (B,) bool accepted into memory.
        """
        if ok.dtype != torch.bool:
            ok = ok.bool()
        if not ok.any():
            return ok

        accept = ok.clone()
        if self.mem_global.size() >= self.min_size:
            sim = self.mem_global.cosine_sim(z_g)
            accept = accept & (sim >= float(self.min_cos_sim))
        if self.mem_boundary.size() >= self.min_size:
            sim = self.mem_boundary.cosine_sim(z_b)
            accept = accept & (sim >= float(self.min_cos_sim))

        if accept.any():
            self.mem_global.add(z_g[accept])
            self.mem_boundary.add(z_b[accept])
        return accept

    def loss(self, z_g: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        zg, _, _ = self.mem_global.zscore(z_g)
        zb, _, _ = self.mem_boundary.zscore(z_b)
        return robust_huber(zg, delta=self.huber_delta) + robust_huber(zb, delta=self.huber_delta)


@dataclass
class MultiModalSymbolicAlignmentTriple:
    """
    Maintains three independent EMA priors (fused/mask/image), each with global+boundary stats,
    and returns a weighted sum of their alignment losses.
    """

    encoder: MultiModalSymbolicEncoder
    ema_fused_g: EMAStats
    ema_fused_b: EMAStats
    ema_mask_g: EMAStats
    ema_mask_b: EMAStats
    ema_img_g: EMAStats
    ema_img_b: EMAStats
    boundary_width: int = 2
    huber_delta: float = 1.0

    w_fused: float = 1.0
    w_mask: float = 0.5
    w_image: float = 0.5

    def compute_all_embeddings(
        self, image: torch.Tensor, p: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns:
          fused: (z_g, z_b)
          mask: (z_g, z_b)
          image: (z_g, z_b)
        """
        bnd = boundary_from_prob(p, width=self.boundary_width)
        x_g = torch.cat([p, bnd], dim=1)
        x_b = torch.cat([bnd, bnd], dim=1)

        zf_g, zm_g, zi_g = self.encoder(image, x_g, pool_mask=p)
        zf_b, zm_b, zi_b = self.encoder(image, x_b, pool_mask=p)
        return (zf_g, zf_b), (zm_g, zm_b), (zi_g, zi_b)

    def update_priors(
        self,
        fused: Tuple[torch.Tensor, torch.Tensor],
        mask: Tuple[torch.Tensor, torch.Tensor],
        image: Tuple[torch.Tensor, torch.Tensor],
        ok: torch.Tensor,
    ) -> None:
        if ok.dtype != torch.bool:
            ok = ok.bool()
        if not ok.any():
            return
        zfg, zfb = fused
        zmg, zmb = mask
        zig, zib = image
        self.ema_fused_g.update(zfg[ok].detach())
        self.ema_fused_b.update(zfb[ok].detach())
        self.ema_mask_g.update(zmg[ok].detach())
        self.ema_mask_b.update(zmb[ok].detach())
        self.ema_img_g.update(zig[ok].detach())
        self.ema_img_b.update(zib[ok].detach())

    def _loss_one(self, ema_g: EMAStats, ema_b: EMAStats, z_g: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        zg, _, _ = ema_g.zscore(z_g)
        zb, _, _ = ema_b.zscore(z_b)
        return robust_huber(zg, delta=self.huber_delta) + robust_huber(zb, delta=self.huber_delta)

    def priors_ready(self) -> bool:
        return (
            self.ema_fused_g.mean is not None
            and self.ema_fused_b.mean is not None
            and self.ema_mask_g.mean is not None
            and self.ema_mask_b.mean is not None
            and self.ema_img_g.mean is not None
            and self.ema_img_b.mean is not None
        )

    def loss(
        self,
        fused: Tuple[torch.Tensor, torch.Tensor],
        mask: Tuple[torch.Tensor, torch.Tensor],
        image: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        zfg, zfb = fused
        zmg, zmb = mask
        zig, zib = image
        lf = self._loss_one(self.ema_fused_g, self.ema_fused_b, zfg, zfb)
        lm = self._loss_one(self.ema_mask_g, self.ema_mask_b, zmg, zmb)
        li = self._loss_one(self.ema_img_g, self.ema_img_b, zig, zib)
        return float(self.w_fused) * lf + float(self.w_mask) * lm + float(self.w_image) * li


@dataclass
class MultiModalSymbolicAlignmentTripleMemory:
    """
    Triple-stream guarded memory banks (fused/mask/image), each with global+boundary stats.
    """

    encoder: MultiModalSymbolicEncoder
    mem_fused_g: MemoryBank
    mem_fused_b: MemoryBank
    mem_mask_g: MemoryBank
    mem_mask_b: MemoryBank
    mem_img_g: MemoryBank
    mem_img_b: MemoryBank
    boundary_width: int = 2
    huber_delta: float = 1.0

    w_fused: float = 1.0
    w_mask: float = 0.5
    w_image: float = 0.5

    min_size: int = 32
    min_cos_sim: float = 0.0

    def compute_all_embeddings(
        self, image: torch.Tensor, p: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        bnd = boundary_from_prob(p, width=self.boundary_width)
        x_g = torch.cat([p, bnd], dim=1)
        x_b = torch.cat([bnd, bnd], dim=1)

        zf_g, zm_g, zi_g = self.encoder(image, x_g, pool_mask=p)
        zf_b, zm_b, zi_b = self.encoder(image, x_b, pool_mask=p)
        return (zf_g, zf_b), (zm_g, zm_b), (zi_g, zi_b)

    def priors_ready(self) -> bool:
        return all(
            mb.size() >= 1
            for mb in (
                self.mem_fused_g,
                self.mem_fused_b,
                self.mem_mask_g,
                self.mem_mask_b,
                self.mem_img_g,
                self.mem_img_b,
            )
        )

    def _accept(self, mb: MemoryBank, z: torch.Tensor, ok: torch.Tensor) -> torch.Tensor:
        accept = ok.clone()
        if mb.size() >= self.min_size:
            accept = accept & (mb.cosine_sim(z) >= float(self.min_cos_sim))
        return accept

    def update_priors(
        self,
        fused: Tuple[torch.Tensor, torch.Tensor],
        mask: Tuple[torch.Tensor, torch.Tensor],
        image: Tuple[torch.Tensor, torch.Tensor],
        ok: torch.Tensor,
    ) -> torch.Tensor:
        if ok.dtype != torch.bool:
            ok = ok.bool()
        if not ok.any():
            return ok

        zfg, zfb = fused
        zmg, zmb = mask
        zig, zib = image

        accept = ok
        accept = accept & self._accept(self.mem_fused_g, zfg, ok)
        accept = accept & self._accept(self.mem_fused_b, zfb, ok)
        accept = accept & self._accept(self.mem_mask_g, zmg, ok)
        accept = accept & self._accept(self.mem_mask_b, zmb, ok)
        accept = accept & self._accept(self.mem_img_g, zig, ok)
        accept = accept & self._accept(self.mem_img_b, zib, ok)

        if accept.any():
            self.mem_fused_g.add(zfg[accept])
            self.mem_fused_b.add(zfb[accept])
            self.mem_mask_g.add(zmg[accept])
            self.mem_mask_b.add(zmb[accept])
            self.mem_img_g.add(zig[accept])
            self.mem_img_b.add(zib[accept])
        return accept

    def _loss_one(self, mb_g: MemoryBank, mb_b: MemoryBank, z_g: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        zg, _, _ = mb_g.zscore(z_g)
        zb, _, _ = mb_b.zscore(z_b)
        return robust_huber(zg, delta=self.huber_delta) + robust_huber(zb, delta=self.huber_delta)

    def loss(
        self,
        fused: Tuple[torch.Tensor, torch.Tensor],
        mask: Tuple[torch.Tensor, torch.Tensor],
        image: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        zfg, zfb = fused
        zmg, zmb = mask
        zig, zib = image
        lf = self._loss_one(self.mem_fused_g, self.mem_fused_b, zfg, zfb)
        lm = self._loss_one(self.mem_mask_g, self.mem_mask_b, zmg, zmb)
        li = self._loss_one(self.mem_img_g, self.mem_img_b, zig, zib)
        return float(self.w_fused) * lf + float(self.w_mask) * lm + float(self.w_image) * li
