from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class EMAStats:
    """
    Simple EMA mean/var tracker for embeddings.
    Uses diagonal variance (per-dimension) for robustness and simplicity.
    """

    dim: int
    momentum: float = 0.99
    eps: float = 1e-6

    n: int = 0
    mean: Optional[torch.Tensor] = None
    var: Optional[torch.Tensor] = None

    def update(self, z: torch.Tensor) -> None:
        """
        z: (B,D) embedding batch (detached or not; caller controls gradients).
        """
        if z.ndim != 2 or z.shape[1] != self.dim:
            raise ValueError(f"Expected z shape (B,{self.dim}), got {tuple(z.shape)}")

        with torch.no_grad():
            batch_mean = z.mean(dim=0)
            batch_var = z.var(dim=0, unbiased=False)

            if self.mean is None:
                self.mean = batch_mean.detach().clone()
                self.var = batch_var.detach().clone()
            else:
                m = float(self.momentum)
                self.mean = m * self.mean + (1.0 - m) * batch_mean.detach()
                self.var = m * self.var + (1.0 - m) * batch_var.detach()

            self.n += int(z.shape[0])

    def zscore(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.mean is None or self.var is None:
            raise RuntimeError("EMAStats not initialized; call update() with some samples first.")
        mean = self.mean.to(device=z.device, dtype=z.dtype)
        var = self.var.to(device=z.device, dtype=z.dtype)
        std = torch.sqrt(var + self.eps)
        return (z - mean) / std, mean, std


def robust_huber(x: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """
    Elementwise Huber, returns mean.
    """
    absx = x.abs()
    quad = torch.minimum(absx, torch.tensor(delta, device=x.device, dtype=x.dtype))
    lin = absx - quad
    return (0.5 * quad * quad + delta * lin).mean()

