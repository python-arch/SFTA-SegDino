from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


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


class MemoryBank:
    """
    Fixed-size ring buffer for normalized embeddings (B,D).

    Intended for guarded priors: compute statistics from a set of admitted embeddings rather than EMA drift.
    """

    def __init__(self, dim: int, capacity: int = 1024, eps: float = 1e-6) -> None:
        self.dim = int(dim)
        self.capacity = int(capacity)
        self.eps = float(eps)
        if self.capacity <= 0:
            raise ValueError("capacity must be > 0")
        self._buf: Optional[torch.Tensor] = None  # (capacity,D) on CPU
        self._ptr: int = 0
        self._full: bool = False

    def size(self) -> int:
        if self._buf is None:
            return 0
        return self.capacity if self._full else self._ptr

    def add(self, z: torch.Tensor) -> None:
        """
        z: (B,D) normalized embeddings; stored on CPU float32.
        """
        if z.ndim != 2 or z.shape[1] != self.dim:
            raise ValueError(f"Expected z shape (B,{self.dim}), got {tuple(z.shape)}")
        if self._buf is None:
            self._buf = torch.zeros((self.capacity, self.dim), dtype=torch.float32, device="cpu")

        zc = z.detach().to(device="cpu", dtype=torch.float32)
        b = int(zc.shape[0])
        if b == 0:
            return

        # If b >= capacity, keep only the most recent capacity items.
        if b >= self.capacity:
            self._buf[:] = zc[-self.capacity :]
            self._ptr = 0
            self._full = True
            return

        end = self._ptr + b
        if end <= self.capacity:
            self._buf[self._ptr : end] = zc
        else:
            first = self.capacity - self._ptr
            self._buf[self._ptr :] = zc[:first]
            self._buf[: end - self.capacity] = zc[first:]
            self._full = True
        self._ptr = end % self.capacity
        if self._ptr == 0 and b > 0:
            self._full = True

    def get(self) -> torch.Tensor:
        """
        Returns current embeddings (N,D) on CPU.
        """
        if self._buf is None:
            return torch.empty((0, self.dim), dtype=torch.float32, device="cpu")
        if self._full:
            return self._buf
        return self._buf[: self._ptr]

    def mean_var(self) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.get()
        if z.numel() == 0:
            raise RuntimeError("MemoryBank is empty.")
        mean = z.mean(dim=0)
        var = z.var(dim=0, unbiased=False)
        return mean, var

    def zscore(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, var = self.mean_var()
        mean = mean.to(device=z.device, dtype=z.dtype)
        var = var.to(device=z.device, dtype=z.dtype)
        std = torch.sqrt(var + self.eps)
        return (z - mean) / std, mean, std

    def proto(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Normalized mean prototype (D,) on given device/dtype.
        """
        mean, _ = self.mean_var()
        p = mean.to(device=device, dtype=dtype)
        return F.normalize(p, dim=0)

    def cosine_sim(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B,D) normalized
        Returns: (B,) cosine similarity to the normalized prototype.
        """
        proto = self.proto(device=z.device, dtype=z.dtype).view(1, -1)
        return (z * proto).sum(dim=-1)
