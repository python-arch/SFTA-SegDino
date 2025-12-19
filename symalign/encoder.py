from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallMaskEncoder(nn.Module):
    """
    Tiny conv encoder for mask structure.
    Input: (B,2,H,W) where channels are [mask, boundary_band].
    Output: (B,k) normalized embedding.
    """

    def __init__(self, in_ch: int = 2, embed_dim: int = 64, width: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(width * 2, width * 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(width * 2, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        z = self.head(z)
        return F.normalize(z, dim=-1)


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """
    SimCLR NT-Xent loss for two views. z1,z2 are (B,D) normalized.
    """
    b = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)  # (2B,D)
    sim = (z @ z.t()) / temperature  # (2B,2B)
    # mask out self-similarity
    mask = torch.eye(2 * b, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e9)

    # positives: i<->i+b and i+b<->i
    pos = torch.cat([torch.arange(b, 2 * b, device=z.device), torch.arange(0, b, device=z.device)])
    loss = F.cross_entropy(sim, pos)
    return loss

