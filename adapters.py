from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F


ATTN_NAME_HINTS = ("attn", "attention", "qkv", "q_proj", "k_proj", "v_proj", "query", "key", "value", "proj", "out")
PLACEMENT_TOKENS = ("Q", "K", "V", "P", "F1", "F2")


def count_parameters(model: nn.Module) -> Tuple[int, int, float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100.0 * trainable / max(1, total)
    return total, trainable, pct


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        self.r = int(r)
        self.alpha = int(alpha)
        self.scaling = (self.alpha / self.r) if self.r > 0 else 1.0
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        if self.r > 0:
            device = base.weight.device
            dtype = base.weight.dtype
            self.lora_A = nn.Parameter(torch.zeros(base.in_features, self.r, device=device, dtype=dtype))
            self.lora_B = nn.Parameter(torch.zeros(self.r, base.out_features, device=device, dtype=dtype))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.r > 0:
            y = y + self.scaling * (self.dropout(x) @ self.lora_A @ self.lora_B)
        return y


class SALTLinear(nn.Linear):
    """
    Minimal SALT-like parameterization:
    - freeze base weight
    - perform SVD once (on first forward) and keep U,S,Vt cached
    - learn a low-dim modification of the spectrum:
      - trainable_scale_A, trainable_shift_B on top singulars
      - low-rank residual on remaining spectrum via X @ Y
    """

    def __init__(self, base: nn.Linear, rank: int = 8, r_lora: int = 8, seed: int = 42) -> None:
        # Try to create parameters on the same device/dtype as the base layer.
        try:
            super().__init__(
                base.in_features,
                base.out_features,
                bias=base.bias is not None,
                device=base.weight.device,
                dtype=base.weight.dtype,
            )
        except TypeError:
            super().__init__(base.in_features, base.out_features, bias=base.bias is not None)
            self.to(device=base.weight.device, dtype=base.weight.dtype)

        self.weight.data.copy_(base.weight.data)
        if base.bias is not None and self.bias is not None:
            self.bias.data.copy_(base.bias.data)

        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        torch.manual_seed(int(seed))

        self.done_svd = False
        self.U: Optional[torch.Tensor] = None
        self.S: Optional[torch.Tensor] = None
        self.Vt: Optional[torch.Tensor] = None

        # bound ranks to layer capacity
        max_rank = min(self.weight.shape[0], self.weight.shape[1])
        self.rank = int(min(rank, max_rank))
        self.r_lora = int(r_lora)
        remaining_rank = max(0, max_rank - self.rank)

        self.trainable_scale_A = nn.Parameter(torch.ones(self.rank))
        self.trainable_shift_B = nn.Parameter(torch.zeros(self.rank))

        device = self.weight.device
        dtype = self.weight.dtype
        self.trainable_X = nn.Parameter(torch.randn(remaining_rank, self.r_lora, device=device, dtype=dtype) * 0.01)
        self.trainable_Y = nn.Parameter(torch.randn(self.r_lora, remaining_rank, device=device, dtype=dtype) * 0.01)

    def _compute_svd(self) -> None:
        U, S, Vt = torch.linalg.svd(self.weight, full_matrices=False)
        self.U, self.S, self.Vt = U, S, Vt
        self.done_svd = True

    def _modified_s_matrix(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        assert self.S is not None
        s = self.S.to(device=device, dtype=dtype)
        max_rank = s.shape[0]

        s_diag = torch.diag(s)
        top_s = s[: self.rank]
        modified_top_s = self.trainable_scale_A.to(device=device, dtype=dtype) * top_s + self.trainable_shift_B.to(
            device=device, dtype=dtype
        )

        new_s = s_diag.clone()
        new_s[: self.rank, : self.rank] = torch.diag(modified_top_s)

        if max_rank > self.rank:
            lo = self.trainable_X.to(device=device, dtype=dtype) @ self.trainable_Y.to(device=device, dtype=dtype)
            new_s[self.rank :, self.rank :] = new_s[self.rank :, self.rank :] + lo

        return F.relu(new_s)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self.done_svd:
            self._compute_svd()
        assert self.U is not None and self.Vt is not None

        U = self.U.to(device=input.device, dtype=input.dtype)
        Vt = self.Vt.to(device=input.device, dtype=input.dtype)
        s_new = self._modified_s_matrix(device=input.device, dtype=input.dtype)
        weight_updated = U @ s_new @ Vt
        return F.linear(input, weight_updated, self.bias)


def _should_wrap_linear(name: str, child_name: str, targets: Sequence[str] = ATTN_NAME_HINTS) -> bool:
    nl = name.lower()
    cl = child_name.lower()
    return any(t in nl or t in cl for t in targets)


def parse_lora_placement(placement: str) -> Tuple[str, ...]:
    """
    Parse a placement string like "Q,K,V,P" into normalized placement tokens.
    """
    parts = [p.strip().upper() for p in str(placement).split(",") if p.strip()]
    if not parts:
        raise ValueError("Empty LoRA placement string.")
    invalid = [p for p in parts if p not in PLACEMENT_TOKENS]
    if invalid:
        raise ValueError(
            f"Invalid LoRA placement token(s): {invalid}. "
            f"Allowed: {', '.join(PLACEMENT_TOKENS)}"
        )
    # keep order stable but de-duplicate
    out: List[str] = []
    for p in parts:
        if p not in out:
            out.append(p)
    return tuple(out)


def _match_lora_placement(name: str, child_name: str, placement_tokens: Sequence[str]) -> bool:
    """
    Match layer names for LoRA placement sweeps.

    Tokens:
    - Q/K/V: attention query/key/value projections (and qkv fused layers)
    - P: attention output projection (attn.proj / out_proj / to_out)
    - F1/F2: first/second FFN linear layers (fc1/fc2)
    """
    fn = f"{name}.{child_name}".lower()
    tokens = set(placement_tokens)

    # Q/K/V include fused qkv projection names.
    if "Q" in tokens and any(h in fn for h in ("q_proj", "query", "to_q", "wq", "qkv")):
        return True
    if "K" in tokens and any(h in fn for h in ("k_proj", "key", "to_k", "wk", "qkv")):
        return True
    if "V" in tokens and any(h in fn for h in ("v_proj", "value", "to_v", "wv", "qkv")):
        return True

    # P should target attention output projection, not generic projections.
    if "P" in tokens and ("attn" in fn or "attention" in fn):
        if any(h in fn for h in ("out_proj", "to_out", ".proj", " proj", ".out")):
            return True

    # FFN projections.
    if "F1" in tokens and any(h in fn for h in ("fc1", "linear1", "mlp.0", "ffn.0")):
        return True
    if "F2" in tokens and any(h in fn for h in ("fc2", "linear2", "mlp.2", "ffn.2")):
        return True

    return False


def inject_lora_with_placement(
    module: nn.Module,
    *,
    r: int,
    alpha: int,
    dropout: float,
    placement: str,
) -> int:
    """
    Inject LoRA only in layers selected by placement string, e.g.:
      "Q", "Q,K,V", "P,F1,F2", "Q,K,V,P,F1,F2".
    """
    tokens = parse_lora_placement(placement)
    replaced = 0
    for name, m in list(module.named_modules()):
        for child_name, child in list(m.named_children()):
            if isinstance(child, nn.Linear) and _match_lora_placement(name, child_name, tokens):
                setattr(m, child_name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
                replaced += 1
    return replaced


def inject_lora(module: nn.Module, *, r: int, alpha: int, dropout: float) -> int:
    replaced = 0
    for name, m in list(module.named_modules()):
        for child_name, child in list(m.named_children()):
            if isinstance(child, nn.Linear) and _should_wrap_linear(name, child_name):
                setattr(m, child_name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
                replaced += 1
    return replaced


def inject_salt(module: nn.Module, *, rank: int, r_lora: int, seed: int) -> int:
    replaced = 0
    for name, m in list(module.named_modules()):
        for child_name, child in list(m.named_children()):
            if isinstance(child, nn.Linear) and _should_wrap_linear(name, child_name):
                setattr(m, child_name, SALTLinear(child, rank=rank, r_lora=r_lora, seed=seed))
                replaced += 1
    return replaced


def set_only_adapter_trainable(model: nn.Module) -> int:
    """
    Freeze everything, then re-enable only adapter parameters (LoRA/SALT trainables).
    Returns number of trainable parameters (count, not numel).
    """
    for p in model.parameters():
        p.requires_grad_(False)
    n = 0
    for m in model.modules():
        if isinstance(m, LoRALinear):
            if isinstance(m.lora_A, torch.Tensor):
                m.lora_A.requires_grad_(True)
            if isinstance(m.lora_B, torch.Tensor):
                m.lora_B.requires_grad_(True)
        if isinstance(m, SALTLinear):
            for p in m.parameters():
                # SALTLinear inherits nn.Linear and has frozen weight/bias; other parameters are trainable.
                if p is not m.weight and p is not m.bias:
                    p.requires_grad_(True)
    for p in model.parameters():
        if p.requires_grad:
            n += 1
    return n


@dataclass(frozen=True)
class LoRASpec:
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05


@dataclass(frozen=True)
class SALTSpec:
    rank: int = 8
    r_lora: int = 8
    seed: int = 42


def apply_peft_to_backbone(
    model: nn.Module,
    *,
    adapter: str,
    lora: Optional[LoRASpec] = None,
    salt: Optional[SALTSpec] = None,
) -> Dict[str, int]:
    """
    Inject adapters into `model.backbone` (expects DPT-style wrapper).
    """
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        raise AttributeError("Model has no `.backbone`; cannot apply adapter to backbone.")

    if adapter == "none":
        return {"wrapped": 0}

    if adapter == "lora":
        if lora is None:
            lora = LoRASpec()
        wrapped = inject_lora(backbone, r=lora.r, alpha=lora.alpha, dropout=lora.dropout)
    elif adapter == "salt":
        if salt is None:
            salt = SALTSpec()
        wrapped = inject_salt(backbone, rank=salt.rank, r_lora=salt.r_lora, seed=salt.seed)
    else:
        raise ValueError(f"Unknown adapter: {adapter}")

    set_only_adapter_trainable(model)
    return {"wrapped": wrapped}


def choose_lora_rank_for_budget(
    model: nn.Module,
    *,
    candidate_r: Sequence[int],
    target_trainable_pct: float,
    alpha: int,
    dropout: float,
) -> int:
    """
    Brute-force choose LoRA rank that gets closest to a target trainable % after injection.
    Note: this mutates the model by injecting LoRA at each trial; call on a fresh model copy.
    """
    best_r = int(candidate_r[0])
    best_err = float("inf")

    for r in candidate_r:
        # trial inject on current model copy
        apply_peft_to_backbone(model, adapter="lora", lora=LoRASpec(r=int(r), alpha=int(alpha), dropout=float(dropout)))
        _, _, pct = count_parameters(model)
        err = abs(pct - float(target_trainable_pct))
        if err < best_err:
            best_err = err
            best_r = int(r)
    return best_r
