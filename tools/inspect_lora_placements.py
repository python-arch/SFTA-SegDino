#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters import (
    FusedQKVLoRALinear,
    LoRALinear,
    _match_lora_placement,
    inject_lora_with_placement,
    parse_lora_placement,
)

ATOMIC = ("Q", "K", "V", "P", "F1", "F2")


def collect_linear_names(model: nn.Module) -> List[str]:
    out: List[str] = []
    for name, m in list(model.named_modules()):
        for child_name, child in list(m.named_children()):
            if isinstance(child, nn.Linear):
                out.append(f"{name}.{child_name}".strip("."))
    return out


def collect_matches(model: nn.Module, placement_tokens: Sequence[str]) -> List[str]:
    out: List[str] = []
    for name, m in list(model.named_modules()):
        for child_name, child in list(m.named_children()):
            if isinstance(child, nn.Linear) and _match_lora_placement(name, child_name, placement_tokens):
                out.append(f"{name}.{child_name}".strip("."))
    return out


def parse_combo_list(s: str) -> List[str]:
    # combos are semicolon-separated, each combo is comma-separated tokens
    return [x.strip() for x in s.split(";") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description="Inspect LoRA placement matches on DINO backbone linear layers.")
    p.add_argument("--repo_dir", type=str, default="./dinov3")
    p.add_argument("--dino_ckpt", type=str, required=True)
    p.add_argument("--dino_size", type=str, default="s", choices=["s", "b"])
    p.add_argument(
        "--combos",
        type=str,
        default="Q;K;V;P;F1;Q,K;Q,V;F1,F2;Q,K,V;P,F1,F2;Q,K,V,P;Q,K,V,P,F1,F2",
        help="Semicolon-separated placement combos to report.",
    )
    p.add_argument("--show_names", action="store_true", help="Print matched module names.")
    p.add_argument("--max_names", type=int, default=200, help="Max names printed per combo.")
    p.add_argument(
        "--verify_combo",
        type=str,
        default="",
        help="If set (e.g. 'Q' or 'Q,K'), inject LoRA and print actual trainable adapter params.",
    )
    p.add_argument("--verify_r", type=int, default=4)
    p.add_argument("--verify_alpha", type=int, default=16)
    p.add_argument("--verify_dropout", type=float, default=0.0)
    args = p.parse_args()

    repo_dir = Path(args.repo_dir)
    ckpt = Path(args.dino_ckpt)
    if not repo_dir.is_dir():
        raise FileNotFoundError(f"repo_dir not found: {repo_dir}")
    if not ckpt.is_file():
        raise FileNotFoundError(f"dino_ckpt not found: {ckpt}")

    if args.dino_size == "b":
        backbone = torch.hub.load(str(repo_dir), "dinov3_vitb16", source="local", weights=str(ckpt))
    else:
        backbone = torch.hub.load(str(repo_dir), "dinov3_vits16", source="local", weights=str(ckpt))
    backbone.eval()

    linear_names = collect_linear_names(backbone)
    print(f"Total nn.Linear children in backbone: {len(linear_names)}")

    # Atomic token summary.
    print("\n[Atomic token coverage]")
    atomic_matches: Dict[str, List[str]] = {}
    for token in ATOMIC:
        names = collect_matches(backbone, (token,))
        atomic_matches[token] = names
        print(f"  {token}: {len(names)}")

    # Requested combo summary.
    print("\n[Combo coverage]")
    combos = parse_combo_list(args.combos)
    for combo in combos:
        tokens = parse_lora_placement(combo)
        names = collect_matches(backbone, tokens)
        print(f"  {combo:<18} -> {len(names)} modules")
        if args.show_names:
            for n in names[: max(1, int(args.max_names))]:
                print(f"    - {n}")
            if len(names) > int(args.max_names):
                print(f"    ... ({len(names) - int(args.max_names)} more)")

    # Optional quick overlap hint for qkv-fused architectures.
    q = set(atomic_matches["Q"])
    k = set(atomic_matches["K"])
    v = set(atomic_matches["V"])
    if q and (q == k == v):
        print(
            "\n[Note] Q/K/V matched sets are identical. "
            "This usually means fused qkv projections in the backbone."
        )

    if args.verify_combo:
        verify_combo = args.verify_combo.strip()
        print(f"\n[Verify injection] combo={verify_combo}")
        wrapped = inject_lora_with_placement(
            backbone,
            r=int(args.verify_r),
            alpha=int(args.verify_alpha),
            dropout=float(args.verify_dropout),
            placement=verify_combo,
        )
        lora_wrappers = 0
        fused_wrappers = 0
        for m in backbone.modules():
            if isinstance(m, LoRALinear):
                lora_wrappers += 1
            if isinstance(m, FusedQKVLoRALinear):
                fused_wrappers += 1
        trainable = [n for n, p in backbone.named_parameters() if p.requires_grad]
        q_params = [n for n in trainable if ".lora_A.q" in n or ".lora_B.q" in n]
        k_params = [n for n in trainable if ".lora_A.k" in n or ".lora_B.k" in n]
        v_params = [n for n in trainable if ".lora_A.v" in n or ".lora_B.v" in n]
        print(f"  wrapped modules: {wrapped}")
        print(f"  LoRALinear wrappers: {lora_wrappers}")
        print(f"  FusedQKVLoRALinear wrappers: {fused_wrappers}")
        print(f"  trainable adapter params: {len(trainable)}")
        print(f"  trainable q-slice params: {len(q_params)}")
        print(f"  trainable k-slice params: {len(k_params)}")
        print(f"  trainable v-slice params: {len(v_params)}")
        if args.show_names:
            print("  sample trainable params:")
            for n in trainable[: max(1, int(args.max_names))]:
                print(f"    - {n}")


if __name__ == "__main__":
    main()
