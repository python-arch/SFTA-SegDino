#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import copy
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

try:
    from segdino.adapters import LoRASpec, SALTSpec, apply_peft_to_backbone, count_parameters
    from segdino.corruption_transform import CorruptionTransform
    from segdino.corruptions import CorruptionSpec, MixedCorruptionSpec
    from segdino.data import (
        ManifestConsistencyDataset,
        ManifestSegmentationDataset,
        ResizeAndNormalize,
        collate_seg_samples,
        collate_seg_views_samples,
    )
    from segdino.metrics import RunningStats, boundary_fscore, dice_iou_binary, hd95_binary
    from segdino.views import WeakStrongViewTransform
except ModuleNotFoundError:  # repo-root execution (no segdino package wrapper)
    from adapters import LoRASpec, SALTSpec, apply_peft_to_backbone, count_parameters
    from corruption_transform import CorruptionTransform
    from corruptions import CorruptionSpec, MixedCorruptionSpec
    from data import (
        ManifestConsistencyDataset,
        ManifestSegmentationDataset,
        ResizeAndNormalize,
        collate_seg_samples,
        collate_seg_views_samples,
    )
    from metrics import RunningStats, boundary_fscore, dice_iou_binary, hd95_binary
    from views import WeakStrongViewTransform

try:
    from segdino.symalign.encoder import SmallMaskEncoder
    from segdino.symalign.prior import EMAStats, MemoryBank
    from segdino.symalign.symbolic_loss import SymbolicAlignment
    from segdino.symalign.multimodal_encoder import MultiModalConfig, MultiModalSymbolicEncoder
    from segdino.symalign.multimodal_symbolic_loss import (
        MultiModalSymbolicAlignment,
        MultiModalSymbolicAlignmentMemory,
        MultiModalSymbolicAlignmentTriple,
        MultiModalSymbolicAlignmentTripleMemory,
    )
except ModuleNotFoundError:  # repo-root execution
    from symalign.encoder import SmallMaskEncoder
    from symalign.prior import EMAStats, MemoryBank
    from symalign.symbolic_loss import SymbolicAlignment
    from symalign.multimodal_encoder import MultiModalConfig, MultiModalSymbolicEncoder
    from symalign.multimodal_symbolic_loss import (
        MultiModalSymbolicAlignment,
        MultiModalSymbolicAlignmentMemory,
        MultiModalSymbolicAlignmentTriple,
        MultiModalSymbolicAlignmentTripleMemory,
    )


def load_ckpt_flex(model: nn.Module, ckpt_path: str, map_location: str) -> None:
    try:
        obj = torch.load(ckpt_path, map_location=map_location, weights_only=True)
    except TypeError:
        obj = torch.load(ckpt_path, map_location=map_location)
    if isinstance(obj, dict) and "state_dict" in obj:
        state = obj["state_dict"]
    else:
        state = obj
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[Warn] Missing keys:", missing)
    if unexpected:
        print("[Warn] Unexpected keys:", unexpected)


def select_tent_params(model: nn.Module) -> List[nn.Parameter]:
    # Freeze everything first, then selectively enable norm affine params.
    for p in model.parameters():
        p.requires_grad_(False)

    params: List[nn.Parameter] = []
    seen: set[int] = set()

    for m in model.modules():
        if not isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.GroupNorm)):
            continue

        w = getattr(m, "weight", None)
        b = getattr(m, "bias", None)

        if isinstance(w, torch.Tensor) and w.requires_grad is False:
            w.requires_grad_(True)
            if id(w) not in seen:
                params.append(w)
                seen.add(id(w))

        if isinstance(b, torch.Tensor) and b.requires_grad is False:
            b.requires_grad_(True)
            if id(b) not in seen:
                params.append(b)
                seen.add(id(b))

    return params


def entropy_loss_from_logits(logits: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = torch.sigmoid(logits)
    p = torch.clamp(p, eps, 1.0 - eps)
    h = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))
    return h.mean()

def kl_to_teacher(student_logits: torch.Tensor, teacher_logits: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    KL( teacher || student ) for Bernoulli per-pixel distributions, averaged.
    This is a stabilizer: keeps student close to a fixed/EMA teacher on the same target images.
    """
    ps = torch.sigmoid(student_logits)
    pt = torch.sigmoid(teacher_logits).detach()
    ps = torch.clamp(ps, eps, 1.0 - eps)
    pt = torch.clamp(pt, eps, 1.0 - eps)
    kl = pt * torch.log(pt / ps) + (1.0 - pt) * torch.log((1.0 - pt) / (1.0 - ps))
    return kl.mean()

def fg_fraction_prior(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """
    Scalar stabilizer: match the mean predicted foreground probability of the student to the teacher.
    Helps prevent the all-background collapse when `teacher_kl_weight` is reduced/disabled.
    """
    ps = torch.sigmoid(student_logits)
    pt = torch.sigmoid(teacher_logits).detach()
    fs = ps.mean()
    ft = pt.mean()
    return F.mse_loss(fs, ft)

def _count_components_cv2(mask01: torch.Tensor) -> int:
    """
    mask01: (H,W) bool/float on CPU
    Returns number of connected components in the foreground (excluding background), or -1 if unavailable.
    """
    try:
        import cv2  # type: ignore
    except Exception:
        return -1
    m = mask01.detach().cpu().numpy()
    m = (m > 0.5).astype("uint8")
    n, _labels = cv2.connectedComponents(m, connectivity=8)
    return int(max(0, n - 1))


def symbolic_ok_mask(
    p: torch.Tensor,
    *,
    conf_thr: float,
    min_fg: float,
    max_fg: float,
    max_components: int,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    p: (B,1,H,W) probability map.
    Returns:
      ok: (B,) bool admission gate
      stats: basic diagnostics tensors (B,)
    """
    conf = torch.maximum(p, 1.0 - p).mean(dim=(1, 2, 3))
    hard = (p > 0.5).float()
    fg_frac = hard.mean(dim=(1, 2, 3))

    ok = conf >= float(conf_thr)
    ok = ok & (fg_frac >= float(min_fg)) & (fg_frac <= float(max_fg))

    comps_list = []
    if max_components and int(max_components) > 0:
        for i in range(p.size(0)):
            comps_list.append(_count_components_cv2(hard[i, 0]))
        comps = torch.tensor(comps_list, device=p.device, dtype=torch.float32)
        ok = ok & (comps <= float(max_components))
    else:
        comps = torch.full_like(conf, -1.0)

    return ok, {"conf": conf, "fg_frac": fg_frac, "components": comps}


def consistency_loss(logits_w: torch.Tensor, logits_s: torch.Tensor) -> torch.Tensor:
    pw = torch.sigmoid(logits_w).detach()
    ps = torch.sigmoid(logits_s)
    return F.mse_loss(ps, pw)


def self_training_loss(logits_s: torch.Tensor, logits_w: torch.Tensor, conf_thr: float = 0.9) -> Tuple[torch.Tensor, float]:
    """
    FixMatch-style for binary segmentation: pseudo-labels from weak view, supervise strong view on confident pixels.
    Returns (loss, fraction_of_pixels_used).
    """
    with torch.no_grad():
        pw = torch.sigmoid(logits_w)
        conf = torch.maximum(pw, 1.0 - pw)
        mask = conf >= conf_thr
        y = (pw >= 0.5).float()
    ps = torch.sigmoid(logits_s)
    if mask.sum().item() == 0:
        return ps.new_tensor(0.0), 0.0
    loss = F.binary_cross_entropy(ps[mask], y[mask])
    frac = float(mask.float().mean().item())
    return loss, frac


def set_trainable(model: nn.Module, trainable: str) -> List[nn.Parameter]:
    """
    Select which parameters are trainable.

    - all: everything trainable (not recommended for source-free baselines)
    - head: only segmentation head trainable (recommended default)
    """
    if trainable not in {"all", "head"}:
        raise ValueError(f"Unknown trainable option: {trainable}")

    if trainable == "all":
        for p in model.parameters():
            p.requires_grad_(True)
        return [p for p in model.parameters() if p.requires_grad]

    # head-only
    for p in model.parameters():
        p.requires_grad_(False)

    head = getattr(model, "head", None)
    if head is None:
        raise AttributeError("Model has no `.head`; cannot use trainable=head mode.")
    for p in head.parameters():
        p.requires_grad_(True)
    return [p for p in model.parameters() if p.requires_grad]


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str, dice_thr: float, boundary_tol_px: int) -> Dict[str, float]:
    model.eval()
    stats = RunningStats()
    for batch in loader:
        if isinstance(batch, dict):
            inputs = batch["image"]
            targets = batch["mask"]
        else:
            inputs = batch.image
            targets = batch.mask

        if inputs.ndim == 3:
            inputs = inputs.unsqueeze(0)
        if targets is not None and targets.ndim == 3:
            targets = targets.unsqueeze(0)

        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs)
        probs = torch.sigmoid(logits)
        preds = (probs > dice_thr).float()

        b = inputs.size(0)
        for i in range(b):
            gt = (targets[i, 0].detach().cpu().numpy() > 0.5)
            pr = (preds[i, 0].detach().cpu().numpy() > 0.5)
            dice, iou = dice_iou_binary(pr, gt)
            bf = boundary_fscore(pr, gt, tolerance_px=boundary_tol_px)
            hd = hd95_binary(pr, gt)
            stats.update(dice=dice, iou=iou, bf=bf, hd95=hd, empty_pred=(not pr.any()), full_pred=bool(pr.all()))
    return stats.means()


def main() -> None:
    parser = argparse.ArgumentParser(description="Source-free baseline adaptation on target_adapt, eval on target_holdout.")
    parser.add_argument("--dataset_root", type=str, default="./segdata/kvasir")
    parser.add_argument("--adapt_manifest", type=str, required=True)
    parser.add_argument("--eval_manifest", type=str, required=True)
    parser.add_argument("--img_dir_name", type=str, default="images")
    parser.add_argument("--mask_dir_name", type=str, default="masks")
    parser.add_argument("--input_h", type=int, default=256)
    parser.add_argument("--input_w", type=int, default=256)

    parser.add_argument("--corruption", type=str, default="mixed", choices=["blur", "noise", "jpeg", "illumination", "mixed"])
    parser.add_argument("--severity", type=int, default=4)
    parser.add_argument("--num_ops", type=int, default=4)
    parser.add_argument("--corruption_id", type=str, default="v1")

    parser.add_argument("--method", type=str, required=True, choices=["entropy", "consistency", "selftrain", "tent"])
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--trainable", type=str, default="head", choices=["head", "all"], help="Which parameters to update for non-TENT methods.")
    parser.add_argument("--dice_thr", type=float, default=0.5)
    parser.add_argument("--boundary_tol_px", type=int, default=2)
    parser.add_argument("--selftrain_conf_thr", type=float, default=0.9)
    parser.add_argument("--teacher_kl_weight", type=float, default=1.0, help="Stabilizer weight (teacher||student KL) on weak view.")
    parser.add_argument(
        "--fg_prior_weight",
        type=float,
        default=0.0,
        help="Stabilizer weight for matching student mean foreground probability to teacher (prevents empty-mask collapse when KL is small).",
    )

    # PEFT-only baselines (adapter injection into backbone attention linears)
    parser.add_argument("--adapter", type=str, default="none", choices=["none", "lora", "salt"])
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--salt_rank", type=int, default=8)
    parser.add_argument("--salt_r_lora", type=int, default=8)
    parser.add_argument("--salt_seed", type=int, default=42)

    # Symbolic alignment (learned E_theta + EMA priors)
    parser.add_argument("--use_symbolic", action="store_true", help="Enable learned-symbolic alignment loss.")
    parser.add_argument(
        "--symbolic_mode",
        type=str,
        default="mask",
        choices=["mask", "multimodal"],
        help="Symbolic descriptor backend: mask-only encoder (mask) or multi-modal encoder (multimodal).",
    )
    parser.add_argument("--symbolic_ckpt", type=str, default=None, help="Path to trained E_theta checkpoint (.pth).")
    parser.add_argument("--multimodal_ckpt", type=str, default=None, help="Path to trained multi-modal symbolic encoder checkpoint (.pth).")
    parser.add_argument(
        "--multimodal_output",
        type=str,
        default="fused",
        choices=["fused", "mask", "image"],
        help="Which multi-modal output to align in symbolic loss (ablation): fused|mask|image.",
    )
    parser.add_argument(
        "--multimodal_prior_mode",
        type=str,
        default="single",
        choices=["single", "triple"],
        help="If multimodal: align a single stream (single) or maintain 3 priors and align fused+mask+image (triple).",
    )
    parser.add_argument("--multimodal_w_fused", type=float, default=1.0)
    parser.add_argument("--multimodal_w_mask", type=float, default=0.5)
    parser.add_argument("--multimodal_w_image", type=float, default=0.5)
    parser.add_argument("--symbolic_lambda", type=float, default=0.1)
    parser.add_argument("--symbolic_warmup_steps", type=int, default=100)
    parser.add_argument("--symbolic_ema_momentum", type=float, default=0.99)
    parser.add_argument("--symbolic_conf_thr", type=float, default=0.9, help="Image-level confidence gate for updating EMA priors.")
    parser.add_argument(
        "--symbolic_prior_type",
        type=str,
        default="ema",
        choices=["ema", "memory"],
        help="Prior mechanism for symbolic alignment: EMA stats (ema) or guarded memory bank (memory).",
    )
    parser.add_argument("--symbolic_mem_capacity", type=int, default=1024)
    parser.add_argument("--symbolic_mem_min", type=int, default=32, help="Minimum memory size before outlier gating activates.")
    parser.add_argument("--symbolic_outlier_cos", type=float, default=0.0, help="Min cosine similarity to memory prototype for admission.")
    parser.add_argument("--symbolic_min_fg", type=float, default=0.0, help="Min foreground fraction for admission (hard threshold at 0.5).")
    parser.add_argument("--symbolic_max_fg", type=float, default=1.0, help="Max foreground fraction for admission (hard threshold at 0.5).")
    parser.add_argument("--symbolic_max_components", type=int, default=0, help="Max connected components for admission (0 disables).")

    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--dino_ckpt", type=str, required=True)
    parser.add_argument("--dino_size", type=str, default="s", choices=["b", "s"])
    parser.add_argument("--repo_dir", type=str, default="./dinov3")

    parser.add_argument("--out_csv", type=str, default="./runs/adapt_baselines_results.csv")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dino_size == "b":
        backbone = torch.hub.load(args.repo_dir, "dinov3_vitb16", source="local", weights=args.dino_ckpt)
        encoder_size = "base"
    else:
        backbone = torch.hub.load(args.repo_dir, "dinov3_vits16", source="local", weights=args.dino_ckpt)
        encoder_size = "small"

    try:
        from segdino.dpt import DPT
    except ModuleNotFoundError:
        from dpt import DPT

    model = DPT(encoder_size=encoder_size, nclass=1, backbone=backbone).to(device)
    load_ckpt_flex(model, args.ckpt, map_location=device)

    # Optional: inject PEFT adapter (then only adapter params remain trainable).
    peft_info: Dict[str, int] = {"wrapped": 0}
    if args.adapter != "none":
        if args.adapter == "lora":
            peft_info = apply_peft_to_backbone(
                model,
                adapter="lora",
                lora=LoRASpec(r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout),
            )
        else:
            peft_info = apply_peft_to_backbone(
                model,
                adapter="salt",
                salt=SALTSpec(rank=args.salt_rank, r_lora=args.salt_r_lora, seed=args.salt_seed),
            )

        # Adapter parameters may have been created on CPU; ensure the whole model is on the selected device.
        model = model.to(device)

        total, trainable, pct = count_parameters(model)
        print(f"[PEFT] adapter={args.adapter} wrapped={peft_info.get('wrapped',0)} trainable={trainable}/{total} ({pct:.3f}%)")

    # Fixed teacher to prevent catastrophic drift/collapse for non-symbolic baselines.
    teacher = copy.deepcopy(model).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # corruption hook (applied before view generation / normalization)
    if args.corruption == "mixed":
        spec = MixedCorruptionSpec(severity=args.severity, num_ops=args.num_ops, corruption_id=args.corruption_id)
    else:
        spec = CorruptionSpec(family=args.corruption, severity=args.severity, corruption_id=args.corruption_id)
    pre = CorruptionTransform(spec=spec)

    view_tf = WeakStrongViewTransform(size=(args.input_h, args.input_w))
    adapt_ds = ManifestConsistencyDataset(
        dataset_root=args.dataset_root,
        split="test",
        manifest_path=args.adapt_manifest,
        img_dir_name=args.img_dir_name,
        mask_dir_name=args.mask_dir_name,
        return_mask=False,
        view_transform=view_tf,
        image_pre_transform=pre,
        mask_size=(args.input_h, args.input_w),
        strict_pair=False,
    )
    adapt_loader = DataLoader(
        adapt_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
        collate_fn=collate_seg_views_samples,
    )

    eval_tf = ResizeAndNormalize(size=(args.input_h, args.input_w))
    eval_ds = ManifestSegmentationDataset(
        dataset_root=args.dataset_root,
        split="test",
        manifest_path=args.eval_manifest,
        img_dir_name=args.img_dir_name,
        mask_dir_name=args.mask_dir_name,
        return_mask=True,
        transform=eval_tf,
        image_pre_transform=pre,
        strict_pair=True,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=collate_seg_samples,
    )

    # configure trainable parameters per method
    if args.method == "tent":
        params = select_tent_params(model)
    else:
        if args.adapter != "none":
            # PEFT path: adapter injection already froze everything except adapter params.
            params = [p for p in model.parameters() if p.requires_grad]
        else:
            params = set_trainable(model, args.trainable)

    if not params:
        raise RuntimeError("No trainable parameters selected. Check method/selection logic.")

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    sym_mask: SymbolicAlignment | None = None
    sym_mm: MultiModalSymbolicAlignment | MultiModalSymbolicAlignmentMemory | None = None
    sym_mm3: MultiModalSymbolicAlignmentTriple | MultiModalSymbolicAlignmentTripleMemory | None = None
    if args.use_symbolic:
        if args.symbolic_mode == "mask":
            if not args.symbolic_ckpt:
                raise ValueError("--use_symbolic with --symbolic_mode mask requires --symbolic_ckpt")

            try:
                obj = torch.load(args.symbolic_ckpt, map_location=device, weights_only=True)
            except TypeError:
                obj = torch.load(args.symbolic_ckpt, map_location=device)

            if isinstance(obj, dict) and "state_dict" in obj:
                state = obj["state_dict"]
                enc_args = obj.get("args", {}) if isinstance(obj.get("args", {}), dict) else {}
            else:
                state = obj
                enc_args = {}

            embed_dim = int(enc_args.get("embed_dim", 64))
            width = int(enc_args.get("width", 32))

            enc = SmallMaskEncoder(in_ch=2, embed_dim=embed_dim, width=width)
            enc.load_state_dict(state, strict=False)
            enc = enc.to(device).eval()
            for p in enc.parameters():
                p.requires_grad_(False)

            sym_mask = SymbolicAlignment(
                encoder=enc,
                ema_global=EMAStats(dim=embed_dim, momentum=args.symbolic_ema_momentum),
                ema_boundary=EMAStats(dim=embed_dim, momentum=args.symbolic_ema_momentum),
            )
            print(
                f"[Symbolic:mask] enabled ckpt={args.symbolic_ckpt} lambda={args.symbolic_lambda} "
                f"warmup={args.symbolic_warmup_steps} ema_m={args.symbolic_ema_momentum} conf_thr={args.symbolic_conf_thr}"
            )
        else:
            if not args.multimodal_ckpt:
                raise ValueError("--use_symbolic with --symbolic_mode multimodal requires --multimodal_ckpt")

            try:
                obj = torch.load(args.multimodal_ckpt, map_location=device, weights_only=True)
            except TypeError:
                obj = torch.load(args.multimodal_ckpt, map_location=device)

            if not (isinstance(obj, dict) and "state_dict" in obj):
                raise ValueError("Expected multimodal checkpoint to be a dict with a 'state_dict' key.")

            state = obj["state_dict"]
            cfg_d = obj.get("cfg", {}) if isinstance(obj.get("cfg", {}), dict) else {}
            embed_dim = int(cfg_d.get("embed_dim", 64))
            mask_width = int(cfg_d.get("mask_width", 32))
            image_width = int(cfg_d.get("image_width", 32))
            fusion = str(cfg_d.get("fusion", "mlp"))
            image_pool = str(cfg_d.get("image_pool", "features"))
            pool_dilate_px = int(cfg_d.get("pool_dilate_px", 0))
            use_imagenet_norm = bool(cfg_d.get("use_imagenet_norm", True))
            gate_hidden = int(cfg_d.get("gate_hidden", 64))

            # Rebuild the multimodal encoder and load weights.
            mask_enc = SmallMaskEncoder(in_ch=2, embed_dim=embed_dim, width=mask_width)
            mm_cfg = MultiModalConfig(
                embed_dim=embed_dim,
                mask_width=mask_width,
                image_width=image_width,
                fusion=fusion,
                image_encoder=str(cfg_d.get("image_encoder", "small_cnn")),
                image_weights=str(cfg_d.get("image_weights", "none")),
                image_pool=image_pool,
                pool_dilate_px=pool_dilate_px,
                use_imagenet_norm=use_imagenet_norm,
                gate_hidden=gate_hidden,
            )
            mm = MultiModalSymbolicEncoder(mask_encoder=mask_enc, cfg=mm_cfg)
            mm.load_state_dict(state, strict=False)
            mm = mm.to(device).eval()
            for p in mm.parameters():
                p.requires_grad_(False)

            if args.multimodal_prior_mode == "triple":
                if args.symbolic_prior_type == "memory":
                    sym_mm3 = MultiModalSymbolicAlignmentTripleMemory(
                        encoder=mm,
                        mem_fused_g=MemoryBank(dim=embed_dim, capacity=args.symbolic_mem_capacity),
                        mem_fused_b=MemoryBank(dim=embed_dim, capacity=args.symbolic_mem_capacity),
                        mem_mask_g=MemoryBank(dim=embed_dim, capacity=args.symbolic_mem_capacity),
                        mem_mask_b=MemoryBank(dim=embed_dim, capacity=args.symbolic_mem_capacity),
                        mem_img_g=MemoryBank(dim=embed_dim, capacity=args.symbolic_mem_capacity),
                        mem_img_b=MemoryBank(dim=embed_dim, capacity=args.symbolic_mem_capacity),
                        w_fused=args.multimodal_w_fused,
                        w_mask=args.multimodal_w_mask,
                        w_image=args.multimodal_w_image,
                        min_size=args.symbolic_mem_min,
                        min_cos_sim=args.symbolic_outlier_cos,
                    )
                else:
                    sym_mm3 = MultiModalSymbolicAlignmentTriple(
                        encoder=mm,
                        ema_fused_g=EMAStats(dim=embed_dim, momentum=args.symbolic_ema_momentum),
                        ema_fused_b=EMAStats(dim=embed_dim, momentum=args.symbolic_ema_momentum),
                        ema_mask_g=EMAStats(dim=embed_dim, momentum=args.symbolic_ema_momentum),
                        ema_mask_b=EMAStats(dim=embed_dim, momentum=args.symbolic_ema_momentum),
                        ema_img_g=EMAStats(dim=embed_dim, momentum=args.symbolic_ema_momentum),
                        ema_img_b=EMAStats(dim=embed_dim, momentum=args.symbolic_ema_momentum),
                        w_fused=args.multimodal_w_fused,
                        w_mask=args.multimodal_w_mask,
                        w_image=args.multimodal_w_image,
                    )
                print(
                    f"[Symbolic:multimodal3] enabled ckpt={args.multimodal_ckpt} prior={args.symbolic_prior_type} "
                    f"w(fused,mask,image)=({args.multimodal_w_fused},{args.multimodal_w_mask},{args.multimodal_w_image}) "
                    f"lambda={args.symbolic_lambda} warmup={args.symbolic_warmup_steps} conf_thr={args.symbolic_conf_thr}"
                )
            else:
                if args.symbolic_prior_type == "memory":
                    sym_mm = MultiModalSymbolicAlignmentMemory(
                        encoder=mm,
                        mem_global=MemoryBank(dim=embed_dim, capacity=args.symbolic_mem_capacity),
                        mem_boundary=MemoryBank(dim=embed_dim, capacity=args.symbolic_mem_capacity),
                        output=args.multimodal_output,
                        min_size=args.symbolic_mem_min,
                        min_cos_sim=args.symbolic_outlier_cos,
                    )
                else:
                    sym_mm = MultiModalSymbolicAlignment(
                        encoder=mm,
                        ema_global=EMAStats(dim=embed_dim, momentum=args.symbolic_ema_momentum),
                        ema_boundary=EMAStats(dim=embed_dim, momentum=args.symbolic_ema_momentum),
                        output=args.multimodal_output,
                    )
                print(
                    f"[Symbolic:multimodal] enabled ckpt={args.multimodal_ckpt} prior={args.symbolic_prior_type} output={args.multimodal_output} lambda={args.symbolic_lambda} "
                    f"warmup={args.symbolic_warmup_steps} conf_thr={args.symbolic_conf_thr}"
                )

    model.train()
    step = 0
    used_frac_ema = None
    it = iter(adapt_loader)
    while step < args.steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(adapt_loader)
            batch = next(it)

        if isinstance(batch, dict):
            xw = batch["weak"]
            xs = batch["strong"]
        else:
            xw = batch.weak
            xs = batch.strong

        xw = xw.to(device)
        xs = xs.to(device)

        optimizer.zero_grad(set_to_none=True)
        lw = model(xw)
        with torch.no_grad():
            lw_t = teacher(xw)

        if args.method == "entropy":
            loss = entropy_loss_from_logits(lw) + args.teacher_kl_weight * kl_to_teacher(lw, lw_t)
        elif args.method == "consistency":
            ls = model(xs)
            # Student strong should match a fixed teacher prediction from weak.
            loss = consistency_loss(lw_t, ls) + args.teacher_kl_weight * kl_to_teacher(lw, lw_t)
        elif args.method == "selftrain":
            ls = model(xs)
            # Use teacher weak logits to generate pseudo-labels, and stabilize student via KL.
            loss, used_frac = self_training_loss(ls, lw_t, conf_thr=args.selftrain_conf_thr)
            loss = loss + args.teacher_kl_weight * kl_to_teacher(lw, lw_t)
            used_frac_ema = used_frac if used_frac_ema is None else 0.9 * used_frac_ema + 0.1 * used_frac
        elif args.method == "tent":
            # TENT typically uses entropy; params already restricted to norm affine
            loss = entropy_loss_from_logits(lw) + args.teacher_kl_weight * kl_to_teacher(lw, lw_t)
        else:
            raise ValueError(f"Unknown method: {args.method}")

        if args.fg_prior_weight and args.fg_prior_weight > 0:
            loss = loss + float(args.fg_prior_weight) * fg_fraction_prior(lw, lw_t)

        if (sym_mask is not None or sym_mm is not None or sym_mm3 is not None) and step >= args.symbolic_warmup_steps:
            p = torch.sigmoid(lw)
            if sym_mm3 is not None:
                fused, mask, image = sym_mm3.compute_all_embeddings(xw, p)
            elif sym_mm is not None:
                z_g, z_b = sym_mm.compute_embeddings(xw, p)
            else:
                z_g, z_b = sym_mask.compute_embeddings(p)  # type: ignore[union-attr]

            # Guarded admission gate for updating priors/memory.
            ok, _ok_stats = symbolic_ok_mask(
                p,
                conf_thr=args.symbolic_conf_thr,
                min_fg=args.symbolic_min_fg,
                max_fg=args.symbolic_max_fg,
                max_components=args.symbolic_max_components,
            )

            # initialize priors on first usable batch
            if sym_mm3 is not None:
                _accepted = sym_mm3.update_priors(fused, mask, image, ok)
                priors_ready = sym_mm3.priors_ready()
            elif sym_mm is not None:
                if isinstance(sym_mm, MultiModalSymbolicAlignmentMemory):
                    _accepted = sym_mm.update_priors(z_g, z_b, ok)
                    priors_ready = sym_mm.priors_ready()
                else:
                    sym_mm.update_priors(z_g, z_b, ok)
                    priors_ready = sym_mm.ema_global.mean is not None
            else:
                sym_mask.update_priors(z_g, z_b, ok)  # type: ignore[union-attr]
                priors_ready = sym_mask.ema_global.mean is not None  # type: ignore[union-attr]

            # only apply loss once priors exist
            if priors_ready:
                if sym_mm3 is not None:
                    loss = loss + float(args.symbolic_lambda) * sym_mm3.loss(fused, mask, image)
                elif sym_mm is not None:
                    loss = loss + float(args.symbolic_lambda) * sym_mm.loss(z_g, z_b)
                else:
                    loss = loss + float(args.symbolic_lambda) * sym_mask.loss(z_g, z_b)  # type: ignore[union-attr]

        loss.backward()
        optimizer.step()

        step += 1
        if step % 50 == 0:
            msg = f"[adapt {args.method}] step={step}/{args.steps} loss={loss.item():.5f}"
            if used_frac_ema is not None:
                msg += f" used_px~{used_frac_ema:.3f}"
            print(msg)

    metrics = evaluate(model, eval_loader, device=device, dice_thr=args.dice_thr, boundary_tol_px=args.boundary_tol_px)
    out = {
        "method": args.method,
        "corruption": args.corruption,
        "severity": args.severity,
        "num_ops": args.num_ops if args.corruption == "mixed" else "",
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "adapter": args.adapter,
        "peft_wrapped": peft_info.get("wrapped", 0),
        "dice": metrics["dice"],
        "iou": metrics["iou"],
        "boundary_f": metrics["boundary_f"],
        "hd95": metrics["hd95"],
        "empty_rate": metrics["empty_rate"],
        "full_rate": metrics["full_rate"],
        "n": int(metrics["n"]),
    }

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_csv.exists()
    with out_csv.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(out)

    print(f"[OK] Appended results to {out_csv}")
    print(out)


if __name__ == "__main__":
    main()
