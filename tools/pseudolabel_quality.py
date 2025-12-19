#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from corruption_transform import CorruptionTransform
from corruptions import CorruptionSpec, MixedCorruptionSpec
from data import ManifestSegmentationDataset, ResizeAndNormalize, collate_seg_samples
from metrics import boundary_fscore, dice_iou_binary, hd95_binary


def load_ckpt_flex(model: torch.nn.Module, ckpt_path: str, map_location: str) -> None:
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


def binary_entropy(p: np.ndarray, eps: float = 1e-8) -> float:
    p = np.clip(p.astype(np.float64), eps, 1.0 - eps)
    h = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
    return float(h.mean())


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Per-image pseudo-label quality diagnostics under corruptions.")
    parser.add_argument("--dataset_root", type=str, default="./segdata/kvasir")
    parser.add_argument("--manifest", type=str, default="./splits/kvasir_target_holdout.txt")
    parser.add_argument("--img_dir_name", type=str, default="images")
    parser.add_argument("--mask_dir_name", type=str, default="masks")
    parser.add_argument("--input_h", type=int, default=256)
    parser.add_argument("--input_w", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--family", type=str, default="mixed", choices=["blur", "noise", "jpeg", "illumination", "mixed"])
    parser.add_argument("--severity", type=int, default=4)
    parser.add_argument("--num_ops", type=int, default=4, help="For mixed only (1..4).")
    parser.add_argument("--corruption_id", type=str, default="v1")

    parser.add_argument("--dice_thr", type=float, default=0.5)
    parser.add_argument("--boundary_tol_px", type=int, default=2)
    parser.add_argument("--conf_thr", type=float, default=0.9, help="Confidence threshold for pseudo-label stats.")

    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--dino_ckpt", type=str, required=True)
    parser.add_argument("--dino_size", type=str, default="s", choices=["b", "s"])
    parser.add_argument("--repo_dir", type=str, default="./dinov3")

    parser.add_argument("--out_csv", type=str, default="./runs/pseudolabel_quality.csv")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dino_size == "b":
        backbone = torch.hub.load(args.repo_dir, "dinov3_vitb16", source="local", weights=args.dino_ckpt)
        encoder_size = "base"
    else:
        backbone = torch.hub.load(args.repo_dir, "dinov3_vits16", source="local", weights=args.dino_ckpt)
        encoder_size = "small"

    from dpt import DPT

    model = DPT(encoder_size=encoder_size, nclass=1, backbone=backbone).to(device)
    load_ckpt_flex(model, args.ckpt, map_location=device)
    model.eval()

    transform = ResizeAndNormalize(size=(args.input_h, args.input_w))
    if args.family == "mixed":
        spec = MixedCorruptionSpec(severity=args.severity, num_ops=args.num_ops, corruption_id=args.corruption_id)
    else:
        spec = CorruptionSpec(family=args.family, severity=args.severity, corruption_id=args.corruption_id)
    pre = CorruptionTransform(spec=spec)

    ds = ManifestSegmentationDataset(
        dataset_root=args.dataset_root,
        split="test",
        manifest_path=args.manifest,
        img_dir_name=args.img_dir_name,
        mask_dir_name=args.mask_dir_name,
        return_mask=True,
        transform=transform,
        image_pre_transform=pre,
        strict_pair=True,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=collate_seg_samples,
    )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for batch in loader:
        inputs = batch["image"]
        targets = batch["mask"]
        ids = batch["meta"]["id"]

        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs)
        probs = torch.sigmoid(logits)
        preds = (probs > args.dice_thr).float()

        b = inputs.size(0)
        for i in range(b):
            image_id = ids[i] if isinstance(ids, list) else str(ids)
            p = probs[i, 0].detach().cpu().numpy()
            pr = (preds[i, 0].detach().cpu().numpy() > 0.5)
            gt = (targets[i, 0].detach().cpu().numpy() > 0.5)

            dice, iou = dice_iou_binary(pr, gt)
            bf = boundary_fscore(pr, gt, tolerance_px=args.boundary_tol_px)
            hd = hd95_binary(pr, gt)

            ent = binary_entropy(p)
            mean_p = float(p.mean())
            fg_frac = float(pr.mean())
            gt_fg_frac = float(gt.mean())

            conf = np.maximum(p, 1.0 - p)
            frac_conf = float((conf >= args.conf_thr).mean())
            frac_conf_fg = float(((conf >= args.conf_thr) & pr).mean())

            # how much confident region aligns with GT foreground (a proxy for pseudo-label usability)
            conf_fg = (conf >= args.conf_thr) & pr
            conf_fg_count = int(conf_fg.sum())
            conf_fg_precision = float(gt[conf_fg].mean()) if conf_fg_count > 0 else 0.0

            rows.append(
                {
                    "id": image_id,
                    "family": args.family,
                    "severity": args.severity,
                    "num_ops": args.num_ops if args.family == "mixed" else "",
                    "dice": dice,
                    "iou": iou,
                    "boundary_f": bf,
                    "hd95": hd,
                    "entropy": ent,
                    "mean_prob": mean_p,
                    "pred_fg_frac": fg_frac,
                    "gt_fg_frac": gt_fg_frac,
                    "conf_thr": args.conf_thr,
                    "frac_conf": frac_conf,
                    "frac_conf_fg": frac_conf_fg,
                    "conf_fg_precision": conf_fg_precision,
                }
            )

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"[OK] Wrote {out_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
