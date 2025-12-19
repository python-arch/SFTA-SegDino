#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from corruption_transform import CorruptionTransform
from corruptions import CorruptionSpec, MixedCorruptionSpec
from data import ManifestSegmentationDataset, ResizeAndNormalize, collate_seg_samples
from metrics import RunningStats, boundary_fscore, dice_iou_binary, hd95_binary


def load_ckpt_flex(model: torch.nn.Module, ckpt_path: str, map_location: str) -> None:
    # Prefer `weights_only=True` when supported to avoid unpickling arbitrary objects.
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


@torch.no_grad()
def eval_one(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    dice_thr: float,
    boundary_tol_px: int,
) -> Dict[str, float]:
    model.eval()
    stats = RunningStats()
    for batch in loader:
        # batch is SegSample (dataclass) collated -> dict-like? default collate will make a dict of lists/tensors
        # We avoid relying on dataclass collation by using attribute access fallback.
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
        if targets is None:
            raise RuntimeError("Evaluation requires masks; dataset was created with return_mask=False.")
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
            empty_pred = not pr.any()
            full_pred = bool(pr.all())
            stats.update(dice=dice, iou=iou, bf=bf, hd95=hd, empty_pred=empty_pred, full_pred=full_pred)

    return stats.means()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained segmentation checkpoint across corruption severities.")
    parser.add_argument("--dataset_root", type=str, default="./segdata/kvasir")
    parser.add_argument("--manifest", type=str, default="./splits/kvasir_target_holdout.txt")
    parser.add_argument("--img_dir_name", type=str, default="images")
    parser.add_argument("--mask_dir_name", type=str, default="masks")
    parser.add_argument("--input_h", type=int, default=256)
    parser.add_argument("--input_w", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--dice_thr", type=float, default=0.5)
    parser.add_argument("--boundary_tol_px", type=int, default=2)

    parser.add_argument("--family", type=str, default="blur", choices=["blur", "noise", "jpeg", "illumination", "mixed"])

    parser.add_argument("--num_ops", type=int, default=2, help="Number of ops for mixed corruption (1..4).")

    parser.add_argument("--max_severity", type=int, default=4, help="Max severity (inclusive). Supports 0..8.")
    parser.add_argument("--corruption_id", type=str, default="v1")

    parser.add_argument("--ckpt", type=str, required=True, help="Trained segmentation checkpoint (.pth).")
    parser.add_argument("--dino_ckpt", type=str, required=True, help="DINOv3 pretrained weights (.pth).")
    parser.add_argument("--dino_size", type=str, default="s", choices=["b", "s"])
    parser.add_argument("--repo_dir", type=str, default="./dinov3", help="Local DINOv3 torch.hub repo (hubconf.py).")

    parser.add_argument("--out_csv", type=str, default="./runs/source_only_corruption_curve.csv")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Backbone
    if args.dino_size == "b":
        backbone = torch.hub.load(args.repo_dir, "dinov3_vitb16", source="local", weights=args.dino_ckpt)
        encoder_size = "base"
    else:
        backbone = torch.hub.load(args.repo_dir, "dinov3_vits16", source="local", weights=args.dino_ckpt)
        encoder_size = "small"

    from dpt import DPT

    model = DPT(encoder_size=encoder_size, nclass=1, backbone=backbone)
    model = model.to(device)
    load_ckpt_flex(model, args.ckpt, map_location=device)

    transform = ResizeAndNormalize(size=(args.input_h, args.input_w))

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for s in range(0, args.max_severity + 1):
        if args.family == "mixed":
            spec = MixedCorruptionSpec(
                severity=s,
                num_ops=args.num_ops,
                corruption_id=args.corruption_id,
            )
        else:
            spec = CorruptionSpec(
                family=args.family,
                severity=s,
                corruption_id=args.corruption_id,
            )

        pre = CorruptionTransform(spec=spec)


        ds = ManifestSegmentationDataset(
            dataset_root=args.dataset_root,
            split="test",  # the manifest lines already include `test/images/...`
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
            pin_memory=(device == "cuda"),
            drop_last=False,
            collate_fn=collate_seg_samples,
        )


        m = eval_one(model, loader, device=device, dice_thr=args.dice_thr, boundary_tol_px=args.boundary_tol_px)
        row = {
            "family": args.family,
            "severity": s,
            "dice": m["dice"],
            "iou": m["iou"],
            "boundary_f": m["boundary_f"],
            "hd95": m["hd95"],
            "empty_rate": m["empty_rate"],
            "full_rate": m["full_rate"],
            "n": int(m["n"]),
        }
        rows.append(row)
        print(
            f"[{args.family} S{s}] dice={row['dice']:.4f} iou={row['iou']:.4f} "
            f"bf={row['boundary_f']:.4f} hd95={row['hd95'] if row['hd95'] != float('inf') else 'inf'} "
            f"empty={row['empty_rate']:.3f} full={row['full_rate']:.3f} n={row['n']}"
        )

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["family", "severity", "dice", "iou", "boundary_f", "hd95", "empty_rate", "full_rate", "n"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"[OK] Wrote {out_csv}")


if __name__ == "__main__":
    main()
