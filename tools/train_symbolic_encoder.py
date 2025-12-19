#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

try:
    from segdino.symalign.augment import MaskAugmentConfig, augment_mask_pair
    from segdino.symalign.encoder import SmallMaskEncoder, nt_xent
    from segdino.symalign.masks import MaskPairDataset
except ModuleNotFoundError:
    from symalign.augment import MaskAugmentConfig, augment_mask_pair
    from symalign.encoder import SmallMaskEncoder, nt_xent
    from symalign.masks import MaskPairDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Train learned symbolic mask descriptor encoder E_theta on source masks.")
    parser.add_argument("--dataset_root", type=str, required=True, help="Dataset root containing train/masks.")
    parser.add_argument("--mask_dir", type=str, default=None, help="Override mask dir (default: <dataset_root>/train/masks).")
    parser.add_argument("--out_dir", type=str, default="./runs/symalign_encoder")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.1)

    parser.add_argument("--out_h", type=int, default=256)
    parser.add_argument("--out_w", type=int, default=256)
    parser.add_argument("--boundary_width", type=int, default=2)
    parser.add_argument("--max_morph_radius", type=int, default=2)

    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--width", type=int, default=32)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset_root = Path(args.dataset_root)
    mask_dir = Path(args.mask_dir) if args.mask_dir else (dataset_root / "train" / "masks")
    ds = MaskPairDataset(mask_dir=mask_dir, boundary_width=args.boundary_width)

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SmallMaskEncoder(in_ch=2, embed_dim=args.embed_dim, width=args.width).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    aug_cfg = MaskAugmentConfig(
        out_size=(args.out_h, args.out_w),
        max_morph_radius=args.max_morph_radius,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in loader:
            x, _ids = batch  # x: (B,2,H,W); ids are unused for training

            b = x.size(0)
            rng = random.Random(args.seed + epoch * 100000 + b)
            x1 = torch.stack([augment_mask_pair(x[i], aug_cfg, rng) for i in range(b)], dim=0).to(device)
            x2 = torch.stack([augment_mask_pair(x[i], aug_cfg, rng) for i in range(b)], dim=0).to(device)

            z1 = model(x1)
            z2 = model(x2)
            loss = nt_xent(z1, z2, temperature=args.temperature)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        mean_loss = sum(losses) / max(1, len(losses))
        print(f"[E_theta] epoch={epoch}/{args.epochs} loss={mean_loss:.4f}")

        if epoch % 10 == 0 or epoch == args.epochs:
            ckpt_path = out_dir / f"encoder_ep{epoch:03d}.pth"
            torch.save({"state_dict": model.state_dict(), "args": vars(args)}, ckpt_path)
            print(f"[Save] {ckpt_path}")

    final_path = out_dir / "encoder_final.pth"
    torch.save({"state_dict": model.state_dict(), "args": vars(args)}, final_path)
    print(f"[OK] Wrote {final_path}")


if __name__ == "__main__":
    main()
