#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

try:
    from segdino.symalign.encoder import SmallMaskEncoder
    from segdino.symalign.image_mask_pairs import ImageMaskPairDataset, collate_image_mask_pairs
    from segdino.symalign.multimodal_encoder import MultiModalConfig, MultiModalSymbolicEncoder
    from segdino.symalign.multimodal_loss import MultiModalContrastiveLoss, MultiModalLossWeights
    from segdino.symalign.pair_augment import PairAugmentConfig, augment_pair
except ModuleNotFoundError:
    from symalign.encoder import SmallMaskEncoder
    from symalign.image_mask_pairs import ImageMaskPairDataset, collate_image_mask_pairs
    from symalign.multimodal_encoder import MultiModalConfig, MultiModalSymbolicEncoder
    from symalign.multimodal_loss import MultiModalContrastiveLoss, MultiModalLossWeights
    from symalign.pair_augment import PairAugmentConfig, augment_pair


def _load_mask_encoder(mask_ckpt: str, device: str) -> tuple[SmallMaskEncoder, Dict]:
    try:
        obj = torch.load(mask_ckpt, map_location=device, weights_only=True)
    except TypeError:
        obj = torch.load(mask_ckpt, map_location=device)

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
    return enc, {"embed_dim": embed_dim, "width": width, "raw_args": enc_args}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train multi-modal symbolic descriptor encoder (mask + image).")
    parser.add_argument("--dataset_root", type=str, required=True, help="Dataset root containing train/images and train/masks.")
    parser.add_argument("--img_dir_name", type=str, default="images")
    parser.add_argument("--mask_dir_name", type=str, default="masks")
    parser.add_argument("--mask_encoder_ckpt", type=str, required=True, help="Path to trained mask-only encoder (encoder_final.pth).")
    parser.add_argument("--out_dir", type=str, default="./runs/symalign_multimodal_encoder")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.1)

    parser.add_argument("--out_h", type=int, default=256)
    parser.add_argument("--out_w", type=int, default=256)
    parser.add_argument("--boundary_width", type=int, default=2)

    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument(
        "--image_encoder",
        type=str,
        default="small_cnn",
        choices=["small_cnn", "resnet18", "resnet34", "mobilenet_v3_small"],
        help="Image encoder backbone. Prefer small_cnn if you want zero external dependencies.",
    )
    parser.add_argument("--image_width", type=int, default=32)
    parser.add_argument(
        "--image_weights",
        type=str,
        default="none",
        choices=["none", "imagenet"],
        help="If using a torchvision backbone, optionally initialize with ImageNet weights (may require local availability).",
    )
    parser.add_argument("--fusion", type=str, default="mlp", choices=["mlp", "attn"])

    parser.add_argument("--w_mask", type=float, default=1.0)
    parser.add_argument("--w_image", type=float, default=1.0)
    parser.add_argument("--w_cross", type=float, default=0.5)
    parser.add_argument("--w_fused", type=float, default=1.0)

    parser.add_argument("--max_rotate_deg", type=float, default=20.0)
    parser.add_argument("--hflip_p", type=float, default=0.5)
    parser.add_argument("--vflip_p", type=float, default=0.0)
    parser.add_argument("--color_jitter_strength", type=float, default=0.2)
    parser.add_argument("--gaussian_noise_std", type=float, default=0.02)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    mask_enc, mask_meta = _load_mask_encoder(args.mask_encoder_ckpt, device=device)
    if args.embed_dim != int(mask_meta["embed_dim"]):
        print(f"[Warn] --embed_dim={args.embed_dim} != mask_encoder.embed_dim={mask_meta['embed_dim']}; using {args.embed_dim} for multimodal head.")

    cfg = MultiModalConfig(
        embed_dim=args.embed_dim,
        mask_width=int(mask_meta["width"]),
        image_encoder=args.image_encoder,
        image_width=args.image_width,
        image_weights=args.image_weights,
        fusion=args.fusion,
    )
    model = MultiModalSymbolicEncoder(mask_encoder=mask_enc, cfg=cfg).to(device)

    # Only train image encoder + fusion.
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    ds = ImageMaskPairDataset(
        dataset_root=args.dataset_root,
        split="train",
        img_dir_name=args.img_dir_name,
        mask_dir_name=args.mask_dir_name,
        out_size=(args.out_h, args.out_w),
        boundary_width=args.boundary_width,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=collate_image_mask_pairs,
    )

    aug_cfg = PairAugmentConfig(
        out_size=(args.out_h, args.out_w),
        max_rotate_deg=args.max_rotate_deg,
        hflip_p=args.hflip_p,
        vflip_p=args.vflip_p,
        color_jitter_strength=args.color_jitter_strength,
        gaussian_noise_std=args.gaussian_noise_std,
    )
    loss_fn = MultiModalContrastiveLoss(
        temperature=args.temperature,
        w=MultiModalLossWeights(mask=args.w_mask, image=args.w_image, cross=args.w_cross, fused=args.w_fused),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_steps = 0

        for images, mask_pairs, _ids in loader:
            images = images.to(device)
            mask_pairs = mask_pairs.to(device)

            # Deterministic augmentation sampling (python RNG); torch RNG is seeded once per epoch.
            rng = random.Random(args.seed + epoch * 100000 + step)
            torch.manual_seed(args.seed + epoch)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed + epoch)

            # Build two augmented views with shared geometry for (image,mask_pair)
            imgs1, masks1 = [], []
            imgs2, masks2 = [], []
            b = images.size(0)
            for i in range(b):
                img1, m1 = augment_pair(images[i], mask_pairs[i], aug_cfg, rng)
                img2, m2 = augment_pair(images[i], mask_pairs[i], aug_cfg, rng)
                imgs1.append(img1)
                masks1.append(m1)
                imgs2.append(img2)
                masks2.append(m2)

            img1 = torch.stack(imgs1, dim=0)
            m1 = torch.stack(masks1, dim=0)
            img2 = torch.stack(imgs2, dim=0)
            m2 = torch.stack(masks2, dim=0)

            zf1, zm1, zi1 = model(img1, m1)
            zf2, zm2, zi2 = model(img2, m2)
            loss, logs = loss_fn(zf1, zm1, zi1, zf2, zm2, zi2)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            epoch_loss += float(loss.detach().cpu().item())
            n_steps += 1
            step += 1

        mean_loss = epoch_loss / max(1, n_steps)
        print(f"[E_mm] epoch={epoch}/{args.epochs} loss={mean_loss:.4f}")

        if epoch % 5 == 0 or epoch == args.epochs:
            ckpt_path = out_dir / f"encoder_ep{epoch:03d}.pth"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "args": vars(args),
                    "mask_encoder_ckpt": args.mask_encoder_ckpt,
                    "mask_encoder_meta": mask_meta,
                    "cfg": cfg.__dict__,
                },
                ckpt_path,
            )
            print(f"[Save] {ckpt_path}")

    final_path = out_dir / "encoder_final.pth"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "args": vars(args),
            "mask_encoder_ckpt": args.mask_encoder_ckpt,
            "mask_encoder_meta": mask_meta,
            "cfg": cfg.__dict__,
        },
        final_path,
    )
    print(f"[OK] Wrote {final_path}")


if __name__ == "__main__":
    main()
