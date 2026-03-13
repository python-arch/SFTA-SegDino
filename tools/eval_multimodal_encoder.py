#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from segdino.symalign.encoder import SmallMaskEncoder
    from segdino.symalign.image_mask_pairs import ImageMaskPairDataset, collate_image_mask_pairs
    from segdino.symalign.multimodal_encoder import MultiModalConfig, MultiModalSymbolicEncoder
    from segdino.symalign.pair_augment import PairAugmentConfig, augment_pair
except ModuleNotFoundError:
    from symalign.encoder import SmallMaskEncoder
    from symalign.image_mask_pairs import ImageMaskPairDataset, collate_image_mask_pairs
    from symalign.multimodal_encoder import MultiModalConfig, MultiModalSymbolicEncoder
    from symalign.pair_augment import PairAugmentConfig, augment_pair


def _load_multimodal_encoder(ckpt_path: str, device: str) -> MultiModalSymbolicEncoder:
    try:
        obj = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        obj = torch.load(ckpt_path, map_location=device)

    if not (isinstance(obj, dict) and "state_dict" in obj):
        raise ValueError("Expected multimodal checkpoint to be a dict with a 'state_dict' key.")

    state = obj["state_dict"]
    cfg_d = obj.get("cfg", {}) if isinstance(obj.get("cfg", {}), dict) else {}
    embed_dim = int(cfg_d.get("embed_dim", 64))
    mask_width = int(cfg_d.get("mask_width", 32))
    image_width = int(cfg_d.get("image_width", 32))
    fusion = str(cfg_d.get("fusion", "mlp"))
    image_encoder = str(cfg_d.get("image_encoder", "small_cnn"))
    image_weights = str(cfg_d.get("image_weights", "none"))
    image_pool = str(cfg_d.get("image_pool", "features"))
    pool_dilate_px = int(cfg_d.get("pool_dilate_px", 0))
    use_imagenet_norm = bool(cfg_d.get("use_imagenet_norm", True))
    gate_hidden = int(cfg_d.get("gate_hidden", 64))

    mask_enc = SmallMaskEncoder(in_ch=2, embed_dim=embed_dim, width=mask_width)
    mm_cfg = MultiModalConfig(
        embed_dim=embed_dim,
        mask_width=mask_width,
        image_width=image_width,
        fusion=fusion,
        image_encoder=image_encoder,
        image_weights=image_weights,
        image_pool=image_pool,
        pool_dilate_px=pool_dilate_px,
        use_imagenet_norm=use_imagenet_norm,
        gate_hidden=gate_hidden,
    )
    model = MultiModalSymbolicEncoder(mask_encoder=mask_enc, cfg=mm_cfg)
    model.load_state_dict(state, strict=False)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # a,b: (N,D) normalized
    return a @ b.t()


@torch.no_grad()
def evaluate_invariance(
    model: MultiModalSymbolicEncoder,
    loader: DataLoader,
    aug_cfg: PairAugmentConfig,
    seed: int,
    max_items: int,
    device: str,
) -> Dict[str, float]:
    """
    Build two augmented views for each sample, compute embeddings for each stream, and evaluate:
    - top1 retrieval: for each item in view1, does its nearest neighbor in view2 share the same id?
    - positive vs negative cosine similarity
    """
    ids: List[str] = []
    z_f1: List[torch.Tensor] = []
    z_m1: List[torch.Tensor] = []
    z_i1: List[torch.Tensor] = []
    z_f2: List[torch.Tensor] = []
    z_m2: List[torch.Tensor] = []
    z_i2: List[torch.Tensor] = []

    n_seen = 0
    for images, mask_pairs, batch_ids in loader:
        if n_seen >= max_items:
            break
        b = images.size(0)
        take = min(b, max_items - n_seen)
        images = images[:take].to(device)
        mask_pairs = mask_pairs[:take].to(device)
        batch_ids = batch_ids[:take]

        rng = random.Random(seed + n_seen)
        imgs1, masks1, imgs2, masks2 = [], [], [], []
        for i in range(take):
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

        f1, m1z, i1 = model(img1, m1)
        f2, m2z, i2 = model(img2, m2)

        ids.extend(batch_ids)
        z_f1.append(f1.cpu())
        z_m1.append(m1z.cpu())
        z_i1.append(i1.cpu())
        z_f2.append(f2.cpu())
        z_m2.append(m2z.cpu())
        z_i2.append(i2.cpu())
        n_seen += take

    if n_seen == 0:
        raise RuntimeError("No samples evaluated (check dataset/split).")

    ids_arr = np.array(ids)
    zf1 = torch.cat(z_f1, dim=0)
    zm1 = torch.cat(z_m1, dim=0)
    zi1 = torch.cat(z_i1, dim=0)
    zf2 = torch.cat(z_f2, dim=0)
    zm2 = torch.cat(z_m2, dim=0)
    zi2 = torch.cat(z_i2, dim=0)

    def score_stream(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float, float]:
        sim = _cosine_sim(a, b)  # (N,N)
        nn = torch.argmax(sim, dim=1).cpu().numpy()
        top1 = float((ids_arr[nn] == ids_arr).mean())
        pos = float(torch.diag(sim).mean().item())
        # sample negatives: off-diagonal mean (cheap proxy)
        neg = float((sim.sum() - torch.diag(sim).sum()) / (sim.numel() - sim.shape[0])).item()
        gap = pos - neg
        return top1, pos, gap

    f_top1, f_pos, f_gap = score_stream(zf1, zf2)
    m_top1, m_pos, m_gap = score_stream(zm1, zm2)
    i_top1, i_pos, i_gap = score_stream(zi1, zi2)

    return {
        "n": float(n_seen),
        "fused_top1": f_top1,
        "mask_top1": m_top1,
        "image_top1": i_top1,
        "fused_pos_cos": f_pos,
        "mask_pos_cos": m_pos,
        "image_pos_cos": i_pos,
        "fused_posneg_gap": f_gap,
        "mask_posneg_gap": m_gap,
        "image_posneg_gap": i_gap,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate multi-modal symbolic encoder invariance via two-view retrieval.")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--img_dir_name", type=str, default="images")
    parser.add_argument("--mask_dir_name", type=str, default="masks")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to multimodal encoder checkpoint (encoder_final.pth).")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_items", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--out_h", type=int, default=256)
    parser.add_argument("--out_w", type=int, default=256)
    parser.add_argument("--boundary_width", type=int, default=2)
    parser.add_argument("--max_rotate_deg", type=float, default=20.0)
    parser.add_argument("--hflip_p", type=float, default=0.5)
    parser.add_argument("--vflip_p", type=float, default=0.0)
    parser.add_argument("--color_jitter_strength", type=float, default=0.2)
    parser.add_argument("--gaussian_noise_std", type=float, default=0.02)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_multimodal_encoder(args.ckpt, device=device)

    ds = ImageMaskPairDataset(
        dataset_root=args.dataset_root,
        split=args.split,
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
        drop_last=False,
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

    metrics = evaluate_invariance(
        model=model,
        loader=loader,
        aug_cfg=aug_cfg,
        seed=args.seed,
        max_items=args.max_items,
        device=device,
    )

    print("[OK] Multi-modal encoder invariance / retrieval")
    for k in sorted(metrics.keys()):
        v = metrics[k]
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
