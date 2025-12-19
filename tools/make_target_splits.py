#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Tuple


SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _iter_images(images_dir: Path) -> List[Path]:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")
    paths: List[Path] = []
    for p in images_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            paths.append(p)
    paths.sort()
    if not paths:
        raise RuntimeError(f"No images found under: {images_dir}")
    return paths


def _mask_path_for_image(img_rel: Path, img_dir_name: str, mask_dir_name: str) -> Path:
    parts = list(img_rel.parts)
    try:
        i = parts.index(img_dir_name)
    except ValueError as e:
        raise RuntimeError(f"Expected '{img_dir_name}' in path: {img_rel}") from e
    parts[i] = mask_dir_name
    return Path(*parts).with_suffix(img_rel.suffix)


def _readable_manifest_lines(paths_rel: Iterable[Path]) -> str:
    return "\n".join(p.as_posix() for p in paths_rel) + "\n"


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SplitMetadata:
    dataset_root: str
    base_split: str
    img_dir_name: str
    mask_dir_name: str
    seed: int
    holdout_ratio: float
    num_total: int
    num_adapt: int
    num_holdout: int
    manifest_adapt: str
    manifest_holdout: str
    sha256_adapt: str
    sha256_holdout: str
    created_at_utc: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic target_adapt/holdout manifests from an existing split.")
    parser.add_argument("--dataset_root", type=str, default="./segdata/kvasir", help="Dataset root containing e.g. test/images.")
    parser.add_argument("--base_split", type=str, default="test", help="Split folder to draw from (default: test).")
    parser.add_argument("--img_dir_name", type=str, default="images")
    parser.add_argument("--mask_dir_name", type=str, default="masks")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout_ratio", type=float, default=0.2)
    parser.add_argument("--out_dir", type=str, default="./splits")
    parser.add_argument("--prefix", type=str, default="kvasir", help="Filename prefix for manifests/metadata.")
    args = parser.parse_args()

    if not (0.0 < args.holdout_ratio < 1.0):
        raise ValueError("--holdout_ratio must be in (0,1)")

    dataset_root = Path(args.dataset_root).resolve()
    images_dir = dataset_root / args.base_split / args.img_dir_name
    masks_dir = dataset_root / args.base_split / args.mask_dir_name
    if not masks_dir.is_dir():
        raise FileNotFoundError(f"Masks dir not found: {masks_dir}")

    img_paths = _iter_images(images_dir)
    img_rel = [p.relative_to(dataset_root) for p in img_paths]

    missing_masks: List[Path] = []
    for r in img_rel:
        m_rel = _mask_path_for_image(r, args.img_dir_name, args.mask_dir_name)
        if not (dataset_root / m_rel).is_file():
            missing_masks.append(m_rel)
    if missing_masks:
        msg = "\n".join(m.as_posix() for m in missing_masks[:20])
        raise FileNotFoundError(f"Missing {len(missing_masks)} masks. First examples:\n{msg}")

    rng = random.Random(args.seed)
    indices = list(range(len(img_rel)))
    rng.shuffle(indices)

    n_total = len(indices)
    n_holdout = max(1, int(round(n_total * args.holdout_ratio)))
    n_adapt = n_total - n_holdout

    holdout_idx = sorted(indices[:n_holdout])
    adapt_idx = sorted(indices[n_holdout:])

    adapt_rel = [img_rel[i] for i in adapt_idx]
    holdout_rel = [img_rel[i] for i in holdout_idx]

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_adapt_name = f"{args.prefix}_target_adapt.txt"
    manifest_holdout_name = f"{args.prefix}_target_holdout.txt"
    manifest_adapt_path = out_dir / manifest_adapt_name
    manifest_holdout_path = out_dir / manifest_holdout_name

    txt_adapt = _readable_manifest_lines(adapt_rel)
    txt_holdout = _readable_manifest_lines(holdout_rel)
    manifest_adapt_path.write_text(txt_adapt, encoding="utf-8")
    manifest_holdout_path.write_text(txt_holdout, encoding="utf-8")

    meta = SplitMetadata(
        dataset_root=str(dataset_root),
        base_split=args.base_split,
        img_dir_name=args.img_dir_name,
        mask_dir_name=args.mask_dir_name,
        seed=args.seed,
        holdout_ratio=float(args.holdout_ratio),
        num_total=n_total,
        num_adapt=n_adapt,
        num_holdout=n_holdout,
        manifest_adapt=manifest_adapt_name,
        manifest_holdout=manifest_holdout_name,
        sha256_adapt=_sha256_text(txt_adapt),
        sha256_holdout=_sha256_text(txt_holdout),
        created_at_utc=datetime.now(timezone.utc).isoformat(),
    )

    meta_path = out_dir / f"{args.prefix}_target_splits.json"
    meta_path.write_text(json.dumps(asdict(meta), indent=2) + "\n", encoding="utf-8")

    print(f"[OK] Wrote {manifest_adapt_path}")
    print(f"[OK] Wrote {manifest_holdout_path}")
    print(f"[OK] Wrote {meta_path}")


if __name__ == "__main__":
    main()
