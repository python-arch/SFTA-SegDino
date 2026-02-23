#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


DATASET_META: Dict[str, Dict[str, int | str]] = {
    "pneumoniamnist": {"display_name": "PneumoniaMNIST", "num_classes": 2, "channels": 1},
    "dermamnist": {"display_name": "DermaMNIST", "num_classes": 7, "channels": 3},
    "bloodmnist": {"display_name": "BloodMNIST", "num_classes": 8, "channels": 3},
    "organmnist": {"display_name": "OrganMNIST", "num_classes": 11, "channels": 1},
    "retinamnist": {"display_name": "RetinaMNIST", "num_classes": 5, "channels": 3},
}


def _to_hwc_uint8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img.astype(np.uint8, copy=False)
    if img.ndim == 3 and img.shape[-1] in (1, 3):
        return img.astype(np.uint8, copy=False)
    if img.ndim == 3 and img.shape[0] in (1, 3):
        return np.transpose(img, (1, 2, 0)).astype(np.uint8, copy=False)
    raise ValueError(f"Unsupported image shape: {img.shape}")


def _read_split(npz: np.lib.npyio.NpzFile, split: str) -> Tuple[np.ndarray, np.ndarray]:
    images_key = f"{split}_images"
    labels_key = f"{split}_labels"
    # tolerate typo variants if present in custom exports
    if images_key not in npz and f"{split}_imag" in npz:
        images_key = f"{split}_imag"
    if images_key not in npz:
        raise KeyError(f"Missing key '{split}_images' in npz (keys: {npz.files})")
    if labels_key not in npz:
        raise KeyError(f"Missing key '{split}_labels' in npz (keys: {npz.files})")
    return npz[images_key], npz[labels_key]


def write_split(
    images: np.ndarray,
    labels: np.ndarray,
    split_dir: Path,
) -> int:
    from PIL import Image

    img_dir = split_dir / "images"
    mask_dir = split_dir / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    labels = labels.reshape(-1)
    count = 0
    for i in range(images.shape[0]):
        img = _to_hwc_uint8(images[i])
        h, w = img.shape[:2]
        cls_id = int(labels[i])
        mask = np.full((h, w), cls_id, dtype=np.uint8)
        fname = f"{i:06d}.png"
        Image.fromarray(img).save(img_dir / fname)
        Image.fromarray(mask).save(mask_dir / fname)
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert MedMNIST .npz files into SegDino folder layout (train/test with images+masks)."
    )
    parser.add_argument("--npz_dir", type=str, default="./medmnist_npz")
    parser.add_argument("--out_dir", type=str, default="./segdata")
    parser.add_argument(
        "--datasets",
        type=str,
        default="pneumoniamnist,dermamnist,bloodmnist,organmnist,retinamnist",
        help="Comma-separated dataset tokens.",
    )
    parser.add_argument(
        "--include_val_in_train",
        action="store_true",
        help="If set, merge val split into train split output.",
    )
    args = parser.parse_args()

    npz_dir = Path(args.npz_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for dataset in [x.strip().lower() for x in args.datasets.split(",") if x.strip()]:
        if dataset not in DATASET_META:
            raise ValueError(f"Unknown dataset token '{dataset}'. Allowed: {', '.join(DATASET_META.keys())}")

        npz_path = npz_dir / f"{dataset}.npz"
        if not npz_path.is_file():
            raise FileNotFoundError(f"Missing file: {npz_path}")

        with np.load(npz_path) as npz:
            train_images, train_labels = _read_split(npz, "train")
            val_images, val_labels = _read_split(npz, "val")
            test_images, test_labels = _read_split(npz, "test")

        if args.include_val_in_train:
            train_images = np.concatenate([train_images, val_images], axis=0)
            train_labels = np.concatenate([train_labels, val_labels], axis=0)

        ds_root = out_dir / dataset
        train_count = write_split(train_images, train_labels, ds_root / "train")
        test_count = write_split(test_images, test_labels, ds_root / "test")

        summary[dataset] = {
            **DATASET_META[dataset],
            "npz_file": str(npz_path),
            "output_root": str(ds_root),
            "train_count": int(train_count),
            "test_count": int(test_count),
            "included_val_in_train": bool(args.include_val_in_train),
        }
        print(f"[ok] {dataset}: train={train_count}, test={test_count}, out={ds_root}")

    summary_path = out_dir / "medmnist_prepared_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[ok] Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
