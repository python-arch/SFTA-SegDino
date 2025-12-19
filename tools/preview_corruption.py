#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from corruptions import CorruptionSpec, MixedCorruptionSpec, apply_corruption_bgr, apply_mixed_corruption_bgr


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview a deterministic corruption on a single image.")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--family", type=str, default="blur", choices=["none", "blur", "noise", "jpeg", "illumination", "mixed"])
    parser.add_argument("--severity", type=int, default=3, help="Severity (supports 0..8).")
    parser.add_argument("--corruption_id", type=str, default="default")
    parser.add_argument("--num_ops", type=int, default=2, help="For mixed only (1..4).")
    args = parser.parse_args()

    img_path = Path(args.image)
    out_path = Path(args.out)
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")

    image_id = img_path.name
    if args.family == "mixed":
        spec = MixedCorruptionSpec(severity=args.severity, num_ops=args.num_ops, corruption_id=args.corruption_id)
        out = apply_mixed_corruption_bgr(img, image_id=image_id, spec=spec)
    else:
        spec = CorruptionSpec(family=args.family, severity=args.severity, corruption_id=args.corruption_id)
        out = apply_corruption_bgr(img, image_id=image_id, spec=spec)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out)
    print(f"[OK] Wrote {out_path}")


if __name__ == "__main__":
    main()
