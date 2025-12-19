#!/usr/bin/env python3
from pathlib import Path
import random
import subprocess

# Paths
IMG_DIR = Path("/home/ahmedjaheen/SegDino/segdino/segdata/kvasir/test/images")
OUT_DIR = Path("/home/ahmedjaheen/SegDino/preview_corruptions")

# Config
NUM_IMAGES = 4
SEVERITIES = [0, 1, 2, 3, 4]
FAMILIES = ["blur", "noise", "jpeg", "illumination", "mixed"]
NUM_OPS_MIXED = 2
CORRUPTION_ID = "default"

def main():
    assert IMG_DIR.exists(), f"Image dir not found: {IMG_DIR}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in IMG_DIR.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    assert len(images) >= NUM_IMAGES, "Not enough images found."

    random.seed(0)  # deterministic
    picked = random.sample(images, NUM_IMAGES)

    print("Picked images:")
    for p in picked:
        print("  ", p.name)

    for img_path in picked:
        img_id = img_path.stem
        for family in FAMILIES:
            for s in SEVERITIES:
                out_dir = OUT_DIR / img_id / family
                out_dir.mkdir(parents=True, exist_ok=True)

                out_path = out_dir / f"S{s}.png"

                cmd = [
                    "python", "segdino/tools/preview_corruption.py",
                    "--image", str(img_path),
                    "--out", str(out_path),
                    "--family", family,
                    "--severity", str(s),
                    "--corruption_id", CORRUPTION_ID,
                ]

                if family == "mixed":
                    cmd += ["--num_ops", str(NUM_OPS_MIXED)]

                subprocess.run(cmd, check=True)

    print(f"\n[OK] Previews written to: {OUT_DIR}")

if __name__ == "__main__":
    main()
