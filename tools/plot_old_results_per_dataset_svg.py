#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

RANKS = [4, 8, 16, 32]
TARGET_MODULES = [
    "Q",
    "K",
    "V",
    "P",
    "F1",
    "Q,K",
    "Q,V",
    "F1,F2",
    "Q,K,V",
    "P,F1,F2",
    "Q,K,V,P",
    "Q,K,V,P,F1,F2",
]
DATASET_ORDER = ["Breast", "DTD", "CUB-200", "FGVC-Aircraft"]
FONT = "Helvetica, Arial, sans-serif"


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def _hex(rgb: Tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _interp(a: Tuple[int, int, int], b: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    t = float(np.clip(t, 0.0, 1.0))
    return (
        int(round(a[0] + (b[0] - a[0]) * t)),
        int(round(a[1] + (b[1] - a[1]) * t)),
        int(round(a[2] + (b[2] - a[2]) * t)),
    )


def orange_rgb(t: float) -> Tuple[int, int, int]:
    # Light -> dark orange ramp (paper-friendly sequential scale).
    stops = [
        (0.00, (255, 245, 235)),
        (0.20, (254, 230, 206)),
        (0.45, (253, 174, 107)),
        (0.70, (241, 105, 19)),
        (1.00, (127, 39, 4)),
    ]
    t = float(np.clip(t, 0.0, 1.0))
    for i in range(len(stops) - 1):
        x0, c0 = stops[i]
        x1, c1 = stops[i + 1]
        if x0 <= t <= x1:
            u = 0.0 if np.isclose(x0, x1) else (t - x0) / (x1 - x0)
            return _interp(c0, c1, u)
    return stops[-1][1]


def val_to_color(v: float, vmin: float, vmax: float) -> str:
    if np.isclose(vmin, vmax):
        t = 0.5
    else:
        t = (v - vmin) / (vmax - vmin)
    return _hex(orange_rgb(float(np.clip(t, 0.0, 1.0))))


def build_mats(df: pd.DataFrame, dataset: str) -> Tuple[np.ndarray, np.ndarray]:
    d = df[df["Dataset"] == dataset]
    means = np.full((len(TARGET_MODULES), len(RANKS)), np.nan, dtype=float)
    stds = np.full((len(TARGET_MODULES), len(RANKS)), np.nan, dtype=float)
    for i, mod in enumerate(TARGET_MODULES):
        dm = d[d["Target_Modules"] == mod]
        for j, rank in enumerate(RANKS):
            row = dm[dm["Rank"] == rank]
            if row.empty:
                continue
            means[i, j] = float(row.iloc[0]["Accuracy"])
            stds[i, j] = float(row.iloc[0]["Std"])
    missing = np.argwhere(np.isnan(means))
    if missing.size:
        miss = [f"{TARGET_MODULES[i]}@{RANKS[j]}" for i, j in missing]
        raise ValueError(f"Missing cells for dataset={dataset}: {', '.join(miss)}")
    return means, stds


def limits(vals: np.ndarray) -> Tuple[float, float]:
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    if np.isclose(lo, hi):
        return lo - 0.5, hi + 0.5
    pad = 0.05 * (hi - lo)
    return lo - pad, hi + pad


def render_dataset_svg(
    out_path: Path,
    dataset: str,
    means: np.ndarray,
    stds: np.ndarray,
    vmin: float,
    vmax: float,
) -> None:
    vals = means.T  # rows=ranks, cols=modules
    devs = stds.T
    n_rows = len(RANKS)
    n_cols = len(TARGET_MODULES)

    # Tight geometry for standalone figure.
    margin_x = 8
    margin_top = 74
    margin_bottom = 26
    rank_w = 24
    cell_w = 34
    cell_h = 23
    grid_w = n_cols * cell_w
    grid_h = n_rows * cell_h
    cbar_gap = 4
    cbar_w = 8
    cbar_label_w = 20

    gx = margin_x + rank_w
    gy = margin_top
    W = margin_x * 2 + rank_w + grid_w + cbar_gap + cbar_w + cbar_label_w
    H = margin_top + grid_h + margin_bottom

    out: List[str] = []
    out.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
    out.append('<rect width="100%" height="100%" fill="white"/>')

    # Rank axis title on the left of the plot.
    rx = margin_x + 5
    ry = gy + grid_h / 2
    out.append(
        f'<text x="{rx:.1f}" y="{ry:.1f}" text-anchor="middle" '
        f'transform="rotate(-90 {rx:.1f} {ry:.1f})" '
        f'font-size="8.0" font-family="{FONT}" font-weight="700">Rank</text>'
    )

    # Rank labels: minimal white space on left.
    for i, r in enumerate(RANKS):
        y = gy + i * cell_h + cell_h * 0.68
        out.append(
            f'<text x="{margin_x + rank_w - 2}" y="{y:.1f}" text-anchor="end" '
            f'font-size="8.0" font-family="{FONT}" font-weight="700">{r}</text>'
        )

    # Heatmap cells.
    for i in range(n_rows):
        for j in range(n_cols):
            x = gx + j * cell_w
            y = gy + i * cell_h
            v = float(vals[i, j])
            s = float(devs[i, j])
            out.append(
                f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" '
                f'fill="{val_to_color(v, vmin, vmax)}" stroke="white" stroke-width="0.85"/>'
            )
            tx = x + cell_w / 2
            ty = y + cell_h / 2
            out.append(
                f'<text x="{tx:.1f}" y="{ty:.1f}" text-anchor="middle" '
                f'font-size="6.8" font-family="{FONT}" fill="black" font-weight="700">'
                f'<tspan x="{tx:.1f}" dy="-1.0">{v:.1f}</tspan>'
                f'<tspan x="{tx:.1f}" dy="7.4">±{s:.1f}</tspan>'
                f"</text>"
            )

    # Target modules: draw after cells so labels stay visible on top.
    for j, mod in enumerate(TARGET_MODULES):
        x = gx + j * cell_w + cell_w / 2
        y = gy - 7
        out.append(
            f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="start" '
            f'transform="rotate(-64 {x:.1f} {y:.1f})" '
            f'font-size="5.7" font-family="{FONT}" font-weight="700">{html.escape(mod)}</text>'
        )

    # Per-figure colorbar.
    cbx = gx + grid_w + cbar_gap
    cby = gy
    steps = 60
    for k in range(steps):
        t0 = k / steps
        y0 = cby + (1 - t0) * grid_h
        h0 = grid_h / steps + 0.45
        val = vmin + t0 * (vmax - vmin)
        out.append(
            f'<rect x="{cbx}" y="{y0:.2f}" width="{cbar_w}" height="{h0:.2f}" '
            f'fill="{val_to_color(val, vmin, vmax)}" stroke="none"/>'
        )
    out.append(
        f'<rect x="{cbx}" y="{cby}" width="{cbar_w}" height="{grid_h}" fill="none" stroke="black" stroke-width="0.55"/>'
    )
    out.append(
        f'<text x="{cbx + cbar_w + 4}" y="{cby + 7}" font-size="7.4" font-family="{FONT}" font-weight="700">{vmax:.1f}</text>'
    )
    out.append(
        f'<text x="{cbx + cbar_w + 4}" y="{cby + grid_h - 1}" font-size="7.4" font-family="{FONT}" font-weight="700">{vmin:.1f}</text>'
    )

    # Dataset name centered below the figure.
    out.append(
        f'<text x="{W/2:.1f}" y="{gy + grid_h + 17:.1f}" text-anchor="middle" '
        f'font-size="11.0" font-family="{FONT}" font-weight="700">{html.escape(dataset)}</text>'
    )

    out.append("</svg>")
    out_path.write_text("\n".join(out), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate one tight SVG heatmap per dataset from old_results.csv.")
    p.add_argument("--csv", type=str, default="old_results.csv")
    p.add_argument("--out_dir", type=str, default="runs/figures/old_results_per_dataset")
    p.add_argument("--datasets", type=str, default="", help="Comma-separated dataset names. Default: all datasets found in CSV.")
    p.add_argument("--global_scale", action="store_true")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    needed = {"Dataset", "Target_Modules", "Rank", "Accuracy", "Std"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df["Rank"] = df["Rank"].astype(int)

    if args.datasets.strip():
        datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    else:
        datasets = list(dict.fromkeys(df["Dataset"].astype(str).tolist()))
    mats: Dict[str, Tuple[np.ndarray, np.ndarray]] = {ds: build_mats(df, ds) for ds in datasets}

    lims: Dict[str, Tuple[float, float]] = {}
    if args.global_scale:
        all_vals = np.concatenate([mats[ds][0].reshape(-1) for ds in datasets], axis=0)
        gl = limits(all_vals)
        for ds in datasets:
            lims[ds] = gl
    else:
        for ds in datasets:
            lims[ds] = limits(mats[ds][0])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ds in datasets:
        means, stds = mats[ds]
        vmin, vmax = lims[ds]
        out_path = out_dir / f"{_slug(ds)}.svg"
        render_dataset_svg(out_path, ds, means, stds, vmin, vmax)
        print(f"[ok] saved: {out_path}")


if __name__ == "__main__":
    main()
