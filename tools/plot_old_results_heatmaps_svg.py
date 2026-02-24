#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
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


def _hex(rgb: Tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _interp(a: Tuple[int, int, int], b: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    t = float(np.clip(t, 0.0, 1.0))
    return (
        int(round(a[0] + (b[0] - a[0]) * t)),
        int(round(a[1] + (b[1] - a[1]) * t)),
        int(round(a[2] + (b[2] - a[2]) * t)),
    )


def viridis_rgb(t: float) -> Tuple[int, int, int]:
    # Compact viridis approximation with 5 control points.
    stops = [
        (0.00, (68, 1, 84)),
        (0.25, (59, 82, 139)),
        (0.50, (33, 145, 140)),
        (0.75, (94, 201, 98)),
        (1.00, (253, 231, 37)),
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
    return _hex(viridis_rgb(float(np.clip(t, 0.0, 1.0))))


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


def main() -> None:
    p = argparse.ArgumentParser(description="Generate 2x2 heatmap figure from old_results.csv as SVG.")
    p.add_argument("--csv", type=str, default="old_results.csv")
    p.add_argument("--out", type=str, default="runs/figures/old_results_heatmaps.svg")
    p.add_argument("--global_scale", action="store_true")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    needed = {"Dataset", "Target_Modules", "Rank", "Accuracy", "Std"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df["Rank"] = df["Rank"].astype(int)

    mats: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for ds in DATASET_ORDER:
        mats[ds] = build_mats(df, ds)

    lims: Dict[str, Tuple[float, float]] = {}
    if args.global_scale:
        all_vals = np.concatenate([mats[ds][0].reshape(-1) for ds in DATASET_ORDER], axis=0)
        gl = limits(all_vals)
        for ds in DATASET_ORDER:
            lims[ds] = gl
    else:
        for ds in DATASET_ORDER:
            lims[ds] = limits(mats[ds][0])

    # Compact 2x2 layout:
    # - each panel is horizontal (Target Modules as columns, Ranks as rows)
    # - dataset name on the left of each panel
    # - per-panel colorbar (no shared colorbar)
    margin_x, margin_y = 14, 14
    panel_gap_x, panel_gap_y = 14, 10
    panel_label_w = 112
    rank_w = 26
    top_header_h = 64
    cell_w, cell_h = 37, 25
    n_cols = len(TARGET_MODULES)
    n_rows = len(RANKS)
    grid_w, grid_h = cell_w * n_cols, cell_h * n_rows
    cbar_w = 11
    cbar_gap = 6
    cbar_label_w = 28

    panel_w = panel_label_w + rank_w + grid_w + cbar_gap + cbar_w + cbar_label_w
    panel_h = top_header_h + grid_h

    panel_order = [
        ["Breast", "DTD"],
        ["CUB-200", "FGVC-Aircraft"],
    ]

    W = margin_x * 2 + panel_w * 2 + panel_gap_x
    H = margin_y * 2 + panel_h * 2 + panel_gap_y

    out: List[str] = []
    out.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
    out.append('<rect width="100%" height="100%" fill="white"/>')

    def draw_panel(ds: str, px: float, py: float) -> None:
        means, stds = mats[ds]
        vals = means.T  # (rank, module)
        devs = stds.T
        vmin, vmax = lims[ds]

        gx = px + panel_label_w + rank_w
        gy = py + top_header_h

        # Panel frame (very subtle).
        out.append(
            f'<rect x="{px}" y="{py}" width="{panel_w}" height="{panel_h}" fill="none" '
            f'stroke="#e6e6e6" stroke-width="0.9"/>'
        )

        # Dataset label (left), aligned with the bottom rank row.
        ds_y = gy + (n_rows - 1) * cell_h + cell_h * 0.70
        out.append(
            f'<text x="{px + 6}" y="{ds_y:.1f}" text-anchor="start" '
            f'font-size="14.5" font-family="{FONT}" font-weight="700">{html.escape(ds)}</text>'
        )

        # Rank header + row labels.
        out.append(
            f'<text x="{px + panel_label_w + rank_w/2:.1f}" y="{py + 15:.1f}" text-anchor="middle" '
            f'font-size="11.5" font-family="{FONT}" font-weight="700">Rank</text>'
        )
        for i, r in enumerate(RANKS):
            y = gy + i * cell_h + cell_h * 0.70
            out.append(
                f'<text x="{px + panel_label_w + rank_w - 5}" y="{y:.1f}" text-anchor="end" '
                f'font-size="10.8" font-family="{FONT}" font-weight="700">{r}</text>'
            )

        # Module header + labels (top, rotated more steeply for readability).
        out.append(
            f'<text x="{gx + grid_w/2:.1f}" y="{py + 16:.1f}" text-anchor="middle" '
            f'font-size="12.2" font-family="{FONT}" font-weight="700">Target Modules</text>'
        )
        for j, mod in enumerate(TARGET_MODULES):
            x = gx + j * cell_w + cell_w / 2
            y = gy - 8
            out.append(
                f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="end" '
                f'transform="rotate(-58 {x:.1f} {y:.1f})" '
                f'font-size="8.8" font-family="{FONT}" font-weight="700">{html.escape(mod)}</text>'
            )

        # Cells with value + std.
        for i in range(n_rows):
            for j in range(n_cols):
                x = gx + j * cell_w
                y = gy + i * cell_h
                v = float(vals[i, j])
                s = float(devs[i, j])
                out.append(
                    f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" '
                    f'fill="{val_to_color(v, vmin, vmax)}" stroke="white" stroke-width="0.9"/>'
                )
                tx = x + cell_w / 2
                ty = y + cell_h / 2
                out.append(
                    f'<text x="{tx:.1f}" y="{ty:.1f}" text-anchor="middle" '
                    f'font-size="7.6" font-family="{FONT}" fill="black" font-weight="700">'
                    f'<tspan x="{tx:.1f}" dy="-1.2">{v:.1f}</tspan>'
                    f'<tspan x="{tx:.1f}" dy="8.1">±{s:.1f}</tspan>'
                    f"</text>"
                )

        # Colorbar (per panel).
        cbx = gx + grid_w + cbar_gap
        cby = gy
        steps = 64
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
            f'<rect x="{cbx}" y="{cby}" width="{cbar_w}" height="{grid_h}" '
            f'fill="none" stroke="black" stroke-width="0.6"/>'
        )
        out.append(
            f'<text x="{cbx + cbar_w + 5}" y="{cby + 8}" font-size="8.8" '
            f'font-family="{FONT}" font-weight="700">{vmax:.1f}</text>'
        )
        out.append(
            f'<text x="{cbx + cbar_w + 5}" y="{cby + grid_h - 1}" font-size="8.8" '
            f'font-family="{FONT}" font-weight="700">{vmin:.1f}</text>'
        )
        out.append(
            f'<text x="{cbx + cbar_w/2}" y="{cby - 6}" text-anchor="middle" '
            f'font-size="8.8" font-family="{FONT}" font-weight="700">Acc</text>'
        )

    for r in range(2):
        for c in range(2):
            ds = panel_order[r][c]
            px = margin_x + c * (panel_w + panel_gap_x)
            py = margin_y + r * (panel_h + panel_gap_y)
            draw_panel(ds, px, py)

    out.append("</svg>")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out), encoding="utf-8")
    print(f"[ok] saved: {out_path}")


if __name__ == "__main__":
    main()
