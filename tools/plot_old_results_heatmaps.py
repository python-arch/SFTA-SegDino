#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
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
DEFAULT_DATASET_ORDER = ["Breast", "DTD", "CUB-200", "FGVC-Aircraft"]


def _parse_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _build_mats(
    df: pd.DataFrame,
    dataset: str,
    *,
    dataset_col: str,
    module_col: str,
    rank_col: str,
    mean_col: str,
    std_col: str,
) -> Tuple[np.ndarray, np.ndarray]:
    d = df[df[dataset_col] == dataset]
    means = np.full((len(TARGET_MODULES), len(RANKS)), np.nan, dtype=float)
    stds = np.full((len(TARGET_MODULES), len(RANKS)), np.nan, dtype=float)

    for i, m in enumerate(TARGET_MODULES):
        dm = d[d[module_col] == m]
        for j, r in enumerate(RANKS):
            cell = dm[dm[rank_col] == r]
            if cell.empty:
                continue
            means[i, j] = float(cell.iloc[0][mean_col])
            stds[i, j] = float(cell.iloc[0][std_col])

    missing = np.argwhere(np.isnan(means))
    if missing.size > 0:
        misses = [f"(module={TARGET_MODULES[i]}, rank={RANKS[j]})" for i, j in missing]
        raise ValueError(f"Missing cells for dataset '{dataset}': {', '.join(misses)}")

    return means, stds


def plot_heatmap_horizontal(
    ax: plt.Axes,
    dataset_name: str,
    means: np.ndarray,
    stds: np.ndarray | None = None,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "viridis",
    show_module_labels: bool = True,
    show_rank_label: bool = True,
) -> plt.AxesImage:
    # means is (modules, ranks). We transpose to horizontal layout:
    # rows=ranks, cols=target modules.
    vals = means.T
    devs = stds.T if stds is not None else None
    im = ax.imshow(vals, aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title(dataset_name, fontsize=14, pad=8, weight="bold", loc="left")

    ax.set_xticks(np.arange(len(TARGET_MODULES)))
    ax.set_xticklabels(TARGET_MODULES if show_module_labels else [], rotation=30, ha="right", fontsize=9, weight="bold")
    ax.set_yticks(np.arange(len(RANKS)))
    ax.set_yticklabels([str(r) for r in RANKS], fontsize=10, weight="bold")

    if show_module_labels:
        ax.set_xlabel("Target Modules", fontsize=11, weight="bold")
    if show_rank_label:
        ax.set_ylabel("Rank", fontsize=11, weight="bold")

    ax.set_xticks(np.arange(-0.5, len(TARGET_MODULES), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(RANKS), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    for s in ax.spines.values():
        s.set_linewidth(0.8)

    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            if devs is None:
                txt = f"{vals[i, j]:.1f}"
            else:
                txt = f"{vals[i, j]:.1f}\n±{devs[i, j]:.1f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.2, color="black", fontweight="bold")
    return im


def _auto_limits(vals: np.ndarray) -> Tuple[float, float]:
    lo = float(np.nanmin(vals))
    hi = float(np.nanmax(vals))
    if np.isclose(lo, hi):
        return lo - 0.5, hi + 0.5
    pad = 0.05 * (hi - lo)
    return lo - pad, hi + pad


def main() -> None:
    p = argparse.ArgumentParser(description="Plot 2x2 heatmaps from old_results.csv style data.")
    p.add_argument("--csv", type=str, default="old_results.csv")
    p.add_argument("--out", type=str, default="runs/figures/old_results_heatmaps.pdf")
    p.add_argument("--datasets", type=str, default=",".join(DEFAULT_DATASET_ORDER))
    p.add_argument("--dataset_col", type=str, default="Dataset")
    p.add_argument("--module_col", type=str, default="Target_Modules")
    p.add_argument("--rank_col", type=str, default="Rank")
    p.add_argument("--mean_col", type=str, default="Accuracy")
    p.add_argument("--std_col", type=str, default="Std")
    p.add_argument("--global_scale", action="store_true", help="Use one color scale for all subplots.")
    p.add_argument("--cmap", type=str, default="viridis")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    for c in [args.dataset_col, args.module_col, args.rank_col, args.mean_col, args.std_col]:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in {csv_path}. Found: {list(df.columns)}")

    df = df.copy()
    df[args.rank_col] = df[args.rank_col].astype(int)
    datasets = _parse_list(args.datasets)
    if len(datasets) != 4:
        raise ValueError("This plotting layout expects exactly 4 datasets (for 2x2 panel).")

    mats: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for ds in datasets:
        mats[ds] = _build_mats(
            df,
            ds,
            dataset_col=args.dataset_col,
            module_col=args.module_col,
            rank_col=args.rank_col,
            mean_col=args.mean_col,
            std_col=args.std_col,
        )

    limits: Dict[str, Tuple[float, float]] = {}
    if args.global_scale:
        all_vals = np.concatenate([mats[ds][0].reshape(-1) for ds in datasets], axis=0)
        gmin, gmax = _auto_limits(all_vals)
        for ds in datasets:
            limits[ds] = (gmin, gmax)
    else:
        for ds in datasets:
            limits[ds] = _auto_limits(mats[ds][0])

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "xtick.major.pad": 1.5,
            "ytick.major.pad": 1.5,
        }
    )
    fig, axs = plt.subplots(2, 2, figsize=(15.0, 6.8))
    # Keep requested arrangement: 2 columns, two datasets per column.
    layout = [
        (0, 0, datasets[0], True),
        (0, 1, datasets[1], False),
        (1, 0, datasets[2], True),
        (1, 1, datasets[3], False),
    ]
    for r, c, ds, show_rank_label in layout:
        means, stds = mats[ds]
        vmin, vmax = limits[ds]
        im = plot_heatmap_horizontal(
            axs[r, c],
            ds,
            means,
            stds,
            vmin=vmin,
            vmax=vmax,
            cmap=args.cmap,
            show_module_labels=True,
            show_rank_label=show_rank_label,
        )
        cbar = fig.colorbar(im, ax=axs[r, c], fraction=0.026, pad=0.01)
        cbar.set_label("Acc (%)", fontsize=10, weight="bold")
        cbar.ax.tick_params(labelsize=9)
        for t in cbar.ax.get_yticklabels():
            t.set_fontweight("bold")

    # Compact, paper-style spacing.
    fig.subplots_adjust(left=0.055, right=0.988, top=0.948, bottom=0.116, wspace=0.032, hspace=0.078)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi)
    print(f"[ok] saved: {out_path}")
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
