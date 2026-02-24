#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

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
RANKS = [4, 8, 16, 32]
EXPECTED_CELLS = len(TARGET_MODULES) * len(RANKS)

DATASET_TOKEN_TO_NAME: Dict[str, str] = {
    "pneumoniamnist": "PneumoniaMNIST",
    "dermamnist": "DermaMNIST (MedMNIST)",
    "bloodmnist": "BloodMNIST (MedMNIST)",
    "organmnist": "OrganMNIST",
    "retinamnist": "RetinaMNIST",
}
DATASET_ORDER = [
    "PneumoniaMNIST",
    "DermaMNIST (MedMNIST)",
    "BloodMNIST (MedMNIST)",
    "OrganMNIST",
    "RetinaMNIST",
]


def _norm_dataset_token(s: str) -> str:
    return "".join(ch.lower() for ch in str(s) if ch.isalnum())


def _canon_dataset_name(dataset: str, dataset_token: str) -> str:
    tok = _norm_dataset_token(dataset_token) if dataset_token else _norm_dataset_token(dataset)
    return DATASET_TOKEN_TO_NAME.get(tok, str(dataset).strip())


def _canon_placement(s: str) -> str:
    parts = [p.strip().upper() for p in str(s).split(",") if p.strip()]
    return ",".join(parts)


def _dataset_sort_key(name: str) -> tuple[int, str]:
    if name in DATASET_ORDER:
        return (DATASET_ORDER.index(name), name)
    return (999, name)


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate MedMNIST LoRA grid CSV into heatmap-ready CSV.")
    p.add_argument("--grid_csv", type=str, default="runs/medmnist_cls/lora_grid_results.csv")
    p.add_argument("--out_csv", type=str, default="runs/medmnist_cls/medmnist_heatmap_results.csv")
    p.add_argument("--metric", type=str, default="test_acc", choices=["test_acc", "best_val_acc", "test_auc"])
    p.add_argument("--percent", action="store_true", help="Convert metric to percent scale (x100).")
    p.add_argument("--only_complete_datasets", action="store_true", help="Keep only datasets with full 12x4 grid.")
    args = p.parse_args()

    in_path = Path(args.grid_csv)
    if not in_path.is_file():
        raise FileNotFoundError(f"Missing grid csv: {in_path}")

    df = pd.read_csv(in_path)
    needed = {"dataset", "lora_placement", "lora_r", "seed", args.metric}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {in_path}: {sorted(missing)}")

    if "dataset_token" not in df.columns:
        df["dataset_token"] = ""

    # Keep last entry if the same (dataset, placement, rank, seed) was re-run.
    df = df.reset_index(drop=True)
    df["dataset_name"] = [
        _canon_dataset_name(d, t) for d, t in zip(df["dataset"].astype(str), df["dataset_token"].astype(str))
    ]
    df["placement"] = df["lora_placement"].map(_canon_placement)
    df["rank"] = df["lora_r"].astype(int)
    df["seed"] = df["seed"].astype(int)
    df = df.dropna(subset=[args.metric])
    df = df.drop_duplicates(subset=["dataset_name", "placement", "rank", "seed"], keep="last")

    grouped = (
        df.groupby(["dataset_name", "placement", "rank"], as_index=False)[args.metric]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "Accuracy", "std": "Std", "count": "N"})
    )
    grouped["Std"] = grouped["Std"].fillna(0.0)
    if args.percent:
        grouped["Accuracy"] = grouped["Accuracy"] * 100.0
        grouped["Std"] = grouped["Std"] * 100.0

    grouped = grouped.rename(columns={"dataset_name": "Dataset", "placement": "Target_Modules", "rank": "Rank"})
    grouped = grouped[grouped["Target_Modules"].isin(TARGET_MODULES) & grouped["Rank"].isin(RANKS)]

    coverage = grouped.groupby("Dataset", as_index=False).size().rename(columns={"size": "cells"})
    coverage["expected"] = EXPECTED_CELLS
    coverage["complete"] = coverage["cells"] == EXPECTED_CELLS
    coverage = coverage.sort_values(by="Dataset", key=lambda s: s.map(lambda x: _dataset_sort_key(x)))

    print("[coverage] dataset cell counts")
    for _, r in coverage.iterrows():
        marker = "OK" if bool(r["complete"]) else "INCOMPLETE"
        print(f"  - {r['Dataset']}: {int(r['cells'])}/{EXPECTED_CELLS} ({marker})")

    if args.only_complete_datasets:
        complete = set(coverage.loc[coverage["complete"], "Dataset"].tolist())
        grouped = grouped[grouped["Dataset"].isin(complete)]
        print(f"[info] keeping complete datasets only: {sorted(complete, key=_dataset_sort_key)}")

    grouped["Dataset"] = pd.Categorical(
        grouped["Dataset"],
        categories=sorted(grouped["Dataset"].dropna().unique().tolist(), key=_dataset_sort_key),
        ordered=True,
    )
    grouped["Target_Modules"] = pd.Categorical(
        grouped["Target_Modules"], categories=TARGET_MODULES, ordered=True
    )
    grouped["Rank"] = pd.Categorical(grouped["Rank"], categories=RANKS, ordered=True)
    grouped = grouped.sort_values(by=["Dataset", "Target_Modules", "Rank"])

    out = grouped[["Dataset", "Target_Modules", "Rank", "Accuracy", "Std", "N"]].copy()
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[ok] wrote: {out_path} ({len(out)} rows)")


if __name__ == "__main__":
    main()
