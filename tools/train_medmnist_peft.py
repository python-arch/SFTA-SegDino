#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters import FusedQKVLoRALinear, LoRALinear, count_parameters, inject_lora_with_placement


DATASET_ALIASES: Dict[str, str] = {
    "pneumoniamnist": "pneumoniamnist",
    "dermamnist": "dermamnist",
    "dermamnistmedmnist": "dermamnist",
    "bloodmnist": "bloodmnist",
    "bloodmnistmedmnist": "bloodmnist",
    "organmnist": "organmnist",
    "organamnist": "organmnist",
    "organcmnist": "organmnist",
    "organsmnist": "organmnist",
    "retinamnist": "retinamnist",
    "aptos2019": "aptos2019",
    "aptos2019blindnessdetection": "aptos2019",
    "ham10000": "ham10000",
    "skincancermnistham10000": "ham10000",
    "breakhis": "breakhis",
}

DATASET_NUM_CLASSES: Dict[str, int] = {
    "pneumoniamnist": 2,
    "dermamnist": 7,
    "bloodmnist": 8,
    "organmnist": 11,
    "retinamnist": 5,
    "aptos2019": 5,
    "ham10000": 7,
    "breakhis": 2,
}

HAM_DX_TO_INT: Dict[str, int] = {
    "akiec": 0,
    "bcc": 1,
    "bkl": 2,
    "df": 3,
    "mel": 4,
    "nv": 5,
    "vasc": 6,
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _normalize_name(name: str) -> str:
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def normalize_placement_tag(placement: str) -> str:
    return "_".join([p.strip().upper() for p in str(placement).split(",") if p.strip()])


def resolve_dataset_token(name: str) -> str:
    key = _normalize_name(name)
    return DATASET_ALIASES.get(key, key)


class MedMNISTNpzDataset(Dataset):
    def __init__(self, npz_path: Path, split: str, input_size: int) -> None:
        super().__init__()
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split: {split}")
        with np.load(npz_path) as data:
            img_key = f"{split}_images"
            if img_key not in data and f"{split}_imag" in data:
                img_key = f"{split}_imag"
            lbl_key = f"{split}_labels"
            if img_key not in data or lbl_key not in data:
                raise KeyError(f"Missing keys for split={split} in {npz_path}. Found keys: {data.files}")
            self.images = np.array(data[img_key])
            self.labels = np.array(data[lbl_key]).reshape(-1).astype(np.int64)

        self.transform = transforms.Compose(
            [
                transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = self.images[idx]
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=2)
        elif img.ndim == 3:
            if img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
                img = np.transpose(img, (1, 2, 0))
            if img.shape[-1] == 1:
                img = np.repeat(img, 3, axis=2)
        else:
            raise ValueError(f"Unsupported image shape: {img.shape}")

        img = img.astype(np.uint8, copy=False)
        x = self.transform(Image.fromarray(img))
        y = int(self.labels[idx])
        return x, y


class ImagePathClassificationDataset(Dataset):
    def __init__(self, items: List[Tuple[str, int]], input_size: int) -> None:
        super().__init__()
        self.items = list(items)
        self.labels = np.array([int(y) for _, y in self.items], dtype=np.int64)
        self.transform = transforms.Compose(
            [
                transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, y = self.items[idx]
        x = Image.open(path).convert("RGB")
        x = self.transform(x)
        return x, int(y)


def _safe_stratified_split(
    indices: np.ndarray,
    labels: np.ndarray,
    *,
    test_size: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    try:
        a, b = train_test_split(
            indices,
            test_size=float(test_size),
            random_state=int(seed),
            shuffle=True,
            stratify=labels,
        )
    except ValueError:
        a, b = train_test_split(
            indices,
            test_size=float(test_size),
            random_state=int(seed),
            shuffle=True,
            stratify=None,
        )
    return np.array(a), np.array(b)


def split_items_train_val_test(
    items: List[Tuple[str, int]],
    *,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], List[Tuple[str, int]]]:
    if not items:
        raise ValueError("Cannot split empty item list.")
    if val_ratio <= 0.0 or test_ratio <= 0.0 or (val_ratio + test_ratio) >= 1.0:
        raise ValueError(f"Invalid val/test split ratios: val={val_ratio}, test={test_ratio}")

    idx = np.arange(len(items))
    labels = np.array([int(y) for _, y in items], dtype=np.int64)
    train_val_idx, test_idx = _safe_stratified_split(
        idx,
        labels,
        test_size=test_ratio,
        seed=seed,
    )
    train_val_labels = labels[train_val_idx]
    val_size_in_train_val = val_ratio / (1.0 - test_ratio)
    train_idx, val_idx = _safe_stratified_split(
        train_val_idx,
        train_val_labels,
        test_size=val_size_in_train_val,
        seed=seed,
    )
    train_items = [items[int(i)] for i in train_idx]
    val_items = [items[int(i)] for i in val_idx]
    test_items = [items[int(i)] for i in test_idx]
    return train_items, val_items, test_items


def _dataset_root_for_token(dataset_root: Path, dataset_token: str) -> Path:
    return dataset_root / dataset_token


def build_aptos_items(dataset_root: Path) -> List[Tuple[str, int]]:
    root = _dataset_root_for_token(dataset_root, "aptos2019")
    csv_path = root / "train.csv"
    img_dir = root / "train_images"
    if not csv_path.is_file():
        raise FileNotFoundError(f"APTOS train.csv not found: {csv_path}")
    if not img_dir.is_dir():
        raise FileNotFoundError(f"APTOS train_images dir not found: {img_dir}")

    df = pd.read_csv(csv_path)
    needed = {"id_code", "diagnosis"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"APTOS CSV missing columns {sorted(missing)} in {csv_path}")

    items: List[Tuple[str, int]] = []
    for r in df.itertuples(index=False):
        p = img_dir / f"{str(r.id_code)}.png"
        if p.is_file():
            items.append((str(p), int(r.diagnosis)))
    if not items:
        raise ValueError(f"No valid labeled APTOS images found under {img_dir}")
    return items


def build_ham10000_items(dataset_root: Path) -> List[Tuple[str, int]]:
    root = _dataset_root_for_token(dataset_root, "ham10000")
    meta_path = root / "HAM10000_metadata.csv"
    if not meta_path.is_file():
        raise FileNotFoundError(f"HAM10000 metadata not found: {meta_path}")
    meta = pd.read_csv(meta_path)
    needed = {"image_id", "dx"}
    missing = needed - set(meta.columns)
    if missing:
        raise ValueError(f"HAM10000 metadata missing columns {sorted(missing)} in {meta_path}")

    img_paths = glob.glob(str(root / "HAM10000_images_part_*" / "*.jpg"))
    img_index = {Path(p).stem: p for p in img_paths}

    items: List[Tuple[str, int]] = []
    for r in meta.itertuples(index=False):
        image_id = str(r.image_id)
        dx = str(r.dx).lower()
        if image_id in img_index and dx in HAM_DX_TO_INT:
            items.append((img_index[image_id], int(HAM_DX_TO_INT[dx])))
    if not items:
        raise ValueError(f"No valid HAM10000 images found under {root}")
    return items


def _extract_breakhis_mag(path: Path) -> str:
    parts = path.stem.split("-")
    if len(parts) >= 2 and parts[-2].isdigit():
        return parts[-2]
    return ""


def build_breakhis_items(dataset_root: Path, mag_filter: str) -> List[Tuple[str, int]]:
    root = _dataset_root_for_token(dataset_root, "breakhis")
    if not root.is_dir():
        raise FileNotFoundError(f"BreakHis root not found: {root}")

    mag_filter = str(mag_filter).strip()
    items: List[Tuple[str, int]] = []
    for p in root.rglob("*.png"):
        lower_parent = str(p.parent).lower()
        if "benign" in lower_parent:
            y = 0
        elif "malignant" in lower_parent:
            y = 1
        else:
            continue
        mag = _extract_breakhis_mag(p)
        if mag_filter and mag != mag_filter:
            continue
        items.append((str(p), y))
    if not items:
        suffix = f" with mag_filter={mag_filter}" if mag_filter else ""
        raise ValueError(f"No BreakHis images found under {root}{suffix}")
    return items


def build_datasets(
    *,
    dataset_token: str,
    npz_dir: str,
    dataset_root: str,
    input_size: int,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    breakhis_mag: str,
) -> Tuple[Dataset, Dataset, Dataset, str]:
    if dataset_token in {"pneumoniamnist", "dermamnist", "bloodmnist", "organmnist", "retinamnist"}:
        npz_path = Path(npz_dir) / f"{dataset_token}.npz"
        if not npz_path.is_file():
            raise FileNotFoundError(f"Missing dataset npz: {npz_path}")
        train_ds = MedMNISTNpzDataset(npz_path=npz_path, split="train", input_size=int(input_size))
        val_ds = MedMNISTNpzDataset(npz_path=npz_path, split="val", input_size=int(input_size))
        test_ds = MedMNISTNpzDataset(npz_path=npz_path, split="test", input_size=int(input_size))
        return train_ds, val_ds, test_ds, str(npz_path)

    root = Path(dataset_root)
    if dataset_token == "aptos2019":
        items = build_aptos_items(root)
    elif dataset_token == "ham10000":
        items = build_ham10000_items(root)
    elif dataset_token == "breakhis":
        items = build_breakhis_items(root, breakhis_mag)
    else:
        raise ValueError(f"Unknown dataset_token={dataset_token}")

    train_items, val_items, test_items = split_items_train_val_test(
        items,
        val_ratio=float(val_ratio),
        test_ratio=float(test_ratio),
        seed=int(seed),
    )
    train_ds = ImagePathClassificationDataset(train_items, input_size=int(input_size))
    val_ds = ImagePathClassificationDataset(val_items, input_size=int(input_size))
    test_ds = ImagePathClassificationDataset(test_items, input_size=int(input_size))
    return train_ds, val_ds, test_ds, str(_dataset_root_for_token(root, dataset_token))


class MedMNISTClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int, input_size: int) -> None:
        super().__init__()
        self.backbone = backbone
        feat_dim = self._infer_feature_dim(input_size)
        self.head = nn.Linear(feat_dim, int(num_classes))

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.backbone, "forward_features"):
            out = self.backbone.forward_features(x)
        else:
            out = self.backbone(x)

        if isinstance(out, dict):
            for key in ("x_norm_clstoken", "cls_token", "x_cls", "pooler_output", "features", "feats"):
                if key in out:
                    out = out[key]
                    break
            else:
                out = next(iter(out.values()))

        if not torch.is_tensor(out):
            raise TypeError(f"Unsupported backbone output type: {type(out)}")

        if out.dim() == 4:
            out = out.mean(dim=(-2, -1))
        elif out.dim() == 3:
            out = out[:, 0, :]
        elif out.dim() != 2:
            raise ValueError(f"Unsupported backbone output shape: {tuple(out.shape)}")
        return out

    def _infer_feature_dim(self, input_size: int) -> int:
        with torch.no_grad():
            device = next(self.backbone.parameters()).device
            dummy = torch.zeros(1, 3, input_size, input_size, device=device)
            feats = self._extract_features(dummy)
            return int(feats.shape[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self._extract_features(x)
        return self.head(feats)


@dataclass
class EvalResult:
    loss: float
    acc: float
    auc: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def enable_lora_and_head_trainable(
    model: MedMNISTClassifier,
    *,
    adapter: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_placement: str,
) -> int:
    for p in model.parameters():
        p.requires_grad_(False)

    wrapped = 0
    if adapter == "lora":
        wrapped = inject_lora_with_placement(
            model.backbone,
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            placement=lora_placement,
        )
        for m in model.modules():
            if isinstance(m, LoRALinear):
                if isinstance(m.lora_A, torch.Tensor):
                    m.lora_A.requires_grad_(True)
                if isinstance(m.lora_B, torch.Tensor):
                    m.lora_B.requires_grad_(True)
            if isinstance(m, FusedQKVLoRALinear):
                for p in m.lora_A.values():
                    p.requires_grad_(True)
                for p in m.lora_B.values():
                    p.requires_grad_(True)
    elif adapter == "none":
        wrapped = 0
    else:
        raise ValueError(f"Unsupported adapter={adapter}; expected one of: none,lora")

    for p in model.head.parameters():
        p.requires_grad_(True)
    return wrapped


def compute_auc(y_true: np.ndarray, probs: np.ndarray, num_classes: int) -> float:
    try:
        if num_classes == 2:
            return float(roc_auc_score(y_true, probs[:, 1]))
        y_bin = label_binarize(y_true, classes=np.arange(num_classes))
        return float(roc_auc_score(y_bin, probs, average="macro", multi_class="ovr"))
    except ValueError:
        return float("nan")


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, num_classes: int) -> EvalResult:
    model.eval()
    total_loss = 0.0
    n = 0
    logits_all = []
    labels_all = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            bs = x.shape[0]
            total_loss += float(loss.item()) * bs
            n += bs
            logits_all.append(logits.detach().cpu())
            labels_all.append(y.detach().cpu())

    logits_np = torch.cat(logits_all, dim=0).numpy()
    labels_np = torch.cat(labels_all, dim=0).numpy()
    probs = torch.softmax(torch.from_numpy(logits_np), dim=1).numpy()
    pred = probs.argmax(axis=1)
    acc = float((pred == labels_np).mean())
    auc = compute_auc(labels_np, probs, num_classes)
    loss = total_loss / max(1, n)
    return EvalResult(loss=loss, acc=acc, auc=auc)


def append_row_csv(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train medical classification (MedMNIST/APTOS/HAM10000/BreakHis) with DINO + LoRA.")
    p.add_argument("--npz_dir", type=str, default="./medmnist_npz")
    p.add_argument("--dataset_root", type=str, default="./data", help="Root for folder-based datasets (aptos2019, ham10000, breakhis).")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--num_classes", type=int, default=0, help="If 0, auto-infer from dataset token.")
    p.add_argument("--val_ratio", type=float, default=0.1, help="Val split ratio for folder-based datasets.")
    p.add_argument("--test_ratio", type=float, default=0.2, help="Test split ratio for folder-based datasets.")
    p.add_argument("--breakhis_mag", type=str, default="", help='BreakHis magnification filter: "", 40, 100, 200, or 400.')
    p.add_argument("--input_size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--repo_dir", type=str, default="./dinov3")
    p.add_argument("--dino_ckpt", type=str, required=True)
    p.add_argument("--dino_size", type=str, default="s", choices=["s", "b"])
    p.add_argument("--adapter", type=str, default="lora", choices=["lora", "none"])
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_placement", type=str, default="Q,K,V,P,F1,F2")
    p.add_argument("--out_dir", type=str, default="./runs/medmnist_cls")
    p.add_argument("--out_csv", type=str, default="./runs/medmnist_cls/results.csv")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="segdino_medmnist_cls")
    p.add_argument("--wandb_entity", type=str, default=None)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    set_seed(int(args.seed))
    args.breakhis_mag = str(args.breakhis_mag).strip()
    if args.breakhis_mag and args.breakhis_mag not in {"40", "100", "200", "400"}:
        raise ValueError(f"Invalid --breakhis_mag={args.breakhis_mag}. Allowed: 40,100,200,400 or empty.")

    dataset_token = resolve_dataset_token(args.dataset)
    if args.num_classes > 0:
        num_classes = int(args.num_classes)
    else:
        if dataset_token not in DATASET_NUM_CLASSES:
            raise ValueError(f"Unknown dataset='{args.dataset}'. Provide --num_classes explicitly.")
        num_classes = DATASET_NUM_CLASSES[dataset_token]
    # Keep args/config consistent for logging systems (e.g. wandb config is immutable by default).
    args.num_classes = num_classes

    if not Path(args.repo_dir).is_dir():
        raise FileNotFoundError(f"repo_dir not found: {args.repo_dir}")
    if not Path(args.dino_ckpt).is_file():
        raise FileNotFoundError(f"dino_ckpt not found: {args.dino_ckpt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.dino_size == "b":
        backbone = torch.hub.load(args.repo_dir, "dinov3_vitb16", source="local", weights=args.dino_ckpt)
    else:
        backbone = torch.hub.load(args.repo_dir, "dinov3_vits16", source="local", weights=args.dino_ckpt)
    backbone.to(device)
    for p in backbone.parameters():
        p.requires_grad_(False)

    model = MedMNISTClassifier(backbone=backbone, num_classes=num_classes, input_size=int(args.input_size)).to(device)
    wrapped = enable_lora_and_head_trainable(
        model,
        adapter=args.adapter,
        lora_r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        lora_placement=str(args.lora_placement),
    )
    total_params, trainable_params, trainable_pct = count_parameters(model)

    train_ds, val_ds, test_ds, data_source_path = build_datasets(
        dataset_token=dataset_token,
        npz_dir=str(args.npz_dir),
        dataset_root=str(args.dataset_root),
        input_size=int(args.input_size),
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
        seed=int(args.seed),
        breakhis_mag=str(args.breakhis_mag),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )

    class_weights = None
    train_labels = np.array(getattr(train_ds, "labels", []), dtype=np.int64)
    if train_labels.size == 0:
        raise RuntimeError("Training dataset does not expose labels; cannot compute class-balanced loss.")
    binc = np.bincount(train_labels, minlength=num_classes).astype(np.float64)
    if np.all(binc > 0):
        inv = 1.0 / binc
        inv = inv / inv.sum() * num_classes
        class_weights = torch.tensor(inv, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=int(args.epochs))

    placement_tag = normalize_placement_tag(args.lora_placement)
    run_dir = Path(args.out_dir) / f"{dataset_token}_{placement_tag}_r{args.lora_r}_seed{args.seed}_{args.adapter}"
    run_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = run_dir / "best.pt"

    wandb_run = None
    wandb_enabled = bool(args.wandb) or ("WANDB_SWEEP_ID" in os.environ)
    if wandb_enabled:
        try:
            import wandb
        except Exception as e:  # pragma: no cover
            raise RuntimeError("wandb logging requested (flag or sweep env), but wandb is not installed.") from e
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            name=f"medical_{dataset_token}_{placement_tag}_r{args.lora_r}_s{args.seed}",
        )
        wandb_run.config.update(
            {
                "dataset_token": dataset_token,
                "num_classes": num_classes,
                "wrapped_lora_layers": wrapped,
                "total_params": total_params,
                "trainable_params": trainable_params,
                "trainable_pct": trainable_pct,
            }
        )

    best_val_acc = -1.0
    best_epoch = -1
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        running = 0.0
        n = 0
        bar = tqdm(train_loader, desc=f"[train e{epoch}] {dataset_token}")
        for x, y in bar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            bs = x.shape[0]
            running += float(loss.item()) * bs
            n += bs
            bar.set_postfix(loss=f"{(running / max(1, n)):.4f}", lr=f"{optim.param_groups[0]['lr']:.2e}")
        sched.step()

        train_loss = running / max(1, n)
        val_res = evaluate(model, val_loader, criterion, device, num_classes)
        print(
            f"[epoch {epoch}] train_loss={train_loss:.4f} "
            f"val_loss={val_res.loss:.4f} val_acc={val_res.acc:.4f} val_auc={val_res.auc:.4f}"
        )
        if wandb_run:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_res.loss,
                    "val_acc": val_res.acc,
                    "val_auc": val_res.auc,
                    "lr": optim.param_groups[0]["lr"],
                }
            )

        if val_res.acc > best_val_acc:
            best_val_acc = val_res.acc
            best_epoch = epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "dataset_token": dataset_token,
                    "num_classes": num_classes,
                    "best_val_acc": best_val_acc,
                    "best_epoch": best_epoch,
                },
                best_ckpt,
            )

    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_res = evaluate(model, test_loader, criterion, device, num_classes)
    print(
        f"[best epoch={best_epoch}] val_acc={best_val_acc:.4f} "
        f"test_loss={test_res.loss:.4f} test_acc={test_res.acc:.4f} test_auc={test_res.auc:.4f}"
    )

    summary = {
        "dataset": args.dataset,
        "dataset_token": dataset_token,
        "num_classes": num_classes,
        "adapter": args.adapter,
        "lora_r": int(args.lora_r),
        "lora_alpha": int(args.lora_alpha),
        "lora_dropout": float(args.lora_dropout),
        "lora_placement": str(args.lora_placement),
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "best_epoch": int(best_epoch),
        "best_val_acc": float(best_val_acc),
        "test_loss": float(test_res.loss),
        "test_acc": float(test_res.acc),
        "test_auc": float(test_res.auc),
        "wrapped_lora_layers": int(wrapped),
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "trainable_pct": float(trainable_pct),
        "train_count": int(len(train_ds)),
        "val_count": int(len(val_ds)),
        "test_count": int(len(test_ds)),
        "val_ratio": float(args.val_ratio),
        "test_ratio": float(args.test_ratio),
        "breakhis_mag": str(args.breakhis_mag),
        "run_dir": str(run_dir),
        "data_source_path": str(data_source_path),
    }

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    append_row_csv(Path(args.out_csv), summary)
    print(f"[ok] wrote summary: {summary_path}")
    print(f"[ok] appended csv: {args.out_csv}")

    if wandb_run:
        wandb_run.log(
            {
                "best_epoch": best_epoch,
                "best_val_acc": best_val_acc,
                "test_loss": test_res.loss,
                "test_acc": test_res.acc,
                "test_auc": test_res.auc,
            }
        )
        wandb_run.finish()


if __name__ == "__main__":
    main()
