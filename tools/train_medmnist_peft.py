#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters import LoRALinear, count_parameters, inject_lora_with_placement


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
}

DATASET_NUM_CLASSES: Dict[str, int] = {
    "pneumoniamnist": 2,
    "dermamnist": 7,
    "bloodmnist": 8,
    "organmnist": 11,
    "retinamnist": 5,
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
    p = argparse.ArgumentParser(description="Train MedMNIST classification with DINO backbone + LoRA.")
    p.add_argument("--npz_dir", type=str, default="./medmnist_npz")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--num_classes", type=int, default=0, help="If 0, auto-infer from dataset token.")
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

    dataset_token = resolve_dataset_token(args.dataset)
    if args.num_classes > 0:
        num_classes = int(args.num_classes)
    else:
        if dataset_token not in DATASET_NUM_CLASSES:
            raise ValueError(f"Unknown dataset='{args.dataset}'. Provide --num_classes explicitly.")
        num_classes = DATASET_NUM_CLASSES[dataset_token]
    # Keep args/config consistent for logging systems (e.g. wandb config is immutable by default).
    args.num_classes = num_classes

    npz_path = Path(args.npz_dir) / f"{dataset_token}.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(f"Missing dataset npz: {npz_path}")
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

    train_ds = MedMNISTNpzDataset(npz_path=npz_path, split="train", input_size=int(args.input_size))
    val_ds = MedMNISTNpzDataset(npz_path=npz_path, split="val", input_size=int(args.input_size))
    test_ds = MedMNISTNpzDataset(npz_path=npz_path, split="test", input_size=int(args.input_size))

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
    binc = np.bincount(train_ds.labels, minlength=num_classes).astype(np.float64)
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
            name=f"medmnist_{dataset_token}_{placement_tag}_r{args.lora_r}_s{args.seed}",
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
        "run_dir": str(run_dir),
        "npz_path": str(npz_path),
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
