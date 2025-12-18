import os
import cv2
import math
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from typing import Type, Tuple, Optional

def tensor_to_rgb(img_t: torch.Tensor, mean=None, std=None) -> np.ndarray:
    img = img_t.detach().cpu().float()
    img = img.clamp(0, 1).numpy()
    img = (img * 255.0).round().astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def mask_to_gray(mask_t: torch.Tensor, thr: float = 0.5, num_classes: int = 1) -> np.ndarray:
    m = mask_t.detach().cpu().float()
    
    if num_classes == 1:
        if m.ndim == 3 and m.shape[0] == 1:
            m = m[0]
        elif m.ndim == 2:
            pass
        else:
            raise ValueError(f"Unexpected mask tensor shape: {m.shape}")
        if m.max() > 1.0 or m.min() < 0.0:
            m = torch.sigmoid(m)
        m_bin = (m > thr).float()
        m_img = (m_bin * 255.0).round().clamp(0, 255).byte().numpy()
    else:
        if m.dim() == 3 and m.shape[0] == num_classes:
            m = torch.argmax(m, dim=0)
        elif m.dim() == 3 and m.shape[0] == 1:
            m = m[0]
        m_img = (m.float() / max(1, num_classes - 1) * 255.0).round().clamp(0, 255).byte().numpy()
    
    return m_img

def save_train_visuals(epoch, inputs, logits, targets, out_dir, max_save=8, thr=0.5, num_classes=1):
    os.makedirs(out_dir, exist_ok=True)
    b = min(inputs.size(0), max_save)
    for i in range(b):
        img_bgr = tensor_to_rgb(inputs[i])
        pred_gray = mask_to_gray(logits[i], thr, num_classes)
        gt_gray   = mask_to_gray(targets[i], thr, num_classes)
        base = os.path.join(out_dir, f"train_ep{epoch:03d}_idx{i:02d}")
        cv2.imwrite(base + "_img.png",  img_bgr)
        cv2.imwrite(base + "_pred.png", pred_gray)
        cv2.imwrite(base + "_gt.png",   gt_gray)

@torch.no_grad()
def save_eval_visuals(idx, inputs, logits, targets, out_dir, thr=0.5, fname_prefix="val", num_classes=1):
    os.makedirs(out_dir, exist_ok=True)
    img_bgr = tensor_to_rgb(inputs)
    pred_gray = mask_to_gray(logits, thr, num_classes)
    gt_gray   = mask_to_gray(targets, thr, num_classes)
    base = os.path.join(out_dir, f"{fname_prefix}_{idx:05d}")
    cv2.imwrite(base + "_img.png",  img_bgr)
    cv2.imwrite(base + "_pred.png", pred_gray)
    cv2.imwrite(base + "_gt.png",   gt_gray)

def iou_binary_torch(pred_logits, target, eps=1e-6, thresh=0.5):
    prob = torch.sigmoid(pred_logits)
    pred = (prob > thresh).float()
    target = (target > 0.5).float()
    inter = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - inter + eps
    iou = (inter + eps) / union
    return iou.view(-1)

def dice_binary_torch(pred_logits, target, eps=1e-6, thresh=0.5):
    prob = torch.sigmoid(pred_logits)
    pred = (prob > thresh).float()
    target = target.float().clamp(0, 1)
    inter = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + eps
    dice = (2 * inter + eps) / union
    return dice

def dice_multi_class_torch(pred_logits, target, num_classes, eps=1e-6):
    probs = torch.softmax(pred_logits, dim=1)
    preds = torch.argmax(probs, dim=1)
    
    if target.dim() == 4 and target.shape[1] == 1:
        target = target.squeeze(1)
    
    dice_scores = []
    for cls in range(num_classes):
        pred_cls = (preds == cls).float()
        target_cls = (target == cls).float()
        
        inter = (pred_cls * target_cls).sum(dim=(1, 2))
        union = pred_cls.sum(dim=(1, 2)) + target_cls.sum(dim=(1, 2)) + eps
        dice_cls = (2 * inter + eps) / union
        dice_scores.append(dice_cls)
    
    return torch.stack(dice_scores).mean(dim=0).mean()

def iou_multi_class_torch(pred_logits, target, num_classes, eps=1e-6):
    probs = torch.softmax(pred_logits, dim=1)
    preds = torch.argmax(probs, dim=1)
    
    if target.dim() == 4 and target.shape[1] == 1:
        target = target.squeeze(1)
    
    iou_scores = []
    for cls in range(num_classes):
        pred_cls = (preds == cls).float()
        target_cls = (target == cls).float()
        
        inter = (pred_cls * target_cls).sum(dim=(1, 2))
        union = pred_cls.sum(dim=(1, 2)) + target_cls.sum(dim=(1, 2)) - inter + eps
        iou_cls = (inter + eps) / union
        iou_scores.append(iou_cls)
    
    return torch.stack(iou_scores).mean(dim=0).mean()

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = r
        self.scaling = (alpha / r) if r and r > 0 else 1.0
        if r and r > 0:
            self.lora_A = nn.Parameter(torch.zeros(base.in_features, r))
            self.lora_B = nn.Parameter(torch.zeros(r, base.out_features))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
            self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        else:
            self.register_parameter('lora_A', None)
            self.register_parameter('lora_B', None)
            self.dropout = nn.Identity()

    @property
    def in_features(self):
        return self.base.in_features
    
    @property
    def out_features(self):
        return self.base.out_features
    
    @property
    def weight(self):
        return self.base.weight
    
    @property
    def bias(self):
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.r and self.r > 0:
            y = y + self.scaling * (self.dropout(x) @ self.lora_A @ self.lora_B)
        return y

class SALTLinear(nn.Linear):
    def __init__(
        self, 
        base: nn.Linear,
        rank: int = 8,
        r_lora: int = 8,
        seed: int = 42
    ) -> None:
        super().__init__(base.in_features, base.out_features, bias=base.bias is not None)
        
        self.weight.data.copy_(base.weight.data)
        if base.bias is not None:
            self.bias.data.copy_(base.bias.data)
            
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
            
        torch.manual_seed(seed)
        
        self.done_svd = False
        self.U, self.S, self.Vt = self._initialize_svd()
        
        max_possible_rank = min(self.U.shape[1], self.S.shape[0], self.Vt.shape[0])
        print(f"\nLayer size: {base.in_features}x{base.out_features}")
        print(f"Max possible rank: {max_possible_rank}")
        print(f"Using rank: {rank}, r_lora: {r_lora}")
        
        scale_shift_params = rank * 2
        lora_params = (max_possible_rank - rank) * r_lora * 2
        total_params = scale_shift_params + lora_params
        print(f"Scale/shift parameters: {scale_shift_params}")
        print(f"LoRA parameters: {lora_params}")
        print(f"Total trainable parameters: {total_params}")

        self.rank = rank
        self.r_lora = r_lora
        
        self.trainable_scale_A = nn.Parameter(torch.ones(rank))
        self.trainable_shift_B = nn.Parameter(torch.zeros(rank))
        
        remaining_rank = max_possible_rank - rank
        self.trainable_X = nn.Parameter(torch.randn(remaining_rank, r_lora) * 0.01)
        self.trainable_Y = nn.Parameter(torch.randn(r_lora, remaining_rank) * 0.01)
        self._verify_parameters()

    def _verify_parameters(self):
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"\nVerifying SALTLinear parameters:")
        print(f"Trainable parameters: {trainable_params}")
        print(f"Total parameters: {total_params}")
        for name, param in self.named_parameters():
            print(f"{name}: {param.shape} (trainable: {param.requires_grad})")

    def _initialize_svd(self):
        return torch.linalg.svd(self.weight, full_matrices=False)

    def perform_svd(self) -> None:
        self.U, self.S, self.Vt = self._initialize_svd()
        self.done_svd = True

    def get_modified_singular_values(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        S_diag = torch.diag(self.S)
        
        top_s = self.S[:self.rank]
        modified_top_s = self.trainable_scale_A * top_s + self.trainable_shift_B
        
        loRA_term = self.trainable_X @ self.trainable_Y
        
        new_s = S_diag.clone()
        new_s[:self.rank, :self.rank] = torch.diag(modified_top_s)
        new_s[self.rank:, self.rank:] += loRA_term
        
        scale_shift_term = torch.zeros_like(S_diag)
        scale_shift_term[:self.rank, :self.rank] = torch.diag(modified_top_s) - torch.diag(top_s)
        
        return new_s, scale_shift_term, loRA_term

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self.done_svd:
            self.perform_svd()

        new_s, scale_shift_term, LoRA_term = self.get_modified_singular_values()
        s_new = F.relu(new_s.to(input.device))

        weight_updated = self.U @ s_new @ self.Vt
        
        return F.linear(input, weight_updated, self.bias)

def inject_lora_into_attention(model: nn.Module, r=8, alpha=16, dropout=0.05):
    targets = ("attn", "attention", "qkv", "q_proj", "k_proj", "v_proj", "query", "key", "value", "proj", "out")
    replaced = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and any(t in child_name.lower() or t in name.lower() for t in targets):
                setattr(module, child_name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
                replaced += 1
    print(f"[LoRA] wrapped {replaced} linear layers (r={r}, alpha={alpha}, p={dropout})")

def inject_salt_into_attention(model: nn.Module, rank=8, r_lora=8, seed=42):
    targets = ("attn", "attention", "qkv", "q_proj", "k_proj", "v_proj", "query", "key", "value", "proj", "out")
    replaced = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and any(t in child_name.lower() or t in name.lower() for t in targets):
                setattr(module, child_name, SALTLinear(child, rank=rank, r_lora=r_lora, seed=seed))
                replaced += 1
    print(f"[SALT] wrapped {replaced} linear layers (rank={rank}, r_lora={r_lora})")

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_percentage = 100 * trainable_params / total_params if total_params > 0 else 0
    return total_params, trainable_params, trainable_percentage

def train_one_epoch(model, train_loader, optimizer, device, num_classes=1, dice_thr=0.5, vis_dir=None, epoch=0, wandb_run=None):
    model.train()
    total_loss = 0.0
    dice_scores, iou_scores = [], []
    criterion = nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss()
    first_batch_logged = False
    pbar = tqdm(train_loader, desc=f"[Train e{epoch}]")
    
    for step, (inputs, targets, _) in enumerate(pbar):
        inputs  = inputs.to(device)
        targets = targets.to(device)
        
        if wandb_run and step == 0 and epoch == 1:
            print("Targets,min:", targets.min().item(),
                  "max:", targets.max().item(),
                  "dtype:", targets.dtype,
                  "unique:", torch.unique(targets))
            print("Inputs, min:", inputs.min().item(),
                  "max:", inputs.max().item(),
                  "dtype:", inputs.dtype)
                  
        optimizer.zero_grad()
        logits = model(inputs)
        if num_classes == 1:
            loss = criterion(logits, targets)
        else:
            loss = criterion(logits, targets.squeeze(1).long())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        with torch.no_grad():
            if num_classes == 1:
                dice = dice_binary_torch(logits, targets, thresh=dice_thr).mean().item()
                iou  = iou_binary_torch(logits, targets, thresh=dice_thr).mean().item()
            else:
                dice = dice_multi_class_torch(logits, targets, num_classes).item()
                iou  = iou_multi_class_torch(logits, targets, num_classes).item()
            dice_scores.append(dice)
            iou_scores.append(iou)
            
        pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{dice:.4f}", iou=f"{iou:.4f}")
        
        if wandb_run and step % 10 == 0:
            wandb_run.log({
                "train_batch_loss": loss.item(),
                "train_batch_dice": dice,
                "train_batch_iou": iou,
                "step": step + len(train_loader) * (epoch - 1)
            })
            
        if (not first_batch_logged) and vis_dir is not None:
            save_train_visuals(epoch, inputs, logits, targets, out_dir=vis_dir, max_save=8, thr=dice_thr, num_classes=num_classes)
            first_batch_logged = True
            
    avg_loss = total_loss / max(1, len(train_loader))
    avg_dice = float(np.mean(dice_scores)) if len(dice_scores) > 0 else 0.0
    avg_iou  = float(np.mean(iou_scores))  if len(iou_scores)  > 0 else 0.0
    print(f"[Train Epoch {epoch}] loss={avg_loss:.4f}  dice={avg_dice:.4f}  iou={avg_iou:.4f}")
    
    return avg_loss, avg_dice, avg_iou

@torch.no_grad()
def evaluate(model, val_loader, device, num_classes=1, dice_thr=0.5, vis_dir=None, epoch=0, wandb_run=None):
    model.eval()
    total_loss = 0.0
    dice_scores, iou_scores = [], []
    criterion = nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss()
    idx_global = 0
    pbar = tqdm(val_loader, desc="[Eval]")
    
    for batch_idx, (inputs, targets, _) in enumerate(pbar):
        inputs  = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        if num_classes == 1:
            loss = criterion(logits, targets)
        else:
            loss = criterion(logits, targets.squeeze(1).long())
        total_loss += loss.item()
        
        if num_classes == 1:
            dice = dice_binary_torch(logits, targets, thresh=dice_thr).mean().item()
            iou  = iou_binary_torch(logits, targets, thresh=dice_thr).mean().item()
        else:
            dice = dice_multi_class_torch(logits, targets, num_classes).item()
            iou  = iou_multi_class_torch(logits, targets, num_classes).item()
        dice_scores.append(dice)
        iou_scores.append(iou)
        
        pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{dice:.4f}", iou=f"{iou:.4f}")
        
        if vis_dir is not None:
            os.makedirs(vis_dir, exist_ok=True)
            B = inputs.size(0)
            for b in range(B):
                save_eval_visuals(idx_global, inputs[b], logits[b], targets[b], out_dir=vis_dir, thr=dice_thr, fname_prefix="val", num_classes=num_classes)
                idx_global += 1
                
    avg_loss = total_loss / max(1, len(val_loader))
    avg_dice = float(np.mean(dice_scores)) if len(dice_scores) > 0 else 0.0
    avg_iou  = float(np.mean(iou_scores))  if len(iou_scores)  > 0 else 0.0
    print(f"[Eval] loss={avg_loss:.4f}  dice={avg_dice:.4f}  iou={avg_iou:.4f}")
    
    return avg_loss, avg_dice, avg_iou

def main():
    import argparse
    import random
    from dataset import FolderDataset, ResizeAndNormalize
    from dpt import DPT
    
    try:
        import wandb
        WANDB_AVAILABLE = True
    except ImportError:
        print("Wandb not available. Install with: pip install wandb")
        WANDB_AVAILABLE = False

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./segdata")
    parser.add_argument("--dataset", type=str, default="tn3k")
    parser.add_argument("--img_ext", type=str, default=".png")
    parser.add_argument("--mask_ext", type=str, default=".jpg")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input_h", type=int, default=256)
    parser.add_argument("--input_w", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--in_ch", type=int, default=1)
    parser.add_argument("--repo_dir", type=str, default="./dinov3")
    parser.add_argument("--dino_ckpt", type=str, required=True)
    parser.add_argument("--dino_size", type=str, default="b", choices=["b", "s"])
    parser.add_argument("--last_layer_idx", type=int, default=-1)
    parser.add_argument("--vis_max_save", type=int, default=8)
    parser.add_argument("--img_dir_name", type=str, default="images")
    parser.add_argument("--label_dir_name", type=str, default="masks")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--adapter", type=str, default="lora", choices=["lora", "salt", "none"])
    parser.add_argument("--salt_rank", type=int, default=8)
    parser.add_argument("--salt_r_lora", type=int, default=8)
    parser.add_argument("--wandb_project", type=str, default="segdino")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

 
    wandb_run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name or f"segdino_{args.dino_size}_{args.dataset}_{args.adapter}",
        config=vars(args)
    )
    print(f"[Wandb] Initialized run: {wandb_run.name}")
  
    save_root = f"./runs/segdino_{args.dino_size}_{args.input_h}_{args.dataset}_{args.adapter}"
    os.makedirs(save_root, exist_ok=True)
    train_vis_dir = os.path.join(save_root, "train_vis")
    val_vis_dir = os.path.join(save_root, "val_vis")
    ckpt_dir = os.path.join(save_root, "ckpts")
    os.makedirs(train_vis_dir, exist_ok=True)
    os.makedirs(val_vis_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    if args.dino_size == "b":
        backbone = torch.hub.load(args.repo_dir, 'dinov3_vitb16', source='local', weights=args.dino_ckpt)
    else:
        backbone = torch.hub.load(args.repo_dir, 'dinov3_vits16', source='local', weights=args.dino_ckpt)

    for p in backbone.parameters():
        p.requires_grad = False
        
    if args.adapter == "lora" and args.lora_r > 0:
        inject_lora_into_attention(backbone, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
        print(f"[LoRA] enabled (r={args.lora_r}, alpha={args.lora_alpha}, p={args.lora_dropout})")
    elif args.adapter == "salt":
        inject_salt_into_attention(backbone, rank=args.salt_rank, r_lora=args.salt_r_lora, seed=args.seed)
        print(f"[SALT] enabled (rank={args.salt_rank}, r_lora={args.salt_r_lora})")
    else:
        print("[Adapter] disabled")

    model = DPT(nclass=args.num_classes, backbone=backbone)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    total_params, trainable_params, trainable_percentage = count_parameters(model)
    print(f"\n{'='*60}")
    print(f"MODEL PARAMETER SUMMARY")
    print(f"{'='*60}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable percentage: {trainable_percentage:.2f}%")
    print(f"{'='*60}\n")

    if wandb_run:
        wandb_run.config.update({
            "total_params": total_params,
            "trainable_params": trainable_params,
            "trainable_percentage": trainable_percentage
        })

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    root = os.path.join(args.data_dir, args.dataset)
    train_transform = ResizeAndNormalize(size=(args.input_h, args.input_w))
    val_transform   = ResizeAndNormalize(size=(args.input_h, args.input_w))
    
    train_dataset = FolderDataset(
        root=root, 
        split="train", 
        img_dir_name=args.img_dir_name, 
        label_dir_name=args.label_dir_name, 
        mask_ext=args.mask_ext,
        transform=train_transform
    )
    val_dataset = FolderDataset(
        root=root, 
        split="test", 
        img_dir_name=args.img_dir_name, 
        label_dir_name=args.label_dir_name, 
        mask_ext=args.mask_ext,
        transform=val_transform 
    )
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,  batch_size=1, shuffle=False, num_workers=args.num_workers, drop_last=False)

    best_val_dice = -1.0
    best_val_dice_epoch = -1
    best_val_iou  = -1.0
    best_val_iou_epoch  = -1

    for epoch in range(1, args.epochs + 1):
        train_loss, train_dice, train_iou = train_one_epoch(
            model, train_loader, optimizer, device, 
            num_classes=args.num_classes, dice_thr=0.5, 
            vis_dir=train_vis_dir, epoch=epoch, wandb_run=wandb_run
        )
        val_loss, val_dice, val_iou = evaluate(
            model, val_loader, device, 
            num_classes=args.num_classes, dice_thr=0.5, 
            vis_dir=val_vis_dir, epoch=epoch, wandb_run=wandb_run
        )

        if wandb_run:
            wandb_run.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_dice": train_dice,
                "train_iou": train_iou,
                "val_loss": val_loss,
                "val_dice": val_dice,
                "val_iou": val_iou,
                "learning_rate": optimizer.param_groups[0]['lr']
            })

        latest_path = os.path.join(ckpt_dir, "latest.pth")
        torch.save({"epoch": epoch, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}, latest_path)

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_val_dice_epoch = epoch
            best_path = os.path.join(ckpt_dir, f"best_ep{epoch:03d}_dice{val_dice:.4f}_{val_iou:.4f}.pth")
            torch.save(model.state_dict(), best_path)
            print(f"[Save] New best ckpt: {best_path}")
            
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_val_iou_epoch = epoch

    print("=" * 60)
    print(f"[Summary] Best Val Dice = {best_val_dice:.4f} @ epoch {best_val_dice_epoch}")
    print(f"[Summary] Best Val IoU  = {best_val_iou:.4f}  @ epoch {best_val_iou_epoch}")
    print("=" * 60)

    if wandb_run:
        wandb_run.summary.update({
            "best_val_dice": best_val_dice,
            "best_val_iou": best_val_iou,
            "best_val_dice_epoch": best_val_dice_epoch,
            "best_val_iou_epoch": best_val_iou_epoch,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "trainable_percentage": trainable_percentage
        })
        wandb_run.finish()

if __name__ == "__main__":
    main()