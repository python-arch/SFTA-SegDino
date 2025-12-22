## What This Repo Is
SegDINO-style segmentation (DINOv3 ViT backbone + DPT head) with:
- PEFT baselines (LoRA / SALT) for adaptation.
- A research pipeline for **source-free target adaptation** under controlled domain shift using deterministic corruptions.
- A newer track: **learned symbolic mask descriptors** (`E_θ`) + target-time **symbolic alignment** (EMA priors) as self-supervision.

Key docs (kept up-to-date as the project evolves):
- `EXPERIMENT_PLAN.md` (problem definition + reviewer-proof protocol)
- `IMPLEMENTATION_PLAN.md` (staged implementation plan + status tracking)
- `FINDINGS_SO_FAR.md` (running lab notebook: results + reasoning)
- `project_journal/` (versioned snapshots + prioritized pivot plans for sharing/feedback)

## Environment Setup
```bash
conda create -n segdino python=3.10.16
conda activate segdino
pip install -r requirements.txt
```

## DINOv3 Dependency
This code expects:
- A local DINOv3 torch.hub repo (contains `hubconf.py`) via `--repo_dir`
- A DINOv3 pretrained weights `.pth` via `--dino_ckpt`

## Dataset Layout (standardized)
Kvasir-SEG is the primary target for the current source-free adaptation pipeline.

Expected structure (images are `.jpg` on the current machine/cluster):
```
segdata/kvasir/
  train/images/*.jpg
  train/masks/*.jpg
  test/images/*.jpg
  test/masks/*.jpg
```

We further split `test/` into:
- `target_adapt` (unlabeled for adaptation) and
- `target_holdout` (evaluation only),
using committed manifests under `splits/`.

## Reproducible Target Splits
Create deterministic manifests:
```bash
python tools/make_target_splits.py \
  --dataset_root ./segdata/kvasir \
  --base_split test \
  --img_dir_name images \
  --mask_dir_name masks \
  --holdout_ratio 0.2 \
  --seed 42 \
  --out_dir ./splits \
  --prefix kvasir
```

## Deterministic Corruptions (severity knob)
Single-family and mixed corruptions are implemented with deterministic per-image seeding.

Preview a corruption on a single image:
```bash
python tools/preview_corruption.py \
  --image ./segdata/kvasir/test/images/<any>.jpg \
  --out ./runs/corruption_preview.jpg \
  --family mixed \
  --severity 4 \
  --num_ops 4
```

## Source-only Degradation Curves
Evaluate a trained segmentation checkpoint across severities:
```bash
python tools/eval_corruption_curve.py \
  --dataset_root ./segdata/kvasir \
  --manifest ./splits/kvasir_target_holdout.txt \
  --family mixed \
  --num_ops 4 \
  --max_severity 4 \
  --ckpt <seg_checkpoint.pth> \
  --dino_ckpt <dinov3_weights.pth> \
  --dino_size s \
  --repo_dir <dinov3_repo_dir> \
  --out_csv ./runs/source_only_mixed_ops4_curve.csv
```

## Baseline Source-free Adaptation (entropy/consistency/self-train/tent)
Run the full baseline suite (adapts on `target_adapt`, evaluates on `target_holdout`):
```bash
bash tools/run_baseline_suite.sh \
  --dataset_root ./segdata/kvasir \
  --adapt_manifest ./splits/kvasir_target_adapt.txt \
  --eval_manifest ./splits/kvasir_target_holdout.txt \
  --corruption mixed \
  --severity 4 \
  --num_ops 4 \
  --corruption_id v1 \
  --ckpt <seg_checkpoint.pth> \
  --dino_ckpt <dinov3_weights.pth> \
  --dino_size s \
  --repo_dir <dinov3_repo_dir> \
  --steps 500 \
  --batch_size 4 \
  --lr 1e-4 \
  --num_workers 4 \
  --out_csv ./runs/adapt_baselines_mixed_ops4_S4.csv
```

## PEFT-only Baselines (LoRA / SALT)
PEFT is treated as a plug-and-play axis (same adaptation runner, same corruption regime).

Example: LoRA on `mixed ops4 S4`:
```bash
bash tools/run_baseline_suite.sh \
  --dataset_root ./segdata/kvasir \
  --adapt_manifest ./splits/kvasir_target_adapt.txt \
  --eval_manifest ./splits/kvasir_target_holdout.txt \
  --corruption mixed \
  --severity 4 \
  --num_ops 4 \
  --ckpt <seg_checkpoint.pth> \
  --dino_ckpt <dinov3_weights.pth> \
  --repo_dir <dinov3_repo_dir> \
  --steps 500 \
  --batch_size 4 \
  --lr 1e-4 \
  --adapter lora \
  --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
  --out_csv ./runs/adapt_peft_lora_mixed_ops4_S4.csv
```

Example: SALT on `mixed ops4 S4`:
```bash
bash tools/run_baseline_suite.sh \
  --dataset_root ./segdata/kvasir \
  --adapt_manifest ./splits/kvasir_target_adapt.txt \
  --eval_manifest ./splits/kvasir_target_holdout.txt \
  --corruption mixed \
  --severity 4 \
  --num_ops 4 \
  --ckpt <seg_checkpoint.pth> \
  --dino_ckpt <dinov3_weights.pth> \
  --repo_dir <dinov3_repo_dir> \
  --steps 500 \
  --batch_size 4 \
  --lr 1e-4 \
  --adapter salt \
  --salt_rank 8 --salt_r_lora 8 --salt_seed 42 \
  --out_csv ./runs/adapt_peft_salt_mixed_ops4_S4.csv
```

## Learned Symbolic Descriptor Encoder (`E_θ`)
Train a tiny mask encoder on source masks (`train/masks`) using structure-preserving augmentations:
```bash
python tools/train_symbolic_encoder.py \
  --dataset_root ./segdata/kvasir \
  --out_dir ./runs/symalign_encoder_kvasir \
  --epochs 50 \
  --batch_size 64 \
  --lr 1e-3 \
  --num_workers 4 \
  --out_h 256 --out_w 256 \
  --boundary_width 2 \
  --max_morph_radius 2 \
  --embed_dim 64 \
  --width 32
```
Outputs: `./runs/symalign_encoder_kvasir/encoder_final.pth`

## Learned Symbolic Alignment (EMA priors)
Run symbolic alignment on target during adaptation (example: `tent` core objective, no PEFT):
```bash
python tools/adapt_baselines.py \
  --dataset_root ./segdata/kvasir \
  --adapt_manifest ./splits/kvasir_target_adapt.txt \
  --eval_manifest ./splits/kvasir_target_holdout.txt \
  --corruption mixed \
  --severity 4 \
  --num_ops 4 \
  --method tent \
  --steps 500 \
  --batch_size 4 \
  --lr 1e-4 \
  --teacher_kl_weight 1.0 \
  --use_symbolic \
  --symbolic_ckpt ./runs/symalign_encoder_kvasir/encoder_final.pth \
  --symbolic_lambda 0.1 \
  --symbolic_warmup_steps 150 \
  --symbolic_ema_momentum 0.99 \
  --symbolic_conf_thr 0.8 \
  --ckpt <seg_checkpoint.pth> \
  --dino_ckpt <dinov3_weights.pth> \
  --repo_dir <dinov3_repo_dir> \
  --out_csv ./runs/adapt_symbolic_none_mixed_ops4_S4.csv
```

## Direction-4 Pivot: Multi-Modal Symbolic Descriptors (mask + image)
Train a multi-modal symbolic encoder that fuses mask structure + feature-pooled appearance (soft region pooling):
```bash
python tools/train_multimodal_encoder.py \
  --dataset_root ./segdata/kvasir \
  --mask_encoder_ckpt ./runs/symalign_encoder_kvasir/encoder_final.pth \
  --out_dir ./runs/symalign_multimodal_encoder_kvasir \
  --epochs 20 \
  --batch_size 32 \
  --lr 1e-4 \
  --num_workers 4 \
  --out_h 256 \
  --out_w 256 \
  --boundary_width 2 \
  --embed_dim 64 \
  --image_encoder resnet18 \
  --image_weights auto \
  --image_pool features \
  --pool_dilate_px 8 \
  --fusion uncertainty \
  --stopgrad_cross \
  --modality_dropout_p 0.2 \
  --mask_perturb_p 0.5 \
  --max_mask_morph_radius 2
```
`--image_weights auto` will fetch ImageNet-pretrained torchvision weights when available and cache them under `~/.cache/symalign_torchvision` (override via `SYALIGN_TORCHVISION_CACHE`).
If you don’t have `torchvision` installed (or want a fully offline path), use `--image_encoder small_cnn --image_weights none`.

Use it during adaptation:
```bash
python tools/adapt_baselines.py \
  --dataset_root ./segdata/kvasir \
  --adapt_manifest ./splits/kvasir_target_adapt.txt \
  --eval_manifest ./splits/kvasir_target_holdout.txt \
  --corruption mixed \
  --severity 4 \
  --num_ops 4 \
  --corruption_id v1 \
  --method tent \
  --steps 500 \
  --batch_size 4 \
  --lr 1e-4 \
  --num_workers 4 \
  --teacher_kl_weight 1.0 \
  --adapter none \
  --use_symbolic \
  --symbolic_mode multimodal \
  --multimodal_ckpt ./runs/symalign_multimodal_encoder_kvasir/encoder_final.pth \
  --multimodal_output fused \
  --multimodal_prior_mode triple \
  --multimodal_w_fused 1.0 \
  --multimodal_w_mask 0.5 \
  --multimodal_w_image 0.5 \
  --symbolic_prior_type memory \
  --symbolic_mem_capacity 1024 \
  --symbolic_mem_min 64 \
  --symbolic_outlier_cos 0.1 \
  --symbolic_min_fg 0.001 \
  --symbolic_max_fg 0.60 \
  --symbolic_max_components 6 \
  --symbolic_lambda 0.1 \
  --symbolic_warmup_steps 150 \
  --symbolic_ema_momentum 0.99 \
  --symbolic_conf_thr 0.8 \
  --ckpt <seg_checkpoint.pth> \
  --dino_ckpt <dinov3_weights.pth> \
  --repo_dir <dinov3_repo_dir> \
  --out_csv ./runs/adapt_symbolic_multimodal_none_mixed_ops4_S4.csv
```

Evaluate the multi-modal encoder invariance (two-view retrieval on the source split):
```bash
python tools/eval_multimodal_encoder.py \
  --dataset_root ./segdata/kvasir \
  --split train \
  --img_dir_name images \
  --mask_dir_name masks \
  --ckpt ./runs/symalign_multimodal_encoder_kvasir/encoder_final.pth \
  --batch_size 32 \
  --num_workers 4 \
  --max_items 512 \
  --seed 42 \
  --out_h 256 \
  --out_w 256 \
  --boundary_width 2
```

## Current findings (high-level)
See `FINDINGS_SO_FAR.md` for full results and reasoning. Key points so far:
- Single-family blur/JPEG/illumination at `S0..S4` are not severe enough; **mixed corruptions** (especially `num_ops=4`) produce a strong stress regime.
- Naive source-free baselines can collapse to all-background; stabilizers (frozen teacher KL, restricted trainable scope) prevent collapse.
- PEFT-only baselines show LoRA and SALT are both viable; SALT can be more parameter-efficient.
- Learned symbolic alignment is currently close to TENT baseline; tuning is ongoing, and “symbolic-only with KL removed” collapses (entropy degeneracy).

### Useful numbers (single seed, holdout n=40)
All numbers below are for `mixed` corruptions with `corruption_id=v1`, adaptation on `target_adapt`, evaluation on `target_holdout`.

**Non-PEFT baselines (no symbolic)**
- Moderate regime (`mixed ops2 S4`): `tent` Dice=0.8362, IoU=0.7420, BoundaryF=0.3785, HD95=26.15
- Stress regime (`mixed ops4 S4`, clean baseline CSV): `tent` Dice=0.7372, IoU=0.6173, BoundaryF=0.2291, HD95=46.66

**PEFT-only baselines (no symbolic)**
- Stress regime (`mixed ops4 S4`):
  - LoRA best Dice (entropy): Dice=0.7418, IoU=0.6202, BoundaryF=0.2375, HD95=47.72 (trainable≈0.92%)
  - SALT best Dice (entropy): Dice=0.7396, IoU=0.6197, BoundaryF=0.2355, HD95=46.25 (trainable≈0.60%)
- Moderate regime (`mixed ops2 S4`):
  - LoRA best Dice (entropy): Dice=0.8381, IoU=0.7446, BoundaryF=0.3739, HD95=26.18
  - SALT best Dice (entropy): Dice=0.8383, IoU=0.7442, BoundaryF=0.3788, HD95=26.08

**Learned symbolic alignment (Eθ + EMA priors)**
- Tuned setting for stress regime (`tent`, no PEFT): `symbolic_lambda=0.1`, `warmup_steps=150`, `conf_thr=0.8`
  - `mixed ops4 S4`: Dice=0.7376, IoU=0.6178, BoundaryF=0.2274, HD95=46.36
- Plug-and-play check (same tuned symbolic settings, `tent` core):
  - `mixed ops4 S4` + LoRA: Dice=0.7317, IoU=0.6113, BoundaryF=0.2285, HD95=47.30
  - `mixed ops4 S4` + SALT: Dice=0.7349, IoU=0.6145, BoundaryF=0.2270, HD95=47.03
- Failure mode ablation: setting `--teacher_kl_weight 0.0` causes all-background collapse (empty_rate=1.0), even with symbolic enabled.

### Current blocker (important for feedback)
- Under the stress regime (`mixed ops4 S4`), removing the frozen-teacher stabilizer (`--teacher_kl_weight 0.0`) collapses to all-background, even with symbolic alignment enabled.
- Current interpretation: symbolic alignment is contributing a **structure/boundary preservation signal**, but it does **not yet** serve as a standalone anti-collapse mechanism.

### Next experiments (small, high-signal)
- **Teacher KL annealing:** start with KL>0 early, decay it toward 0 late; test whether symbolic loss can “take over” without collapse.
- **No-teacher constraints:** explicit anti-collapse penalties (foreground-mass bounds, smoothness/TV, fragmentation proxies), evaluated under `mixed ops4 S4`.
