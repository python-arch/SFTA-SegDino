# Plan — 2025-12-19 (Direction 4: Multi-Modal Symbolic Alignment Pivot)

This plan is a prioritized pivot: extend symbolic descriptors from **mask-only** → **mask + image appearance** so the symbolic signal remains informative under severe appearance corruptions.

## Why this pivot (based on current evidence)
- Under `mixed ops4 S4`, symbolic alignment (mask-only) is currently ~tied with TENT and does not outperform PEFT-only.
- Removing teacher KL collapses, indicating we need a stronger “meaningful signal” + stronger anti-collapse story.
- Mixed corruptions target appearance; adding an appearance descriptor provides a complementary channel.

## Scope (in / out)
In scope:
- Implement multi-modal descriptor encoder under `symalign/` (do not create/move anything into a new `segdino/` package).
- Train it on **source train split** (images + GT masks).
- Integrate it into `tools/adapt_baselines.py` as an alternative symbolic backend.
- Run targeted stress tests on `mixed ops4 S4`.
- Add ablations: mask-only vs image-only vs fused.

MICCAI-hardening upgrades (added):
- **Decouple appearance from pred mask** via **feature-map soft region pooling** (not raw `image * mask` pixels).
- **Uncertainty-conditioned fusion** (gating uses prediction uncertainty stats; outputs weights for logging).
- **Guarded memory-bank priors** (strict admission + outlier gate) to prevent drift.
- **Robust descriptor training**: mask-only perturbations + modality dropout + stop-grad cross-modal alignment.

Out of scope (Phase-2):
- Topological persistence / equivariant encoders (higher risk/time).
- “SALT variants” (SALT-G/SALT-S/A/SALT-Dial) beyond LoRA/SALT baselines.

## Deliverables (what to implement)
### New/updated files (planned)
- `symalign/multimodal_encoder.py`
  - appearance from **feature maps** + soft region pooling (with optional mask dilation)
  - `MultiModalSymbolicEncoder` returns `(s_fused, s_mask, s_img)` and can optionally return:
    - fusion weights (`w_mask,w_region,w_global`)
    - uncertainty stats (`conf/entropy/sharpness/area`)
  - fusion supports `mlp|attn|uncertainty` (uncertainty-gated is the MICCAI method)
- `symalign/multimodal_symbolic_loss.py`
  - supports **EMA priors** and **guarded MemoryBank priors**
  - single-stream alignment (fused/mask/image) and triple-stream alignment (fused+mask+image)
- `symalign/multimodal_loss.py`
  - multi-objective contrastive: intra-mask, intra-image, cross-modal, fused consistency
  - optional stop-gradient cross-modal terms for stability
- `tools/train_multimodal_encoder.py`
  - trains multi-modal encoder on source train split (images + GT masks)
  - writes `runs/symalign_multimodal_encoder_kvasir/encoder_final.pth`
  - adds mask-only perturbations and modality dropout options
- `tools/eval_multimodal_encoder.py`
  - encoder sanity check: two-view retrieval / invariance for fused/mask/image streams
- `tools/adapt_baselines.py`
  - add CLI flags to switch symbolic encoder type:
    - `--symbolic_mode mask` (current)
    - `--symbolic_mode multimodal` (new)
    - `--multimodal_ckpt <path>`
  - during adaptation, compute symbolic descriptors using:
    - prediction probs `p` + boundary view (existing) for mask channel
    - `masked_image = image * p` for appearance channel
  - keep EMA priors and confidence gating logic consistent.

### Training objective (MICCAI-hardened)
Train multi-modal encoder on source with:
- structure-preserving geometric augs (shared image/mask)
- photometric/noise augs (image only)
- **mask-only perturbations** (dilation/erosion) + boundary recompute
- **modality dropout** (drop mask/image branch sometimes)
Loss components:
- `L_mask`: contrastive on mask descriptors between two views of same mask
- `L_img`: contrastive on image descriptors between two views of same image region
- `L_cross`: align mask descriptor and image descriptor for same sample (cross-modal agreement), with **stop-grad** option
- `L_fused`: contrastive on fused descriptors between two views

## Evaluation plan (what to run)
### Stage A: encoder sanity (source-side)
1. Train encoder on `train/` (images + GT masks).
2. Validate embeddings qualitatively:
   - two-view top-1 retrieval (same sample id under aug)
   - positive vs negative cosine gap (quick separability proxy)
   - optionally UMAP of descriptors colored by mask area / complexity.

### Stage B: target adaptation stress test (primary)
Run adaptation on `mixed ops4 S4`:
- Baseline: `tent` (no symbolic), `tent` + mask-symbolic, `tent` + multimodal-symbolic
- Evaluate Dice/IoU/BoundaryF/HD95 and failure rates.

### Stage C: ablations (must-have)
- mask-only symbolic vs image-only symbolic vs fused symbolic
- corruption-family breakdown (blur/noise/jpeg/illumination) at S4 if time permits

## Success criteria (decision point)
Primary: beat TENT on stress regime with meaningful margin, and/or improve boundary metrics without harming Dice.
- Target: Dice ≥ 0.755 on `mixed ops4 S4` **or** BoundaryF +0.02 while Dice ≥ baseline.

If it fails:
- fallback plan: Direction 3 (stratified contrastive symbolic learning) to improve mask-only encoder training.

## Exact commands (to be run once implemented)
Placeholders below assume:
- dataset root: `./segdata/kvasir`
- existing mask encoder: `./runs/symalign_encoder_kvasir/encoder_final.pth`

Train multi-modal encoder:
```bash
PYTHONPATH="$(pwd)" python tools/train_multimodal_encoder.py \
  --dataset_root ./segdata/kvasir \
  --mask_encoder_ckpt ./runs/symalign_encoder_kvasir/encoder_final.pth \
  --out_dir ./runs/symalign_multimodal_encoder_kvasir \
  --epochs 20 \
  --batch_size 32 \
  --lr 0.0001 \
  --num_workers 4 \
  --out_h 256 \
  --out_w 256 \
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

Evaluate encoder invariance (two-view retrieval):
```bash
PYTHONPATH="$(pwd)" python tools/eval_multimodal_encoder.py \
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

Adaptation run (stress regime, TENT core, multimodal symbolic):
```bash
PYTHONPATH="$(pwd)" python tools/adapt_baselines.py \
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
  --num_workers 4 \
  --lr 0.0001 \
  --teacher_kl_weight 1.0 \
  --adapter none \
  --use_symbolic \
  --symbolic_mode multimodal \
  --multimodal_ckpt ./runs/symalign_multimodal_encoder_kvasir/encoder_final.pth \
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
  --ckpt /absolute/path/to/best_seg_checkpoint.pth \
  --dino_ckpt /absolute/path/to/dinov3_weights.pth \
  --dino_size s \
  --repo_dir /absolute/path/to/dinov3_repo_dir \
  --out_csv ./runs/adapt_symbolic_multimodal_none_mixed_ops4_S4.csv
```
