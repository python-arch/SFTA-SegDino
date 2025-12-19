# Symbolic Alignment for Source-Free Segmentation Adaptation (SegDINO + PEFT)

## Problem Definition
We study **source-free, unlabeled target adaptation** for polyp segmentation (Kvasir-SEG) with a SegDINO-style model (DINOv3 ViT backbone + DPT head).

- **Source training:** uses labeled source-domain data (here: clean Kvasir-SEG `train/`).
- **Target adaptation:** uses **only unlabeled target images** (no source images, no source labels, no source statistics).
- **Evaluation:** uses a **held-out labeled target set** never used for adaptation/tuning.

### Core capability claim (refined)
Enable **source-free target adaptation** under **severe appearance shift** where pseudo-label self-training is unreliable, and **preserve mask structure/boundaries** during adaptation by aligning **learned mask-structure descriptors** during PEFT adaptation (LoRA/SALT and other adapters).

Important nuance (based on current findings):
- The current source-free baselines require an explicit anti-collapse stabilizer (teacher KL / constraints) under severe shift; learned symbolic alignment **does not yet replace** that stabilizer by itself.
  - Current positioning for feedback: symbolic alignment is a **structure/boundary preservation signal** that complements standard anti-collapse stabilizers; replacing the stabilizer is an explicit research question (see “Teacher dependence” experiments).

## Experimental Protocol (Reviewer-Proof)
### Data contract (standardized)
Dataset layout (canonical):
- `segdata/kvasir/train/images/*`, `segdata/kvasir/train/masks/*`
- `segdata/kvasir/test/images/*`, `segdata/kvasir/test/masks/*` (provided split)

We further split the provided `test/` once (fixed seed; filelists committed):
- `target_adapt/` (unlabeled, used for adaptation and development iteration)
- `target_holdout/` (labeled, **evaluation only**, never used for adaptation or model selection)

Default: `target_holdout = 20%` of original `test/` (min 15%, max 25%).

### Target shift construction (severity knob)
We construct target domains by applying corruptions to `target_adapt/images` and `target_holdout/images` with **deterministic per-image seeds** (seed = hash(filename, corruption_id, severity)).

#### Stage 1 (main paper): single-family ladders
We run separate severity ladders `S0..S4` for each corruption family:
- Blur
- Noise
- JPEG/compression
- Illumination (brightness/contrast/gamma)

Headline reporting aggregates across families (average AUSC and S4; also provide per-family breakdown).

#### Stage 2 (appendix): mixed corruptions
Per image, compose 1–2 corruptions sampled deterministically (same seed scheme), to emulate “in-the-wild” shifts.

**Implementation note (current code):** mixed corruptions support `num_ops` up to 4 families and severity up to `S8`. The paper’s main ladder remains `S0..S4` unless explicitly extended.

### “Source-free” enforcement
During adaptation:
- No access to `segdata/kvasir/train/*`
- No access to `target_holdout/images`
- No labels used for any selection

## Methods
### Model family
SegDINO-like model: DINOv3 backbone + DPT head (binary segmentation).

Primary adaptation scope (to keep early experiments tight):
- Apply PEFT to the **segmentation head first** (and only expand scope to backbone projections after this is stable).

### Baseline groups (non-negotiable)
1. **No adaptation**
   - Source-only (train on clean, evaluate on target severities)
2. **Source-free adaptation (non-PEFT)**
   - Entropy minimization
   - Augmentation consistency
   - **TENT-style** adaptation (update norm affine params, entropy objective)
3. **Pseudo-label baseline**
   - Self-training with confidence threshold + augmentation (included to demonstrate failure under severe shift)
4. **PEFT-only (non-symbolic)**, adapter ∈ {LoRA, SALT, …}
   - Same best non-symbolic objective as above; update only adapter params
5. **Ours: PEFT + symbolic alignment**, adapter ∈ {LoRA, SALT, …}
   - Learned-symbolic alignment loss + anti-collapse safeguards + same adaptation budget

### PEFT is a pluggable axis
Primary comparisons:
- LoRA vs SALT under **matched trainable parameter budgets**
Secondary (optional): other PEFTs (e.g., IA³/adapters/BitFit) as long as parameter budgets are matched.

### Symbolically-aligned reparameterization (SALT-like variants)
We keep LoRA and SALT as the **primary PEFT axis** for the first milestone (clean, budget-matched, reviewer-expected).

**Phase-2 / optional (after the core symbolic method shows clear gains):** test SALT-like “symbolically aligned reparameterizations” as ablations to understand how update-space design interacts with symbolic alignment:
- **SALT-G (symbolic-gradient gated subspace):** use symbolic-loss signals to decide which layers/components update (others frozen).
- **SALT-S/A (structure vs appearance spectrum split):** partition the singular spectrum into “structure” vs “appearance” components and couple them to different losses (e.g., `L_sym` vs `L_core`).
- **SALT-Dial (domain dial):** learn a small set of per-layer scalars that modulate adaptation strength; symbolic loss supervises the dial to avoid over-adaptation.

These are treated strictly as **ablation variants** of the reparameterized update space (not new datasets or new supervision), and should be de-scoped if they distract from the main claim.

**Scope note (phase-1 vs phase-2):**
- Phase-1: validate learned-symbolic alignment + PEFT plug-and-play (LoRA/SALT) with strong baselines.
- Phase-2: implement and compare SALT-G / SALT-S/A / SALT-Dial (keep as ablations / follow-up unless a clear winner emerges).

## Symbolic Alignment (what “symbolic” means)
We replace hand-crafted shape/topology statistics with a **learned mask-structure descriptor encoder** `E_θ`.

### Learnable symbolic encoder `E_θ` (trained once; then frozen)
`E_θ` maps a mask view to a `k`-dim descriptor:
- Input views:
  - Soft mask `p` (probabilities/logits passed through sigmoid)
  - Boundary view `b(p)` (e.g., boundary band / mask-gradient magnitude / thin edge map)
- Outputs:
  - Global descriptor: `s_g = E_θ(p)`
  - Boundary descriptor: `s_b = E_θ(b(p))` (same encoder, different input view)

Training of `E_θ` happens **once before target adaptation** using **source-domain ground truth masks** from `train/`:
- Use structure-preserving transforms (flip/rotate, resize/crop with aligned remap, conservative morphological perturbations).
- Objective: contrastive (positives = same mask under different transforms; negatives = different masks in batch) or BYOL-style.
- Freeze `E_θ` during target adaptation (first paper version) to reduce instability and avoid leakage concerns.

### Symbolic prior (source-free; target-only)
Primary: **self-bootstrapped EMA prior** on target descriptors:
- Maintain EMA summaries over confident, non-degenerate predictions on `target_adapt/`, separately for:
  - Global: `(μ_g, Σ_g)` (mean + variance/cov)
  - Boundary: `(μ_b, Σ_b)`

Optional: task-knowledge priors (soft ranges) as secondary experiments, if defensible.

### Alignment loss (two-scale)
For each target prediction `p_t`, compute `(s_g(p_t), s_b(p_t))` and align to EMA priors via a robust distance:
- Huber on z-scored deviations, or clipped Mahalanobis distance.

`L_sym = D(s_g, μ_g, Σ_g) + D(s_b, μ_b, Σ_b)`

### Core adaptation objective (non-symbolic)
Primary choice (locked): **augmentation consistency** on unlabeled target images (entropy minimization remains a baseline).

### Anti-collapse safeguards (explicit, separable from symbols)
Non-negotiable safeguards:
- Confidence gating (pixel-/image-level) + warmup/ramp schedule
- Robust loss on symbolic deviations (outlier clipping / Huber / quantile-style)
- Explicit penalties/guards for degenerate masks (all-foreground / all-background)

## Causality Ablations (separate “symbols” vs “safeguards”)
To avoid “it’s just regularization” critiques, include:
- **Safeguards-only** (no symbolic terms)
- **Symbols-only** (no safeguards; expected to collapse; report failure rates)
- **Symbols + safeguards** (full)
- Global-only (remove boundary descriptor)
- Boundary-only (remove global descriptor)

Reparameterization ablations (SALT family):
- SALT baseline vs SALT-G vs SALT-S/A vs SALT-Dial under the same objective/budget
- Where applicable: disable gating/splitting/dial to isolate effect

Run ablations at least on `S2` and `S4`, plus include in AUSC if feasible.

## Stability Ablations (teacher dependence)
Given the known degeneracy of entropy-based adaptation in binary segmentation, include explicit stability ablations:
- `teacher_kl_weight > 0` vs `teacher_kl_weight = 0` (expected collapse under severe shift)
- KL annealing schedule (start >0 then decay toward 0) to test if symbolic alignment can take over late in adaptation
- Explicit collapse penalties/constraints (foreground mass bounds, fragmentation proxies) vs none

## Parameter/Compute Budgets (claims)
Report trainable parameters and time.

Default budgets:
- Main: **0.5% trainable params**
- Curves: **0.1%**, **1%**

All PEFT comparisons must be budget-matched.

## Metrics and Reporting
Primary:
- Dice, IoU

Boundary:
- **Boundary F-score** (primary for endoscopy-style segmentation)
- **HD95** (secondary; framed as robust boundary error under shift)

Interpretation note:
- We expect a Dice–boundary trade-off for some baselines (e.g., TENT can improve Dice while hurting boundaries); part of the goal is to **preserve boundaries/structure** without sacrificing Dice.

Stability / failure:
- Empty prediction rate, full prediction rate
- Fragmentation proxy (components/fragmentation)

Headline reporting:
- **AUSC** over `S1–S4`
- Stress-test: **S4** results
- Keep `S0` as sanity check (not the headline)

Reproducibility:
- 3 seeds minimum for headline tables
- Fixed filelists for splits and fixed corruption spec (IDs + severity mapping committed)

## Reproducibility artifacts (what to run)
This section records the concrete tooling used to instantiate the protocol.

### Split generation
- Generate deterministic manifests:
  - `python tools/make_target_splits.py --dataset_root ./segdata/kvasir --base_split test --img_dir_name images --mask_dir_name masks --holdout_ratio 0.2 --seed 42 --out_dir ./splits --prefix kvasir`
- Outputs:
  - `splits/kvasir_target_adapt.txt`
  - `splits/kvasir_target_holdout.txt`
  - `splits/kvasir_target_splits.json`

### Corruption preview (sanity)
- `python tools/preview_corruption.py --image <path/to/img.jpg> --out ./runs/preview.jpg --family mixed --severity 4 --num_ops 4`

### Source-only degradation curves
- Evaluate a fixed checkpoint across severities:
  - `python tools/eval_corruption_curve.py --dataset_root ./segdata/kvasir --manifest ./splits/kvasir_target_holdout.txt --family noise --max_severity 4 --ckpt <seg_ckpt.pth> --dino_ckpt <dinov3.pth> --dino_size s --repo_dir ./dinov3 --out_csv ./runs/source_only_noise_curve.csv`

### Pseudo-label quality diagnostics (stress regime justification)
- Per-image confidence/entropy proxies (e.g. `mixed ops4 S4`):
  - `python tools/pseudolabel_quality.py --dataset_root ./segdata/kvasir --manifest ./splits/kvasir_target_holdout.txt --family mixed --num_ops 4 --severity 4 --ckpt <seg_ckpt.pth> --dino_ckpt <dinov3.pth> --dino_size s --repo_dir ./dinov3 --out_csv ./runs/pseudolabel_quality_mixed_ops4_S4.csv`

### Source-free baseline adaptation (adapt on target_adapt, eval on target_holdout)
- Single baseline run:
  - `python tools/adapt_baselines.py --dataset_root ./segdata/kvasir --adapt_manifest ./splits/kvasir_target_adapt.txt --eval_manifest ./splits/kvasir_target_holdout.txt --corruption mixed --severity 4 --num_ops 4 --method consistency --steps 500 --batch_size 4 --lr 1e-4 --ckpt <seg_ckpt.pth> --dino_ckpt <dinov3.pth> --repo_dir ./dinov3 --out_csv ./runs/adapt_baselines_mixed_ops4_S4.csv`
- Full baseline suite (entropy/consistency/selftrain/tent):
  - `bash tools/run_baseline_suite.sh --dataset_root ./segdata/kvasir --adapt_manifest ./splits/kvasir_target_adapt.txt --eval_manifest ./splits/kvasir_target_holdout.txt --corruption mixed --severity 4 --num_ops 4 --ckpt <seg_ckpt.pth> --dino_ckpt <dinov3.pth> --repo_dir ./dinov3 --out_csv ./runs/adapt_baselines_mixed_ops4_S4.csv`

## Model selection / stopping (no label leakage)
If any early stopping is used, it must be **unsupervised**, e.g.:
- plateau in target entropy/consistency loss
- stability of symbolic-stat EMA deltas
- “no-collapse” constraints satisfied for N steps

## Minimal Execution Plan (run order)
Phase 1: establish stress regime
1. Train source on clean; evaluate on clean and `S0..S4` per family
2. Identify pseudo-label failure severities (typically S3/S4)

Phase 2: learn the symbolic encoder
3. Train `E_θ` on source masks with structure-preserving transforms; freeze for adaptation
4. Sanity-check `E_θ` invariance qualitatively (same-mask views cluster; different masks separate)

Phase 3: baselines (two severities first)
5. Run non-PEFT baselines on `S2` and `S4`
6. Run PEFT-only LoRA/SALT on `S2` and `S4` (0.5% budget)

Phase 4: ours MVP (S4-first)
7. Ours (learned symbols + safeguards) on `S4` first, then sweep `S1..S4`

Phase 5: ablations + budget curves
8. Causality ablations on `S2` and `S4`
9. Budget curves at `S4` + AUSC for LoRA vs SALT (0.1/0.5/1%)

Phase 6 (optional): symbolic-aligned reparameterization ablations (SALT variants)
10. Compare SALT baseline vs SALT-G vs SALT-S/A vs SALT-Dial on:
   - moderate regime: `mixed ops2 S4`
   - stress regime: `mixed ops4 S4`
   - (optional) severity ladder aggregation (AUSC) on `mixed ops4 S2–S4`

## Teacher dependence experiments (explicit, due to observed collapse)
Current evidence: setting `--teacher_kl_weight 0.0` collapses to all-background under the stress regime (`mixed ops4 S4`), even with symbolic alignment enabled. This is a central discussion point for professor/reviewer feedback.

We therefore plan the following targeted experiments (small, high-signal) before expanding scope:
1. **KL annealing schedule:** start with teacher KL > 0 to prevent collapse, then decay to 0 so symbolic loss “takes over” late.
2. **Explicit anti-collapse constraints (no teacher):** replace teacher KL with constraints that directly penalize degenerate solutions (foreground-mass bounds, TV/boundary smoothness, fragmentation proxy bounds).
3. **Symbols-only vs safeguards-only vs full:** demonstrate causal separation between “symbolic structure signal” and “anti-collapse engineering”.

Success criteria (for the “replace teacher” goal): stable non-degenerate predictions (empty/full rates near 0) and competitive Dice/BoundaryF without teacher KL in the stress regime.

## Research inspirations (optional extensions)
If reviewers/professors push for stronger novelty, the following directions are compatible with this pipeline:
- **Stratified contrastive symbolic learning:** stratify masks by size/complexity during `E_θ` training to avoid the descriptor collapsing to trivial “area only”.
- **Multi-modal symbolic descriptors (image+mask):** fuse mask-structure descriptors with appearance descriptors from the image (e.g., masked pooling of backbone features).
- **Topological descriptors:** topology-aware descriptors (e.g., persistent homology style summaries) to penalize fragmentation/holes in a principled way (higher complexity; phase-2).
- **Equivariant symbolic descriptors:** rotation/scale equivariant encoders to reduce augmentation reliance (higher complexity; phase-2).

## Plan history (for feedback / reproducibility)
We keep a versioned “journal” with snapshots and pivot plans:
- Snapshot of current baselines/PEFT/symbolic status: `project_journal/snapshots/2025-12-19_baselines_peft_symbolic.md`
- Current prioritized pivot plan (Direction 4, multi-modal): `project_journal/plans/2025-12-19_direction4_multimodal_symbolic_alignment.md`

Appendix: mixed corruptions; optional “domain dial” (scale adapter strength at inference).

## Figures Checklist (paper narrative)
1. Protocol schematic: source train → target_adapt → target_holdout (+ severities)
2. Source-only degradation: Dice/IoU vs severity (per family)
3. Descriptor learning sanity: UMAP/t-SNE of `E_θ` outputs (optional but strong)
4. Main results: AUSC and S4 (methods grouped: non-PEFT, PEFT-only, ours)
4. Pseudo-label failure: self-training collapses at S4; ours remains stable (include failure rates)
5. Causality ablations: safeguards-only vs symbols-only vs full (+ term removals)
6. Efficiency curve: performance vs trainable params (LoRA vs SALT)
7. Qualitative grids on S4 (input/pred/gt) + symbolic stats over adaptation steps
