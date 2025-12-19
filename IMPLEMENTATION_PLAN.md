# Implementation Plan (Staged) — Symbolic Alignment + Learned Descriptors

This document breaks the experimental protocol in `EXPERIMENT_PLAN.md` into concrete implementation stages, with intended artifacts, interfaces, and validation checks. It is an execution plan only (no code in this step).

## Guiding principles
- **Repro first:** deterministic splits, deterministic corruptions, committed manifests.
- **Source-free strictness:** adaptation code must never read `train/` and must never see `target_holdout/`.
- **Pluggable axes:** PEFT adapter type and loss components (core/symbolic/safeguards) are modular and switchable.
- **Minimal intrusion:** keep existing scripts working where possible; add new modules gradually.

## Repo structure (locked constraint)
We keep the current project layout stable to avoid breaking paths on the machine/cluster.

Important constraint (user request): **do not migrate code into a new `segdino/` package**; if we add a package for symbolic alignment / descriptor learning it should live under a separate name (currently: `symalign/`).

Current structure used by the pipeline:
- Top-level modules (model, adapters, data, corruptions, metrics).
- `tools/`: runnable CLIs (splits, corruption evaluation, adaptation baselines, symbolic encoder training).
- `symalign/`: learned symbolic descriptor encoder + symbolic alignment utilities (kept separate from `segdino`).
- `splits/`: committed target manifest files.
- `segdata/`: dataset root (external, not committed).

## Current repo mapping → staged refactor (keep scripts working)
This repo currently uses top-level scripts and utilities:
- `dataset.py`: folder dataset + resize/normalize
- `dpt.py` / `blocks.py`: segmentation head
- `train_segdino.py`: baseline training
- `PEFT_segdino.py`: PEFT training + (currently) wandb logging
- `test_segdino.py`: evaluation metrics + visualization

### Migration strategy (no breakage)
We will not “big bang” rewrite. Instead:
1. Keep existing entrypoints working by:
   - leaving their CLI intact where possible
   - gradually replacing duplicated internal logic with shared imports (top-level modules and `symalign/` where applicable)
2. Add **new** entrypoints only when needed (e.g., `tools/train_symbolic_encoder.py`, `tools/adapt_baselines.py`) so the baseline scripts remain usable.

Note: `segdino/` exists in this repo (legacy/utilities), but the current plan is to **avoid expanding it**; new symbolic work stays in `symalign/` and new CLIs stay under `tools/`.

### What gets moved/duplicated first vs later
Early stages (unblock experiments):
- `dataset.py` functionality becomes `segdino/data.py` (manifest selection + target splits + view generation), but we can keep `dataset.py` as a thin wrapper initially.
- Corruptions and deterministic seeding are new (`segdino/corruptions.py`) and used by new adaptation/eval paths first.
- Metrics from `test_segdino.py` (Dice/IoU/HD95) become `segdino/metrics.py`; `test_segdino.py` can import them.

Later stages (after results are flowing):
- Consolidate duplicated training loops between `train_segdino.py` and `PEFT_segdino.py` into a shared runner module (optional).
- Normalize logging (CSV + optional wandb) behind a small interface to avoid coupling experiments to wandb.

### “Known sharp edges” to address during migration
- **`dinov3/` dependency:** scripts expect a local torch.hub repo under `--repo_dir` (needs `hubconf.py`). We will document and validate this early to avoid silent failures.
- **Dataset directory naming:** standardize to `images/` + `masks/` everywhere; keep backwards-compatible CLI flags for older names if needed.
- **W&B coupling:** `PEFT_segdino.py` currently initializes W&B unconditionally; we will gate it so local runs don’t fail when wandb is absent (when we reach that file).

## Stage 0 — Baseline wiring & repo hygiene
**Goal:** ensure there is a single canonical dataset naming convention and minimal config drift.

Deliverables:
- Confirm `images/` + `masks/` everywhere (CLI defaults; docs).
- A single “paths and splits” contract referenced by all training/adaptation/eval entrypoints.
- Optional: a short `README` section pointing to `EXPERIMENT_PLAN.md`.

Validation:
- Dry-run CLI parsing and path resolution without reading data.

## Stage 1 — Deterministic split manifests (target_adapt vs target_holdout)
**Goal:** create `target_adapt/` and `target_holdout/` logically (and optionally physically) with committed filelists.

Tasks:
- Implement a splitter that:
  - reads `segdata/kvasir/test/images` filenames
  - produces `splits/kvasir_target_adapt.txt` and `splits/kvasir_target_holdout.txt`
  - uses fixed seed and stable sorting
- Decide whether to:
  - (A) keep images in place and drive selection via manifests only (preferred), or
  - (B) materialize `target_adapt/` and `target_holdout/` folders (higher I/O).

Deliverables:
- `splits/` text manifests + a `splits/metadata.json` capturing seed, holdout %, creation time.

Validation:
- Consistency check: no overlaps; counts match expected ratios.
- Dataset loader can load “subset by manifest” and return paired image/mask paths.

## Stage 2 — Deterministic corruption pipeline (single-family ladders + mixed)
**Goal:** create a reproducible corruption system with severity `S0..S4`, per-family ladders first.

Tasks:
- Define a corruption spec:
  - families: blur, noise, jpeg, illumination
  - severity mapping `S0..S4` per family
  - deterministic per-image RNG seed = f(filename, family, severity)
- Decide caching strategy:
  - (A) on-the-fly corruptions in the dataset pipeline (fast iteration; deterministic), or
  - (B) offline materialization into `segdata/kvasir_<family>_S{0..4}/...` (faster training; more disk)
  - Recommended: start on-the-fly; add offline cache option later.
- Implement “mixed corruption” mode for appendix:
  - choose 1–2 corruptions per image deterministically and compose.

Deliverables:
- `segdino/corruptions.py` spec (family implementations + severity maps + seeding rules).
- Config representation: `configs/corruptions/*.yaml` (or JSON) with the fixed IDs and mappings.

Validation:
- Determinism test: same image+spec yields identical pixels across runs/machines.
- Mask alignment check: masks remain unchanged and correctly paired.

## Stage 3 — Unified dataset + augmentation views for consistency training
**Goal:** one dataset interface that supports:
- clean source training
- corrupted target adaptation
- paired weak/strong views for consistency objectives
- subset selection by manifest

Tasks:
- Standardize sample dict fields: `{image, mask(optional), id, path, domain_meta}`
- Add a view generator:
  - `weak_view(x)` and `strong_view(x)` for consistency objective
  - ensure augmentations are deterministic under a seed when needed for debugging

Deliverables:
- `segdino/data.py` (dataset + transforms + manifest selection).
- One CLI surface shared by train/adapt/test scripts for `--dataset_root`, `--split_manifest`, `--img_dir_name images`, `--label_dir_name masks`.

Validation:
- Smoke-load 10 samples from each split and each severity.

## Stage 4 — Learned symbolic descriptor encoder `E_θ` (pretrain once)
**Goal:** implement and train a tiny mask-structure encoder on **source masks only**, then freeze it.

Design constraints (from protocol):
- Input views: soft mask-like tensors and boundary-band tensors
- Tiny architecture (2–4 conv blocks + MLP), `k=32/64`
- Training: contrastive or BYOL-style on structure-preserving mask augmentations

Tasks:
- Implement mask augmentation pipeline for descriptor learning (structure-preserving).
- Implement `E_θ` model + projection head (as needed).
- Implement training script:
  - trains on `segdata/kvasir/train/masks`
  - saves `E_θ` checkpoint + config

Deliverables:
- `segdino/symbolic_encoder.py` (model + boundary view function).
- `train_symbolic_encoder.py` (entrypoint).
- Saved weights path convention under `runs/` (ignored by git).

Validation:
- Qualitative: retrieval sanity (nearest neighbors in embedding space).
- Optional: UMAP/t-SNE figure generator for paper.

## Stage 5 — Pluggable PEFT adapter framework (budget-matched)
**Goal:** make adapter type a config switch: LoRA, SALT, and future PEFTs.

Tasks:
- Define a minimal adapter API:
  - `apply_adapter(model, spec) -> adapted_model`
  - `count_trainable_params(model)` utility
- Implement adapters:
  - LoRA on selected linear layers
  - SALT on selected linear layers
  - (optional later) IA³ / adapters / BitFit
- Implement “budget matching” helpers:
  - given a target % budget, select ranks or layer subsets to match within tolerance

Deliverables:
- `segdino/adapters/` package with `lora.py`, `salt.py`, `registry.py`.
- A config schema for adapter specs (`configs/adapters/*.yaml`).

Validation:
- Unit-ish checks (fast):
  - adapted model forward matches shape
  - only adapter params require grad
  - trainable param counts match requested budget roughly

## Stage 6 — Core adaptation objectives + baselines (source-free)
**Goal:** implement baselines and the core objective used by your method.

Primary core objective (locked):
- augmentation consistency (weak/strong views), plus optional entropy term

Baselines to implement:
- source-only evaluation pipeline
- entropy minimization
- consistency-only
- TENT-style (norm affine only)
- pseudo-label self-training (with confidence threshold)

Deliverables:
- `segdino/objectives.py` with modular losses
- `adapt.py` entrypoint supporting method presets via flags/config

Validation:
- Overfit check on a tiny subset (sanity) and confirm losses decrease.
- Confirm adaptation runner never touches source split paths (explicit path guard).

## Stage 7 — Symbolic EMA priors + two-scale alignment loss (global + boundary)
**Goal:** implement the key contribution: learned symbolic alignment with strict gating and robust distances.

Tasks:
- Compute descriptors during adaptation:
  - `s_g = E_θ(p)`
  - `s_b = E_θ(boundary_view(p))`
- Maintain EMA stats on `target_adapt` confident predictions:
  - mean + (diag variance initially; full cov optional later)
- Implement gating:
  - image-level confidence
  - degenerate mask checks
  - fragmentation proxy bounds
- Implement robust distance:
  - z-score + Huber, or clipped Mahalanobis
- Implement warmup/ramp schedule for `λ_sym`

Deliverables:
- `segdino/symbolic_prior.py` (EMA stats + update rules + gating)
- `segdino/symbolic_loss.py` (two-scale alignment)

Validation:
- Logging: distribution of accepted/rejected samples; EMA drift curves.
- Failure-rate reduction vs PEFT-only on S4 (quick regression test).

## Stage 8 — Metrics: Dice/IoU + Boundary F-score + HD95 + failure rates
**Goal:** compute the locked metrics consistently across all runs.

Tasks:
- Implement:
  - Dice, IoU
  - Boundary F-score (with fixed tolerance in pixels; document it)
  - HD95 (existing in `test_segdino.py` can be reused/ported)
  - failure rates and fragmentation proxy

Deliverables:
- `segdino/metrics.py` used by evaluation scripts
- CSV outputs compatible with aggregate scripts

Validation:
- Metric sanity on a few hand-checked examples (empty/full masks).

## Stage 9 — Experiment orchestration (configs, sweeps, multi-node)
**Goal:** make the run matrix easy to launch and hard to mess up.

Tasks:
- Add config templates:
  - corruption family + severity
  - adapter type + budget
  - method preset (baseline vs ours vs ablation)
- Add a launcher script that expands a grid into commands (or integrate with W&B sweeps if desired).
- Ensure every run writes:
  - config snapshot
  - seed
  - split manifests
  - corruption spec ID

Deliverables:
- `configs/` for reproducible runs
- `scripts/launch_grid.py` (optional) or W&B sweep YAMLs updated to new CLI

Validation:
- “Resume and rerun” determinism: rerunning same config reproduces identical corruption and split selection.

## Stage 10 — Paper-facing artifacts
**Goal:** produce the figures/checklists specified in `EXPERIMENT_PLAN.md`.

Tasks:
- Add plotting/aggregation utilities:
  - AUSC computation over severities
  - per-family and averaged tables
  - plots: degradation curves, budget curves, failure rates
  - optional: UMAP/t-SNE for `E_θ`

Deliverables:
- `analysis/` or `notebooks/` (optional) for aggregation scripts
- One “make figures” doc with exact commands

Validation:
- End-to-end reproduction on a new machine using only configs + manifests.

## Implementation order (what we will actually do next)
1. Stage 1 (split manifests) + Stage 3 (unified dataset selection) — unblock everything else.
2. Stage 2 (corruptions) — produce severity ladders and source-only degradation curves.
3. Stage 5 (pluggable adapters) + Stage 6 (core objectives + baselines).
4. Stage 4 (`E_θ` training) — train once and freeze.
5. Stage 7 (symbolic EMA + alignment) — implement full method + causality ablations.
6. Stage 8–10 (metrics, orchestration, paper artifacts).
7. Stage 5b (SALT-family symbolic-aligned variants) — only after `E_θ` + `L_sym` are working end-to-end.

## Implementation status (what exists in code now)
This section is a living checklist to keep the plan synchronized with what is actually implemented in the repo.

### Completed
- Stage 1 (split manifests)
  - Generator: `tools/make_target_splits.py`
  - Manifest docs: `splits/README.md`
  - Typical outputs: `splits/kvasir_target_adapt.txt`, `splits/kvasir_target_holdout.txt`, `splits/kvasir_target_splits.json`
- Stage 2 (corruptions, deterministic seeding)
  - Corruption specs + implementation: `corruptions.py`
  - Wrapper hook for datasets: `corruption_transform.py`
  - Preview utility: `tools/preview_corruption.py`
  - Supported:
    - single-family: `blur`, `noise`, `jpeg`, `illumination`
    - mixed: `mixed` with `--num_ops` up to 4
    - severity ladder currently supports `S0..S8` in `corruptions.py` (protocol still reports `S0..S4` as the main ladder unless explicitly extended).
- Stage 3 (manifest-aware datasets + view generation)
  - Dataset utilities: `data.py`
    - `ManifestSegmentationDataset` (mask-supervised loading, manifest-driven)
    - `ManifestConsistencyDataset` (weak/strong views for consistency, manifest-driven)
    - `image_pre_transform(img_bgr, image_id)` hook for corruptions
  - Weak/strong views: `views.py` via `WeakStrongViewTransform`
- Stage 8 (metrics)
  - Shared metrics: `metrics.py` (Dice/IoU, Boundary F-score, HD95, failure rates accumulator)
- Stage 6 (baseline evaluation utilities)
  - Corruption curve evaluator: `tools/eval_corruption_curve.py` (includes mixed support)

### Completed (PEFT axis)
- Pluggable PEFT injection (LoRA/SALT) for adaptation runs
  - Core implementation: `adapters.py`
  - CLI integration: `tools/adapt_baselines.py` via `--adapter {none,lora,salt}`
  - Compatibility fixes included:
    - device/dtype placement (avoid CPU/GPU mismatch)
    - wrapper attributes expected by DINOv3 attention (e.g., `in_features`)

### Completed (learned symbolic descriptors)
- Learned symbolic descriptor encoder `E_θ`
  - Package: `symalign/` (kept separate from `segdino/`)
  - Training CLI: `tools/train_symbolic_encoder.py`
  - Output checkpoint: `runs/symalign_encoder_kvasir/encoder_final.pth`
- Symbolic alignment integrated into adaptation
  - CLI integration: `tools/adapt_baselines.py` via `--use_symbolic` and `--symbolic_*` flags
  - Uses target-only EMA priors with confidence gating and warmup

### Newly added for “Step 1/2” execution
- Step 1 (pseudo-label quality diagnostics)
  - Tool: `tools/pseudolabel_quality.py`
  - Output: per-image CSV with confidence/entropy proxies under a chosen corruption regime.
- Step 2 (source-free adaptation baselines)
  - Tool: `tools/adapt_baselines.py`
  - Methods: `entropy`, `consistency`, `selftrain`, `tent`
  - Adapt split: `target_adapt` manifest (unlabeled)
  - Eval split: `target_holdout` manifest (labeled)
  - Convenience runner: `tools/run_baseline_suite.sh` (runs all 4 methods into one CSV)

### In progress / next
- Teacher-dependence experiments
  - Add a `--teacher_kl_schedule` option (or equivalent) to test KL annealing without changing the protocol.
  - Add explicit anti-collapse constraints (foreground-mass bounds / smoothness / fragmentation proxies) to test “no teacher” regimes.
- Stage 5 (pluggable PEFT adapter framework + budget matching)
  - Implement adapter registry + LoRA/SALT knobs in a shared module (avoid duplicating logic between scripts/tools).
- Stage 5b (symbolically-aligned SALT-like variants)
  - Implement three SALT-family variants (as ablations) under a unified interface:
    - `SALT-G`: symbolic-gradient gated update subspace (layer/component selection driven by `L_sym` signal)
    - `SALT-S/A`: spectrum split into “structure” vs “appearance” components with separate loss coupling
    - `SALT-Dial`: per-layer scalar dial(s) controlling adaptation strength, supervised/regularized by `L_sym`
  - Add config knobs to toggle these variants and log their additional parameters and compute.
- Stage 4 (`E_θ` learned symbolic encoder pretraining)
  - Implement encoder, mask augmentations, and a training script; save weights under `runs/`.
- Stage 7 (symbolic EMA priors + two-scale alignment loss)
  - Implement EMA stats, gating, robust distances, and ramp schedule; integrate into adaptation runner.

## Execution order decision (locked)
We will **not** implement SALT-G / SALT-S/A / SALT-Dial until the symbolic pipeline exists.

Order:
1) Implement and validate `E_θ` (descriptor learning on source masks).
2) Implement symbolic EMA priors + two-scale alignment loss (global+boundary) on target.
3) Demonstrate “Ours works with PEFT plug-and-play” using `{none, LoRA, SALT}`.
4) Only then implement SALT-family symbolic-aligned variants as ablations (SALT-Dial → SALT-S/A → SALT-G).

### Notes on repo layout in the cluster
If the repo is checked out under a parent folder named `segdino` and run as a module (e.g. `python -m segdino.tools.eval_corruption_curve`), keep `PYTHONPATH` set to the repo root and ensure the scripts import from the same package namespace consistently.

## Implementation notes for SALT-family ablations
To keep this tractable and comparable:
- All SALT-family variants should share the same injection points (e.g., attention `qkv`/`proj` linears) and the same base parameter budget.
- We will implement variants as small extensions around a shared SALT module:
  - `SALT-G`: add a gating mask per layer/component (or per singular index bucket) and drive it with symbolic-loss gradients/statistics (stop-grad where needed).
  - `SALT-S/A`: explicitly separate “top-k structure” vs “remaining appearance” components; apply `L_sym` only to structure bucket and `L_core` primarily to appearance bucket (or vice versa as an ablation).
  - `SALT-Dial`: attach per-layer scalar(s) that modulate the effective update (e.g., scale the spectrum modification); include a monotonicity/regularization prior if needed.
  
We will keep these as ablations (not all enabled in the main method table) unless one variant consistently dominates.
