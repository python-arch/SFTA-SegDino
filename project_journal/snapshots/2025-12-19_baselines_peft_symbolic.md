# Snapshot — 2025-12-19 (Kvasir, corruption ladder, baselines/PEFT/symbolic)

This snapshot records the **current stable pipeline** (splits → corruptions → source-only curves → adaptation baselines → PEFT axis → learned symbolic encoder → symbolic alignment runs) and the **high-signal numbers** we will show for professor feedback.

## Locked protocol elements
- Dataset: Kvasir-SEG (`segdata/kvasir`).
- Split contract:
  - `splits/kvasir_target_adapt.txt`: 160 images
  - `splits/kvasir_target_holdout.txt`: 40 images
  - Verified overlap: 0
- Target shift: deterministic corruptions; focus on `mixed` as the “stress knob”.
- Metrics: Dice/IoU (primary), Boundary F-score + HD95 (boundary), failure rates (empty/full).

## Regimes (what counts as “moderate” vs “stress”)
- Moderate: `mixed`, `num_ops=2`, `severity=4` (“ops2 S4”)
- Stress: `mixed`, `num_ops=4`, `severity=4` (“ops4 S4”)

Reason: `ops4 S4` produces a large degradation in source-only and large per-image variance in pseudo-label quality.

## Key results (single seed, holdout n=40)
All runs: `corruption_id=v1`, adapt on `target_adapt`, eval on `target_holdout`.

### Source-only (no adaptation)
- `mixed ops4 S4`: Dice≈0.7188, IoU≈0.5989, BoundaryF≈0.2299, HD95≈50.31

### Non-PEFT baselines (no symbolic, stabilized)
- `mixed ops2 S4`: `tent` Dice=0.8362, IoU=0.7420, BoundaryF=0.3785, HD95=26.15
- `mixed ops4 S4` (clean baseline CSV): `tent` Dice=0.7372, IoU=0.6173, BoundaryF=0.2291, HD95=46.66
- AUSC proxy: mean over `mixed ops4` severities `S2–S4`: `tent` mean Dice≈0.8010 (best among the 4 baselines)

Interpretation: TENT is the strongest stabilized baseline; boundaries remain poor under stress.

### PEFT-only baselines (no symbolic)
Stress regime (`mixed ops4 S4`, best objective = entropy):
- LoRA: Dice=0.7418, IoU=0.6202, BoundaryF=0.2375, HD95=47.72 (trainable≈0.92%, wrapped=24)
- SALT: Dice=0.7396, IoU=0.6197, BoundaryF=0.2355, HD95=46.25 (trainable≈0.60%, wrapped=24)

Moderate regime (`mixed ops2 S4`, best objective = entropy):
- LoRA: Dice=0.8381, IoU=0.7446, BoundaryF=0.3739, HD95=26.18
- SALT: Dice=0.8383, IoU=0.7442, BoundaryF=0.3788, HD95=26.08

Interpretation: PEFT is genuinely “plug-and-play” across LoRA/SALT; SALT is more parameter-efficient; performance differences are modest.

### Learned symbolic descriptors + symbolic alignment (current status)
- Trained a tiny mask descriptor encoder `E_θ` via `tools/train_symbolic_encoder.py`
  - Output: `runs/symalign_encoder_kvasir/encoder_final.pth`

Best tuned symbolic settings so far (no PEFT, `tent` core):
- `symbolic_lambda=0.1`, `symbolic_warmup_steps=150`, `symbolic_conf_thr=0.8`
- `mixed ops4 S4`: Dice=0.7376, IoU=0.6178, BoundaryF=0.2274, HD95=46.36

Plug-and-play check (same tuned symbolic settings, `tent` core):
- +LoRA: Dice=0.7317, IoU=0.6113, BoundaryF=0.2285, HD95=47.30
- +SALT: Dice=0.7349, IoU=0.6145, BoundaryF=0.2270, HD95=47.03

Interpretation: symbolic alignment currently matches (but does not exceed) the best baseline in stress; interactions with PEFT need care.

## Critical failure mode (teacher dependence)
Observed: removing teacher KL collapses to all-background under stress.
- `mixed ops4 S4`: setting `--teacher_kl_weight 0.0` collapses (empty_rate=1.0) even with symbolic enabled.
- Simple `fg_prior_weight` was insufficient to prevent collapse.

Interpretation: currently, teacher KL is a **necessary anti-collapse stabilizer** in the severe regime. “Symbolic replaces teacher” is not yet supported; this motivates KL annealing and/or explicit constraints.

## What this means for positioning
- Current strongest defensible framing: symbolic alignment is a **structure/boundary preservation signal** that complements standard stabilizers.
- Next high-signal experiments: KL annealing + no-teacher explicit constraints, and a pivot to multi-modal symbolic descriptors (mask+appearance).

## Canonical docs for this snapshot
- `EXPERIMENT_PLAN.md`
- `IMPLEMENTATION_PLAN.md`
- `FINDINGS_SO_FAR.md`
- `README.md`

