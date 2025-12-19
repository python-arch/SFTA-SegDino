# Findings so far (Kvasir + corruption ladder + source-free baselines)

This note records what we observed empirically so far, and why the results make sense. It is meant as a running lab notebook that can later be distilled into the paper narrative.

## Setup recap (what is being tested)
- Task: binary polyp segmentation (Kvasir-SEG).
- Protocol: **source training** on `train/` only; **source-free target adaptation** on `target_adapt` only; evaluation only on `target_holdout`.
- Target shift: deterministic corruptions applied to target images with a severity ladder `S0..S4`.
- Stress regime focus: **mixed corruptions** where pseudo-label self-training is expected to be unreliable.

## Splits are correct (source-free enforcement foundation)
We generated deterministic manifests:
- `splits/kvasir_target_adapt.txt`: 160 images
- `splits/kvasir_target_holdout.txt`: 40 images
- Verified no overlap between the two (0 overlapping lines).

This is critical because it ensures:
- adaptation never sees holdout images (no test leakage),
- the holdout size is enough to estimate Dice/HD95 trends,
- the evaluation protocol is reviewer-proof.

## Source-only degradation curves (why blur/jpg/illumination looked “too easy”)
### Observation
On the initial single-family ladders `S0..S4`:
- **Blur**: Dice stayed ~flat (near invariant).
- **JPEG**: small/no consistent degradation.
- **Illumination**: mild degradation.
- **Noise**: the only family that produced a meaningful drop at `S4`.

### Interpretation / reasoning
This is plausible because:
- DINOv3-style features can be relatively robust to moderate blur/JPEG changes, especially after resizing to a fixed input resolution.
- Blur can sometimes act like denoising and reduce high-frequency clutter, causing non-monotone behavior on small evaluation sets.
- Noise directly corrupts fine texture and local cues; segmentation is more sensitive to this.

Conclusion: **single-family blur/JPEG/illumination at S0..S4 were not “hard” enough** to create the desired pseudo-label failure regime.

## Mixed corruptions created the desired “pseudo-label failure” regime
### Observation
We evaluated mixed corruptions across severities for `num_ops=2` and `num_ops=4`:
- `mixed ops2`:
  - `S4` Dice ≈ **0.8187** (mild shift; small drop from clean).
- `mixed ops4`:
  - `S4` Dice ≈ **0.7188** (large drop).
  - Boundary F-score drops sharply; HD95 increases (worse boundary errors).
  - No degenerate collapse in source-only evaluation (`empty_rate=0`, `full_rate=0`).

### Interpretation / reasoning
This is exactly the “stress knob” we need:
- `ops2` is a “moderate” shift regime where reasonable methods should stay close to source-only.
- `ops4 S4` is a “severe” shift regime where:
  - pseudo-label quality is expected to deteriorate,
  - boundary metrics degrade strongly,
  - adaptation methods have room to improve.

This supports the protocol design: **use `mixed ops4 S4` as the stress-test regime** and `mixed ops2` as a moderate regime.

## Pseudo-label quality diagnostics (holdout, per-image)
We ran `tools/pseudolabel_quality.py` to produce per-image CSVs that quantify:
- per-image Dice/IoU/BoundaryF/HD95
- mean pixel entropy of the soft mask
- “confidence mass” proxies such as `frac_conf` and `frac_conf_fg` at `conf_thr=0.9`
- a precision-only proxy `conf_fg_precision` (how often confident predicted-FG pixels fall on GT-FG)

### Observation
- `mixed ops4 S4` shows **high dispersion**:
  - some images remain easy (Dice > 0.9),
  - some images are catastrophic (Dice ~0.1–0.4, very large HD95).
- `mixed ops2 S4` is **much easier overall**, with many images in the ~0.8–0.95 Dice range, but still contains a few hard outliers.

### Interpretation / reasoning
This supports a key paper claim: under severe shift, pseudo-label reliability is **heterogeneous and brittle**—even if a subset of pixels is high-confidence, global mask quality can be poor. This is exactly the regime where “structure/symbolic” regularization may provide signal beyond naive pseudo-labeling.

## First attempt at adaptation baselines collapsed (and why)
### Observation
Initial baseline adaptation runs (entropy / consistency / self-train / tent) collapsed to **all-empty predictions** (empty_rate=1.0).

### Interpretation / reasoning
This is a known failure mode:
- Entropy minimization has a trivial solution for binary segmentation: push probabilities toward 0 everywhere.
- Consistency can also collapse if the model is free to drift without a stabilizer.
- Self-training collapses when pseudo-labels are poor and feedback loops reinforce errors.

This was *useful* diagnostically: it showed our initial baseline implementation was missing standard stabilizers needed for severe shifts.

## Stabilized source-free baselines (teacher regularization + restricted trainable scope)
### Implementation changes that fixed collapse
We added two key stabilizers:
1) **Trainable scope restriction (default head-only)** for non-TENT baselines, to avoid catastrophic drift.
2) A **frozen teacher copy** of the initial model and a KL regularizer `KL(teacher || student)` on weak-view predictions, plus using teacher predictions for pseudo-label generation in self-training.

### Observation (mixed ops4 S4, 500 steps, bs=4, lr=1e-4)
All methods became stable (no collapse, empty/full rates = 0):
- `entropy`: Dice **0.7258**, IoU **0.6070**, BF **0.2472**, HD95 **48.83**
- `consistency`: Dice **0.7235**, IoU **0.6029**, BF **0.2503**, HD95 **48.59**
- `selftrain`: Dice **0.7145**, IoU **0.5968**, BF **0.2292**, HD95 **44.15**
- `tent`: Dice **0.7349**, IoU **0.6152**, BF **0.2289**, HD95 **46.92**

Relative to source-only `mixed ops4 S4` (Dice ~0.7188):
- entropy / consistency: small gains (≈ +0.5 to +0.7 Dice points)
- selftrain: slightly worse than source-only (consistent with “pseudo-labels are unreliable”)
- tent: best Dice/IoU among non-symbolic baselines here

### Interpretation / reasoning
This is consistent with expectations:
- With a strong frozen teacher, entropy/consistency behave more like “small refinement” rather than unconstrained drift.
- Self-training remains sensitive to pseudo-label noise even with stabilization; under severe shift it can underperform simpler methods.
- TENT often gives modest gains by adjusting normalization statistics/affines without changing the model’s core semantics.

## Baseline ladder results (mixed ops4 across severities)
We ran the full baseline suite (entropy / consistency / self-train / TENT) under `mixed ops4` for severities `S2`, `S3`, and a **clean** `S4` run (to avoid earlier pre-stabilization collapsed rows).

### AUSC-style aggregation (mean across S2–S4 under mixed ops4)
Ranking observed:
- `tent` best overall (highest mean Dice/IoU and best mean boundary F-score)
- `entropy` and `consistency` are close behind
- `selftrain` is the weakest overall (consistent with pseudo-label brittleness)

### Stress test snapshot (mixed ops4 S4, clean)
Representative results (holdout `n=40`):
- `tent`: best Dice/IoU (≈0.737 Dice)
- `entropy`: slightly lower Dice but competitive; can be best on boundary F-score in some runs
- `selftrain`: may achieve better HD95 but is less reliable and can produce rare empty predictions

## Moderate regime baseline (mixed ops2 S4)
Under `mixed ops2 S4`, all methods are stable and high-performing (Dice ≈0.81–0.84), and `tent` again tends to be best overall.

## PEFT-only baselines (LoRA vs SALT)
We evaluated PEFT-only baselines (adapter injected into backbone attention linears; ~24 layers wrapped) under the same stabilized adaptation runner.

### Stress regime: mixed ops4 S4
Key observations:
- **LoRA** achieved the best Dice among PEFT runs with `entropy` (≈0.742 Dice), and was competitive with TENT.
- **SALT** achieved similar performance with fewer trainable parameters (≈0.60% trainable vs ≈0.92% for LoRA in the tested configuration), but performance depended more on the choice of core objective (entropy vs consistency/self-train).

Interpretation:
- PEFT improves or matches the best non-PEFT baselines, but it is not uniformly beneficial across all objectives.
- The parameter-efficiency of SALT is a credible advantage to emphasize when we move to symbolically aligned reparameterizations.

### Moderate regime: mixed ops2 S4
Key observations:
- Both **LoRA** and **SALT** are strong and stable (no collapse).
- Entropy and TENT remain the most competitive objectives in this moderate regime; self-training tends to underperform.

Interpretation:
- The PEFT axis appears “plug-and-play” across at least two adapters (LoRA, SALT), supporting the design choice to keep PEFT modular.

## Practical takeaways for the next stage (learned-symbolic method)
- The regime `mixed ops4 S4` is sufficiently hard to be a meaningful stress test.
- We now have stable baselines to beat; **TENT is currently the strongest** on Dice/IoU in this regime.
- A credible contribution must improve:
  - Dice/IoU **and/or**
  - boundary metrics (BF, HD95) **and**
  - maintain low failure rates (no collapse).

## Symbolic alignment stability ablations (teacher KL dependence)
We tested whether learned symbolic alignment can replace the frozen-teacher KL stabilizer.

### Observation 1: removing teacher KL collapses
Setting `--teacher_kl_weight 0.0` (even with symbolic enabled) collapses to the trivial binary-segmentation optimum:
- all-background predictions (`empty_rate=1.0`),
- Dice≈0, HD95=inf.

### Reasoning
This is expected: entropy minimization/TENT for binary segmentation has a degenerate minimum where all logits push to background.
The teacher KL term is not just “a regularizer”; it is currently a **necessary anti-collapse constraint** in the severe regime.

### Observation 2: simple foreground-fraction prior was insufficient (current implementation)
We attempted an additional stabilizer that matches the mean foreground probability of student to teacher (`fg_prior_weight`), but the run still collapsed under `teacher_kl_weight=0.0`.

### Implication / next fix direction
To decouple from teacher KL (if desired), we likely need one of:
- a stronger/structured stabilizer (e.g., spatial prior, boundary-aware constraint, or per-image foreground mass constraint),
- or a schedule: keep KL early to prevent collapse, then anneal it to allow symbolic losses to shape the solution.

For professor feedback: this is a concrete “failure mode + hypothesis” we can discuss.

## “Useful numbers” snapshot (single seed, holdout n=40)
All numbers below: corruption `mixed`, `corruption_id=v1`, adapt on `target_adapt`, eval on `target_holdout`.

### Source-only (no adaptation)
- Stress regime (`mixed ops4 S4`): Dice≈0.7188, IoU≈0.5989, BoundaryF≈0.2299, HD95≈50.31

### Non-PEFT baselines (no symbolic, stabilized)
- Moderate regime (`mixed ops2 S4`): `tent` Dice=0.8362, IoU=0.7420, BoundaryF=0.3785, HD95=26.15
- Stress regime (`mixed ops4 S4`, clean baseline CSV): `tent` Dice=0.7372, IoU=0.6173, BoundaryF=0.2291, HD95=46.66
- AUSC proxy (mean over `mixed ops4` severities `S2–S4`): `tent` mean Dice≈0.8010 (best among the four baselines)

### PEFT-only baselines (no symbolic)
- Stress regime (`mixed ops4 S4`, best objective = entropy):
  - LoRA: Dice=0.7418, IoU=0.6202, BoundaryF=0.2375, HD95=47.72 (trainable≈0.92%)
  - SALT: Dice=0.7396, IoU=0.6197, BoundaryF=0.2355, HD95=46.25 (trainable≈0.60%)
- Moderate regime (`mixed ops2 S4`, best objective = entropy):
  - LoRA: Dice=0.8381, IoU=0.7446, BoundaryF=0.3739, HD95=26.18
  - SALT: Dice=0.8383, IoU=0.7442, BoundaryF=0.3788, HD95=26.08

### Learned symbolic alignment (Eθ + EMA priors)
- Tuned setting (no PEFT, `tent` core): `symbolic_lambda=0.1`, `warmup_steps=150`, `conf_thr=0.8`
  - Stress regime (`mixed ops4 S4`): Dice=0.7376, IoU=0.6178, BoundaryF=0.2274, HD95=46.36
- Plug-and-play check (same tuned symbolic settings, `tent` core):
  - +LoRA: Dice=0.7317, IoU=0.6113, BoundaryF=0.2285, HD95=47.30
  - +SALT: Dice=0.7349, IoU=0.6145, BoundaryF=0.2270, HD95=47.03

### Teacher dependence (hard failure)
- Stress regime (`mixed ops4 S4`): setting `--teacher_kl_weight 0.0` collapses to all-background (empty_rate=1.0), even with symbolic enabled.

## Known limitations of current evidence
- Holdout is 40 images (fine for iteration but still somewhat noisy); we should run 3 seeds later.
- Symbolic alignment currently matches (but does not exceed) the best baseline in the stress regime; we need to either:
  - improve the symbolic signal (e.g., differentiable boundary view, loss calibration), and/or
  - explicitly target the boundary tradeoff (BoundaryF/HD95), and/or
  - revisit the stabilizer story (KL annealing vs explicit constraints).

## Immediate next milestones (high-signal, low-scope-creep)
1. **Teacher KL annealing experiment** on `mixed ops4 S4`:
   - keep KL early to prevent collapse, decay toward 0 late, measure stability and metrics.
2. **No-teacher explicit anti-collapse constraints** on `mixed ops4 S4`:
   - foreground-mass bounds + smoothness/TV + fragmentation proxy (evaluate if any can replace teacher KL).
3. **Symbolic tuning grid (small)** on `mixed ops4 S4`:
   - `symbolic_lambda ∈ {0.03, 0.1, 0.3}`
   - `conf_thr ∈ {0.8, 0.9}`
   - `warmup_steps ∈ {100, 150, 200}`
