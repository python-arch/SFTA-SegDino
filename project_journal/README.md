# Project Journal (Versioned Plans + Findings)

This folder is a lightweight “tagged notebook” so we can share **what we believed at a point in time** with professors and always reproduce the context later.

Git already tracks line-by-line diffs of `README.md`, `EXPERIMENT_PLAN.md`, `IMPLEMENTATION_PLAN.md`, and `FINDINGS_SO_FAR.md`. This folder adds:
- curated snapshots (what mattered at that time),
- prioritized pivot plans (e.g., multi-modal symbolic alignment),
- decision points + next commands.

## Structure
- `project_journal/snapshots/`: dated snapshots of results + key decisions.
- `project_journal/plans/`: dated plans (what we intend to implement next).

## Conventions
- One folder per date: `YYYY-MM-DD/`.
- Each snapshot should include:
  - dataset + corruption regime
  - checkpoints used
  - key results table (the “good numbers”)
  - interpretation + next decisions
- Each plan should include:
  - scope (what is in/out)
  - file-level implementation steps (what to edit/add)
  - exact commands to run (no abbreviations)

