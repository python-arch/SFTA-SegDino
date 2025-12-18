#!/usr/bin/env bash
set -uo pipefail

# Run a baseline adaptation suite (source-free) on a chosen corruption regime.
#
# Expected usage (example):
#   PYTHONPATH="$(pwd)" bash tools/run_baseline_suite.sh \
#     --dataset_root ./segdino/segdata/kvasir \
#     --adapt_manifest ./segdino/splits/kvasir_target_adapt.txt \
#     --eval_manifest ./segdino/splits/kvasir_target_holdout.txt \
#     --corruption mixed --severity 4 --num_ops 4 \
#     --ckpt /path/to/seg_ckpt.pth \
#     --dino_ckpt /path/to/dino.pth \
#     --repo_dir ./segdino/dinov3 \
#     --out_csv ./segdino/runs/adapt_baselines_mixed_ops4_S4.csv

DATASET_ROOT=""
ADAPT_MANIFEST=""
EVAL_MANIFEST=""
CORRUPTION="mixed"
SEVERITY="4"
NUM_OPS="4"
CORRUPTION_ID="v1"

CKPT=""
DINO_CKPT=""
DINO_SIZE="s"
REPO_DIR="./dinov3"

STEPS="500"
BATCH_SIZE="4"
LR="1e-4"
WEIGHT_DECAY="0.0"
NUM_WORKERS="4"

DICE_THR="0.5"
BOUNDARY_TOL_PX="2"
SELFTRAIN_CONF_THR="0.9"
TRAINABLE="head"
TEACHER_KL_WEIGHT="1.0"
ADAPTER="none"
LORA_R="8"
LORA_ALPHA="16"
LORA_DROPOUT="0.05"
SALT_RANK="8"
SALT_R_LORA="8"
SALT_SEED="42"

OUT_CSV="./runs/adapt_baselines.csv"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset_root) DATASET_ROOT="$2"; shift 2;;
    --adapt_manifest) ADAPT_MANIFEST="$2"; shift 2;;
    --eval_manifest) EVAL_MANIFEST="$2"; shift 2;;

    --corruption) CORRUPTION="$2"; shift 2;;
    --severity) SEVERITY="$2"; shift 2;;
    --num_ops) NUM_OPS="$2"; shift 2;;
    --corruption_id) CORRUPTION_ID="$2"; shift 2;;

    --ckpt) CKPT="$2"; shift 2;;
    --dino_ckpt) DINO_CKPT="$2"; shift 2;;
    --dino_size) DINO_SIZE="$2"; shift 2;;
    --repo_dir) REPO_DIR="$2"; shift 2;;

    --steps) STEPS="$2"; shift 2;;
    --batch_size) BATCH_SIZE="$2"; shift 2;;
    --lr) LR="$2"; shift 2;;
    --weight_decay) WEIGHT_DECAY="$2"; shift 2;;
    --num_workers) NUM_WORKERS="$2"; shift 2;;

    --dice_thr) DICE_THR="$2"; shift 2;;
    --boundary_tol_px) BOUNDARY_TOL_PX="$2"; shift 2;;
    --selftrain_conf_thr) SELFTRAIN_CONF_THR="$2"; shift 2;;
    --trainable) TRAINABLE="$2"; shift 2;;
    --teacher_kl_weight) TEACHER_KL_WEIGHT="$2"; shift 2;;
    --adapter) ADAPTER="$2"; shift 2;;
    --lora_r) LORA_R="$2"; shift 2;;
    --lora_alpha) LORA_ALPHA="$2"; shift 2;;
    --lora_dropout) LORA_DROPOUT="$2"; shift 2;;
    --salt_rank) SALT_RANK="$2"; shift 2;;
    --salt_r_lora) SALT_R_LORA="$2"; shift 2;;
    --salt_seed) SALT_SEED="$2"; shift 2;;

    --out_csv) OUT_CSV="$2"; shift 2;;
    -h|--help)
      sed -n '1,120p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${DATASET_ROOT}" || -z "${ADAPT_MANIFEST}" || -z "${EVAL_MANIFEST}" || -z "${CKPT}" || -z "${DINO_CKPT}" ]]; then
  echo "Missing required args. Run with --help for usage." >&2
  exit 2
fi

METHODS=(entropy consistency selftrain tent)

for METHOD in "${METHODS[@]}"; do
  echo "============================================================"
  echo "[RUN] method=${METHOD} corruption=${CORRUPTION} severity=${SEVERITY} num_ops=${NUM_OPS}"
  echo "============================================================"

  if python -m segdino.tools.adapt_baselines \
    --dataset_root "${DATASET_ROOT}" \
    --adapt_manifest "${ADAPT_MANIFEST}" \
    --eval_manifest "${EVAL_MANIFEST}" \
    --img_dir_name images \
    --mask_dir_name masks \
    --input_h 256 \
    --input_w 256 \
    --corruption "${CORRUPTION}" \
    --severity "${SEVERITY}" \
    --num_ops "${NUM_OPS}" \
    --corruption_id "${CORRUPTION_ID}" \
    --method "${METHOD}" \
    --steps "${STEPS}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --lr "${LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --dice_thr "${DICE_THR}" \
    --boundary_tol_px "${BOUNDARY_TOL_PX}" \
    --selftrain_conf_thr "${SELFTRAIN_CONF_THR}" \
    --trainable "${TRAINABLE}" \
    --teacher_kl_weight "${TEACHER_KL_WEIGHT}" \
    --adapter "${ADAPTER}" \
    --lora_r "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout "${LORA_DROPOUT}" \
    --salt_rank "${SALT_RANK}" \
    --salt_r_lora "${SALT_R_LORA}" \
    --salt_seed "${SALT_SEED}" \
    --ckpt "${CKPT}" \
    --dino_ckpt "${DINO_CKPT}" \
    --dino_size "${DINO_SIZE}" \
    --repo_dir "${REPO_DIR}" \
    --out_csv "${OUT_CSV}"
  then
    echo "[OK] method=${METHOD}"
  else
    echo "[FAIL] method=${METHOD} (continuing)" >&2
  fi
done

echo "[OK] Baseline suite complete: ${OUT_CSV}"
