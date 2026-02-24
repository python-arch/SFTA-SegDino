#!/usr/bin/env bash
set -euo pipefail

# One-command medical classification LoRA grid:
# 1) Download requested datasets (MedMNIST/APTOS/HAM10000/BreakHis)
# 2) Run classification training on dataset x placement x rank x seed

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

NPZ_DIR="${REPO_ROOT}/medmnist_npz"
DATA_ROOT="${REPO_ROOT}/data"
DATASETS="PneumoniaMNIST,DermaMNIST (MedMNIST),BloodMNIST (MedMNIST),OrganMNIST,RetinaMNIST"
ORGAN_VARIANT="a"
SIZE="28"
BREAKHIS_SOURCE="official"

DINO_CKPT=""
REPO_DIR="${REPO_ROOT}/dinov3"
DINO_SIZE="s"
EPOCHS="30"
BATCH_SIZE="128"
NUM_WORKERS="8"
LR="1e-4"
WEIGHT_DECAY="1e-4"
INPUT_SIZE="224"
VAL_RATIO="0.1"
TEST_RATIO="0.2"
BREAKHIS_MAG=""

LORA_RANKS="4,8,16,32,64"
SEEDS="42,7,2023"
LORA_PLACEMENTS="Q;K;V;P;F1;Q,K;Q,V;F1,F2;Q,K,V;P,F1,F2;Q,K,V,P;Q,K,V,P,F1,F2"
OUT_CSV="${REPO_ROOT}/runs/medmnist_cls/lora_grid_results.csv"

WANDB="false"
WANDB_PROJECT="segdino_medmnist_cls"
WANDB_ENTITY=""

SKIP_DOWNLOAD="false"
SKIP_PREFLIGHT="false"
DRY_RUN="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dino_ckpt) DINO_CKPT="$2"; shift 2;;
    --repo_dir) REPO_DIR="$2"; shift 2;;
    --dino_size) DINO_SIZE="$2"; shift 2;;
    --npz_dir) NPZ_DIR="$2"; shift 2;;
    --data_root) DATA_ROOT="$2"; shift 2;;
    --datasets) DATASETS="$2"; shift 2;;
    --organ_variant) ORGAN_VARIANT="$2"; shift 2;;
    --size) SIZE="$2"; shift 2;;
    --breakhis_source) BREAKHIS_SOURCE="$2"; shift 2;;
    --epochs) EPOCHS="$2"; shift 2;;
    --batch_size) BATCH_SIZE="$2"; shift 2;;
    --num_workers) NUM_WORKERS="$2"; shift 2;;
    --lr) LR="$2"; shift 2;;
    --weight_decay) WEIGHT_DECAY="$2"; shift 2;;
    --input_size) INPUT_SIZE="$2"; shift 2;;
    --val_ratio) VAL_RATIO="$2"; shift 2;;
    --test_ratio) TEST_RATIO="$2"; shift 2;;
    --breakhis_mag) BREAKHIS_MAG="$2"; shift 2;;
    --lora_ranks) LORA_RANKS="$2"; shift 2;;
    --seeds) SEEDS="$2"; shift 2;;
    --lora_placements) LORA_PLACEMENTS="$2"; shift 2;;
    --out_csv) OUT_CSV="$2"; shift 2;;
    --wandb) WANDB="true"; shift 1;;
    --wandb_project) WANDB_PROJECT="$2"; shift 2;;
    --wandb_entity) WANDB_ENTITY="$2"; shift 2;;
    --skip_download) SKIP_DOWNLOAD="true"; shift 1;;
    --skip_preflight) SKIP_PREFLIGHT="true"; shift 1;;
    --dry_run) DRY_RUN="true"; shift 1;;
    -h|--help)
      cat <<'EOF'
Run a medical classification LoRA grid (MedMNIST/APTOS/HAM10000/BreakHis).

Usage:
  bash tools/run_medmnist_lora_grid.sh --dino_ckpt /path/to/dino_weights.pth [options]

Core options:
  --dino_ckpt PATH            Required DINO checkpoint path
  --datasets CSV              Dataset list (friendly names or tokens)
                             Examples:
                             MedMNIST: PneumoniaMNIST,DermaMNIST (MedMNIST),BloodMNIST (MedMNIST),OrganMNIST,RetinaMNIST
                             Others: APTOS2019,HAM10000,BreakHis
  --lora_ranks CSV            Default: 4,8,16,32,64
  --seeds CSV                 Default: 42,7,2023
  --lora_placements STR       Semicolon-separated placement combos
                              Default: Q;K;V;P;F1;Q,K;Q,V;F1,F2;Q,K,V;P,F1,F2;Q,K,V,P;Q,K,V,P,F1,F2
  --dry_run                   Print commands only

Data options:
  --npz_dir PATH              Default: ./medmnist_npz
  --data_root PATH            Default: ./data
  --size 28|64|128|224        MedMNIST size variant for download (default: 28)
  --organ_variant a|c|s       OrganMNIST variant when downloading (default: a)
  --breakhis_source SRC       official|kaggle (default: official)
  --skip_download             Assume npz files already exist

Train options:
  --epochs INT                Default: 30
  --batch_size INT            Default: 128
  --input_size INT            Default: 224
  --lr FLOAT                  Default: 1e-4
  --weight_decay FLOAT        Default: 1e-4
  --val_ratio FLOAT           Split ratio for folder datasets (default: 0.1)
  --test_ratio FLOAT          Split ratio for folder datasets (default: 0.2)
  --breakhis_mag MAG          BreakHis magnification filter: 40|100|200|400 (optional)
  --out_csv PATH              Aggregate results CSV path
  --wandb                     Enable wandb logging
  --wandb_project NAME        wandb project name
  --wandb_entity NAME         wandb entity
  --skip_preflight            Skip dependency/path checks
EOF
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$DINO_CKPT" ]]; then
  echo "--dino_ckpt is required" >&2
  exit 2
fi
if [[ ! -f "$DINO_CKPT" ]]; then
  echo "dino checkpoint not found: $DINO_CKPT" >&2
  exit 2
fi
if [[ ! -d "$REPO_DIR" ]]; then
  echo "dinov3 repo_dir not found: $REPO_DIR" >&2
  exit 2
fi

if [[ "$SKIP_PREFLIGHT" != "true" ]]; then
  if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required but not found in PATH" >&2
    exit 2
  fi

  python3 - <<'PY'
mods = ["numpy", "pandas", "PIL", "torch", "torchvision", "einops", "tqdm", "sklearn"]
missing = []
for m in mods:
    try:
        __import__(m)
    except Exception:
        missing.append(m)
if missing:
    raise SystemExit("Missing Python packages: " + ", ".join(missing))
print("[ok] Python package preflight passed")
PY
fi

if [[ "$SKIP_DOWNLOAD" != "true" ]]; then
  bash "${REPO_ROOT}/tools/download_medical_datasets.sh" \
    --npz_dir "$NPZ_DIR" \
    --data_root "$DATA_ROOT" \
    --datasets "$DATASETS" \
    --size "$SIZE" \
    --organ_variant "$ORGAN_VARIANT" \
    --breakhis_source "$BREAKHIS_SOURCE"
fi

IFS=',' read -r -a DATASET_ITEMS <<< "$DATASETS"
for dataset in "${DATASET_ITEMS[@]}"; do
  dataset="$(echo "$dataset" | sed 's/^ *//; s/ *$//')"
  IFS=';' read -r -a PLACEMENT_ITEMS <<< "$LORA_PLACEMENTS"
  for placement in "${PLACEMENT_ITEMS[@]}"; do
    placement="$(echo "$placement" | sed 's/^ *//; s/ *$//')"
    for lora_r in ${LORA_RANKS//,/ }; do
      for seed in ${SEEDS//,/ }; do
        CMD=(
          python3 "${REPO_ROOT}/tools/train_medmnist_peft.py"
          --npz_dir "$NPZ_DIR"
          --dataset_root "$DATA_ROOT"
          --dataset "$dataset"
          --dino_ckpt "$DINO_CKPT"
          --repo_dir "$REPO_DIR"
          --dino_size "$DINO_SIZE"
          --input_size "$INPUT_SIZE"
          --epochs "$EPOCHS"
          --batch_size "$BATCH_SIZE"
          --num_workers "$NUM_WORKERS"
          --lr "$LR"
          --weight_decay "$WEIGHT_DECAY"
          --val_ratio "$VAL_RATIO"
          --test_ratio "$TEST_RATIO"
          --adapter "lora"
          --lora_r "$lora_r"
          --lora_placement "$placement"
          --seed "$seed"
          --out_csv "$OUT_CSV"
        )
        if [[ -n "$BREAKHIS_MAG" ]]; then
          CMD+=(--breakhis_mag "$BREAKHIS_MAG")
        fi
        if [[ "$WANDB" == "true" ]]; then
          CMD+=(--wandb --wandb_project "$WANDB_PROJECT")
          if [[ -n "$WANDB_ENTITY" ]]; then
            CMD+=(--wandb_entity "$WANDB_ENTITY")
          fi
        fi

        echo "[run] dataset=${dataset} placement=${placement} lora_r=${lora_r} seed=${seed}"
        if [[ "$DRY_RUN" == "true" ]]; then
          printf ' %q' "${CMD[@]}"
          echo
        else
          "${CMD[@]}"
        fi
      done
    done
  done
done

echo "[ok] Medical LoRA classification grid finished"
