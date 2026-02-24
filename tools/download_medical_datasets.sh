#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

NPZ_DIR="${REPO_ROOT}/medmnist_npz"
DATA_ROOT="${REPO_ROOT}/data"
DATASETS="pneumoniamnist,dermamnist,bloodmnist,organmnist,retinamnist,aptos2019,ham10000,breakhis"
SIZE="28"
ORGAN_VARIANT="a"
BREAKHIS_SOURCE="official"  # official|kaggle

while [[ $# -gt 0 ]]; do
  case "$1" in
    --npz_dir) NPZ_DIR="$2"; shift 2;;
    --data_root) DATA_ROOT="$2"; shift 2;;
    --datasets) DATASETS="$2"; shift 2;;
    --size) SIZE="$2"; shift 2;;
    --organ_variant) ORGAN_VARIANT="$2"; shift 2;;
    --breakhis_source) BREAKHIS_SOURCE="$2"; shift 2;;
    -h|--help)
      cat <<'EOF'
Download medical classification datasets used by the LoRA grid.

Usage:
  bash tools/download_medical_datasets.sh [options]

Options:
  --npz_dir PATH             MedMNIST npz output directory (default: ./medmnist_npz)
  --data_root PATH           Root for folder datasets (default: ./data)
  --datasets CSV             Dataset list, tokens or friendly names
  --size 28|64|128|224       MedMNIST resolution variant (default: 28)
  --organ_variant a|c|s      OrganMNIST variant for MedMNIST download (default: a)
  --breakhis_source SRC      official|kaggle (default: official)

Supported dataset tokens:
  pneumoniamnist, dermamnist, bloodmnist, organmnist, retinamnist,
  aptos2019, ham10000, breakhis
EOF
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

normalize_ds() {
  local x
  x="$(echo "$1" | tr -cd '[:alnum:]' | tr '[:upper:]' '[:lower:]')"
  case "$x" in
    pneumoniamnist) echo "pneumoniamnist" ;;
    dermamnist|dermamnistmedmnist) echo "dermamnist" ;;
    bloodmnist|bloodmnistmedmnist) echo "bloodmnist" ;;
    organmnist|organamnist|organcmnist|organsmnist) echo "organmnist" ;;
    retinamnist) echo "retinamnist" ;;
    aptos2019|aptos2019blindnessdetection) echo "aptos2019" ;;
    ham10000|skincancermnistham10000) echo "ham10000" ;;
    breakhis) echo "breakhis" ;;
    *) echo "$x" ;;
  esac
}

need_kaggle="false"
need_medmnist="false"
need_aptos="false"
need_ham="false"
need_breakhis="false"
medmnist_list=()

IFS=',' read -r -a DATASET_ITEMS <<< "$DATASETS"
for ds in "${DATASET_ITEMS[@]}"; do
  ds="$(echo "$ds" | sed 's/^ *//; s/ *$//')"
  tok="$(normalize_ds "$ds")"
  case "$tok" in
    pneumoniamnist|dermamnist|bloodmnist|organmnist|retinamnist)
      need_medmnist="true"
      medmnist_list+=("$tok")
      ;;
    aptos2019)
      need_aptos="true"
      need_kaggle="true"
      ;;
    ham10000)
      need_ham="true"
      need_kaggle="true"
      ;;
    breakhis)
      need_breakhis="true"
      if [[ "$BREAKHIS_SOURCE" == "kaggle" ]]; then
        need_kaggle="true"
      fi
      ;;
    *)
      echo "Unknown dataset token/name: '$ds'" >&2
      exit 2
      ;;
  esac
done

if [[ "$need_kaggle" == "true" ]]; then
  if ! command -v kaggle >/dev/null 2>&1; then
    echo "kaggle CLI is required for selected downloads but not found in PATH." >&2
    exit 2
  fi
fi

if [[ "$need_medmnist" == "true" ]]; then
  unique_medmnist_csv="$(printf "%s\n" "${medmnist_list[@]}" | awk '!seen[$0]++' | paste -sd, -)"
  bash "${REPO_ROOT}/tools/download_medmnist.sh" \
    --out_dir "$NPZ_DIR" \
    --datasets "$unique_medmnist_csv" \
    --size "$SIZE" \
    --organ_variant "$ORGAN_VARIANT"
fi

if [[ "$need_aptos" == "true" ]]; then
  aptos_dir="${DATA_ROOT}/aptos2019"
  mkdir -p "$aptos_dir"
  if [[ -f "${aptos_dir}/train.csv" && -d "${aptos_dir}/train_images" ]]; then
    echo "[skip] APTOS already present: ${aptos_dir}"
  else
    echo "[download] APTOS 2019 -> ${aptos_dir}"
    kaggle competitions download -c aptos2019-blindness-detection -p "$aptos_dir"
    unzip -qo "${aptos_dir}/aptos2019-blindness-detection.zip" -d "$aptos_dir"
  fi
fi

if [[ "$need_ham" == "true" ]]; then
  ham_dir="${DATA_ROOT}/ham10000"
  mkdir -p "$ham_dir"
  if [[ -f "${ham_dir}/HAM10000_metadata.csv" ]]; then
    echo "[skip] HAM10000 already present: ${ham_dir}"
  else
    echo "[download] HAM10000 -> ${ham_dir}"
    kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p "$ham_dir"
    unzip -qo "${ham_dir}/skin-cancer-mnist-ham10000.zip" -d "$ham_dir"
  fi
fi

if [[ "$need_breakhis" == "true" ]]; then
  breakhis_dir="${DATA_ROOT}/breakhis"
  mkdir -p "$breakhis_dir"
  if find "$breakhis_dir" -type f -name "*.png" -print -quit | grep -q .; then
    echo "[skip] BreakHis already present: ${breakhis_dir}"
  else
    if [[ "$BREAKHIS_SOURCE" == "official" ]]; then
      echo "[download] BreakHis (official) -> ${breakhis_dir}"
      curl -fL -o "${breakhis_dir}/BreaKHis_v1.tar.gz" "http://www.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz"
      tar -xzf "${breakhis_dir}/BreaKHis_v1.tar.gz" -C "$breakhis_dir"
    elif [[ "$BREAKHIS_SOURCE" == "kaggle" ]]; then
      echo "[download] BreakHis (kaggle) -> ${breakhis_dir}"
      kaggle datasets download -d ambarish/breakhis -p "$breakhis_dir"
      unzip -qo "${breakhis_dir}/breakhis.zip" -d "$breakhis_dir"
    else
      echo "Invalid --breakhis_source '${BREAKHIS_SOURCE}'. Use 'official' or 'kaggle'." >&2
      exit 2
    fi
  fi
fi

echo "[ok] Medical dataset download step finished"
