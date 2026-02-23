#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="./medmnist_npz"
SIZE="28"
ORGAN_VARIANT="a"
DATASETS="pneumoniamnist,dermamnist,bloodmnist,organmnist,retinamnist"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out_dir) OUT_DIR="$2"; shift 2;;
    --size) SIZE="$2"; shift 2;;
    --organ_variant) ORGAN_VARIANT="$2"; shift 2;;
    --datasets) DATASETS="$2"; shift 2;;
    -h|--help)
      cat <<'EOF'
Download MedMNIST .npz files needed by the PEFT MedMNIST sweep.

Usage:
  bash tools/download_medmnist.sh [options]

Options:
  --out_dir PATH            Output directory for .npz files (default: ./medmnist_npz)
  --size 28|64|128|224      MedMNIST image size variant (default: 28)
  --organ_variant a|c|s     OrganMNIST view: a=axial, c=coronal, s=sagittal (default: a)
  --datasets CSV            Comma-separated list from:
                            pneumoniamnist,dermamnist,bloodmnist,organmnist,retinamnist
                            or friendly names:
                            PneumoniaMNIST,DermaMNIST (MedMNIST),BloodMNIST (MedMNIST),OrganMNIST,RetinaMNIST
EOF
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

if [[ "$SIZE" != "28" && "$SIZE" != "64" && "$SIZE" != "128" && "$SIZE" != "224" ]]; then
  echo "Invalid --size '$SIZE'. Allowed: 28,64,128,224" >&2
  exit 2
fi

if [[ "$ORGAN_VARIANT" != "a" && "$ORGAN_VARIANT" != "c" && "$ORGAN_VARIANT" != "s" ]]; then
  echo "Invalid --organ_variant '$ORGAN_VARIANT'. Allowed: a,c,s" >&2
  exit 2
fi

SUFFIX=""
if [[ "$SIZE" != "28" ]]; then
  SUFFIX="_${SIZE}"
fi

mkdir -p "$OUT_DIR"

download_npz() {
  local remote_name="$1"
  local out_name="$2"
  local url="https://zenodo.org/records/10519652/files/${remote_name}.npz?download=1"
  local out_path="${OUT_DIR}/${out_name}.npz"
  echo "[download] ${url} -> ${out_path}"
  curl -fsSL "$url" -o "$out_path"
}

normalize_ds() {
  local x
  x="$(echo "$1" | tr -cd '[:alnum:]' | tr '[:upper:]' '[:lower:]')"
  case "$x" in
    pneumoniamnist) echo "pneumoniamnist" ;;
    dermamnist|dermamnistmedmnist) echo "dermamnist" ;;
    bloodmnist|bloodmnistmedmnist) echo "bloodmnist" ;;
    organmnist|organamnist|organcmnist|organsmnist) echo "organmnist" ;;
    retinamnist) echo "retinamnist" ;;
    *) echo "$x" ;;
  esac
}

IFS=',' read -r -a DATASET_ITEMS <<< "$DATASETS"
for ds in "${DATASET_ITEMS[@]}"; do
  ds="$(echo "$ds" | sed 's/^ *//; s/ *$//')"
  ds="$(normalize_ds "$ds")"
  case "$ds" in
    pneumoniamnist)
      download_npz "pneumoniamnist${SUFFIX}" "pneumoniamnist"
      ;;
    dermamnist)
      download_npz "dermamnist${SUFFIX}" "dermamnist"
      ;;
    bloodmnist)
      download_npz "bloodmnist${SUFFIX}" "bloodmnist"
      ;;
    retinamnist)
      download_npz "retinamnist${SUFFIX}" "retinamnist"
      ;;
    organmnist)
      case "$ORGAN_VARIANT" in
        a) download_npz "organamnist${SUFFIX}" "organmnist" ;;
        c) download_npz "organcmnist${SUFFIX}" "organmnist" ;;
        s) download_npz "organsmnist${SUFFIX}" "organmnist" ;;
      esac
      ;;
    *)
      echo "Unknown dataset token: '$ds'" >&2
      exit 2
      ;;
  esac
done

echo "[ok] Downloaded requested datasets into ${OUT_DIR}"
