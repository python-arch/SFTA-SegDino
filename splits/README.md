# Splits manifests

This directory stores **committed filelists** that define experiment splits (for reproducibility and strict source-free protocols).

## Manifest format
- UTF-8 text file
- One entry per line
- Empty lines and lines starting with `#` are ignored
- Each entry is a **relative path from the dataset root** (e.g. `test/images/0001.png`)

For Kvasir-SEG, manifests are generated from `segdata/kvasir/test/images/*` and written as:
- `splits/kvasir_target_adapt.txt` (unlabeled; used for adaptation)
- `splits/kvasir_target_holdout.txt` (labeled; evaluation only)

## Generator
Use `tools/make_target_splits.py` to generate these manifests deterministically.

