#!/usr/bin/env bash
set -euo pipefail

python run_experiments.py \
  --hidden_sizes 8 \
  --lookbacks    1 \
  --epochs       80 \
  --lr           0.001 \
  --out_root     results
