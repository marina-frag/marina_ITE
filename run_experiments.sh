set -euo pipefail

python run_experiments.py \
  --hidden_sizes 128 \
  --lookbacks    1 \
  --lr           0.001 \
  --out_root     results
