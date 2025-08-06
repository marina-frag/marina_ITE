set -euo pipefail

python run_experiments_groups_series.py \
  --hidden_sizes 1 2 4 8 12 16 \
  --lookbacks    4 \
  --lr           0.0001 \
  --output_sizes 1 \
  --out_root     results
