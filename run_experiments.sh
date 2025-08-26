set -euo pipefail

python run_experiments_groups_series.py \
  --thresholds   0.5 \
  --hidden_sizes 2 \
  --lookbacks    2 \
  --lr           0.0001 \
  --output_sizes 1 \
  --out_root     results
