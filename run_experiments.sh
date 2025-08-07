set -euo pipefail

python run_experiments_groups_series.py \
  --thresholds   0.05 0.1 0.3 0.5 0.6 0.8 \
  --hidden_sizes 4 \
  --lookbacks    4 \
  --lr           0.0001 \
  --output_sizes 1 \
  --out_root     results
