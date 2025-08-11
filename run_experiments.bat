@echo on

python run_experiments_groups_series.py ^
  --thresholds   0.1 ^
  --hidden_sizes 1 ^
  --lookbacks   1^
  --lr           0.001 ^
  --output_size 1 ^
  --out_root     results

pause
