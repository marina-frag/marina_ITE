@echo on

python run_experiments_groups_series.py ^
  --hidden_sizes 1 ^
  --lookbacks   1 ^
  --lr           0.001 ^
  --out_root     results

pause
