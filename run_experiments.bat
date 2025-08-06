@echo on

python run_experiments_groups_series.py ^
  --hidden_sizes 1 2 4 6 8 10 ^
  --lookbacks   1 2 4 6 8 10^
  --lr           0.001 ^
  --output_size 1 2 3 4 5 6^
  --out_root     results

pause
