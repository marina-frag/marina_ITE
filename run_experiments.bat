@echo on

python run_experiments.py ^
  --hidden_sizes 1 ^
  --lookbacks   1 ^
  --lr           0.001 ^
  --out_root     results

pause
