@echo on

python run_experiments.py ^
  --hidden_sizes 64 ^
  --lookbacks    32 ^
  --epochs       10 ^
  --lr           0.001 ^
  --out_root     results

pause
