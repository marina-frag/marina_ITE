@echo on

python run_experiments.py ^
  --hidden_sizes 10 ^
  --lookbacks    10 ^
  --lr           0.001 ^
  --out_root     results

pause
