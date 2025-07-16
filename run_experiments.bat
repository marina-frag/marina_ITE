@echo on

python run_experiments.py ^
  --hidden_sizes 16 ^
  --lookbacks    1 ^
  --epochs       5 ^
  --lr           0.001 ^
  --out_root     results

pause
