@echo on

python run_experiments.py ^
  --hidden_sizes 10 ^
  --lookbacks   10 ^
  --epochs       8 ^
  --lr           0.001 ^
  --out_root     results

pause
