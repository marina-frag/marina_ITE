@echo on

python run_experiments.py ^
<<<<<<< Updated upstream
  --hidden_sizes 10 ^
  --lookbacks    10 ^
=======
  --hidden_sizes 16 ^
  --lookbacks    1 ^
>>>>>>> Stashed changes
  --lr           0.001 ^
  --out_root     results

pause
