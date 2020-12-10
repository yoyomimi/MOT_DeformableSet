PARTITION=VA
srun --partition=$PARTITION --mpi=pmi2 -n 1 --gres=gpu:1 --ntasks-per-node=1 --job-name=DET_VISUAL --kill-on-bad-exit=1 python3 tools/visual.py
