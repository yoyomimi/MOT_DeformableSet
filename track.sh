PARTITION=Test
srun --partition=$PARTITION --mpi=pmi2 -n 1 --gres=gpu:1 --ntasks-per-node=1 --job-name=EVAL --kill-on-bad-exit=1 python3 tools/match_track.py --cfg configs/deformable_track.yaml \
     MODEL.RESUME_PATH output/deformable_track_refine_DeformableTrack_epoch015_checkpoint.pth