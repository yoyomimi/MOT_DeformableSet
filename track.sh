PARTITION=VA
srun --partition=$PARTITION --mpi=pmi2 -n 8 --gres=gpu:8 --ntasks-per-node=8 --job-name=EVAL --kill-on-bad-exit=1 python3 tools/match_track_simple.py --cfg configs/deformable_track_warp_test.yaml \
     MODEL.RESUME_PATH output/deformable_track_warp_right_DeformableTrack_epoch030_checkpoint.pth
     # MODEL.RESUME_PATH output/deformable_track_DeformableTrack_epoch020_checkpoint.pth
     # MODEL.RESUME_PATH output/deformable_track_refine_DeformableTrack_epoch015_checkpoint.pth