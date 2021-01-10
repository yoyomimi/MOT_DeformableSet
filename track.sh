PARTITION=Test
srun --partition=$PARTITION --mpi=pmi2 -n 2 --gres=gpu:2 --ntasks-per-node=8 --job-name=EVAL --kill-on-bad-exit=1 python3 tools/offset_track.py --cfg configs/deformable_macthtrack_test.yaml \
     MODEL.RESUME_PATH output/matchtrack_15_nodetach_DeformableMatchTrack_epoch005_checkpoint.pth
     # matchtrack_15_DeformableMatchTrack_epoch012_checkpoint.pth
     # MODEL.RESUME_PATH output/deformable_track_DeformableTrack_epoch020_checkpoint.pth
     # MODEL.RESUME_PATH output/deformable_track_refine_DeformableTrack_epoch015_checkpoint.pth