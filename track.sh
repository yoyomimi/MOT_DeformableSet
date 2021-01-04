PARTITION=VA
MASTER_PORT=22263
srun --partition=$PARTITION --mpi=pmi2 -n 8 --gres=gpu:8 --ntasks-per-node=8 --job-name=EVAL --kill-on-bad-exit=1 python3 tools/match_track_simple.py --cfg configs/deformable_track_qp_test.yaml \
     MODEL.RESUME_PATH output/deformable_track_qp_fixid_DeformableQPTrack_epoch010_checkpoint.pth
     # MODEL.RESUME_PATH output/deformable_track_DeformableTrack_epoch020_checkpoint.pth
     # MODEL.RESUME_PATH output/deformable_track_refine_DeformableTrack_epoch015_checkpoint.pth