PARTITION=VA
srun --partition=$PARTITION --mpi=pmi2 -n 2 --gres=gpu:2 --ntasks-per-node=8 --job-name=EVAL --kill-on-bad-exit=1 python3 tools/offset_track_simsam.py --cfg configs/deformable_macthtrack_simsam_ref.yaml \
     MODEL.RESUME_PATH output/matchtrack_simsam_refid_mot_DeformableMatchTrack_epoch025_checkpoint.pth
     # matchtrack_15_bidetach_x_DeformableMatchTrack_epoch002_checkpoint.pth
     # matchtrack_15_DeformableMatchTrack_epoch012_checkpoint.pth
     # matchtrack_15_nodetach_DeformableMatchTrack_epoch008_checkpoint.pth
     # MODEL.RESUME_PATH output/deformable_track_DeformableTrack_epoch020_checkpoint.pth
     # MODEL.RESUME_PATH output/deformable_track_refine_DeformableTrack_epoch015_checkpoint.pth