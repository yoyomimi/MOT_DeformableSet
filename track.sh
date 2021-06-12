PARTITION=pat_largescale
srun --partition=$PARTITION --mpi=pmi2 -n 1 --gres=gpu:1 --ntasks-per-node=8 --job-name=EVAL --kill-on-bad-exit=1 python tools/offset_track.py --cfg configs/deformable_macthtrack_ch.yaml \
     MODEL.RESUME_PATH output/matchtrack_15_ch_6w_layer6_joint_DeformableMatchTrack_epoch010_checkpoint.pth
# srun --partition=$PARTITION --mpi=pmi2 -n 1 --gres=gpu:1 --ntasks-per-node=8 --job-name=EVAL --kill-on-bad-exit=1 -x BJ-IDC1-10-10-16-[53,54,58,59,62,64,68,86,85] python3 tools/half_offset_track.py --cfg configs/deformable_macthtrack.yaml \
#      MODEL.RESUME_PATH output/matchtrack_halfmot17_fix_posnorm_bugs_norefidx_layer6_DeformableMatchTrack_epoch025_checkpoint.pth
     # matchtrack_fullmot17_fix_posnorm_bugs_norefidx_layer6_DeformableMatchTrack_epoch025_checkpoint.pth
     # matchtrack_fullmot17_fix_posnorm_bugs_norefidx_layer6_iterative_DeformableMatchTrack_epoch025_checkpoint.pth
     # matchtrack_halfmot17_fix_posnorm_bugs_norefidx_DeformableMatchTrack_epoch025_checkpoint.pth
     # matchtrack_halfmot17_fix_posnorm_bugs_norefidx_layer6_DeformableMatchTrack_epoch025_checkpoint.pth
     # matchtrack_halfmot17_fix_posnorm_bugs_norefidx_layer6_DeformableMatchTrack_epoch025_checkpoint.pth
     # matchtrack_halfmot17_fix_posnorm_bugs_norefidx_iterative_DeformableMatchTrack_epoch025_checkpoint.pth