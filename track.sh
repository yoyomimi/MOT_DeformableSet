PARTITION=VA
srun --partition=$PARTITION --mpi=pmi2 -n 8 --gres=gpu:8 --ntasks-per-node=8 --job-name=EVAL --kill-on-bad-exit=1 python3 tools/track.py --cfg configs/deformable_track_single_test.yaml \
     MODEL.RESUME_PATH /mnt/lustre/chenmingfei/code/MOT_DeformableSet/output/deformable_motid_right_DeformableBaseTrack_epoch015_checkpoint.pth
    # MODEL.RESUME_PATH /mnt/lustre/chenmingfei/code/MOT_DeformableSet/output/deformable_motid_DeformableBaseTrack_epoch040_checkpoint.pth
    # MODEL.RESUME_PATH /mnt/lustre/chenmingfei/code/MOT_DeformableSet/output/deformable_step2_DeformableBaseTrack_epoch040_checkpoint.pth