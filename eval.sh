PARTITION=VA
srun --partition=$PARTITION --mpi=pmi2 -n 1 --gres=gpu:1 --ntasks-per-node=1 --job-name=HOI_EVAL --kill-on-bad-exit=1 python3 tools/eval.py --cfg configs/deformable_det.yaml \
    MODEL.RESUME_PATH /mnt/lustre/chenmingfei/code/MOT_DeformableSet/output/deformable_det_more_DeformableDETR_epoch030_checkpoint.pth