rm -rf work_dirs
rm -rf output

python3 tools/train.py --cfg configs/deformable_det.yaml

# PARTITION=VA_Test
# srun --partition=$PARTITION --mpi=pmi2 -n 8 --gres=gpu:8 --ntasks-per-node=8 --job-name=MOT_DEBUG --kill-on-bad-exit=1 python3 tools/train.py --cfg configs/MOT_Transformer_base.yaml

# PARTITION=VA
# srun --partition=$PARTITION --mpi=pmi2 -n 8 --gres=gpu:8 --ntasks-per-node=8 --job-name=Deform_DETR --kill-on-bad-exit=1 python tools/train.py --cfg configs/deformable_det.yaml
