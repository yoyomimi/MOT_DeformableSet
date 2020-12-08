srun --partition=VA_Test --mpi=pmi2 -n 1 --gres=gpu:1 --ntasks-per-node=1 \
 --job-name=MOT_TEST --kill-on-bad-exit=1 python tools/visual.py
cp test_out/MOT17-02-SDP/result.txt result/MOT17/MOT17-02-SDP.txt
python -m motmetrics.apps.evaluateTracking  \
 /mnt/lustre/chenmingfei/code/Puppet_MOT/data/MOT17/train \
  /mnt/lustre/chenmingfei/code/Puppet_MOT/result/MOT17 \
   /mnt/lustre/chenmingfei/code/Puppet_MOT/py-motmetrics/seqmaps/MOT17-train.txt