import os
import motmetrics as mm

import _init_paths
from libs.tracking_utils.evaluation import Evaluator


data_dir = '/mnt/lustre/chenmingfei/code/MOT_DeformableSet'
seqs_str = ['MOT17-02-SDP',
            'MOT17-04-SDP',
            'MOT17-05-SDP',
            'MOT17-09-SDP',
            'MOT17-10-SDP',
            'MOT17-11-SDP',
            'MOT17-13-SDP',]
data_root = os.path.join(data_dir, 'data/MOT17/train')
# result_root = '/mnt/lustre/chenmingfei/code/MOT_DeformableSet/work_dirs/deformable_track_qp_nearfix_lightid_test/2021-01-04-09-53/results/test/'
# result_root = '/mnt/lustre/chenmingfei/code/MOT_DeformableSet/work_dirs/deformable_track_qp_nearfix_test/2021-01-04-09-57/results/test/'
result_root = '/mnt/lustre/chenmingfei/code/MOT_DeformableSet/work_dirs/deformable_track_qp_nearfix_test/2021-01-04-09-55/results/test/'
accs = []
for seq in seqs_str:
    result_filename = result_root + seq + '.txt'
    evaluator = Evaluator(data_root, seq, 'mot')
    accs.append(evaluator.eval_file(result_filename))

metrics = mm.metrics.motchallenge_metrics
mh = mm.metrics.create()
summary = Evaluator.get_summary(accs, seqs_str, metrics)
strsummary = mm.io.render_summary(
    summary,
    formatters=mh.formatters,
    namemap=mm.io.motchallenge_metric_names
)
print(strsummary)
# Evaluator.save_summary(summary, os.path.join(result_root, 'summary_{}.xlsx'.format(exp_name)))