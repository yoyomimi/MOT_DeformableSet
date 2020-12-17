from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import cv2
import importlib
import logging
import motmetrics as mm
import numpy as np
import os
import os.path as osp
from PIL import Image

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.transforms.functional as F_trans

import _init_paths
from libs.datasets.collate import collect
from libs.datasets.transform import EvalTransform
from libs.models.tracker.multitracker import JDETracker
from libs.tracking_utils import visualization as vis
from libs.tracking_utils.timer import Timer
from libs.tracking_utils.evaluation import Evaluator
from libs.utils.box_ops import hard_nms
from libs.utils.utils import create_logger
from libs.utils.utils import get_model
from libs.utils.utils import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='HOI Transformer Task')
    parser.add_argument(
        '--cfg',
        dest='yaml_file',
        help='experiment configure file name, e.g. configs/track.yaml',
        required=True,
        type=str)    
    # default distributed training
    parser.add_argument(
        '--distributed',
        action='store_true',
        default=False,
        help='if use distribute train')
    parser.add_argument(
        '--dist-url',
        dest='dist_url',
        default='tcp://10.5.38.36:23456',
        type=str,
        help='url used to set up distributed training')
    parser.add_argument(
        '--world-size',
        dest='world_size',
        default=1,
        type=int,
        help='number of nodes for distributed training')
    parser.add_argument(
        '--rank',
        default=0,
        type=int,
        help='node rank for distributed training, machine level')
    parser.add_argument(
        'opts',
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER)
    args = parser.parse_args()
          
    return args

def get_ip(ip_addr):
    ip_list = ip_addr.split('-')[2:6]
    for i in range(4):
        if ip_list[i][0] == '[':
            ip_list[i] = ip_list[i][1:].split(',')[0]
    return f'tcp://{ip_list[0]}.{ip_list[1]}.{ip_list[2]}.{ip_list[3]}:23456'

def read_img(img_path):
    image = Image.open(img_path).convert('RGB')
    image, _ = resize(image, None, 800, max_size=1333)
    image = F_trans.to_tensor(image)
    image = F_trans.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return image

def process_img(img_path, model, postprocessors, device, threshold=0.12, nms_threshold=0.5):
    model.eval()
    ori_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    h, w = ori_img.shape[1:]
    size = torch.from_numpy(np.array([h, w]))
    image = read_img(img_path)
    imgs = [image.to(device)]
    bs = len(imgs)
    target_sizes = size.expand(bs, 2)
    target_sizes = target_sizes.to(device)
    outputs_dict = model(imgs)
    pred_out = postprocessors(outputs_dict, img_path,
        target_sizes)
    res = pred_out[-1]
    valid_inds = torch.where(res['scores']>threshold)[0]
    boxes = res['boxes'][valid_inds].clamp(min=0)
    labels = res['labels'][valid_inds]
    scores = res['scores'][valid_inds]
    valid_id = torch.where(labels==1)[0]
    boxes_with_scores = torch.cat([boxes[valid_id].reshape(
        -1, 4), scores[valid_id].reshape(-1, 1)], dim=1)
    keep = hard_nms(boxes_with_scores, nms_threshold, return_pick=True)
    boxes_np = boxes_with_scores[keep, :4].reshape(
        -1, 4).data.cpu().numpy()
    scores_np = boxes_with_scores[keep, -1].reshape(
        -1, 1).data.cpu().numpy()
    # TODO
    return ori_img, dets, id_feature

def write_results(filename, results, data_type, logger=None):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
    if logger is not None:
        logger.info('save results to {}'.format(filename))


def write_results_score(filename, results, data_type, logger=None):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},{s},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h, s=score)
                f.write(line)
    if logger is not None:
        logger.info('save results to {}'.format(filename))


def eval_seq(cfg, device, img_path_list, model, postprocessors, data_type,
             result_filename, save_dir=None, show_image=True,
             min_box_area=100., frame_rate=30, use_cuda=True, logger=None):
    if logger is not None and save_dir and not osp.exists(save_dir)
        os.makedirs(save_dir)
    tracker = JDETracker(cfg.DATASET.MEAN, cfg.DATASET.STD, frame_rate=frame_rate)
    timer = Timer()
    results = []
    frame_id = 0
    for i, path in enumerate(img_path_list):
        #if i % 8 != 0:
            #continue
        if logger is not None and frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        # run tracking
        timer.tic()
        ori_img, dets, id_feature = process_img(path, model, postprocessors, device)
        online_targets = tracker.update(dets, id_feature)
        online_tlwhs = []
        online_ids = []
        #online_scores = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                #online_scores.append(t.score)
        timer.toc()
        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids))
        #results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))
        if show_image or save_dir is not None:
            online_im = vis.plot_tracking(ori_img, online_tlwhs, online_ids, frame_id=frame_id,
                                          fps=1. / timer.average_time)
        if show_image:
            cv2.imshow('online_im', online_im)
        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
        frame_id += 1
    # save results
    write_results(result_filename, results, data_type)
    #write_results_score(result_filename, results, data_type)
    return frame_id, timer.average_time, timer.calls


def main(data_root='/data/MOT16/train', seqs=('MOT16-05',), exp_name='demo',
         save_images=False, save_videos=False, show_image=True):
    args = parse_args()
    update_config(cfg, args)
    ngpus_per_node = torch.cuda.device_count()
    if 'SLURM_PROCID' in os.environ.keys():
        proc_rank = int(os.environ['SLURM_PROCID'])
        local_rank = proc_rank % ngpus_per_node
        args.world_size = int(os.environ['SLURM_NTASKS'])
    else:
        proc_rank = 0
        local_rank = 0
        args.world_size = 1
    args.distributed = (args.world_size > 1 or args.distributed)
    #create logger
    if proc_rank == 0:
        logger, root_output_dir = create_logger(cfg, proc_rank)
    else:
        logger = None
    # distribution
    if args.distributed:
        dist_url = get_ip(os.environ['SLURM_STEP_NODELIST'])
        if proc_rank == 0:
            logger.info(
                f'Init process group: dist_url: {dist_url},  '
                f'world_size: {args.world_size}, '
                f'proc_rank: {proc_rank}, '
                f'local_rank:{local_rank}'
            )  
        dist.init_process_group(
            backend=cfg.DIST_BACKEND,
            init_method=dist_url,
            world_size=args.world_size,
            rank=proc_rank
        )
        torch.distributed.barrier()
        torch.cuda.set_device(local_rank)
        device = torch.device(cfg.DEVICE)
        model, criterion, postprocessors = get_model(cfg, device)  
        model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True
        )
    else:
        assert proc_rank == 0, ('proc_rank != 0, it will influence '
                                'the evaluation procedure')
        if cfg.DEVICE == 'cuda':
            torch.cuda.set_device(local_rank)
        device = torch.device(cfg.DEVICE)
        model, criterion, postprocessors = get_model(cfg, device)  
        model = torch.nn.DataParallel(model).to(device)
    
    model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    resume_path = cfg.MODEL.RESUME_PATH
    if os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location='cpu')
        # resume
        if 'state_dict' in checkpoint:
            model.module.load_state_dict(checkpoint['state_dict'], strict=True)
            logging.info(f'==> model loaded from {resume_path} \n')

    if logger is not None:
        logger.setLevel(logging.INFO)
        result_root = os.path.join(root_output_dir, 'results', exp_name)
        if not osp.exists(result_root):
            os.makedirs(result_root)

    data_type = 'mot'
    # run tracking
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []
    for seq in seqs:
        output_dir = osp.join(result_root, seq) if save_images or save_videos else None
        if logger is not None:
            logger.info('start seq: {}'.format(seq))
        device = torch.device(cfg.DEVICE)
        img_path_list = os.listdir(osp.join(data_root, seq, 'img1'))
        nf, ta, tc = eval_seq(cfg, device, img_path_list, model, postprocessors, data_type,
            result_filename, save_dir=output_dir, show_image=show_image, frame_rate=frame_rate)
        n_frame += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)

        # eval
        if logger is not None:
            logger.info('Evaluate seq: {}'.format(seq))
        evaluator = Evaluator(data_root, seq, data_type)
        accs.append(evaluator.eval_file(result_filename))
        if save_videos:
            output_video_path = osp.join(output_dir, '{}.mp4'.format(seq))
            cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}'.format(output_dir, output_video_path)
            os.system(cmd_str)
    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    if logger is not None:
        logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))

    # get summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    Evaluator.save_summary(summary, os.path.join(result_root, 'summary_{}.xlsx'.format(exp_name)))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # TODO
    # cfg = cfgs().init()



    seqs_str = '''MOT16-02
                  MOT16-04
                  MOT16-05
                  MOT16-09
                  MOT16-10
                  MOT16-11
                  MOT16-13'''
    data_root = os.path.join(cfg.data_dir, 'MOT16/train')
    if cfg.test_mot16:
        seqs_str = '''MOT16-01
                      MOT16-03
                      MOT16-06
                      MOT16-07
                      MOT16-08
                      MOT16-12
                      MOT16-14'''
        #seqs_str = '''MOT16-01 MOT16-07 MOT16-12 MOT16-14'''
        #seqs_str = '''MOT16-06 MOT16-08'''
        data_root = os.path.join(cfg.data_dir, 'MOT16/test')
    if cfg.test_mot15:
        seqs_str = '''ADL-Rundle-1
                      ADL-Rundle-3
                      AVG-TownCentre
                      ETH-Crossing
                      ETH-Jelmoli
                      ETH-Linthescher
                      KITTI-16
                      KITTI-19
                      PETS09-S2L2
                      TUD-Crossing
                      Venice-1'''
        data_root = os.path.join(cfg.data_dir, 'MOT15/images/test')
    if cfg.test_mot17:
        seqs_str = '''MOT17-01-SDP
                      MOT17-03-SDP
                      MOT17-06-SDP
                      MOT17-07-SDP
                      MOT17-08-SDP
                      MOT17-12-SDP
                      MOT17-14-SDP'''
        #seqs_str = '''MOT17-01-SDP
                      #MOT17-06-SDP
                      #MOT17-07-SDP
                      #MOT17-12-SDP
                      #'''
        #seqs_str = '''MOT17-01-SDP MOT17-07-SDP MOT17-12-SDP MOT17-14-SDP'''
        #seqs_str = '''MOT17-03-SDP'''
        #seqs_str = '''MOT17-06-SDP MOT17-08-SDP'''
        data_root = os.path.join(cfg.data_dir, 'MOT17/images/test')
    if cfg.val_mot17:
        seqs_str = '''MOT17-02-SDP
                      MOT17-04-SDP
                      MOT17-05-SDP
                      MOT17-09-SDP
                      MOT17-10-SDP
                      MOT17-11-SDP
                      MOT17-13-SDP'''
        #seqs_str = '''MOT17-02-SDP'''
        data_root = os.path.join(cfg.data_dir, 'MOT17/images/train')
    if cfg.val_mot15:
        seqs_str = '''Venice-2
                      KITTI-13
                      KITTI-17
                      ETH-Bahnhof
                      ETH-Sunnyday
                      PETS09-S2L1
                      TUD-Campus
                      TUD-Stadtmitte
                      ADL-Rundle-6
                      ADL-Rundle-8
                      ETH-Pedcross2
                      TUD-Stadtmitte'''
        #seqs_str = '''Venice-2'''
        data_root = os.path.join(cfg.data_dir, 'MOT15/images/train')
    if cfg.val_mot20:
        seqs_str = '''MOT20-01
                      MOT20-02
                      MOT20-03
                      MOT20-05
                      '''
        data_root = os.path.join(cfg.data_dir, 'MOT20/images/train')
    if cfg.test_mot20:
        seqs_str = '''MOT20-04
                      MOT20-06
                      MOT20-07
                      MOT20-08
                      '''
        data_root = os.path.join(cfg.data_dir, 'MOT20/images/test')
    seqs = [seq.strip() for seq in seqs_str.split()]

    main(cfg,
         data_root=data_root,
         seqs=seqs,
         exp_name='MOT17_test_public_dla34',
         show_image=False,
         save_images=False,
         save_videos=False)