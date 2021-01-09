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
from configs import cfg
from configs import update_config
from libs.datasets.collate import collect
from libs.datasets.transform import EvalTransform
from libs.models.tracker.simple_multitracker import SimpleTracker
from libs.tracking_utils import visualization as vis
from libs.tracking_utils.timer import Timer
from libs.tracking_utils.evaluation import Evaluator
from libs.datasets.transform import resize
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

def get_warp_matrix(src, dst, warp_mode = cv2.MOTION_HOMOGRAPHY, eps = 1e-5,
        max_iter = 100, scale = None, align = False):
    """Compute the warp matrix from src to dst.
​
    Parameters
    ----------
    src : ndarray
        An NxM matrix of source img(BGR or Gray), it must be the same format as dst.
    dst : ndarray
        An NxM matrix of target img(BGR or Gray).
    warp_mode: flags of opencv
        translation: cv2.MOTION_TRANSLATION
        rotated and shifted: cv2.MOTION_EUCLIDEAN
        affine(shift,rotated,shear): cv2.MOTION_AFFINE
        homography(3d): cv2.MOTION_HOMOGRAPHY
    eps: float
        the threshold of the increment in the correlation coefficient between two iterations
    max_iter: int
        the number of iterations.
    scale: float or [int, int]
        scale_ratio: float
        scale_size: [W, H]
    align: bool
        whether to warp affine or perspective transforms to the source image
​
    Returns
    -------
    warp matrix : ndarray
        Returns the warp matrix from src to dst.
        if motion model is homography, the warp matrix will be 3x3, otherwise 2x3
    src_aligned: ndarray
        aligned source image of gray
    """
    assert src.shape == dst.shape, "the source image must be the same format to the target image!"
    # BGR2GRAY
    if src.ndim == 3:
        # Convert images to grayscale
        src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    # make the imgs smaller to speed up
    if scale is not None:
        if isinstance(scale, float) or isinstance(scale, int):
            if scale != 1:
                src_r = cv2.resize(src, (0, 0), fx = scale, fy = scale,interpolation =  cv2.INTER_LINEAR)
                dst_r = cv2.resize(dst, (0, 0), fx = scale, fy = scale,interpolation =  cv2.INTER_LINEAR)
                scale = [scale, scale]
            else:
                src_r, dst_r = src, dst
                scale = None
        else:
            if scale[0] != src.shape[1] and scale[1] != src.shape[0]:
                src_r = cv2.resize(src, (scale[0], scale[1]), interpolation = cv2.INTER_LINEAR)
                dst_r = cv2.resize(dst, (scale[0], scale[1]), interpolation=cv2.INTER_LINEAR)
                scale = [scale[0] / src.shape[1], scale[1] / src.shape[0]]
            else:
                src_r, dst_r = src, dst
                scale = None
    else:
        src_r, dst_r = src, dst
    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else :
        warp_matrix = np.eye(2, 3, dtype=np.float32)
    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, eps)
    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC (src_r, dst_r, warp_matrix, warp_mode, criteria, None, 1)
    if scale is not None:
        warp_matrix[0, 2] = warp_matrix[0, 2] / scale[0]
        warp_matrix[1, 2] = warp_matrix[1, 2] / scale[1]
    if align:
        sz = src.shape
        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            # Use warpPerspective for Homography
            src_aligned = cv2.warpPerspective(src, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR)
        else :
            # Use warpAffine for Translation, Euclidean and Affine
            src_aligned = cv2.warpAffine(src, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR)
        return warp_matrix, src_aligned
    else:
        return warp_matrix, None

def read_img(img_path):
    image = Image.open(img_path).convert('RGB')
    image, _ = resize(image, None, 608, max_size=1088)
    image = F_trans.to_tensor(image)
    image = F_trans.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return image

def process_img(img_path, model, postprocessors, device, threshold=0.35, references=None):
    model.eval()
    ori_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    h, w = ori_img.shape[:2]
    size = torch.from_numpy(np.array([h, w]))
    image = read_img(img_path)
    imgs = [image.to(device)]
    bs = len(imgs)
    target_sizes = size.expand(bs, 2)
    target_sizes = target_sizes.to(device)
    outputs_dict = model(imgs, references)
    prev_memory = model.module.out_memory.detach()
    pred_out = postprocessors(outputs_dict, img_path,
        target_sizes, references)
    res = pred_out[-1]
    valid_inds = torch.where(res['scores']>threshold)[0]
    boxes = res['boxes'][valid_inds]
    labels = res['labels'][valid_inds]
    scores = res['scores'][valid_inds]
    valid_id = torch.where(labels==1)[0]
    boxes_with_scores = torch.cat([boxes[valid_id].reshape(
        -1, 4), scores[valid_id].reshape(-1, 1)], dim=1)
    # keep = hard_nms(boxes_with_scores, nms_threshold, return_pick=True)
    dets_np = boxes_with_scores[..., :5].reshape(
        -1, 5).data.cpu().numpy()
    id_features_np = res['id_features'][valid_inds][valid_id].reshape(-1,
        res['id_features'].shape[-1]).data.cpu().numpy()
    if references is not None:
        track_idx = res['track_idx'][valid_inds][valid_id].reshape(
            -1,).data.cpu().numpy()
        ref_coords_np = res['ref_coords'].reshape(-1, 2).data.cpu().numpy()
        ref_id_features = res['ref_id_features'].reshape(-1,
            res['id_features'].shape[-1]).data.cpu().numpy()
    else:
        track_idx = np.array([]).reshape(-1,)
        ref_coords_np = np.array([]).reshape(-1, 2)
        ref_id_features = np.array([]).reshape(-1, res['id_features'].shape[-1])
    assert len(id_features_np) == len(dets_np)
    return ori_img, dets_np, id_features_np, track_idx, ref_coords_np, ref_id_features, prev_memory

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
    if logger is not None and save_dir and not osp.exists(save_dir):
        os.makedirs(save_dir)
    tracker = SimpleTracker(frame_rate=frame_rate)
    timer = Timer()
    results = []
    frame_id = 0
    references = None
    prev_img = None
    for i, path in enumerate(img_path_list):
        #if i % 8 != 0:
            #continue
        if logger is not None and frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        # run tracking
        timer.tic()
        ori_img, dets, id_feature, track_idx, ref_coords, ref_id_features, prev_memory = process_img(path, model, postprocessors,
            device, references=references)
        scale = torch.as_tensor([ori_img.shape[1], ori_img.shape[0], ori_img.shape[1], ori_img.shape[0]])
        offset_np = None
        if references is not None:
            ori_ref_boxes = (references[0]['ref_boxes'].data.cpu() * scale)[..., :2]
            offset_np = np.hstack([ori_ref_boxes.numpy(), ref_coords])

        # if prev_img is not None:
        #     warp_matrix, _ = get_warp_matrix(prev_img, ori_img)
        # else:
        #     warp_matrix = None
        
        online_targets, references = tracker.update(dets, id_feature, ref_id_features, track_idx, logger)
        
        # if prev_img is not None:
        #     warp_matrix = torch.as_tensor(warp_matrix).to(references[0]['ref_boxes'].device)
        #     x0, y0 = references[0]['ref_boxes'][..., 0], references[0]['ref_boxes'][..., 1]     
        #     warp_matrix = warp_matrix.unsqueeze(0).repeat(x0.shape[0], 1, 1).reshape(-1, 9)
        #     X = warp_matrix[..., 0] * x0 + warp_matrix[..., 1] * y0 + warp_matrix[..., 2]
        #     Y = warp_matrix[..., 3] * x0 + warp_matrix[..., 4] * y0 + warp_matrix[..., 5]
        #     Z = warp_matrix[..., 6] * x0 + warp_matrix[..., 7] * y0 + warp_matrix[..., 8]
        #     references[0]['ref_boxes'][..., :2] = torch.stack([X/Z, Y/Z], dim=-1).reshape(references[0]['ref_boxes'][..., :2].shape)

        
        references[0]['ref_boxes'] /= scale.to(references[0]['ref_boxes'].device)
        references[0]['prev_memory'] = prev_memory
        references[0]['input_size'] = torch.as_tensor([ori_img.shape[1], ori_img.shape[0]]).reshape(1, 2).long()
        references = [{k: v.to(device) for k, v in r.items()} for r in references]
        prev_img = ori_img.copy()
        online_tlwhs = []
        online_ids = []
        #online_scores = []
        for t in online_targets:
            tlwh = t._tlwh
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
        if logger is not None and (show_image or save_dir is not None):
            online_im = vis.plot_tracking(ori_img, online_tlwhs, online_ids, frame_id=frame_id, fps=1. / timer.average_time)
        if offset_np is not None:
            for offset in offset_np:
                x1, y1, x2, y2 = np.array(offset, dtype=np.int32)
                cv2.line(online_im, (x1, y1), (x2, y2), (0, 0, 255) , 2)

        if show_image and logger is not None:
            cv2.imshow('online_im', online_im)
        if save_dir is not None and logger is not None:
            cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
        frame_id += 1
    # save results
    if result_filename is not None:
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
        logger, root_output_dir = create_logger(cfg, proc_rank)
    
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
        result_root = os.path.join(root_output_dir, 'results', exp_name)
        logger.setLevel(logging.INFO)
        if not osp.exists(result_root):
            os.makedirs(result_root)

    data_type = 'mot'
    # run tracking
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []
    for seq in seqs:
        output_dir = None
        result_filename = None
        if logger is not None:
            output_dir = osp.join(result_root, seq) if save_images or save_videos else None
            result_filename = osp.join(result_root, '{}.txt'.format(seq))
            logger.info('start seq: {}'.format(seq))
            if output_dir is not None and not osp.exists(output_dir):
                os.makedirs(output_dir)
        device = torch.device(cfg.DEVICE)
        img_path_list = sorted(os.listdir(osp.join(data_root, seq, 'img1')))
        img_path_list = [osp.join(data_root, seq, 'img1', path) for path in img_path_list]
        
        meta_info = open(osp.join(data_root, seq, 'seqinfo.ini')).read()
        frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])
        nf, ta, tc = eval_seq(cfg, device, img_path_list, model, postprocessors, data_type, result_filename, save_dir=output_dir, show_image=show_image, frame_rate=frame_rate, logger=logger)
        n_frame += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)

        # eval
        if logger is not None:
            logger.info('Evaluate seq: {}'.format(seq))
            evaluator = Evaluator(data_root, seq, data_type)
            accs.append(evaluator.eval_file(result_filename))
            if save_videos and output_dir is not None:
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
    if logger is not None:
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
    data_dir = '/mnt/lustre/chenmingfei/code/MOT_DeformableSet'
    # val mot16
    # seqs_str = '''MOT16-02
    #               MOT16-04
    #               MOT16-05
    #               MOT16-09
    #               MOT16-10
    #               MOT16-11
    #               MOT16-13'''
    # data_root = os.path.join(data_dir, 'MOT16/train')
    # # test mot16
    # seqs_str = '''MOT16-01
    #               MOT16-03
    #               MOT16-06
    #               MOT16-07
    #               MOT16-08
    #               MOT16-12
    #               MOT16-14'''
    # data_root = os.path.join(data_dir, 'MOT16/test')
    # # test mot15
    # seqs_str = '''ADL-Rundle-1
    #               ADL-Rundle-3
    #               AVG-TownCentre
    #               ETH-Crossing
    #               ETH-Jelmoli
    #               ETH-Linthescher
    #               KITTI-16
    #               KITTI-19
    #               PETS09-S2L2
    #               TUD-Crossing
    #               Venice-1'''
    # data_root = os.path.join(data_dir, 'MOT15/images/test')
    # # test mot17
    # seqs_str = '''MOT17-01-SDP
    #               MOT17-03-SDP
    #               MOT17-06-SDP
    #               MOT17-07-SDP
    #               MOT17-08-SDP
    #               MOT17-12-SDP
    #               MOT17-14-SDP'''
    # data_root = os.path.join(data_dir, 'data/MOT17/test')
    # val mot17
    # seqs_str = '''MOT17-02-SDP
    #               MOT17-04-SDP
    #               MOT17-05-SDP
    #               MOT17-09-SDP
    #               MOT17-10-SDP
    #               MOT17-11-SDP
    #               MOT17-13-SDP'''
    seqs_str = '''MOT17-13-SDP'''
    data_root = os.path.join(data_dir, 'data/MOT17/train')
    # # val mot15
    # seqs_str = '''Venice-2
    #               KITTI-13
    #               KITTI-17
    #               ETH-Bahnhof
    #               ETH-Sunnyday
    #               PETS09-S2L1
    #               TUD-Campus
    #               TUD-Stadtmitte
    #               ADL-Rundle-6
    #               ADL-Rundle-8
    #               ETH-Pedcross2
    #               TUD-Stadtmitte'''
    # data_root = os.path.join(data_dir, 'MOT15/images/train')
    # # val mot20
    # seqs_str = '''MOT20-01
    #               MOT20-02
    #               MOT20-03
    #               MOT20-05
    #               '''
    # data_root = os.path.join(data_dir, 'MOT20/images/train')
    # # test mot20
    # seqs_str = '''MOT20-04
    #               MOT20-06
    #               MOT20-07
    #               MOT20-08
    #               '''
    # data_root = os.path.join(data_dir, 'MOT20/images/test')

    seqs = [seq.strip() for seq in seqs_str.split()]

    main(data_root=data_root,
         seqs=seqs,
         exp_name='test',
         show_image=False,
         save_images=True,
         save_videos=False)