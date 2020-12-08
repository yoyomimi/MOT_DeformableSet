# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Created by Mingfei Chen (chenmingfei@sensetime.com)
# Created On: 2020-9-17
# ------------------------------------------------------------------------------
import argparse
import cv2
import mmcv
import numpy as np
import os
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.transforms.functional as F_trans
import torch.nn.functional as F

import _init_paths
from configs import cfg
from configs import update_config
from libs.datasets.transform import resize
from libs.utils import box_ops
from libs.utils.utils import get_model
from libs.utils.utils import write_dict_to_json

from libs.models.puppet_tracker import PuppetTracker

import logging
    
def parse_args():
    parser = argparse.ArgumentParser(description='HOI Transformer Task')
    parser.add_argument(
        '--cfg',
        dest='yaml_file',
        default='configs/MOT_Transformer_base.yaml',
        help='experiment configure file name, e.g. configs/fcos_detector.yaml',
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

def read_img(img_path):
    image = Image.open(img_path).convert('RGB')
    w, h = image.size
    size = torch.from_numpy(np.array([h, w]))
    image, _ = resize(image, None, 800, max_size=1333)
    image = F_trans.to_tensor(image)
    image = F_trans.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return image, size

def process_img(img_path, next_img_path, model, postprocessors,
                device, topk=150, nms_thr=0.35, min_thr=0.35):
    model.eval()
    image, size = read_img(img_path)
    next_image, _ = read_img(next_img_path)  
    imgs = [image.to(device)]
    next_imgs = [next_image.to(device)]
    bs = len(imgs)
    target_sizes = size.expand(bs, 2)
    target_sizes = target_sizes.to(device)
    outputs_dict = model(imgs, next_imgs)
    pred_out = postprocessors(outputs_dict, target_sizes, topk=topk,
        nms_thr=nms_thr, min_thr=min_thr)

    return pred_out, size


if __name__ == '__main__':
    args = parse_args()
    update_config(cfg, args)
    resume_path = 'data/mot_base_100.pth'
    device = torch.device(cfg.DEVICE)
    model, criterion, postprocessors = get_model(cfg, device)  
    model.to(device)
    if os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location='cpu')
        # resume
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            print(f'==> model pretrained from {resume_path} \n')

    results = {}
    img_root = 'data/MOT17/train/MOT17-02-SDP/img1'
    out_root = 'test_out/MOT17-02-SDP'
    if not os.path.exists(out_root):
        os.makedirs(out_root)
    
    count = 0
    path_list = sorted(os.listdir(img_root))
    Tracker = PuppetTracker(track_buffer=30, det_match_thr=0.35, lost_match_thr=0.35,
        min_track_conf=0.15)
    last_pred_out = None
    size = (800, 1333)
    for i, path in enumerate(tqdm(path_list)):
        if path.split('.')[-1] == 'json':
            continue
        if i == len(path_list) - 1:
            break
        img_path = os.path.join(img_root, path)
        next_img_path = os.path.join(img_root, path_list[i+1])
        pred_out, size = process_img(img_path, next_img_path, model,
            postprocessors, device, topk=150, nms_thr=0.35, min_thr=0.4)

        h, w = size
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (w, h))

        if i > 0:
            det_in = [pred_out['boxes'], pred_out['det_scores'], pred_out['clses']]
            pred_in = [last_pred_out['puppet_boxes'], last_pred_out['mot_scores'], last_pred_out['puppet_clses']]
            Tracker.update(det_in, pred_in)
            for box in last_pred_out['puppet_boxes']:
                # front-green; next-red
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255) , 1) #r
            for box in pred_out['boxes']:
                # front-green; next-red
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0) , 1) #g
        else:
            det_in = [pred_out['boxes'], pred_out['det_scores'], pred_out['clses']]
            Tracker.update(det_in)
            for box in pred_out['boxes']:
                # front-green; next-red
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0) , 1) #g
        last_pred_out = pred_out
        logging.info(f'processing {img_path}')
        out_path = os.path.join('test_out', path)
        cv2.imwrite(out_path, img)

        count += 1
        # if count > 20:
        #     break

    Tracker.frame_summary(count, out_root=out_root, vis=True, log=True,
        video_root=img_root, size=size, min_len=6)