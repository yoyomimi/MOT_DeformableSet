import argparse
import os
import cv2
import numpy as np
from PIL import Image

import torch
import torchvision.transforms.functional as F_trans
import torch.nn.functional as F

import _init_paths
from configs import cfg
from configs import update_config
from libs.datasets.transform import resize
from libs.utils import box_ops
from libs.utils.box_ops import hard_nms
from libs.utils.utils import get_model
from libs.utils.utils import write_dict_to_json


def parse_args():
    parser = argparse.ArgumentParser(description='HOI Transformer Task')
    parser.add_argument(
        '--cfg',
        dest='yaml_file',
        default='configs/deformable_track_single_test.yaml',
        help='experiment configure file name, e.g. configs/deformable_track_single_test.yaml',
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
    image, _ = resize(image, None, 800, max_size=1333)
    image = F_trans.to_tensor(image)
    image = F_trans.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return image

def process_img(img_path, model, postprocessors, device, threshold=0.12):
    model.eval()
    image = read_img(img_path)
    h, w = image.shape[1:]
    size = torch.from_numpy(np.array([h, w]))
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
    keep = hard_nms(boxes_with_scores, 0.5, return_pick=True)
    boxes_np = boxes_with_scores[keep, :4].reshape(
        -1, 4).data.cpu().numpy()
    scores_np = boxes_with_scores[keep, -1].reshape(
        -1, 1).data.cpu().numpy()
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (w, h))
    for box, score in zip(boxes_np, scores_np):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255) , 2)
        cv2.putText(img, str(score)[1:5], (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL,
            1, (255, 0, 0), 1, cv2.LINE_AA)
    return img, pred_out


if __name__ == '__main__':
    args = parse_args()
    update_config(cfg, args)
    resume_path = '/mnt/lustre/chenmingfei/code/MOT_DeformableSet/output/deformable_step2_DeformableBaseTrack_epoch040_checkpoint.pth'
    device = torch.device(cfg.DEVICE)
    model, criterion, postprocessors = get_model(cfg, device)  
    model.to(device)
    if os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location='cpu')
        # resume
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            print(f'==> model pretrained from {resume_path} \n')

    results = []
    img_root = '/mnt/lustre/chenmingfei/code/MOT_DeformableSet/data/MOT17/train/MOT17-02-SDP/img1'
    out_root = 'test_out/MOT17-02-SDP'
    if not os.path.exists(out_root):
        os.makedirs(out_root)
    
    count = 0
    for path in sorted(os.listdir(img_root)):
        if path.split('.')[-1] == 'json':
            continue
        img_path = os.path.join(img_root, path)
        img, pred_out = process_img(img_path, model, postprocessors, device, threshold=0.12)
        out_path = os.path.join(out_root, path)
        results.append(pred_out)
        cv2.imwrite(out_path, img)
        print(f'{out_path}')
        count += 1