import argparse
import cv2
import json
import matplotlib.pyplot as plt
import math
import mmcv
import numpy as np
import os
from PIL import Image
import random

import torch
import torchvision.transforms.functional as F_trans
import torch.nn.functional as F

import _init_paths
from configs import cfg
from configs import update_config
from libs.datasets.transform import resize
from libs.utils import box_ops
from libs.utils.box_ops import box_xyxy_to_cxcywh
from libs.utils.utils import get_model

features_grad = []

def parse_args():
    parser = argparse.ArgumentParser(description='HOI Transformer Task')
    parser.add_argument(
        '--cfg',
        dest='yaml_file',
        default='/mnt/lustre/chenmingfei/code/MOT_DeformableSet/configs/deformable_macthtrack_kitti.yaml',
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
        default='tcp://10.5.38.36:22456',
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


def get_target(anno, img_root='/mnt/lustre/chenmingfei/data/MOT_data/'):
    filename = anno['filename']
    img_path = os.path.join(img_root, filename)
    if not os.path.exists(img_path):
        logging.error("Cannot find image data: " + img_path)
        raise FileNotFoundError
    img = Image.open(img_path).convert('RGB')
    ori_img = img.copy()
    w, h = img.size
    boxes = anno['ann']['bboxes']
    labels = anno['ann']['labels']
    dataset_track_ids = anno['ann']['extra_anns']
    boxes = torch.from_numpy(boxes.reshape(-1, 4).astype(np.float32))
    labels = torch.from_numpy(labels.reshape(-1, ).astype(np.int64))
    dataset_track_ids = torch.from_numpy(dataset_track_ids.reshape(-1, ).astype(np.float32))
    target = dict(
        boxes=boxes,
        labels=labels,
        sizes=torch.from_numpy(np.array([h, w])),
        ids=dataset_track_ids
    )
    image, target = resize(img, target, 608, max_size=1088)
    w, h = image.size
    target['boxes'] = box_xyxy_to_cxcywh(target['boxes'])
    target['boxes'] = target['boxes'] / torch.tensor([w, h, w, h], dtype=torch.float32)
    image = F_trans.to_tensor(image)
    image = F_trans.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return image, target, img_path, ori_img

def save_heatmap(i, img, weights, loc, shape, ref, h, w, out_root, im_name, type_str, visual_heatmap, eps, extra=None):
    dec_shape = torch.cat([shape[..., 1].reshape(4, 1), shape[..., 0].reshape(4, 1)], dim=-1)
    dec_ref_loc = ref[0][i] # n_level, 6
    if dec_ref_loc.shape[-1] == 2:
        dec_ref = dec_ref_loc
    else:
        dec_box = torch.cat([dec_ref_loc[..., :2] - dec_ref_loc[..., 2:4], dec_ref_loc[..., :2] + dec_ref_loc[..., 4:]], dim=-1)
        # dec_box = dec_box * torch.cat([dec_shape[0], dec_shape[0]], dim=-1)
        dec_ref = 0.5 * (dec_box[..., :2] + dec_box[..., 2:])
    weights = weights[0][i]
    locs = loc[0][i]
    dec_locs = locs * dec_shape[0][None, None, None, :]
    dec_sample_locs = dec_locs.long()
    dec_heatmap = torch.zeros(h, w).float()
    for k, (x, y) in enumerate(dec_sample_locs.reshape(-1, 2)):
        if (x>=0 and x<w and y>=0 and y<h):
            dec_heatmap[y,x] += weights.reshape(-1)[k] 
    heatmap = dec_heatmap.detach().numpy()
    heatmap = np.maximum(heatmap, eps)
    heatmap /= np.max(heatmap)
    # 可视化原始热力图
    if visual_heatmap:
        plt.matshow(heatmap)
        plt.show()
    if extra is not None:
        img = cv2.addWeighted(img, 0.5, extra, 0.5, 0)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
    heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
    superimposed_img = heatmap * 0.4 + img  # 这里的0.4是热力图强度因子
    save_path = os.path.join(out_root, f'{im_name}_{type_str}_{i}.jpg')
    dec_ref = dec_ref * torch.tensor([img.shape[1], img.shape[0]]).reshape(-1, 2).to(dec_ref.device)
    coord = dec_ref[0].long().cpu().detach().numpy()
    x, y = coord[0], coord[1]
    if extra is None:
        x1 = x - 1
        x2 = x + 1
        y1 = y - 4
        y2 = y + 4
        superimposed_img = cv2.rectangle(superimposed_img, (x1, y1), (x2, y2), (0, 215, 255), 2)
        x1 = x - 4
        x2 = x + 4
        y1 = y - 1
        y2 = y + 1
        superimposed_img = cv2.rectangle(superimposed_img, (x1, y1), (x2, y2), (29, 215, 255), 2)
    # superimposed_img = cv2.circle(superimposed_img, tuple(), 1, (255, 0, 0), thickness=2)
    cv2.imwrite(save_path, superimposed_img)
    print(f'{save_path}')

def draw_CAM(i, model, criterion, postprocessors, device, anno, out_root,
             visual_heatmap=False, eps=1e-10, sum_heatmap=None, references=None):
    # 为了能读取到中间梯度定义的辅助函数
    # if i not in [2, 6]:
    #     return range(16)
    model.eval()
    criterion.eval()
    conv_features, enc_attn_weights, dec_attn_weights = [], [], []
    dec_ref_attn_weights, dec_det_attn_weights = [], []

    hooks = [
        model.backbone[-2].register_forward_hook(
            lambda self, input, output: conv_features.append(output)
        ),
        model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
            lambda self, input, output: enc_attn_weights.append(model.transformer.encoder.layers[-1].self_attn.out[0])
        ),
        model.transformer.decoder.layers[-1].cross_attn.register_forward_hook(
            lambda self, input, output: dec_det_attn_weights.append(model.transformer.decoder.layers[-1].cross_attn.out[0])
        ), # 1 * 100 * (w*h)
        model.transformer.match_decoder.layers[-1].cross_attn.register_forward_hook(
            lambda self, input, output: dec_ref_attn_weights.append(model.transformer.match_decoder.layers[-1].cross_attn.out[0])
        ), # 1 * 100 * (w*h)
    ]

    image, target, img_path, ori_img = get_target(anno)
    if image is None or target is None:
        return range(16)
    imgs = [image.to(device)]
    targets = [{k: v.to(device) for k, v in t.items()} for t in [target]]
    h, w = image.shape[1:]
    size = torch.from_numpy(np.array([h, w]))
    bs = len(imgs)
    target_sizes = size.expand(bs, 2)
    img_h, img_w = target_sizes.unbind(1)
    target_sizes = target_sizes.to(device)
    file_name = img_path.split('/')[-3] + '-' + img_path.split('/')[-1]

    outputs_dict = model(imgs, references)
    prev_memory = model.out_memory.detach()
    pred_out = postprocessors(outputs_dict, img_path,
        target_sizes, references)
    res = pred_out[-1]
    valid_inds = torch.where(res['scores']>0.4)[0]
    boxes = res['boxes'][valid_inds]
    labels = res['labels'][valid_inds]
    scores = res['scores'][valid_inds]
    valid_id = torch.where(labels==1)[0]
    boxes_with_scores = torch.cat([boxes[valid_id].reshape(
        -1, 4), scores[valid_id].reshape(-1, 1)], dim=1)
    # keep = hard_nms(boxes_with_scores, 0.7, return_pick=True)
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



    for hook in hooks:
        hook.remove()
    # don't need the list anymore
    conv_features = conv_features[0]
    enc_attn_weights = enc_attn_weights[0]
    dec_attn_weights = dec_det_attn_weights[0]
    enc_loc, enc_index, enc_shape, enc_ref = model.transformer.encoder.layers[-1].self_attn.out[1:]
    dec_loc, dec_index, dec_shape, dec_ref = model.transformer.decoder.layers[-1].cross_attn.out[1:]
    if references is not None:
        ref_attn_weights = dec_ref_attn_weights[0]
        ref_loc, ref_index, ref_shape, ref_ref = model.transformer.match_decoder.layers[-1].cross_attn.out[1:]

    img = cv2.imread(img_path)  # 用cv2加载原始图像
    img = cv2.resize(img, (img_w, img_h))
    h, w = conv_features['0'].tensors.shape[-2:]

    # n_head average
    # N, Len_q, n_heads, n_levels, n_points, 2 (w, h)
    # sampling_offsets / offset_normalizer[None, None, None, :, None, :]
    # tensor([[ 76, 135],
        # [ 38,  68],
        # [ 19,  34],
        # [ 10,  17]], device='cuda:0')
    # conv_features['0'].tensors.shape[-2:]: torch.Size([76, 135])
    # TODO -> weights (w*h), reference points loc for (img.shape[1], img.shape[0])
    # further: query summation
    # further: joint memory   
    # save_heatmap(i, img.copy(), dec_attn_weights, dec_loc, dec_shape, dec_ref, h, w, out_root, file_name, 'detdec', visual_heatmap, eps)
    # save_heatmap(i, img.copy(), enc_attn_weights, enc_loc, enc_shape, enc_ref, h, w, out_root, file_name, 'enc', visual_heatmap, eps)
    if references is not None:
        pre_path = references[0]['prev_img_path']
        pre_img = cv2.imread(pre_path)  # 用cv2加载原始图像
        pre_img = cv2.resize(pre_img, (img_w, img_h))
        save_heatmap(i, pre_img, ref_attn_weights, ref_loc, ref_shape, ref_ref, h, w, out_root, file_name, 'refdec', visual_heatmap, eps, extra=img)

    
    return ori_img, dets_np, id_features_np, track_idx, ref_coords_np, ref_id_features, prev_memory, img_path, target_sizes


if __name__ == '__main__':
    args = parse_args()
    update_config(cfg, args)
    resume_path = '/mnt/lustre/chenmingfei/code/MOT_DeformableSet/output/matchtrack_kitti_DeformableMatchTrack_epoch045_checkpoint.pth'
    device = torch.device(cfg.DEVICE)
    model, criterion, postprocessors = get_model(cfg, device)  
    model.to(device)
    if os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location='cpu')
        # resume
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            print(f'==> model pretrained from {resume_path} \n')

    anno_path = '/mnt/lustre/chenmingfei/code/MOT_DeformableSet/data/mot17_pkl/train/MOT17_train.pkl'
    annotations = mmcv.load(anno_path)


    out_root = 'CAM_out_low'
    if not os.path.exists(out_root):
        os.mkdir(out_root)
    
    count = 0
    references = None
    for j, anno in enumerate(annotations):
        if j % 5 != 0:
            continue
        # if anno['filename'].split('/')[2] != 'MOT17-13-SDP' or int(anno['filename'].split('/')[-1].split('.')[0]) < 465:
        #     continue
        # if anno['filename'].split('/')[2] == 'MOT17-13-SDP' and int(anno['filename'].split('/')[-1].split('.')[0]) > 520:
        #     break
        if anno['filename'].split('/')[2] != 'MOT17-13-SDP':
            continue
        if references is None:
            num = 20
        else:
            num = min(20, len(references[0]['ref_boxes']))
        for i in range(num):
            ori_img, dets_np, id_features_np, track_idx, ref_coords_np, ref_id_features, prev_memory, img_path, target_sizes = draw_CAM(
                i, model, criterion, postprocessors, device, anno, out_root, references=references)
        id_features = torch.as_tensor(id_features_np)
        dets_np[..., 2:4] = dets_np[..., 2:4] - dets_np[..., :2]
        dets_np[..., :2] = dets_np[..., :2] + 0.5 * dets_np[..., 2:4]
        dets_np = torch.as_tensor(dets_np)
        target_sizes = target_sizes[0]
        ref_boxes = dets_np[..., :4] / torch.as_tensor([target_sizes[1], target_sizes[0], target_sizes[1], target_sizes[0]]).reshape(-1).to(dets_np.device)
        idx_map = torch.range(0, len(ref_boxes)-1)
        references = [
            dict(ref_features=id_features.float(), ref_boxes=ref_boxes.float(), idx_map=idx_map)
        ]
        wh_boxes = references[0]['ref_boxes'].clone()
        cx = wh_boxes[..., 0].clamp(min=0, max=1.)
        cy = wh_boxes[..., 1].clamp(min=0, max=1.)
        left_xy = wh_boxes[..., :2] - 0.5 * wh_boxes[..., 2:]
        right_xy = wh_boxes[..., :2] + 0.5 * wh_boxes[..., 2:]
        lw = (cx - left_xy[..., 0]).unsqueeze(-1)
        lh = (cy - left_xy[..., 1]).unsqueeze(-1)
        rw = (right_xy[..., 0] - cx).unsqueeze(-1)
        rb = (right_xy[..., 1] - cy).unsqueeze(-1)
        cx = cx.unsqueeze(-1)
        cy = cy.unsqueeze(-1)
        references[0]['ref_boxes'] = torch.cat([cx, cy, lw, lh, rw, rb], dim=-1)
        references[0]['prev_memory'] = prev_memory
        references[0]['input_size'] = torch.as_tensor([ori_img.size[0], ori_img.size[1]]).reshape(1, 2).long()
        references = [{k: v.to(device) for k, v in r.items()} for r in references]
        video_name = anno['filename'].split('/')[2]
        frame = int(anno['filename'].split('/')[-1].split('.')[0])
        if j < len(annotations) - 1:
            next_anno = annotations[j+1]
            next_video_name = next_anno['filename'].split('/')[2]
            next_frame = int(next_anno['filename'].split('/')[-1].split('.')[0])
            # if next_video_name == video_name and next_frame == frame + 1:
            if next_video_name == video_name:
                references[0]['prev_img_path'] = img_path
        else:
            references = None
        count += 1
