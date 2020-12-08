import argparse
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

import torch
import torchvision.transforms.functional as F_trans
import torch.nn.functional as F

import _init_paths
from configs import cfg
from configs import update_config
from libs.datasets.transform import resize
from libs.models.hoi_transformer_base import SetCriterion
from libs.utils import box_ops
from libs.utils.box_ops import box_xyxy_to_cxcywh
from libs.utils.utils import get_model


features_grad = []

def parse_args():
    parser = argparse.ArgumentParser(description='HOI Transformer Task')
    parser.add_argument(
        '--cfg',
        dest='yaml_file',
        default='configs/hoi_transformer.yaml',
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
    image, _ = resize(image, None, 800, max_size=1333)
    image = F_trans.to_tensor(image)
    image = F_trans.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return image

def process_img(img_path, model, postprocessors, device, rel_topk=10):
    model.eval()
    image = read_img(img_path)
    h, w = image.shape[1:]
    size = torch.from_numpy(np.array([h, w]))
    imgs = [image.to(device)]
    bs = len(imgs)
    target_sizes = size.expand(bs, 2)
    target_sizes = target_sizes.to(device)
    outputs_dict = model(imgs)
    file_name = img_path.split('/')[-1]
    pred_out = postprocessors(outputs_dict, file_name, target_sizes, rel_topk=rel_topk)
    boxes_np = []
    labels_np = []
    for pred in pred_out['predictions']:
        boxes_np.append(pred['bbox'])
        labels_np.append(pred['category_id'])
    boxes_np = np.array(boxes_np)
    labels_np = np.array(labels_np)
    r_labels_np = []
    rel_h_cent_np = []
    rel_o_cent_np = []
    for rel in pred_out['hoi_prediction']:
        r_labels_np.append(rel['category_id'])
        h_box = pred_out['predictions'][rel['subject_id']]['bbox']
        o_box = pred_out['predictions'][rel['object_id']]['bbox']
        h_cent = 0.5 * (h_box[:2] + h_box[2:])
        o_cent = 0.5 * (o_box[:2] + o_box[2:])
        rel_h_cent_np.append(h_cent)
        rel_o_cent_np.append(o_cent)
    r_labels_np = np.array(r_labels_np)
    rel_h_cent_np = np.array(rel_h_cent_np)
    rel_o_cent_np = np.array(rel_o_cent_np)

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (w, h))
    for box, label in zip(boxes_np, labels_np):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0) , 1) #g

    rel_str = ''
    for h, o, rel in zip(rel_h_cent_np, rel_o_cent_np, r_labels_np):
        xh, yh = h.astype(int)
        xo, yo = o.astype(int)
        cv2.circle(img, (xh, yh), 5, (0, 0, 255))
        cv2.circle(img, (xo, yo), 5, (255, 0, 0))
        cv2.line(img,(xh, yh), (xo, yo), (0, 0, 255), 3)
        rel_str += hico_verb_dict[rel] +'_'
    return img, pred_out, rel_str

def get_target(img_path, labels_path='data/test_hico.json', num_classes_verb=117):
    def multi_dense_to_one_hot(labels, num_classes):
        num_labels = labels.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels.ravel()] = 1
        # index: id-1
        return np.sum(labels_one_hot, axis=0)[1:]

    image = Image.open(img_path).convert('RGB')
    w, h = image.size
    hoi_annotations = json.load(open(labels_path, 'r'))
    for test_ann in hoi_annotations:
        file_name = test_ann['file_name']
        if file_name != img_path.split('/')[-1]:
            continue
        hoi_anns = test_ann['hoi_annotation']
        anns =  test_ann['annotations']
        boxes = []
        labels = []
        for ann in anns:
            boxes.append(np.asarray(ann['bbox']))
            if isinstance(ann['category_id'], str):
                ann['category_id'] =  int(ann['category_id'].replace('\n', ''))
            cls_id = int(ann['category_id'])
            labels.append(cls_id)
        boxes = torch.from_numpy(np.vstack(boxes).reshape(-1, 4).astype(np.float32))
        labels = np.array(labels).reshape(-1,)
        target = dict(
            boxes=boxes,
            labels=labels
        )
        image, target = resize(image, target, 800, max_size=1333)
        target['boxes'] = box_xyxy_to_cxcywh(target['boxes'])
        target['boxes'] = target['boxes'] / torch.tensor([w, h, w, h], dtype=torch.float32)
        image = F_trans.to_tensor(image)
        image = F_trans.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        target['labels'] = torch.from_numpy(target['labels'])
        boxes = target['boxes']
        hoi_labels = []
        hoi_boxes = [] # sub_ctx, sub_cty, obj_ctx, obj_cty
        for hoi in hoi_anns:
            hoi_label_np = np.array([hoi['category_id']])
            hoi_labels.append(multi_dense_to_one_hot(hoi_label_np,
                                                     num_classes_verb+1))

            sub_ct_coord = boxes[hoi['subject_id']][..., :2]     
            obj_ct_coord = boxes[hoi['object_id']][..., :2]
            hoi_boxes.append(torch.cat([sub_ct_coord, obj_ct_coord], dim=-1).reshape(-1, 4))
            
         # gt_rel label 1 -> index 0
        hoi_labels = np.array(hoi_labels).reshape(-1, num_classes_verb)

        target['rel_labels'] = torch.from_numpy(hoi_labels)
        target['rel_boxes'] = torch.cat(hoi_boxes)
        target['size'] = torch.from_numpy(np.array([h, w]))
        
        return image, target
    
    return None, {}
        

def get_rel_loss(criterion, outputs_dict, targets):
    outputs = outputs_dict.copy()
    del outputs['aux_outputs']
    # outputs['pred_rel']['pred_logits'] = outputs_dict['pred_rel']['pred_logits'][..., i, :].unsqueeze(1)
    # outputs['pred_rel']['pred_boxes'] = outputs_dict['pred_rel']['pred_boxes'][..., i, :].unsqueeze(1)
    loss_dict = criterion(outputs, targets)
    weight_dict = criterion.weight_dict
    cal_loss_dict = {
        'rel_loss_ce': criterion.out_rel_loss.sum(-1),
        'rel_loss_bbox': criterion.out_rel_loss_box.sum(-1)
    }
    rel_loss_ce = criterion.out_rel_loss.sum(-1) * weight_dict['rel_loss_ce']
    rel_loss_bbox = criterion.out_rel_loss_box.sum(-1) * weight_dict['rel_loss_bbox']
    return (rel_loss_ce + rel_loss_bbox)


def draw_CAM(i, model, criterion, postprocessors, device, img_path,
             visual_heatmap=False, eps=1e-10):
    # 为了能读取到中间梯度定义的辅助函数
    model.eval()
    criterion.eval()
    image, target = get_target(img_path)
    imgs = [image.to(device)]
    targets = [{k: v.to(device) for k, v in t.items()} for t in [target]]
    outputs_dict = model(imgs)
    features = model.out_feat
    h, w = image.shape[1:]
    size = torch.from_numpy(np.array([h, w]))
    bs = len(imgs)
    target_sizes = size.expand(bs, 2)
    target_sizes = target_sizes.to(device)
    file_name = img_path.split('/')[-1]
    
    heat_map_list = []
    rel_loss = get_rel_loss(criterion, outputs_dict, targets)
    print(criterion.out_idx)
    img = cv2.imread(img_path)  # 用cv2加载原始图像
    features_grad =[]
    def extract(g):
        features_grad.append(g)
    features.register_hook(extract)
    rel_loss[..., i].backward() # 计算梯度
    grads = features_grad[-1]   # 获取梯度

    pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1))

    # 此处batch size默认为1，所以去掉了第0维（batch size维）
    pooled_grads = pooled_grads[0]
    new_features = features[0]
        # 512是最后一层feature的通道数
    for j in range(256):
        new_features[j, ...] *= pooled_grads[j, ...]

    heatmap = new_features.detach().numpy()
    heatmap = np.mean(heatmap, axis=0)

    heatmap = np.maximum(heatmap, eps)
    heatmap /= np.max(heatmap)
    heat_map_list.append(heatmap)

    # 可视化原始热力图
    if visual_heatmap:
        plt.matshow(heatmap)
        plt.show()
    
    out_root = 'CAM_out'
    if not os.path.exists(out_root):
        os.mkdir(out_root)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
    heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
    superimposed_img = heatmap * 0.4 + img  # 这里的0.4是热力图强度因子
    save_path = os.path.join(out_root, f'{im_name}_{i}.jpg')
    cv2.imwrite(save_path, superimposed_img)
    print(f'{save_path}')


if __name__ == '__main__':
    args = parse_args()
    update_config(cfg, args)
    # resume_path = 'data/hoi_focal_decode_res101_HOI_Transformer_epoch060_checkpoint.pth'
    resume_path = 'data/hoi_10lr_117_zero_focal_cross_HOI_Transformer_epoch085_checkpoint.pth'
    device = torch.device(cfg.DEVICE)
    model, criterion, postprocessors = get_model(cfg, device)  
    model.to(device)
    if os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location='cpu')
        # resume
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            print(f'==> model pretrained from {resume_path} \n')

    img_root = 'data/hico_20160224_det/images/test'
    out_root = 'CAM_out'
    if not os.path.exists(out_root):
        os.mkdir(out_root)
    
    count = 0
    path_list = ['HICO_test2015_00000072.jpg']
    # for path in sorted(os.listdir(img_root)):
    for path in path_list:
        if path.split('.')[-1] == 'json':
            continue
        img_path = os.path.join(img_root, path)
        im_name = img_path[:-4].split('_')[-1]
        for i in range(16):
            draw_CAM(i, model, criterion, postprocessors, device, img_path)
        count += 1
