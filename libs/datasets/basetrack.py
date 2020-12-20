import json
import logging
import numpy as np
import os
import os.path as osp
from PIL import Image
import random

import mmcv
import torch
from torch.utils.data import Dataset

class BaseTrackDataset(Dataset):

    def __init__(self,
                 data_root,
                 img_root,
                 transform=None,
                 istrain=False,
                 max_obj=100,
                 ):
        self.max_objs = max_obj
        self.transform = transform

        video_anno = {}
        for img_dir in os.listdir(data_root):
            if img_dir.split('-')[-1] != 'SDP':
                continue
            seq_path = os.path.join(data_root, img_dir, 'seqinfo.ini')
            f = open(seq_path)
            seq_info = f.readlines()
            video_name = seq_info[1].strip().split('=')[-1]
            img_dir_name = seq_info[2].strip().split('=')[-1]
            seq_len = int(seq_info[4].strip().split('=')[-1])
            width = int(seq_info[5].strip().split('=')[-1])
            height = int(seq_info[6].strip().split('=')[-1])
            imext = seq_info[7].strip().split('=')[-1]
            video_anno[video_name] = {
                'img_root': os.path.join(data_root, video_name, img_dir_name),
                'seq_len': seq_len,
                'width': width,
                'height': height,
                'imext': imext,
                'anno':{}
            }
            f.close()

            anno_path = os.path.join(data_root, video_name, 'gt/gt.txt')
            f = open(anno_path)
            lines = f.readlines()
            for line in lines:
                # frame_id from 1
                frame_id, track_id, x1, y1, w, h, valid, \
                    label, vis_ratio = line.strip().split(',')
                box = [float(x1), float(y1), float(x1)+float(w), float(y1)+float(h)]
                if int(valid) == 1:
                    video_anno[video_name]['anno'].setdefault(int(frame_id), {
                        'boxes': [],
                        'labels': [],
                        'track_ids': [],
                        'vis_ratios': []
                    })
                    video_anno[video_name]['anno'][int(frame_id)]['boxes'].append(np.array(box, dtype=np.float))
                    video_anno[video_name]['anno'][int(frame_id)]['labels'].append(int(label))
                    video_anno[video_name]['anno'][int(frame_id)]['track_ids'].append(int(track_id))
                    video_anno[video_name]['anno'][int(frame_id)]['vis_ratios'].append(float(vis_ratio))
            
        self.anno_info = []
        count_id = 0
        last_id_count = 0
        for video_name in video_anno.keys():
            max_track_id = 0
            for frame_id in video_anno[video_name]['anno'].keys():
                anno = video_anno[video_name]['anno'][frame_id]
                seq_len = video_anno[video_name]['seq_len']
                if frame_id >= seq_len:
                    next_ann_id = -1
                else:
                    next_ann_id = count_id + 1
                img_ext = video_anno[video_name]['imext']
                img_name = str(frame_id).zfill(6) + img_ext
                img_path = os.path.join(video_anno[video_name]['img_root'], img_name)
                cur_anno_info = {
                    'video_name': video_name,
                    'img_path': img_path,
                    'seq_len': seq_len,
                    'width': video_anno[video_name]['width'],
                    'height': video_anno[video_name]['height'],
                    'frame_id': frame_id,
                    'boxes': np.vstack(anno['boxes']).reshape(-1, 4),
                    'labels': np.array(anno['labels']).reshape(-1, ),
                    'track_ids': np.array(anno['track_ids']).reshape(-1, ),
                    'dataset_track_ids': np.array(anno['track_ids']).reshape(-1, ) + last_id_count,
                    'vis_ratios': np.array(anno['vis_ratios']).reshape(-1, ),
                    'next_ann_id': next_ann_id
                }
                max_track_id = max(max(anno['track_ids']), max_track_id)
                self.anno_info.append(cur_anno_info)
                count_id += 1
            last_id_count += max_track_id
        print(f'last_id_count: {last_id_count}')
                
    def __len__(self):
        return len(self.anno_info)

    def __getitem__(self, index):
        """
        Return:
            data (tensor): a image
            bboxes (tensor): shape: `(num_object, 4)`
                box = bboxes[:, :4]， label = bboxes[:, 4]
            index (int): image index
        """
        # next frame
        while(self.anno_info[index]['next_ann_id']==-1):
            index = random.choice(np.arange(len(self.anno_info)))
        anno = self.anno_info[index]
        img_path = anno['img_path']
        video_name = anno['video_name']
        if not osp.exists(img_path):
            logging.error("Cannot found image data: " + img_path)
            raise FileNotFoundError
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        boxes = anno['boxes']
        labels = anno['labels']
        track_ids = anno['track_ids']
        dataset_track_ids = anno['dataset_track_ids']
        vis_ratios = anno['vis_ratios']
        num_object = min(len(boxes), self.max_objs)
        if num_object == 0:
            boxes = np.array([])
            labels = np.array([])
            vis_ratios = np.array([])
        boxes = torch.from_numpy(boxes.reshape(-1, 4)[:num_object].astype(np.float32))
        labels = torch.from_numpy(labels.reshape(-1, )[:num_object].astype(np.int64))
        vis_ratios = torch.from_numpy(vis_ratios.reshape(-1, )[:num_object].astype(np.float32))
        dataset_track_ids = torch.from_numpy(dataset_track_ids.reshape(-1, )[:num_object].astype(np.float32))
        target = dict(
            boxes=boxes,
            labels=labels,
            vis_ratios=vis_ratios,
            sizes=torch.from_numpy(np.array([h, w])),
            ids=dataset_track_ids
        )

        next_anno = self.anno_info[anno['next_ann_id']]
        next_img_path = next_anno['img_path']
        if not osp.exists(next_img_path):
            logging.error("Cannot found image data: " + next_img_path)
            raise FileNotFoundError
        next_img = Image.open(next_img_path).convert('RGB')
        assert next_img.size == img.size
        ori_next_boxes = next_anno['boxes']
        ori_next_labels = next_anno['labels']
        next_dataset_track_ids = next_anno['dataset_track_ids']
        next_vis_ratios = next_anno['vis_ratios']
        num_object = min(len(ori_next_boxes), self.max_objs)
        if num_object == 0:
            ori_next_boxes = np.array([])
            ori_next_labels = np.array([])
            next_vis_ratios = np.array([])
        ori_next_boxes = torch.from_numpy(ori_next_boxes.reshape(-1, 4)[:num_object].astype(np.float32))
        ori_next_labels = torch.from_numpy(ori_next_labels.reshape(-1, )[:num_object].astype(np.int64))
        next_vis_ratios = torch.from_numpy(next_vis_ratios.reshape(-1, )[:num_object].astype(np.float32))
        next_dataset_track_ids = torch.from_numpy(next_dataset_track_ids.reshape(-1, )[:num_object].astype(np.float32))
        next_target = dict(
            boxes=ori_next_boxes,
            labels=ori_next_labels,
        )
        img_list = [img, next_img]
        target_list = [target, next_target]
        if self.transform is not None:
            img_list, target_list = self.transform(
                img_list, target_list
            )
        img, next_img = img_list
        target, next_target = target_list

        boxes = target['boxes']
        labels = target['labels']
        ori_next_boxes = next_target['boxes']
        ori_next_labels = next_target['labels']
        assert len(dataset_track_ids) == len(boxes)
        assert len(next_dataset_track_ids) == len(ori_next_boxes)
        # match track_id
        match_mask = []
        valid_pre_mask = []
        for pre_idx, tid in enumerate(dataset_track_ids):
            match_index = torch.where(next_dataset_track_ids==tid)[0]
            if len(match_index) > 0:
                match_mask.append(match_index[0])
                valid_pre_mask.append(pre_idx)
        match_mask = torch.as_tensor(match_mask).reshape(-1, ).long()
        valid_pre_mask = torch.as_tensor(valid_pre_mask).reshape(-1, ).long()

        next_boxes = torch.zeros(boxes.shape)
        next_boxes[valid_pre_mask] = ori_next_boxes[match_mask]
        next_labels = np.zeros(labels.shape)
        next_labels[valid_pre_mask] = ori_next_labels[match_mask]
    
        # TODO map_idx, matched_idx, next_boxes, for the joint training
        ref_boxes = torch.cat([next_boxes[valid_pre_mask, :2], boxes[
            valid_pre_mask, 2:]], dim=1)
        assert ref_boxes.shape[-1] == 4
        out_target = dict(
            boxes=boxes,
            next_boxes=ori_next_boxes,
            ids=dataset_track_ids,
            next_ids=next_dataset_track_ids,
            next_centers=next_boxes[..., :2],
            labels=labels,
            next_labels=ori_next_labels,
            idx_map=match_mask,
            matched_idx=valid_pre_mask,
            ref_boxes=ref_boxes,
        )
        return img, next_img, out_target, video_name
