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

class MatchCustomTaskDataset(Dataset):

    def __init__(self,
                 anno_root,
                 img_root,
                 transform=None,
                 istrain=False,
                 max_obj=100,
                 ):
        self.img_root = img_root
        self.transform = transform
        annotations = []
        count = [0]
        for path in os.listdir(anno_root):
            data = mmcv.load(f'{anno_root}/{path}')
            count.append(len(data)+count[-1])
            annotations.extend(data)
        self.annotations = annotations
        self.ids = []
        self.max_objs = max_obj
        id_base = 439047
        max_id = 439046
        min_id = 10000000
        pre_mot_video = None
        pre_mot_frame = None
        for i, anno in enumerate(self.annotations):
            if i > 0 and i in count:
                id_base = max_id + 1
            anno['ann']['extra_anns'] = np.array(anno['ann']['extra_anns']).reshape(-1, )
            if(len(anno['ann']['extra_anns'])>0):
                anno['ann']['extra_anns'][anno['ann']['extra_anns']>-1] += id_base - 1
                max_id = max(max_id, max(anno['ann']['extra_anns']))
                min_id = min(min_id, min(anno['ann']['extra_anns']))

            # get next_anno
            anno['next_ann_id'] = -1
            if istrain is False:
                flag_bad = 0
                self.ids.append(i)
            else:
                flag_bad = 1
                boxes = anno['ann']['bboxes']
                labels = anno['ann']['labels']
                if (len(boxes) > 0 and len(boxes) <= max_obj and 
                     labels.sum()==len(labels)):
                    flag_bad = 0
                    self.ids.append(i)
                    
            # if flag_bad == 0 and anno['filename'][:3] == 'MOT':
            if flag_bad == 0:
                video_name = anno['filename'].split('/')[2]
                frame = int(anno['filename'].split('/')[-1].split('.')[0])
                if pre_mot_video is None or pre_mot_frame is None:
                    pre_mot_video = video_name
                    pre_mot_frame = frame
                else:
                    if video_name == pre_mot_video and frame == pre_mot_frame + 1 and i > 0:
                        self.annotations[i-1]['next_ann_id'] = i
                        pre_mot_video = video_name
                        pre_mot_frame = frame
                    else:
                        pre_mot_frame = None
                        pre_mot_video = None
        # print(cur_id)
        id_base = max_id
        print(id_base, min_id)
        
    def __len__(self):
        return len(self.ids)

    def getraw_item(self, index):
        """
        Return:
            data (tensor): a image
            bboxes (tensor): shape: `(num_object, 4)`
                box = bboxes[:, :4]， label = bboxes[:, 4]
            index (int): image index
        """
        # affine the same pic
        affine_flag = (self.annotations[self.ids[index]]['next_ann_id']==-1)
        anno = self.annotations[self.ids[index]]
        filename = anno['filename']
        img_path = os.path.join(self.img_root, filename)
        if not osp.exists(img_path):
            logging.error("Cannot find image data: " + img_path)
            raise FileNotFoundError
        img = Image.open(img_path).convert('RGB')
        w, h = img.size

        boxes = anno['ann']['bboxes']
        labels = anno['ann']['labels']
        dataset_track_ids = anno['ann']['extra_anns']
        num_object = min(len(boxes), self.max_objs)
        if num_object == 0:
            boxes = np.array([])
            labels = np.array([])
        boxes = torch.from_numpy(boxes.reshape(-1, 4)[:num_object].astype(np.float32))
        labels = torch.from_numpy(labels.reshape(-1, )[:num_object].astype(np.int64))
        dataset_track_ids = torch.from_numpy(dataset_track_ids.reshape(-1, )[:num_object].astype(np.float32))
        target = dict(
            boxes=boxes,
            labels=labels,
            sizes=torch.from_numpy(np.array([h, w])),
            ids=dataset_track_ids
        )

        if affine_flag is False:
            next_anno = self.annotations[anno['next_ann_id']]
            next_filename = next_anno['filename']
            next_img_path = os.path.join(self.img_root, next_filename)
            if not osp.exists(next_img_path):
                logging.error("Cannot found image data: " + next_img_path)
                raise FileNotFoundError
            next_img = Image.open(next_img_path).convert('RGB')
            if next_img.size != img.size:
                import pdb; pdb.set_trace()
            assert next_img.size == img.size
            ori_next_boxes = next_anno['ann']['bboxes']
            ori_next_labels = next_anno['ann']['labels']
            next_dataset_track_ids = next_anno['ann']['extra_anns']
            num_object = min(len(ori_next_boxes), self.max_objs)
            if num_object == 0:
                ori_next_boxes = np.array([])
                ori_next_labels = np.array([])
            ori_next_boxes = torch.from_numpy(ori_next_boxes.reshape(-1, 4)[:num_object].astype(np.float32))
            ori_next_labels = torch.from_numpy(ori_next_labels.reshape(-1, )[:num_object].astype(np.int64))
            next_dataset_track_ids = torch.from_numpy(next_dataset_track_ids.reshape(-1, )[:num_object].astype(np.float32))
            next_target = dict(
                boxes=ori_next_boxes,
                labels=ori_next_labels,
                ids=next_dataset_track_ids
            )
            img_list = [img, next_img]
            target_list = [target, next_target]

            # TODO centertrack image transform for the offset train

            if self.transform is not None:
                img_list, target_list = self.transform(
                    img_list, target_list
                )
            img, next_img = img_list
            target, next_target = target_list
        else:
            if self.transform is not None:
                img_list, target_list = self.transform(
                    img, target
                )
            img, next_img = img_list
            target, next_target = target_list

        boxes = target['boxes']
        labels = target['labels']
        dataset_track_ids = target['ids']
        ori_next_boxes = next_target['boxes']
        ori_next_labels = next_target['labels']
        next_dataset_track_ids = next_target['ids']
        assert len(dataset_track_ids) == len(boxes)
        assert len(next_dataset_track_ids) == len(ori_next_boxes)
        # match track_id
        if affine_flag is False:
            match_mask = []
            valid_pre_mask = []
            for pre_idx, tid in enumerate(dataset_track_ids):
                match_index = torch.where(next_dataset_track_ids==tid)[0]
                if len(match_index) > 0:
                    match_mask.append(match_index[0])
                    valid_pre_mask.append(pre_idx)
        else:
            if 'valid_idx' in next_target:
                valid_idx = next_target['valid_idx']
            else:
                valid_idx = []
            match_mask = list(range(len(next_dataset_track_ids)))
            valid_pre_mask = np.array(list(range(len(dataset_track_ids))))[valid_idx]
        match_mask = torch.as_tensor(match_mask).reshape(-1, ).long()
        valid_pre_mask = torch.as_tensor(valid_pre_mask).reshape(-1, ).long()

        next_boxes = torch.zeros(boxes.shape)
        next_boxes[valid_pre_mask] = ori_next_boxes[match_mask]
        next_labels = np.zeros(labels.shape)
        next_labels[valid_pre_mask] = ori_next_labels[match_mask]
    
        # TODO map_idx, matched_idx, next_boxes, for the joint training
        ref_boxes = torch.cat([next_boxes[valid_pre_mask, :2], boxes[
            valid_pre_mask, 2:]], dim=1)
        boxes_cxcy = boxes[..., :2]-boxes[..., 2:4] + 0.5 * (boxes[..., 2:4]+boxes[..., 4:])
        boxes = torch.cat([boxes_cxcy, boxes[..., 2:4]+boxes[..., 4:]], dim=-1)

        ref_cxcy = boxes_cxcy[valid_pre_mask]
        valid = (ref_cxcy[..., 0]>=0) & (ref_cxcy[..., 0]<=1) & (ref_cxcy[..., 1]>=0) & (ref_cxcy[..., 1]<=1)
        ref_boxes = ref_boxes[valid]
        valid_pre_mask = valid_pre_mask[valid]
        match_mask = match_mask[valid]

        gt_ref_ids = dataset_track_ids[valid_pre_mask]
        ori_next_boxes_cxcy = ori_next_boxes[..., :2]-ori_next_boxes[..., 2:4] + 0.5 * (
            ori_next_boxes[..., 2:4]+ori_next_boxes[..., 4:])
        ori_next_boxes = torch.cat([ori_next_boxes_cxcy, ori_next_boxes[..., 2:4]+ori_next_boxes[..., 4:]], dim=-1)
        assert ref_boxes.shape[-1] == 6
        out_target = dict(
            size=torch.from_numpy(np.array([h, w])),
            boxes=boxes,
            next_boxes=ori_next_boxes,
            ids=dataset_track_ids,
            next_ids=next_dataset_track_ids,
            labels=labels,
            next_labels=ori_next_labels,
            idx_map=match_mask,
            matched_idx=valid_pre_mask,
            ref_boxes=ref_boxes,
            gt_ref_ids=gt_ref_ids
        )
        return img, next_img, out_target, filename, len(match_mask)

    def __getitem__(self, index):
        valid = 0
        while(valid == 0):
            img, next_img, out_target, filename, valid = self.getraw_item(index)
            if valid == 0:
                index = random.choice(np.arange(len(self.ids)))
        return img, next_img, out_target, filename