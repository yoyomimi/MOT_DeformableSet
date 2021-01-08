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
        id_base = 0
        max_id = 0
        # cur_id = 0
        for i, anno in enumerate(self.annotations):
            # anno['ann']['extra_anns'] = np.array(anno['ann']['extra_anns']).reshape(-1, )
            # if len(anno['ann']['extra_anns']) == 0:
            #     anno['ann']['extra_anns'] = -np.ones(len(anno['ann']['bboxes'])).reshape(-1, )
            # for j, single_id in enumerate(anno['ann']['extra_anns']):
            #     if cur_id > 355567:
            #         anno['ann']['extra_anns'][j] = -1
            #     else:
            #         anno['ann']['extra_anns'][j] = cur_id
            #         cur_id += 1
            if i > 0 and i in count:
                id_base = max_id + 1
            anno['ann']['extra_anns'] = np.array(anno['ann']['extra_anns']).reshape(-1, )
            if(len(anno['ann']['extra_anns'])>0):
                anno['ann']['extra_anns'][anno['ann']['extra_anns']>-1] += id_base
                max_id = max(max_id, max(anno['ann']['extra_anns']))
            if istrain is False:
                self.ids.append(i)
            else:
                flag_bad = 0
                boxes = anno['ann']['bboxes']
                labels = anno['ann']['labels']
                if (len(boxes) >= 0 and len(boxes) <= max_obj and 
                     labels.sum()==len(labels)):
                    self.ids.append(i)
        # print(cur_id)
        id_base = max_id
        print(id_base)
        
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        """
        Return:
            data (tensor): a image
            bboxes (tensor): shape: `(num_object, 4)`
                box = bboxes[:, :4]， label = bboxes[:, 4]
            index (int): image index
        """
        # next frame
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

        next_anno = self.anno_info[anno['next_ann_id']]
        next_filename = next_anno['filename']
        next_img_path = os.path.join(self.img_root, next_filename)
        if not osp.exists(next_img_path):
            logging.error("Cannot found image data: " + next_img_path)
            raise FileNotFoundError
        next_img = Image.open(next_img_path).convert('RGB')
        if next_img.size != img.size:
            import pdb; pdb.set_trace()
        assert next_img.size == img.size
        # get warp matrix
        # im1 = cv2.cvtColor(np.array(img),cv2.COLOR_RGB2BGR)
        # im2 = cv2.cvtColor(np.array(next_img),cv2.COLOR_RGB2BGR)
        # try:
        #     warp_matrix, _ = self.get_warp_matrix(im1, im2)
        #     warp_matrix = torch.as_tensor(warp_matrix).reshape(3, 3)
        # except:
        #     warp_matrix = torch.as_tensor([])

        ori_next_boxes = next_anno['ann']['bboxes']
        ori_next_labels = next_anno['ann']['labels']
        next_dataset_track_ids = next_anno['ann']['extra_anns']
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

        # TODO centertrack image transform for the offset train

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
        gt_ref_ids = dataset_track_ids[valid_pre_mask]
        assert ref_boxes.shape[-1] == 4
        out_target = dict(
            size=torch.from_numpy(np.array([h, w])),
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
            warp_matrix=warp_matrix,
            gt_ref_ids=gt_ref_ids
        )
        return img, next_img, out_target, filename
