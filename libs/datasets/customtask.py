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

class CustomTaskDataset(Dataset):

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
        id_set = set()
        for i, anno in enumerate(self.annotations):
            if i in count:
                id_base = id_base + max_id + 1
                max_id = 0
            anno['ann']['extra_anns'] = np.array(anno['ann']['extra_anns']).reshape(-1, )
            if(len(anno['ann']['extra_anns'])>0):
                max_id = max(max_id, max(anno['ann']['extra_anns']))
                anno['ann']['extra_anns'][anno['ann']['extra_anns']>-1] += id_base
                for single_id in anno['ann']['extra_anns']:
                    id_set.add(single_id)

            # if len(np.where(anno['ann']['extra_anns']>-1)[0]) < 1:
            #     continue
            
            if istrain is False:
                self.ids.append(i)
            else:
                flag_bad = 0
                boxes = anno['ann']['bboxes']
                labels = anno['ann']['labels']
                if (len(boxes) > 0 and len(boxes) <= max_obj and 
                     labels.sum()==len(labels)):
                    self.ids.append(i)
        id_base = id_base + max_id + 1 
        
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
        ids = anno['ann']['extra_anns']
        num_object = len(boxes)
        if num_object == 0:
            boxes = np.array([])
            labels = np.array([])
        boxes = torch.from_numpy(boxes.reshape(-1, 4).astype(np.float32))
        labels = torch.from_numpy(labels.reshape(-1, ).astype(np.int64))
        ids = torch.from_numpy(ids.reshape(-1, ).astype(np.int64))
        target = dict(
            boxes=boxes,
            labels=labels,
            sizes=torch.from_numpy(np.array([h, w])),
            ids=ids,
        )

        if self.transform is not None:
            img, target = self.transform(
                img, target
            )

        return img, target, filename
