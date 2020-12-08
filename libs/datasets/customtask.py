import json
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
                 data_root,
                 img_root,
                 transform=None,
                 istrain=False,
                 max_obj=100,
                 ):
        self.img_root = img_root
        self.transform = transform
        pkl_path = f'{data_root}/pkl'
        annotations = []
        for path in os.listdir(pkl_path):
            if path.split('.')[-1] != 'pkl':
                continue
            if istrain and path.find('train') != -1:
                annotations.extend(mmcv.load(f'{pkl_path}/{path}'))
            elif istrain is False:
                annotations.extend(mmcv.load(f'{pkl_path}/{path}'))
            else:
                continue
        self.annotations = annotations
        self.ids = []
        self.max_objs = 100
        for i, anno in enumerate(self.annotations):
            if istrain is False:
                self.ids.append(i)
            else:
                flag_bad = 0
                boxes = anno['ann']['bboxes']
                labels = anno['ann']['labels']
                if (len(boxes) > 0 and len(boxes) <= max_obj and 
                     labels.sum()==len(labels)):
                    self.ids.append(i)

        
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
        num_object = len(boxes)
        if num_object == 0:
            boxes = np.array([])
            labels = np.array([])
        boxes = torch.from_numpy(boxes.reshape(-1, 4).astype(np.float32))
        labels = torch.from_numpy(labels.reshape(-1, ).astype(np.int64))
        target = dict(
            boxes=boxes,
            labels=labels,
            sizes=torch.from_numpy(np.array([h, w]))
        )

        if self.transform is not None:
            img, target = self.transform(
                img, target
            )

        return img, target, filename