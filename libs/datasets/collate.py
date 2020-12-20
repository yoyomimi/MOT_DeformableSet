import torch
from libs.utils.misc import nested_tensor_from_tensor_list

def collect(batch):
    """Collect the data for one batch.
    """
    imgs = []
    targets = []
    filenames = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(sample[1])
        filenames.append(sample[2])
    return imgs, targets, filenames

def joint_collect(batch):
    """Collect the data for one batch.
    """
    imgs = []
    next_imgs = []
    targets = []
    filenames = []
    for sample in batch:
        imgs.append(sample[0])
        next_imgs.append(sample[1])
        targets.append(sample[2])
        filenames.append(sample[3])
    return imgs, next_imgs, targets, filenames