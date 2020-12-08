# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Transforms and data augmentation for both image + bbox.
"""
import random

import PIL
import torch
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as F

from libs.utils.box_ops import box_xyxy_to_cxcywh


def hflip(image, target):
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)

    return flipped_image, target

def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    if float(torchvision.__version__[:3]) < 0.7:
        if input.numel() > 0:
            return torch.nn.functional.interpolate(
                input, size, scale_factor, mode, align_corners
            )

        output_shape = _output_size(2, input, size, scale_factor)
        output_shape = list(input.shape[:-2]) + list(output_shape)
        return _new_empty_tensor(input, output_shape)
    else:
        return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)


def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, target


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            if isinstance(img, list):
                assert len(img) == len(target)
                out_img = []
                out_target = []
                for im, t in zip(img, target):
                    out_im, out_t = hflip(im, t)
                    out_img.append(out_im)
                    out_target.append(out_t)
                return out_img, out_target
            else:
                return hflip(img, target)
        return img, target


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        if isinstance(img, list):
            assert len(img) == len(target)
            out_img = []
            out_target = []
            for im, t in zip(img, target):
                out_im, out_t = resize(im, t, size, self.max_size)
                out_img.append(out_im)
                out_target.append(out_t)
            return out_img, out_target
        else:
            return resize(img, target, size, self.max_size)


class ToTensor(object):
    def __call__(self, img, target):
        if isinstance(img, list):
            out_img = [F.to_tensor(im) for im in img]
        else:
            out_img = F.to_tensor(img)
        return out_img, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        if isinstance(image, list):
            out_image = [F.normalize(im, mean=self.mean, std=self.std) for im in image]
            if target is None:
                return out_image, None
            assert len(out_image) == len(target)
            out_target = []
            for im, t in zip(out_image, target):
                out_t = t.copy()
                h, w = im.shape[-2:]
                if "boxes" in out_t:
                    boxes = out_t["boxes"]
                    boxes = box_xyxy_to_cxcywh(boxes)
                    boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
                    out_t["boxes"] = boxes
                    out_target.append(out_t)
            return out_image, out_target
        else:
            image = F.normalize(image, mean=self.mean, std=self.std)
            if target is None:
                return image, None
            target = target.copy()
            h, w = image.shape[-2:]
            if "boxes" in target:
                boxes = target["boxes"]
                boxes = box_xyxy_to_cxcywh(boxes)
                boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
                target["boxes"] = boxes
            return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class TrainTransform(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], 
                 scales=[608, 640, 672, 704, 736, 768, 800], max_size=1333):
        normalize = Compose([
            ToTensor(),
            Normalize(mean, std)
        ])
        self.augment = Compose([
            RandomHorizontalFlip(),
            RandomResize(scales, max_size=max_size),
            normalize,
        ])

    def __call__(self, img, target):
        # target["boxes"] xyxy; "masks"(optional)
        return self.augment(img, target)


class EvalTransform(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], 
                 max_size=1333):
        normalize = Compose([
            ToTensor(),
            Normalize(mean, std)
        ])
        self.augment = Compose([
            RandomResize([800], max_size=max_size),
            normalize,
        ])
    
    def __call__(self, img, target):
        return self.augment(img, target)