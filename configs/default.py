# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Created by Mingfei Chen (lasiafly@gmail.com)
# Created On: 2020-1-20
# ------------------------------------------------------------------------------
import numpy as np

from yacs.config import CfgNode as CN


INF = 1e8

_C = CN()

# working dir
_C.OUTPUT_ROOT = ''

# distribution
_C.DIST_BACKEND = 'nccl'

_C.DEVICE = 'cuda'

_C.WORKERS = 4

_C.PI = 'mAP'

_C.SEED = 42

# cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# dataset
_C.DATASET = CN()
_C.DATASET.FILE = 'hoi_det'
_C.DATASET.NAME = 'HICODetDataset'
_C.DATASET.ROOT = ''
_C.DATASET.MEAN = []
_C.DATASET.STD = []
_C.DATASET.MAX_SIZE = 1333
_C.DATASET.SCALES = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
_C.DATASET.IMG_NUM_PER_GPU = 2
_C.DATASET.NUM_CLASSES = 91
_C.DATASET.PREFIX = ''

# model
_C.MODEL = CN()
# specific model 
_C.MODEL.FILE = ''
_C.MODEL.NAME = ''
# resume
_C.MODEL.RESUME_PATH = ''
_C.MODEL.MASKS = False

# backbone
_C.BACKBONE = CN()
_C.BACKBONE.NAME = 'resnet50'
_C.BACKBONE.DIALATION = False
_C.BACKBONE.DEPTH = 50
_C.BACKBONE.STRIDE_IN_1X1 = False
_C.BACKBONE.OUT_FEATURES = ["res2", "res3", "res4", "res5"]

# deformable detr
_C.DEFORMABLE = CN()
_C.DEFORMABLE.WITH_BOX_REFINE = False
_C.DEFORMABLE.TWO_STAGE = False

# transformer
_C.TRANSFORMER = CN()
_C.TRANSFORMER.POSITION_EMBEDDING = 'sine' # choices=('sine', 'learned')
_C.TRANSFORMER.POSITION_EMBEDDING_SCALE = 2 * np.pi
_C.TRANSFORMER.NUM_FEATURE_LEVELS = 4
_C.TRANSFORMER.HIDDEN_DIM = 256
_C.TRANSFORMER.ENC_LAYERS = 6
_C.TRANSFORMER.DEC_LAYERS = 6
_C.TRANSFORMER.DIM_FEEDFORWARD = 2048
_C.TRANSFORMER.DROPOUT = 0.1
_C.TRANSFORMER.NHEADS = 8
_C.TRANSFORMER.NUM_QUERIES = 300
_C.TRANSFORMER.DEC_N_POINTS = 4
_C.TRANSFORMER.ENC_N_POINTS = 4
_C.TRANSFORMER.PRE_NORM = False

# matcher
_C.MATCHER = CN()
_C.MATCHER.COST_CLASS = 1
_C.MATCHER.COST_BBOX = 5
_C.MATCHER.COST_GIOU = 2

# LOSS
_C.LOSS = CN()
_C.LOSS.AUX_LOSS = True
_C.LOSS.DICE_LOSS_COEF = 1
_C.LOSS.CLS_LOSS_COEF = 1
_C.LOSS.BBOX_LOSS_COEF = 5
_C.LOSS.GIOU_LOSS_COEF = 2
_C.LOSS.FOCAL_ALPHA = 0.25
_C.LOSS.EOS_COEF = 0.1

# trainer
_C.TRAINER = CN()
_C.TRAINER.FILE = ''
_C.TRAINER.NAME = ''

# train
_C.TRAIN = CN()
_C.TRAIN.OPTIMIZER = ''
_C.TRAIN.LR = 0.0002
_C.TRAIN.LR_BACKBONE_NAMES = ["backbone.0"]
_C.TRAIN.LR_BACKBONE = 0.00002
_C.TRAIN.LR_PROJ_NAMES = ['reference_points', 'sampling_offsets']
_C.TRAIN.LR_PROJ_MULT = 0.1
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WEIGHT_DECAY = 0.0001
# optimizer SGD
_C.TRAIN.NESTEROV = False
# learning rate scheduler
_C.TRAIN.LR_SCHEDULER = 'MultiStepWithWarmup'
_C.TRAIN.LR_STEPS = [120000, ]
_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_DROP = 200
_C.TRAIN.CLIP_MAX_NORM = 0.1
_C.TRAIN.WARMUP_STEP = 500
_C.TRAIN.WARMUP_INIT_FACTOR = 1.0 / 3
_C.TRAIN.MAX_EPOCH = 360000
# train resume
_C.TRAIN.RESUME = False
# input size
_C.TRAIN.INPUT_MIN = 788
_C.TRAIN.INPUT_MAX = 1400
# print freq
_C.TRAIN.PRINT_FREQ = 20
# save checkpoint during train
_C.TRAIN.SAVE_INTERVAL = 5000
_C.TRAIN.SAVE_EVERY_CHECKPOINT = False
# val when train
_C.TRAIN.VAL_WHEN_TRAIN = False

# test
_C.TEST = CN()
# input size
_C.TEST.SCORE_THRESHOLD = 0.3
_C.TEST.NMS_THRESHOLD = 0.5
# test image dir
_C.TEST.IMAGE_DIR = ''
_C.TEST.TEST_SIZE = (512, 768)
_C.TEST.CROP_SZIE = (182, 182)
_C.TEST.MAX_PER_IMG = 100
_C.TEST.NMS_PRE = 1000
_C.TEST.PR_CURVE = False
_C.TEST.OUT_DIR = ''
_C.TEST.AP_IOU = 0.5


def update_config(config, args):
    config.defrost()
    # set cfg using yaml config file
    config.merge_from_file(args.yaml_file)
    # update cfg using args
    config.merge_from_list(args.opts)
    config.freeze()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)