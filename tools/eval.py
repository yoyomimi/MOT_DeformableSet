# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Created by Mingfei Chen (chenmingfei@sensetime.com)
# Created On: 2020-7-24
# ------------------------------------------------------------------------------
from __future__ import division
from __future__ import print_function
from __future__ import with_statement

import argparse
import logging
import os
import random

import pprint
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

import _init_paths
from configs import cfg
from configs import update_config

from libs.datasets.collate import collect
from libs.utils.utils import create_logger
from libs.utils.utils import get_model
from libs.utils.utils import get_dataset
from libs.utils.utils import get_trainer
from libs.utils.utils import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='HOI Transformer Task')
    parser.add_argument(
        '--cfg',
        dest='yaml_file',
        help='experiment configure file name, e.g. configs/fcos_detector.yaml',
        required=True,
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

def get_ip(ip_addr):
    ip_list = ip_addr.split('-')[2:6]
    for i in range(4):
        if ip_list[i][0] == '[':
            ip_list[i] = ip_list[i][1:].split(',')[0]
    return f'tcp://{ip_list[0]}.{ip_list[1]}.{ip_list[2]}.{ip_list[3]}:23456'

def main_per_worker():
    args = parse_args()

    update_config(cfg, args)
    ngpus_per_node = torch.cuda.device_count()
    
    # torch seed
    torch.cuda.manual_seed(random.random())

    # cudnn
    # cudnn.benchmark = cfg.CUDNN.BENCHMARK
    # torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    if 'SLURM_PROCID' in os.environ.keys():
        proc_rank = int(os.environ['SLURM_PROCID'])
        local_rank = proc_rank % ngpus_per_node
        args.world_size = int(os.environ['SLURM_NTASKS'])
    else:
        proc_rank = 0
        local_rank = 0
        args.world_size = 1

    args.distributed = (args.world_size > 1 or args.distributed)
    
    #create logger
    if proc_rank == 0:
        logger, output_dir = create_logger(cfg, proc_rank)
    # logger.info(pprint.pformat(args))
    # logger.info(cfg)

    # distribution
    if args.distributed:
        dist_url = get_ip(os.environ['SLURM_STEP_NODELIST'])
        if proc_rank == 0:
            logger.info(
                f'Init process group: dist_url: {dist_url},  '
                f'world_size: {args.world_size}, '
                f'proc_rank: {proc_rank}, '
                f'local_rank:{local_rank}'
            )  
        dist.init_process_group(
            backend=cfg.DIST_BACKEND,
            init_method=dist_url,
            world_size=args.world_size,
            rank=proc_rank
        )
        torch.distributed.barrier()
        torch.cuda.set_device(local_rank)
        device = torch.device(cfg.DEVICE)
        model, criterion, postprocessors = get_model(cfg, device)  
        model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True
        )
        _, eval_dataset = get_dataset(cfg)
        batch_size = cfg.DATASET.IMG_NUM_PER_GPU

    else:
        assert proc_rank == 0, ('proc_rank != 0, it will influence '
                                'the evaluation procedure')
        if cfg.DEVICE == 'cuda':
            torch.cuda.set_device(local_rank)
        device = torch.device(cfg.DEVICE)
        model, criterion, postprocessors = get_model(cfg, device)  
        model = torch.nn.DataParallel(model).to(device)
        _, eval_dataset = get_dataset(cfg)
        if ngpus_per_node == 0:
            batch_size = cfg.DATASET.IMG_NUM_PER_GPU
        else:
            batch_size = cfg.DATASET.IMG_NUM_PER_GPU * ngpus_per_node
    
    model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters()
                        if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters()
                        if "backbone" in n and p.requires_grad],
            "lr": cfg.TRAIN.LR_BACKBONE,
        },
    ]

    resume_path = cfg.MODEL.RESUME_PATH
    if os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location='cpu')
        # resume
        if 'state_dict' in checkpoint:
            model.module.load_state_dict(checkpoint['state_dict'], strict=True)
            logging.info(f'==> model pretrained from {resume_path} \n')


    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        collate_fn=collect,
        num_workers=cfg.WORKERS
    )

    Trainer = get_trainer(
        cfg,
        model,
        criterion=criterion,
        optimizer=None,
        lr_scheduler=None,
        postprocessors=postprocessors,
        log_dir='output',
        performance_indicator='mAP',
        last_iter=-1,
        rank=proc_rank,
        device=device,
        max_norm=None
    )

    print('start eval...')
    Trainer.evaluate(eval_loader)


if __name__ == '__main__':
    main_per_worker()