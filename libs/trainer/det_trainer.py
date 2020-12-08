# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Created by Mingfei Chen (chenmingfei@sensetime.com)
# Created On: 2020-7-27
# ------------------------------------------------------------------------------
import datetime
import logging
import math
import numpy as np
import sys
import time
from tqdm import tqdm

import torch
from torch import autograd
from tensorboardX import SummaryWriter

from libs.trainer.trainer import BaseTrainer

from libs.utils.utils import AverageMeter, save_checkpoint
import libs.utils.misc as utils
from libs.utils.utils import write_dict_to_json



class DetTrainer(BaseTrainer):

    def __init__(self,
                 cfg,
                 model,
                 criterion,
                 optimizer,
                 lr_scheduler,
                 postprocessors,
                 log_dir='output',
                 performance_indicator='mAP',
                 last_iter=-1,
                 rank=0,
                 device='cuda',
                 max_norm=0,
                 logger=None):

        super().__init__(cfg, model, criterion, optimizer, lr_scheduler, 
            log_dir, performance_indicator, last_iter, rank)
        self.postprocessors = postprocessors
        self.device = device
        self.max_norm = max_norm
        self.logger = logger
        
    def _read_inputs(self, inputs):
        imgs, targets, filenames = inputs
        imgs = [img.to(self.device) for img in imgs]
        # targets are list type in det tasks
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
        return imgs, targets

    def _forward(self, data):
        imgs = data[0]
        targets = data[1]
        outputs = self.model(imgs)
        loss_dict = self.criterion(outputs, targets)
        return loss_dict

    def train(self, train_loader, eval_loader):
        start_time = time.time()
        self.model.train()
        self.criterion.train()
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        header = 'Epoch: [{}]'.format(self.epoch)
        print_freq = self.cfg.TRAIN.PRINT_FREQ

        if self.epoch > self.max_epoch:
            logging.info("Optimization is done !")
            sys.exit(0)
        for data in metric_logger.log_every(train_loader, print_freq, header, self.logger):
            data = self._read_inputs(data)
            loss_dict = self._forward(data)   
            weight_dict = self.criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                        for k, v in loss_dict_reduced.items()}
            loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                        for k, v in loss_dict_reduced.items() if k in weight_dict}
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

            loss_value = losses_reduced_scaled.item()

            if not math.isfinite(loss_value):
                if self.rank == 0:
                    self.logger.info("Loss is {}, stopping training".format(loss_value))
                    self.logger.info(loss_dict_reduced)
                    sys.exit(1)
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)

            self.optimizer.zero_grad()
            losses.backward()
            if self.max_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
            self.optimizer.step()

            metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
            metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        if self.rank == 0:
            self.logger.info("Averaged stats:", metric_logger)
        train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': self.epoch}
        if self.rank == 0:
            for (key, val) in log_stats.items():
                self.writer.add_scalar(key, val, log_stats['epoch'])
        self.lr_scheduler.step()

        # save checkpoint
        if self.rank == 0 and self.epoch > 0 and self.epoch % self.cfg.TRAIN.SAVE_INTERVAL == 0:
            # evaluation
            if self.cfg.TRAIN.VAL_WHEN_TRAIN:
                self.model.eval()
                performance = self.evaluate(eval_loader)
                self.writer.add_scalar(self.PI, performance, self.epoch)  
                if performance > self.best_performance:
                    self.is_best = True
                    self.best_performance = performance
                else:
                    self.is_best = False
                logging.info(f'Now: best {self.PI} is {self.best_performance}')
            else:
                performance = -1

            # save checkpoint
            try:
                state_dict = self.model.module.state_dict() # remove prefix of multi GPUs
            except AttributeError:
                state_dict = self.model.state_dict()

            if self.rank == 0:
                if self.cfg.TRAIN.SAVE_EVERY_CHECKPOINT:
                    filename = f"{self.model_name}_epoch{self.epoch:03d}_checkpoint.pth"
                else:
                    filename = "checkpoint.pth"
                save_checkpoint(
                    {
                        'epoch': self.epoch,
                        'model': self.model_name,
                        f'performance/{self.PI}': performance,
                        'state_dict': state_dict,
                        'optimizer': self.optimizer.state_dict(),
                    },
                    self.is_best,
                    self.log_dir,
                    filename=f'{self.cfg.OUTPUT_ROOT}_{filename}'
                )
        
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        if self.rank == 0:
            self.logger.info('Training time {}'.format(total_time_str))
        self.epoch += 1
        
    
    def evaluate(self, eval_loader, rel_topk=100):
        self.model.eval()
        results = []
        count = 0
        total_time = 0
        for data in tqdm(eval_loader):
            imgs, targets, filenames = data
            imgs = [img.to(self.device) for img in imgs]
            # targets are list type in det tasks
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            bs = len(imgs)
            # c, h, w = imgs[0].shape
            # target_sizes = torch.tensor([h, w]).expand(bs, 2)
            target_sizes = targets[0]['size'].expand(bs, 2)
            target_sizes = target_sizes.to(self.device)
            start_time = time.time()
            outputs_dict = self.model(imgs)
            file_name = filenames[0]
            pred_out = self.postprocessors(outputs_dict, file_name, target_sizes, rel_topk=rel_topk)
            total_time += self.postprocessors.end_time - start_time
            results.append(pred_out)
            count += 1
            if count % 1000 == 0:
                avg_t = total_time * 1.0 / 1000
                if self.rank == 0:
                    self.logger.info(avg_t)
                f = open('speed_log_512.txt', 'a')
                f.writelines(f'{count}: avg time {avg_t}\n')
                f.close()
                total_time = 0