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

import mmcv

import torch
from torch import autograd
from torchvision.ops import nms
from tensorboardX import SummaryWriter

from libs.evaluation.mean_ap import eval_map
from libs.trainer.trainer import BaseTrainer
from libs.utils.box_ops import box_cxcywh_to_xyxy
from libs.utils.utils import AverageMeter, save_checkpoint
import libs.utils.misc as utils
from libs.utils.utils import write_dict_to_json



class TrackTrainer(BaseTrainer):

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
        self.prev_features = None
        
    def _read_inputs(self, inputs):
        imgs, next_imgs, targets, filenames = inputs
        imgs = [img.to(self.device) for img in imgs]
        next_imgs = [img.to(self.device) for img in next_imgs]
        # targets are list type in det tasks
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
        return imgs, next_imgs, targets

    def _forward(self, data):
        imgs = data[0]
        next_imgs = data[1]
        targets = data[2]
        outputs = self.model(imgs)
        loss_dict_prev = self.criterion(outputs, targets)
        ### Modified ###
        indices = self.criterion.out_indices
        out_id_features = self.model.module.out_id_features.detach()
        out_pred_next_boxes = self.model.module.out_pred_next_boxes.detach()
        assert len(indices) == len(targets)
        references = []
        for i in range(len(indices)):
            src, tgt = indices[i]
            matched_idx = targets[i]['matched_idx']
            if len(matched_idx) == 0:
                references = []
                break
            idx_map = targets[i]['idx_map']
            ref_boxes = targets[i]['ref_boxes']
            id_features = out_id_features[i]
            pred_next_boxes = out_pred_next_boxes[i]
            valid_idx = [torch.where(tgt==idx)[0][0] for idx in matched_idx]
            prev_features = id_features[src[torch.as_tensor(valid_idx).reshape(-1, ).long()]]
            input_size = torch.as_tensor([imgs[i].shape[2], imgs[i].shape[1]]).reshape(1, 2).long()
            # ref_boxes = pred_next_boxes[src[torch.stack(valid_idx)]]
            references.append(dict(ref_features=prev_features, ref_boxes=ref_boxes, idx_map=idx_map,
                                   input_size=input_size))
        if len(references) == 0:
            references = None
        outputs = self.model(next_imgs, references)
        loss_dict_next = self.criterion(outputs,targets, self.model.module.ref_indices, is_next=True)
        loss_dict = {k: 0.5*(v + loss_dict_next[k]) for k, v in loss_dict_prev.items()}
        ##########
        return loss_dict

    def train(self, train_loader, eval_loader):
        start_time = time.time()
        self.model.train()
        self.criterion.train()
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        metric_logger.add_meter('id_class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
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
            metric_logger.update(id_class_error=loss_dict_reduced['id_class_error'])
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
        
    
    def evaluate(self, eval_loader, num_classes, threshold=0.05):
        self.model.eval()
        det_results = []
        annotations = []
        results_dict = {}
        count = 0
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
            outputs_dict = self.model(imgs)
            file_name = filenames[0]
            pred_out = self.postprocessors(outputs_dict, file_name, target_sizes)

            # for level, res in enumerate(pred_out):
            #     results_dict.setdefault(level, [])
            #     valid_inds = torch.where(res['scores']>threshold)[0]
            #     boxes = res['boxes'][valid_inds].clamp(
            #         min=0)
            #     labels = res['labels'][valid_inds]
            #     scores = res['scores'][valid_inds]
            #     det_boxes = []
            #     for i in range(1, num_classes):
            #         valid_id = torch.where(labels==i)[0]
            #         keep = nms(boxes[valid_id], scores[valid_id], 0.5)
            #         boxes = boxes[valid_id][keep].reshape(
            #             -1, 4).data.cpu().numpy()
            #         scores = scores[valid_id][keep].reshape(
            #             -1, 1).data.cpu().numpy()
            #         valid_det_res = np.hstack([boxes, scores])
            #         det_boxes.append(valid_det_res)
            #     results_dict[level].append(det_boxes)
            
            target = targets[0]
            labels = target['labels'].reshape(-1, )
            img_h, img_w = target['size']
            scale_fct = np.array([img_w, img_h, img_w, img_h])
            boxes = box_cxcywh_to_xyxy(target['boxes'])
            boxes = boxes.clamp(min=0).reshape(-1, 4)
            anno = {
                'bboxes': boxes.data.cpu().numpy() * scale_fct,
                'labels': labels.data.cpu().numpy()-1,
                'bboxes_ignore': np.array([]).reshape(-1, 4),
                'labels_ignore': np.array([]).reshape(-1, )
            }
            annotations.append(anno)
            det_results.append(pred_out)
            
        mmcv.dump(det_results, 'data/det_results_more_epoch30.pkl')
        mmcv.dump(annotations, 'data/test_annos.pkl')
        
        # for i in results_dict.keys():
        #     det_results = results_dict[i]
        #     eval_map(det_results, annotations, i, iou_thr=0.5, nproc=4)
