import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

import os
import sys
# comment out this line after debugging
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from libs.utils import box_ops
from libs.utils.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from libs.models.backbone import build_backbone
from libs.models.matcher import build_matcher
from libs.models.transformer import build_transformer


class MOT_Transformer(nn.Module):
    """ This is the HOI Transformer module that performs HOI detection """
    def __init__(self, 
                 backbone, 
                 transformer, 
                 num_classes, 
                 num_queries=150, 
                 aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: dict of number of sub clses, obj clses and relation clses, 
                         omitting the special no-object category
                         keys: ["sub_labels", "obj_labels", "rel_labels"]
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.mot_class_embed = nn.Linear(hidden_dim, num_classes['obj_labels'])
        self.mot_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.mot_query_embed = nn.Embedding(num_queries, hidden_dim)

        # puppet
        self.puppet_mot_class_embed = nn.Linear(hidden_dim, num_classes['obj_labels'])
        self.puppet_mot_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.puppet_mot_query_embed = nn.Embedding(num_queries, hidden_dim)

        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

    def forward(self, samples, next_samples):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        samples.extend(next_samples)
        cat_input = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(cat_input)
        src, mask = features[-1].decompose()
        assert mask is not None
        hs, puppet_hs = self.transformer(self.input_proj(src), mask, self.mot_query_embed.weight,
            self.puppet_mot_query_embed.weight, pos[-1])[:2]
        outputs_class = self.mot_class_embed(hs)
        outputs_coord = self.mot_bbox_embed(hs).sigmoid()
        puppet_outputs_class = self.puppet_mot_class_embed(puppet_hs)
        puppet_outputs_coord = self.puppet_mot_bbox_embed(puppet_hs).sigmoid()

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1],
               'puppet_pred_logits': puppet_outputs_class[-1], 'puppet_pred_boxes': puppet_outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, puppet_outputs_class,
                puppet_outputs_coord)
        
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, puppet_outputs_class, puppet_outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b,
                 'puppet_pred_logits': p_a, 'puppet_pred_boxes': p_b,}
                for a, b, p_a, p_b in zip(outputs_class[:-1], outputs_coord[:-1],
                                          puppet_outputs_class[:-1], puppet_outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for HOI Transformer.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self,  
                 matcher, 
                 losses,
                 weight_dict,
                 eos_coef,  
                 num_classes):
        """ Create the criterion.
        Parameters:
            num_classes: dict of number of sub clses, obj clses and relation clses, 
                         omitting the special no-object category
                         keys: ["sub_labels", "obj_labels", "rel_labels"]
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes['obj_labels']
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True,
                     topk=1, alpha=0.25, gamma=2, loss_reduce='sum'):
        assert 'pred_logits' in outputs
        losses = {}
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_obj = torch.cat([t["labels"][J].to(src_logits.device) for t, (_, J) in zip(targets, indices)])
        target_classes = torch.zeros(src_logits.shape[0], src_logits.shape[1],
            self.num_classes).type_as(src_logits).to(src_logits.device)
        target_classes[idx] = target_classes_obj.type_as(src_logits)
        label = target_classes.long()
        
        pred_sigmoid = src_logits.sigmoid()
        pt = (1 - pred_sigmoid) * label + pred_sigmoid * (1 - label)
        focal_weight = (alpha * label + (1 - alpha) * (1 - label)) * pt.pow(gamma)
        loss_ce = F.binary_cross_entropy_with_logits(src_logits,
            target_classes, reduction='none') * focal_weight

        # for puppet
        puppet_src_logits = outputs['puppet_pred_logits']
        puppet_target_classes_obj = torch.cat([t["next_labels"][J].to(puppet_src_logits.device)
                                                for t, (_, J) in zip(targets, indices)])
        puppet_target_classes = torch.zeros(puppet_src_logits.shape[0], puppet_src_logits.shape[1],
            self.num_classes).type_as(puppet_src_logits).to(puppet_src_logits.device)
        puppet_target_classes[idx] = puppet_target_classes_obj.type_as(puppet_src_logits)
        puppet_label = puppet_target_classes.long() 
        puppet_pred_sigmoid = puppet_src_logits.sigmoid()
        puppet_pt = (1 - puppet_pred_sigmoid) * puppet_label + puppet_pred_sigmoid * (1 - puppet_label)
        puppet_focal_weight = (alpha * puppet_label + (1 - alpha) * (1 - puppet_label)) * puppet_pt.pow(gamma)
        puppet_loss_ce = F.binary_cross_entropy_with_logits(puppet_src_logits,
            puppet_target_classes, reduction='none') * puppet_focal_weight
        
        if loss_reduce == 'mean':
            losses['loss_ce'] = loss_ce.mean()
            losses['puppet_loss_ce'] = puppet_loss_ce.mean()
        else:
            losses['loss_ce'] = loss_ce.sum()
            losses['puppet_loss_ce'] = puppet_loss_ce.sum()
        if log:
            acc = (src_logits.sigmoid()[idx] > 0.5).sum()
            losses['class_error'] = (100 - 100 * acc / len(target_classes_obj)).to(src_logits.device).float()
            puppet_acc = (puppet_src_logits.sigmoid()[idx] > 0.5).sum()
            losses['puppet_class_error'] = (100 - 100 * puppet_acc / len(puppet_target_classes_obj)).to(
                puppet_src_logits.device).float()
            # can be used on multi-class classification
            # _, pred = src_logits[idx].topk(topk, 1, True, True)
            # acc = 0.0
            # for tid, target in enumerate(target_classes_obj):
            #     tgt_idx = torch.where(target==1)[0]
            #     if len(tgt_idx) == 0:
            #         continue
            #     acc_pred = 0.0
            #     for tgt in tgt_idx:
            #         acc_pred += (tgt in pred[tid])
            #     acc += acc_pred / len(tgt_idx)
            # labels_error = 100 - 100 * acc / len(target_classes_obj)
            # losses['class_error'] = torch.from_numpy(np.array(
            #     labels_error)).to(src_logits.device).float()
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.max(-1)[0] > 0.5).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        
        # puppet
        puppet_pred_logits = outputs['puppet_pred_logits']
        puppet_device = puppet_pred_logits.device
        puppet_tgt_lengths = torch.as_tensor([len(v["match_mask"]) for v in targets], device=puppet_device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        puppet_card_pred = (puppet_pred_logits.max(-1)[0] > 0.5).sum(1)
        puppet_card_err = F.l1_loss(puppet_card_pred.float(), puppet_tgt_lengths.float())
        
        losses = {'cardinality_error': card_err, 'puppet_cardinality_error': puppet_card_err}

        return losses


    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        puppet_src_boxes = outputs['puppet_pred_boxes'][idx]
        puppet_target_boxes = torch.cat([t['next_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        puppet_target_classes_obj = torch.cat([t["next_labels"][J].to(puppet_src_boxes.device)
                                                for t, (_, J) in zip(targets, indices)])
        puppet_loss_bbox = F.l1_loss(puppet_src_boxes, puppet_target_boxes, reduction='none') * puppet_target_classes_obj

        # for debug
        assert sum(puppet_target_classes_obj) == sum(len(t["match_mask"]) for t in targets)

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes['num_boxes']
        losses['puppet_loss_bbox'] = puppet_loss_bbox.sum() / num_boxes['puppet_num_boxes']

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        puppet_loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(puppet_src_boxes),
            box_ops.box_cxcywh_to_xyxy(puppet_target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes['num_boxes']
        losses['puppet_loss_giou'] = puppet_loss_giou.sum() / num_boxes['puppet_num_boxes']

        return losses


    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        puppet_num_boxes = sum(len(t["match_mask"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        puppet_num_boxes = torch.as_tensor([puppet_num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
            torch.distributed.all_reduce(puppet_num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        puppet_num_boxes = torch.clamp(puppet_num_boxes / get_world_size(), min=1).item()
        num_boxes_dict = {
            'num_boxes': num_boxes,
            'puppet_num_boxes': puppet_num_boxes
        }

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes_dict))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes_dict, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

# todo
class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def process_output(self, out_logits, out_bbox, scale_fct):
        prob = out_logits.sigmoid()
        scores, labels = prob.max(-1)
        labels = labels + 1
        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        boxes = boxes * scale_fct[:, None, :]
        scores = scores.reshape(-1)
        labels = labels.reshape(-1)
        boxes = boxes.reshape(-1, 4)
        return boxes, labels, scores

    @torch.no_grad()
    def forward(self, outputs, target_sizes, topk, nms_thr=0.35, min_thr=0.25):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        puppet_out_logits, puppet_out_bbox = outputs['puppet_pred_logits'], outputs['puppet_pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)

        boxes, labels, scores = self.process_output(out_logits, out_bbox, scale_fct)
        puppet_boxes, puppet_labels, puppet_scores = self.process_output(puppet_out_logits,
            puppet_out_bbox, scale_fct)

        valid_idx = torch.where(scores>=min_thr)[0]
        boxes = boxes[valid_idx]
        labels = labels[valid_idx]
        scores = scores[valid_idx]
        puppet_boxes = puppet_boxes[valid_idx]
        puppet_labels = puppet_labels[valid_idx]
        puppet_scores = puppet_scores[valid_idx]

        boxes, scores, labels, picked = box_ops.multiclass_nms(boxes, labels, scores, max_num=topk, nms_thr=nms_thr)
        assert len(boxes) == len(picked)
        # topk = min(topk, len(mot_scores))
        # _, seq_id = torch.topk(mot_scores, k=topk)
        # mot_scores = mot_scores[seq_id]
        # labels = labels[seq_id]
        puppet_labels = puppet_labels[picked]
        mot_scores = puppet_scores[picked] * scores
        # boxes = boxes[seq_id]
        puppet_boxes = puppet_boxes[picked]
        
        boxes = boxes.data.cpu().numpy()
        labels = labels.data.cpu().numpy()
        scores = scores.data.cpu().numpy()
        puppet_boxes = puppet_boxes.data.cpu().numpy()
        puppet_labels = puppet_labels.data.cpu().numpy()
        mot_scores = mot_scores.data.cpu().numpy()

        results = {'det_scores': scores, 'mot_scores': mot_scores, 'clses': labels, 'boxes': boxes,
                   'puppet_clses': puppet_labels, 'puppet_boxes': puppet_boxes}

        return results

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_model(cfg, device):
    backbone = build_backbone(cfg)

    transformer = build_transformer(cfg)

    num_classes=dict(
        obj_labels=cfg.DATASET.OBJ_NUM_CLASSES,
    )
    model = MOT_Transformer(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=cfg.TRANSFORMER.NUM_QUERIES,
        aux_loss=cfg.LOSS.AUX_LOSS,
    )
    matcher = build_matcher(cfg)
    weight_dict = {'loss_ce': cfg.LOSS.DET_CLS_COEF[0], 'loss_bbox': cfg.LOSS.BBOX_LOSS_COEF[0],
                   'puppet_loss_ce': cfg.LOSS.DET_CLS_COEF[0], 'puppet_loss_bbox': cfg.LOSS.BBOX_LOSS_COEF[0]}
    weight_dict['loss_giou'] = cfg.LOSS.GIOU_LOSS_COEF[0]
    weight_dict['puppet_loss_giou'] = cfg.LOSS.GIOU_LOSS_COEF[0]

    if cfg.LOSS.AUX_LOSS:
        aux_weight_dict = {}
        for i in range(cfg.TRANSFORMER.DEC_LAYERS - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    criterion = SetCriterion(matcher=matcher, losses=losses, weight_dict=weight_dict,
                             eos_coef=cfg.LOSS.EOS_COEF, num_classes=num_classes)
    criterion.to(device)
    postprocessors = PostProcess()
    return model, criterion, postprocessors
    