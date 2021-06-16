# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Deformable DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
import math
import yaml

from libs.models.yolov5.utils.general import xyxy2xywh, non_max_suppression
from libs.models.yolov5.utils.loss import ComputeLoss, sigmoid_focal_loss
from libs.models.yolov5.yolo import Model as YOLOv5
from libs.models.matcher import build_detmatcher, build_matchtrack_matcher
from libs.models.position_encoding import build_position_encoding
from libs.models.reference_search import build_refersearch
from libs.utils import box_ops
from libs.utils.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)
import copy


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class YOLOv5Match(nn.Module):
    def __init__(self, detector, transformer, position_embedding, num_classes, num_queries,
                 num_feature_levels=3, emb_dim=64, dataset_nids=5777, aux_loss=True,
                 num_match_decoder_layers=6):
        # ch: 439046 14687 60000 776 2324 897 757
        """ Initializes the model.
        Parameters:
            detector: YOLOv5 detector
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.detector = detector
        self.position_embedding = position_embedding
        self.transformer = transformer
        self.dataset_nids = dataset_nids
        hidden_dim = transformer.d_model
        self.id_embed = nn.ModuleList(MLP(x, hidden_dim, emb_dim, 3)
             for x in self.detector.model[-1].ch)
        self.nID = dataset_nids
        self.emb_dim = emb_dim
        self.id_head = nn.Linear(self.emb_dim, self.nID)
        nn.init.normal_(self.id_head.weight, std=0.01)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.id_head.bias, bias_value)
        self.emb_scale = math.sqrt(2) * math.log(self.nID - 1)
        self.num_feature_levels = num_feature_levels
        num_backbone_outs = detector.model[-1].nl
        num_backbone_channels = detector.model[-1].ch
        if num_feature_levels > 1:
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = num_backbone_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(num_backbone_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.aux_loss = aux_loss

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        
        # match
        self.ref_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        nn.init.constant_(self.ref_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.ref_embed.layers[-1].bias.data, 0)
        self.ref_embed = _get_clones(self.ref_embed, num_match_decoder_layers)
        nn.init.constant_(self.ref_embed[0].layers[-1].bias.data[2:], -2.0)
        self.ref_id_embed = MLP(hidden_dim, hidden_dim, emb_dim, 3)

    def forward(self, samples: NestedTensor, references=None):
        """ The forward expects a NestedTensor, which consists of:
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
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        xs, det_preds = self.detector(samples.tensors)
        features, pos = [], []
        for x in xs:
            m = samples.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            nested_x = NestedTensor(x.clone(), mask)
            features.append(nested_x)
            pos.append(self.position_embedding(nested_x).to(nested_x.tensors.dtype))
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.position_embedding(NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        self.out_memory = srcs

        if references is not None:
            query_embeds = None
            match_hs, match_inter_references = self.transformer(
                srcs, masks, pos, query_embeds, references=references)
            ref_coords = []
            ref_boxes = torch.stack([v["ref_boxes"] for v in references], dim=0)
            for lvl in range(match_hs.shape[0]):
                if lvl == 0:
                    reference = ref_boxes[..., :2]
                else:
                    reference = match_inter_references[lvl - 1]
                reference = inverse_sigmoid(reference)
                if lvl == match_hs.shape[0] - 1:
                    ref_id_embed = self.ref_id_embed(match_hs[lvl])
                    ref_id_feature = ref_id_embed.clone()
                    ref_id_embed = self.emb_scale * F.normalize(ref_id_embed)
                    ref_id_embed = self.id_head(ref_id_embed.contiguous())
                tmp = self.ref_embed[lvl](match_hs[lvl])
                tmp = tmp + reference
                ref_coords.append(tmp.sigmoid())
        else:
            match_hs = None

        outputs_classes = []
        outputs_coords = []
        outputs_id_embeds = []
        outputs_id_features = []
        outputs_next_centers = []
    
        z = []
        grid = [torch.zeros(1)] * len(det_preds)
        for i in range(len(det_preds)):
            bs, _, ny, nx, _ = det_preds[i].shape
            yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
            grid[i] = torch.stack((xv, yv), 2).view((
                1, 1, ny, nx, 2)).float().to(det_preds[i].device)
            y = det_preds[i].sigmoid()
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid[i]) * self.detector.model[
                -1].stride[i]  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.detector.model[
                -1].anchor_grid[i]  # wh
            outputs_id_embed = self.id_embed[i](xs[i].permute(0,2,3,1).unsqueeze(1).repeat(
                1, self.detector.model[-1].na, 1, 1, 1))
            outputs_id_features.append(outputs_id_embed.contiguous().view(bs, -1, self.emb_dim))
            outputs_id_embed = self.emb_scale * F.normalize(outputs_id_embed)
            outputs_id_embed = self.id_head(outputs_id_embed)
            outputs_id_embeds.append(outputs_id_embed)
            z.append(y.view(bs, -1, self.detector.model[-1].no))
            
        out = torch.cat(z, 1)
        outputs_id_features = torch.cat(outputs_id_features, 1)
        final_out_preds, final_id_features = non_max_suppression(
            out, 0.05, 0.45, max_det=self.num_queries, addict=outputs_id_features)
        outputs_coords = []
        out_id_features = []
        out_probs = []

        for i in range(len(final_out_preds)):
            coord = torch.zeros(self.num_queries, 4).to(out.device)
            prob = torch.zeros(self.num_queries, 1).to(out.device)
            id_embs = torch.zeros(self.num_queries, self.emb_dim).to(out.device)
            id_features = torch.zeros(self.num_queries, self.emb_dim).to(out.device)
            ref_num = len(final_out_preds[i])
            coord[:ref_num] = xyxy2xywh(final_out_preds[i][..., :4].view(-1, 4))
            prob[:ref_num] = final_out_preds[i][..., 4].view(-1, 1)
            id_features[:ref_num] = final_id_features[i]
            outputs_coords.append(coord)
            out_probs.append(prob)
            out_id_features.append(id_features)
        
        outputs_coords = torch.stack(outputs_coords)
        out_probs = torch.stack(out_probs)
        out_id_features = torch.stack(out_id_features)

        self.outputs_coords = outputs_coords.clone()
        self.out_probs = out_probs.clone()
        self.out_id_features = out_id_features.clone()
        
        self.outputs_id_embeds = outputs_id_embeds
        self.outputs_id_features = outputs_id_features.clone()
    
        out_dict = {'pred_logits': out_probs, 'pred_boxes': outputs_coords,
            'id_embeds': outputs_id_embeds, 'id_features': outputs_id_features,
            'det_out': det_preds
        }
        

        if references is not None and match_hs is not None:
            out_dict['ref_outputs'] = {'ref_coords': ref_coords[-1],
               'ref_id_embeds': ref_id_embed, 'ref_id_features': ref_id_feature}
            out_dict['aux_ref_outputs'] = [{'ref_coords': a} for a in ref_coords[:-1]]
        return out_dict

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b,}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, detector, num_classes, matcher, weight_dict, losses, refer_matcher=None, focal_alpha=0.25):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.det_loss = ComputeLoss(detector)
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.refer_matcher = refer_matcher

    def loss_boxes(self, outputs, targets, num_boxes, log=True, is_next=False):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        losses = {}
        det_preds = outputs['det_out']
        outputs_id_embeds = outputs['id_embeds']
        cat_targets = []
        if is_next is True:
            for i, t in enumerate(targets):
                target = torch.zeros(t['next_boxes'].shape[0], 7).to(det_preds[0].device)
                target[..., 0] = i
                target[..., 1] = t['next_labels']
                target[..., 2:6] = t['next_boxes']
                target[..., 6] = t['next_ids']
                cat_targets.append(target)
        else:
            for i, t in enumerate(targets):
                target = torch.zeros(t['boxes'].shape[0], 7).to(det_preds[0].device)
                target[..., 0] = i
                target[..., 1] = t['labels']
                target[..., 2:6] = t['boxes']
                target[..., 6] = t['ids']
                cat_targets.append(target)

        cat_targets = torch.cat(cat_targets, dim=0)

        lbox, lobj, _, lids = self.det_loss([det_preds, outputs_id_embeds], cat_targets)
        lbox /= num_boxes
        lids /= num_boxes
        lobj /= len(det_preds)
        losses = {
            'loss_ce': lobj, 'loss_bbox': lbox,
            'loss_ids': lids
        }

        return losses

    def loss_match(self, outputs, targets, indices, num_boxes, q_padding_mask=None, log=True, is_next=False):
        assert is_next is True
        losses = {}
        if 'ref_outputs' not in outputs:
            device = outputs['pred_boxes'].device
            losses['loss_offset'] = torch.Tensor([0.]).mean().to(device)
            losses['aux_loss_offset_0'] = torch.Tensor([0.]).mean().to(device)
            losses['aux_loss_offset_1'] = torch.Tensor([0.]).mean().to(device)
            losses['aux_loss_offset_2'] = torch.Tensor([0.]).mean().to(device)
            losses['aux_loss_offset_3'] = torch.Tensor([0.]).mean().to(device)
            losses['aux_loss_offset_4'] = torch.Tensor([0.]).mean().to(device)
            return losses

        target_ids = torch.cat([t['gt_ref_ids'] for t in targets], dim=0).long()
        valids_ids = torch.where(target_ids>-1)[0]
        if len(valids_ids) == 0:
            flag = 0.0
        else:
            flag = 1.0
        target_next_centers = torch.cat([t['gt_ref_boxes'][..., :2] for t in targets], dim=0)
        src_next_centers = torch.cat([coord[~q_padding_mask[i]] for i, coord in enumerate(
            outputs['ref_outputs']['ref_coords'])], dim=0)
        loss_offset = F.l1_loss(src_next_centers, target_next_centers, reduction='none')
        losses['loss_offset'] = loss_offset.sum() / num_boxes * flag
        aux_outputs = outputs['aux_ref_outputs']
        for i, aux_output in enumerate(aux_outputs):
            src_next_centers = torch.cat([coord[~q_padding_mask[i]] for i, coord in enumerate(
                aux_output['ref_coords'])], dim=0)
            loss_offset = F.l1_loss(src_next_centers, target_next_centers, reduction='none')
            losses[f'aux_loss_offset_{i}'] = loss_offset.sum() / num_boxes * flag

        # id
        target_ids = target_ids[valids_ids]
        outputs_src_id_logits = torch.cat([ref_id_emb[~q_padding_mask[i]] for i, ref_id_emb in enumerate(
            outputs['ref_outputs']['ref_id_embeds'])], dim=0)
        outputs_src_id_logits = outputs_src_id_logits[valids_ids].contiguous()
        # loss_ids = F.cross_entropy(outputs_src_id_logits, target_ids, ignore_index=-1)
        target_ids_onehot = torch.zeros([outputs_src_id_logits.shape[0], outputs_src_id_logits.shape[1] + 1],
                                        dtype=outputs_src_id_logits.dtype, layout=outputs_src_id_logits.layout, device=outputs_src_id_logits.device)
        target_ids_onehot.scatter_(1, target_ids.unsqueeze(-1), 1)

        target_ids_onehot = target_ids_onehot[:,:-1]
        loss_ids = sigmoid_focal_loss(outputs_src_id_logits, target_ids_onehot, reduction='sum', alpha=self.focal_alpha, gamma=2) / max(
            1, outputs_src_id_logits.shape[0])
        if log:
            losses['id_class_error'] = 100 - accuracy(outputs_src_id_logits, target_ids)[0]
        losses['match_loss_ids'] = loss_ids
        return losses
        
    def get_loss(self, loss, outputs, targets, num_boxes, **kwargs):
        loss_map = {
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, num_boxes, **kwargs)

    def forward(self, outputs, targets, is_next=False, references=None):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
             references(optional): This is a list of references (len(references) = batch_size), where each reference is a dict containing:
                "ref_features": Tensor of dim [num_refer_boxes, emb_dim] (where num_refer_boxes is the number of detected boxes 
                                in the previous frame) containing the reference appearance embeddings
                "ref_boxes": Tensor of dim [num_refer_boxes, 4] containing the reference box coordinates predicted 
                             in the previous frame
                "idx_map": Tensor of dim [num_refer_boxes, num_targets], map the refer index to the target index
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs' and k != 'ref_outputs' and k != 'aux_ref_outputs'}
        # Retrieve the matching between the outputs of the last layer and the targets
        ref_indices = None
        indices = self.matcher(outputs_without_aux, targets, ref_indices, is_next)
        self.out_indices = indices
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        if is_next is True:
            num_boxes = sum(len(t["next_labels"]) for t in targets)
        else:
            num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            kwargs['is_next'] = is_next
            losses.update(self.get_loss(loss, outputs, targets, num_boxes, **kwargs))

        # ref outputs
        if is_next is True:
            # regression loss for ref outputs
            if references is None:
                ref_num_boxes = num_boxes
            else:
                ref_num_boxes = sum(len(r["gt_ref_boxes"]) for r in references)
                ref_num_boxes = torch.as_tensor([ref_num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
                if is_dist_avail_and_initialized():
                    torch.distributed.all_reduce(ref_num_boxes)
                ref_num_boxes = torch.clamp(ref_num_boxes / get_world_size(), min=1).item()
                q_padding_mask = torch.stack([r["padding_mask"] for r in references], dim=0)
                losses.update(self.loss_match(outputs, references, None, ref_num_boxes, q_padding_mask, is_next=is_next))

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self):
        super().__init__()
        from libs.models.matcher import MatchTrackMatcher
        self.refer_matcher = MatchTrackMatcher(det_thr=0.4, cost_limit=1.0)

    @torch.no_grad()
    def forward(self, outputs, filename, target_sizes, references=None):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        id_features = outputs['id_features']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)

        track_idx = -torch.ones(out_bbox.shape[1]).long()
        # TODO get strack idx map with box
        if references is not None:
            ref_outputs = outputs['ref_outputs']
            _, ref_indices = self.refer_matcher(ref_outputs, outputs, references)
            assert len(ref_indices) == 1
            src_idx, tgt_idx = ref_indices[0]
            for sid, tid in zip(src_idx, tgt_idx):
                track_idx[sid] = tid
            ref_coords = ref_outputs['ref_coords']
            ref_id_features = ref_outputs['ref_id_features']
        prob = out_logits
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        boxes = boxes * scale_fct[:, None, :]

        filenames = [filename] * len(scores)
        results = [{'filename': f, 'scores': s, 'labels': l, 'boxes': b} for f, s, l, b in zip(
            filenames, scores, labels, boxes)]
        results[-1]['out_pred_boxes'] = boxes[-1]
        results[-1]['id_features'] = id_features.reshape(-1, id_features.shape[
            -1])[topk_boxes[0]]
        results[-1]['track_idx'] = track_idx[topk_boxes[0]]
        if references is not None:
            results[-1]['ref_coords'] = ref_coords * scale_fct[:, None, :2]
            results[-1]['ref_id_features'] = ref_id_features

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
    num_classes = cfg.DATASET.NUM_CLASSES

    detector = YOLOv5(cfg.MODEL.MODEL_CONFIG)
    with open(cfg.TRAIN.HYP_CONFIG) as f:
        hyp = yaml.safe_load(f)
    nl = detector.model[-1].nl
    imgsz = cfg.DATASET.MAX_SIZE
    nc = 1
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    detector.nc = nc  # attach number of classes to model
    detector.hyp = hyp  # attach hyperparameters to model
    detector.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    detector.hyp = hyp

    transformer = build_refersearch(cfg)
    matcher = build_detmatcher(cfg)
    position_embedding = build_position_encoding(cfg)

    ref_matcher = build_matchtrack_matcher(cfg)
    model = YOLOv5Match(
        detector,
        transformer,
        position_embedding,
        num_classes=num_classes,
        num_queries=cfg.TRANSFORMER.NUM_QUERIES,
        num_feature_levels=nl,
        aux_loss=cfg.LOSS.AUX_LOSS,
    )

    weight_dict = {'loss_ce': 1.0, 'loss_bbox': 1.0}
    # TODO this is a hack
    if cfg.LOSS.AUX_LOSS:
        aux_weight_dict = {}
        for i in range(cfg.TRANSFORMER.DEC_LAYERS - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    weight_dict['loss_offset'] = cfg.LOSS.OFFSET_LOSS_COEF
    weight_dict['aux_loss_offset_0'] = cfg.LOSS.OFFSET_LOSS_COEF
    weight_dict['aux_loss_offset_1'] = cfg.LOSS.OFFSET_LOSS_COEF
    weight_dict['aux_loss_offset_2'] = cfg.LOSS.OFFSET_LOSS_COEF
    weight_dict['aux_loss_offset_3'] = cfg.LOSS.OFFSET_LOSS_COEF
    weight_dict['aux_loss_offset_4'] = cfg.LOSS.OFFSET_LOSS_COEF
    weight_dict['match_loss_ids'] = cfg.LOSS.ID_LOSS_COEF
    weight_dict['loss_ids'] = cfg.LOSS.ID_LOSS_COEF

    losses = ['boxes']
    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
    criterion = SetCriterion(detector, num_classes, matcher, weight_dict, losses, 
        focal_alpha=cfg.LOSS.FOCAL_ALPHA, refer_matcher=ref_matcher)
    criterion.to(device)
    postprocessors = PostProcess()

    return model, criterion, postprocessors
