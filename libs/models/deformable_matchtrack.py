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

from libs.utils import box_ops
from libs.utils.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from libs.models.backbone import build_backbone
from libs.models.matcher import build_matcher, build_matchtrack_matcher
from libs.models.deformable_transformer import build_deforamble_transformer
import copy


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def sigmoid_focal_loss(inputs, targets, num_boxes=None, reduction='mean', alpha: float = 0.25, gamma: float = 2,):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if reduction == 'mean':
        return loss.mean(1).sum() / num_boxes
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss.sum()


class DeformableMatchTrack(nn.Module):
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 emb_dim=128, dataset_nids=14687, aux_loss=True, with_box_refine=False, two_stage=False, num_match_decoder_layers=3):
        # ch: 355567
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.det_class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        ###### modified ##########
        self.id_embed =  MLP(hidden_dim, hidden_dim, emb_dim, 3)
        self.nID = dataset_nids
        self.emb_dim = emb_dim
        self.light_id_head = nn.Linear(self.emb_dim, self.nID)
        nn.init.normal_(self.light_id_head.weight, std=0.01)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.light_id_head.bias, bias_value)
        # self.IDLoss = nn.CrossEntropyLoss(ignore_index=-1)
        self.emb_scale = math.sqrt(2) * math.log(self.nID - 1)
        # self.offset_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        ##########################
        self.num_feature_levels = num_feature_levels
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim*2)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
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
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.det_class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            self.det_class_embed = _get_clones(self.det_class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            # nn.init.constant_(self.offset_embed.layers[-1].bias.data[2:], -2.0)
            self.det_class_embed = nn.ModuleList([self.det_class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.det_class_embed = self.det_class_embed
            ###### modified ##########
            # self.transformer.decoder.id_embed =  self.id_embed
            # for offset_embed in self.offset_embed:
            #     nn.init.constant_(offset_embed.layers[-1].bias.data[2:], 0.0)
            ##########################
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)
        # match
        self.ref_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        nn.init.constant_(self.ref_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.ref_embed.layers[-1].bias.data, 0)
        # nn.init.constant_(self.ref_embed.layers[-1].bias.data[2:], -2.0)
        self.ref_embed = _get_clones(self.ref_embed, num_match_decoder_layers)
        nn.init.constant_(self.ref_embed[0].layers[-1].bias.data[2:], -2.0)
        self.ref_id_embed = copy.deepcopy(self.id_embed)
        

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
        features, pos = self.backbone(samples)

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
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, ref_indices, memory, match_hs, match_inter_references = self.transformer(
            srcs, masks, pos, query_embeds, references=references)
        self.out_memory = memory

        if references is not None and match_hs is not None:
            # print(match_hs.shape)
            # ref_boxes = torch.cat([v["ref_boxes"] for v in references])
            # match_reference = ref_boxes[..., :2]
            # ref_id_embed = self.ref_id_embed(match_hs)
            # ref_id_feature = ref_id_embed.clone()
            # ref_id_embed = self.emb_scale * F.normalize(ref_id_embed)
            # ref_id_embed = self.light_id_head(ref_id_embed.contiguous())
            # tmp = self.ref_embed(match_hs)
            # tmp += ref_boxes
            # ref_coords = tmp.sigmoid()
            ref_coords = []
            ref_boxes = torch.cat([v["ref_boxes"] for v in references])
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
                    ref_id_embed = self.light_id_head(ref_id_embed.contiguous())
                tmp = self.ref_embed[lvl](match_hs[lvl])
                tmp += reference
                ref_coords.append(tmp.sigmoid())

        outputs_classes = []
        outputs_coords = []
        outputs_id_embeds = []
        outputs_id_features = []
        outputs_next_centers = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.det_class_embed[lvl](hs[lvl])
            ###### modified ##########
            if lvl == hs.shape[0] - 1:
                outputs_id_embed = self.id_embed(hs[lvl])
                outputs_id_features.append(outputs_id_embed.clone())
                outputs_id_embed = self.emb_scale * F.normalize(outputs_id_embed)
                outputs_id_embed = self.light_id_head(outputs_id_embed.contiguous())
                # outputs_id_embed = None
                outputs_id_embeds.append(outputs_id_embed)
            # next_center_tmp = self.offset_embed[lvl](hs[lvl])
            ##########################
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
                # next_center_tmp += reference[..., :2]
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
                # next_center_tmp += reference
            outputs_coord = tmp.sigmoid()
            # outputs_next_center = next_center_tmp.sigmoid()

            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            # outputs_next_centers.append(outputs_next_center)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        outputs_id_embed = outputs_id_embeds[-1]
        outputs_id_features = outputs_id_features[-1]
        self.out_id_features = outputs_id_features
        self.out_pred_boxes = outputs_coord[-1]
        # outputs_next_center = torch.stack(outputs_next_centers)

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1],
               'id_embeds': outputs_id_embed, 'id_features': outputs_id_features
            #    'next_cenetrs': outputs_next_center[-1]
               }
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
        
        if references is not None and match_hs is not None:
            out['ref_outputs'] = {'ref_coords': ref_coords[-1],
               'ref_id_embeds': ref_id_embed, 'ref_id_features': ref_id_feature}
            out['aux_ref_outputs'] = [{'ref_coords': a} for a in ref_coords[:-1]]
        return out

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
    def __init__(self, num_classes, matcher, weight_dict, losses, refer_matcher=None, focal_alpha=0.25):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.refer_matcher = refer_matcher
        
    def loss_labels(self, outputs, targets, indices, num_boxes, log=True, is_next=False):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        if is_next is True:
            target_classes_o = torch.cat([t["next_labels"][J] for t, (_, J) in zip(targets, indices)])
        else:
            target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes, is_next=False):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        if is_next is True:
            tgt_lengths = torch.as_tensor([len(v["next_labels"]) for v in targets], device=device)
        else:
            tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes, log=True, is_next=False):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        losses = {}
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        ###### modified ##########
        if is_next is True:
            target_boxes = torch.cat([t['next_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            target_ids = torch.cat([t['next_ids'][i] for t, (_, i) in zip(targets, indices)], dim=0).long()
        else:
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            target_ids = torch.cat([t['ids'][i] for t, (_, i) in zip(targets, indices)], dim=0).long()
        if 'id_embeds' in outputs:
            valids_ids = torch.where(target_ids>-1)[0]
            target_ids = target_ids[valids_ids]
            outputs_src_id_logits = outputs['id_embeds'][idx][valids_ids].contiguous()
            # loss_ids = self.IDLoss(outputs_src_id_logits, target_ids)
            target_ids_onehot = torch.zeros([outputs_src_id_logits.shape[0], outputs_src_id_logits.shape[1] + 1],
                                             dtype=outputs_src_id_logits.dtype, layout=outputs_src_id_logits.layout, device=outputs_src_id_logits.device)
            target_ids_onehot.scatter_(1, target_ids.unsqueeze(-1), 1)

            target_ids_onehot = target_ids_onehot[:,:-1]
            loss_ids = sigmoid_focal_loss(outputs_src_id_logits, target_ids_onehot, reduction='sum', alpha=self.focal_alpha, gamma=2) / max(1, outputs_src_id_logits.shape[0])
            losses = {'loss_ids': loss_ids}
            if log:
                # TODO this should probably be a separate loss, not hacked in this one here
                losses['id_class_error'] = 100 - accuracy(outputs_src_id_logits, target_ids)[0]
        ##########################
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_match(self, outputs, targets, indices, num_boxes, log=True, is_next=False):
        assert is_next is True
        losses = {}
        if 'ref_outputs' not in outputs:
            device = outputs['pred_boxes'].device
            losses['loss_offset'] = torch.Tensor([0.]).mean().to(device)
            losses['aux_loss_offset_0'] = torch.Tensor([0.]).mean().to(device)
            losses['aux_loss_offset_1'] = torch.Tensor([0.]).mean().to(device)
            losses['match_loss_ids'] = torch.Tensor([0.]).mean().to(device)
            return losses

        target_next_centers = torch.cat([t['gt_ref_boxes'][..., :2] for t in targets], dim=0)
        src_next_centers = outputs['ref_outputs']['ref_coords'].flatten(0, 1)
        loss_offset = F.l1_loss(src_next_centers, target_next_centers, reduction='none')
        losses['loss_offset'] = loss_offset.sum() / num_boxes
        aux_outputs = outputs['aux_ref_outputs']
        for i, aux_output in enumerate(aux_outputs):
            src_next_centers = aux_output['ref_coords'].flatten(0, 1)
            loss_offset = F.l1_loss(src_next_centers, target_next_centers, reduction='none')
            losses[f'aux_loss_offset_{i}'] = loss_offset.sum() / num_boxes
        # id
        target_ids = torch.cat([t['gt_ref_ids'] for t in targets], dim=0).long()
        valids_ids = torch.where(target_ids>-1)[0]
        target_ids = target_ids[valids_ids]
        outputs_src_id_logits = outputs['ref_outputs']['ref_id_embeds'].flatten(0, 1)
        outputs_src_id_logits = outputs_src_id_logits[valids_ids].contiguous()
        # loss_ids = self.IDLoss(outputs_src_id_logits, target_ids)
        target_ids_onehot = torch.zeros([outputs_src_id_logits.shape[0], outputs_src_id_logits.shape[1] + 1],
                                        dtype=outputs_src_id_logits.dtype, layout=outputs_src_id_logits.layout, device=outputs_src_id_logits.device)
        target_ids_onehot.scatter_(1, target_ids.unsqueeze(-1), 1)

        target_ids_onehot = target_ids_onehot[:,:-1]
        loss_ids = sigmoid_focal_loss(outputs_src_id_logits, target_ids_onehot, reduction='sum', alpha=self.focal_alpha, gamma=2) / max(1, outputs_src_id_logits.shape[0])
        losses['match_loss_ids'] = loss_ids
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
        if references is not None and 'ref_outputs' in outputs:
            ref_outputs = outputs['ref_outputs']
            ori_indices, ref_indices = self.refer_matcher(ref_outputs, outputs_without_aux, references)
        else:
            ref_indices = None
        # Retrieve the matching between the outputs of the last layer and the targets
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
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets, ref_indices, is_next)
                for loss in self.losses:
                    kwargs = {}
                    kwargs['is_next'] = is_next
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                if is_next is True:
                    bt['next_labels'] = torch.zeros_like(bt['next_labels'])
                else:
                    bt['labels'] = torch.zeros_like(bt['labels'])
            indices = self.matcher(enc_outputs, bin_targets, None, is_next)
            for loss in self.losses:
                kwargs = {}
                kwargs['is_next'] = is_next
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)
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
            losses.update(self.loss_match(outputs, references, None, ref_num_boxes, is_next=is_next))

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self):
        super().__init__()
        from libs.models.matcher import MatchTrackMatcher
        self.refer_matcher = MatchTrackMatcher(det_thr=0.35, cost_limit=1.0)

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
        prob = out_logits.sigmoid()
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

    backbone = build_backbone(cfg)

    transformer = build_deforamble_transformer(cfg)
    matcher = build_matcher(cfg)
    ref_matcher = build_matchtrack_matcher(cfg)
    model = DeformableMatchTrack(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=cfg.TRANSFORMER.NUM_QUERIES,
        num_feature_levels=cfg.TRANSFORMER.NUM_FEATURE_LEVELS,
        aux_loss=cfg.LOSS.AUX_LOSS,
        with_box_refine=cfg.DEFORMABLE.WITH_BOX_REFINE,
        two_stage=cfg.DEFORMABLE.TWO_STAGE,
    )
    weight_dict = {'loss_ce': cfg.LOSS.CLS_LOSS_COEF, 'loss_bbox': cfg.LOSS.BBOX_LOSS_COEF}
    weight_dict['loss_giou'] = cfg.LOSS.GIOU_LOSS_COEF
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
    weight_dict['match_loss_ids'] = cfg.LOSS.ID_LOSS_COEF
    weight_dict['loss_ids'] = cfg.LOSS.ID_LOSS_COEF

    losses = ['labels', 'boxes', 'cardinality']
    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
    criterion = SetCriterion(num_classes, matcher, weight_dict, losses, 
        focal_alpha=cfg.LOSS.FOCAL_ALPHA, refer_matcher=ref_matcher)
    criterion.to(device)
    postprocessors = PostProcess()

    return model, criterion, postprocessors
