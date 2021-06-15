# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import lap
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn

from libs.utils.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, generalized_box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self,
                 cost_class: float = 1,
                 cost_bbox: float = 1,
                 cost_giou: float = 1,
                 cost_ids: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_ids = cost_ids
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0 or cost_ids != 0, "all costs cant be 0"

    def forward(self, outputs, targets, ref_indices=None, is_next=False, inf=1e10):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]

            # We flatten to compute the cost matrices in a batch
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
            out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

            # Also concat the target labels and boxes
            if is_next is True:
                tgt_ids = torch.cat([v["next_labels"] for v in targets])
                tgt_bbox = torch.cat([v["next_boxes"] for v in targets])
                sizes = [len(v["next_boxes"]) for v in targets]
                tgt_feat_ids = torch.cat([v["next_ids"] for v in targets])
            else:
                tgt_ids = torch.cat([v["labels"] for v in targets])
                tgt_bbox = torch.cat([v["boxes"] for v in targets])
                sizes = [len(v["boxes"]) for v in targets]
                tgt_feat_ids = torch.cat([v["ids"] for v in targets])

            # Compute the classification cost.
            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

            # Compute the giou cost betwen boxes
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox),
                                             box_cxcywh_to_xyxy(tgt_bbox))

            # Final cost matrix
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou

            C = C.view(bs, num_queries, -1)

            if is_next is True and ref_indices is not None:
                batch_ref_idx = torch.cat([torch.full_like(out_idx, i) for i, (
                    out_idx, _) in enumerate(ref_indices)])
                matched_out_idx = torch.cat([out_idx for (out_idx, _) in ref_indices])
                tgt_ref_idx = torch.cat([tgt_idx for (_, tgt_idx) in ref_indices])
                C[batch_ref_idx, matched_out_idx] = inf
                C[batch_ref_idx, ..., tgt_ref_idx] = inf
                C[batch_ref_idx, matched_out_idx, tgt_ref_idx] = 0.0

            C = C.cpu()
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class DetMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self,
                 cost_class: float = 1,
                 cost_bbox: float = 1,
                 cost_giou: float = 1,
                 cost_ids: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_ids = cost_ids
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0 or cost_ids != 0, "all costs cant be 0"

    def forward(self, outputs, targets, ref_indices=None, is_next=False, inf=1e10):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]

            # We flatten to compute the cost matrices in a batch
            out_prob = outputs["pred_logits"].flatten(0, 1)
            out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

            # Also concat the target labels and boxes
            if is_next is True:
                tgt_ids = torch.cat([v["next_labels"] for v in targets])
                tgt_bbox = torch.cat([v["next_boxes"] for v in targets])
                sizes = [len(v["next_boxes"]) for v in targets]
                tgt_feat_ids = torch.cat([v["next_ids"] for v in targets])
            else:
                tgt_ids = torch.cat([v["labels"] for v in targets])
                tgt_bbox = torch.cat([v["boxes"] for v in targets])
                sizes = [len(v["boxes"]) for v in targets]
                tgt_feat_ids = torch.cat([v["ids"] for v in targets])

            # Compute the classification cost.
            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())

            cost_class = pos_cost_class[:, tgt_ids-1] - neg_cost_class[:, tgt_ids-1]
            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

            # Compute the giou cost betwen boxes
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox),
                                             box_cxcywh_to_xyxy(tgt_bbox))

            # Final cost matrix
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou

            C = C.view(bs, num_queries, -1)

            if is_next is True and ref_indices is not None:
                batch_ref_idx = torch.cat([torch.full_like(out_idx, i) for i, (
                    out_idx, _) in enumerate(ref_indices)])
                matched_out_idx = torch.cat([out_idx for (out_idx, _) in ref_indices])
                tgt_ref_idx = torch.cat([tgt_idx for (_, tgt_idx) in ref_indices])
                C[batch_ref_idx, matched_out_idx] = inf
                C[batch_ref_idx, ..., tgt_ref_idx] = inf
                C[batch_ref_idx, matched_out_idx, tgt_ref_idx] = 0.0

            C = C.cpu()
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class ReferTrackMatcher(nn.Module):
    """This class computes an assignment between the last detected objects and the predictions of the network
       stage one.

    We do a 1-to-1 matching of the best predictions, while the others are un-matched (new detections).
    """

    def __init__(self,
                 cost_feat: float = 0.98,
                 dist_thr: float = 9.4877,
                 cost_limit: float = 0.5,
                 ):
        """Creates the matcher

        Params:
            cost_feat: This is the relative weight of the feature similarity in the matching cost
        """
        super().__init__()
        self.cost_feat = cost_feat
        self.dist_thr = dist_thr
        # self.cost_limit = cost_limit
        self.cost_limit = 1.0
        assert cost_feat != 0 or cost_loc != 0, "all costs cant be 0"
    
    def cosine_distance(self, x1, x2, eps=1e-8):
        w1 = x1.norm(p=2, dim=1, keepdim=True)
        w2 = x2.norm(p=2, dim=1, keepdim=True)
        return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)

    def forward(self, enc_outputs, references, inf=1e10):
        """ Performs the matching

        Params:
            enc_outputs: This is a dict that contains at least these entries:
                 "id_features": Tensor of dim [batch_size, num_queries, emb_dim] with the appearance embeddings
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            references: This is a list of references (len(references) = batch_size), where each reference is a dict containing:
                 "ref_features": Tensor of dim [num_refer_boxes, emb_dim] (where num_refer_boxes is the number of detected boxes 
                                 in the previous frame) containing the reference appearance embeddings, detached.
                 "ref_boxes": Tensor of dim [num_refer_boxes, 4] containing the reference box coordinates predicted 
                              in the previous frame
                 "idx_map": Tensor of dim [num_refer_boxes, num_targets], map the refer index to the target index

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected reference (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_refer_boxes)
        """
        with torch.no_grad():
            bs, num_queries = enc_outputs["id_features"].shape[:2]
            # We flatten to compute the cost matrices in a batch
            id_features = enc_outputs["id_features"].flatten(0, 1)
            pred_boxes = enc_outputs["pred_boxes"].flatten(0, 1) # [batch_size * num_queries, 4]

            # Also concat the target labels and boxes
            ref_features = torch.cat([v["ref_features"] for v in references])
            ref_boxes = torch.cat([v["ref_boxes"] for v in references])
            input_size = torch.cat([v["input_size"] for v in references])
            scale = torch.cat([input_size, input_size], dim=1)
            ref_boxes = ref_boxes * scale
            pred_boxes = pred_boxes * scale
            ref_boxes[..., 2] /= ref_boxes[..., 3]
            pred_boxes[..., 2] /= pred_boxes[..., 3]
            
            # ref_boxes = box_cxcywh_to_xyxy(ref_boxes)
            # pred_boxes = box_cxcywh_to_xyxy(pred_boxes)
            assert len(ref_features) == len(ref_boxes)
            # Compute the feature similarity
            cost_feature = self.cosine_distance(id_features, ref_features)
            # Compute the distance ** 2 between boxes 
            cost_distance = torch.cdist(pred_boxes, ref_boxes, p=2)
            # TODO use dist_thr or not
            # cost_feature[cost_distance>self.dist_thr] = inf

            # Final cost matrix
            C = self.cost_feat * cost_feature + (1 - self.cost_feat) * cost_distance

            C = C.view(bs, num_queries, -1).cpu()

            sizes = [len(v["ref_boxes"]) for v in references]
            ref_indices = [lap.lapjv(c[i].data.cpu().numpy(), extend_cost=True,
                cost_limit=self.cost_limit)[1] for i, c in enumerate(C.split(sizes, -1))]
            ori_indices = [(torch.as_tensor(torch.range(0, len(idx)-1)[idx>=0], dtype=torch.int64), torch.as_tensor(idx[
                idx>=0], dtype=torch.int64).long()) for i, idx in enumerate(ref_indices)]
            ref_indices = [(torch.as_tensor(torch.range(0, len(idx)-1)[idx>=0], dtype=torch.int64), r['idx_map'][torch.as_tensor(idx[
                idx>=0], dtype=torch.int64)].long()) for i, (r, idx) in enumerate(zip(references, ref_indices))]
            return ori_indices, ref_indices


class MatchTrackMatcher(nn.Module):
    """This class computes an assignment between the last detected objects and the predictions of the network
       stage one.

    We do a 1-to-1 matching of the best predictions, while the others are un-matched (new detections).
    """
    def __init__(self,
                 cost_feat: float = 0.98,
                 det_thr: float = 0.3,
                 cost_limit: float = 0.5,
                 ):
        """Creates the matcher

        Params:
            cost_feat: This is the relative weight of the feature similarity in the matching cost
        """
        super().__init__()
        self.cost_feat = cost_feat
        # self.cost_limit = cost_limit
        self.det_thr = det_thr
        self.cost_limit = 0.8
        # self.cost_limit = -0.1
        # self.det_thr = 1.1
        assert cost_feat != 0 or cost_loc != 0, "all costs cant be 0"
    
    def cosine_distance(self, x1, x2, eps=1e-8):
        w1 = x1.norm(p=2, dim=1, keepdim=True)
        w2 = x2.norm(p=2, dim=1, keepdim=True)
        return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)

    def forward(self, ref_outputs, outputs, references, inf=1e10):
        """ Performs the matching
        """
        with torch.no_grad():
            bs, num_queries = outputs["id_features"].shape[:2]
            # We flatten to compute the cost matrices in a batch
            id_features = ref_outputs["ref_id_features"].flatten(0, 1)
            
            pred_boxes = ref_outputs["ref_coords"].flatten(0, 1) # [batch_size * num_queries, 4]

            src_id_features = outputs["id_features"].flatten(0, 1)
            src_boxes = outputs["pred_boxes"].flatten(0, 1)
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()[..., 1]

            # Also concat the target labels and boxes
            ref_features = torch.cat([v["ref_features"] for v in references])
            ref_boxes = torch.cat([v["ref_boxes"] for v in references])
            input_size = torch.cat([v["input_size"] for v in references])
            scale = torch.cat([input_size, input_size], dim=1).to(ref_boxes.device)
            # pred_boxes = torch.cat([pred_boxes, ref_boxes[..., 2:]], dim=1)
            pred_boxes = torch.cat([pred_boxes-ref_boxes[..., 2:4], pred_boxes+ref_boxes[
                ..., 4:]], dim=-1) 
            ref_boxes = torch.cat([ref_boxes[..., :2]-ref_boxes[..., 2:4], ref_boxes[
                ..., :2]+ref_boxes[..., 4:]], dim=-1)
            pred_boxes = box_xyxy_to_cxcywh(pred_boxes)
            ref_boxes = box_xyxy_to_cxcywh(ref_boxes)

            ref_boxes = ref_boxes * scale
            pred_boxes = pred_boxes * scale
            src_boxes = src_boxes * scale
            pred_boxes[..., 2] /= pred_boxes[..., 3]
            src_boxes[..., 2] /= src_boxes[..., 3]
            
            assert len(ref_features) == len(ref_boxes)
            # Compute the feature similarity
            ref_distance = torch.diag(self.cosine_distance(id_features, ref_features))
            # cost_feature = (self.cosine_distance(src_id_features, id_features) + ref_distance) / 2.0
            cost_feature = torch.sqrt(ref_distance * self.cosine_distance(src_id_features, id_features))
            # cost_feature = self.cosine_distance(src_id_features, id_features)

            # cost_feature = torch.diag(self.cosine_distance(src_id_features, ref_features))
            # Compute the distance ** 2 between boxes 
            cost_distance = torch.cdist(src_boxes, pred_boxes, p=2)
            
            # Final cost matrix
            C = self.cost_feat * cost_feature + (1 - self.cost_feat) * cost_distance
            # C = (1 - self.cost_feat) * cost_distance
            C[out_prob<self.det_thr] = inf
            C = C.view(bs, num_queries, -1).cpu()

            sizes = [len(v["ref_boxes"]) for v in references]
            ref_indices = [lap.lapjv(c[i].data.cpu().numpy(), extend_cost=True,
                cost_limit=self.cost_limit)[1] for i, c in enumerate(C.split(sizes, -1))]
            ori_indices = [(torch.as_tensor(torch.range(0, len(idx)-1)[idx>=0], dtype=torch.int64), torch.as_tensor(idx[
                idx>=0], dtype=torch.int64).long()) for i, idx in enumerate(ref_indices)]
            ref_indices = [(torch.as_tensor(torch.range(0, len(idx)-1)[idx>=0], dtype=torch.int64), r['idx_map'][torch.as_tensor(idx[
                idx>=0], dtype=torch.int64)].long()) for i, (r, idx) in enumerate(zip(references, ref_indices))]
            return ori_indices, ref_indices


def build_matcher(cfg):
    return HungarianMatcher(cost_class=cfg.MATCHER.COST_CLASS,
        cost_bbox=cfg.MATCHER.COST_BBOX, cost_giou=cfg.MATCHER.COST_GIOU)

def build_detmatcher(cfg):
    return DetMatcher(cost_class=cfg.MATCHER.COST_CLASS,
        cost_bbox=cfg.MATCHER.COST_BBOX, cost_giou=cfg.MATCHER.COST_GIOU)

def build_refer_matcher(cfg):
    return ReferTrackMatcher()

def build_matchtrack_matcher(cfg):
    return MatchTrackMatcher()