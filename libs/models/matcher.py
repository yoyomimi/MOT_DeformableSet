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
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from libs.utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


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
                 cost_ids: float = 0.5):
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

    def forward(self, outputs, targets):
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
            tgt_ids = torch.cat([v["labels"] for v in targets])
            tgt_bbox = torch.cat([v["boxes"] for v in targets])

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
            
            # feat ids
            # out_feat_logits = outputs['id_embeds'].flatten(0, 1).contiguous().sigmoid()
            # tgt_feat_ids = torch.cat([v["ids"] for v in targets])
            # valids_ids = torch.where(tgt_feat_ids>-1)[0]
            # if len(valid_ids) == len(tgt_bbox):
            #     neg_cost_ids = (1 - alpha) * (out_feat_logits ** gamma) * (-(1 - out_feat_logits + 1e-8).log())
            #     pos_cost_ids = alpha * ((1 - out_feat_logits) ** gamma) * (-(out_feat_logits + 1e-8).log())
            #     cost_ids = neg_cost_ids[:, tgt_feat_ids] - pos_cost_ids[:, tgt_feat_ids]
            #     C = C + self.cost_ids * cost_ids

            C = C.view(bs, num_queries, -1).cpu()

            sizes = [len(v["boxes"]) for v in targets]
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class ReferTrackMatcher(nn.Module):
    """This class computes an assignment between the last detected objects and the predictions of the network
       stage one.

    We do a 1-to-1 matching of the best predictions, while the others are un-matched (new detections).
    """

    def __init__(self,
                 cost_emb: float = 0.98,
                 cost_loc: float = 0.02,
                 ):
        """Creates the matcher

        Params:
            cost_emb: This is the relative weight of the appearance similarity in the matching cost
            cost_loc: This is the relative weight of the locations overlap in the matching cost
        """
        super().__init__()
        self.cost_emb = cost_emb
        self.cost_loc = cost_loc
        assert cost_emb != 0 or cost_loc != 0, "all costs cant be 0"

    def forward(self, enc_outputs, references):
        """ Performs the matching

        Params:
            enc_outputs: This is a dict that contains at least these entries:
                 "id_features": Tensor of dim [batch_size, num_queries, emb_dim] with the appearance embeddings
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            references: This is a list of references (len(references) = batch_size), where each reference is a dict containing:
                 "ref_embeds": Tensor of dim [num_refer_boxes, emb_dim] (where num_refer_boxes is the number of detected boxes 
                               in the previous frame) containing the reference appearance embeddings
                 "ref_centers": Tensor of dim [num_refer_boxes, 2] containing the reference box centers predicted 
                               in the previous frame

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
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
            out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

            # Also concat the target labels and boxes
            tgt_ids = torch.cat([v["labels"] for v in targets])
            tgt_bbox = torch.cat([v["boxes"] for v in targets])

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

            C = C.view(bs, num_queries, -1).cpu()

            sizes = [len(v["boxes"]) for v in targets]
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(cfg):
    return HungarianMatcher(cost_class=cfg.MATCHER.COST_CLASS,
        cost_bbox=cfg.MATCHER.COST_BBOX, cost_giou=cfg.MATCHER.COST_GIOU)
