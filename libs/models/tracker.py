# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Created by Mingfei Chen (chenmingfei@sensetime.com)
# Created On: 2020-9-17
# ------------------------------------------------------------------------------
import cv2
import numpy as np
import os

import logging

from libs.utils import box_ops
from libs.utils.visualize import getDistinguishableColors


class Tracker():
    def __init__(self, track_id, cls_id, track_buffer=30, min_conf=0.05):
        self.track_buffer = track_buffer
        self.min_conf = min_conf
        self.track_id = track_id
        self.class_id = cls_id

        self.is_activated = True
        self.track_list = []
        self.conf = 0.0
        self.start_frame = -1
        self.lost_frame = 0

    def refine_coord_with_det(self, pred_box, score, matched_det):
        det_box, det_score, det_iou = matched_det
        # refined_box = (det_box + pred_box) // 2
        refined_box = det_box
        self.conf = det_score
        # self.conf = det_score * det_iou + (1 - det_iou) * score
        return refined_box

    def update(self, frame_id=None, box=None, score=None, matched_det=None):
        """
        Args:
            box: numpy.array(float), shape`(1, 4)`. Pred coord from puppet branch.
            score: float. Pred conf from puppet branch.
            frame_id: int.
            matched_det: List. [0] det_box: numpy.array(float), shape`(1, 4)`;
                         [1] det_score: float; [2] iou: float, range from 0 to 1.
        """
        if self.is_activated is False:
            self.lost_frame += 1
            return

        if matched_det is None:
            self.conf = score
        else:
            box = self.refine_coord_with_det(box, score, matched_det)

        if self.conf <= self.min_conf:
            print(f'deactivate tracker: {self.track_id}')
            self.is_activated = False
        else:
            self.is_activated = True
            self.lost_frame = 0
            if len(self.track_list) == 0:
                self.start_frame = int(frame_id)
            coord =  np.array(box, dtype=np.float).reshape(-1, 4)
            self.track_list.append({
                'frame_id': int(frame_id),
                'coord': coord,
                'score': self.conf
            })
            self.coord = coord
            self.last_frame = int(frame_id)
    
    def re_activate(self, frame_id, matched_det):
        frame_gap = int(frame_id) - self.last_frame
        det_box, det_score, det_iou = matched_det
        self.conf = det_score
        coord_gap = (det_box - self.coord) // frame_gap
        for f_id in range(self.last_frame+1, int(frame_id)+1):
            self.coord = self.coord + coord_gap
            self.track_list.append({
                'frame_id': f_id,
                'coord': self.coord,
                'score': self.conf
            })
        self.last_frame = int(frame_id)
        assert self.coord.sum() != det_box.sum()
        self.is_activated = True
        self.lost_frame = 0
        
    def __len__(self):
        return len(self.track_list)


class PuppetTracker():
    def __init__(self, track_buffer=30, det_match_thr=0.1, lost_match_thr=0.35,
                 min_track_conf=0.15):
        self.track_buffer = track_buffer
        self.det_match_thr = det_match_thr
        self.lost_match_thr = lost_match_thr
        self.min_track_conf = min_track_conf

        self.cur_id = 1 # current tracker id
        self.cur_frame_id = 1
        self.last_tracker_indexs = []

        self.active_trackers = []
        self.lost_trackers = []
        self.removed_trackers = []

    def sort_trackers(self, tracker_list):
        def get_conf(tracker):
            return tracker.conf
        tracker_list.sort(key=get_conf, reverse=True)
        return tracker_list

    def generate_overlap_mat(self, pred_boxes, det_boxes):
        overlap_mat = np.zeros((len(pred_boxes), len(det_boxes)))
        for i, pred_box in enumerate(pred_boxes):
            pred_box = pred_box.reshape(-1, 4)
            overlap_mat[i] = box_ops.compute_iou_np(pred_box.repeat(len(det_boxes), axis=0),
                det_boxes)
        return overlap_mat

    def greedy_match(self, overlap_mat): 
        idx1 = [] 
        idx2 = [] 
        new_overlap_mat = overlap_mat.copy()
        while True:
            # find all (obj_1_id, obj_2_id) pair which matches best to each other
            idx = np.unravel_index(np.argmax(new_overlap_mat, axis=None), new_overlap_mat.shape) 
            if new_overlap_mat[idx] < self.det_match_thr: 
                # no matched pair satisfies the threshold constriant
                break 
            else: 
                idx1.append(idx[0]) 
                idx2.append(idx[1]) 
                new_overlap_mat[idx[0],:] = 0 
                new_overlap_mat[:,idx[1]] = 0
        idx1 = np.array(idx1)
        idx2 = np.array(idx2)
        return idx1, idx2

    def update(self, dets, preds=None):
        if preds is None:
            now_tracker_indexs = []
            assert self.cur_frame_id == 1
            # initial track with det [det_b, det_s, clses]
            boxes, scores, cls_ids = dets
            for i, box in enumerate(boxes):
                score = scores[i]
                cls_id = cls_ids[i]
                # print(f'active {self.cur_id}')
                if score <= self.min_track_conf:
                    now_tracker_indexs.append(-1)
                    continue
                tracker = Tracker(self.cur_id, cls_id, min_conf=self.min_track_conf)
                self.cur_id += 1
                tracker.update(self.cur_frame_id, box.reshape(-1, 4), score, matched_det=None)
                self.active_trackers.append(tracker)
                now_tracker_indexs.append(i) # in sequence
            self.last_tracker_indexs = now_tracker_indexs
            self.cur_frame_id += 1
            self.lost_boxes = np.array([]).reshape(-1, 4)
            return
        det_boxes, det_scores, det_cls_ids = dets
        pred_boxes, mot_scores, pred_cls_ids = preds
        assert len(self.last_tracker_indexs) == len(pred_boxes)
        # match preds and dets (iou > det_match_thr)
        pre_boxes = np.vstack([pred_boxes, self.lost_boxes])
        overlap_mat = self.generate_overlap_mat(pre_boxes, det_boxes)
        pre_idx, det_idx = self.greedy_match(overlap_mat)
        now_tracker_indexs = {}
        det_match_valid = np.ones((len(det_boxes)), dtype=int)
        for i in range(len(pred_boxes)):
            pred_box = pred_boxes[i].reshape(-1, 4)
            mot_score = mot_scores[i]
            pred_cls_id = pred_cls_ids[i]
            tracker_idx = self.last_tracker_indexs[i]
            if tracker_idx < 0:
                # lost
                continue
            tracker_i = self.active_trackers[tracker_idx]
            match_flag = False
            if i in pre_idx:
                index = np.where(pre_idx==i)[0]
                match_det_idx = det_idx[index]
                if det_cls_ids[match_det_idx] == pred_cls_id:
                    #  match
                    match_box = det_boxes[match_det_idx].reshape(-1, 4)
                    match_score = det_scores[match_det_idx]
                    match_iou = overlap_mat[i][match_det_idx]
                    matched_det = [match_box, match_score, match_iou]
                    det_match_valid[match_det_idx] = 0
                    tracker_id = tracker_i.track_id
                    assert tracker_id not in now_tracker_indexs.keys()
                    # print(f'match actived {tracker_id}')
                    now_tracker_indexs[tracker_id] = match_det_idx
                    match_flag = True
            if match_flag is False:
                # print(f'det lost actived {tracker_i.track_id}')
                # lost
                tracker_i.is_activated = False
                matched_det = None
            # update existed tracker
            tracker_i.update(self.cur_frame_id, pred_box, mot_score, matched_det=matched_det)
                
        # link unmatched det with lost trackers
        for i, tracker_i in enumerate(self.lost_trackers):
            tracker_id = tracker_i.track_id
            track_cls_id = tracker_i.class_id
            track_coord = tracker_i.coord # `(1, 4)`
            if (i+len(pred_boxes)) in pre_idx:
                index = np.where(pre_idx==(i+len(pred_boxes)))[0]
                match_det_idx = det_idx[index]
                match_iou = overlap_mat[i+len(pred_boxes)][match_det_idx]
                if match_iou > self.lost_match_thr and det_cls_ids[
                    match_det_idx] == track_cls_id:
                    #  match
                    match_box = det_boxes[match_det_idx].reshape(-1, 4)
                    match_score = det_scores[match_det_idx]
                    matched_det = [match_box, match_score, match_iou]
                    det_match_valid[match_det_idx] = 0
                    assert tracker_id not in now_tracker_indexs.keys()
                    now_tracker_indexs[tracker_id] = match_det_idx
                    # print(f'refind {tracker_id}')
                    tracker_i.re_activate(self.cur_frame_id, matched_det)
        
        # create new track for valid unmatched det
        det_valid_idxs = np.where(det_match_valid==1)[0]
        for det_idx in det_valid_idxs:
            score = det_scores[det_idx]
            cls_id = det_cls_ids[det_idx]
            box = det_boxes[det_idx].reshape(-1, 4)
            # print(f'det active {self.cur_id}')
            if score <= self.min_track_conf:
                continue
            tracker = Tracker(self.cur_id, cls_id, min_conf=self.min_track_conf)
            tracker.update(self.cur_frame_id, box, score, matched_det=None)
            self.active_trackers.append(tracker)
            assert self.cur_id not in now_tracker_indexs.keys()
            now_tracker_indexs[self.cur_id] = det_idx
            self.cur_id += 1

        # update lost tracker
        remove_list = []
        for tracker_i in self.active_trackers:
            if tracker_i.is_activated is False:
                self.lost_trackers.append(tracker_i)
                remove_list.append(tracker_i)
        for tracker_i in remove_list:
            self.active_trackers.remove(tracker_i)

        remove_list = []
        for tracker_i in self.lost_trackers:
            if tracker_i.is_activated is True:
                self.active_trackers.append(tracker_i)
                remove_list.append(tracker_i)
        for tracker_i in remove_list:
            self.lost_trackers.remove(tracker_i)

        remove_list = []
        for tracker_i in self.lost_trackers:
            tracker_i.update()
            # lost too much frames to hold
            if tracker_i.lost_frame > self.track_buffer:
                # print(f'dead {self.cur_id}')
                self.removed_trackers.append(tracker_i)
                remove_list.append(tracker_i)
        for tracker_i in remove_list:
            self.lost_trackers.remove(tracker_i)
        self.lost_trackers = self.sort_trackers(self.lost_trackers)

        self.lost_boxes = []
        for tracker_i in self.lost_trackers:
            self.lost_boxes.append(tracker_i.coord[0])
        self.lost_boxes = np.array(self.lost_boxes).reshape(-1, 4)

        # map now_tracker_indexs
        self.last_tracker_indexs = np.zeros((len(det_boxes)), dtype=int) - 1 # 1-1 for det
        for i, tracker_i in enumerate(self.active_trackers):
            if tracker_i.track_id in now_tracker_indexs.keys():
                det_idx = now_tracker_indexs[tracker_i.track_id]
                assert self.last_tracker_indexs[det_idx] == -1
                self.last_tracker_indexs[det_idx] = i
            # else:
            #     print(f'not active {tracker_i.track_id}')
        
        # update frame_id
        self.cur_frame_id += 1

    def frame_summary(self, last_frame_id, log=False, out_root='./',
                      vis=False, min_len=10, video_root=None, size=None):
        if not os.path.exists(out_root):
            os.mkdir(out_root)
        tracker_list = []
        valid = 1
        vis_ratio = 1.0
        tracker_list.extend(self.active_trackers)
        tracker_list.extend(self.lost_trackers)
        tracker_list.extend(self.removed_trackers)
        result = {}
        log_result = {}
        track_num = 0
        for tracker_i in tracker_list:
            tracker_id = tracker_i.track_id
            log_result.setdefault(tracker_id, [])
            label = tracker_i.class_id
            if len(tracker_i.track_list) <= min_len:
                continue
            track_num += 1
            for track in tracker_i.track_list:
                x1, y1, x2, y2 = track['coord'][0]
                w = x2 - x1
                h = y2 - y1
                frame_id = track['frame_id']
                result.setdefault(frame_id, []) # [track_id, x1, y1, x2, y2, w, h, 1, ]
                result[frame_id].append([tracker_id, x1, y1, x2, y2, w, h, valid, vis_ratio])
                log_result[tracker_id].append([frame_id, tracker_id, x1, y1, w, h, valid, vis_ratio])
                
        if vis is True:
            colors = getDistinguishableColors(5*len(tracker_list))
            print(f'{track_num} tracked')
            assert video_root is not None
            for frame_id in range(1, last_frame_id+1):
                img_path = os.path.join(video_root, str(frame_id).zfill(6)+'.jpg')
                # size (h, w)
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if size is not None:
                    h, w = size
                    img = cv2.resize(img, (w, h))
                if frame_id in result.keys():
                    for track in result[frame_id]:
                        tracker_id, x1, y1, x2, y2, w, h, valid, vis_ratio = track
                        r, g, b = colors[5*(tracker_id-1)]
                        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)),
                            (int(255*r), int(255*g), int(255*b)), 1)
                out_path = os.path.join(out_root, str(frame_id).zfill(6)+'.jpg')
                cv2.imwrite(out_path, img)

        if log is True:
            out_path = os.path.join(out_root, 'result.txt')
            f = open(out_path, 'w')
            h, w = size
            for tracker_id in log_result.keys():
                for track in log_result[tracker_id]:
                    frame_id, tracker_id, x1, y1, w, h, valid, vis_ratio = track
                    result_str = f'{frame_id},{tracker_id},{x1},{y1},{w},{h},{valid},{vis_ratio}\n'
                    f.write(result_str)
            f.close()
        return 
        




    

