import numpy as np
from numba import jit
from collections import deque
import itertools
import os
import os.path as osp
import time
import torch
import cv2
import torch.nn.functional as F

from libs.models.tracker.basetrack import BaseTrack, TrackState
from libs.models.tracker import matching


class STrack(BaseTrack):
    def __init__(self, tlwh, motion, score, temp_feat, logger=None, buffer_size=30):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float) # cur loc
        self.motion = np.asarray(motion, dtype=np.float)
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        self.update_features(temp_feat)
        self.features = deque([], maxlen=buffer_size)
        self.alpha = 0.9

        self.logger = logger

    def update_features(self, feat):
        # feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        # self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def activate(self, frame_id):
        """Start a new tracklet"""
        self.track_id = self.next_id()

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        #self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
    
    @staticmethod
    def multi_predict(stracks, warp_matrix=None):
        if len(stracks) > 0:
            for i in range(len(stracks)):
                stracks[i]._tlwh = stracks[i].tlwh
                if warp_matrix is not None:
                    warp_matrix = warp_matrix.reshape(-1, 9)
                    x0, y0 = stracks[i]._tlwh[0], stracks[i]._tlwh[1]
                    X = warp_matrix[..., 0] * x0 + warp_matrix[..., 1] * y0 + warp_matrix[..., 2]
                    Y = warp_matrix[..., 3] * x0 + warp_matrix[..., 4] * y0 + warp_matrix[..., 5]
                    Z = warp_matrix[..., 6] * x0 + warp_matrix[..., 7] * y0 + warp_matrix[..., 8]
                    stracks[i]._tlwh[..., 0] = X/Z
                    stracks[i]._tlwh[..., 1] = Y/Z

    def re_activate(self, new_track, frame_id, new_id=False):
        self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()

    def update(self, new_track, frame_id, update_feature=True):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        self._tlwh = new_track._tlwh
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        if update_feature:
            self.update_features(new_track.curr_feat)

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        ret = self._tlwh.copy()
        if self.motion is None:
            return ret
        ret[:2] += self.motion
        return ret
    
    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self._tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class SimpleTracker(object):
    def __init__(self, track_buffer=30, frame_rate=30, ltrb=True):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        self.unconfirmed = []
        self.strack_pool = []

        self.frame_id = 0
        self.ltrb = ltrb
        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        self.max_time_lost = self.buffer_size

    def update(self, dets, id_feature, motion, track_idx, logger=None, warp_matrix=None):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), m, tlbrs[4], f, 30) for
                          (tlbrs, m, f) in zip(dets[:, :5], motion, id_feature)]
        else:
            detections = []

        STrack.multi_predict(self.strack_pool, warp_matrix) # pred cur location
        ''' Step 1: Referred association, tracked_indices'''
        # TODO
        u_detection = []
        u_track = []
        track_valid =  np.ones(len(self.strack_pool))
        for idet, itracked in enumerate(track_idx):
            if itracked >= len(self.strack_pool):
                import pdb; pdb.set_trace()
            if itracked < 0:
                u_detection.append(idet)
                continue
            track = self.strack_pool[itracked]
            track_valid[itracked] = 0
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        
        ''' Step 2: First association, with embedding'''
        if len(track_idx) > 0:
            detections = [detections[i] for i in u_detection]
            u_track = np.where(track_valid == 1)[0]
            r_tracked_stracks = [self.strack_pool[i] for i in u_track if self.strack_pool[i].state == TrackState.Tracked]
        else:
            r_tracked_stracks = []


        ''' Step 3: Second association, with IOU'''
        dists = matching.iou_distance(r_tracked_stracks, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
                
        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(self.unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            self.unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(self.unconfirmed[itracked])
        for it in u_unconfirmed:
            track = self.unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            track.activate(self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        ''' Add newly detected tracklets to tracked_stracks'''
        self.unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                self.unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
        self.strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # TODO return additional references
        references = None
        id_features = torch.cat([torch.as_tensor(track.smooth_feat).reshape(1, -1) for track in self.strack_pool])
        ref_boxes = torch.cat([torch.as_tensor(track.tlwh.reshape(1, -1)) for track in self.strack_pool])
        ref_boxes[..., :2] += 0.5 * ref_boxes[..., 2:]
        idx_map = torch.range(0, len(ref_boxes)-1)
        references = [
            dict(ref_features=id_features.float(), ref_boxes=ref_boxes.float(), idx_map=idx_map)
        ]
        if logger is not None:
            logger.debug('===========Frame {}=========='.format(self.frame_id))
            logger.debug('Activated: {}'.format([track.track_id for track in activated_starcks]))
            logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
            logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
            logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))

        return output_stracks, references


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res

def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())

def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb