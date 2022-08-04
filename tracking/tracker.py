## TransCenter: Transformers with Dense Representations for Multiple-Object Tracking
## Copyright Inria
## Year 2022
## Contact : yihong.xu@inria.fr
##
## TransCenter is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.

## TransCenter is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program, TransCenter.  If not, see <http://www.gnu.org/licenses/> and the LICENSE file.
##
##
## TransCenter has code derived from
## (1) 2020 fundamentalvision.(Apache License 2.0: https://github.com/fundamentalvision/Deformable-DETR)
## (2) 2020 Philipp Bergmann, Tim Meinhardt. (GNU General Public License v3.0 Licence: https://github.com/phil-bergmann/tracking_wo_bnw)
## (3) 2020 Facebook. (Apache License Version 2.0: https://github.com/facebookresearch/detr/)
## (4) 2020 Xingyi Zhou.(MIT License: https://github.com/xingyizhou/CenterTrack)
## (5) 2021 Wenhai Wang. (Apache License Version 2.0: https://github.com/whai362/PVT/blob/v2/LICENSE)
##
## TransCenter uses packages from
## (1) 2019 Charles Shang. (BSD 3-Clause Licence: https://github.com/CharlesShang/DCNv2)
## (2) 2020 fundamentalvision.(Apache License 2.0: https://github.com/fundamentalvision/Deformable-DETR)
import copy
import sys
import os

from scipy.misc import derivative

# dirty insert path #
cur_path = os.path.realpath(__file__)
cur_dir = "/".join(cur_path.split('/')[:-2])
sys.path.insert(0, cur_dir)

import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

import cv2
from PIL import Image, ImageDraw, ImageFont
from util.tracker_util import bbox_overlaps

from torchvision.ops.boxes import clip_boxes_to_image, nms
import lap
from post_processing.decode import generic_decode
from util import box_ops
import matplotlib.pyplot as plt
import cv2
# from BOTSORT.tracker.gmc import GMC
from StrongSORT.deep_sort.kalman_filter import KalmanFilter
import pandas as pd
from fast_reid.fast_reid_interface import FastReIDInterface
from scipy.spatial.distance import cdist

def transparent_cmap(cmap, N=255):
        "Copy colormap and set alpha values"
        mycmap = cmap
        mycmap._init()
        mycmap._lut[:,-1] = np.linspace(0, 0.8, N+4)
        return mycmap

chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}

class Tracker:
    """The main tracking file, here is where magic happens."""
    # only track pedestrian
    cl = 1

    def __init__(self, obj_detect, reid_network, flownet, tracker_cfg, postprocessor=None, main_args=None, seq_name=None):

        self.obj_detect = obj_detect
        self.public_detections = tracker_cfg['public_detections']
        self.inactive_patience = tracker_cfg['inactive_patience']
        self.do_reid = tracker_cfg['do_reid']
        self.max_features_num = tracker_cfg['max_features_num']
        self.reid_sim_threshold = tracker_cfg['reid_sim_threshold']
        self.reid_iou_threshold = tracker_cfg['reid_iou_threshold']
        self.sim_threshold = tracker_cfg['sim_threshold']
        self.iou_threshold = tracker_cfg['iou_threshold']
        self.do_align = tracker_cfg['do_align']
        self.motion_model_cfg = tracker_cfg['motion_model']
        self.postprocessor = postprocessor
        self.main_args = main_args

        self.inactive_tracks = []
        self.track_num = 0
        self.im_index = 0
        self.results = {}
        self.img_features = None
        self.encoder_pos_encoding = None
        self.transforms = transforms.ToTensor()
        self.last_image = None
        self.pre_sample = None
        self.sample = None
        self.pre_img_features = None
        self.pre_encoder_pos_encoding = None
        self.flow = None
        self.det_thresh = main_args.track_thresh + 0.1
        
        self.tracks = []
        #for plotting kalman error:
        self.covx_minus_list = []
        self.covx_list = []
        self.covy_minus_list = []
        self.covy_list = []
        self.meanx_minus_list = []
        self.meanx_list = []
        self.meany_minus_list = []
        self.meany_list = []
        self.posx_list = []
        self.posy_list = []
        self.kalman_outputs = {}
        # output inactive tracks:
        # self.show_inactive_age = 3 # show inactive tracks for 3 frames TODO: make it configurable
        self.encoder = FastReIDInterface(tracker_cfg['fast_reid_config'], tracker_cfg['fast_reid_weights'], tracker_cfg['device'])


    def reset(self, hard=True, seq_name=None):
        self.tracks = []
        self.inactive_tracks = []
        self.last_image = None
        self.pre_sample = None
        self.obj_detect.pre_memory = None
        self.sample = None
        self.pre_img_features = None
        self.pre_encoder_pos_encoding = None
        self.flow = None
        self.obj_detect.masks_flatten = None
        # self.gmc = GMC(method='file', verbose=[seq_name, False]) #Amit: added for camera correction

        if hard:
            self.track_num = 0
            self.results = {}
            self.im_index = 0

    def linear_assignment(self, cost_matrix, thresh):
        if cost_matrix.size == 0:
            return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
        matches, unmatched_a, unmatched_b = [], [], []

        cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)

        for ix, mx in enumerate(x):
            if mx >= 0:
                matches.append([ix, mx])
        unmatched_a = np.where(x < 0)[0]
        unmatched_b = np.where(y < 0)[0]
        matches = np.asarray(matches)

        # matches = [[match_row_idx, match_column_idx]...], it gives you all the matches (assignments)
        # unmatched_a gives all the unmatched row indexes
        # unmatched_b gives all the unmatched column indexes
        return matches, unmatched_a, unmatched_b

    def tracks_to_inactive(self, tracks):
        self.tracks = [t for t in self.tracks if t not in tracks]
        for t in tracks:
            t.pos = t.last_pos[-1]
        self.inactive_tracks += tracks

    def add(self, new_det_pos, new_det_scores, new_det_features):
        """Initializes new Track objects and saves them."""
        num_new = new_det_pos.size(0)
        for i in range(num_new):
            self.tracks.append(Track(
                new_det_pos[i].view(1, -1),
                new_det_scores[i],
                self.track_num + i,
                new_det_features[i].view(1, -1),
                self.inactive_patience,
                self.max_features_num,
                self.motion_model_cfg['n_steps'] if self.motion_model_cfg['n_steps'] > 0 else 1
            ))
        self.track_num += num_new 

    def tracks_dets_matching_tracking(self, blob, raw_dets, raw_scores, pre2cur_cts, pos=None, reid_cts=None, reid_feats=None, batch = None, detection_list = None):
        """
        raw_dets and raw_scores are clean (only ped class and filtered by a threshold
        """
        if pos is None:
            pos = self.get_pos().clone()

        # iou matching #

        assert pos.nelement() > 0 and pos.shape[0] == pre2cur_cts.shape[0]
        # todo we can directly output warped_pos for faster inference #
        if raw_dets.nelement() > 0:
            assert raw_dets.shape[0] == raw_scores.shape[0]
            pos_w = pos[:, [2]] - pos[:, [0]]
            pos_h = pos[:, [3]] - pos[:, [1]]

            warped_pos = torch.cat([pre2cur_cts[:, [0]] - 0.5 * pos_w,
                                    pre2cur_cts[:, [1]] - 0.5 * pos_h,
                                    pre2cur_cts[:, [0]] + 0.5 * pos_w,
                                    pre2cur_cts[:, [1]] + 0.5 * pos_h], dim=1) 

            # warp positions to current frame using gmc #
            # warp = self.gmc.apply(batch['img'])
            # warped_pos = Track.multi_gmc(self.tracks, warp) 
            # warped_pos = Track.tlwh_pos_to_tlbr(warped_pos) #transcenter works with tlbr pos but the mean is in tlwh pos
            # index low-score dets #
            inds_low = raw_scores > 0.1  # was 0.1 TODO: change back after testing 
            inds_high = raw_scores < self.main_args.track_thresh
            inds_second = torch.logical_and(inds_low, inds_high)
            dets_second = raw_dets[inds_second]
            inds_second_list = torch.where(inds_second)[0].tolist() #added for detection list, can't slice list with mask tensor
            detection_list_second = [detection_list[i] for i in inds_second_list]
            scores_second = raw_scores[inds_second]
            reid_cts_second = reid_cts[inds_second]

            # index high-score dets #
            remain_inds = raw_scores > self.main_args.track_thresh
            dets = raw_dets[remain_inds]
            remain_inds_list = torch.where(remain_inds)[0].tolist() #added for detection list, can't slice list with mask tensor
            detection_list_first = [detection_list[i] for i in remain_inds_list]
            scores_keep = raw_scores[remain_inds]
            reid_cts_keep = reid_cts[remain_inds]

            # extract embedding features #
            features_keep_first = self.encoder.inference(blob['img'][0].permute(1, 2, 0), detection_list_first) #image shape 3,1080,1920 to 1080,1920,3
            features_keep_second = self.encoder.inference(blob['img'][0].permute(1, 2, 0), detection_list_second)
            for i,d in enumerate(detection_list_first):
                d.feature = features_keep_first[i]
            for i,d in enumerate(detection_list_second):
                d.feature = features_keep_second[i]

            # Step 1: first assignment #
            if len(dets) > 0:
                assert dets.shape[0] == scores_keep.shape[0]
                # matching with gIOU
                iou_dist = box_ops.generalized_box_iou(warped_pos, dets)

                # todo fuse with dets scores here.
                if self.main_args.fuse_scores:
                    iou_dist *= scores_keep[None, :]

                iou_dist = 1 - iou_dist
                iou_dist_mask = (iou_dist > self.iou_threshold)

                # todo recover inactive tracks here ?
                if self.main_args.iou_recover: 
                    det_feats = F.grid_sample(reid_feats, reid_cts_keep.unsqueeze(0).unsqueeze(0),
                                                mode='bilinear', padding_mode='zeros', align_corners=False)[:, :, 0, :] #was after assignment, moved to try appearance matching
                    matches, u_track, u_detection = self.linear_assignment(iou_dist.cpu().numpy(),
                                                                        thresh=self.main_args.match_thresh)

                else: # try fuse matching (appearance and iou) 
                    emb_dists = self.embedding_distance(self.tracks, detection_list_first)
                    raw_emb_dists = emb_dists.copy()
                    emb_dists[emb_dists > self.sim_threshold] = 1.0
                    emb_dists[iou_dist_mask] = 1.0
                    dist = np.minimum(iou_dist, emb_dists) #picking the min of iou and appearance

                    matches, u_track, u_detection = self.linear_assignment(dist, thresh=self.main_args.match_thresh)
                # dist_mat, pos = [], []
                # for t in self.tracks:
                    # dist_mat.append(torch.cat([t.test_features(feat.view(1, -1))
                                            #    for feat in det_feats], dim=1))
                    # pos.append(t.pos)
                # if len(dist_mat) > 1:
                    # dist_mat = torch.cat(dist_mat, 0)
                    # pos = torch.cat(pos, 0)
                # else:
                    # dist_mat = dist_mat[0]
                    # pos = pos[0]
                # dist_mat_np = dist_mat.cpu().numpy()
                # lambda_ = 0.5
                
                # for row, tracks in enumerate(self.tracks):

                    # dist_mat_np[row] = lambda_ * dist_mat_np[row] + (1 - lambda_) * iou_dist.cpu().numpy()[row]

                    # assigned by appearance & iou fuse
                # matches, u_track, u_detection = self.linear_assignment(dist_mat_np,
                                                                        #    thresh=self.match_thresh)
                


                if matches.shape[0] > 0:
                    
                    # update track dets, scores #
                    self.update(detections=detection_list_first, matches=matches, unmatched_tracks=u_track, unmatched_detections=u_detection) #from strongSORT TODO: return to for loop if doesn't work here
                    
                    for idx_track, idx_det in zip(matches[:, 0], matches[:, 1]):
                        t = self.tracks[idx_track]
                        t.pos = dets[[idx_det]]
                        # t.mean = Track.tlbr_pos_to_tlwh(t.pos) #added for gmc
                        t.add_features(det_feats[:, :, idx_det])
                        t.score = scores_keep[[idx_det]]


                pos_birth = dets[u_detection, :] # dets are the high score dets
                if len(u_detection)>0:
                    # u_det_list_idx = torch.where(u_detection)[0].tolist() #added for detection list, can't slice list with mask tensor
                    detection_list_reid = [detection_list_first[i] for i in u_detection] #update detection list for reid
                else:
                    detection_list_reid = []
                scores_birth = scores_keep[u_detection]
                dets_features_birth = det_feats[0, :, u_detection].transpose(0,1)


            else:
                # no detection, kill all
                u_track = list(range(len(self.tracks)))
                pos_birth = torch.zeros(size=(0, 4), device=pos.device, dtype=pos.dtype)
                scores_birth = torch.zeros(size=(0,), device=pos.device).long()
                dets_features_birth = torch.zeros(size=(0, 64), device=pos.device, dtype=pos.dtype)


            # Step 2: second assignment #
            # get remained tracks
            if len(u_track) > 0:

                if len(dets_second) > 0:
                    remained_tracks_pos = warped_pos[u_track]
                    track_indices = copy.deepcopy(u_track)
                    # print("track_indices: ", track_indices)
                    if self.main_args.iou_recover:
                        # matching with gIOU
                        iou_dist = 1 - box_ops.generalized_box_iou(remained_tracks_pos, dets_second)  # [0, 2]
                        iou_dist_mask = (iou_dist > self.iou_threshold)

                        matches, u_track_second, u_detection_second = self.linear_assignment(iou_dist.cpu().numpy(),thresh=0.4)  # stricter with low-score dets
                    else: # try fuse matching (appearance and iou)
                        emb_dists = self.embedding_distance(self.tracks[u_track], detection_list_second)
                        raw_emb_dists = emb_dists.copy()
                        emb_dists[emb_dists > self.sim_threshold] = 1.0
                        emb_dists[iou_dist_mask] = 1.0
                        dist = np.minimum(iou_dist, emb_dists)
                        matches, u_track_second, u_detection_second = self.linear_assignment(dist, thresh=self.main_args.match_thresh) #TODO: change to smaller thresh like in iou

                    # update u_track here
                    u_track = [track_indices[t_idx] for t_idx in u_track_second]

                    if matches.shape[0] > 0:
                        second_det_feats = F.grid_sample(reid_feats,
                                                         reid_cts_second[matches[:, 1]].unsqueeze(0).unsqueeze(0),
                                                         mode='bilinear', padding_mode='zeros', align_corners=False)[:,
                                           :, 0, :]
                        # update track dets, scores #
                        self.update(detections=detection_list_second, matches=matches, unmatched_tracks=u_track_second, unmatched_detections=u_detection_second, is_2nd_assignment=True,track_indices=track_indices) #from strongSORT
                        
                        for cc, (idx_match, idx_det) in enumerate(zip(matches[:, 0], matches[:, 1])):
                            idx_track = track_indices[idx_match]
                            # print("low score match:", idx_track)
                            t = self.tracks[idx_track]
                            t.pos = dets_second[[idx_det]]
                            # t.mean = Track.tlbr_pos_to_tlwh(t.pos) #added for gmc
                            gather_feat_t = second_det_feats[:, :, cc]
                            t.add_features(gather_feat_t)
                            t.score = scores_second[[idx_det]]
        else:
            # no detection, kill all
            u_track = list(range(len(self.tracks)))
            pos_birth = torch.zeros(size=(0, 4), device=pos.device, dtype=pos.dtype)
            scores_birth = torch.zeros(size=(0,), device=pos.device).long()
            dets_features_birth = torch.zeros(size=(0, 64), device=pos.device, dtype=pos.dtype)


        # put inactive tracks
        self.new_tracks = []
        for i, t in enumerate(self.tracks):
            if i in u_track:  # inactive
                # t.pos = t.last_pos[-1]
                t.pos = t.xyah_to_tlbr(t.mean[:4])
                assert (t.pos[:,2:]>t.pos[:,:2]).all(), "wrong pos: {}".format(t.pos)
                # t.mean = Track.tlbr_pos_to_tlwh(t.pos) #added for gmc
                self.inactive_tracks += [t]
            else: # keep
                self.new_tracks.append(t)
        self.tracks = self.new_tracks

        return [pos_birth, scores_birth, dets_features_birth, detection_list_reid]

    def detect_tracking_duel_vit(self, batch):

        [ratio, padw, padh] = batch['trans']
        mypos = self.get_pos().clone()

        no_pre_cts = False

        if mypos.shape[0] > 0:
            # make pre_cts #
            # bboxes to centers
            hm_h, hm_w = self.sample.tensors.shape[2], self.sample.tensors.shape[3]
            bboxes = mypos.clone()
            # bboxes

            bboxes[:, 0] += bboxes[:, 2]
            bboxes[:, 1] += bboxes[:, 3]
            pre_cts = bboxes[:, 0:2] / 2.0

            # to input image plane
            pre_cts *= ratio
            pre_cts[:, 0] += padw
            pre_cts[:, 1] += padh
            pre_cts[:, 0] = torch.clamp(pre_cts[:, 0], 0, hm_w - 1)
            pre_cts[:, 1] = torch.clamp(pre_cts[:, 1], 0, hm_h - 1)

            # to output image plane
            pre_cts /= self.main_args.down_ratio
        else:
            pre_cts = torch.zeros(size=(2, 2), device=mypos.device, dtype=mypos.dtype)

            no_pre_cts = True
            print("No Pre Cts!")


        outputs = self.obj_detect(samples=self.sample, pre_samples=self.pre_sample,
                                  pre_cts=pre_cts.clone().unsqueeze(0))
        # # post processing #
        output = {k: v[-1] for k, v in outputs.items() if k != 'boxes'}

        # 'hm' is not _sigmoid!
        output['hm'] = torch.clamp(output['hm'].sigmoid(), min=1e-4, max=1 - 1e-4)

        decoded = generic_decode(output, K=self.main_args.K, opt=self.main_args)

        out_scores = decoded['scores'][0]
        labels_out = decoded['clses'][0].int() + 1
        #print hm - selected frames#
        frame_num = int(batch['frame_name'][-8:-4])
        if frame_num>636 and frame_num<636: #only printing heatmaps for debugging in selected frames
            image = Image.fromarray((255*batch['img'])[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8), 'RGB')
            tensor_hm = output['hm'][0,0,:]
            w, h = image.size
            y, x = np.mgrid[0:h, 0:w]
            hm_np = tensor_hm.cpu().numpy()
            hm_r = cv2.resize(hm_np, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
            mycmap = transparent_cmap(plt.cm.YlOrRd) #was Reds
            fig, ax1 = plt.subplots(1, 1)
            ax1.imshow(image)
            cb = ax1.contourf(x,y,hm_r, cmap=mycmap)
            plt.colorbar(cb)
            frame_nums = batch['frame_name'][-8:-4]
            name="hm_for_17-11_frame_" +frame_nums + ".png"
            plt.savefig(name)

        # # reid features #
        # torch.Size([1, 64, 152, 272])

        if no_pre_cts:
            pre2cur_cts = torch.zeros_like(mypos)[..., :2]
        else:
            pre2cur_cts = self.main_args.down_ratio * (decoded['tracking'][0] + pre_cts)
            pre2cur_cts[:, 0] -= padw
            pre2cur_cts[:, 1] -= padh
            pre2cur_cts /= ratio

        # extract reid features #
        boxes = decoded['bboxes'][0].clone()
        reid_cts = torch.stack([0.5*(boxes[:, 0]+boxes[:, 2]), 0.5*(boxes[:, 1]+boxes[:, 3])], dim=1)
        reid_cts[:, 0] /= outputs['reid'][0].shape[3]
        reid_cts[:, 1] /= outputs['reid'][0].shape[2]
        reid_cts = torch.clamp(reid_cts, min=0.0, max=1.0)
        reid_cts = (2.0 * reid_cts - 1.0)
        # print(reid_cts.shape)

        out_boxes = decoded['bboxes'][0] * self.main_args.down_ratio
        out_boxes[:, 0::2] -= padw
        out_boxes[:, 1::2] -= padh
        out_boxes /= ratio

        # filtered by scores #
        filtered_idx = labels_out == 1 # todo warning, wrong for multiple classes
        out_scores = out_scores[filtered_idx]
        out_boxes = out_boxes[filtered_idx]

        reid_cts = reid_cts[filtered_idx]
        if self.main_args.clip:  # for mot20 clip box
            _, _, orig_h, orig_w = batch['img'].shape
            out_boxes[:, 0::2] = torch.clamp(out_boxes[:, 0::2], 0, orig_w-1)
            out_boxes[:, 1::2] = torch.clamp(out_boxes[:, 1::2], 0, orig_h-1)

        #create detection objects
        detection_list = []
        for i in range(out_boxes.shape[0]):
            detection_list.append(Detection(out_boxes[i,:], out_scores[i], feature=None))
        
        # post processing #

        return out_boxes, out_scores, pre2cur_cts, mypos, reid_cts, outputs['reid'][0], detection_list

    def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
        if cost_matrix.size == 0:
            return cost_matrix
        gating_dim = 2 if only_position else 4
        gating_threshold = chi2inv95[gating_dim]
        measurements = np.asarray([det.to_xyah() for det in detections])
        for row, track in enumerate(tracks):
            gating_distance = kf.gating_distance(
                track.mean, track.covariance, measurements, only_position, metric='maha')
            cost_matrix[row, gating_distance > gating_threshold] = np.inf
            cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
        return cost_matrix

    def embedding_distance(tracks, detections, metric='cosine'):
        """
        :param tracks: list[STrack]
        :param detections: list[BaseTrack]
        :param metric:
        :return: cost_matrix np.ndarray
        """

        cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
        if cost_matrix.size == 0:
            return cost_matrix
        det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float)
        track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float)

        cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # / 2.0  # Nomalized features
        return cost_matrix

    def get_pos(self):
        """Get the positions of all active tracks."""
        if len(self.tracks) == 1:
            pos = self.tracks[0].pos
        elif len(self.tracks) > 1:
            pos = torch.cat([t.pos for t in self.tracks], dim=0)
        else:
            pos = torch.zeros(size=(0, 4), device=self.sample.tensors.device).float()

        return pos

    def get_features(self):
        """Get the features of all active tracks."""
        if len(self.tracks) == 1:
            features = self.tracks[0].features
        elif len(self.tracks) > 1:
            features = torch.cat([t.features for t in self.tracks], 0)
        else:
            features = torch.zeros(size=(0,), device=self.sample.tensors.device).float()
        return features

    def get_inactive_features(self):
        """Get the features of all inactive tracks."""
        if len(self.inactive_tracks) == 1:
            features = self.inactive_tracks[0].features
        elif len(self.inactive_tracks) > 1:
            features = torch.cat([t.features for t in self.inactive_tracks], 0)
        else:
            features = torch.zeros(0).cuda()
        return features

    def reid(self, blob, new_det_pos, new_det_scores, new_det_features, detection_list=None):
        """Tries to ReID inactive tracks with provided detections."""

        if self.do_reid:

            if len(self.inactive_tracks) > 0:
                # calculate appearance distances
                dist_mat, pos = [], []
                for t in self.inactive_tracks:
                    dist_mat.append(torch.cat([t.test_features(feat.view(1, -1))
                                               for feat in new_det_features], dim=1))
                    pos.append(t.pos)
                if len(dist_mat) > 1:
                    dist_mat = torch.cat(dist_mat, 0)
                    pos = torch.cat(pos, 0)
                else:
                    dist_mat = dist_mat[0]
                    pos = pos[0]
                dist_mat_np = dist_mat.cpu().numpy()
                
                # # calculate IoU distances
                if self.main_args.iou_recover:
                    iou_dist = 1 - box_ops.generalized_box_iou(pos, new_det_pos)

                    matches, u_track, u_detection = self.linear_assignment(iou_dist.cpu().numpy(),
                                                                           thresh=self.main_args.match_thresh)
                else:

                    #assigned by fusing iou and appearance
                    # lambda_ = 0.2
                    # iou_dist = 1 - box_ops.generalized_box_iou(pos, new_det_pos)
                    # for row, tracks in enumerate(self.inactive_tracks):

                        # dist_mat_np[row] = lambda_ * dist_mat_np[row] + (1 - lambda_) * iou_dist.cpu().numpy()[row]

                    # assigned by appearance
                    matches, u_track, u_detection = self.linear_assignment(dist_mat_np,
                                                                           thresh=self.reid_sim_threshold)

                assigned = []
                remove_inactive = []
                if matches.shape[0] > 0:
                    #update kalman filter
                    self.update(detections=detection_list, matches=matches, unmatched_tracks=u_track, unmatched_detections=u_detection, is_reid=True) #from strongSORT

                    for r, c in zip(matches[:, 0], matches[:, 1]):
                        # inactive tracks reactivation #
                        # print('dist:', dist_mat_np[r, c])
                        # print('sim threshold:', self.reid_sim_threshold)
                        if dist_mat[r, c] <= self.reid_sim_threshold or not self.main_args.iou_recover: #TODO: used to depend on iou, remove if doesn't work
                            t = self.inactive_tracks[r]
                            self.tracks.append(t)
                            t.count_inactive = 0
                            t.pos = new_det_pos[c].view(1, -1)
                            t.reset_last_pos()
                            t.add_features(new_det_features[c].view(1, -1))
                            assigned.append(c)
                            remove_inactive.append(t)
                    

                for t in remove_inactive:
                    self.inactive_tracks.remove(t)

                keep = [i for i in range(new_det_pos.size(0)) if i not in assigned]
                if len(keep) > 0:
                    new_det_pos = new_det_pos[keep]
                    new_det_scores = new_det_scores[keep]
                    new_det_features = new_det_features[keep]
                    
                else:
                    new_det_pos = torch.zeros(size=(0, 4), device=self.sample.tensors.device).float()
                    new_det_scores = torch.zeros(size=(0,), device=self.sample.tensors.device).long()
                    new_det_features = torch.zeros(size=(0, 128), device=self.sample.tensors.device).float()

        return new_det_pos, new_det_scores, new_det_features

    def add_features(self, new_features):
        """Adds new appearance features to active tracks."""
        for t, f in zip(self.tracks, new_features):
            t.add_features(f.view(1, -1))
    
    def predict(self): # from strongSORT
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        pred_positions = []
        for track in self.tracks:
            assert track.mean[3] > 0, "Error: track height is negative before prediction!"
            pred_pos = track.predict()
            assert track.mean[3] > 0, "Error: track height is negative!"
            pred_positions.append(pred_pos)
        pred_positions = torch.stack([torch.from_numpy(item).float() for item in pred_positions]).to(device='cuda:0') #original pre2cur_cts is on cuda:0
        for track in self.inactive_tracks:
            if track.mean[3] < 0:
                self.inactive_tracks.remove(track) #remove tracks with negative height
                
            elif track.count_inactive < track._max_age:
                assert track.mean[3] > 0, "Error: track height is negative before prediction! track id: {} track mean: {}".format(track.id, track.mean)
                inactive_pred_pos = track.predict() #predict inactive track locations as well
        return pred_positions

    def update(self, detections, matches, unmatched_tracks, unmatched_detections, is_reid=False, is_2nd_assignment=False,track_indices=None): #from strongSORT 
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        # matches, unmatched_tracks, unmatched_detections = \
            # self._match(detections)

        # Update track set.
        for track_idx, detection_idx in matches:
            if is_reid: #in reid mode, the updated track is in inactive_tracks
                self.inactive_tracks[track_idx].update(detections[detection_idx])
                self.inactive_tracks[track_idx].mean = self.inactive_tracks[track_idx].mean.reshape(8) 
            elif is_2nd_assignment:
                actual_track_idx = track_indices[track_idx]
                self.tracks[actual_track_idx].update(detections[detection_idx])
                self.tracks[actual_track_idx].mean = self.tracks[actual_track_idx].mean.reshape(8)
            else:
                self.tracks[track_idx].update(detections[detection_idx])
                self.tracks[track_idx].mean = self.tracks[track_idx].mean.reshape(8) #TODO: remove this line if doesn't work
                assert self.tracks[track_idx].mean[3] > 1, "Error: track height is very small!"
        # for track_idx in unmatched_tracks:
            # self.tracks[track_idx].mark_missed()
        # for detection_idx in unmatched_detections:
            # self._initiate_track(detections[detection_idx]) #TODO: add this if we use tentative tracks
        # self.tracks = [t for t in self.tracks] #if not t.is_deleted()]

        # Update distance metric.
        # active_targets = [t.id for t in self.tracks] #if t.is_confirmed()
        # features, targets = [], []
        # for track in self.tracks:
            # if not track.is_confirmed():
                # continue
            # features += track.features
            # targets += [track.id for _ in track.features]
            # if not opt.EMA:
            # track.features = []
        # self.metric.partial_fit(
            # np.asarray(features), np.asarray(targets), active_targets)

    @torch.no_grad()
    def step_reidV3_pre_tracking_vit(self, blob):
        """This function should be called every timestep to perform tracking with a blob
        containing the image information.
        """
        # Nested tensor #
        self.sample = blob['samples']

        if self.pre_sample is None:
            self.pre_sample = self.sample

        # # plot #
        # if int(blob['frame_name'][:-4])>62:
            # img_pil = Image.fromarray((255*blob['img'])[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8), 'RGB')
            # img_draw = ImageDraw.Draw(img_pil)
        # #
            # if self.last_image is not None:
                # pre_img_pil = Image.fromarray((255 * self.last_image)[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8), 'RGB')
                # pre_img_draw = ImageDraw.Draw(pre_img_pil)
            # else:
                # pre_img_pil = img_pil
                # pre_img_draw = img_draw
        # # # # plot #

        ###########################
        # Look for new detections #
        ###########################
        # detect

        det_pos, det_scores, pre2cur_cts, mypos, reid_cts, reid_features, detection_list = self.detect_tracking_duel_vit(blob) #added detection_list

        ##################
        # Predict tracks #
        ##################
        plot_kf = False
        if len(self.tracks):
            predicted_pre2cur_cts = self.predict() #from strongSORT, added output of predicted_pre2cur_cts
            #############################
            # save Kalman filter result # TODO: remove this section after testing
            #############################
            # chosen_idx = 4
            # t_idx = None
            # for i,t in enumerate(self.tracks):
            #     if t.id ==chosen_idx:
            #         t_idx = i
            # if t_idx is None: # chosen idx not found in tracks, search in inactive tracks
            #     for i,t in enumerate(self.inactive_tracks):
            #         if t.id ==chosen_idx:
            #             t_idx = i
            # assert t_idx is not None, "chosen_idx not found, track id: {} probably gone".format(chosen_idx)
            # plot_kf = True
            # track_cov_x = self.tracks[t_idx].covariance[0,0]
            # track_mean_x = self.tracks[t_idx].mean[0]
            # track_posx = self.tracks[t_idx].pos[:,0]
            # track_cov_y = self.tracks[t_idx].covariance[1,1]
            # track_mean_y = self.tracks[t_idx].mean[1]
            # track_posy = self.tracks[t_idx].pos[:,1]

            # self.covx_minus_list.append(track_cov_x)
            # self.posx_list.append(track_posx)
            # self.covy_minus_list.append(track_cov_y)
            # self.posy_list.append(track_posy)
            # self.meanx_minus_list.append(track_mean_x)
            # self.meany_minus_list.append(track_mean_y)


            #############################

            [det_pos, det_scores, dets_features_birth,detection_list_reid] = self.tracks_dets_matching_tracking(
                blob=blob, raw_dets=det_pos, raw_scores=det_scores, pre2cur_cts=predicted_pre2cur_cts, pos=mypos, reid_cts=reid_cts,
                reid_feats=reid_features, batch = blob, detection_list=detection_list) #changed pre2cur_cts to predicted_pre2cur_cts from strongSORT kalman
        else:
            dets_features_birth = F.grid_sample(reid_features, reid_cts.unsqueeze(0).unsqueeze(0), mode='bilinear', padding_mode='zeros', align_corners=False)[:, :, 0, :].transpose(1, 2)[0]
            detection_list_reid = detection_list
        #####################
        # Create new tracks #
        #####################
        # filter birth candidates by scores
        valid_dets_idx = det_scores >= self.det_thresh  # track_thresh+0.1
        det_pos = det_pos[valid_dets_idx]
        valid_dets_list_idx = torch.where(valid_dets_idx)[0].tolist()
        detection_list_reid = [detection_list_reid[j] for j in valid_dets_list_idx]
        det_scores = det_scores[valid_dets_idx]
        dets_features_birth = dets_features_birth[valid_dets_idx]

        if self.public_detections:
            # no pub dets => in def detect = no private detection
            # case 1: No pub det, private dets OR
            # case 2: No pub det, no private dets

            if blob['dets'].shape[0] == 0:
                det_pos = torch.zeros(size=(0, 4), device=self.sample.tensors.device).float()
                det_scores = torch.zeros(size=(0,), device=self.sample.tensors.device).long()
                dets_features_birth = torch.zeros(size=(0, 64), device=self.sample.tensors.device).float()

            # case 3: Pub det, private dets
            elif det_pos.shape[0] > 0:
                _, _, orig_h, orig_w = blob['img'].shape
                pub_dets = blob['dets']
                # using centers
                M = pub_dets.shape[0]

                # # iou of shape [#private, #public]#
                if self.main_args.clip:  # for mot20 clip box
                    iou = bbox_overlaps(det_pos, clip_boxes_to_image(pub_dets, (orig_h-1, orig_w-1)))
                else:
                    iou = bbox_overlaps(det_pos, pub_dets)
                # having overlap ?
                valid_private_det_idx = []
                for j in range(M):
                    # print("pub dets")
                    i = iou[:, j].argmax()
                    if iou[i, j] > 0:
                        iou[i, :] = -1
                        valid_private_det_idx.append(i.item())
                det_pos = det_pos[valid_private_det_idx]
                valid_dets_list_idx = torch.where(valid_private_det_idx)[0].tolist()
                detection_list_reid = [detection_list_reid[j] for j in valid_dets_list_idx]
                det_scores = det_scores[valid_private_det_idx]
                dets_features_birth = dets_features_birth[valid_private_det_idx]

            # case 4: No pub det, no private dets
            else:
                det_pos = torch.zeros(size=(0, 4), device=self.sample.tensors.device).float()
                det_scores = torch.zeros(size=(0,), device=self.sample.tensors.device).long()
                dets_features_birth = torch.zeros(size=(0, 64), device=self.sample.tensors.device).float()

        else:
            pass

        if det_pos.nelement() > 0:

            assert det_pos.shape[0] == dets_features_birth.shape[0] == det_scores.shape[0]
            # try to re-identify tracks
            
            det_pos, det_scores, dets_features_birth = self.reid(blob, det_pos, det_scores, dets_features_birth, detection_list=detection_list_reid)

            assert det_pos.shape[0] == dets_features_birth.shape[0] == det_scores.shape[0]

            # add new
            if det_pos.nelement() > 0:
                self.add(det_pos, det_scores, dets_features_birth)
        #############################
        # save Kalman filter result # TODO: remove this section after testing
        #############################
        if plot_kf:
            t_idx = None
            for i,t in enumerate(self.tracks):
                if t.id ==chosen_idx:
                    t_idx = i
            if t_idx is None:
                for i,t in enumerate(self.inactive_tracks):
                    if t.id ==chosen_idx:
                        t_idx = i
            assert t_idx is not None, "chosen_idx not found, track id: {} probably gone".format(chosen_idx)
            track_cov_x = self.tracks[t_idx].covariance[0,0]
            track_mean_x = self.tracks[t_idx].mean[0]
            track_cov_y = self.tracks[t_idx].covariance[1,1]
            track_mean_y = self.tracks[t_idx].mean[1]

            self.covx_list.append(track_cov_x)
            self.meanx_list.append(track_mean_x)
            self.covy_list.append(track_cov_y)
            self.meany_list.append(track_mean_y)
            
            self.kalman_outputs = {'covx_minus': self.covx_minus_list, 'meanx_minus': self.meanx_minus_list,
                                    'covy_minus': self.covy_minus_list, 'meany_minus': self.meany_minus_list,
                                    'covx': self.covx_list, 'meanx': self.meanx_list, 'covy': self.covy_list, 'meany': self.meany_list}
        #############################    
        


        ####################
        # Generate Results #
        ####################
        for t in self.tracks:
            if t.id not in self.results.keys():
                self.results[t.id] = {}
            self.results[t.id][self.im_index] = np.concatenate([t.pos[0].cpu().numpy(), np.array([t.score.cpu()])])

        #     # # # plot tracks
        #     bb = t.pos[0].clone()
        #     bb = np.array(bb.cpu(), dtype=int)
        #     img_draw.rectangle([(bb[0], bb[1]), (bb[2], bb[3])], fill=None, outline="red")
        #     img_draw.text((bb[0], (bb[1] + bb[3]) // 2), f" {t.id}", fill=(255, 0, 0, 255),
        #                   font=ImageFont.truetype("./UbuntuMono-BI.ttf", 25))
        #
        #     img_draw.text((bb[0], bb[3]), f" {t.score.item():.02f}", fill=(0, 255, 0, 255),
        #                   font=ImageFont.truetype("./UbuntuMono-BI.ttf", 20))
        #     # #
        # # # save plot #
        # os.makedirs(
        #     "./check_plot/" + blob['video_name'],
        #     exist_ok=True)
        # img_pil.save(
        #     "./check_plot/" + blob['video_name'] + f'/' + blob['frame_name'])
        #
        # # # # save plot #
        new_inactive_tracks = []
        for t in self.inactive_tracks:
            t.count_inactive += 1
            if t.has_positive_area() and t.count_inactive <= self.inactive_patience:
                new_inactive_tracks.append(t)

        self.inactive_tracks = new_inactive_tracks

        # Add young inactive tracks to results #
        # for t in self.inactive_tracks:
            # if t.count_inactive <= self.show_inactive_age: #only inactives younger than X frames are shown
                # if t.id not in self.results.keys():
                    # self.results[t.id] = {}
                # self.results[t.id][self.im_index] = np.concatenate([t.pos[0].cpu().numpy(), np.array([t.score.cpu()])])

        self.im_index += 1

    def get_results(self):
        return self.results


class Track(object):
    """This class contains all necessary for every individual track."""

    def __init__(self, pos, score, track_id, features, inactive_patience, max_features_num, mm_steps, n_init=3,max_age=30, feat=None):
        self.id = track_id
        self.pos = pos
        self.score = score
        self.features = deque([features])
        self.ims = deque([])
        self.count_inactive = 0
        self.inactive_patience = inactive_patience
        self.max_features_num = max_features_num
        self.last_pos = deque([pos.clone()], maxlen=mm_steps + 1)
        self.last_v = torch.Tensor([])
        self.gt_id = None

        # self.mean = self.tlbr_pos_to_tlwh(pos)
        # epsilon = 1e-8
        # self.covariance = np.ones((4, 4)) * epsilon
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        #fast RE-ID:
        self.smooth_feat = None
        self.curr_feat = None
        self.FR_features = deque([], maxlen=max_features_num)
        if feat is not None:
            self.update_features(feat)
        self.alpha = 0.9
        self._n_init = n_init
        self._max_age = max_age

        self.kf = KalmanFilter()
        self.mean, self.covariance = self.kf.initiate(self.tlbr_to_xyah(pos).reshape(4))

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.FR_features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)  
        

    def has_positive_area(self):
        return self.pos[0, 2] > self.pos[0, 0] and self.pos[0, 3] > self.pos[0, 1]

    def add_features(self, features):
        """Adds new appearance features to the object."""
        self.features.append(features)
        if len(self.features) > self.max_features_num:
            self.features.popleft()

    def test_features(self, test_features):
        """Compares test_features to features of this Track object"""
        if len(self.features) > 1:
            features = torch.cat(list(self.features), dim=0)
        else:
            features = self.features[0]
        features = features.mean(0, keepdim=True)
        dist = F.pairwise_distance(features, test_features, keepdim=True)
        return dist

    def reset_last_pos(self):
        self.last_pos.clear()
        self.last_pos.append(self.pos.clone())

    @staticmethod
    def tlbr_pos_to_tlwh(tlbr):
        tlbr = tlbr.cpu().numpy()
        tlwh = np.zeros_like(tlbr)
        tlwh[:, 0] = tlbr[:, 0]
        tlwh[:, 1] = tlbr[:, 1]
        tlwh[:, 2] = tlbr[:, 2] - tlbr[:, 0]
        tlwh[:, 3] = tlbr[:, 3] - tlbr[:, 1]

        # tlwh = tlwh.tolist()
        # tlwh = [torch.from_numpy(item).float() for item in tlwh]
        return tlwh
    
    @staticmethod
    def tlwh_pos_to_tlbr(tlwh):
        tlwh = np.asarray(tlwh)
        tlbr = np.zeros_like(tlwh)
        tlbr[:, 0] = tlwh[:, 0]
        tlbr[:, 1] = tlwh[:, 1]
        tlbr[:, 2] = tlwh[:, 0] + tlwh[:, 2]
        tlbr[:, 3] = tlwh[:, 1] + tlwh[:, 3]
        # tlbr = [torch.from_numpy(item).float() for item in tlbr]
        tlbr = torch.from_numpy(tlbr).float().squeeze().cuda()
        return tlbr
    
    @staticmethod
    def tlbr_to_xyah(tlbr):
        tlbr = tlbr.cpu().numpy()
        xyah = np.zeros_like(tlbr)
        xyah[:, 0] = (tlbr[:, 0] + tlbr[:, 2]) / 2 # x center
        xyah[:, 1] = (tlbr[:, 1] + tlbr[:, 3]) / 2 # y center
        xyah[:, 2] = (tlbr[:, 2] - tlbr[:, 0])/(tlbr[:, 3] - tlbr[:, 1]) # aspect ratio (width/height)
        xyah[:, 3] = tlbr[:, 3]-tlbr[:, 1] # height
        # xyah = xyah.tolist()
        # xyah = [torch.from_numpy(item).float() for item in xyah]
        return xyah.T #kalman filter expects a column vector
    
    @staticmethod
    def xyah_to_tlbr(xyah):
        xyah = np.asarray(xyah)
        tlbr = np.zeros_like(xyah)
        width = xyah[2] * xyah[3]
        tlbr[0] = xyah[0] - width / 2
        tlbr[1] = xyah[1] - xyah[3] / 2
        tlbr[2] = xyah[0] + width / 2
        tlbr[3] = xyah[1] + xyah[3] / 2
        tlbr = tlbr.reshape(1, 4)
        tlbr = torch.from_numpy(tlbr).float().cuda()
        return tlbr
    
    # @staticmethod
    # def multi_gmc(stracks, H=np.eye(2, 3)):
    #     if len(stracks) > 0:
    #         multi_mean = np.asarray([st.mean.copy() for st in stracks])
    #         multi_covariance = np.asarray([st.covariance for st in stracks])

    #         R = H[:2, :2] #size (2,2)
    #         R8x8 = np.kron(np.eye(4, dtype=float), R) #size (8, 8) 
    #         R4x4 = np.kron(np.eye(2, dtype=float), R) #size (4, 4)
    #         t = H[:2, 2, np.newaxis] #it raised a valueError without the newaxis
    #         means = []
    #         for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
    #             # mean = R8x8.dot(mean)
    #             mean = mean.T
    #             mean = R4x4.dot(mean) # changed since mean is now a 4x1 vector 
    #             mean[:2] += t
    #             # cov = R8x8.dot(cov).dot(R8x8.transpose())
    #             cov = R4x4.dot(cov).dot(R4x4.transpose())

    #             stracks[i].mean = mean.T
    #             stracks[i].covariance = cov
    #             means.append(mean)


    #         return means
    
    def predict(self):
        " Kalman prediction step"
        self.mean, self.covariance = self.kf.predict(self.mean.reshape(8), self.covariance)
        self.age += 1
        self.time_since_update += 1
        pred_pos = self.mean[:2] #added to force using the predicted pos in matching
        return pred_pos

    
    def update(self, detection):
        """Kalman filter measurement update
       
        detection : The associated detection object.

        """
        self.mean, self.covariance = self.kf.update(self.mean.reshape(8), self.covariance, detection.to_xyah().reshape(1,4), detection.confidence)

        if detection.feature:
            self.update_features(detection.feature)
        # feature = detection.feature / np.linalg.norm(detection.feature) #TODO: declare detection dict with feature key
        # if opt.EMA:
        #     smooth_feat = opt.EMA_alpha * self.features[-1] + (1 - opt.EMA_alpha) * feature
        #     smooth_feat /= np.linalg.norm(smooth_feat)
        #     self.features = [smooth_feat]
        # else:
        # self.features.append(feature)

        self.hits += 1
        self.time_since_update = 0
        # if self.state == TrackState.Tentative and self.hits >= self._n_init:
        #     self.state = TrackState.Confirmed

class Detection(object): #from strongSORT
    """
    This class represents a bounding box detection in a single image.

    Parameters
    ----------
    tlwh : array_like
        Bounding box in format `(x, y, w, h)`.
    confidence : float
        Detector confidence score.
    feature : array_like
        A feature vector that describes the object contained in this image.

    Attributes
    ----------
    tlwh : ndarray
        Bounding box in format `(top left x, top left y, width, height)`.
    confidence : ndarray
        Detector confidence score.
    feature : ndarray | NoneType
        A feature vector that describes the object contained in this image.

    """

    def __init__(self, tlbr, confidence, feature):
        self.tlbr = tlbr.cpu().numpy() #np.asarray(tlwh, dtype=np.float)
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32)

    # def to_tlbr(self):
        # """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        # `(top left, bottom right)`.
        # """
        # ret = self.tlwh.copy()
        # ret[2:] += ret[:2]
        # return ret

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        tlbr = self.tlbr.copy()
        xyah = tlbr
        xyah[0] = (tlbr[0] + tlbr[2]) / 2
        xyah[1] = (tlbr[1] + tlbr[3]) / 2
        xyah[3] = (tlbr[3]-tlbr[1]) #height
        xyah[2] = (tlbr[2]-tlbr[0]) / xyah[3] # width / height
        return xyah