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


class Tracker:
    """The main tracking file, here is where magic happens."""
    # only track pedestrian
    cl = 1

    def __init__(self, obj_detect, reid_network, flownet, tracker_cfg, postprocessor=None, main_args=None):

        self.obj_detect = obj_detect
        self.public_detections = tracker_cfg['public_detections']
        self.inactive_patience = tracker_cfg['inactive_patience']
        self.do_reid = tracker_cfg['do_reid']
        self.max_features_num = tracker_cfg['max_features_num']
        self.reid_sim_threshold = tracker_cfg['reid_sim_threshold']
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

    def reset(self, hard=True):
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

    def tracks_dets_matching_tracking(self, raw_dets, raw_scores, pre2cur_cts, pos=None, reid_cts=None, reid_feats=None):
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

            # index low-score dets #
            inds_low = raw_scores > 0.1
            inds_high = raw_scores < self.main_args.track_thresh
            inds_second = torch.logical_and(inds_low, inds_high)
            dets_second = raw_dets[inds_second]
            scores_second = raw_scores[inds_second]
            reid_cts_second = reid_cts[inds_second]

            # index high-score dets #
            remain_inds = raw_scores > self.main_args.track_thresh
            dets = raw_dets[remain_inds]
            scores_keep = raw_scores[remain_inds]
            reid_cts_keep = reid_cts[remain_inds]

            # Step 1: first assignment #
            if len(dets) > 0:
                assert dets.shape[0] == scores_keep.shape[0]
                # matching with gIOU
                iou_dist = box_ops.generalized_box_iou(warped_pos, dets)

                # todo fuse with dets scores here.
                if self.main_args.fuse_scores:
                    iou_dist *= scores_keep[None, :]

                iou_dist = 1 - iou_dist

                # todo recover inactive tracks here ?

                matches, u_track, u_detection = self.linear_assignment(iou_dist.cpu().numpy(),
                                                                       thresh=self.main_args.match_thresh)

                det_feats = F.grid_sample(reid_feats, reid_cts_keep.unsqueeze(0).unsqueeze(0),
                                              mode='bilinear', padding_mode='zeros', align_corners=False)[:, :, 0, :]


                if matches.shape[0] > 0:

                    # update track dets, scores #
                    for idx_track, idx_det in zip(matches[:, 0], matches[:, 1]):
                        t = self.tracks[idx_track]
                        t.pos = dets[[idx_det]]
                        t.add_features(det_feats[:, :, idx_det])
                        t.score = scores_keep[[idx_det]]

                pos_birth = dets[u_detection, :]
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
                    # matching with gIOU
                    iou_dist = 1 - box_ops.generalized_box_iou(remained_tracks_pos, dets_second)  # [0, 2]

                    matches, u_track_second, u_detection_second = self.linear_assignment(iou_dist.cpu().numpy(),thresh=0.4)  # stricter with low-score dets

                    # update u_track here
                    u_track = [track_indices[t_idx] for t_idx in u_track_second]

                    if matches.shape[0] > 0:
                        second_det_feats = F.grid_sample(reid_feats,
                                                         reid_cts_second[matches[:, 1]].unsqueeze(0).unsqueeze(0),
                                                         mode='bilinear', padding_mode='zeros', align_corners=False)[:,
                                           :, 0, :]
                        # update track dets, scores #
                        for cc, (idx_match, idx_det) in enumerate(zip(matches[:, 0], matches[:, 1])):
                            idx_track = track_indices[idx_match]
                            # print("low score match:", idx_track)
                            t = self.tracks[idx_track]
                            t.pos = dets_second[[idx_det]]
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
                t.pos = t.last_pos[-1]
                self.inactive_tracks += [t]
            else: # keep
                self.new_tracks.append(t)
        self.tracks = self.new_tracks

        return [pos_birth, scores_birth, dets_features_birth]

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

        # post processing #

        return out_boxes, out_scores, pre2cur_cts, mypos, reid_cts, outputs['reid'][0]

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

    def reid(self, blob, new_det_pos, new_det_scores, new_det_features):
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

                # # calculate IoU distances
                if self.main_args.iou_recover:
                    iou_dist = 1 - box_ops.generalized_box_iou(pos, new_det_pos)

                    matches, u_track, u_detection = self.linear_assignment(iou_dist.cpu().numpy(),
                                                                           thresh=self.main_args.match_thresh)
                else:

                    # assigned by appearance
                    matches, u_track, u_detection = self.linear_assignment(dist_mat.cpu().numpy(),
                                                                           thresh=self.reid_sim_threshold)

                assigned = []
                remove_inactive = []
                if matches.shape[0] > 0:
                    for r, c in zip(matches[:, 0], matches[:, 1]):
                        # inactive tracks reactivation #
                        if dist_mat[r, c] <= self.reid_sim_threshold or not self.main_args.iou_recover:
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
        # img_pil = Image.fromarray((255*blob['img'])[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8), 'RGB')
        # img_draw = ImageDraw.Draw(img_pil)
        # #
        # if self.last_image is not None:
        #     pre_img_pil = Image.fromarray((255 * self.last_image)[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8), 'RGB')
        #     pre_img_draw = ImageDraw.Draw(pre_img_pil)
        # else:
        #     pre_img_pil = img_pil
        #     pre_img_draw = img_draw
        # # # # plot #

        ###########################
        # Look for new detections #
        ###########################
        # detect

        det_pos, det_scores, pre2cur_cts, mypos, reid_cts, reid_features = self.detect_tracking_duel_vit(blob)

        ##################
        # Predict tracks #
        ##################
        if len(self.tracks):

            [det_pos, det_scores, dets_features_birth] = self.tracks_dets_matching_tracking(
                raw_dets=det_pos, raw_scores=det_scores, pre2cur_cts=pre2cur_cts, pos=mypos, reid_cts=reid_cts,
                reid_feats=reid_features)
        else:
            dets_features_birth = F.grid_sample(reid_features, reid_cts.unsqueeze(0).unsqueeze(0), mode='bilinear', padding_mode='zeros', align_corners=False)[:, :, 0, :].transpose(1, 2)[0]

        #####################
        # Create new tracks #
        #####################
        # filter birth candidates by scores
        valid_dets_idx = det_scores >= self.det_thresh
        det_pos = det_pos[valid_dets_idx]
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
            det_pos, det_scores, dets_features_birth = self.reid(blob, det_pos, det_scores, dets_features_birth)

            assert det_pos.shape[0] == dets_features_birth.shape[0] == det_scores.shape[0]

            # add new
            if det_pos.nelement() > 0:
                self.add(det_pos, det_scores, dets_features_birth)

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

        self.im_index += 1

    def get_results(self):
        return self.results


class Track(object):
    """This class contains all necessary for every individual track."""

    def __init__(self, pos, score, track_id, features, inactive_patience, max_features_num, mm_steps):
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