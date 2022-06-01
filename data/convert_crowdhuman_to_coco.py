## TransCenter: Transformers with Dense Queries for Multiple-Object Tracking
## Copyright Inria
## Year 2021
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
##
## TransCenter uses packages from
## (1) 2019 Charles Shang. (BSD 3-Clause Licence: https://github.com/CharlesShang/DCNv2)
## (2) 2017 NVIDIA CORPORATION. (Apache License, Version 2.0: https://github.com/NVIDIA/flownet2-pytorch/tree/master/networks/correlation_package)
## (3) 2019 Simon Niklaus. (GNU General Public License v3.0: https://github.com/sniklaus/pytorch-liteflownet)
## (4) 2018 Tak-Wai Hui. (Copyright (c), see details in the LICENSE file: https://github.com/twhui/LiteFlowNet)
import os
import numpy as np
import json
import cv2

DATA_PATH = 'CrowdHumanPath'
OUT_PATH = DATA_PATH + 'annotations/'
SPLITS = ['val', 'train']
DEBUG = False


def load_func(fpath):
    print('fpath', fpath)
    assert os.path.exists(fpath)
    with open(fpath, 'r') as fid:
        lines = fid.readlines()
    records = [json.loads(line.strip('\n')) for line in lines]
    return records


if __name__ == '__main__':
    import copy
    def bb_IoU(boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        # return the intersection over union value
        return iou


    if not os.path.exists(OUT_PATH):
        os.mkdir(OUT_PATH)
    for split in SPLITS:
        data_path = DATA_PATH + split
        out_path = OUT_PATH + '{}.json'.format(split)
        out = {'images': [], 'annotations': [],
               'categories': [{'id': 1, 'name': 'person'}]}
        ann_path = DATA_PATH + '/annotation_{}.odgt'.format(split)
        anns_data = load_func(ann_path)
        image_cnt = 0
        ann_cnt = 0
        video_cnt = 0
        trackid_cnt = 0
        for ann_data in anns_data:
            # if (image_cnt >= 3750 and split == 'train') or \
            #     (image_cnt >= 1092 and split == 'val'):
            #     break
            image_cnt += 1
            image_info = {'file_name': '{}.jpg'.format(ann_data['ID']),
                          'id': image_cnt}
            out['images'].append(image_info)
            if split != 'test':
                anns = ann_data['gtboxes']
                for i in range(len(anns)):

                    if anns[i]['tag'] == 'person':
                        full_box = np.array(anns[i]['fbox'], dtype=np.float32).copy()
                        full_box[2:] = full_box[:2] + full_box[2:]
                        v_box = np.array(anns[i]['vbox'], dtype=np.float32).copy()
                        v_box[2:] = v_box[:2] + v_box[2:]
                        vis_level = bb_IoU(full_box, v_box)
                        # print("vis level:", vis_level)
                        iscrowd = ('extra' in anns[i] and
                                   'ignore' in anns[i]['extra'] and
                                   anns[i]['extra']['ignore'] == 1) or vis_level < 0.1
                        trackid_cnt += 1
                        ann_cnt += 1
                        ann = {'id': ann_cnt,
                               'category_id': 1,
                               'image_id': image_cnt,
                               'bbox_vis': anns[i]['vbox'],
                               'bbox': anns[i]['fbox'],
                               'iscrowd': int(iscrowd),
                               'area': anns[i]['fbox'][2]*anns[i]['fbox'][3],
                               'track_id': trackid_cnt}
                        out['annotations'].append(ann)
        print('loaded {} for {} images and {} samples'.format(
            split, len(out['images']), len(out['annotations'])))
        json.dump(out, open(out_path, 'w'))
