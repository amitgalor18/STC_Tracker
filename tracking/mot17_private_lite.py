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
import sys
import os
# dirty insert path #
cur_path = os.path.realpath(__file__)
cur_dir = "/".join(cur_path.split('/')[:-2])
sys.path.insert(0, cur_dir)
import torch

import numpy as np
from datasets.generic_dataset_test import GenericDataset_val
import csv
import os.path as osp
import yaml
from tracking.tracker import Tracker
from tracking.deformable_detr_lite import build as build_model
import argparse
from torch.utils.data import DataLoader

from shutil import copyfile
torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True
curr_pth = '/'.join(osp.dirname(__file__).split('/'))

def get_args_parser():
    parser = argparse.ArgumentParser('TransCenter Vit Encoder Mode', add_help=False)
    parser.add_argument('--ignoreIsCrowd', action='store_true')

    parser.add_argument('--d_model', default=[32, 64, 160, 256], type=int, nargs='+',
                        help="model dimensions in the transformer")

    parser.add_argument('--nheads', default=[1, 2, 5, 8], type=int, nargs='+',
                        help="Number of attention heads inside the transformer's attentions")

    parser.add_argument('--num_encoder_layers', default=[2, 2, 2, 2], type=int, nargs='+',
                        help="Number of encoding layers in the transformer")

    parser.add_argument('--num_decoder_layers', default=4, type=int,
                        help="Number of decoding layers in the transformer")

    parser.add_argument('--dim_feedforward_ratio', default=[8, 8, 4, 4], type=int, nargs='+',
                        help="Intermediate size of the feedforward layers dim ratio in the transformer blocks")

    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")

    parser.add_argument('--dec_n_points', default=9, type=int)

    parser.add_argument('--enc_n_points', default=[8, 8, 8, 8], type=int, nargs='+')

    parser.add_argument('--down_sample_ratio', default=[8, 4, 2, 1], type=int, nargs='+')

    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--linear', action='store_true',
                        help='linear vit')
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')


    # dataset parameters
    parser.add_argument('--dataset_file', default='mot17')
    parser.add_argument('--data_dir', default='MOT17', type=str)

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)

    # centers
    parser.add_argument('--num_classes', default=1, type=int)
    parser.add_argument('--input_h', default=608, type=int)
    parser.add_argument('--input_w', default=1088, type=int)
    parser.add_argument('--down_ratio', default=4, type=int)
    parser.add_argument('--trainval', action='store_true',
                        help='include validation in training and '
                             'test on test set')
    parser.add_argument('--half', default=False, action='store_true', help='half precision')

    parser.add_argument('--K', type=int, default=300,
                        help='max number of output objects.')
    # tracking
    parser.add_argument('--tracking', action='store_true')
    parser.add_argument('--pre_hm', action='store_true')
    parser.add_argument('--zero_pre_hm', action='store_true')
    parser.add_argument('--pre_thresh', type=float, default=-1)
    parser.add_argument('--track_thresh', type=float, default=0.3)
    parser.add_argument('--new_thresh', type=float, default=0.3)
    parser.add_argument('--public_det', action='store_true')
    parser.add_argument('--no_pre_img', action='store_true')
    parser.add_argument('--zero_tracking', action='store_true')
    parser.add_argument('--max_age', type=int, default=-1)

    parser.add_argument('--small', action='store_true', help='smaller dataset')
    parser.add_argument('--pretrained', type=str,
                        default="./model_zoo/pvtv2_backbone/pvt_v2_b4.pth",
                        help="pretrained")

    parser.add_argument('--recover', action='store_true',
                        help='recovery optimizer.')

    parser.add_argument('--mode', default='duel vit', type=str)


    return parser


def write_results(all_tracks, out_dir, seq_name=None, frame_offset=0):
    output_dir = out_dir + "/txt/"
    """Write the tracks in the format for MOT16/MOT17 sumbission

    all_tracks: dictionary with 1 dictionary for every track with {..., i:np.array([x1,y1,x2,y2]), ...} at key track_num

    Each file contains these lines:
    <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
    """
    #format_str = "{}, -1, {}, {}, {}, {}, {}, -1, -1, -1"

    assert seq_name is not None, "[!] No seq_name, probably using combined database"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file = osp.join(output_dir, seq_name+'.txt')

    with open(file, "w") as of:
        writer = csv.writer(of, delimiter=',')
        for i, track in all_tracks.items():
            for frame, bb in track.items():
                x1 = bb[0]
                y1 = bb[1]
                x2 = bb[2]
                y2 = bb[3]
                writer.writerow([frame+frame_offset, i+1, x1+1, y1+1, x2-x1+1, y2-y1+1, -1, -1, -1, -1])

    # copy to FRCNN, DPM.txt, private setting
    copyfile(file, file[:-7]+"FRCNN.txt")
    copyfile(file, file[:-7]+"DPM.txt")


def main(tracktor):
    torch.manual_seed(tracktor['seed'])
    torch.cuda.manual_seed(tracktor['seed'])
    np.random.seed(tracktor['seed'])
    torch.backends.cudnn.deterministic = True

    # load model
    main_args = get_args_parser().parse_args()
    main_args.pre_hm = True
    main_args.tracking = True
    main_args.clip = False
    main_args.fuse_scores = True
    main_args.iou_recover = True

    device = torch.device(main_args.device)
    ds = GenericDataset_val(root=main_args.data_dir, valset='val', select_seq='SDP')

    ds.default_resolution[0], ds.default_resolution[1] = main_args.input_h, main_args.input_w
    print(main_args.input_h, main_args.input_w)
    main_args.output_h = main_args.input_h // main_args.down_ratio
    main_args.output_w = main_args.input_w // main_args.down_ratio
    main_args.input_res = max(main_args.input_h, main_args.input_w)
    main_args.output_res = max(main_args.output_h, main_args.output_w)
    # threshold
    main_args.track_thresh = tracktor['tracker']["track_thresh"]
    main_args.match_thresh = tracktor['tracker']["match_thresh"]

    model, criterion, postprocessors = build_model(main_args)


    # load flowNet
    liteFlowNet = None



    # dataloader
    def collate_fn(batch):
        batch = list(zip(*batch))
        return tuple(batch)
    data_loader = DataLoader(ds, 1, shuffle=False, drop_last=False, num_workers=4,
                             pin_memory=True, collate_fn=collate_fn)
    for th in [0.3]:
        main_args.track_thresh = 0.3
        main_args.track_thresh = th
        # tracker
        tracker = Tracker(model, None, liteFlowNet, tracktor['tracker'], postprocessor=postprocessors['bbox'],
                          main_args=main_args)
        tracker.public_detections = False
        tracker.mode = main_args.mode

        models = [
            "./model_zoo/MOT17_ch_lite.pth",
        ]
        output_dirs = [
            curr_pth + '/test_models/MOT17_test_ch_lite/',
        ]

        for model_dir, output_dir in zip(models, output_dirs):
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            # load pretrained #
            tracktor['obj_detect_model'] = model_dir
            tracktor['output_dir'] = output_dir
            print("Loading: ", tracktor['obj_detect_model'])
            model.load_state_dict(torch.load(tracktor['obj_detect_model'])["model"])
            model.to(device)
            model.eval()
            from util.misc import NestedTensor

            pre_seq_name = ''
            frame_offset = 0
            num_frames = 0
            pub_dets = None
            start = 0
            first_frame = True
            for idx, [samples, meta] in enumerate(data_loader):

                num_frames += 1
                [orig_size, im_name, video_name, orig_img, trans] = meta[0]

                if os.path.exists(output_dir + "txt/" + video_name + '.txt'):
                    continue

                if video_name != pre_seq_name:
                    print("video_name", video_name)
                    # save results #
                    if not os.path.exists(output_dir + "txt/" + pre_seq_name + '.txt') and idx != 0:
                        # save results #
                        results = tracker.get_results()
                        print(f"Tracks found: {len(results)}")
                        write_results(results, tracktor['output_dir'], seq_name=pre_seq_name, frame_offset=frame_offset)

                    # update pre_seq_name #
                    pre_seq_name = video_name
                    first_frame = True

                    pub_dets = ds.VidPubDet[video_name]

                    # reset tracker #
                    tracker.reset()
                    # update inactive patience according to framerate
                    seq_info_path = os.path.join(main_args.data_dir, "train", video_name, 'seqinfo.ini')
                    print("seq_info_path ", seq_info_path)
                    assert os.path.exists(seq_info_path)
                    with open(seq_info_path, 'r') as f:
                        reader = csv.reader(f, delimiter='=')
                        for row in reader:
                            if 'frameRate' in row:
                                framerate = int(row[1])

                    print('frameRate', framerate)
                    tracker.inactive_patience = framerate/30 * tracktor['tracker']['inactive_patience']

                    # init offset #
                    frame_offset = int(im_name[:-4])
                    print("frame offset : ", frame_offset)

                # starts with 0 #
                pub_det = pub_dets[int(im_name[:-4]) - 1]

                print("step frame: ", im_name)

                batch = {'frame_name': im_name, 'video_name': video_name, 'img': orig_img.to(device),
                         'samples': samples[0].to(device), 'orig_size': orig_size.unsqueeze(0).to(device),
                         'dets': torch.FloatTensor(pub_det)[:, :-1].to(device) if len(pub_det) > 0 else torch.zeros(0,
                                                                                                                    4),
                         'trans': trans}

                tracker.step_reidV3_pre_tracking_vit(batch)
                first_frame = False

            # save last results #
            if not os.path.exists(output_dir + "txt/" + video_name + '.txt'):
                # save results #
                results = tracker.get_results()
                print(f"Tracks found: {len(results)}")
                write_results(results, tracktor['output_dir'], seq_name=video_name, frame_offset=frame_offset)


with open(curr_pth + '/cfgs/transcenter_cfg.yaml', 'r') as f:
    tracktor = yaml.load(f)['tracktor']

main(tracktor)
