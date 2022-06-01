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
"""
Deformable DETR model and criterion classes.
"""
import torch
from torch import nn
from losses.utils import _sigmoid
from util.misc import NestedTensor
from post_processing.decode import generic_decode
from post_processing.post_process import generic_post_process
from models.deformable_transformer import build_deforamble_transformer
import copy
from models.dla import IDAUpV3_bis
from torch import Tensor


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, transformer, num_classes, output_shape, half=True):
        """ Initializes the model.
        """
        super().__init__()
        self.transformer = transformer
        self.output_shape = output_shape

        # # different ida up for tracking and detection
        # self.ida_up_tracking = IDAUpV3(
        #     64, [256, 256, 256], [])

        self.ida_up = IDAUpV3_bis(
            64, [256, 256, 256, 256])

        '''
        (0): Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU(inplace=True)
        (2): Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1))
        '''

        self.hm = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=3//2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )

        self.ct_offset = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=3// 2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 2, kernel_size=1, stride=1, padding=0, bias=True)
        )

        self.reg = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=3// 2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 2, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True)
        )

        self.wh = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=3// 2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 2, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True)
        )
        # future tracking offset
        self.tracking = nn.Sequential(
            nn.Linear(130, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )

        self.ida_up_tracking = nn.ModuleList([nn.Sequential(nn.Linear(256, 64), nn.ReLU(inplace=True)),
                                              nn.Sequential(nn.Linear(256, 64), nn.ReLU(inplace=True))])

        # init weights #
        # prior bias
        self.hm[-1].bias.data.fill_(-4.6)
        fill_fc_weights(self.reg)
        fill_fc_weights(self.wh)
        fill_fc_weights(self.ct_offset)
        fill_fc_weights(self.tracking)

        self.pre_memory = None
        self.masks_flatten = None

    def forward(self, samples: NestedTensor, pre_samples: NestedTensor, pre_cts: Tensor):
        assert isinstance(samples, NestedTensor)

        merged_hs, self.pre_memory, pre_gathered_features, self.masks_flatten = self.transformer(samples, pre_samples,
                                                                                                 pre_cts=pre_cts,
                                                                             pre_memories=self.pre_memory,
                                                                             masks_flatten=self.masks_flatten)

        hs = []
        pre_hs = []

        for hs_m, pre_hs_m in merged_hs:
            hs.append(hs_m)
            pre_hs.append(pre_hs_m)

        outputs_hms = []
        outputs_regs = []
        outputs_whs = []
        outputs_ct_offsets = []
        outputs_tracking = []
        outputs_reid = []

        for layer_lvl in range(len(hs)):
            hs[layer_lvl] = self.ida_up(hs[layer_lvl], 0, len(hs[layer_lvl]))[-1]
            pre_hs[layer_lvl] = self.ida_up_tracking[0](pre_hs[layer_lvl])

            ct_offset = self.ct_offset(hs[layer_lvl])
            wh_head = self.wh(hs[layer_lvl])
            reg_head = self.reg(hs[layer_lvl])
            hm_head = self.hm(hs[layer_lvl])
            outputs_reid.append(hs[layer_lvl])

            # gather features #
            # (x,y) to index
            pre_reference_points = pre_cts.clone()
            # normalize
            pre_reference_points[:, :, 0] /= self.output_shape[1]
            pre_reference_points[:, :, 1] /= self.output_shape[0]
            # clamp #
            pre_reference_points = torch.clamp(pre_reference_points, min=0.0, max=1.0)
            assert pre_gathered_features.shape[:-1] == pre_hs[layer_lvl].shape[:-1] == pre_reference_points.shape[:-1]
            tracking_head = self.tracking(torch.cat([self.ida_up_tracking[1](pre_gathered_features.detach()),
                                                     pre_hs[layer_lvl], pre_reference_points], dim=2))

            outputs_whs.append(wh_head)
            outputs_ct_offsets.append(ct_offset)
            outputs_regs.append(reg_head)
            outputs_hms.append(hm_head)
            outputs_tracking.append(tracking_head)

        out = {'hm': torch.stack(outputs_hms), 'wh': torch.stack(outputs_whs), 'reg': torch.stack(outputs_regs),
               'center_offset': torch.stack(outputs_ct_offsets), 'tracking': torch.stack(outputs_tracking),
               'reid': torch.stack(outputs_reid)}

        return out


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    def __init__(self, args, valid_ids):
        self.args = args
        self._valid_ids = valid_ids
        print("valid_ids: ", self._valid_ids)
        super().__init__()

    def _sigmoid_output(self, output):
        if 'hm' in output:
            output['hm'] = _sigmoid(output['hm'])
        return output
    @torch.no_grad()
    def forward(self, outputs, target_sizes, pre_cts, filter_score=True):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        # for map you don't need to filter
        if filter_score:
            out_thresh = self.args.pre_thresh
        else:
            out_thresh = 0.0
        # get the output of last layer of transformer
        output = {k: v[-1].cpu() for k, v in outputs.items() if k != 'boxes'}

        # 'hm' is not _sigmoid!
        output = self._sigmoid_output(output)

        dets = generic_decode(output, K=self.args.K, opt=self.args)

        dws = []
        dhs = []
        ratios = []
        height, width = self.args.input_h, self.args.input_w

        for target_size in target_sizes:
            shape = target_size.cpu().numpy()  # shape = [height, width]
            ratio = min(float(height) / shape[0], float(width) / shape[1])
            new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]
            dw = (width - new_shape[0]) / 2  # width padding
            dh = (height - new_shape[1]) / 2  # height padding

            dws.append(dw)
            dhs.append(dh)
            ratios.append(ratio)
        results, pre_results = generic_post_process(opt=self.args, pre_cts=pre_cts.cpu(), dets=dets, dws=dws, dhs=dhs, ratios=ratios, filter_by_scores=out_thresh)
        coco_results = []
        for btch_idx in range(len(results)):
            boxes = []
            scores = []
            labels = []

            for det in results[btch_idx]:
                if det['bbox'][2] - det['bbox'][0] < 1 or det['bbox'][3] - det['bbox'][1] < 1:
                    continue
                boxes.append(det['bbox'])
                scores.append(det['score'])
                labels.append(self._valid_ids[det['class']-1])

            if len(boxes) > 0:
                coco_results.append({'scores': torch.as_tensor(scores).float(),
                                     'labels': torch.as_tensor(labels).int(),
                                     'boxes': torch.as_tensor(boxes).float(),
                                     'pre2cur_cts': pre_results[btch_idx]['pre2cur_cts']})
            else:
                coco_results.append({'scores': torch.zeros(0).float(),
                                     'labels': torch.zeros(0).int(),
                                     'boxes': torch.zeros(0, 4).float(),
                                     'pre2cur_cts': torch.zeros(0, 2).float()
                                     })
        return coco_results


def build(args):
    num_classes = 1 if args.dataset_file != 'coco' else 80

    if args.dataset_file == 'coco':
        valid_ids = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
            14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
            37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
            48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
            58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
            72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
            82, 84, 85, 86, 87, 88, 89, 90]
    else:

        valid_ids = [1]

    device = torch.device(args.device)
    print("we set arg.half=True for inference")
    args.half = True
    transformer = build_deforamble_transformer(args)
    print("num_classes", num_classes)
    model = DeformableDETR(
        transformer,
        num_classes=num_classes,
        output_shape=(args.input_h/args.down_ratio, args.input_w/args.down_ratio),
        half=args.half
    )

    postprocessors = {'bbox': PostProcess(args, valid_ids)}
    return model, None, postprocessors
