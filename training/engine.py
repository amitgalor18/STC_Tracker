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
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable
import torch
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
import copy


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, adaptive_clip: bool=False,
                    scaler: torch.nn.Module=None, half:bool=True):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50
    data_type = torch.float16 if half else torch.float32

    for ret in metric_logger.log_every(data_loader, print_freq, header):

        samples = utils.NestedTensor(ret['image'], ret['pad_mask'])
        # samples.tensors = samples.tensors.to(data_type)
        samples = samples.to(device)
        pre_samples = utils.NestedTensor(ret['pre_img'], ret['pre_pad_mask'])
        # pre_samples.tensors = pre_samples.tensors.to(data_type)
        pre_samples = pre_samples.to(device)

        targets = {k: v.to(device) for k, v in ret.items() if
                   k != 'orig_image' and k != 'image' and 'pad_mask' not in k and 'pre_img' not in k}

        # save memory, reduce max_dets#
        max_dets, _ = torch.max(targets["valid_num_pre_dets"], dim=0)
        max_dets = int(max_dets)
        if max_dets == 0:
            max_dets = 5

        targets['pre_cts'] = targets['pre_cts'][:, :max_dets, :]
        targets['tracking'] = targets['tracking'][:, :max_dets, :]
        targets['tracking_mask'] = targets['tracking_mask'][:, :max_dets, :]
        with torch.cuda.amp.autocast(enabled=half):
            outputs = model(samples, pre_samples=pre_samples, pre_hm=targets['pre_cts'])
            loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        assert len(weight_dict.keys()) == len(loss_dict_reduced.keys())

        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        for param in model.parameters():
            param.grad = None
        scaler.scale(losses).backward()

        if adaptive_clip:
            if max_norm > 0:
                scaler.unscale_(optimizer)
                utils.clip_grad_norm(model.parameters())
                grad_total_norm = utils.get_total_grad_norm(model.parameters())
            else:
                grad_total_norm = utils.get_total_grad_norm(model.parameters())
        else:
            if max_norm > 0:
                scaler.unscale_(optimizer)
                grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            else:
                grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)

        scaler.step(optimizer)
        scaler.update()


        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, half=True):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    data_type = torch.float16 if half else torch.float32


    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # set max Dets to 300
    coco_evaluator.coco_eval[iou_types[0]].params.maxDets = [300, 300, 300]

    for ret in metric_logger.log_every(data_loader, 50, header):
        samples = utils.NestedTensor(ret['image'], ret['pad_mask'])
        # samples.tensors = samples.tensors.to(data_type)
        samples = samples.to(device)
        pre_samples = utils.NestedTensor(ret['pre_img'], ret['pre_pad_mask'])
        # pre_samples.tensors = pre_samples.tensors.to(data_type)
        # pre_hm = ret['pre_hm'].to(device)
        pre_samples = pre_samples.to(device)

        targets = {k: v.to(device) for k, v in ret.items() if k != 'orig_image' and k != 'image' and 'pad_mask' not in k and 'pre_img' not in k}

        # max_dets, _ = torch.max(targets['tracking_mask'][:,:,0].sum(-1), dim=0)
        max_dets, _ = torch.max(targets["valid_num_pre_dets"], dim=0)
        max_dets = int(max_dets)
        if max_dets == 0:
            max_dets = 5

        targets['pre_cts'] = targets['pre_cts'][:, :max_dets, :]
        targets['tracking'] = targets['tracking'][:, :max_dets, :]
        targets['tracking_mask'] = targets['tracking_mask'][:, :max_dets, :]
        with torch.cuda.amp.autocast(enabled=half):
            outputs = model(samples, pre_samples=pre_samples, pre_hm=targets['pre_cts'].clone())
            loss_dict = criterion(copy.deepcopy(outputs), targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)

        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        results = postprocessors['bbox'](outputs, targets['orig_size'], pre_cts=targets['pre_cts'], filter_score=False)
        res = {img_id.item(): output for img_id, output in zip(targets['image_id'], results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
    return stats, coco_evaluator
