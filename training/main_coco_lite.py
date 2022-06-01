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
os.environ["OMP_NUM_THREADS"] = "4"
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import util.misc as utils
import datasets.samplers as samplers

from training.engine import evaluate, train_one_epoch
from training.deformable_detr_lite import build as build_model
from datasets.coco import COCO

# xyh #
from torch.utils.tensorboard import SummaryWriter

torch.backends.cudnn.benchmark = False


def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--ignoreIsCrowd', action='store_true')
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    parser.add_argument('--pretrained', type=str,
                        default="./model_zoo/pvtv2_backbone/pvt_v2_b0.pth",
                        help="pretrained")
    parser.add_argument('--half', default=False, action='store_true', help='half precision')


    # * Transformer
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

    parser.add_argument('--num_feature_levels', default=1, type=int, help='number of feature levels')

    parser.add_argument('--dec_n_points', default=9, type=int)

    parser.add_argument('--enc_n_points', default=[8, 8, 8, 8], type=int, nargs='+')

    parser.add_argument('--down_sample_ratio', default=[8, 4, 2, 1], type=int, nargs='+')

    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--linear', action='store_true',
                        help='linear vit')

    parser.add_argument('--heads', default=['hm', 'reg', 'wh', 'center_offset', 'tracking'], type=str, nargs='+')

    # * Loss coefficients
    parser.add_argument('--hm_weight', default=1, type=float)
    parser.add_argument('--off_weight', default=1, type=float)
    parser.add_argument('--wh_weight', default=0.1, type=float)
    parser.add_argument('--ct_offset_weight', default=0.1, type=float)
    parser.add_argument('--boxes_weight', default=0.5, type=float)
    parser.add_argument('--giou_weight', default=0.4, type=float)
    parser.add_argument('--norm_factor', default=1.0, type=float)
    parser.add_argument('--tracking_weight', default=1, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='cocoperson')
    parser.add_argument('--data_dir', default='cocoperson', type=str)
    parser.add_argument('--coco_panargsic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    # centers
    parser.add_argument('--num_classes', default=1, type=int)
    parser.add_argument('--input_h', default=608, type=int)
    parser.add_argument('--input_w', default=1088, type=int)
    parser.add_argument('--down_ratio', default=4, type=int)
    parser.add_argument('--dense_reg', type=int, default=1, help='')
    parser.add_argument('--trainval', action='store_true',
                        help='include validation in training and '
                             'test on test set')

    parser.add_argument('--K', type=int, default=300,
                        help='max number of output objects.')

    parser.add_argument('--debug', action='store_true')

    # noise
    parser.add_argument('--not_rand_crop', action='store_true',
                        help='not use the random crop data augmentation'
                             'from CornerNet.')
    parser.add_argument('--not_max_crop', action='store_true',
                        help='used when the training dataset has'
                             'inbalanced aspect ratios.')
    parser.add_argument('--shift', type=float, default=0.05,
                        help='when not using random crop'
                             'apply shift augmentation.')
    parser.add_argument('--scale', type=float, default=0.05,
                        help='when not using random crop'
                             'apply scale augmentation.')
    parser.add_argument('--rotate', type=float, default=0,
                        help='when not using random crop'
                             'apply rotation augmentation.')
    parser.add_argument('--flip', type=float, default=0.5,
                        help='probability of applying flip augmentation.')
    parser.add_argument('--no_color_aug', action='store_true',
                        help='not use the color augmenation '
                             'from CornerNet')
    parser.add_argument('--aug_rot', type=float, default=0,
                        help='probability of applying '
                             'rotation augmentation.')

    # tracking
    parser.add_argument('--max_frame_dist', type=int, default=3)
    parser.add_argument('--merge_mode', type=int, default=1)
    parser.add_argument('--tracking', action='store_true')
    parser.add_argument('--pre_hm', action='store_true')
    parser.add_argument('--same_aug_pre', action='store_true')
    parser.add_argument('--zero_pre_hm', action='store_true')
    parser.add_argument('--hm_disturb', type=float, default=0.05)
    parser.add_argument('--lost_disturb', type=float, default=0.4)
    parser.add_argument('--fp_disturb', type=float, default=0.1)
    parser.add_argument('--pre_thresh', type=float, default=-1)
    parser.add_argument('--track_thresh', type=float, default=0.3)
    parser.add_argument('--new_thresh', type=float, default=0.3)
    parser.add_argument('--ltrb_amodal', action='store_true')
    parser.add_argument('--ltrb_amodal_weight', type=float, default=0.1)
    parser.add_argument('--public_det', action='store_true')
    parser.add_argument('--no_pre_img', action='store_true')
    parser.add_argument('--zero_tracking', action='store_true')
    parser.add_argument('--hungarian', action='store_true')
    parser.add_argument('--max_age', type=int, default=-1)
    parser.add_argument('--out_thresh', type=float, default=-1,
                        help='')
    parser.add_argument('--image_blur_aug', action='store_true',
                        help='blur image for aug.')
    parser.add_argument('--adaptive_clip', action='store_true',
                        help='adaptive_clip')

    parser.add_argument('--small', action='store_true',
                        help='smaller dataset')

    parser.add_argument('--recover', action='store_true',
                        help='recovery optimizer.')
    return parser


def main(args):
    utils.init_distributed_mode(args)

    if utils.is_main_process():
        logger = SummaryWriter(os.path.join(args.output_dir, 'log'))
    my_map = 0
    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"

    device = torch.device(args.device)

    dataset_train = COCO(args, 'train')
    dataset_val = COCO(args, 'val')

    # input output shapes
    dataset_val.default_resolution[0], dataset_val.default_resolution[1] = args.input_h, args.input_w
    print(args.input_h, args.input_w)
    print(dataset_val.default_resolution)
    args.output_h = args.input_h // args.down_ratio
    args.output_w = args.input_w // args.down_ratio
    args.input_res = max(args.input_h, args.input_w)
    args.output_res = max(args.output_h, args.output_w)
    # threshold
    args.out_thresh = max(args.track_thresh, args.out_thresh)
    args.pre_thresh = max(args.track_thresh, args.pre_thresh)
    args.new_thresh = max(args.track_thresh, args.new_thresh)
    print(args)
    print("trainset #samples: ", len(dataset_train))
    print("valset #samples: ", len(dataset_val))

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)

    def worker_init_fn(worker_id):
        np.random.seed(seed + worker_id)
        random.seed(seed + worker_id)

    model, criterion, postprocessors = build_model(args)
    model.to(device)
    # print(model)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
            sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
            sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, num_workers=args.num_workers,
                                   pin_memory=True, worker_init_fn=worker_init_fn)
    data_loader_val = DataLoader(dataset_val, 10, sampler=sampler_val,
                                 drop_last=False, num_workers=args.num_workers,
                                 pin_memory=True)

    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if
                       match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    if args.distributed:
        # sync BN #
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    print("Loading base ds...")
    base_ds = dataset_val.coco
    print("Loading base ds done.")

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        missing_keys = []
        unexpected_keys = []
        print("Loading ", args.resume)
        print("loading all.")

        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint and args.recover:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            # print(optimizer.param_groups)
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler
            #  (e.g., decrease lr in advance).
            args.override_resumed_lr_drop = True
            if args.override_resumed_lr_drop:
                print(
                    'Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                lr_scheduler.step_size = args.lr_drop
                lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1
        # check the resumed model
        if not args.eval:
            test_stats, coco_evaluator = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir, args.half
            )

            # xyh, valbest save#
            avg_map = np.mean([
                test_stats['coco_eval_bbox'][0],
                test_stats['coco_eval_bbox'][1],
                test_stats['coco_eval_bbox'][3],
                test_stats['coco_eval_bbox'][4],
                test_stats['coco_eval_bbox'][5]

            ])
            if avg_map >= my_map and args.start_epoch > 1:
                my_map = float(avg_map)
                print("resume map ", my_map)
            else:
                print("first epoch, init map:", my_map)

    if args.eval:
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir, args.half)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    print("Start training")
    start_time = time.time()

    scaler = torch.cuda.amp.GradScaler(enabled=args.half)
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm,
            adaptive_clip=args.adaptive_clip, scaler=scaler, half=args.half)
        lr_scheduler.step()

        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 5 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 50 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

            # tensorboard logger #
            if utils.is_main_process():
                logger.add_scalar("LR/train", log_stats['train_lr'], epoch)
                logger.add_scalar("Loss/train", log_stats['train_loss'], epoch)
                logger.add_scalar("HMLoss/train", log_stats['train_hm'], epoch)
                logger.add_scalar("REGLoss/train", log_stats['train_reg'], epoch)
                logger.add_scalar("WHLoss/train", log_stats['train_wh'], epoch)
                logger.add_scalar("GIOULoss/train", log_stats['train_giou'], epoch)
                logger.add_scalar("BOXLoss/train", log_stats['train_boxes'], epoch)
                logger.add_scalar("TrackingLoss/train", log_stats['train_tracking'], epoch)

        if epoch % 4 == 0:
            test_stats, coco_evaluator = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir, half=args.half
            )

            # valbest save#
            avg_map = np.mean([
                test_stats['coco_eval_bbox'][0],
                test_stats['coco_eval_bbox'][1],
                test_stats['coco_eval_bbox'][3],
                test_stats['coco_eval_bbox'][4],
                test_stats['coco_eval_bbox'][5]

            ])
            if avg_map >= my_map:
                my_map = float(avg_map)
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'mAP': [
                        test_stats['coco_eval_bbox'][0],
                        test_stats['coco_eval_bbox'][1],
                        test_stats['coco_eval_bbox'][3],
                        test_stats['coco_eval_bbox'][4],
                        test_stats['coco_eval_bbox'][5]

                    ],
                }, output_dir / 'val_best.pth')

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

            # tensorboard logger #
            if args.output_dir and utils.is_main_process():
                logger.add_scalar("Loss/test", log_stats['test_loss'], epoch)
                logger.add_scalar("HMLoss/test", log_stats['test_hm'], epoch)
                logger.add_scalar("REGLoss/test", log_stats['test_reg'], epoch)
                logger.add_scalar("WHLoss/test", log_stats['test_wh'], epoch)
                logger.add_scalar("GIOULoss/test", log_stats['test_giou'], epoch)
                logger.add_scalar("BOXLoss/test", log_stats['test_boxes'], epoch)
                logger.add_scalar("TrackingLoss/test", log_stats['test_tracking'], epoch)

                logger.add_scalar("mAP_ALL/test", log_stats['test_coco_eval_bbox'][0], epoch)
                logger.add_scalar("mAP_ALL_05/test", log_stats['test_coco_eval_bbox'][1], epoch)
                logger.add_scalar("mAP_SMALL/test", log_stats['test_coco_eval_bbox'][3], epoch)
                logger.add_scalar("mAP_MEDIUM/test", log_stats['test_coco_eval_bbox'][4], epoch)
                logger.add_scalar("mAP_Large/test", log_stats['test_coco_eval_bbox'][5], epoch)

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    args.clip = False
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
