## TransCenter V2: Transformers with Dense Representations for Multiple-Object Tracking <br />
## An update towards a more efficient and powerful TransCenter, TransCenter-Lite. ##

**TransCenter: Transformers with Dense Representations for Multiple-Object Tracking** <br />
[Yihong Xu](https://team.inria.fr/robotlearn/team-members/yihong-xu/), [Yutong Ban](https://team.inria.fr/perception/team-members/yutong-ban/), [Guillaume Delorme](https://team.inria.fr/robotlearn/team-members/guillaume-delorme/), [Chuang Gan](https://people.csail.mit.edu/ganchuang/), [Daniela Rus](http://danielarus.csail.mit.edu/), [Xavier Alameda-Pineda](http://xavirema.eu/) <br />
**[[Paper](https://arxiv.org/abs/2103.15145)]** **[[Project](https://team.inria.fr/robotlearn/transcenter-transformers-with-dense-queriesfor-multiple-object-tracking/)]**<br />

<div align="center">
  <img src="https://github.com/yihongXU/TransCenter/raw/main/eTransCenter_pipeline.png" width="1200px" />
</div>

## Bibtex
**If you find this code useful, please star the project and consider citing:** <br />
```
@misc{xu2021transcenter,
      title={TransCenter: Transformers with Dense Representations for Multiple-Object Tracking}, 
      author={Yihong Xu and Yutong Ban and Guillaume Delorme and Chuang Gan and Daniela Rus and Xavier Alameda-Pineda},
      year={2021},
      eprint={2103.15145},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Environment Preparation 
### Option 1 (recommended):
We provide a singularity image (similar to docker) containing all the packages we need for TransCenter:

1) Install singularity > 3.7.1:  [https://sylabs.io/guides/3.0/user-guide/installation.html#install-on-linux](https://sylabs.io/guides/3.0/user-guide/installation.html#install-on-linux)
2) Download one of the singularity images:

[**transcenter_singularity.sif**](https://drive.google.com/file/d/1ln18FLon2HczviuTOxVjEW_1mJ2zhkZF/view?usp=sharing) tested with Nvidia RTX TITAN, Quadro RTX 8000, RTX 2080Ti, Quadro RTX 4000.

- Launch a Singularity image
```shell
singularity shell --nv --bind yourLocalPath:yourPathInsideImage YourSingularityImage.sif
```
**- -bind: to link a singularity path with a local path. By doing this, you can find data from local PC inside Singularity image;** <br />
**- -nv: use the local Nvidia driver.**

### Option 2:

You can also build your own environment:
1) we use anaconda to simplify the package installations, you can download anaconda (4.10.3) here: [https://www.anaconda.com/products/individual](https://www.anaconda.com/products/individual)
2) you can create your conda env by doing 
```
conda env create -n <env_name> -f eTransCenter.yml
```
3) TransCenter uses Deformable transformer from Deformable DETR. Therefore, we need to install deformable attention modules:
```
cd ./to_install/ops
sh ./make.sh
# unit test (should see all checking is True)
python test.py
```
4.for the up-scale and merge module in TransCenter, we use deformable convolution module, you can install it with:
```
cd ./to_install/DCNv2
./make.sh         # build
python testcpu.py    # run examples and gradient check on cpu
python testcuda.py   # run examples and gradient check on gpu
```
See also known issues from [https://github.com/CharlesShang/DCNv2](https://github.com/CharlesShang/DCNv2).
If you have issues related to cuda of the third-party modules, please try to recompile them in the GPU that you use for training and testing. 
The dependencies are compatible with Pytorch 1.6, cuda 10.1.

If you install the DCNv2 and Deformable Transformer packages from other implementations, please replace the corresponding files with dcn_v2.py and ms_deform_attn.py in ./toinstall
for allowing half-precision operations with the customized packages.
## Data Preparation ##
[ms coco](https://cocodataset.org/#download): we use only the *person* category for pretraining TransCenter. The code for filtering is provided in ./data/coco_person.py.
```
@inproceedings{lin2014microsoft,
  title={Microsoft coco: Common objects in context},
  author={Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Doll{\'a}r, Piotr and Zitnick, C Lawrence},
  booktitle={European conference on computer vision},
  pages={740--755},
  year={2014},
  organization={Springer}
}
```

[CrowdHuman](https://www.crowdhuman.org/): CrowdHuman labels are converted to coco format, the conversion can be done through ./data/convert_crowdhuman_to_coco.py.

```
@article{shao2018crowdhuman,
    title={CrowdHuman: A Benchmark for Detecting Human in a Crowd},
    author={Shao, Shuai and Zhao, Zijian and Li, Boxun and Xiao, Tete and Yu, Gang and Zhang, Xiangyu and Sun, Jian},
    journal={arXiv preprint arXiv:1805.00123},
    year={2018}
  }
```

[MOT17](https://motchallenge.net/data/MOT17/): MOT17 labels are converted to coco format, the conversion can be done through ./data/convert_mot_to_coco.py.

```
@article{milan2016mot16,
  title={MOT16: A benchmark for multi-object tracking},
  author={Milan, Anton and Leal-Taix{\'e}, Laura and Reid, Ian and Roth, Stefan and Schindler, Konrad},
  journal={arXiv preprint arXiv:1603.00831},
  year={2016}
}
```

[MOT20](https://motchallenge.net/data/MOT20/): MOT20 labels are converted to coco format, the conversion can be done through ./data/convert_mot20_to_coco.py.

```
@article{dendorfer2020mot20,
  title={Mot20: A benchmark for multi object tracking in crowded scenes},
  author={Dendorfer, Patrick and Rezatofighi, Hamid and Milan, Anton and Shi, Javen and Cremers, Daniel and Reid, Ian and Roth, Stefan and Schindler, Konrad and Leal-Taix{\'e}, Laura},
  journal={arXiv preprint arXiv:2003.09003},
  year={2020}
}
```

We also provide the filtered/converted labels:

[ms coco person labels](https://drive.google.com/drive/folders/1PuVXRQV10fRW8MTBG8txhamSJqaSWBbc?usp=sharing): please put the *annotations* folder inside *cocoperson* to your ms coco dataset root folder.

[CrowdHuman coco format labels](https://drive.google.com/drive/folders/152K_-FjltstDPkW3jKUEaRHrtxes6mr8?usp=sharing): please put the *annotations* folder inside *crowdhuman* to your CrowdHuman dataset root folder.

[MOT17 coco format labels](https://drive.google.com/drive/folders/1SxaVF4KddLp7t_twF53wpOifrLNzDNXE?usp=sharing): please put the *annotations* and *annotations_onlySDP* folders inside *MOT17* to your MOT17 dataset root folder.

[MOT20 coco format labels](https://drive.google.com/drive/folders/12svjv5V7-pC2BHJfyxfo_9wYcEUWGs27?usp=sharing): please put the *annotations* folder inside *MOT20* to your MOT20 dataset root folder.


## Model Zoo
***For TransCenter V2***:

[PVTv2 pretrained](https://drive.google.com/drive/folders/1h4mvNhYbUqwH04Cd63JZOah4xo97U2H3?usp=sharing): pretrained model from deformable-DETR.

[coco_pretrained](https://drive.google.com/file/d/11FAdJoXS3tPjRSzv4G-yEVa9ABiCmJGE/view?usp=sharing): model trained with coco person dataset.

[MOT17_fromCoCo](https://drive.google.com/file/d/1hURJ9QSSWoX2NsE0rxfyJD3bfpwlAC3z/view?usp=sharing): model pretrained on coco person and fine-tuned on MOT17 trainset.

[MOT17_trained_with_CH](https://drive.google.com/file/d/1f_zzZWK3QA0wNhHOiH04hasWWSBbs8gu/view?usp=sharing): model trained on CrowdHuman and MOT17 trainset.

[MOT20_fromCoCo](https://drive.google.com/file/d/14WeLNNpWkNEyOIx6zrA4vuVxg4RrNkCh/view?usp=sharing): model pretrained on coco person and fine-tuned on MOT20 trainset.

[MOT20_trained_with_CH](https://drive.google.com/file/d/1F65LpeMj7nkvjpoKcweMCymo9aP2Z9ag/view?usp=sharing): model trained on CrowdHuman and MOT20 trainset.


***For TransCenter-Lite***:

[coco_pretrained_lite](https://drive.google.com/file/d/1g5rHNee5jOKH14aAucnzZd4fLmMpuJG7/view?usp=sharing): model trained with coco person dataset.

[MOT17_trained_with_CH_lite](https://drive.google.com/file/d/1HObK_SUntlD0lmO63T8K-9fLp7kstV6m/view?usp=sharing): model trained on CrowdHuman and MOT17 trainset.

[MOT20_trained_with_CH_lite](https://drive.google.com/file/d/10IUHW_k8TKTcwSdR_3dWgevDQG6OJ2mL/view?usp=sharing): model trained on CrowdHuman and MOT20 trainset.

Please put all the pretrained models to *./model_zoo* .
## Training
***For TransCenter V2***:

- Pretrained on coco person dataset:
```
cd TransCenter_official
python -m torch.distributed.launch --nproc_per_node=4 --use_env ./training/main_coco.py --output_dir=./outputs/whole_coco --batch_size=4 --num_workers=8 --pre_hm --tracking --nheads 1 2 5 8 --num_encoder_layers 3 4 6 3 --dim_feedforward_ratio 8 8 4 4 --d_model 64 128 320 512 --data_dir=YourPathTo/cocodataset/
```

- Pretrained on CrowdHuman dataset:
```
cd TransCenter_official
python -m torch.distributed.launch --nproc_per_node=4 --use_env ./training/main_crowdHuman.py --output_dir=./outputs/whole_ch_from_COCO --batch_size=4 --num_workers=8 --resume=./model_zoo/coco_pretrained.pth --pre_hm --tracking --nheads 1 2 5 8 --num_encoder_layers 3 4 6 3 --dim_feedforward_ratio 8 8 4 4 --d_model 64 128 320 512 --data_dir=YourPathTo/crowd_human/
```

- Train MOT17 from CoCo pretrained model:
```
cd TransCenter_official
python -m torch.distributed.launch --nproc_per_node=2 --use_env ./training/main_mot17.py --output_dir=./outputs/mot17_from_coco --batch_size=4 --num_workers=8 --data_dir=YourPathTo/MOT17/ --epochs=50 --lr_drop=40 --nheads 1 2 5 8 --num_encoder_layers 3 4 6 3 --dim_feedforward_ratio 8 8 4 4 --d_model 64 128 320 512 --pre_hm --tracking --resume=./model_zoo/coco_pretrained.pth --same_aug_pre --image_blur_aug --clip_max_norm=35
```

- Train MOT17 together with CrowdHuman:
```
cd TransCenter_official
python -m torch.distributed.launch --nproc_per_node=2 --use_env ./training/main_mot17_mix_ch.py --output_dir=./outputs/CH_mot17 --batch_size=4 --num_workers=8 --data_dir=YourPathTo/MOT17/  --data_dir_ch=YourPathTo/crowd_human/ --epochs=150 --lr_drop=100 --nheads 1 2 5 8 --num_encoder_layers 3 4 6 3 --dim_feedforward_ratio 8 8 4 4 --d_model 64 128 320 512 --pre_hm --tracking --same_aug_pre --image_blur_aug --clip_max_norm=35
```

- Train MOT20 from CoCo pretrained model:
```
cd TransCenter_official
python -m torch.distributed.launch --nproc_per_node=2 --use_env ./training/main_mot20.py --output_dir=./outputs/mot20_from_coco --batch_size=4 --num_workers=8 --data_dir=YourPathTo/MOT20/ --epochs=50 --lr_drop=40 --nheads 1 2 5 8 --num_encoder_layers 3 4 6 3 --dim_feedforward_ratio 8 8 4 4 --d_model 64 128 320 512 --pre_hm --tracking --resume=./model_zoo/coco_pretrained.pth --same_aug_pre --image_blur_aug --clip_max_norm=35
```

- Train MOT20 together with CrowdHuman:
```
cd TransCenter_official
python -m torch.distributed.launch --nproc_per_node=2 --use_env ./training/main_mot20_mix_ch.py --output_dir=./outputs/CH_mot20 --batch_size=4 --num_workers=8 --data_dir=YourPathTo/MOT20/  --data_dir_ch=YourPathTo/crowd_human/ --epochs=150 --lr_drop=100 --nheads 1 2 5 8 --num_encoder_layers 3 4 6 3 --dim_feedforward_ratio 8 8 4 4 --d_model 64 128 320 512 --pre_hm --tracking --same_aug_pre --image_blur_aug --clip_max_norm=35
```

***For TransCenter-Lite***:
- Pretrained on coco person dataset:
```
cd TransCenter_official
python -m torch.distributed.launch --nproc_per_node=4 --use_env ./training/main_coco_lite.py --output_dir=./outputs/whole_coco_lite --batch_size=4 --num_workers=8 --pre_hm --tracking --nheads 1 2 5 8 --num_encoder_layers 2 2 2 2 --dim_feedforward_ratio 8 8 4 4 --d_model 32 64 160 256 --num_decoder_layers 4 --data_dir=YourPathTo/cocodataset/
```

- Pretrained on CrowdHuman dataset:
```
cd TransCenter_official
python -m torch.distributed.launch --nproc_per_node=4 --use_env ./training/main_crowdHuman_lite.py --output_dir=./outputs/whole_ch_from_coco_lite --batch_size=4 --num_workers=8 --resume=./model_zoo/coco_pretrained_lite.pth --pre_hm --tracking --nheads 1 2 5 8 --num_encoder_layers 2 2 2 2 --dim_feedforward_ratio 8 8 4 4 --d_model 32 64 160 256 --num_decoder_layers 4 --data_dir=YourPathTo/crowd_human/
```
- Train MOT17 from CoCo pretrained model:
```
cd TransCenter_official
python -m torch.distributed.launch --nproc_per_node=2 --use_env ./training/main_mot17_lite.py --output_dir=./outputs/mot17_from_coco_lite --batch_size=4 --num_workers=8 --data_dir=YourPathTo/MOT17/ --epochs=50 --lr_drop=40 --nheads 1 2 5 8 --num_encoder_layers 2 2 2 2 --dim_feedforward_ratio 8 8 4 4 --d_model 32 64 160 256 --num_decoder_layers 4 --pre_hm --tracking --resume=./model_zoo/coco_pretrained_lite.pth --same_aug_pre --image_blur_aug --clip_max_norm=35
```

- Train MOT17 together with CrowdHuman:
```
cd TransCenter_official
python -m torch.distributed.launch --nproc_per_node=2 --use_env ./training/main_mot17_mix_ch_lite.py --output_dir=./outputs/CH_mot17_lite --batch_size=4 --num_workers=8 --data_dir=YourPathTo/MOT17/  --data_dir_ch=YourPathTo/crowd_human/ --epochs=150 --lr_drop=100 --nheads 1 2 5 8 --num_encoder_layers 2 2 2 2 --dim_feedforward_ratio 8 8 4 4 --d_model 32 64 160 256 --num_decoder_layers 4 --pre_hm --tracking --same_aug_pre --image_blur_aug --clip_max_norm=35
```
- Train MOT20 from CoCo pretrained model:
```
cd TransCenter_official
python -m torch.distributed.launch --nproc_per_node=2 --use_env ./training/main_mot20_lite.py --output_dir=./outputs/mot20_from_coco_lite --batch_size=4 --num_workers=8 --data_dir=YourPathTo/MOT20/ --epochs=50 --lr_drop=40 --nheads 1 2 5 8 --num_encoder_layers 2 2 2 2 --dim_feedforward_ratio 8 8 4 4 --d_model 32 64 160 256 --num_decoder_layers 4 --pre_hm --tracking --resume=./model_zoo/coco_pretrained_lite.pth --same_aug_pre --image_blur_aug --clip_max_norm=35
```

- Train MOT20 together with CrowdHuman:
```
cd TransCenter_official
python -m torch.distributed.launch --nproc_per_node=2 --use_env ./training/main_mot20_mix_ch_lite.py --output_dir=./outputs/CH_mot20_lite --batch_size=4 --num_workers=8 --data_dir=YourPathTo/MOT20/  --data_dir_ch=YourPathTo/crowd_human/ --epochs=150 --lr_drop=100 --nheads 1 2 5 8 --num_encoder_layers 2 2 2 2 --dim_feedforward_ratio 8 8 4 4 --d_model 32 64 160 256 --num_decoder_layers 4 --pre_hm --tracking --same_aug_pre --image_blur_aug --clip_max_norm=35
```

Tips:
1) If you encounter *RuntimeError: cuDNN error: CUDNN_STATUS_INTERNAL_ERROR* in some GPUs, please try to set *torch.backends.cudnn.benchmark=False*. In most of the cases, setting *torch.backends.cudnn.benchmark=True* is more memory-efficient.
2) Depending on your environment and GPUs, you might experience MOTA jitter in your final models.
3) You may see training noise during fine-tuning, especially for MOT17/MOT20 training with well-pretrained models. You can slow down the training rate by 1/10, apply early stopping, increase batch size with GPUs having more memory.  
4) If you have GPU memory issues, try to lower the batch size for training and evaluation in main_****.py, freeze the resnet backbone and use our coco/CH pretrained models.


## Tracking
###Using Private detections:

***For TransCenter V2***:

- MOT17:
```
cd TransCenter_official
python ./tracking/mot17_private_test.py --data_dir=YourPathTo/MOT17/
```
- MOT20:
```
cd TransCenter_official
python ./tracking/mot20_private_test.py --data_dir=YourPathTo/MOT20/
```
***For TransCenter-Lite***:
- MOT17:
```
cd TransCenter_official
python ./tracking/mot17_private_lite_test.py --data_dir=YourPathTo/MOT17/
```
- MOT20:
```
cd TransCenter_official
python ./tracking/mot20_private_lite_test.py --data_dir=YourPathTo/MOT20/
```

###Using Public detections:

***For TransCenter V2***:

- MOT17:
```
cd TransCenter_official
python ./tracking/mot17_pub_test.py --data_dir=YourPathTo/MOT17/
```
- MOT20:
```
cd TransCenter_official
python ./tracking/mot20_pub_test.py --data_dir=YourPathTo/MOT20/
```

***For TransCenter-Lite***:
- MOT17:
```
cd TransCenter_official
python ./tracking/mot17_pub_lite_test.py --data_dir=YourPathTo/MOT17/
```
- MOT20:
```
cd TransCenter_official
python ./tracking/mot20_pub_lite_test.py --data_dir=YourPathTo/MOT20/
```

## MOTChallenge Results
***For TransCenter V2***:

MOT17 public detections:
     
| Pretrained| MOTA     | MOTP     | IDF1 |  FP    | FN    | IDS |
|-----------|----------|----------|--------|-------|------|----------------|
|   CoCo  |  71.9%   |  80.5%   | 64.1% | 27,356  | 126,860  |     4,118     |
|   CH    |  75.9%   |  81.2%   | 65.9%  | 30,190 | 100,999 |     4,626    |

MOT20 public detections:
   
| Pretrained| MOTA     | MOTP     | IDF1 |  FP    | FN    | IDS |
|-----------|----------|----------|--------|-------|------|----------------|
|   CoCo    |  67.7%   |  79.8%   | 58.9%  | 54,967   | 108,376  |     3,707     |
|   CH      |  72.8%   |  81.0%   | 57.6%  | 28,026  | 110,312  |     2,621     |


MOT17 private detections:
   
| Pretrained| MOTA     | MOTP     | IDF1 |  FP    | FN    | IDS |
|-----------|----------|----------|--------|-------|------|----------------|
|   CoCo  |  72.7%   |  80.3%   | 64.0% | 33,807   | 115,542  |    4,719     |
|   CH    |  76.5%   |  81.1%   | 65.5% | 40,101 | 88,827 |     5,394    |

MOT20 private detections:

| Pretrained| MOTA     | MOTP     | IDF1 |  FP    | FN    | IDS |
|-----------|----------|----------|--------|-------|------|----------------|
|   CoCo   |  67.7%   |  79.8%   | 58.7% | 56,435  | 107,163 |     3,759     |
|   CH   |  72.9%   |  81.0%   | 57.7%  | 28,596  | 108,982  |     2,625     |


**Note:** 
- The results can be slightly different depending on the running environment.
- We might keep updating the results in the near future.

## Acknowledgement

The code for TransCenterV2, TransCenter-Lite is modified and network pre-trained weights are obtained from the following repositories:

1) The PVTv2 backbone pretrained models from PVTv2.
2) The data format conversion code is modified from CenterTrack.

[**CenterTrack**](https://github.com/xingyizhou/CenterTrack), [**Deformable-DETR**](https://github.com/fundamentalvision/Deformable-DETR), [**Tracktor**](https://github.com/phil-bergmann/tracking_wo_bnw).
```
@article{zhou2020tracking,
  title={Tracking Objects as Points},
  author={Zhou, Xingyi and Koltun, Vladlen and Kr{\"a}henb{\"u}hl, Philipp},
  journal={ECCV},
  year={2020}
}

@InProceedings{tracktor_2019_ICCV,
author = {Bergmann, Philipp and Meinhardt, Tim and Leal{-}Taix{\'{e}}, Laura},
title = {Tracking Without Bells and Whistles},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {October},
year = {2019}}

@article{zhu2020deformable,
  title={Deformable DETR: Deformable Transformers for End-to-End Object Detection},
  author={Zhu, Xizhou and Su, Weijie and Lu, Lewei and Li, Bin and Wang, Xiaogang and Dai, Jifeng},
  journal={arXiv preprint arXiv:2010.04159},
  year={2020}
}

@article{zhang2021bytetrack,
  title={ByteTrack: Multi-Object Tracking by Associating Every Detection Box},
  author={Zhang, Yifu and Sun, Peize and Jiang, Yi and Yu, Dongdong and Yuan, Zehuan and Luo, Ping and Liu, Wenyu and Wang, Xinggang},
  journal={arXiv preprint arXiv:2110.06864},
  year={2021}
}

@article{wang2021pvtv2,
  title={Pvtv2: Improved baselines with pyramid vision transformer},
  author={Wang, Wenhai and Xie, Enze and Li, Xiang and Fan, Deng-Ping and Song, Kaitao and Liang, Ding and Lu, Tong and Luo, Ping and Shao, Ling},
  journal={Computational Visual Media},
  volume={8},
  number={3},
  pages={1--10},
  year={2022},
  publisher={Springer}
}
```
Several modules are from:

**MOT Metrics in Python**: [**py-motmetrics**](https://github.com/cheind/py-motmetrics)

**Soft-NMS**: [**Soft-NMS**](https://github.com/DocF/Soft-NMS)

**DETR**: [**DETR**](https://github.com/facebookresearch/detr)

**DCNv2**: [**DCNv2**](https://github.com/CharlesShang/DCNv2)

**PVTv2**: [**PVTv2**](https://github.com/whai362/PVT)

**ByteTrack**: [**ByteTrack**](https://github.com/ifzhang/ByteTrack)

