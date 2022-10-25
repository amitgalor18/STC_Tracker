## Strong-TransCenter: Improved Multi-Object Tracking based on Transformers with Dense Representations <br />

**[[Paper](https://arxiv.org/abs/TBD)]** <br />

## Results on MOT17 and MOT20 compared to transformer-based trackers:
<div align="center">
  <img src="https://github.com/amitgalor18/STC_Tracker/raw/main/dubblebubble.png" width="1000px" />
</div>
(as of October 2022)

## Algorithm flowchart:
<div align="center">
  <img src="https://github.com/amitgalor18/STC_Tracker/raw/main/flowchart_v5.PNG" width="1200px" />
</div>

## Bibtex
**If you find this code useful, please star the project and consider citing:** <br />
```
@misc{

}
```

## Results examples:

https://user-images.githubusercontent.com/46008959/197331505-b6ffc6eb-ad4d-4ea5-8f7b-f67891c0b38c.mp4

https://user-images.githubusercontent.com/46008959/197331524-700992c2-af2c-4993-9c3d-a8af84e02dc0.mp4

## Environment Preparation 


1) we use anaconda to simplify the package installations, you can download anaconda (4.10.3) here: [https://www.anaconda.com/products/individual](https://www.anaconda.com/products/individual)
2) you can create your conda env by doing 
```
conda env create -n <env_name> -f eTransCenter.yml
```
Alternatively, you can use the added 'requirements.txt':
```
pip install requirements.txt
```
*Make sure to install the correct torch and torchvision versions matching your CUDA version from the pytorch website: [https://pytorch.org/get-started/previous-versions/]

3) STC uses Deformable transformer from Deformable DETR. Therefore, we need to install deformable attention modules:
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


[MOT17 coco format labels](https://drive.google.com/drive/folders/1SxaVF4KddLp7t_twF53wpOifrLNzDNXE?usp=sharing): please put the *annotations* and *annotations_onlySDP* folders inside *MOT17* to your MOT17 dataset root folder.

[MOT20 coco format labels](https://drive.google.com/drive/folders/12svjv5V7-pC2BHJfyxfo_9wYcEUWGs27?usp=sharing): please put the *annotations* folder inside *MOT20* to your MOT20 dataset root folder.


## Model Zoo
***For main TransCenterV2 transformer model***:

[PVTv2 pretrained](https://drive.google.com/file/d/17K7qfXXxwsM-afxaIV_4YaATLUNd7iWM/view?usp=sharing): pretrained model from deformable-DETR.

[MOT17_trained_with_CH](https://drive.google.com/file/d/1dVUqokxvBWhzvnI6URT0_bZn9tfJ4Oed/view?usp=sharing): model trained on CrowdHuman and MOT17 trainset.

[MOT20_trained_with_CH](https://drive.google.com/file/d/1Aef8AOUz4M__H_5aVrCElJPPlkwDIjnU/view?usp=sharing): model trained on CrowdHuman and MOT20 trainset.

***For embedding network fastReID model***:

[MOT17_SBS_S50](https://drive.google.com/file/d/1V8BSF9mzgMTAMijVEpCkGQxEep4lH0AA/view?usp=sharing): model trained on mot17 train set

[MOT20_SBS_S50](https://drive.google.com/file/d/1vccLHckIYn6vrcOZY6IFCfL-bzyOKfsw/view?usp=sharing): model trained on mot20 train set


Please put all the pretrained models to *./model_zoo*, and the PVTv2 model in *.model_zoo/pvtv2_backbone*

***For Training, see the original TransCenterV2 for instructions***: [TransCenter](https://github.com/yihongXU/TransCenter)



## Tracking
###Using Private detections:

- MOT17:
```
cd STC_Tracker
python ./tracking/mot17_private_test.py --data_dir=YourPathTo/MOT17/
```
- MOT20:
```
cd STC_Tracker
python ./tracking/mot20_private_test.py --data_dir=YourPathTo/MOT20/
```


###Using Public detections:


- MOT17:
```
cd STC_Tracker
python ./tracking/mot17_pub_test.py --data_dir=YourPathTo/MOT17/
```
- MOT20:
```
cd STC_Tracker
python ./tracking/mot20_pub_test.py --data_dir=YourPathTo/MOT20/
```

You may also run the inference on a single file from the dataset, e.g.:
```
python ./tracking/mot17_private.py --data_dir YourPathTo/MOT17/ --output_dir mot17_results_dir_name --custom MOT17-02-SDP
```


## MOTChallenge Results

MOT17 public detections:
     
| Tracker             |HOTA     |MOTA     | MOTP     | IDF1 |  FP    | FN    | IDSW |
|--------------------|----------|---------|---------|-------|------|---------|-------|
|   TransCenterV2    | 56.7%    | 75.9%   |  81.2%   | 66.0%  | 30,220 | 100,995 |     4,622    |
|   STC_Tracker      | 59.5%    | 75.8%   |  81.3%   | 70.8%  | 33,833 | 99,074 |     3,787    |

MOT20 public detections:
   
| Tracker            | HOTA     |MOTA     | MOTP     | IDF1 |  FP    | FN    | IDSW |
|--------------------|----------|----------|--------|-------|------|---------|-------|
|   TransCenterV2    |   50.1%  | 72.8%   |  81.0%   | 57.6%  | 28,012  | 110,274  |     2,620     |
|   STC_Tracker      |   56.1%  | 73.0%   |  80.9%   | 67.6%  | 30,880  | 106,876  |     2,172     |


MOT17 private detections:
   
| Tracker            | HOTA      |MOTA     | MOTP     | IDF1 |  FP    | FN    | IDSW |
|--------------------|----------|----------|---------|-------|-------|--------|------|
|   TransCenterV2    |  56.7%   | 76.2%   |  81.1%   | 65.5% | 40,107 | 88,827 |     5,397    |
|   STC_Tracker      |  59.8%   | 75.8%   |  81.1%   | 70.9% | 44,952 | 87,039 |     4,533    |

MOT20 private detections:

| Tracker           | HOTA     | MOTA     | MOTP     | IDF1 |  FP    | FN    | IDSW |
|-------------------|----------|----------|---------|-------|-------|--------|-------|
|   TransCenterV2   |  50.2%   | 72.9%   |  81.0%   | 57.8%  | 28,588  | 108,950  |     2,620     |
|   STC_Tracker     |  56.3%   | 73.0%   |  81.0%   | 67.5% | 30,215 | 107,701 |     2,011    |


**Note:**
- Results from the original TransCenterV2 code were submitted independently for comparison on the same environment, using the same model version, pretrained on CrowdHuman
- The results can be slightly different depending on the running environment.
- We might keep updating the results in the near future.

## Acknowledgement

The code for STC Tracker is modified and network pre-trained weights are obtained from the following repositories:

1) The main framework code is derived from TransCenter
2) The PVTv2 backbone pretrained models from PVTv2.
3) The data format conversion code is modified from CenterTrack.
4) The Kalman filter implementation is modified from StrongSORT
5) The embedding network code is modified from fastReID
6) The embedding network trained network and association implementation is from BoT-SORT

[**TransCenter**](https://github.com/yihongXU/TransCenter), [**CenterTrack**](https://github.com/xingyizhou/CenterTrack), [**Deformable-DETR**](https://github.com/fundamentalvision/Deformable-DETR), [**Tracktor**](https://github.com/phil-bergmann/tracking_wo_bnw), [**BoT-SORT**](https://github.com/NirAharon/BoT-SORT), [**FastReID**](https://github.com/JDAI-CV/fast-reid), [**StrongSORT**](https://github.com/dyhBUPT/StrongSORT).
```
@article{xu2021transcenter,
      title={TransCenter: Transformers with Dense Representations for Multiple-Object Tracking}, 
      author={Yihong Xu and Yutong Ban and Guillaume Delorme and Chuang Gan and Daniela Rus and Xavier Alameda-Pineda},
      year={2021},
      eprint={2103.15145},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@article{Aharon2022,
   author = {Nir Aharon and Roy Orfaig and Ben-Zion Bobrovsky},
   journal = {arXiv},
   month = {6},
   title = {BoT-SORT: Robust Associations Multi-Pedestrian Tracking},
   year = {2022},
}
@article{FastReID,
   author = {Lingxiao He and Xingyu Liao and Wu Liu and Xinchen Liu and Peng Cheng and Tao Mei},
   journal = {arXiv},
   month = {6},
   title = {FastReID: A Pytorch Toolbox for General Instance Re-identification},
   year = {2020},
}
@article{StrongSORT,
   author = {Yunhao Du and Yang Song and Bo Yang and Yanyun Zhao},
   journal = {arXiv},
   month = {2},
   title = {StrongSORT: Make DeepSORT Great Again},
   year = {2022},
}

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

