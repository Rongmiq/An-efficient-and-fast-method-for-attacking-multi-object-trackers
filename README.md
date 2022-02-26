# EAAMOT: Efficient Adversarial Attacks for Multiple Object Tracking.
An efficient and fast attack multi-object tracker method.

<img src="images/MOT-03-III.gif" width="500" hight='250'/>   <img src="images/MOT17-03-IDI.gif" width="500" hight='250'/> 

## Abstract:
Multiple Object Tracking (MOT) is an essential but challenging task in computer vision. In recent years, most state-of-the-art trackers have formulated the MOT task as two sub-problem: object detection and data association. They have achieved the top performance on the most popular benchmarks with the help of excellent detectors and robust ReID modules. However, whether MOT is as vulnerable as DNNs remains to be studied. To explore the robustness of detector and ReID modules in MOT, we propose an effective adversarial attack method to fool them. A series of detection and ReID losses is presented for training disturbance generators. Specifically, we build a vanishing and shrinkage loss to cool the heatmap of the detector and shrink the detection box. For ReID models, we propose a ReID loss to blind and confuse it so that the tracker cannot locate different targets correctly in the videos. TraDeS is the primary tracker of the white-box attack in this work. Further, we transfer our method as a black-box attack method to other trackers, including CenterTrack, FairMOT, and ByteTrack. Extensive experiments on COCO, MOT16, MOT17, and MOT20 datasets show that our method is effective and universal.

# Attacking performance
Visualization results on MOT17 val set. ‘-AD’ and ‘-AR’ represent attack detector and ReID module respectively. 

| Method      | MOTA  | MOTA | MOTA   | MOTA    | IDF1  |  IDF1|  IDF1  |  IDF1  | IDs   | IDs |  IDs   | IDs    |
|-------------|-------|------|--------|---------|-------|------|--------|--------|-------|-----|--------|--------|
|             | Clean | AD   | AR+III | AR+IDI  | Clean | AD   | AR+III | AR+IDI | Clean | AD  | AR+III | AR+IDI |
| TraDeS      | 68.2  | 45.4 | 4.7    | -399.2  | 71.7  | 53.6 | 7      | 4.3    | 285   | 338 | 315    | 1393   |
| CenterTrack | 66.1  | 41.8 | 8.7    | -236.7  | 64.2  | 43.3 | 10.4   | 7.4    | 528   | 797 | 436    | 920    |
| FairMOT     | 69.1  | 58.6 | 31.3   | 31      | 72.8  | 50.3 | 35     | 33.4   | 299   | 376 | 470    | 1130   |
| ByteTrack   | 76.6  | 75.4 | 73.6   | -1252.6 | 79.3  | 75.5 | 73.3   | 7.3    | 159   | 187 | 198    | 543    |

Visualization results on MOT17 test set. ‘-AD’ and ‘-AR’ represent attack detector and ReID module respectively. 
| Method                                               | MOTA   | IDF1 | HOTA | MT    | ML    | IDs  |
|------------------------------------------------------|--------|------|------|-------|-------|------|
| TraDeS                                               | 69.1   | 63.9 | 52.7 | 36.4% | 21.5% | 3555 |
| TraDeS-AD                                            | 49.4   | 46.9 | 38.9 | 12.5% | 48.7% | 4095 |
| TraDeS-AR+III                                        | 3.0    | 4.2  | 6.4  | 0.1%  | 95.9% | 681  |
| TraDeS-AR+IDI                                        | -238.1 | 4.4  | 6.2  | 0%    | 77.7% | 5831 |
| FairMOT                                              | 73.7   | 72.3 | 59.3 | 43.2% | 17.3% | 3303 |
| FairMOT-AR+III                                       | 31.4   | 33.3 | 27.2 | 5.0%  | 62.9% | 3057 |
| FairMOT-AR+IDI                                       | 19.8   | 23.3 | 19.5 | 1.4%  | 69.0% | 3120 |
| ByteTrack                                            | 80.3   | 77.3 | 63.1 | 53.2% | 14.5% | 2196 |
| ByteTrack-AR+III                                     | 75.6   | 72.8 | 58.3 | 41.5% | 19.9% | 2205 |
| ByteTrack-AR+IDI                                     | -357.0 | 17.2 | 20.9 | 26.7% | 36.7% | 4831 |

## Installation
### 1. Installing on the host machine
Step1. Install dependencies. We use python 3.7 and pytorch 1.3.1 (it also work on pytorch 1.7.0)
```
conda create -n EAAMOT python=3.7
conda activate EAAMOT
conda install pytorch==1.3.1 torchvision==0.4.2 cudatoolkit=11.3 -c pytorch
```

Step2. Install EAAMOT.
```shell
git clone https://github.com/Rongmiq/EAAMOT.git
cd EAAMOT
pip install -r requirements.txt
```
Then, install [TraDeS](https://github.com/JialianW/TraDeS), [CenterTrack](https://github.com/xingyizhou/CenterTrack), [FairMOT](https://github.com/ifzhang/FairMOT), and [ByteTrack](https://github.com/ifzhang/ByteTrack) which you want attack. ByteTrack may be not exist in the same environment as TraDeS/FairMOT/CenterTrack. 

After that, you should set PYTHONPATH as follow to avoid environment errors.
```
export PYTHONPATH={EAAMOT_root}/EAAMO:$PYTHONPATH
export PYTHONPATH={EAAMOT_root}/EAAMO/pix2pix:$PYTHONPATH
export PYTHONPATH={EAAMOT_root}/EAAMO/TraDeS:$PYTHONPATH
export PYTHONPATH={EAAMOT_root}/EAAMO/FairMOT:$PYTHONPATH (optional)
export PYTHONPATH={EAAMOT_root}/EAAMO/CenterTrack:$PYTHONPATH (optional)
export PYTHONPATH={EAAMOT_root}/EAAMO/ByteTrack:$PYTHONPATH (optional)
```

Step3. Install COCOAPI:
```
pip install cython; pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

Step3. Install DCNv2: (you should choose one of them and we choose the first)

1. For [TraDeS](https://github.com/JialianW/TraDeS/blob/master/readme/INSTALL.md) 
Compile deformable convolutional (Successuflly compiled with both gcc v5.4.0 and v8.4.0. gcc version should be higher than v4.8). 
```
cd $EAAMOT_root/TraDeS/src/lib/model/networks/DCNv2
. make.sh
```
2. or for [FairMOT](https://github.com/ifzhang/FairMOT/edit/master/README.md)

FairMOT use [DCNv2_pytorch_1.7](https://github.com/ifzhang/DCNv2/tree/pytorch_1.7) in its backbone network (pytorch_1.7 branch). Previous versions can be found in [DCNv2](https://github.com/CharlesShang/DCNv2).
```
git clone -b pytorch_1.7 https://github.com/ifzhang/DCNv2.git
cd DCNv2
./make.sh
```
pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

## Data preparation
Download [MOT16](https://motchallenge.net/), [MOT17](https://motchallenge.net/), [MOT20](https://motchallenge.net/), [CrowdHuman](https://www.crowdhuman.org/), [COCO2017](https://https://cocodataset.org/#download), and put them under <Your_dir>/datasets in the following structure:
```
MOT16/17/20
   |——————annotations
   |        └——————test.sjon
   └——————train
   |         └——————MOT-xx-xx
   |                  └——————det
   |                          └——————det.txt
   |                  └——————gt
   |                          └——————gt.txt
   |                  └——————img1
   |                         └——————000001.jpg
   |                         └——————...
   |                  └——————seqinfo.ini
   └——————test
            └——————MOT-xx-xx
                     └——————det
                             └——————det.txt
                     └——————gt (empty)
                     └——————img1
                             └——————000001.jpg
                             └——————...
                     └——————seqinfo.ini

Crowdhuman
   |——————CrowdHuman_train
   |        └——————xxxxxx1.jpg
   |        └——————xxxxxx2.jpg
   └——————CrowdHuman_val
   |        └——————xxxxxx1.jpg
   |        └——————xxxxxx2.jpg
   |——————annotation_train.odgt
   |——————annotation_val.odgt
   |——————train.json
   |——————val.json
   
COCO
   |——————annotations
   |         └——————....sjon
   └——————train2017
   |         └——————xxxxxx.jpg
   └——————test2017
   |         └——————xxxxxx.jpg
   └——————val2017
             └——————xxxxxx.jpg

```
Mix data and formate convert please see [TraDeS](https://github.com/JialianW/TraDeS), [CenterTrack](https://github.com/xingyizhou/CenterTrack), [FairMOT](https://github.com/ifzhang/FairMOT), and [ByteTrack](https://github.com/ifzhang/ByteTrack). After that, you will get .json files for training and testing trackers. 

Then, you should mix Crowdhuman and MOT17 for training detection loss and ReID loss.
```
# detection loss (Crowdhuman and MOT17)
python tools/mix_crownhuman_mot17.py

# ReID loss (300 frames in MOT17 train set only)
python tools/mix_mot17_half_train.py
```

## Model zoo
we train it on crownhuman_mot17/mot_half_train and attacking CenterNet/TradeS.
| Model       |   | MOTA   | IDF1 | AP  |
|-------------|---|--------|------|-----|
| AD          |   | 45.4   | 53.6 | 0.7 |
| AR+III      |   | 4.7    | 7    | \   |
| AR+IDI+500  |   | -399.2 | 4.3  | \   |
| AR+IDI+1000 |   | -376.4 | 7.8  | \   |
| AR+IDI+1500 |   | -105.9 | 31.6 | \   |

## Training
Before training, you should choose the model you want to train. Specifically, modify the 
```
'--name', '--model' in  {EAAMOT_root}/EAAMOT/pix2pix/options/base_options.py, 
```
and
```
'GAN_opt.name', 'GAN_opt.model' in {EAAMOT_root}/EAAMOT/pix2pix/options/GAN_utils.py, 
```                                       
make sure it in {models_dir} and correct.
```
{EAAMOT_root}/EAAMOT/pix2pix/models
```
Other parameters used in training (e.g. epoch, devices, batch_size can be found in base_options.py and train_options.py)

Then, you can start the training by
```
cd {EAAMOT_root}
python3 train.py
```
If you want to train other models, you can change the '--name' and '--model' in base_option.py and GAN_utils.py. And create a {your_models}.py as G_reid2_L2_500_model.py in pix2pix/models.

## Testing
You should choose the model you want to test firstly, the details are same as training.
How to run a tracker, please refer [TraDeS](https://github.com/JialianW/TraDeS), [CenterTrack](https://github.com/xingyizhou/CenterTrack), [FairMOT](https://github.com/ifzhang/FairMOT), and [ByteTrack](https://github.com/ifzhang/ByteTrack). 
To avoid parameter collisions, the tracker parameters need to be written to the ```{EAAMOT_root}/TraDeS/src/lib/opts.py``` first. 

Here is an example to attack TraDeS:
```
cd {EAAMOT_root}/TraDeS/src/
python test.py --adv True
```
## Demo
If you want to visualize the results. You should set as testing and run
```
cd {EAAMOT_root}/TraDeS/src/
python demot.py --adv True
```

## Citation
```
Ours
```
```
@inproceedings{Wu2021TraDeS,
title={Track to Detect and Segment: An Online Multi-Object Tracker},
author={Wu, Jialian and Cao, Jiale and Song, Liangchen and Wang, Yu and Yang, Ming and Yuan, Junsong},
booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
year={2021}}

@article{zhou2020tracking,
  title={Tracking Objects as Points},
  author={Zhou, Xingyi and Koltun, Vladlen and Kr{\"a}henb{\"u}hl, Philipp},
  journal={ECCV},
  year={2020}
}

@article{zhang2021fairmot,
  title={Fairmot: On the fairness of detection and re-identification in multiple object tracking},
  author={Zhang, Yifu and Wang, Chunyu and Wang, Xinggang and Zeng, Wenjun and Liu, Wenyu},
  journal={International Journal of Computer Vision},
  volume={129},
  pages={3069--3087},
  year={2021},
  publisher={Springer}
}

@article{zhang2021bytetrack,
  title={ByteTrack: Multi-Object Tracking by Associating Every Detection Box},
  author={Zhang, Yifu and Sun, Peize and Jiang, Yi and Yu, Dongdong and Yuan, Zehuan and Luo, Ping and Liu, Wenyu and Wang, Xinggang},
  journal={arXiv preprint arXiv:2110.06864},
  year={2021}
}

@inproceedings{zhou2019objects,
  title={Objects as Points},
  author={Zhou, Xingyi and Wang, Dequan and Kr{\"a}henb{\"u}hl, Philipp},
  booktitle={arXiv preprint arXiv:1904.07850},
  year={2019}
}
```

## Acknowledgement
A large part of the code is borrowed from  [TraDeS](https://github.com/JialianW/TraDeS), [CenterTrack](https://github.com/xingyizhou/CenterTrack), [FairMOT](https://github.com/ifzhang/FairMOT), and [ByteTrack](https://github.com/ifzhang/ByteTrack), [CenterNet](https://github.com/xingyizhou/CenterNet), and [pix2pix](https://github.com/phillipi/pix2pix). Thanks for their wonderful works.



