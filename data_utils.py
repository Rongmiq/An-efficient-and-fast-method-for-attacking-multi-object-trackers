from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import openpyxl as op
import json
import os
import cv2
import torch
from collections import defaultdict
from torch.utils.data.dataset import Dataset
from TraDeS.src.lib.dataset.generic_dataset import GenericDataset


def pre_process_normalize_np(images):
    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                     dtype=np.float32).reshape(1, 1, 3)
    std  = np.array([0.28863828, 0.27408164, 0.27809835],
                     dtype=np.float32).reshape(1, 1, 3)
    images_normalize = ((images / 255. - mean) / std).float()

    return images_normalize

def post_process_normalize_np(images):
    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                     dtype=np.float32).reshape(1, 3, 1, 1)
    std  = np.array([0.28863828, 0.27408164, 0.27809835],
                     dtype=np.float32).reshape(1, 3, 1, 1)
    images_post_normalize = (images * std + mean) * 255.

    return images_post_normalize

def pre_process_normalize(images):
    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                     dtype=np.float32).reshape(1, 3, 1, 1)
    std  = np.array([0.28863828, 0.27408164, 0.27809835],
                     dtype=np.float32).reshape(1, 3, 1, 1)
    mean_tensor = torch.from_numpy(mean).cuda()
    std_tensor = torch.from_numpy(std).cuda()
    images_normalize = ((images / 255. - mean_tensor) / std_tensor).float()
    images_normalize = images_normalize.to()
    return images_normalize

def pre_process_normalize_bytetrack(images):
    mean = np.array([0.485, 0.456, 0.406],
                     dtype=np.float32).reshape(1, 3, 1, 1)
    std  = np.array([0.229, 0.224, 0.225],
                     dtype=np.float32).reshape(1, 3, 1, 1)
    mean_tensor = torch.from_numpy(mean).cuda()
    std_tensor = torch.from_numpy(std).cuda()
    images_normalize = ((images / 255. - mean_tensor) / std_tensor).float()

    return images_normalize


def post_process_normalize(images):
    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                     dtype=np.float32).reshape(1, 3, 1, 1)
    std  = np.array([0.28863828, 0.27408164, 0.27809835],
                     dtype=np.float32).reshape(1, 3, 1, 1)
    mean_tensor = torch.from_numpy(mean).cuda()
    std_tensor = torch.from_numpy(std).cuda()

    images_post_normalize = (images * std_tensor + mean_tensor) * 255.

    return images_post_normalize

def post_process_normalize_bytetrack(images):
    mean = np.array([0.485, 0.456, 0.406],
                     dtype=np.float32).reshape(1, 3, 1, 1)
    std  = np.array([0.229, 0.224, 0.225],
                     dtype=np.float32).reshape(1, 3, 1, 1)
    mean_tensor = torch.from_numpy(mean).cuda()
    std_tensor = torch.from_numpy(std).cuda()

    images_post_normalize = (images * std_tensor + mean_tensor) * 255.

    return images_post_normalize

def normalize(image_tensor):
    l = [-1.41317548, -1.63160516, -1.6909886]
    r = [2.05136844, 2.01694276, 1.90486153]
    for i in range(3):
        image_tensor[:,i,:,:] = torch.clamp(image_tensor[:,i,:,:], min=l[i], max=r[i])
    return image_tensor

def post_process_resize(images):
    batch, channel, height, width = images.shape[0:4]
    zeros_pad = torch.zeros(batch, channel, height+32, width+32).float().cuda()
    zeros_pad[:, :, 16:528, 16:528] = images

    return zeros_pad

class PrefetchDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.images = dataset.images
        self.load_image_func = dataset.coco.loadImgs
        self.img_dir = dataset.img_dir

    def __getitem__(self, index):
        img_id = self.images[index]
        img_info = self.load_image_func(ids=[img_id])[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        if not os.path.exists(img_path):
            raise ValueError
        image = cv2.imread(img_path)
        # image = self.resize_image(image,target_size=(512,256))  # for COCO dataset
        # image = self.crop_pad_image(image, target_size=(1088,608))  # for FairMOT and MOT dataset
        image = self.resize_image(image, target_size=(960,544))  #for CenterTracking/TraDeS and MOT dataset
        # image = self.resize_image(image, target_size=(1440,800))  #for ByteTrack/CrowdHuman and MOT dataset
        # image = self.crop_pad_image(image, target_size=(1152,640))  #for ByteTrack/CrowdHuman and MOT dataset
        # image = self.resize_image(image, target_size=(1024,512))  #for CrowdHuman centernet
        return image


    def __len__(self):
        return len(self.images)


    def crop_pad_image(self, image, target_size=(512,512)):
        h, w, c = image.shape[:]
        new_wh = max(h, w)
        zeros = np.zeros([new_wh, new_wh, c])
        pad = int((new_wh - min(h, w)) / 2)
        if h <= w:
            zeros[pad:pad + h, :, :] = image
        else:
            zeros[:, pad:pad + w, :] = image
        images = cv2.resize(zeros, target_size)
        return  images

    def resize_image(self, image, target_size=(512,512)):
        image = cv2.resize(image, target_size)
        return  image

    def resize_pad_image(self, image, target_size=(960,540)):
        image = cv2.resize(image, target_size)
        pad = np.zeros([2,960,3])
        image_pad = np.concatenate((pad,image),axis=0)
        image_pad = np.concatenate((image_pad, pad),axis=0)
        return  image_pad

class COCO(torch.utils.data.Dataset):
    num_classes = 80
    default_resolution = [512, 512]
    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)

    def __init__(self, split):
        super(COCO, self).__init__()
        # if split == 'train_half':
        #     self.data_dir = os.path.join(opt.data_dir, 'MOT17')
        #     self.img_dir = os.path.join(self.data_dir, 'train')
        # elif split == 'val_half':
        #     self.data_dir = os.path.join(opt.data_dir, 'MOT17')
        #     self.img_dir = os.path.join(self.data_dir, 'val')
        # else:
        #     self.data_dir = os.path.join(opt.data_dir, 'coco')
        #     self.img_dir = os.path.join(self.data_dir, '{}2017'.format(split))
        # if split == 'test':
        #     self.annot_path = os.path.join(
        #         self.data_dir, 'annotations',
        #         'image_info_test-dev2017.json').format(split)
        # elif split == 'train_half':
        #     self.annot_path = os.path.join('/home/marq/Desktop/datasets/MOT17'
        #                                    , 'annotations',
        #                                    '{}.json').format(split)
        # elif split == 'val_half':
        #     self.annot_path = os.path.join('/home/marq/Desktop/datasets/MOT17'
        #                                    , 'annotations',
        #                                    '{}.json').format(split)
        if split == 'mix':
            self.img_dir = os.path.join('/home/marq/Desktop/datasets')
            self.annot_path = os.path.join('/home/marq/Desktop/datasets/mix_det/annotations/train.json')
        elif split == 'coco':
            self.img_dir = os.path.join('/home/marq/Desktop/datasets/coco/train2017')
            self.annot_path = os.path.join('/home/marq/Desktop/datasets/coco/annotations/instances_train2017.json')
        elif split == 'train_mot17':
            self.img_dir = os.path.join('/home/marq/Desktop/datasets/MOT17/train')
            # self.annot_path = os.path.join('/home/marq/Desktop/datasets/MOT17/annotations/train_half.json')
            # self.annot_path = os.path.join('/home/marq/Desktop/datasets/MOT17/annotations/train_mot17_2.json')
            self.annot_path = os.path.join('/home/marq/Desktop/datasets/MOT17/annotations/train_300_mot17.json')
        elif split == 'train_mot20':
            self.img_dir = os.path.join('/home/marq/Desktop/datasets/MOT20/train')
            self.annot_path = os.path.join('/home/marq/Desktop/datasets/MOT20/annotations/train_mot20.json')
        # else:
        #     if opt.task == 'exdet':
        #         self.annot_path = os.path.join(
        #             self.data_dir, 'annotations',
        #             'instances_extreme_{}2017.json').format(split)
        #     else:
        #         self.annot_path = os.path.join(
        #             self.data_dir, 'annotations',
        #             'instances_{}2017.json').format(split)
        self.max_objs = 128
        self.class_name = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
            'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
            'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
            'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
            'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
            'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        self._valid_ids = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
            14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
            37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
            48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
            58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
            72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
            82, 84, 85, 86, 87, 88, 89, 90]
        self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
        self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) \
                          for v in range(1, self.num_classes + 1)]
        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                 dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)
        # self.mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
        # self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

        self.split = split
        if 'train_half' == split:
            print('==> initializing MOT17 {} data.'.format(split))
        elif 'mix' == split:
            print('==> initializing CrowdHuman and MOT17 data.')
        elif 'coco' == split:
            print('==> initializing coco data.')
        self.coco = coco.COCO(self.annot_path)
        self.images = self.coco.getImgIds()
        self.num_samples = len(self.images)

        print('Loaded {} {} samples'.format(split, self.num_samples))

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_eval_format(self, all_bboxes):
        # import pdb; pdb.set_trace()
        detections = []
        for image_id in all_bboxes:
            for cls_ind in all_bboxes[image_id]:
                category_id = self._valid_ids[cls_ind - 1]
                for bbox in all_bboxes[image_id][cls_ind]:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                    score = bbox[4]
                    bbox_out = list(map(self._to_float, bbox[0:4]))

                    detection = {
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "bbox": bbox_out,
                        "score": float("{:.2f}".format(score))
                    }
                    if len(bbox) > 5:
                        extreme_points = list(map(self._to_float, bbox[5:13]))
                        detection["extreme_points"] = extreme_points
                    detections.append(detection)
        return detections

    def __len__(self):
        return self.num_samples

    def save_results(self, results, save_dir):
        json.dump(self.convert_eval_format(results),
                  open('{}/results.json'.format(save_dir), 'w'))

    def run_eval(self, results, save_dir):
        # result_json = os.path.join(save_dir, "results.json")
        # detections  = self.convert_eval_format(results)
        # json.dump(detections, open(result_json, "w"))
        self.save_results(results, save_dir)
        coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
        coco_eval = COCOeval(self.coco, coco_dets, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()


