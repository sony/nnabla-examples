# Copyright 2021 Sony Corporation.
# Copyright 2021 Sony Group Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import json

import pycocotools.coco as coco
from utils.reval import from_dets

from datasets.dataset.dataset_config import DatasetConfig


class PascalVOCDefaultParams():
    mean = np.array([0.485, 0.456, 0.406],
                    dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225],
                   dtype=np.float32)
    num_classes = 20
    default_resolution = [384, 384]
    max_objs = 128
    train_size = 16550
    eval_size = 4952

    class_name = [
        '__background__', "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
        "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
        "train", "tvmonitor"]
    _valid_ids = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    cat_ids = {v: i for i, v in enumerate(_valid_ids)}
    _eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                        dtype=np.float32)
    _eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)


class PascalVOC(DatasetConfig):

    def __init__(self, opt, split, mixed_precision=False, channel_last=False, shuffle=False, rng=None):
        super(PascalVOC, self).__init__(mixed_precision=mixed_precision,
                                        channel_last=channel_last, shuffle=shuffle, rng=rng)
        self.data_dir = os.path.join(opt.data_dir, 'voc')
        self.img_dir = os.path.join(self.data_dir, 'images')
        _ann_name = {
            'train': 'trainval0712',
            'val': 'test2007'
        }
        self.annot_path = os.path.join(
            self.data_dir, 'annotations',
            'pascal_{}.json').format(_ann_name[split])
        self.params = PascalVOCDefaultParams()
        self.split = split
        self.opt = opt
        self.coco = coco.COCO(self.annot_path)
        self.images = self.coco.getImgIds()
        self._size = len(self.images)
        self._variables = ('img', 'hm', 'ind', 'wh', 'reg', 'reg_mask', 'cls')

        print('==> initializing pascal {} data.'.format(_ann_name[split]))
        print('Loaded {} {} samples'.format(split, self._size))
        if rng is None:
            rng = np.random.RandomState(313)
        self.rng = rng
        self.reset()

    def convert_eval_format(self, all_bboxes):
        detections = [[[] for __ in range(self._size)]
                      for _ in range(self.params.num_classes + 1)]
        for i in range(self._size):
            img_id = self.images[i]
            for j in range(1, self.params.num_classes + 1):
                if isinstance(all_bboxes[img_id][j], np.ndarray):
                    detections[j][i] = all_bboxes[img_id][j].tolist()
                else:
                    detections[j][i] = all_bboxes[img_id][j]
        return detections

    def save_results(self, results, save_dir):
        json.dump(self.convert_eval_format(results),
                  open(os.path.join(save_dir, 'results.json'), 'w'))

    def run_eval(self, results, save_dir, data_dir):
        self.save_results(results, save_dir)
        avg_map = from_dets(os.path.join(save_dir, 'results.json'), data_dir)
        return avg_map
