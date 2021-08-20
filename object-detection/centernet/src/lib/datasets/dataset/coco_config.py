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
from pycocotools.cocoeval import COCOeval

from datasets.dataset.dataset_config import DatasetConfig


class COCODefaultParams():
    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                    dtype=np.float32)
    std = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32)
    num_classes = 80
    default_resolution = [512, 512]
    max_objs = 128
    train_size = 118287
    eval_size = 5000

    class_name = [
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
    _valid_ids = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
            14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
            37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
            48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
            58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
            72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
            82, 84, 85, 86, 87, 88, 89, 90]
    cat_ids = {v: i for i, v in enumerate(_valid_ids)}
    voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32)
                 for v in range(1, num_classes + 1)]
    _eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                        dtype=np.float32)
    _eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)


class COCO(DatasetConfig):

    def __init__(self, opt, split, shuffle=False, rng=None, mixed_precision=False, channel_last=False):
        super(COCO, self).__init__(shuffle=shuffle, rng=rng,
                                   mixed_precision=mixed_precision, channel_last=channel_last)
        self.data_dir = os.path.join(opt.data_dir, 'coco')
        self.img_dir = os.path.join(self.data_dir, '{}2017'.format(split))
        if split == 'test':
            self.annot_path = os.path.join(
                self.data_dir, 'annotations',
                'image_info_test-dev2017.json').format(split)
        else:
            self.annot_path = os.path.join(
                self.data_dir, 'annotations',
                'instances_{}2017.json').format(split)
        self.params = COCODefaultParams()
        self.split = split
        self.opt = opt

        self.coco = coco.COCO(self.annot_path)
        self.images = self.coco.getImgIds()
        self._size = len(self.images)
        self._variables = ('img', 'hm', 'ind', 'wh', 'reg', 'reg_mask', 'cls')

        if rng is None:
            rng = np.random.RandomState(313)
        self.rng = rng
        self.reset()

        print('==> initializing coco 2017 {} data.'.format(split))
        print('Loaded {} {} samples'.format(split, self._size))

    @property
    def default_resolution(self):
        return self._default_resolution

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_eval_format(self, all_bboxes):
        # import pdb; pdb.set_trace()
        detections = []
        for image_id in all_bboxes:
            for cls_ind in all_bboxes[image_id]:
                category_id = self.params._valid_ids[cls_ind - 1]
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

    def save_results(self, results, save_dir):
        json.dump(self.convert_eval_format(results), open(
            '{}/results.json'.format(save_dir), 'w'))

    def run_eval(self, results, save_dir, data_dir):
        self.save_results(results, save_dir)
        coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
        coco_eval = COCOeval(self.coco, coco_dets, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return coco_eval.stats[0]
