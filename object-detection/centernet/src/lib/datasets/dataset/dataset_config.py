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

import numpy as np
import os
import cv2
cv2.setNumThreads(1)
import json
import math

import pycocotools.coco as coco

from nnabla.utils.data_source import DataSource

from utils.image import fast_pad
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian


class DatasetConfig(DataSource):
    '''
    Implements Data Loader for serving the iterators
    channel_last sets input image data as NHWC
    mixed_precision zero pads the input image channel to 4 for better utilization of Tensor Cores
    '''

    def __init__(self, mixed_precision=False, channel_last=False, shuffle=False, rng=None):
        super(DatasetConfig, self).__init__(shuffle=shuffle, rng=rng)
        self.channel_last = channel_last
        self.mixed_precision = mixed_precision

    def save_results(self, results, save_dir):
        raise NotImplementedError

    def convert_eval_format(self, all_boxes):
        raise NotImplementedError

    def run_eval(self, results, save_dir, data_dir):
        raise NotImplementedError

    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                        dtype=np.float32)
        return bbox

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def __len__(self):
        return self._size

    def reset(self):
        if self._shuffle:
            self._indexes = self.rng.permutation(self._size)
        else:
            self._indexes = np.arange(self._size)
        super(DatasetConfig, self).reset()

    def _get_data(self, position):
        img_id = self.images[self._indexes[position]]
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        img_path = os.path.join(self.img_dir, file_name)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = min(len(anns), self.params.max_objs)

        img = cv2.imread(img_path)
        height, width = img.shape[0], img.shape[1]
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        if self.opt.keep_res:
            input_h = (height | self.opt.pad) + 1
            input_w = (width | self.opt.pad) + 1
            s = np.array([input_w, input_h], dtype=np.float32)
        else:
            s = max(img.shape[0], img.shape[1]) * 1.0
            input_h, input_w = self.opt.input_h, self.opt.input_w

        flipped = False
        if self.split == 'train':
            if not self.opt.not_rand_crop:
                s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
                w_border = self._get_border(128, img.shape[1])
                h_border = self._get_border(128, img.shape[0])
                c[0] = np.random.randint(
                    low=w_border, high=img.shape[1] - w_border)
                c[1] = np.random.randint(
                    low=h_border, high=img.shape[0] - h_border)
            else:
                sf = self.opt.scale
                cf = self.opt.shift
                c[0] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                c[1] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)

            if np.random.random() < self.opt.flip:
                flipped = True
                assert (len(
                    img.shape) == 3), f"The dimensions of img should be 3. Filename: {img_path}, shape: {img.shape}"
                img = img[:, ::-1, :]
                c[0] = width - c[0] - 1
        trans_input = get_affine_transform(
            c, s, 0, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input,
                             (input_w, input_h),
                             flags=cv2.INTER_LINEAR)
        inp = (inp.astype(np.float32) / 255.)
        if self.split == 'train' and not self.opt.no_color_aug:
            color_aug(self._rng, inp, self.params._eig_val,
                      self.params._eig_vec)
        inp = (inp - self.params.mean) / self.params.std

        if self.mixed_precision:
            inp = fast_pad(inp)
            # Transpose to NCHW if channel_last is not enabled
        if not self.channel_last:
            inp = inp.transpose(2, 0, 1)
        output_h = input_h // self.opt.down_ratio
        output_w = input_w // self.opt.down_ratio
        num_classes = self.params.num_classes
        trans_output = get_affine_transform(c, s, 0, [output_w, output_h])

        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        ind = np.zeros((self.params.max_objs), dtype=np.int32)
        wh = np.zeros((self.params.max_objs, 2), dtype=np.float32)
        reg = np.zeros((self.params.max_objs, 2), dtype=np.float32)
        reg_mask = np.zeros((self.params.max_objs, 1), dtype=np.float32)
        cls = np.zeros((self.params.max_objs, 1), dtype=np.int32)

        draw_gaussian = draw_umich_gaussian
        for k in range(num_objs):
            ann = anns[k]
            bbox = self._coco_box_to_bbox(ann['bbox'])
            cls_id = int(self.params.cat_ids[ann['category_id']])
            if flipped:
                bbox[[0, 2]] = width - bbox[[2, 0]] - 1
            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                draw_gaussian(hm[cls_id], ct_int, radius)
                wh[k] = 1. * w, 1. * h
                ind[k] = ct_int[1] * output_w + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1
                cls[k] = cls_id
        # Transpose heatmap to NHWC if channel last is enabled
        if self.channel_last:
            hm = np.transpose(hm, (1, 2, 0))
        ret = (inp, hm, ind, wh, reg, reg_mask, cls)
        return ret
