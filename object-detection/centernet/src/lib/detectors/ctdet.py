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

import cv2
import numpy as np
import time

from models.decode import ctdet_decode
from utils.post_process import ctdet_post_process
import nnabla as nn
import nnabla.functions as F
from .base_detector import BaseDetector


class CtdetDetector(BaseDetector):
    def __init__(self, opt):
        super(CtdetDetector, self).__init__(opt)

    def process(self, images, return_time=False):
        """ Apply detection to input images.

        :param images: input images with "NCHW" format.
        :param return_time: if True, return processing time.
        :return:
        """
        inputs = nn.Variable.from_numpy_array(images)
        outputs = self.model(inputs)
        hm = outputs[0]
        hm = F.sigmoid(hm)
        wh = outputs[1]
        reg = outputs[2]
        if self.opt.channel_last:
            hm = F.transpose(hm, (0, 3, 1, 2))
            wh = F.transpose(wh, (0, 3, 1, 2))
            reg = F.transpose(reg, (0, 3, 1, 2))
        forward_time = time.time()
        dets = ctdet_decode(hm, wh, reg=reg, K=self.opt.K)

        if return_time:
            return outputs, dets, forward_time
        else:
            return outputs, dets

    def post_process(self, dets, meta, scale=1):
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = ctdet_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], self.opt.num_classes)
        for j in range(1, self.opt.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
            dets[0][j][:, :4] /= scale
        return dets[0]

    def merge_outputs(self, detections):
        results = {}
        for j in range(1, self.opt.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0).astype(np.float32)
        scores = np.hstack(
            [results[j][:, 4] for j in range(1, self.opt.num_classes + 1)])
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.opt.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        return results

    def debug(self, debugger, images, dets, output, scale=1):
        detection = dets.copy()
        detection[:, :, :4] *= self.opt.down_ratio
        hm = output[0]
        hm = F.sigmoid(hm)
        if self.opt.channel_last:
            hm = F.transpose(hm, (0, 3, 1, 2))
        for i in range(1):
            if self.opt.mixed_precision:
                # Removing pad from input image for drawing
                img = images[i][:, :, :3]
            if not self.opt.channel_last:
                img = images[i].transpose(1, 2, 0)
            img = ((img * self.std + self.mean) * 255).astype(np.uint8)
            pred = debugger.gen_colormap(hm[i].d)
            debugger.add_blend_img(img, pred, 'pred_hm_{:.1f}'.format(scale))
            debugger.add_img(img, img_id='out_pred_{:.1f}'.format(scale))
            for k in range(len(dets[i])):
                if detection[i, k, 4] > self.opt.vis_thresh:
                    debugger.add_coco_bbox(detection[i, k, :4],
                                           detection[i, k, -1],
                                           detection[i, k, 4],
                                           img_id='out_pred_{:.1f}'.format(scale))
            for j in range(hm[i].shape[0]):
                hmap = hm[i][j].d
                hmap *= 255
                hmap = hmap.astype('uint8')
                print("max at channel {}:".format(j), np.max(hmap))
                hmap = cv2.applyColorMap(hmap, cv2.COLORMAP_JET)
                debugger.add_img(
                    hmap, img_id='heatmap_{}_{:.1f}'.format(j, scale))

    def show_results(self, debugger, image, results):
        debugger.add_img(image, img_id='ctdet')
        for j in range(1, self.opt.num_classes + 1):
            for bbox in results[j]:
                if bbox[4] > self.opt.vis_thresh:
                    debugger.add_coco_bbox(
                        bbox[:4], j - 1, bbox[4], img_id='ctdet')
        debugger.show_all_imgs(path=self.opt.save_dir)
