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
import nnabla as nn
import nnabla.functions as F
from .utils import _gather_feat, _tranpose_and_gather_feat
import numpy as np


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2
    hmax = F.max_pooling(
        heat, (kernel, kernel), stride=(1, 1), pad=(pad, pad))
    keep = F.equal(hmax, heat)
    return heat*keep


def _topk(scores, K=40):
    batch, cat, height, width = scores.shape
    # Gather topk inds and scores using argpartition
    topk_inds = np.argsort(scores.d.reshape(
        batch, cat, -1))[:, :, ::-1][:, :, :K]
    topk_scores = np.sort(scores.d.reshape(
        batch, cat, -1))[:, :, ::-1][:, :, :K]
    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).astype(int).astype(np.float32)
    topk_xs = (topk_inds % width).astype(int).astype(np.float32)
    # Gather topk inds and scores per class using argpartition
    topk_ind = np.argsort(topk_scores.reshape(batch, -1))[:, ::-1][:, :K]
    topk_score = np.sort(topk_scores.reshape(batch, -1))[:, ::-1][:, :K]
    topk_clses = (topk_ind / K).astype(int).astype(np.float32)
    topk_inds = _gather_feat(topk_inds.reshape(
        (batch, -1, 1)), topk_ind).reshape((batch, K))
    topk_xs = _gather_feat(topk_xs.reshape(
        (batch, -1, 1)), topk_ind).reshape((batch, K))
    topk_ys = _gather_feat(topk_ys.reshape(
        (batch, -1, 1)), topk_ind).reshape((batch, K))

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def ctdet_decode(heat, wh, reg=None, K=128):
    heat = _nms(heat)
    batch, cat, height, width = heat.shape
    scores, inds, clses, ys, xs = _topk(heat, K=K)
    if reg is not None:
        reg = _tranpose_and_gather_feat(reg.d, inds)
        reg = reg.reshape((batch, K, 2))
        xs = xs.reshape((batch, K, 1))
        ys = ys.reshape((batch, K, 1))
        xs += reg[:, :, 0:1]
        ys += reg[:, :, 1:2]
    else:
        xs = xs.reshape((batch, K, 1)) + 0.5
        ys = ys.reshape((batch, K, 1)) + 0.5
    wh = _tranpose_and_gather_feat(wh.d, inds).reshape((batch, K, 2))

    clses = clses.reshape((batch, K, 1))
    scores = scores.reshape((batch, K, 1))
    bboxes = np.concatenate([xs - wh[..., 0:1] / 2,
                             ys -
                             wh[..., 1:2] / 2,
                             xs +
                             wh[..., 0:1] / 2,
                             ys +
                             wh[..., 1:2] / 2],
                            axis=2)
    detections = np.concatenate([
            bboxes,
            scores,
            clses],
            axis=2)
    return detections
