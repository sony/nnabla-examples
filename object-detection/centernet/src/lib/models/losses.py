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

from .utils import _tranpose_and_gather_feat
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import numpy as np
import os
import cv2


def _focal_loss(pred, gt):
    '''Modified focal loss. Exactly the same as CornerNet.

    Modified for more stability by using log_sigmoid function

      Arguments:
        pred (batch x c x h x w): logit (must be values before sigmoid activation)
        gt_regr (batch x c x h x w)
    '''
    alpha = 2
    beta = 4
    pos_inds = F.greater_equal_scalar(gt, 1)
    neg_inds = 1 - pos_inds
    neg_weights = F.pow_scalar(1.0 - gt, beta)
    prob_pred = F.sigmoid(pred)
    pos_loss = F.log_sigmoid(pred)*F.pow_scalar(1.0-prob_pred, alpha)*pos_inds
    pos_loss = F.sum(pos_loss)
    neg_loss = F.log_sigmoid(-pred)*F.pow_scalar(prob_pred,
                                                 alpha)*neg_weights*neg_inds
    neg_loss = F.sum(neg_loss)
    num_pos = F.maximum_scalar(F.sum(pos_inds), 1)
    loss = -(1/num_pos) * (pos_loss + neg_loss)
    return loss


class FocalLoss():

    '''nn.Module wrapper for focal loss'''

    def __init__(self):
        super(FocalLoss, self).__init__()
        self.loss = _focal_loss

    def forward(self, out, target):
        return self.loss(out, target)


class L1Loss():
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, output, inds, gt, reg_mask, channel_last=False):
        # TODO refactor loss implementation for channel_last without transposing
        if channel_last:
            output = F.transpose(output, (0, 3, 1, 2))
        b = inds.shape[0]
        c = output.shape[1]
        max_objs = inds.shape[1]
        # divide by number of :
        num_objs = F.sum(reg_mask)*2
        f_map_size = output.shape[2]*output.shape[3]
        output = F.reshape(output, (-1, f_map_size))
        inds = F.broadcast(inds.reshape((b, 1, max_objs)), (b, c, max_objs))
        inds = inds.reshape((-1, max_objs))
        y = output[F.broadcast(F.reshape(
            F.arange(0, b*c), (b*c, 1)), (b*c, max_objs)), inds].reshape((b, c, max_objs))
        y = F.transpose(y, (0, 2, 1))
        loss = F.sum(reg_mask*F.absolute_error(y, gt))
        loss = loss / (num_objs + 1e-4)
        return loss
