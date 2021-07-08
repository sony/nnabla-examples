
# Copyright 2021 Sony Group Corporation
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
import sys

import nnabla as nn
import nnabla.functions as F
import numpy as np

from preprocess import uncenter_l
from args import get_config

conf = get_config()


def vgg_pre_process(x):
    x_bgr = F.concatenate(
        x[:, 2:3, :, :], x[:, 1:2, :, :], x[:, 0:1, :, :], axis=1)
    # tensor_bgr = tensor[:, [2, 1, 0], ...]
    x_sub = F.reshape(nn.Variable.from_numpy_array(
        np.array([0.40760392, 0.45795686, 0.48501961])), (1, 3, 1, 1))
    x_bgr_ml = x_bgr - x_sub
    x_rst = x_bgr_ml * 255
    return x_rst


def feature_normalize(feature_in):
    feature_in_norm = F.norm(
        feature_in,
        p=2,
        axis=1,
        keepdims=True) + sys.float_info.epsilon
    feature_in_norm = F.div2(feature_in, feature_in_norm)
    return feature_in_norm


def gray2rgb_batch(l):
    # gray image tensor to rgb image tensor
    l_uncenter = uncenter_l(l, conf)
    l_uncenter = l_uncenter / (2 * conf.l_mean)
    return F.concatenate(l_uncenter, l_uncenter, l_uncenter, axis=1)
