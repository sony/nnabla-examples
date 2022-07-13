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


def _gather_feat(feat, ind):
    """Gather feature according to index.

    Args:
        feat (numpy.ndarray): Target feature map. Shape: (batch, A, B)
        ind (numpy.ndarray): Target coord index. Shape: (batch, K)

    Returns:
        numpy.ndarray: Gathered feature. Shape: (batch, 1, K, 1, B)
    """
    ind = np.expand_dims(ind, axis=2).astype(int)
    result = np.take(feat, ind, axis=1)
    return result


def _transpose_and_gather_feat(feat, ind):
    """Transpose feature map and get gathered feature according to the corresponding index array.

    Args:
        feat (numpy.ndarray): Target feature map with (batch, C, H, W) shape
        ind (numpy.ndarray): Corresponding index array with (batch, K) shape.

    Returns:
        numpy.ndarray: Gathered feature. Shape: (batch, 1, K, 1, C)
    """
    # output shape: (batch, C, H, W) -> (batch, H, W, C)
    feat = feat.transpose(0, 2, 3, 1)
    # output shape: (batch, (W * H), C)
    feat = feat.reshape((feat.shape[0], -1, feat.shape[3]))
    # output shape: (batch, 1, K, 1, C)
    feat = _gather_feat(feat, ind)
    return feat
