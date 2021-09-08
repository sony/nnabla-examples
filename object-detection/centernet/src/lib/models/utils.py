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
import nnabla as nn
import nnabla.functions as F


def _gather_feat(feat, ind, mask=None):
    ind = np.expand_dims(ind, axis=2).astype(int)
    result = np.take(feat, ind, axis=1)
    return result


def _tranpose_and_gather_feat(feat, ind):
    feat = feat.transpose(0, 2, 3, 1)
    feat = feat.reshape((feat.shape[0], -1, feat.shape[3]))
    feat = _gather_feat(feat, ind)
    return feat
