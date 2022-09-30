# Copyright 2023 Sony Group Corporation.
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


import random

import nnabla as nn
import nnabla.functions as F
import numpy as np
from PIL import Image


def calc_acc(y_pred, y_test):
    y_pred = nn.Variable.from_numpy_array(y_pred)
    prob = F.softmax(y_pred)
    prob.forward()
    y_pred_tags = np.argmax(prob.d, axis=1)

    acc = np.sum(y_pred_tags == y_test.flatten()) / len(y_test)

    acc = acc * 100

    return acc


def colorize_mask(mask, palette):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return np.array(new_mask.convert('RGB'))


def to_labels(masks, palette):
    results = np.zeros((256, 256), dtype=np.int32).reshape(-1)
    mask_flat = masks.reshape(-1, 3)  # 256*256, 3
    label = 0
    num = 0
    for color in np.array(palette).reshape(-1, 3):
        idxs = np.where((mask_flat == color).all(-1))
        results[idxs] = label
        label += 1
    return results.reshape(256, 256)
