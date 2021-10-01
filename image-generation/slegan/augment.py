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


import nnabla as nn
import nnabla.functions as F
import numpy as np


def augment(batch, aug_list, p_aug=1.0):

    if isinstance(p_aug, float):
        p_aug = nn.Variable.from_numpy_array(p_aug * np.ones((1,)))

    if "flip" in aug_list:
        rnd = F.rand(shape=[batch.shape[0], ])
        batch_aug = F.random_flip(batch, axes=(2, 3))
        batch = F.where(
            F.greater(F.tile(p_aug, batch.shape[0]), rnd), batch_aug, batch)

    if "lrflip" in aug_list:
        rnd = F.rand(shape=[batch.shape[0], ])
        batch_aug = F.random_flip(batch, axes=(3,))
        batch = F.where(
            F.greater(F.tile(p_aug, batch.shape[0]), rnd), batch_aug, batch)

    if "translation" in aug_list and batch.shape[2] >= 8:
        rnd = F.rand(shape=[batch.shape[0], ])
        # Currently nnabla does not support random_shift with border_mode="noise"
        mask = np.ones((1, 3, batch.shape[2], batch.shape[3]))
        mask[:, :, :, 0] = 0
        mask[:, :, :, -1] = 0
        mask[:, :, 0, :] = 0
        mask[:, :, -1, :] = 0
        batch_int = F.concatenate(
            batch, nn.Variable().from_numpy_array(mask), axis=0)
        batch_int_aug = F.random_shift(batch_int, shifts=(
            batch.shape[2]//8, batch.shape[3]//8), border_mode="nearest")
        batch_aug = F.slice(batch_int_aug, start=(
            0, 0, 0, 0), stop=batch.shape)
        mask_var = F.slice(batch_int_aug, start=(
            batch.shape[0], 0, 0, 0), stop=batch_int_aug.shape)
        batch_aug = batch_aug * F.broadcast(mask_var, batch_aug.shape)
        batch = F.where(
            F.greater(F.tile(p_aug, batch.shape[0]), rnd), batch_aug, batch)

    if "color" in aug_list:
        rnd = F.rand(shape=[batch.shape[0], ])
        rnd_contrast = 1.0 + 0.5 * \
            (2.0 * F.rand(shape=[batch.shape[0], 1, 1, 1]
                          ) - 1.0)  # from 0.5 to 1.5
        rnd_brightness = 0.5 * \
            (2.0 * F.rand(shape=[batch.shape[0], 1, 1, 1]
                          ) - 1.0)  # from -0.5 to 0.5
        rnd_saturation = 2.0 * \
            F.rand(shape=[batch.shape[0], 1, 1, 1])  # from 0.0 to 2.0
        # Brightness
        batch_aug = batch + rnd_brightness
        # Saturation
        mean_s = F.mean(batch_aug, axis=1, keepdims=True)
        batch_aug = rnd_saturation * (batch_aug - mean_s) + mean_s
        # Contrast
        mean_c = F.mean(batch_aug, axis=(1, 2, 3), keepdims=True)
        batch_aug = rnd_contrast * (batch_aug - mean_c) + mean_c
        batch = F.where(
            F.greater(F.tile(p_aug, batch.shape[0]), rnd), batch_aug, batch)

    if "cutout" in aug_list and batch.shape[2] >= 16:
        batch = F.random_erase(batch, prob=p_aug.d[0], replacements=(0.0, 0.0))

    return batch
