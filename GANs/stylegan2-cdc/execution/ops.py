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
from nnabla.utils.image_utils import imsave

import numpy as np
import random


def save_generations(rgb_output, filepath, drange=[-1, 1], return_images=False):
    """
    Save generated images
    """
    if return_images:
        images = []
    for i in range(rgb_output.shape[0]):

        scale = 255 / (drange[1] - drange[0])
        if isinstance(rgb_output, nn.Variable):
            image = rgb_output.d[i] * scale + (0.5 - drange[0] * scale)
        else:
            image = rgb_output.data[i] * scale + (0.5 - drange[0] * scale)

        if return_images:
            images.append(np.uint8(np.clip(image, 0, 255)))
        else:
            imsave(f'{filepath}_{i}.png', np.uint8(
                np.clip(image, 0, 255)), channel_first=True)
            print(f'Output saved. Saved {filepath}_{i}.png')

    if return_images:
        return images


def collect_data(data):
    data = [np.expand_dims(d, 1) for d in data]
    data = np.concatenate(data, 1)
    return data


def mixing_noise(batch_size, latent_dim, mixing_prob, seed):

    rnd = np.random.RandomState(seed=seed[0])
    z = rnd.randn(batch_size, latent_dim).astype(np.float32)

    if mixing_prob > 0 and random.random() < mixing_prob:
        rnd_2 = np.random.RandomState(seed=seed[1])
        z_1 = z
        z_2 = rnd_2.randn(batch_size, latent_dim).astype(np.float32)
    else:
        z_1 = z_2 = z

    return z_1, z_2


def slerp(noise_1, noise_2, ratio):
    interpolated_noises = []
    for a, b in zip(noise_1, noise_2):
        a_norm = F.pow_scalar(
            F.sum(F.pow_scalar(a, 2), axis=1, keepdims=True), 0.5)
        b_norm = F.pow_scalar(
            F.sum(F.pow_scalar(b, 2), axis=1, keepdims=True), 0.5)

        a /= a_norm
        b /= b_norm

        d = F.sum(a*b, axis=1, keepdims=True)
        p = ratio*F.acos(d)
        c = b-d*a
        c_norm = F.pow_scalar(
            F.sum(F.pow_scalar(c, 2), axis=1, keepdims=True), 0.5)
        c /= c_norm

        d = a*F.cos(p) + c*F.sin(p)
        d = d/F.pow_scalar(F.sum(F.pow_scalar(d, 2),
                           axis=1, keepdims=True), 0.5)

        interpolated_noises.append(d)
    return interpolated_noises


def lerp(a, b, t):
    return a + (b - a) * t
