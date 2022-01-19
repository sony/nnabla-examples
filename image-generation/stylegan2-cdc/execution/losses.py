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


def disc_logistic_loss(real_disc_out, fake_disc_out):

    real_loss = F.softplus(-real_disc_out)
    fake_loss = F.softplus(fake_disc_out)

    return F.mean(real_loss) + F.mean(fake_loss)


def disc_r1_loss(real_disc_out, real_img):
    gradient = nn.grad([F.sum(real_disc_out)], [real_img])[0]
    gradient_penalty = F.pow_scalar(gradient, 2)
    gradient_penalty = F.reshape(gradient_penalty, (gradient.shape[0], -1))
    return F.mean(F.sum(gradient_penalty, axis=1))


def gen_nonsaturating_loss(fake_disc_out):
    loss = F.softplus(-fake_disc_out)
    return F.mean(loss)


def gen_path_regularize(fake_img, latents, mean_path_length, decay=0.01, pl_weight=2.0):

    noise = F.randn(shape=fake_img.shape) / \
                    np.sqrt(fake_img.shape[2]*fake_img.shape[3])

    gradient = nn.grad([F.sum(fake_img*noise)], [latents])[0]
    path_lengths = F.mean(F.sum(F.pow_scalar(gradient, 2), axis=1), axis=0)
    path_lengths = F.pow_scalar(path_lengths, 0.5)

    path_mean = mean_path_length + decay * \
        (F.mean(path_lengths) - mean_path_length)

    path_penalty = F.mean(F.pow_scalar(
        path_lengths-F.reshape(path_mean, (1,), inplace=False), 1))
    return path_penalty*pl_weight, path_mean, path_lengths
