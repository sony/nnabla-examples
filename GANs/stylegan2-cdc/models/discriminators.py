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
import nnabla.parametric_functions as PF
import nnabla.initializer as I

import numpy as np
import subprocess as sp

from .networks import conv_layer, res_block
from .ops import weight_init_fn


class Discriminator(object):

    def __init__(self, discriminator_config, img_size):

        self.stddev_group = discriminator_config['stddev_group']
        self.stddev_feat = discriminator_config['stddev_feat']

        resolutions = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
        channels = [32, 64, 128, 256, 512, 512, 512, 512, 512]

        img_idx = resolutions.index(img_size)

        self.resolutions = resolutions[0:img_idx+1]
        # self.channels = channels[len(channels)-img_idx-1:]
        # for image size = 256
        self.channels = channels[len(channels)-img_idx-2:]

    def __call__(self, gen_rgb_out):

        out = conv_layer(gen_rgb_out, inmaps=3,
                         outmaps=self.channels[0], kernel_size=1, name_scope='Discriminator/Convinitial')

        inmaps = self.channels[0]
        for i in range(1, len(self.resolutions)):
            res = out.shape[2]
            outmaps = self.channels[i]
            out = res_block(out, res=res, outmaps=outmaps, inmaps=inmaps)
            inmaps = outmaps

        N, C, H, W = out.shape
        group = min(N, self.stddev_group)
        stddev_mean = F.reshape(
            out, (group, -1, self.stddev_feat, C // self.stddev_feat, H, W), inplace=False)

        # mean = F.mean(stddev_mean, axis=0, keepdims=True)
        mean = F.mul_scalar(F.sum(stddev_mean, axis=0, keepdims=True),
                            1.0/stddev_mean.shape[0], inplace=False)

        stddev_mean = F.mean(F.pow_scalar(F.sub2(stddev_mean, F.broadcast(
            mean, stddev_mean.shape)), 2.), axis=0, keepdims=False)
        stddev_mean = F.pow_scalar(F.add_scalar(
            stddev_mean, 1e-8, inplace=False), 0.5, inplace=False)

        stddev_mean = F.mean(stddev_mean, axis=[2, 3, 4], keepdims=True)
        stddev_mean = F.reshape(
            stddev_mean, stddev_mean.shape[:2]+stddev_mean.shape[3:], inplace=False)

        out = F.concatenate(out, F.tile(stddev_mean, (group, 1, H, W)), axis=1)

        out = conv_layer(out, inmaps=out.shape[1], outmaps=self.channels[-1],
                         kernel_size=3, name_scope='Discriminator/Convfinal')

        out = F.reshape(out, (N, -1), inplace=False)

        # Linear Layers
        lrmul = 1
        scale = 1/(out.shape[1]**0.5)*lrmul
        W, bias = weight_init_fn(
            (out.shape[-1], self.channels[-1]), weight_var='Discriminator/final_linear_1/affine')
        out = F.affine(out, W*scale, bias*lrmul)
        out = F.mul_scalar(F.leaky_relu(
            out, alpha=0.2, inplace=False), np.sqrt(2), inplace=False)

        scale = 1/(out.shape[1]**0.5)*lrmul
        W, bias = weight_init_fn(
            (out.shape[-1], 1), weight_var='Discriminator/final_linear_2/affine')
        out = F.affine(out, W*scale, bias*lrmul)

        return out
