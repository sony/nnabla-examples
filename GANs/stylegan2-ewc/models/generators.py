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
import nnabla.initializer as I

import numpy as np
import subprocess as sp

from .ops import upsample_2d, lerp
from .networks import mapping_network, styled_conv_block


class Generator(object):

    def __init__(self, generator_config, img_size, mix_after=None, global_scope=''):
        self.rnd = np.random.RandomState(generator_config['rnd_seed'])
        self.mapping_network_dim = generator_config['mapping_network_dim']
        self.mapping_network_num_layers = generator_config['mapping_network_num_layers']

        self.feature_map_dim = generator_config['feature_map_dim']

        resolutions = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
        img_idx = resolutions.index(img_size)
        self.resolutions = resolutions[0:img_idx+1]

        self.num_conv_layers = 2*(img_idx+1)

        if mix_after == None:
            self.mix_after = np.random.randint(1, self.num_conv_layers-1)
        else:
            self.mix_after = mix_after

        self.global_scope = global_scope

    def synthesis(self, w_mixed, constant_bc, seed=-1, noises_in=None):

        batch_size = w_mixed.shape[0]

        if noises_in is None:
            noise = F.randn(shape=(batch_size, 1, 4, 4), seed=seed)
        else:
            noise = noises_in[0]
        w = F.reshape(F.slice(w_mixed, start=(0, 0, 0), stop=(w_mixed.shape[0], 1, w_mixed.shape[2]), step=(
            1, 1, 1)), (w_mixed.shape[0], w_mixed.shape[2]), inplace=False)
        h = styled_conv_block(constant_bc, w, noise, res=self.resolutions[0],
                              outmaps=self.feature_map_dim, namescope="Conv")
        torgb = styled_conv_block(h, w, noise=None, res=self.resolutions[0], outmaps=3, inmaps=self.feature_map_dim,
                                  kernel_size=1, pad_size=0, demodulate=False, namescope="ToRGB", act=F.identity)

        # initial feature maps
        outmaps = self.feature_map_dim
        inmaps = self.feature_map_dim

        downsize_index = 4 if self.resolutions[-1] in [512, 1024] else 3

        # resolution 8 x 8 - 1024 x 1024
        for i in range(1, len(self.resolutions)):

            i1 = (2+i)*2-5
            i2 = (2+i)*2-4
            i3 = (2+i)*2-3
            w_ = F.reshape(F.slice(w_mixed, start=(0, i1, 0), stop=(
                w_mixed.shape[0], i1+1, w_mixed.shape[2]), step=(1, 1, 1)), w.shape, inplace=False)
            if i > downsize_index:
                outmaps = outmaps // 2
            curr_shape = (
                batch_size, 1, self.resolutions[i], self.resolutions[i])
            if noises_in is None:
                noise = F.randn(shape=curr_shape, seed=seed)
            else:
                noise = noises_in[2*i-1]

            h = styled_conv_block(h, w_, noise, res=self.resolutions[i], outmaps=outmaps, inmaps=inmaps,
                                  kernel_size=3, up=True, namescope="Conv0_up")

            w_ = F.reshape(F.slice(w_mixed, start=(0, i2, 0), stop=(
                w_mixed.shape[0], i2+1, w_mixed.shape[2]), step=(1, 1, 1)), w.shape, inplace=False)
            if i > downsize_index:
                inmaps = inmaps // 2
            if noises_in is None:
                noise = F.randn(shape=curr_shape, seed=seed)
            else:
                noise = noises_in[2*i]
            h = styled_conv_block(h, w_, noise, res=self.resolutions[i], outmaps=outmaps, inmaps=inmaps,
                                  kernel_size=3, pad_size=1, namescope="Conv1")

            w_ = F.reshape(F.slice(w_mixed, start=(0, i3, 0), stop=(
                w_mixed.shape[0], i3+1, w_mixed.shape[2]), step=(1, 1, 1)), w.shape, inplace=False)
            curr_torgb = styled_conv_block(h, w_, noise=None, res=self.resolutions[i], outmaps=3, inmaps=inmaps,
                                           kernel_size=1, pad_size=0, demodulate=False, namescope="ToRGB", act=F.identity)

            torgb = F.add2(curr_torgb, upsample_2d(torgb, k=[1, 3, 3, 1]))

        return torgb

    def __call__(self, batch_size, style_noises, truncation_psi=1.0, return_latent=False, mixing_layer_index=None, dlatent_avg_beta=0.995):

        with nn.parameter_scope(self.global_scope):
            # normalize noise inputs
            for i in range(len(style_noises)):
                style_noises[i] = F.div2(style_noises[i], F.pow_scalar(F.add_scalar(F.mean(
                    style_noises[i] ** 2., axis=1, keepdims=True), 1e-8, inplace=False), 0.5, inplace=False))

            # get latent code
            w = [mapping_network(style_noises[0], outmaps=self.mapping_network_dim,
                                 num_layers=self.mapping_network_num_layers)]
            w += [mapping_network(style_noises[1], outmaps=self.mapping_network_dim,
                                  num_layers=self.mapping_network_num_layers)]

            dlatent_avg = nn.parameter.get_parameter_or_create(
                name="dlatent_avg", shape=(1, 512))

            # Moving average update of dlatent_avg
            batch_avg = F.mean((w[0] + w[1])*0.5, axis=0, keepdims=True)
            update_op = F.assign(dlatent_avg, lerp(
                batch_avg, dlatent_avg, dlatent_avg_beta))
            update_op.name = 'dlatent_avg_update'
            dlatent_avg = F.identity(dlatent_avg) + 0*update_op

            # truncation trick
            w = [lerp(dlatent_avg, _, truncation_psi) for _ in w]

            # generate output from generator
            constant_bc = nn.parameter.get_parameter_or_create(
                            name="G_synthesis/4x4/Const/const",
                            shape=(1, 512, 4, 4), initializer=np.random.randn(1, 512, 4, 4).astype(np.float32))
            constant_bc = F.broadcast(
                constant_bc, (batch_size,) + constant_bc.shape[1:])

            if mixing_layer_index is None:
                mixing_layer_index_var = F.randint(
                    1, len(self.resolutions)*2, (1,))
            else:
                mixing_layer_index_var = F.constant(
                    val=mixing_layer_index, shape=(1,))
            mixing_switch_var = F.clip_by_value(
                F.arange(0, len(self.resolutions)*2) - mixing_layer_index_var, 0, 1)
            mixing_switch_var_re = F.reshape(
                mixing_switch_var, (1, mixing_switch_var.shape[0], 1), inplace=False)
            w0 = F.reshape(w[0], (batch_size, 1, w[0].shape[1]), inplace=False)
            w1 = F.reshape(w[1], (batch_size, 1, w[0].shape[1]), inplace=False)
            w_mixed = w0 * mixing_switch_var_re + \
                w1 * (1 - mixing_switch_var_re)

            rgb_output = self.synthesis(w_mixed, constant_bc)

            if return_latent:
                return rgb_output, w_mixed
            else:
                return rgb_output
