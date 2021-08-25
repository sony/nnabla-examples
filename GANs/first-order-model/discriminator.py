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

import functools
import numpy as np

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.initializer as I

from modules import kp2gaussian


def downblock(x, out_features, norm=False, kernel_size=4, pool=False, sn=False, test=False):
    out = x

    if sn:
        def apply_w(w): return PF.spectral_norm(w, dim=0, test=test)
    else:
        apply_w = None

    inmaps, outmaps = out.shape[1], out_features
    k_w = I.calc_normal_std_he_forward(
        inmaps, outmaps, kernel=(kernel_size, kernel_size)) / np.sqrt(2.)
    k_b = I.calc_normal_std_he_forward(inmaps, outmaps) / np.sqrt(2.)
    w_init = I.UniformInitializer((-k_w, k_w))
    b_init = I.UniformInitializer((-k_b, k_b))

    out = PF.convolution(out, out_features,
                         kernel=(kernel_size, kernel_size), pad=(0, 0),
                         stride=(1, 1), w_init=w_init, b_init=b_init,
                         apply_w=apply_w)

    if norm:
        out = PF.instance_normalization(out)

    out = F.leaky_relu(out, 0.2)

    if pool:
        out = F.average_pooling(out, kernel=(2, 2))

    return out


def discriminator(x, kp=None, num_channels=3, block_expansion=64,
                  num_blocks=4, max_features=512, sn=False, use_kp=False,
                  num_kp=10, kp_variance=0.01, test=False, **kwargs):

    down_blocks = []
    for i in range(num_blocks):
        down_blocks.append(
                functools.partial(downblock,
                                  out_features=min(
                                      max_features, block_expansion * (2 ** (i + 1))),
                                  norm=(i != 0), kernel_size=4,
                                  pool=(i != num_blocks - 1), sn=sn,
                                  test=test))

    feature_maps = []
    out = x

    if use_kp:
        heatmap = kp2gaussian(kp, x.shape[2:], kp_variance)
        out = F.concatenate(out, heatmap, axis=1)

    for i, down_block in enumerate(down_blocks):
        with nn.parameter_scope(f"downblock_{i}"):
            feature_maps.append(down_block(out))
            out = feature_maps[-1]

    if sn:
        def apply_w(w): return PF.spectral_norm(w, dim=0, test=test)
    else:
        apply_w = None

    with nn.parameter_scope("prediction"):
        inmaps, outmaps = out.shape[1], 1
        k_w = I.calc_normal_std_he_forward(
            inmaps, outmaps, kernel=(1, 1)) / np.sqrt(2.)
        k_b = I.calc_normal_std_he_forward(inmaps, outmaps) / np.sqrt(2.)
        w_init = I.UniformInitializer((-k_w, k_w))
        b_init = I.UniformInitializer((-k_b, k_b))
        prediction_map = PF.convolution(out, 1, kernel=(1, 1), pad=(0, 0),
                                        stride=(1, 1),
                                        w_init=w_init,
                                        b_init=b_init,
                                        apply_w=apply_w)
    return feature_maps, prediction_map


def multiscale_discriminator(x, kp=None, scales=[1], **kwargs):
    discs = dict()
    for scale in scales:
        discs[str(scale).replace('.', '-')
              ] = functools.partial(discriminator, **kwargs)

    out_dict = {}
    # this is executed only once since in most cases scales = [1].
    for scale, disc in discs.items():
        scale = str(scale).replace('-', '.')
        key = 'prediction_' + scale
        with nn.parameter_scope(f"{scale}"):
            feature_maps, prediction_map = disc(x[key], kp)
        out_dict['feature_maps_' + scale] = feature_maps
        out_dict['prediction_map_' + scale] = prediction_map
    return out_dict
