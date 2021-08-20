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

import numpy as np

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.initializer as I

from modules import hourglass, make_coordinate_grid, anti_alias_interpolate


def gaussian2kp(heatmap):
    shape = heatmap.shape
    heatmap = F.reshape(heatmap, shape + (1,), inplace=False)
    grid = make_coordinate_grid(shape[2:])
    grid = F.reshape(grid, (1, 1) + grid.shape)
    value = F.sum(heatmap * grid, axis=(2, 3))

    kp = {'value': value}

    return kp


def detect_keypoint(x, block_expansion, num_kp, num_channels, max_features,
                    num_blocks, temperature, estimate_jacobian=False, scale_factor=1,
                    single_jacobian_map=False, pad=0,
                    test=False, comm=None):

    if scale_factor != 1:
        x = anti_alias_interpolate(x, num_channels, scale_factor)

    with nn.parameter_scope("hourglass"):
        feature_map = hourglass(x, block_expansion, num_blocks=num_blocks,
                                max_features=max_features, test=test, comm=comm)

    with nn.parameter_scope("keypoint_detector"):
        inmaps, outmaps = feature_map.shape[1], num_kp
        k_w = I.calc_normal_std_he_forward(
            inmaps, outmaps, kernel=(7, 7)) / np.sqrt(2.)
        k_b = I.calc_normal_std_he_forward(inmaps, outmaps) / np.sqrt(2.)
        w_init = I.UniformInitializer((-k_w, k_w))
        b_init = I.UniformInitializer((-k_b, k_b))
        prediction = PF.convolution(feature_map, outmaps=num_kp,
                                    kernel=(7, 7), pad=(pad, pad),
                                    w_init=w_init, b_init=b_init)

    final_shape = prediction.shape

    heatmap = F.reshape(prediction, (final_shape[0], final_shape[1], -1))
    heatmap = F.softmax(heatmap / temperature, axis=2)
    heatmap = F.reshape(heatmap, final_shape, inplace=False)

    out = gaussian2kp(heatmap)  # {"value": value}, keypoint positions.

    if estimate_jacobian:
        if single_jacobian_map:
            num_jacobian_maps = 1
        else:
            num_jacobian_maps = num_kp

        with nn.parameter_scope("jacobian_estimator"):
            jacobian_map = PF.convolution(feature_map,
                                          outmaps=4*num_jacobian_maps,
                                          kernel=(7, 7), pad=(pad, pad),
                                          w_init=I.ConstantInitializer(0),
                                          b_init=np.array([1, 0, 0, 1]*num_jacobian_maps))

        jacobian_map = F.reshape(
            jacobian_map, (final_shape[0], num_jacobian_maps, 4, final_shape[2], final_shape[3]))
        heatmap = F.reshape(
            heatmap, heatmap.shape[:2] + (1,) + heatmap.shape[2:], inplace=False)

        jacobian = heatmap * jacobian_map
        jacobian = F.sum(jacobian, axis=(3, 4))
        jacobian = F.reshape(
            jacobian, (jacobian.shape[0], jacobian.shape[1], 2, 2), inplace=False)
        out['jacobian'] = jacobian  # jacobian near each keypoint.

    # out is a dictionary containing {"value": value, "jacobian": jacobian}

    return out
