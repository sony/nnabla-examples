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

from modules import hourglass, anti_alias_interpolate, make_coordinate_grid, kp2gaussian


def predict_dense_motion(source_image, kp_driving, kp_source,
                         block_expansion, num_blocks, max_features,
                         num_kp, num_channels, estimate_occlusion_map=False,
                         scale_factor=1, kp_variance=0.01,
                         test=False, comm=None):
    if scale_factor != 1:
        source_image = anti_alias_interpolate(
            source_image, num_channels, scale_factor)

    bs, _, h, w = source_image.shape

    out_dict = dict()
    heatmap_representation = create_heatmap_representations(
        source_image, kp_driving, kp_source, kp_variance)
    sparse_motion = create_sparse_motions(
        source_image, kp_driving, kp_source, num_kp)
    deformed_source = create_deformed_source_image(
        source_image, sparse_motion, num_kp)
    out_dict['sparse_deformed'] = deformed_source

    input = F.concatenate(heatmap_representation, deformed_source, axis=2)
    input = F.reshape(input, (bs, -1, h, w))

    with nn.parameter_scope("hourglass"):
        prediction = hourglass(input,
                               block_expansion=block_expansion, num_blocks=num_blocks,
                               max_features=max_features, test=test, comm=comm)

    with nn.parameter_scope("mask"):
        inmaps, outmaps = prediction.shape[1], num_kp + 1
        k_w = I.calc_normal_std_he_forward(
            inmaps, outmaps, kernel=(7, 7)) / np.sqrt(2.)
        k_b = I.calc_normal_std_he_forward(inmaps, outmaps) / np.sqrt(2.)
        w_init = I.UniformInitializer((-k_w, k_w))
        b_init = I.UniformInitializer((-k_b, k_b))
        mask = PF.convolution(prediction, outmaps=num_kp + 1, kernel=(7, 7),
                              pad=(3, 3), w_init=w_init, b_init=b_init)

    mask = F.softmax(mask, axis=1)
    out_dict['mask'] = mask
    reshaped_mask = F.reshape(
        mask, mask.shape[:2] + (1,) + mask.shape[2:], inplace=False)
    sparse_motion = F.transpose(sparse_motion, (0, 1, 4, 2, 3))
    deformation = F.sum(sparse_motion * reshaped_mask, axis=1)
    deformation = F.transpose(deformation, (0, 2, 3, 1))

    out_dict['deformation'] = deformation

    if estimate_occlusion_map:
        with nn.parameter_scope("occlusion_map"):
            occlusion_map = F.sigmoid(
                                PF.convolution(prediction,
                                               outmaps=1, kernel=(7, 7),
                                               pad=(3, 3), w_init=w_init, b_init=b_init))
        out_dict['occlusion_map'] = occlusion_map
    else:
        occlusion_map = None

    return out_dict


def create_heatmap_representations(source_image, kp_driving, kp_source, kp_variance):
    spatial_size = source_image.shape[2:]
    gaussian_driving = kp2gaussian(
        kp_driving, spatial_size=spatial_size, kp_variance=kp_variance)
    gaussian_source = kp2gaussian(
        kp_source, spatial_size=spatial_size, kp_variance=kp_variance)
    heatmap = gaussian_driving - gaussian_source

    # background feature
    zeros = F.constant(
        0, (heatmap.shape[0], 1, spatial_size[0], spatial_size[1]))
    heatmap = F.concatenate(zeros, heatmap, axis=1)
    heatmap = F.reshape(heatmap, heatmap.shape[:2] + (1,) + heatmap.shape[2:])
    return heatmap


def create_sparse_motions(source_image, kp_driving, kp_source, num_kp):
    bs, _, h, w = source_image.shape
    identity_grid = make_coordinate_grid((h, w))
    identity_grid = F.reshape(
        identity_grid, (1, 1, h, w, 2))  # (1, 1, h, w, 2)
    coordinate_grid = identity_grid - \
        F.reshape(kp_driving['value'], (bs, num_kp, 1, 1, 2), inplace=False)

    if 'jacobian' in kp_driving:
        jacobian = F.batch_matmul(kp_source['jacobian'],
                                  F.reshape(
                                    F.batch_inv(
                                        F.reshape(kp_driving['jacobian'],
                                                  (-1,) + kp_driving['jacobian'].shape[-2:], inplace=False)
                                                ),
                                    kp_driving['jacobian'].shape))
        # what it does
        # batched_driving_jacobian = F.reshape(kp_driving['jacobian'], (-1) + kp_driving['jacobian'].shape[-2:])
        # batched_inverse_jacobian = F.batch_inv(batched_driving_jacobian)
        # inverse_jacobian = F.reshape(batched_inverse_jacobian, kp_driving['jacobian'].shape)

        jacobian = F.reshape(
            jacobian, jacobian.shape[:-2] + (1, 1) + jacobian.shape[-2:])
        jacobian = F.broadcast(
            jacobian, jacobian.shape[:2] + (h, w) + jacobian.shape[-2:])

        coordinate_grid = F.batch_matmul(jacobian, F.reshape(
            coordinate_grid, coordinate_grid.shape + (1,)))
        coordinate_grid = F.reshape(
            coordinate_grid, coordinate_grid.shape[:-1])

    driving_to_source = coordinate_grid + \
        F.reshape(kp_source['value'], (bs, num_kp, 1, 1, 2), inplace=False)

    # background feature
    identity_grid = F.broadcast(identity_grid, (bs, 1, h, w, 2))

    sparse_motions = F.concatenate(identity_grid, driving_to_source, axis=1)
    return sparse_motions


def create_deformed_source_image(source_image, sparse_motions, num_kp):
    bs, c, h, w = source_image.shape
    source_repeat = F.reshape(source_image, (bs, 1, 1, c, h, w))
    source_repeat = F.broadcast(source_repeat, (bs, num_kp + 1, 1, c, h, w))

    source_repeat = F.reshape(source_repeat, (bs * (num_kp + 1), -1, h, w))
    sparse_motions = F.reshape(sparse_motions, (bs * (num_kp + 1), h, w, -1))

    sparse_deformed = F.warp_by_grid(
        source_repeat, sparse_motions, align_corners=True)
    sparse_deformed = F.reshape(sparse_deformed, (bs, num_kp + 1, -1, h, w))

    return sparse_deformed
