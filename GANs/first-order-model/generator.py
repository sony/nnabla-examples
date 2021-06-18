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

from modules import resblock, sameblock, upblock, downblock
from dense_motion import predict_dense_motion


def deform_input(inp, deformation):
    _, h_old, w_old, _ = deformation.shape
    _, _, h, w = inp.shape

    if h_old != h or w_old != w:
        deformation = F.transpose(deformation, (0, 3, 1, 2))
        deformation = F.interpolate(deformation, output_size=(
            h, w), mode="linear", align_corners=False, half_pixel=True)
        deformation = F.transpose(deformation, (0, 2, 3, 1))

    return F.warp_by_grid(inp, deformation, align_corners=True)


def occlusion_aware_generator(source_image, kp_driving, kp_source,
                              num_channels, num_kp, block_expansion, max_features,
                              num_down_blocks, num_bottleneck_blocks,
                              estimate_occlusion_map=False, dense_motion_params=None,
                              estimate_jacobian=False, test=False, comm=None):

    # pre-downsampling
    out = sameblock(source_image, out_features=block_expansion,
                    kernel_size=7, padding=3, test=test, comm=comm)

    # downsampling
    for i in range(num_down_blocks):
        with nn.parameter_scope(f"downblock_{i}"):
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            out = downblock(out, out_features=out_features,
                            kernel_size=3, padding=1, test=test, comm=comm)

    output_dict = {}
    if dense_motion_params is not None:
        with nn.parameter_scope("dense_motion_prediction"):
            dense_motion = predict_dense_motion(source_image=source_image,
                                                kp_driving=kp_driving, kp_source=kp_source,
                                                num_kp=num_kp, num_channels=num_channels,
                                                estimate_occlusion_map=estimate_occlusion_map,
                                                test=test, comm=comm, **dense_motion_params)
        # dense_motion is a dictionay containing:
        # 'sparse_deformed': <Variable((8, 11, 3, 256, 256)),
        # 'mask': <Variable((8, 11, 256, 256)),
        # 'deformation': <Variable((8, 256, 256, 2)),
        # 'occlusion_map': <Variable((8, 1, 256, 256))}

        output_dict['mask'] = dense_motion['mask']
        output_dict['sparse_deformed'] = dense_motion['sparse_deformed']

        # Transform feature representation by deformation (+ occlusion)
        if 'occlusion_map' in dense_motion:
            occlusion_map = dense_motion['occlusion_map']
            output_dict['occlusion_map'] = occlusion_map
        else:
            occlusion_map = None
        deformation = dense_motion['deformation']
        out = deform_input(out, deformation)

        if occlusion_map is not None:
            if out.shape[2] != occlusion_map.shape[2] or out.shape[3] != occlusion_map.shape[3]:
                resized_occlusion_map = F.interpolate(occlusion_map,
                                                      output_size=out.shape[2:], mode="linear",
                                                      align_corners=False, half_pixel=True)
            else:
                resized_occlusion_map = F.identity(occlusion_map)
            out = out * resized_occlusion_map

        if test:
            output_dict["deformed"] = deform_input(source_image, deformation)

    # intermediate residual blocks
    in_features = min(max_features, block_expansion * (2 ** num_down_blocks))
    for i in range(num_bottleneck_blocks):
        with nn.parameter_scope(f"residual_block_{i}"):
            out = resblock(out, in_features=in_features,
                           kernel_size=3, padding=1, test=test, comm=comm)

    # upsampling
    for i in range(num_down_blocks):
        with nn.parameter_scope(f"upblock_{i}"):
            out_features = min(max_features, block_expansion *
                               (2 ** (num_down_blocks - i - 1)))
            out = upblock(out, out_features=out_features,
                          kernel_size=3, padding=1, test=test, comm=comm)

    with nn.parameter_scope("final_conv"):
        inmaps, outmaps = out.shape[1], num_channels
        k_w = I.calc_normal_std_he_forward(
            inmaps, outmaps, kernel=(7, 7)) / np.sqrt(2.)
        k_b = I.calc_normal_std_he_forward(inmaps, outmaps) / np.sqrt(2.)
        w_init = I.UniformInitializer((-k_w, k_w))
        b_init = I.UniformInitializer((-k_b, k_b))
        out = PF.convolution(out, outmaps=num_channels, kernel=(7, 7),
                             pad=(3, 3), w_init=w_init, b_init=b_init)
    out = F.sigmoid(out)
    output_dict["prediction"] = out

    return output_dict
