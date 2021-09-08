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

from model import PretrainedVgg19


def perceptual_loss(pyramide_real, pyramide_fake, scales, weights, vgg_param_path):
    """
        Compute Perceptual Loss using VGG19 as a feature extractor.
    """
    vgg19 = PretrainedVgg19(param_path=vgg_param_path)
    variable_not_exist = True
    for scale in scales:
        x_vgg = vgg19(pyramide_fake[f'prediction_{scale}'])
        y_vgg = vgg19(pyramide_real[f'prediction_{scale}'])

        for i, weight in enumerate(weights):
            value = F.mean(F.absolute_error(x_vgg[i], y_vgg[i]))
            if variable_not_exist:
                loss = weight * value
                variable_not_exist = False
            else:
                loss += weight * value
    return loss


def lsgan_loss(real, weight, fake=None):
    if fake:
        loss = weight * F.mean(F.squared_error(F.constant(1, real.shape), real)
                               + F.pow_scalar(fake, 2))
    else:
        loss = weight * \
            F.mean(F.squared_error(F.constant(1, real.shape), real))
    return loss


def feature_matching_loss(dis_maps_real, dis_maps_fake, model_params, weights):
    variable_not_exist = True
    for j, scale in enumerate(model_params['discriminator_params']['scales']):
        key = f"feature_maps_{scale}".replace('.', '-')
        for i, (a, b) in enumerate(zip(dis_maps_real[key], dis_maps_fake[key])):
            if weights[i] == 0:
                continue
            if variable_not_exist:
                loss = F.mean(F.absolute_error(a, b)) * weights[i]
                variable_not_exist = False
            else:
                loss += F.mean(F.absolute_error(a, b)) * weights[i]
    return loss


def equivariance_value_loss(kp_driving_value, warped_kp_value, weight):
    value_loss = F.mean(F.absolute_error(kp_driving_value, warped_kp_value))
    loss = weight * value_loss
    return loss


def equivariance_jacobian_loss(kp_driving_jacobian,
                               arithmetic_jacobian,
                               trans_kp_jacobian, weight):
    jacobian_transformed = F.batch_matmul(arithmetic_jacobian,
                                          trans_kp_jacobian)

    normed_driving = F.reshape(
                        F.batch_inv(
                            F.reshape(kp_driving_jacobian,
                                      (-1,) + kp_driving_jacobian.shape[-2:])),
                        kp_driving_jacobian.shape)

    normed_transformed = jacobian_transformed
    value = F.batch_matmul(normed_driving, normed_transformed)

    eye = nn.Variable.from_numpy_array(np.reshape(np.eye(2), (1, 1, 2, 2)))

    jacobian_loss = F.mean(F.absolute_error(eye, value))
    loss = weight * jacobian_loss
    return loss
