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

import os
import numpy as np

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.logger as logger

from modules import make_coordinate_grid, anti_alias_interpolate


def vgg_prediction(image, nmaps=64, config="VGG19", with_bias=True, with_bn=False, test=False, finetune=True):

    def convblock(x, nmaps, layer_idx, with_bias, with_bn=False):
        h = x
        scopenames = ["conv{}".format(_) for _ in layer_idx]
        for scopename in scopenames:
            with nn.parameter_scope(scopename):
                if scopename not in ["conv1", "conv13"] and scopename == scopenames[-1]:
                    nmaps *= 2
                h = PF.convolution(h, nmaps, kernel=(3, 3), pad=(
                    1, 1), with_bias=with_bias, fix_parameters=finetune)
                if with_bn:
                    h = PF.batch_normalization(
                        h, batch_stat=not test, fix_parameters=finetune)
            h = F.relu(h)
            if len(scopenames) != 1 and scopename == scopenames[-2]:
                h = F.max_pooling(h, kernel=(2, 2), stride=(2, 2))
        return h

    assert config in ["VGG19"]
    if config == "VGG19":
        layer_indices = [(1,), (2, 3), (4, 5),
                         (6, 7, 8, 9), (10, 11, 12, 13)]
    else:
        raise NotImplementedError

    h1 = convblock(image, 64, layer_indices[0], with_bias, with_bn)
    h2 = convblock(h1, 64, layer_indices[1], with_bias, with_bn)
    h3 = convblock(h2, 128, layer_indices[2], with_bias, with_bn)
    h4 = convblock(h3, 256, layer_indices[3], with_bias, with_bn)
    h5 = convblock(h4, 512, layer_indices[4], with_bias, with_bn)

    out = [h1, h2, h3, h4, h5]

    return out


class PretrainedVgg19(object):
    def __init__(self, param_path=None):
        assert os.path.isfile(
            param_path), "pretrained VGG19 weights not found."
        self.h5_file = param_path
        if not os.path.exists(self.h5_file):
            print("Pretrained VGG19 parameters not found. Downloading. Please wait...")
            url = "https://nnabla.org/pretrained-models/nnabla-examples/GANs/first-order-model/vgg19.h5"
            from nnabla.utils.data_source_loader import download
            download(url, url.split('/')[-1], False)

        with nn.parameter_scope("VGG19"):
            logger.info('loading vgg19 parameters...')
            nn.load_parameters(self.h5_file)
            # drop all the affine layers.
            drop_layers = ['classifier/0/affine',
                           'classifier/3/affine', 'classifier/6/affine']
            for layers in drop_layers:
                nn.parameter.pop_parameter((layers + '/W'))
                nn.parameter.pop_parameter((layers + '/b'))
            self.mean = nn.Variable.from_numpy_array(
                np.asarray([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1))
            self.std = nn.Variable.from_numpy_array(
                np.asarray([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1))

    def __call__(self, x):
        with nn.parameter_scope("VGG19"):
            self.x = F.div2(F.sub2(x, self.mean), self.std)
            return vgg_prediction(self.x, finetune=True)


def get_image_pyramid(x, scales, num_channels):
    out_dict = dict()
    for scale in scales:
        out_dict['prediction_' + str(scale).replace('-', '.')
                 ] = anti_alias_interpolate(x, num_channels, scale)
    return out_dict


def unlink_all(kp, need_grad=False):
    """
        works like torch.Tensor.detach.
    """
    return {key: value.get_unlinked_variable(need_grad=need_grad) for key, value in kp.items()}


def persistent_all(variable_dict):
    for variable in variable_dict.values():
        variable.persistent = True
    return


class Transform:
    def __init__(self, bs, **kwargs):
        noise = F.randn(mu=0, sigma=kwargs['sigma_affine'], shape=(bs, 2, 3))
        self.theta = noise + \
            nn.Variable.from_numpy_array(
                np.array([[[1., 0., 0.], [0., 1., 0.]]]))
        self.bs = bs

        if ('sigma_tps' in kwargs) and ('points_tps' in kwargs):
            self.tps = True
            self.control_points = make_coordinate_grid(
                (kwargs['points_tps'], kwargs['points_tps']))
            self.control_points = F.reshape(
                self.control_points, (1,) + self.control_points.shape)
            self.control_params = F.randn(
                mu=0, sigma=kwargs['sigma_tps'], shape=(bs, 1, kwargs['points_tps'] ** 2))
        else:
            self.tps = False

    def transform_frame(self, frame):
        grid = make_coordinate_grid(frame.shape[2:])
        grid = F.reshape(grid, (1, frame.shape[2] * frame.shape[3], 2))
        grid = self.warp_coordinates(grid)
        grid = F.reshape(grid, (self.bs, frame.shape[2], frame.shape[3], 2))

        return F.warp_by_grid(frame, grid, padding_mode="reflect", align_corners=True)

    def warp_coordinates(self, coordinates):
        theta = self.theta
        theta = F.reshape(
            theta, theta.shape[:1] + (1,) + theta.shape[1:], inplace=False)
        if coordinates.shape[0] == self.bs:
            transformed = F.batch_matmul(
                            F.tile(theta[:, :, :, :2],
                                   (1, coordinates.shape[1], 1, 1)),
                            F.reshape(coordinates, coordinates.shape + (1,), inplace=False)) + theta[:, :, :, 2:]
        else:
            transformed = F.batch_matmul(
                            F.tile(theta[:, :, :, :2],
                                   (1, coordinates.shape[1], 1, 1)),
                            F.tile(F.reshape(coordinates, coordinates.shape + (1,), inplace=False),
                                   (self.bs / coordinates.shape[0], 1, 1, 1))) + theta[:, :, :, 2:]
        transformed = F.reshape(
            transformed, transformed.shape[:-1], inplace=False)

        if self.tps:
            control_points = self.control_points
            control_params = self.control_params
            distances = F.reshape(
                coordinates, (coordinates.shape[0], -1, 1, 2), inplace=False) - F.reshape(control_points, (1, 1, -1, 2))
            distances = F.sum(F.abs(distances), axis=distances.ndim - 1)

            result = distances ** 2
            result = result * F.log(distances + 1e-6)
            result = result * control_params
            result = F.sum(result, axis=2)
            result = F.reshape(
                result, (self.bs, coordinates.shape[1], 1), inplace=False)
            transformed = transformed + result

        return transformed

    def jacobian(self, coordinates):
        new_coordinates = self.warp_coordinates(coordinates)
        new_coordinates_x = F.slice(new_coordinates, start=(
            0, 0, 0), stop=new_coordinates.shape[:2] + (1,))
        grad_x = nn.grad([F.sum(new_coordinates_x)], [coordinates])
        new_coordinates_y = F.slice(new_coordinates, start=(
            0, 0, 1), stop=new_coordinates.shape[:2] + (2,))
        grad_y = nn.grad([F.sum(new_coordinates_y)], [coordinates])
        gx = F.reshape(grad_x[0], grad_x[0].shape[:-1] +
                       (1,) + grad_x[0].shape[-1:])
        gy = F.reshape(grad_y[0], grad_y[0].shape[:-1] +
                       (1,) + grad_y[0].shape[-1:])
        jacobian = F.concatenate(gx, gy, axis=gy.ndim-2)
        return jacobian
