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
from .ops import upsample_conv_2d, downsample_2d, weight_init_fn

import subprocess as sp


def mapping_network(z, outmaps=512, num_layers=8, net_scope='G_mapping/Dense'):

    lrmul = 0.01
    runtime_coef = 0.00044194172

    out = z
    for i in range(num_layers):
        with nn.parameter_scope(f'{net_scope}{i}'):
            W, bias = weight_init_fn(
                shape=(out.shape[1], outmaps), lrmul=lrmul)
            out = F.affine(out, W*runtime_coef, bias*lrmul)
            out = F.mul_scalar(F.leaky_relu(
                out, alpha=0.2, inplace=False), np.sqrt(2), inplace=False)
    return out


def styled_conv_block(conv_input, w, noise=None, res=4, inmaps=512, outmaps=512, kernel_size=3,
                      pad_size=1, demodulate=True, namescope="Conv", up=False, act=F.leaky_relu):
    """
    Conv block with skip connection for Generator
    """
    batch_size = conv_input.shape[0]

    with nn.parameter_scope(f'G_synthesis/{res}x{res}/{namescope}'):
        W, bias = weight_init_fn(shape=(w.shape[1], inmaps))
        runtime_coef = (1. / np.sqrt(512)).astype(np.float32)
        style = F.affine(w, W*runtime_coef, bias) + 1.0
    runtime_coef_for_conv = (
        1/np.sqrt(np.prod([inmaps, kernel_size, kernel_size]))).astype(np.float32)

    if up:
        init_function = weight_init_fn(
            shape=(inmaps, outmaps, kernel_size, kernel_size), return_init=True)
        conv_weight = nn.parameter.get_parameter_or_create(name=f'G_synthesis/{res}x{res}/{namescope}/conv/W',
                                                           shape=(inmaps, outmaps, kernel_size, kernel_size), initializer=init_function)
    else:
        init_function = weight_init_fn(
            shape=(outmaps, inmaps, kernel_size, kernel_size), return_init=True)
        conv_weight = nn.parameter.get_parameter_or_create(name=f'G_synthesis/{res}x{res}/{namescope}/conv/W',
                                                           shape=(outmaps, inmaps, kernel_size, kernel_size), initializer=init_function)
    conv_weight = F.mul_scalar(conv_weight, runtime_coef_for_conv)
    if up:
        scale = F.reshape(
            style, (style.shape[0], style.shape[1], 1, 1, 1), inplace=False)
    else:
        scale = F.reshape(
            style, (style.shape[0], 1, style.shape[1], 1, 1), inplace=False)

    mod_w = F.mul2(F.reshape(conv_weight, (1,) +
                   conv_weight.shape, inplace=False), scale)

    if demodulate:
        if up:
            denom_w = F.pow_scalar(F.sum(F.pow_scalar(mod_w, 2.), axis=[
                                   1, 3, 4], keepdims=True) + 1e-8, 0.5)
        else:
            denom_w = F.pow_scalar(F.sum(F.pow_scalar(mod_w, 2.), axis=[
                                   2, 3, 4], keepdims=True) + 1e-8, 0.5)
        demod_w = F.div2(mod_w, denom_w)

    else:
        demod_w = mod_w

    conv_input = F.reshape(
        conv_input, (1, -1, conv_input.shape[2], conv_input.shape[3]), inplace=False)
    demod_w = F.reshape(
        demod_w, (-1, demod_w.shape[2], demod_w.shape[3], demod_w.shape[4]), inplace=False)

    if up:
        k = [1, 3, 3, 1]
        conv_out = upsample_conv_2d(
            conv_input, demod_w, k, factor=2, gain=1, group=batch_size)
    else:
        conv_out = F.convolution(conv_input, demod_w, pad=(
            pad_size, pad_size), group=batch_size)
        conv_out = F.reshape(
            conv_out, (batch_size, -1, conv_out.shape[2], conv_out.shape[3]), inplace=False)

    if noise is not None:
        noise_coeff = nn.parameter.get_parameter_or_create(
            name=f'G_synthesis/{res}x{res}/{namescope}/noise_strength', shape=())
        conv_out = F.add2(conv_out, noise*F.reshape(noise_coeff, (1, 1, 1, 1)))
    else:
        conv_out = conv_out

    bias = nn.parameter.get_parameter_or_create(name=f'G_synthesis/{res}x{res}/{namescope}/conv/b', shape=(
        outmaps,), initializer=np.random.randn(outmaps,).astype(np.float32))
    conv_out = F.add2(conv_out, F.reshape(
        bias, (1, outmaps, 1, 1), inplace=False))

    if act == F.leaky_relu:
        conv_out = F.mul_scalar(F.leaky_relu(
            conv_out, alpha=0.2, inplace=False), np.sqrt(2), inplace=False)
    else:
        conv_out = act(conv_out)

    return conv_out


def conv_layer(conv_input, inmaps, outmaps, kernel_size, downsample=False,
               bias=True, act=F.leaky_relu, name_scope='Conv'):
    """
    Conv layer for the residual block of the discriminator
    """

    if downsample:
        k = [1, 3, 3, 1]
        out = downsample_2d(conv_input, k, factor=2,
                            gain=1, kernel_size=kernel_size)
        stride = 2
        pad = 0
    else:
        stride = 1
        pad = kernel_size//2
        out = conv_input

    init_function = weight_init_fn(
        shape=(outmaps, inmaps, kernel_size, kernel_size), return_init=True)
    scale = 1/np.sqrt(inmaps*kernel_size**2)

    conv_weight = nn.parameter.get_parameter_or_create(name=f'{name_scope}/W', initializer=init_function,
                                                            shape=(outmaps, inmaps, kernel_size, kernel_size))
    if bias:
        conv_bias = nn.parameter.get_parameter_or_create(
            name=f'{name_scope}/b', shape=(outmaps,))
    else:
        conv_bias = None

    out = F.convolution(out, conv_weight*scale, bias=conv_bias,
                        stride=(stride, stride), pad=(pad, pad))

    if act == F.leaky_relu:
        out = F.mul_scalar(F.leaky_relu(
            out, alpha=0.2, inplace=False), np.sqrt(2), inplace=False)
    else:
        out = act(out)

    return out


def res_block(res_input, res, inmaps, outmaps, block_scope='res_block'):
    """
    Residual block for Discriminator
    """

    name_scope = f'Discriminator/{block_scope}_{res}x{res}'

    out = conv_layer(res_input, inmaps, inmaps, kernel_size=3,
                     name_scope=f'{name_scope}/Conv1')
    out = conv_layer(out, inmaps, outmaps, kernel_size=3,
                     downsample=True, name_scope=f'{name_scope}/Conv2')

    skip = conv_layer(res_input, inmaps, outmaps, kernel_size=1, downsample=True,
                      bias=False, act=F.identity, name_scope=f'{name_scope}/ConvSkip')

    out = F.mul_scalar(F.add2(out, skip), 1 /
                       np.sqrt(2).astype(np.float32), inplace=False)

    return out
