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


def make_coordinate_grid(spatial_size):
    assert isinstance(spatial_size, tuple)

    h, w = spatial_size
    x = F.arange(0, w)
    y = F.arange(0, h)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    yy = F.tile(F.reshape(y, (-1, 1)), (1, w))
    xx = F.tile(F.reshape(x, (1, -1)), (h, 1))

    meshed = F.concatenate(F.reshape(xx, xx.shape + (1,)),
                           F.reshape(yy, xx.shape + (1,)), axis=2)

    return meshed


def kp2gaussian(kp, spatial_size, kp_variance):
    mean = kp['value']

    coordinate_grid = make_coordinate_grid(spatial_size)
    number_of_leading_dimensions = len(mean.shape) - 1
    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape
    coordinate_grid = F.reshape(coordinate_grid, shape)
    coordinate_grid = F.broadcast(
        coordinate_grid, mean.shape[:number_of_leading_dimensions] + coordinate_grid.shape[number_of_leading_dimensions:])

    # Preprocess kp shape
    shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 2)
    mean = F.reshape(mean, shape, inplace=False)

    mean_sub = coordinate_grid - mean

    out = F.exp(-0.5 * F.sum((mean_sub ** 2),
                             axis=mean_sub.ndim - 1) / kp_variance)

    return out


def resblock(x, in_features: int, kernel_size: int, padding: int, test: bool = False, comm=None):
    if comm:
        batchnorm = functools.partial(PF.sync_batch_normalization,
                                      comm=comm, group='world',
                                      axes=[1], decay_rate=0.9, eps=1e-05,
                                      batch_stat=not test)
    else:
        # 1 GPU
        batchnorm = functools.partial(PF.batch_normalization,
                                      axes=[1], decay_rate=0.9, eps=1e-05,
                                      batch_stat=not test)

    inmaps, outmaps = x.shape[1], in_features
    k_w = I.calc_normal_std_he_forward(
        inmaps, outmaps, kernel=(kernel_size, kernel_size)) / np.sqrt(2.)
    k_b = I.calc_normal_std_he_forward(inmaps, outmaps) / np.sqrt(2.)
    w_init = I.UniformInitializer((-k_w, k_w))
    b_init = I.UniformInitializer((-k_b, k_b))

    with nn.parameter_scope("convblock_0"):
        out = batchnorm(x)
        out = F.relu(out)
        out = PF.convolution(out, outmaps=in_features,
                             kernel=(kernel_size, kernel_size),
                             pad=(padding, padding),
                             w_init=w_init, b_init=b_init)

    with nn.parameter_scope("convblock_2"):
        out = batchnorm(out)
        out = F.relu(out)
        out = PF.convolution(out, outmaps=in_features,
                             kernel=(kernel_size, kernel_size),
                             pad=(padding, padding),
                             w_init=w_init, b_init=b_init)
    out = F.add2(out, x)
    return out


def upblock(x, out_features, kernel_size=3, padding=1, groups=1, test=False, comm=None):
    if comm:
        batchnorm = functools.partial(PF.sync_batch_normalization,
                                      comm=comm, group='world',
                                      axes=[1], decay_rate=0.9, eps=1e-05,
                                      batch_stat=not test)
    else:
        # 1 GPU
        batchnorm = functools.partial(PF.batch_normalization,
                                      axes=[1], decay_rate=0.9, eps=1e-05,
                                      batch_stat=not test)

    inmaps, outmaps = x.shape[1], out_features
    k_w = I.calc_normal_std_he_forward(
        inmaps, outmaps, kernel=(kernel_size, kernel_size)) / np.sqrt(2.)
    k_b = I.calc_normal_std_he_forward(inmaps, outmaps) / np.sqrt(2.)
    w_init = I.UniformInitializer((-k_w, k_w))
    b_init = I.UniformInitializer((-k_b, k_b))

    out = F.interpolate(x, scale=(2, 2), mode="nearest")
    with nn.parameter_scope("upblock"):
        out = PF.convolution(out, outmaps=out_features,
                             kernel=(kernel_size, kernel_size),
                             pad=(padding, padding),
                             group=groups,
                             w_init=w_init, b_init=b_init)
        out = batchnorm(out)
    out = F.relu(out)
    return out


def downblock(x, out_features, kernel_size=3, padding=1, groups=1, test=False, comm=None):
    if comm:
        batchnorm = functools.partial(PF.sync_batch_normalization,
                                      comm=comm, group='world',
                                      axes=[1], decay_rate=0.9, eps=1e-05,
                                      batch_stat=not test)
    else:
        # 1 GPU
        batchnorm = functools.partial(PF.batch_normalization,
                                      axes=[1], decay_rate=0.9, eps=1e-05,
                                      batch_stat=not test)

    inmaps, outmaps = x.shape[1], out_features
    k_w = I.calc_normal_std_he_forward(
        inmaps, outmaps, kernel=(kernel_size, kernel_size)) / np.sqrt(2.)
    k_b = I.calc_normal_std_he_forward(inmaps, outmaps) / np.sqrt(2.)
    w_init = I.UniformInitializer((-k_w, k_w))
    b_init = I.UniformInitializer((-k_b, k_b))

    with nn.parameter_scope("downblock"):
        out = PF.convolution(x, outmaps=out_features,
                             kernel=(kernel_size, kernel_size),
                             pad=(padding, padding),
                             group=groups,
                             w_init=w_init, b_init=b_init)
        out = batchnorm(out)
    out = F.relu(out)
    out = F.average_pooling(out, kernel=(2, 2))
    return out


def sameblock(x, out_features, kernel_size=3, padding=1, groups=1, test=False, comm=None):
    if comm:
        batchnorm = functools.partial(PF.sync_batch_normalization,
                                      comm=comm, group='world',
                                      axes=[1], decay_rate=0.9, eps=1e-05,
                                      batch_stat=not test)
    else:
        # 1 GPU
        batchnorm = functools.partial(PF.batch_normalization,
                                      axes=[1], decay_rate=0.9, eps=1e-05,
                                      batch_stat=not test)

    inmaps, outmaps = x.shape[1], out_features
    k_w = I.calc_normal_std_he_forward(
        inmaps, outmaps, kernel=(kernel_size, kernel_size)) / np.sqrt(2.)
    k_b = I.calc_normal_std_he_forward(inmaps, outmaps) / np.sqrt(2.)
    w_init = I.UniformInitializer((-k_w, k_w))
    b_init = I.UniformInitializer((-k_b, k_b))

    with nn.parameter_scope("downblock"):
        out = PF.convolution(x, outmaps=out_features,
                             kernel=(kernel_size, kernel_size),
                             pad=(padding, padding),
                             group=groups,
                             w_init=w_init, b_init=b_init)
        out = batchnorm(out)
    out = F.relu(out)
    return out


def encoder(x, block_expansion: int, num_blocks=3, max_features=256, test=False, comm=None):
    down_blocks = []
    outs = [x]

    for i in range(num_blocks):
        down_block = functools.partial(downblock,
                                       out_features=min(
                                           max_features, block_expansion * (2 ** (i + 1))),
                                       kernel_size=3,
                                       padding=1,
                                       test=test,
                                       comm=comm)
        down_blocks.append(down_block)

    for i, down_block in enumerate(down_blocks):
        with nn.parameter_scope(f"downblock_{i}"):
            outs.append(down_block(outs[-1]))
    return outs


def decoder(x: list, block_expansion: int, num_blocks=3, max_features=256, test=False, comm=None):
    up_blocks = []

    for i in range(num_blocks)[::-1]:
        up_block = functools.partial(upblock,
                                     out_features=min(
                                         max_features, block_expansion * (2 ** i)),
                                     kernel_size=3,
                                     padding=1,
                                     test=test,
                                     comm=comm)
        up_blocks.append(up_block)

    out = x.pop()  # Variable((B, 256, 32, 32)), the last feature from encoder

    for i, up_block in enumerate(up_blocks):
        with nn.parameter_scope(f"upblock_{i}"):
            out = up_block(out)
        skip = x.pop()
        out = F.concatenate(out, skip, axis=1)

    return out


def hourglass(x, block_expansion, num_blocks=3, max_features=256, test=False, comm=None):
    with nn.parameter_scope("encoder"):
        encoder_outputs = encoder(x,
                                  block_expansion=block_expansion,
                                  num_blocks=num_blocks,
                                  max_features=max_features,
                                  test=test, comm=comm)

    with nn.parameter_scope("decoder"):
        output = decoder(encoder_outputs,
                         block_expansion=block_expansion,
                         num_blocks=num_blocks,
                         max_features=max_features,
                         test=test, comm=comm)
    return output


def anti_alias_interpolate(input, channels, scale):
    # no trainable parameters exist.
    if scale == 1.0:
        # no interpolation executed
        return F.identity(input)

    sigma = (1 / scale - 1) / 2
    kernel_size = 2 * round(sigma * 4) + 1
    ka = kernel_size // 2
    if kernel_size % 2 == 0:
        kb = ka - 1
    else:
        kb = ka

    kernel_size = [kernel_size, kernel_size]
    sigma = [sigma, sigma]
    kernel = 1

    xa = F.reshape(F.arange(0, kernel_size[0]), (-1, 1))
    ya = F.reshape(F.arange(0, kernel_size[1]), (1, -1))
    meshgrids = (F.tile(xa, (1, kernel_size[1])), F.tile(
        ya, (kernel_size[0], 1)))

    for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
        mean = (size - 1) / 2
        kernel *= F.exp(-(mgrid - mean) ** 2 / (2 * std ** 2))

    kernel = kernel / F.sum(kernel, keepdims=True)
    # Reshape to depthwise convolutional weight
    kernel = F.reshape(kernel, (1, 1) + kernel.shape)
    kernel = F.broadcast(kernel, (channels, 1) + tuple(kernel_size))
    # if using the pre-computed kernel, no need to compute here.

    out = F.pad(input, (ka, kb, ka, kb))
    out = F.convolution(out, weight=kernel, group=channels)
    out = F.interpolate(out, scale=(scale, scale), mode="nearest")

    return out
