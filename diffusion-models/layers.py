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


def sinusoidal_embedding(timesteps, embedding_dim):
    """
    Sinusoidal embeddings originally proposed in "Attention Is All You Need" (https://arxiv.org/abs/1706.03762).
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    denominator = -np.log(10000) / half_dim
    emb = F.exp(denominator * F.arange(start=0, stop=half_dim))
    emb = F.reshape(timesteps, (-1, 1)) * F.reshape(emb, (1, -1))
    emb = F.concatenate(F.cos(emb), F.sin(emb), axis=1)

    if embedding_dim & 1:  # zero pad to be divisible by two
        emb = F.pad(emb, [[0, 0], [0, 1]])

    assert emb.shape == (timesteps.shape[0], embedding_dim)

    return emb


def nonlinearity(x):
    return F.swish(x)


def normalize(x, name):
    with nn.parameter_scope(name):
        return PF.group_normalization(x, num_groups=32)


def conv(x, c, name, kernel=(3, 3), pad=(1, 1), stride=(1, 1), zeroing_w=False):
    # init weight and bias with uniform, which is the same as pytorch
    lim = I.calc_normal_std_he_forward(x.shape[1] * 2, c, tuple(kernel))
    w_init = I.UniformInitializer(lim=(-lim, lim), rng=None)
    b_init = I.UniformInitializer(lim=(-lim, lim), rng=None)

    if zeroing_w:
        w_init = I.ConstantInitializer(0)
        b_init = I.ConstantInitializer(0)

    return PF.convolution(x, c, kernel,
                          pad=pad, stride=stride, name=name,
                          w_init=w_init, b_init=b_init)


def nin(x, c, name, zeroing_w=False):
    lim = np.sqrt(x.shape[1]) ** -1
    w_init = I.UniformInitializer(lim=(-lim, lim))  # same as pytorch's default
    b_init = I.UniformInitializer(lim=(-lim, lim))  # same as pytorch's default

    if zeroing_w:
        w_init = I.ConstantInitializer(0)
        b_init = I.ConstantInitializer(0)

    return PF.convolution(x, c,
                          kernel=(1, 1),
                          pad=(0, 0), stride=(1, 1), name=name,
                          w_init=w_init, b_init=b_init)


def upsample(x, name, with_conv):
    with nn.parameter_scope(name):
        B, C, H, W = x.shape
        x = F.interpolate(x, scale=(2, 2), mode="nearest", align_corners=True)
        assert x.shape == (B, C, H * 2, W * 2)
        if with_conv:
            x = conv(x, C, "upsample_conv")
            assert x.shape == (B, C, H * 2, W * 2)
        return x


def downsample(x, name, with_conv):
    with nn.parameter_scope(name):
        B, C, H, W = x.shape
        if with_conv:
            x = conv(x, C, "downsample_conv",
                     kernel=(3, 3), pad=(1, 1), stride=(2, 2))
        else:
            x = F.average_pooling(x, (2, 2), stride=(2, 2))

        assert x.shape == (B, C, H // 2, W // 2)
        return x


def chunk(x, num_chunk, axis):
    """
    Split `x` to `num_chunk` arrays along specified axis.
    """
    shape = x.shape
    C = shape[axis]
    num_elems = (C + num_chunk - 1) // num_chunk

    ret = []
    for i in range(num_chunk):
        start = [0 for _ in shape]
        stop = [s for s in shape]
        start[axis] = i * num_elems
        stop[axis] = start[axis] + num_elems

        segment = F.slice(x, start=start, stop=stop)
        assert len(segment.shape) == len(x.shape)
        ret.append(segment)

    return ret


def sqrt(x):
    assert isinstance(x, (nn.Variable, nn.NdArray))
    return x ** 0.5
