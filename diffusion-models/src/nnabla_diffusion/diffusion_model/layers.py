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

from .utils import Shape4D, force_float

recompute = False


def set_recompute(flag):
    global recompute
    recompute = flag


def pad_for_faster_conv(x, *, channel_last=False):
    """
    Pad channel to meet the condition that tensor core is available to accelerate computation.
    """
    x_shape = Shape4D(x.shape, channel_last=channel_last)
    C = x_shape.c
    if C == 4 or C % 8 == 0:
        # no need to pad
        return x

    pad_width = 0
    # increment pad_width until the condition is satisfied.
    while C + pad_width != 4 and (C + pad_width) % 8 > 0:
        pad_width += 1

    if channel_last:
        return F.pad(x, (0, pad_width))
    else:
        return F.pad(x, (0, pad_width, 0, 0, 0, 0))


@force_float
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
    global recompute
    with nn.recompute(recompute):
        return F.swish(x)


@force_float
def group_norm(x, name, *, channel_axis=1, batch_axis=0):
    global recompute
    with nn.parameter_scope(name), nn.recompute(recompute):
        return PF.group_normalization(x,
                                      num_groups=32,
                                      # todo: use more stable eps for float16?
                                      channel_axis=channel_axis,
                                      batch_axis=batch_axis)


@force_float
def layer_norm(x, name, *, batch_axis=0):
    global recompute
    with nn.parameter_scope(name), nn.recompute(recompute):
        return PF.layer_normalization(x,
                                      # todo: use more stable eps for float16?
                                      batch_axis=batch_axis)


def conv(x, c, name, *, kernel=(3, 3), pad=(1, 1), stride=(1, 1), zeroing_w=False, channel_last=False):
    # init weight and bias with uniform, which is the same as pytorch
    c_axis = x.ndim - 1 if channel_last else 1
    lim = I.calc_normal_std_he_forward(x.shape[c_axis] * 2, c, tuple(kernel))
    w_init = I.UniformInitializer(lim=(-lim, lim), rng=None)
    b_init = I.UniformInitializer(lim=(-lim, lim), rng=None)

    if zeroing_w:
        w_init = I.ConstantInitializer(0)
        b_init = I.ConstantInitializer(0)

    return PF.convolution(x, c, kernel,
                          pad=pad, stride=stride, name=name,
                          w_init=w_init, b_init=b_init,
                          channel_last=channel_last)


def nin(x, c, name, *, zeroing_w=False, channel_last=False, recompute=False):
    c_axis = x.ndim - 1 if channel_last else 1
    lim = np.sqrt(x.shape[c_axis]) ** -1
    w_init = I.UniformInitializer(lim=(-lim, lim))  # same as pytorch's default
    b_init = I.UniformInitializer(lim=(-lim, lim))  # same as pytorch's default

    if zeroing_w:
        w_init = I.ConstantInitializer(0)
        b_init = I.ConstantInitializer(0)

    with nn.recompute(recompute):
        return PF.convolution(x, c,
                              kernel=(1 for _ in range(x.ndim - 2)),
                              pad=(0 for _ in range(x.ndim - 2)),
                              stride=(1 for _ in range(x.ndim - 2)),
                              name=name,
                              w_init=w_init,
                              b_init=b_init,
                              channel_last=channel_last)


def upsample(x, name, with_conv, *, channel_last=False, recompute=False):
    with nn.parameter_scope(name), nn.recompute(recompute):
        B, C, H, W = Shape4D(
            x.shape, channel_last=channel_last).get_as_tuple("bchw")
        x = F.interpolate(x, scale=(2, 2), mode="nearest",
                          align_corners=True, channel_last=channel_last)

        if with_conv:
            x = conv(x, C, "upsample_conv", channel_last=channel_last)

        assert Shape4D(x.shape, channel_last=channel_last) == \
            Shape4D((B, C, H * 2, W * 2), channel_last=False)  # reference

    return x


def downsample(x, name, with_conv, *, channel_last=False, recompute=False):
    with nn.parameter_scope(name), nn.recompute(recompute):
        B, C, H, W = Shape4D(
            x.shape, channel_last=channel_last).get_as_tuple("bchw")
        if with_conv:
            x = conv(x, C, "downsample_conv",
                     kernel=(3, 3), pad=(1, 1), stride=(2, 2), channel_last=channel_last)
        else:
            x = F.average_pooling(x, (2, 2), stride=(
                2, 2), channel_last=channel_last)

        assert Shape4D(x.shape, channel_last=channel_last) == \
            Shape4D((B, C, H // 2, W // 2), channel_last=False)  # reference

    return x


def chunk(x, num_chunk, axis, recompute=False):
    """
    Split `x` to `num_chunk` arrays along specified axis.
    """
    with nn.recompute(recompute):
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


def sqrt(x, recompute=False):
    assert isinstance(x, (nn.Variable, nn.NdArray))
    with nn.recompute(recompute):
        return x ** 0.5


def interp_like(x, arr, channel_last, mode="linear"):
    # interpolate spatial size of `x`` to be the same as `arr`.

    # get target shape from arr
    h, w = Shape4D(arr.shape, channel_last=channel_last).get_as_tuple("hw")

    # return x if x and arr has the same spatial size.
    if Shape4D(x.shape, channel_last=channel_last).get_as_tuple("hw") == (h, w):
        return x

    x_interp = F.interpolate(x, output_size=(
        h, w), mode=mode, channel_last=channel_last)

    return x_interp


def adaptive_pooling_2d(x, output_shape, channel_last, mode):
    # output_shape: (hight, width)
    assert len(output_shape) == 2

    h, w = Shape4D(x.shape, channel_last=channel_last).get_as_tuple("hw")
    mode = mode.lower()

    stride_h = int(h / output_shape[0])
    stride_w = int(w / output_shape[1])
    kernel_h = int(h - (output_shape[0] - 1) * stride_h)
    kernel_w = int(w - (output_shape[1] - 1) * stride_w)

    if mode == "average":
        return F.average_pooling(x, kernel=(kernel_h, kernel_w), stride=(stride_h, stride_w), channel_last=channel_last)
    if mode == "max":
        return F.max_pooling(x, kernel=(kernel_h, kernel_w), stride=(stride_h, stride_w), channel_last=channel_last)
    else:
        raise NotImplementedError(f"mode {mode} is not implemented.")
