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


"""
DLA primitives and full network models.
"""

import numpy as np
import nnabla as nn
import nnabla.parametric_functions as PF
import nnabla.functions as F

from nnabla.initializer import UniformInitializer, ConstantInitializer, NormalInitializer, calc_normal_std_he_forward, calc_normal_std_he_backward
from nnabla.logger import logger
from nnabla.utils.save import save
from nnabla.utils.nnp_graph import NnpNetworkPass

from models.networks.initializers import he_initializer, bilinear_depthwise_initializer, bilinear_initializer

RNG = np.random.RandomState(214)


def pf_depthwise_deconvolution(x, kernel, stride=(1, 1), pad=(1, 1), dilation=(2, 2), with_bias=False, w_init=None, b_init=None, channel_last=False):
    out_map = x.shape[3] if channel_last else x.shape[1]
    if channel_last:
        w_init = np.transpose(w_init, (0, 2, 3, 1))
    x = PF.deconvolution(
        x,
        out_map,
        kernel,
        pad=pad,
        stride=stride,
        dilation=dilation,
        w_init=w_init,
        with_bias=with_bias,
        b_init=b_init,
        group=out_map,
        channel_last=channel_last
        )
    return x


def pf_affine(r, num_classes=1000, channel_last=False):
    r = PF.convolution(r, num_classes, (1, 1), channel_last=channel_last,
                       w_init=NormalInitializer(sigma=0.01, rng=RNG), name='fc')
    return F.reshape(r, (r.shape[0], -1), inplace=False)


def pf_convolution(x, ochannels, kernel, pad=None, stride=(1, 1), dilation=None, with_bias=False, w_init=None, b_init=None, channel_last=False):
    return PF.convolution(x, ochannels, kernel, stride=stride, pad=pad, dilation=dilation,
                          with_bias=with_bias, w_init=w_init, b_init=b_init, channel_last=channel_last)


def shortcut(x, ochannels, stride, shortcut_type, test, channel_last=False):
    axes = 3 if channel_last else 1
    ichannels = x.shape[axes]
    use_conv = shortcut_type.lower() == 'c'
    if ichannels != ochannels:
        assert (ichannels * 2 == ochannels) or (ichannels * 4 == ochannels)
        if shortcut_type.lower() == 'b':
            use_conv = True
    if use_conv:
        # Convolution does everything.
        # Matching channels, striding.
        with nn.parameter_scope("shortcut_conv"):
            x = PF.convolution(x, ochannels, (1, 1),
                               stride=(stride, stride), with_bias=False, channel_last=channel_last)
            x = PF.batch_normalization(x, axes=[axes], batch_stat=not test)
    else:
        # shortcut block is slightly different for dla
        if stride != 1:
            # Stride
            x = F.max_pooling(
                x, kernel=(
                    stride, stride), stride=(
                    stride, stride), channel_last=channel_last)
        if ichannels != ochannels:
            x = PF.convolution(
                x, ochannels, (1, 1), stride=(
                    1, 1), with_bias=False, channel_last=channel_last)
            x = PF.batch_normalization(x, axes=[axes], batch_stat=not test)

    return x


def basicblock(x, residual, ochannels, stride, test, channel_last=False):
    def bn(h):
        axes = [3 if channel_last else 1]
        return PF.batch_normalization(h, axes=axes, batch_stat=not test)
    if residual is None:
        residual = x
    with nn.parameter_scope("basicblock1"):
        h = F.relu(bn(PF.convolution(x, ochannels, (3, 3), stride=(
            stride, stride), pad=(1, 1), with_bias=False, channel_last=channel_last)))
    with nn.parameter_scope("basicblock2"):
        h = bn(
            PF.convolution(
                h, ochannels, (3, 3), pad=(
                    1, 1), with_bias=False, channel_last=channel_last))
    return F.relu(F.add2(h, residual))


def bottleneck(x, ochannels, shortcut_type, stride, test, channel_last=False):
    def bn(h):
        axes = [3 if channel_last else 1]
        return PF.batch_normalization(h, axes=axes, batch_stat=not test)
    assert ochannels % 4 == 0
    hchannels = ochannels / 4
    with nn.parameter_scope("bottleneck1"):
        h = F.relu(
            bn(PF.convolution(x, hchannels, (1, 1),
                              with_bias=False, channel_last=channel_last))
            )
    with nn.parameter_scope("bottleneck2"):
        h = F.relu(
            bn(PF.convolution(h, hchannels, (3, 3), pad=(1, 1),
                              stride=stride, with_bias=False, channel_last=channel_last)))
    with nn.parameter_scope("bottleneck3"):
        h = bn(PF.convolution(h, ochannels, (1, 1),
                              with_bias=False, channel_last=channel_last))
    with nn.parameter_scope("bottleneck_s"):
        s = shortcut(x, ochannels, stride, shortcut_type, test, channel_last)
    return F.relu(F.add2(h, s))


def layer(x, block, ochannels, count, stride, shortcut_type, test, channel_last=False):
    for i in range(count):
        with nn.parameter_scope("layer{}".format(i + 1)):
            x = block(x, ochannels, stride if i ==
                      0 else (1, 1), shortcut_type, test, channel_last=channel_last)
    return x


def _make_conv_level(x, ochannels, convs, test, stride=1, dilation=1, channel_last=False):
    axes = [3 if channel_last else 1]
    for i in range(convs):
        with nn.parameter_scope("conv{}".format(i + 1)):
            s = (stride, stride) if i == 0 else (1, 1)
            x = pf_convolution(
                x, ochannels, (3, 3), stride=s,
                pad=(dilation, dilation),
                dilation=(dilation, dilation),
                with_bias=False,
                channel_last=channel_last)
            x = F.relu(PF.batch_normalization(
                x, axes=axes, batch_stat=not test))
    return x


def root(x, children, ochannels, test, concat_axis=1, kernel_size=1, channel_last=False):
    axes = 3 if channel_last else 1
    with nn.parameter_scope("root"):
        rng = np.random.RandomState(313)
        x = F.concatenate(x, *children, axis=axes)
        x = pf_convolution(
            x, ochannels, (kernel_size, kernel_size), pad=((kernel_size-1)//2, (kernel_size-1)//2), stride=(
                1, 1),
            with_bias=False,
            w_init=he_initializer(ochannels, kernel_size, rng),
            channel_last=channel_last
        )
        x = PF.batch_normalization(x, axes=[axes], batch_stat=not test)
        x = F.relu(x)
    return x


def upsample(x, ochannels, test, kernel_size=4, channel_last=False):
    rng = np.random.RandomState(313)
    axes = 3 if channel_last else 1
    with nn.parameter_scope("up"):
        x = pf_convolution(
            x, ochannels, (1, 1), stride=(
                1, 1),
            with_bias=False,
            w_init=he_initializer(ochannels, kernel_size, rng),
            channel_last=channel_last
        )
        x = F.relu(
            PF.batch_normalization(
                x,
                axes=[axes],
                batch_stat=not test)
            )
        ichannels = x.shape[axes]
        x = pf_depthwise_deconvolution(
            x,
            (kernel_size, kernel_size),
            pad=(1, 1),
            stride=(2, 2),
            dilation=(1, 1),
            with_bias=False,
            w_init=bilinear_depthwise_initializer(ichannels, kernel_size),
            channel_last=channel_last
        )
    return x


def _make_tree_level1(
        x,
        children,
        block,
        ochannels,
        level,
        test,
        level_root=False,
        stride=1,
        channel_last=False
        ):
    axes = 3 if channel_last else 1
    ichannels = x.shape[axes]
    bottom = F.max_pooling(
        x,
        kernel=(stride, stride),
        stride=(stride, stride),
        channel_last=channel_last
        ) if stride > 1 else x
    if ichannels != ochannels:
        residual = pf_convolution(
            bottom, ochannels, (1, 1), stride=(1, 1), pad=None, with_bias=False, channel_last=channel_last)
        residual = PF.batch_normalization(
            residual, axes=[axes], batch_stat=not test)
    else:
        residual = bottom
    with nn.parameter_scope('block1'):
        b1 = block(x, residual, ochannels, stride,
                   test, channel_last=channel_last)
    with nn.parameter_scope('block2'):
        b2 = block(b1, b1, ochannels, 1, test, channel_last=channel_last)
    _children = [bottom, b2] if level_root else [b2]
    if children:
        _children += children
    x = root(b1, _children, ochannels, test,
             kernel_size=1, channel_last=channel_last)
    return x, bottom


def _make_tree_level2(
        x,
        children,
        block,
        ochannels,
        level,
        test,
        level_root=False,
        stride=1,
        channel_last=False):
    with nn.parameter_scope('node1'):
        ag1, bottom1 = _make_tree_level1(
            x, None, block, ochannels, level, test, False, stride, channel_last=channel_last)
    with nn.parameter_scope('node2'):
        x, _ = _make_tree_level1(
            ag1, [bottom1], block, ochannels, level, test, level_root, 1, channel_last=channel_last)
    return x


def dla_imagenet(
        x,
        num_classes,
        num_layers,
        test,
        residual_root=False,
        tiny=False,
        channel_last=False):
    """
    Args:
        x : Variable
        num_classes : Number of classes of outputs
        num_layers : Number of layers of DLA chosen from (34).
        test : Construct net for testing.
        tiny (bool): Tiny imagenet mode. Input image must be (3, 56, 56).
    """
    layers = {
        # 18: ((2, 2, 2, 2), basicblock, 1),
        34: ((1, 1, 1, 2, 2, 1), (False, False, False, True, True, True), basicblock)
        # 50: ((3, 4, 6, 3), bottleneck, 4),
        # 101: ((3, 4, 23, 3), bottleneck, 4),
        # 152: ((3, 8, 36, 3), bottleneck, 4)
    }

    ochannels = [16, 32, 64, 128, 256, 512]
    levels, levels_root, block = layers[num_layers]
    strides = [1, 2, 2, 2, 2, 2]
    logger.debug(x.shape)
    axes = 3 if channel_last else 1

    with nn.parameter_scope("conv1"):
        stride = (1, 1)
        r = pf_convolution(x, 16, (7, 7),
                           pad=(3, 3), stride=stride, with_bias=False, channel_last=channel_last)
        r = F.relu(PF.batch_normalization(
            r, axes=[axes], batch_stat=not test))
    hidden = {}
    hidden['conv0'] = r
    logger.debug(r.shape)
    with nn.parameter_scope("level0"):
        r = _make_conv_level(
            r,
            ochannels[0],
            levels[0],
            test=test,
            stride=strides[0],
            channel_last=channel_last)
        hidden['level0'] = r
        logger.debug(r.shape)
    with nn.parameter_scope("level1"):
        r = _make_conv_level(
            r,
            ochannels[1],
            levels[1],
            test=test,
            stride=strides[1],
            channel_last=channel_last)
        hidden['level1'] = r
        logger.debug(r.shape)
    with nn.parameter_scope("level2"):
        r, _ = _make_tree_level1(
            r, None, block, ochannels[2], levels[2], test, levels_root[2], stride=strides[2], channel_last=channel_last)
        hidden['level2'] = r
        logger.debug(r.shape)
    with nn.parameter_scope("level3"):
        r = _make_tree_level2(
            r,
            None,
            block,
            ochannels[3],
            levels[3],
            test,
            levels_root[3],
            stride=strides[3],
            channel_last=channel_last)
        hidden['level3'] = r
        logger.debug(r.shape)
    with nn.parameter_scope("level4"):
        r = _make_tree_level2(
            r,
            None,
            block,
            ochannels[4],
            levels[4],
            test,
            levels_root[4],
            stride=strides[4],
            channel_last=channel_last)
        hidden['level4'] = r
        logger.debug(r.shape)
    with nn.parameter_scope("level5"):
        r, _ = _make_tree_level1(
            r, None, block, ochannels[5], levels[5], test, levels_root[5], stride=strides[5], channel_last=channel_last)
        hidden['level5'] = r
        logger.debug(r.shape)
    pool_shape = r.shape[-2:]
    if channel_last:
        pool_shape = r.shape[1:3]
    r = F.average_pooling(r, pool_shape, channel_last=channel_last)
    with nn.parameter_scope("fc"):
        r = pf_affine(r, num_classes, channel_last=channel_last)
    logger.debug(r.shape)

    return r, hidden

# Upsampling portion of DLA


def DLAUp(x, test, residual_root=False, channel_last=False):
    r, hidden = dla_imagenet(
        x, num_classes=1000, num_layers=34, test=test, channel_last=channel_last)
    callback = NnpNetworkPass(True)
    callback.remove_and_rewire('fc')
    ochannels = [256, 128, 64, 32]
    with nn.parameter_scope("up16"):
        x = upsample(hidden['level5'], ochannels[0], test,
                     kernel_size=4, channel_last=channel_last)
        hidden['up16'] = x
    with nn.parameter_scope("up8"):
        x = root(x, [hidden['level4']], ochannels[0], test,
                 kernel_size=3, channel_last=channel_last)
        x = upsample(x, ochannels[1], test,
                     kernel_size=4, channel_last=channel_last)
        hidden['up8'] = x
    with nn.parameter_scope("up4"):
        with nn.parameter_scope("residual_level3"):
            level4up = upsample(
                hidden['level4'], ochannels[1], test, kernel_size=4, channel_last=channel_last)
            with nn.parameter_scope("level3up_root"):
                level3up = root(
                    level4up, [hidden['level3']], ochannels[1], test, kernel_size=3, channel_last=channel_last)
            with nn.parameter_scope("x_root"):
                x = root(x, [level3up], ochannels[1], test,
                         kernel_size=1, channel_last=channel_last)
        x = upsample(x, ochannels[2], test,
                     kernel_size=4, channel_last=channel_last)
        hidden['up4'] = x
    with nn.parameter_scope("up2_b"):
        level3up_b = upsample(
            level3up, ochannels[2], test, kernel_size=4, channel_last=channel_last)
    with nn.parameter_scope("up2_c"):
        level3up_c = upsample(
            hidden['level3'], ochannels[2], test, kernel_size=4, channel_last=channel_last)
        with nn.parameter_scope("level3up_c_root"):
            level3up_c = root(hidden['level2'], [
                              level3up_c], ochannels[2], test, kernel_size=3, channel_last=channel_last)
        with nn.parameter_scope("level2up_root"):
            level2up = root(level3up_b, [level3up_c],
                            ochannels[2], test, kernel_size=3, channel_last=channel_last)
        with nn.parameter_scope("x_root"):
            x = root(x, [level2up], ochannels[2], test,
                     kernel_size=3, channel_last=channel_last)
    return x
