# Copyright (c) 2017 Sony Corporation. All Rights Reserved.
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
import nnabla.parametric_functions as PF
from neu.tts.module import Module
from nnabla.initializer import NormalInitializer


def wn_conv(*args, **kwargs):
    return PF.convolution(
        *args, **kwargs,
        apply_w=PF.weight_normalization,
        w_init=NormalInitializer(0.01)
    )


def sn_conv(*args, **kwargs):
    return PF.convolution(
        *args, **kwargs,
        apply_w=lambda w: PF.spectral_norm(w, dim=0),
        w_init=NormalInitializer(0.01)
    )


def wn_deconv(*args, **kwargs):
    return PF.deconvolution(
        *args, **kwargs,
        apply_w=PF.weight_normalization,
        w_init=NormalInitializer(0.01)
    )


def res_block_1(x, c, k, d):
    r"""Residual block of type 1.

    Args:
        x (nn.Variable): Input variable of shape (B, C, L).
        c (int): Number of channels.
        k (int): Kernel size.
        d (tuple of `int`): Dilations.

    Returns:
        nn.Variable: Output variable.
    """
    for i in range(len(d)):
        with nn.parameter_scope(f"first_conv_{i}"):
            out = F.leaky_relu(x, 0.1)
            out = wn_conv(out, c, (k,), dilation=(d[i],),
                          pad=((k*d[i] - d[i])//2,))
        with nn.parameter_scope(f"second_conv_{i}"):
            out = F.leaky_relu(out, 0.1)
            out = wn_conv(out, c, (k, ), pad=((k-1)//2,))
            x = x + out
    return x


def res_block_2(x, c, k, d):
    r"""Residual block of type 2.

    Args:
        x (nn.Variable): Input variable of shape (B, C, L).
        c (int): Number of channels.
        k (int): Kernel size.
        d (tuple of `int`): Dilations.

    Returns:
        nn.Variable: Output variable.
    """
    for i in range(len(d)):
        with nn.parameter_scope(f"conv_{i}"):
            out = F.leaky_relu(x, 0.1)
            out = wn_conv(out, c, (k,), dilation=(d[i],),
                          pad=((k*d[i] - d[i])//2,))
            x = x + out
    return x


class UpBlock(Module):
    def __init__(self, hp):
        self.hp = hp

    def call(self, x, i):
        hp = self.hp
        dilations = hp.resblock_dilation_sizes
        kernel_sizes = hp.resblock_kernel_sizes
        res_block = res_block_1 if hp.resblock == "1" else res_block_2
        channel = hp.upsample_initial_channel // (2**(i + 1))
        r = hp.upsample_rates[i]

        with nn.parameter_scope("deconv"):
            x = F.leaky_relu(x, 0.1)
            x = wn_deconv(
                x, channel, (r * 2, ),
                stride=(r,), pad=(r // 2 + r % 2,),
            )

        out = list()
        for j, (k, d) in enumerate(zip(kernel_sizes, dilations)):
            with nn.parameter_scope(f"resblock_{j}"):
                out.append(res_block(x, channel, k, d))

        x = sum(out) / len(kernel_sizes)

        return x
