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

import os
import sys
ops_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'stylegan2'))
sys.path.append(ops_path)
from ops import _setup_kernel, lerp, upsample_2d, upsample_conv_2d, weight_init_fn


def _simple_upfirdn_2d(x, k, up=1, down=1, pad0=0, pad1=0):
    assert x.ndim == 4
    y = x
    y = F.reshape(y, [-1, y.shape[2], y.shape[3], 1], inplace=False)
    y = upfirdn_2d(y, k, upx=up, upy=up, downx=down, downy=down,
                   padx0=pad0, padx1=pad1, pady0=pad0, pady1=pad1)
    y = F.reshape(y, [-1, x.shape[1], y.shape[1], y.shape[2]], inplace=False)
    return y


def upfirdn_2d(x, k, upx=1, upy=1, downx=1, downy=1, padx0=0, padx1=0, pady0=0, pady1=0):
    assert isinstance(x, nn.Variable) or (x, nn.NdArray)
    k = np.asarray(k, dtype=np.float32)
    assert x.ndim == 4
    inH = x.shape[1]
    inW = x.shape[2]
    minorDim = x.shape[3]
    kernelH, kernelW = k.shape
    assert inW >= 1 and inH >= 1
    assert kernelW >= 1 and kernelH >= 1
    assert isinstance(upx, int) and isinstance(upy, int)
    assert isinstance(downx, int) and isinstance(downy, int)
    assert isinstance(padx0, int) and isinstance(padx1, int)
    assert isinstance(pady0, int) and isinstance(pady1, int)

    x = F.reshape(x, [-1, inH, 1, inW, 1, minorDim], inplace=False)
    x = F.pad(x, [0, 0, 0, 0, 0, upy - 1, 0, 0, 0, upx - 1, 0, 0])
    x = F.reshape(x, [-1, inH * upy, inW * upx, minorDim], inplace=False)

    x = F.pad(x, [0, 0, max(pady0, 0), max(pady1, 0),
                  max(padx0, 0), max(padx1, 0), 0, 0])
    x = x[:, max(-pady0, 0): x.shape[1] - max(-pady1, 0),
          max(-padx0, 0): x.shape[2] - max(-padx1, 0), :]

    # Convolve with filter.
    x = F.transpose(x, [0, 3, 1, 2])
    x = F.reshape(x, [-1, 1, inH * upy + pady0 + pady1,
                      inW * upx + padx0 + padx1], inplace=False)
    w = nn.Variable.from_numpy_array(k[np.newaxis, np.newaxis, ::-1, ::-1])

    x = F.convolution(x, w)

    x = F.reshape(x, [-1, minorDim, inH * upy + pady0 + pady1 - kernelH +
                      1, inW * upx + padx0 + padx1 - kernelW + 1], inplace=False)
    x = F.transpose(x, [0, 2, 3, 1])

    if downx == 1 and downy == 1:
        return x
    return x[:, ::downy, ::downx, :]


def downsample_2d(x, k=None, factor=2, gain=1, kernel_size=3):
    assert isinstance(factor, int) and factor >= 1
    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * (gain)
    p = (len(k) - factor) + (kernel_size - 1)
    return _simple_upfirdn_2d(x, k, pad0=(p + 1) // 2, pad1=p // 2)


def downsample_conv_2d(x, w, k=None, factor=2, gain=1):
    assert isinstance(factor, int) and factor >= 1

    # Check weight shape.
    assert w.ndim == 4
    convH = w.shape[2]
    convW = w.shape[3]
    assert convW == convH

    # Setup filter kernel.
    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * (gain * (factor ** 2))
    p = (k.shape[0] - factor) + (convW - 1)

    # Execute.
    w = w[:, :, ::-1, ::-1]
    x = F.convolution(x, w, stride=(factor, factor))

    return _simple_upfirdn_2d(x, k, pad0=(p + 1)//2, pad1=p//2)
