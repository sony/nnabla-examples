# Copyright 2020,2021 Sony Corporation.
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
from nnabla.parameter import get_parameter_or_create
import numpy as np


def invertible_conv(x, reverse, rng, scope):
    r"""Invertible 1x1 Convolution Layer.

    Args:
        x (nn.Variable): Input variable.
        reverse (bool): Whether it's a reverse direction.
        rng (numpy.random.RandomState): A random generator.
        scope (str): The scope.

    Returns:
        nn.Variable: The output variable.
    """
    batch_size, c, n_groups = x.shape
    with nn.parameter_scope(scope):
        # initialize w by an orthonormal matrix
        w_init = np.linalg.qr(rng.randn(c, c))[0][None, ...]
        W_var = get_parameter_or_create("W", (1, c, c), w_init, True, True)
        W = F.batch_inv(W_var) if reverse else W_var
        x = F.convolution(x, F.reshape(W, (c, c, 1)), None, stride=(1,))
    if reverse:
        return x
    log_det = batch_size*n_groups*F.log(F.abs(F.batch_det(W)))
    return x, log_det


def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    in_act = F.add2(input_a, input_b)
    t_act = F.tanh(in_act[:, :n_channels, :])
    s_act = F.sigmoid(in_act[:, n_channels:, :])
    return t_act * s_act
