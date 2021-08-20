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
from nnabla.initializer import UniformInitializer
import nnabla.parametric_functions as PF
import numpy as np


def calculate_gain(nonlinearity):
    if nonlinearity == 'tanh':
        return 5.0 / 3.0
    elif nonlinearity == 'relu':
        return np.sqrt(2.0)
    return 1.0


def xavier_uniform_bound(inp_shape, outmaps, kernel=(1, 1), base_axis=1,
                         nonlinearity=None, is_affine=False):
    inmaps = np.prod(inp_shape[base_axis:]
                     ) if is_affine else inp_shape[base_axis]
    gain = calculate_gain(nonlinearity)
    d = gain * np.sqrt(6. / (np.prod(kernel) * (inmaps + outmaps)))
    return -d, d


def affine_norm(inputs, out_channels, base_axis, with_bias, w_init_gain, scope, **kargs):
    r"""Affine Layer.

    Args:
        inputs (nn.Variable): An input variable of shape (B,...)
        out_channels (int): The number of output channels.
        base_axis (int): The base axis.
        with_bias (bool): Whether to use bias.
        w_init_gain (str): The non-linear function.
        scope (str): The parameter scope name.

    Returns:
        nn.Variable: An output variable.
    """
    with nn.parameter_scope(scope):
        lim = xavier_uniform_bound(inputs.shape, out_channels, kernel=(1, 1), base_axis=base_axis,
                                   nonlinearity=w_init_gain, is_affine=True)
        w_init = UniformInitializer(lim)
        return PF.affine(inputs, out_channels, base_axis=base_axis,
                         w_init=w_init, with_bias=with_bias, **kargs)


def conv_norm(inputs, out_channels, kernel_size, stride,
              padding, dilation, bias, w_init_gain, scope, **kargs):
    r"""1D convolutional layer.

    Args:
        inputs (nn.Variable): An input variable of shape (B, C, T).
        out_channels (int): The number of ouput channels.
        kernel_size (int): The kernel size.
        stride (int): The stride.
        padding (int): The number of paddings.
        dilation (int): The dilation.
        bias (bool): Whether bias is used.
        w_init_gain (str): The non-linear function.
        scope (str): The parameter scope name.

    Returns:
        nn.Variable: An output variable.
    """
    with nn.parameter_scope(scope):
        base_axis = len(inputs.shape) - \
                        1 if kargs.get('channel_last', False) else 1
        lim = xavier_uniform_bound(inputs.shape, out_channels, (kernel_size,), base_axis,
                                   nonlinearity=w_init_gain, is_affine=False)
        w_init = UniformInitializer(lim)
        out = PF.convolution(inputs, out_channels, kernel=(kernel_size,),
                             stride=(stride,), pad=(padding,), w_init=w_init,
                             dilation=(dilation,), with_bias=bias, **kargs)
    return out


def prenet(inputs, layer_sizes, is_training, scope):
    r"""Return Prenet.

    Args:
        inputs (nn.Variable): A NNabla variable of shape (B, T, C).
        layer_sizes (list of int): A list of layer sizes.
        is_training (bool): Whether training module is activated.
        scope (str): The parameter scope name.

    Returns:
        nn.Variable: An output variable.
    """
    out = inputs
    with nn.parameter_scope(scope):
        for i, size in enumerate(layer_sizes):
            out = affine_norm(out, size, base_axis=2, with_bias=False,
                              w_init_gain='affine', scope=f'affine_{i}')
            out = F.dropout(F.relu(out), p=0.5)  # always chooses dropout
    return out


def location_sensitive_attention(query, values, attention_weights_cat,
                                 attention_location_kernel_size,
                                 attention_n_filters,
                                 attention_dim,
                                 is_training,
                                 scope):
    r"""Returns the location-sensitive attention mechanism.

    Args:
        query (nn.Variable): A query of size (B, 1, C1).
        values (nn.Variable): Values of size (B, T, C2).
        attention_weights_cat (nn.Variable): A variable of shape (B, 2, T).
        attention_dim (int): The projected dimensionality.
        scope (str): Parameter scope.

    Returns:
        nn.Variable: The context vector.
        nn.Variable: The attention weight vector.

    References:
        J. K. Chorowski, et al., "Attention-based models for speech recognition"
        in Advances in Neural Information Processing Systems, 2015, pp. 577-585.
    """

    with nn.parameter_scope(scope):
        x = affine_norm(query, attention_dim, base_axis=2,
                        with_bias=False, w_init_gain='tanh', scope='query')
        y = affine_norm(values, attention_dim, base_axis=2,
                        with_bias=False, w_init_gain='tanh', scope='memory')

        # apply a 1D-convolutional filter
        z = conv_norm(attention_weights_cat, attention_n_filters,
                      kernel_size=attention_location_kernel_size, stride=1,
                      padding=(attention_location_kernel_size-1) // 2,
                      dilation=1, bias=False, w_init_gain='affine', scope='conv_norm_lsa')
        z = F.transpose(z, (0, 2, 1))

        # location of shape (B, T, attention_dim)
        location = affine_norm(z, attention_dim, base_axis=2,
                               with_bias=False, w_init_gain='tanh', scope='location')

        # scores of shape (B, T, 1)
        scores = affine_norm(F.tanh(x + y + location), 1, base_axis=2,
                             with_bias=False, w_init_gain='affine', scope='scores')

        # attention_weights of shape (B, 1, T)
        attention_weights = F.softmax(
            scores, axis=1).reshape((query.shape[0], 1, -1))

        # context_vector shape after sum == (B, 1, C)
        context_vector = F.batch_matmul(attention_weights, values)

    return context_vector, attention_weights
