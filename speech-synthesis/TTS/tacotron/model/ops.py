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

from functools import partial

import nnabla as nn
import nnabla.functions as F
from nnabla.initializer import ConstantInitializer
import nnabla.parametric_functions as PF


def conv1d(inputs, kernel_size, channels, activation, is_training, scope):
    r"""Create an 1D convolutional layer.

    Args:
        inputs (nn.Variable): The input sequence of shape B x C x T
        kernel_size (int): The kernel size.
        channels (list of int): A list of integers representing the
            channel sizes.
        activation (nn.function): Activation function, which will be applied.
        is_training (bool): If `is_training` is `True`, then batch_stat
            will be computed.
        scope (str): The parameter scope name.

    Returns:
        nn.Variable: Output variable.
    """
    if kernel_size % 2 == 0:
        inputs = F.pad(inputs, (0,)*5 + (1,),
                       mode='constant', constant_value=0)
    with nn.parameter_scope(scope):
        out = PF.convolution(inputs, channels, kernel=(kernel_size,),
                             pad=((kernel_size-1)//2,), with_bias=False)
        if activation is not None:
            out = activation(out)
        out = PF.batch_normalization(out, batch_stat=is_training)
    return out


def highwaynet(inputs, depth, scope):
    r"""Return the Highway network.

    Args:
        inputs (nn.Variable): An input variable.
        depth (int): The hidden size used in affine.
        scope (str): The parameter scope name.

    Returns:
        nn.Variable: Output variable.

    References:
        https://arxiv.org/abs/1505.00387
    """
    with nn.parameter_scope(scope):
        H = F.relu(PF.affine(inputs, depth, base_axis=2, name='H'))
        T = F.sigmoid(PF.affine(inputs, depth, name='T', base_axis=2,
                                b_init=ConstantInitializer(-1.0)))
    return H * T + inputs * (1.0 - T)


def cbhg(inputs, K, projections, depth, is_training, scope):
    r"""Returns the 1D Convolution Bank Highwaynet bindirectional
    GRU (CBHG) module.

    Args:
        inputs (nn.Variable): NNabla Variable of shape (B, C, T).
        K (int): Maximum kernel size.
        projections (list of int): A list of channels.
        depth (int): A depth. This should be an even number.
        is_training (bool): Whether training mode is activated.
        scope (str): The parameter scope name.

    Returns:
        nn.Variable: Output variable.
    """

    with nn.parameter_scope(scope):
        # Convolution bank: concatenate channels from all 1D convolutions
        with nn.parameter_scope('conv_bank'):
            conv = partial(conv1d, inputs, channels=128,
                           activation=F.relu, is_training=is_training)
            conv_outputs = [
                conv(kernel_size=k, scope=f'conv1d_{k}') for k in range(1, K+1)]
            conv_outputs = F.concatenate(*conv_outputs, axis=1)

        # make sure a valid input to max_pooling
        x = F.pad(conv_outputs, (0,)*5+(1,), mode='constant')

        # Maxpooling: reshape is needed because nnabla does support 1D pooling
        maxpool_output = F.max_pooling(
            x.reshape(x.shape + (1,)),
            kernel=(2, 1), stride=(1, 1)
        ).reshape(conv_outputs.shape)

        # Two projection layers:
        proj1_output = conv1d(
            maxpool_output,
            kernel_size=3,
            channels=projections[0],
            activation=F.relu,
            is_training=is_training,
            scope='proj_1'
        )
        proj2_output = conv1d(
            proj1_output,
            kernel_size=3,
            channels=projections[1],
            activation=None,
            is_training=is_training,
            scope='proj_2'
        )

        # Residual connection:
        highway_input = proj2_output + inputs

        assert depth % 2 == 0
        half_depth = depth // 2

        with nn.parameter_scope('highwaynet'):
            # transposing to shape (B, T, C)
            highway_input = F.transpose(highway_input, (0, 2, 1))

            # Handle dimensionality mismatch:
            if highway_input.shape[2] != half_depth:
                highway_input = PF.affine(
                    highway_input, half_depth, base_axis=2,
                    name='adjust_dim'
                )

            # 4-layer HighwayNet:
            for i in range(4):
                highway_input = highwaynet(
                    highway_input, half_depth,
                    scope=f'highway_{i+1}'
                )

        with nn.parameter_scope('rnn_net'):
            # transpose to shape (T, B, C)
            rnn_input = F.transpose(highway_input, (1, 0, 2))
            outputs, _ = PF.gru(
                rnn_input,
                F.constant(shape=(2, 2, rnn_input.shape[1], half_depth)),
                training=is_training,
                bidirectional=True
            )  # (T, B, C)

    return outputs


def encoder_cbhg(inputs, depth, is_training):
    # inputs of shape (B, C, T)
    input_channels = inputs.shape[1]
    return cbhg(inputs, K=16,
                projections=[128, input_channels],
                depth=depth,
                is_training=is_training,
                scope='encoder_cbhg')


def post_cbhg(inputs, input_dim, is_training, depth):
    return cbhg(inputs, K=8,
                projections=[256, input_dim],
                depth=depth,
                is_training=is_training,
                scope='post_cbhg')


def prenet(inputs, layer_sizes, is_training, scope):
    r"""Return Prenet.

    Args:
        inputs (nn.Variable): A NNabla variable of shape (B, T, C).
        layer_sizes (list of int): A list of layer sizes.
        is_training (bool): Whether training module is activated.
        scope (str): The parameter scope name.

    Returns:
        nn.Variable: Output variable.
    """
    out = inputs
    with nn.parameter_scope(scope):
        for i, size in enumerate(layer_sizes):
            out = PF.affine(out, size, base_axis=2, name=f'affine_{i}')
            out = F.relu(out)
            out = F.dropout(out, p=0.5)  # always chooses dropout
    return out


def Bahdanau_attention(query, values, out_features, scope):
    r"""Return the Bahdanau attention mechanism.

    Args:
        query (nn.Variable): A query of size (B, 1, C).
        values (nn.Variable): Values of size (B, T, C).
        out_features (int): The projected dimensionality.
        scope (str): Parameter scope.

    Returns:
        nn.Variable: The context vector.
        nn.Variable: The attention weight vector.
    """
    with nn.parameter_scope(scope):
        x = PF.affine(query, out_features, base_axis=2,
                      with_bias=False, name='query')
        y = PF.affine(values, out_features, base_axis=2,
                      with_bias=False, name='values')
        # scores of shape (B, T, 1)
        scores = PF.affine(F.tanh(x + y), 1, base_axis=2,
                           with_bias=False, name='scores')
        # attention_weights of shape (B, 1, T)
        attention_weights = F.softmax(
            scores, axis=1).reshape((query.shape[0], 1, -1))
        # context_vector shape after sum == (B, 1, C)
        context_vector = F.batch_matmul(attention_weights, values)

    return context_vector, attention_weights
