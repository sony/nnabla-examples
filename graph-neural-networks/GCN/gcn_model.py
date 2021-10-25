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
import nnabla.parametric_functions as PF


def gcn(A_hat, X, num_classes=7, dropout=0.5):
    """
    Two layer GCN model.
    """

    H = gcn_layer(A_hat, X, out_features=16,
                  name='gcn_layer_0', dropout=dropout)
    H = gcn_layer(A_hat, H, out_features=num_classes,
                  name='gcn_layer_1', dropout=dropout, activation=F.softmax)

    return H


def gcn_layer(A_hat, X, out_features, name, dropout=0.5, activation=F.relu):
    '''
    GCN layer

    Parameters
    ----------
    A_hat: nnabla.Variable
      Normalized graph Laplacian
    X: nnabla.Variable
      Feature matrix
    out_features: int
      Number of dimensions of output
    name: str
      Name of parameter scope
    dropout: float
      Parameter of dropout. If 0, not to use dropout
    activaton: nnabla.functons
      Activation function

    Returns
    -------
    H: nnabla.Variable
      Output of GCN layer
    '''

    with nn.parameter_scope(name):
        if dropout > 0:
            X = F.dropout(X, dropout)

        H = PF.affine(X, (out_features, ), with_bias=False)
        H = F.dot(A_hat, H)

        if activation is not None:
            H = activation(H)

    return H
