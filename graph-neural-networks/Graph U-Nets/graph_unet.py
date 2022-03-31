# Copyright 2022 Sony Group Corporation.
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
import numpy as np


def model(A, H, k_list, num_classes, train=False):
    '''
    Graph U-Net for node classification
    '''
    if train:
        dropout = 0.5
    else:
        dropout = 0.

    out = graph_unet(A, H, k_list, dropout)
    out = gcn_layer(A, out, 128, 'classifier_0', dropout, activation=F.relu)
    out = gcn_layer(A, out, num_classes, 'classifier_1',
                    dropout, activation=F.softmax)

    return out


def graph_unet(A, H, k_list, dropout):

    depth = len(k_list)
    hidden_dim = H.shape[1]

    pool_adj_list = []
    pool_hidden_list = []
    indices_list = []
    H_origin = H

    # pooling
    for i in range(depth):
        H = gcn_layer(A, H, hidden_dim, 'pool_gcn_' + str(i), dropout, F.relu)
        pool_adj_list.append(A)
        pool_hidden_list.append(H)
        A, H, indices = g_pool(A, H, k_list[i], 'pool_' + str(i), dropout)
        indices_list.append(indices)

    # bottom
    H = gcn_layer(A, H, hidden_dim, 'bottom_gcn', dropout, F.relu)

    # unpooling
    for i in range(depth):
        reverse_idx = depth - i - 1
        A, indices = pool_adj_list[reverse_idx], indices_list[reverse_idx]
        H = g_unpool(A, H, indices)
        H = gcn_layer(A, H, hidden_dim, 'unpool_gcn_' +
                      str(i), dropout, F.relu)
        H = H + pool_hidden_list[reverse_idx]

    H = H + H_origin

    return H


def g_pool(A, H, k, name, dropout):
    '''
    graph pooling

    '''
    with nn.parameter_scope(name):
        Z = F.dropout(H, dropout)

        # projection feature with trainable vector 'p' in paper
        weights = PF.affine(Z, 1)
        scores = F.sigmoid(weights)

        A, H_pooled, indices = top_k_node(scores, A, H, k)

    return A, H_pooled, indices


def g_unpool(A, H, indices):
    '''
    graph unpooling
    '''
    indices = F.reshape(indices, (1, indices.shape[0]))
    H_unpooled = F.scatter_nd(H, indices, shape=(A.shape[0], H.shape[1]))

    return H_unpooled


def top_k_node(scores, A, H, k):
    '''
    get top-k nodes and pooling
    '''

    num_nodes = A.shape[0]
    k = max(2, int(num_nodes * k))  # minimal number of nodes is 2

    _, indices = F.sort(scores, axis=0, with_index=True)

    indices = F.slice(indices, start=(0, 0), stop=(k, 1))  # top-k nodes
    indices = F.reshape(indices, (indices.shape[0], ))

    H_pooled = F.gather(H, indices)
    A = F.greater_scalar(A, 0).apply(need_grad=False)
    A = F.dot(A, A)  # graph connectivity augumentation
    A_pooled = F.transpose(F.gather(A, indices), (1, 0))
    A_pooled = F.transpose(F.gather(A_pooled, indices), (1, 0))

    A_pooled = normalize(A_pooled)

    return A_pooled, H_pooled, indices


def normalize(A):
    degrees = F.sum(A, axis=1)
    degrees = F.broadcast(degrees, A.shape)
    A = F.div2(A, degrees)
    return A


def gcn_layer(A, H, out_dim, name, dropout, activation):
    with nn.parameter_scope(name):

        if dropout > 0.:
            H = F.dropout(H, dropout)

        H = PF.affine(H, (out_dim, ), with_bias=False)
        H = F.dot(A, H)

        if activation is not None:
            H = activation(H)

    return H


def metrics(out, labels, mask):
    '''
    Return loss and accuracy
    '''

    # weight decay
    loss_wd = 0.0
    for v in nn.get_parameters().values():
        loss_wd += 0.5*F.sum(F.pow_scalar(v, 2.0))

    loss = F.mean(F.categorical_cross_entropy(
        out[mask], labels[mask])) + 5e-4*loss_wd
    pred = F.max(out, axis=1, only_index=True)
    pred = F.reshape(pred, (pred.shape[0], 1))
    acc = F.mean(F.equal(pred[mask], labels[mask]))

    return loss, acc
