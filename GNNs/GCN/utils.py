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
import networkx as nx
import pandas as pd
import os
import tarfile
from urllib.request import urlopen
from sklearn.preprocessing import LabelEncoder

CORA_URL = "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"


def download_cora(url=CORA_URL):
    '''
    Download cora dataset.
    '''
    if not os.path.exists('./cora.tgz'):
        print('Downloading cora dataset...')
        data = urlopen(url).read()
        with open('./cora.tgz', 'wb') as f:
            f.write(data)
    else:
        print('Cora dataset already exists.')

    if not os.path.exists('./cora'):
        print('Extracting cora dataset...')
        with tarfile.open('./cora.tgz', 'r') as f:
            f.extractall('./')
    else:
        print('Cora dataset is already extracted.')


def load_cora(shuffle=True):
    """
    Download and load cora dataset.
    Return NetworkX graph and feature matrix and numerical labels.
    """

    download_cora()

    edge_df = pd.read_csv(os.path.join('./cora', 'cora.cites'),
                          sep='\t', header=None, names=["target", "source"])
    if shuffle:
        edge_df = edge_df.sample(frac=1)
    G = nx.from_pandas_edgelist(edge_df)

    feature_names = ['word_{}'.format(i) for i in range(1433)]
    column_names = feature_names + ['subject']
    node_df = pd.read_csv(os.path.join('./cora', 'cora.content'),
                          sep='\t', header=None, names=column_names)

    node_index = [i for i in G.nodes]
    node_df = node_df.reindex(index=node_index)
    feature_matrix = np.array(node_df.iloc[:, 0:-1])
    feature_matrix = row_normalization(feature_matrix)

    labels = node_df['subject']
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    return G, feature_matrix, labels


def row_normalization(matrix):
    '''
    Normalize feature matrix.
    '''
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / norm

    return matrix


def get_mask(labels_per_class, num_valid, num_test, num_nodes, num_classes, labels):
    '''
    Return mask index for semi-supervised training.
    '''
    all_index = np.arange(num_nodes)

    train_index = []
    cnt = [0 for _ in range(num_classes)]
    for i, label in enumerate(labels):
        if cnt[label] < labels_per_class:
            train_index.append(i)
            cnt[label] += 1
        elif len(train_index) == num_classes * labels_per_class:
            break

    all_index = np.delete(all_index, train_index)

    valid_test_index = np.random.choice(
        all_index, num_valid + num_test, replace=False)

    valid_index = valid_test_index[:num_valid]
    test_index = valid_test_index[num_valid: num_valid + num_test]

    return np.array(train_index), valid_index, test_index


def normalize_adj(G):
    """
    Normalize adjacency matrix.
    """
    A = nx.adjacency_matrix(G).todense().astype(np.float)
    A_tilda = A + np.eye(A.shape[0]).astype(np.float)  # A + I
    degree = np.array(np.sum(A_tilda, axis=0))[0]
    D_hat_inv_sqrt = np.diag(np.power(degree, -0.5))
    A_hat = np.matmul(D_hat_inv_sqrt, np.matmul(
        A_tilda, D_hat_inv_sqrt))  # D^(-0.5) * A * D^(-0.5)

    return A_hat


def get_accuracy(predict, label, mask):
    '''
    Calculate accuray.
    '''
    mask = np.squeeze(mask)
    predict = np.argmax(predict[mask], axis=1)
    label = np.squeeze(label[mask])
    correct = np.count_nonzero(predict == label)
    accuracy = (correct / len(mask))

    return accuracy
