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
import nnabla.solvers as S
import numpy as np

from graph_unet import model, metrics
from utils import load_cora, get_mask, normalize_adj

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--seed', type=int, default=915)

args = parser.parse_args()

if __name__ == '__main__':
    try:
        from nnabla.ext_utils import get_extension_context
        ctx = get_extension_context('cudnn', device_id='0')
        nn.set_default_context(ctx)
    except:
        pass

    nn.seed(args.seed)

    print('Loading dataset...')
    G, feature_matrix, labels = load_cora(seed=args.seed)

    num_nodes = len(G.nodes)
    num_classes = max(labels) + 1

    train_mask, valid_mask, test_mask = get_mask(
        20, 500, 1000, num_nodes, num_classes, labels, seed=args.seed)

    print('Preprocessing data...')
    A_hat = normalize_adj(G)
    A_hat = nn.Variable.from_numpy_array(A_hat)

    print('Building model...')

    X = nn.Variable.from_numpy_array(feature_matrix)
    labels = nn.Variable.from_numpy_array(np.expand_dims(labels, axis=1))
    train_mask = nn.Variable.from_numpy_array(
        np.expand_dims(train_mask, axis=1))
    valid_mask = nn.Variable.from_numpy_array(
        np.expand_dims(valid_mask, axis=1))

    pool_rate = [0.25, 0.25, 0.25, 0.25]
    out = model(A_hat, X, pool_rate, num_classes, train=True)
    out_valid = model(A_hat, X, pool_rate, num_classes, train=False)

    loss, acc = metrics(out, labels, train_mask)
    _, acc_valid = metrics(out_valid, labels, valid_mask)

    # Solver / Optimizer
    solver = S.Adam(alpha=0.01)
    solver.set_parameters(nn.get_parameters())

    print('Begin training loop...')
    best_acc = 0.
    for i in range(args.epoch):

        solver.zero_grad()
        F.sink(loss, acc).forward()
        loss.backward()
        solver.update()

        acc_valid.forward()

        best_acc = max(acc_valid.d, best_acc)

        print("loss: {:.3f} acc: {:.3f} acc_valid: {:.3f}".format(
            loss.d, acc.d, acc_valid.d))

    _, acc_test = metrics(out_valid, labels, test_mask)
    acc_test.forward()

    print('Training finished.')
    print('Best validation accuracy: {:.3f}'.format(best_acc))
    print('Test accuracy: {:.3f}'.format(acc_test.d))
