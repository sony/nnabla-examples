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
import nnabla.solvers as S
import numpy as np

from gcn_model import gcn
from utils import load_cora, get_mask, normalize_adj, get_accuracy

if __name__ == '__main__':
    try:
        from nnabla.ext_utils import get_extension_context
        ctx = get_extension_context('cudnn', device_id='0')
        nn.set_default_context(ctx)
    except:
        pass

    print('Loading dataset...')
    G, feature_matrix, labels = load_cora()

    num_nodes = len(G.nodes)
    num_classes = max(labels) + 1

    train_mask, valid_mask, test_mask = get_mask(
        20, 500, 500, num_nodes, num_classes, labels)

    print('Preprocessing data...')
    A_hat = normalize_adj(G)

    print('Building model...')
    A_hat = nn.Variable.from_numpy_array(A_hat)
    X = nn.Variable.from_numpy_array(feature_matrix)
    labels = nn.Variable.from_numpy_array(np.expand_dims(labels, axis=1))
    train_mask = nn.Variable.from_numpy_array(
        np.expand_dims(train_mask, axis=1))
    valid_mask = nn.Variable.from_numpy_array(
        np.expand_dims(valid_mask, axis=1))
    test_mask = nn.Variable.from_numpy_array(np.expand_dims(test_mask, axis=1))

    H = gcn(A_hat, X, num_classes, 0.5)
    H_valid = gcn(A_hat, X, num_classes, 0)

    # Solver / Optimizer
    solver = S.Adam(alpha=0.01)
    solver.set_parameters(nn.get_parameters())

    # Weight decay
    loss_wd = 0.0
    for v in nn.get_parameters().values():
        loss_wd += 0.5*F.sum(F.pow_scalar(v, 2.0))

    loss = F.mean(F.categorical_cross_entropy(
        H[train_mask], labels[train_mask])) + 5e-4*loss_wd
    loss_valid = F.mean(F.categorical_cross_entropy(
        H_valid[valid_mask], labels[valid_mask]))

    print('Begin training loop...')
    best_score = 0.
    for i in range(200):

        solver.zero_grad()
        loss.forward()
        loss.backward()
        solver.update()

        loss_valid.forward()

        acc = get_accuracy(H.d, labels.d, train_mask.d)
        acc_val = get_accuracy(H_valid.d, labels.d, valid_mask.d)

        if acc_val > best_score:
            best_score = acc_val

        print("loss: {:.3f} acc: {:.3f} loss_val: {:.3f} acc_val: {:.3f}".format(
            loss.d, acc, loss_valid.d, acc_val))

    acc_test = get_accuracy(H_valid.d, labels.d, test_mask.d)
    print('Training finished.')
    print('Best validation acc: {:.3f}'.format(best_score))
    print('Test acc: {:.3f}'.format(acc_test))
