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

import functools
import numpy as np
import nnabla as nn
import nnabla.parametric_functions as PF
import nnabla.functions as F


def block(x, scope_name, n_channels, kernel, pad, test):
    with nn.parameter_scope(scope_name):
        with nn.parameter_scope('conv1'):
            h = PF.convolution(x, n_channels, kernel=kernel,
                               pad=pad, with_bias=True)
            h = PF.batch_normalization(h, batch_stat=not test)
            h = F.relu(h)

        with nn.parameter_scope('conv2'):
            h = PF.convolution(h, n_channels, kernel=kernel,
                               pad=pad, with_bias=True)
            h = F.relu(h)
            h = F.max_pooling(h, kernel=(2, 2), stride=(2, 2))
    return h


def cifarnet(x, test=False, n_classes=10):
    maps = [32, 64, 128]
    kernel = (3, 3)
    pad = (1, 1)

    h = block(x, 'block1', maps[0], kernel, pad, test)
    h = block(h, 'block2', maps[1], kernel, pad, test)
    h = block(h, 'block3', maps[2], kernel, pad, test)
    h = PF.affine(h, n_classes)
    return h


def loss_function(pred, label, reduction='mean'):
    loss_dict = {
        'mean': F.mean(F.softmax_cross_entropy(pred, label)),
        'sum': F.sum(F.softmax_cross_entropy(pred, label))
    }
    return loss_dict[reduction]


def calc_acc(pred, label, method='mean'):
    acc_sum = (np.argmax(pred, axis=1).reshape(-1, 1)
               == np.array(label).reshape(-1, 1)).sum()
    acc_dict = {
        'mean': acc_sum / len(label),
        'sum': acc_sum
    }
    return acc_dict[method]


def setup_model(net_func, n_classes, n_channels, resize_size, batch_size, test, reduction='mean'):
    prediction = functools.partial(net_func, n_classes=n_classes)
    image = nn.Variable(
        (batch_size, n_channels, resize_size[0], resize_size[1]))
    label = nn.Variable((batch_size, 1))
    pred = prediction(image, test)
    loss_fn = loss_function(pred, label, reduction)
    input_image = {"image": image, "label": label}
    return pred, loss_fn, input_image


def adjust_batch_size(model, solver, batch_size, loss_fn=None):
    params = nn.get_parameters(grad_only=False)
    has_loss = loss_fn is not None
    if has_loss:
        loss_d, loss_g = loss_fn.d, loss_fn.g
    pred, loss_fn, input_image = model(batch_size=batch_size)
    solver.set_parameters(params)
    if has_loss:
        loss_fn.d = loss_d
        loss_fn.g = loss_g
    return pred, loss_fn, input_image


def get_config(args, is_eval=False):
    class Congig(object):
        def __init__(self, cfg, is_eval):
            lr = 0.05
            batch_size = 64

            self.model_info_dict = {
                'lr': lr,
                'batch_size': batch_size,
                'device_id': cfg.device_id,
                'seed': cfg.seed,
                'net_func': cifarnet,
                'num_epochs': cfg.n_epochs,
            }

            self.file_dir_dict = {
                'train_csv': cfg.input_train,
                'val_csv': cfg.input_val,
                'score_filename': cfg.score_output,
                'save_dir': cfg.weight_output,
                'infl_filename': cfg.output,
                'model_filename': 'final_model.h5',
                'info_filename': 'info.npy'
            }
            if is_eval:
                self._init_for_eval(cfg)
            else:
                self._init_for_train_infl(cfg)

        def _init_for_eval(self, cfg):
            self.file_dir_dict['test_csv'] = cfg.input_test
            start_epoch = 0 if cfg.retrain_all else cfg.n_epochs - 1
            self.model_info_dict['start_epoch'] = start_epoch

        def _init_for_train_infl(self, cfg):
            self.calc_infl_with_all_params = cfg.calc_infl_with_all_params
            self.need_evaluate = False if cfg.score_output is None else True
            end_epoch_dict = {'last': cfg.n_epochs - 1, 'all': 0}
            self.model_info_dict['end_epoch'] = end_epoch_dict[cfg.calc_infl_method]

    return Congig(args, is_eval)
