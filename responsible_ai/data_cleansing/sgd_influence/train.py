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

import os
import numpy as np
from tqdm import tqdm
import nnabla as nn
import functools
import nnabla.solvers as S
from .model import setup_model, calc_acc, adjust_batch_size
from .dataset import get_batch_indices, get_batch_data, init_dataset, get_image_size
from .utils import ensure_dir, get_indices, save_to_csv


def save_all_params(params_dict, c, k, j, bundle_size, step_size, save_dir, epoch):
    params_dict[c] = nn.get_parameters(grad_only=False).copy()
    c += 1
    if c == bundle_size or j == step_size - 1:
        dn = os.path.join(save_dir, 'epoch%02d' % (epoch), 'weights')
        ensure_dir(dn)
        for cc, params in params_dict.items():
            fn = '%s/model_step%04d.h5' % (dn, k + cc)
            nn.save_parameters(fn, params=params, extension=".h5")
        k += c
        c = 0
        params_dict = {}
    return params_dict, c, k


def eval_model(val_model, solver, dataset, idx_list_to_data, batch_size, resize_size):
    loss = 0
    acc = 0
    n = len(idx_list_to_data)
    idx = np.array_split(np.arange(n), batch_size)
    loss_fn = None
    for _, i in enumerate(idx):
        X, y = get_batch_data(dataset, idx_list_to_data,
                              i, resize_size, test=True)
        pred, loss_fn, input_image = adjust_batch_size(
            val_model, solver, len(X), loss_fn)
        input_image["image"].d = X
        input_image["label"].d = y
        loss_fn.forward()
        loss += loss_fn.d * len(X)
        acc += calc_acc(pred.d, y, method='sum')
    loss /= n
    acc /= n
    return loss, acc


def train(model_info_dict, file_dir_dict, use_all_params, need_evaluate, bundle_size=200):
    # params
    lr = model_info_dict['lr']
    seed = model_info_dict['seed']
    net_func = model_info_dict['net_func']
    batch_size = model_info_dict['batch_size']
    num_epochs = model_info_dict['num_epochs']
    infl_end_epoch = model_info_dict['end_epoch']
    # files and dirs
    save_dir = file_dir_dict['save_dir']
    info_filename = file_dir_dict['info_filename']
    model_filename = file_dir_dict['model_filename']
    score_filename = file_dir_dict['score_filename']
    # setup
    trainset, valset, image_shape, n_classes, ntr, nval = init_dataset(
        file_dir_dict['train_csv'], file_dir_dict['val_csv'], seed)
    n_channels, _h, _w = image_shape
    resize_size = get_image_size((_h, _w))
    # Create training graphs
    test = False
    train_model = functools.partial(
        setup_model, net_func=net_func, n_classes=n_classes, n_channels=n_channels, resize_size=resize_size, test=test)
    # Create validation graphs
    test = True
    val_model = functools.partial(
        setup_model, net_func=net_func, n_classes=n_classes, n_channels=n_channels, resize_size=resize_size, test=test)
    # setup optimizer (SGD)
    solver = S.Sgd(lr=lr)
    solver.set_parameters(nn.get_parameters(grad_only=False))

    # get shuffled index using designated seed
    idx_train = get_indices(ntr, seed)
    idx_val = get_indices(nval, seed)

    # training
    seed_train = 0
    info = []
    score = []
    loss_train = None
    for epoch in tqdm(range(num_epochs), desc='training (1/3 steps)'):
        idx = get_batch_indices(ntr, batch_size, seed=epoch)
        epoch_info = []
        c = 0
        k = 0
        params_dict = {}
        for j, i in enumerate(idx):
            seeds = list(range(seed_train, seed_train + i.size))
            seed_train += i.size
            epoch_info.append({'epoch': epoch, 'step': j,
                               'idx': i, 'lr': lr, 'seeds': seeds})
            if (use_all_params) & (epoch >= infl_end_epoch):
                params_dict, c, k = save_all_params(
                    params_dict, c, k, j, bundle_size, len(idx), save_dir, epoch)
            X, y = get_batch_data(trainset, idx_train, i,
                                  resize_size, test=False, seeds=seeds)
            _, loss_train, input_image_train = adjust_batch_size(
                train_model, solver, len(X), loss_train)
            input_image_train["image"].d = X
            input_image_train["label"].d = y

            loss_train.forward()
            solver.zero_grad()
            loss_train.backward(clear_buffer=True)
            solver.update()
        info.append(epoch_info)
        # save if params are necessary for calculating influence
        if epoch >= infl_end_epoch-1:
            dn = os.path.join(save_dir, 'epoch%02d' % (epoch), 'weights')
            ensure_dir(dn)
            nn.save_parameters(os.path.join(dn, model_filename), params=nn.get_parameters(
                grad_only=False), extension=".h5")
        # evaluation
        if need_evaluate:
            loss_tr, acc_tr = eval_model(
                val_model, solver, trainset, idx_train, batch_size, resize_size)
            loss_val, acc_val = eval_model(
                val_model, solver, valset, idx_val, batch_size, resize_size)
            score.append((loss_tr, loss_val, acc_tr, acc_val))
    # save epoch and step info
    np.save(os.path.join(save_dir, info_filename), arr=info)
    # save score
    if need_evaluate:
        save_to_csv(filename=score_filename, header=[
                    'train_loss', 'val_loss', 'train_accuracy', 'val_accuracy'], list_to_save=score, data_type='float,float,float,float')
