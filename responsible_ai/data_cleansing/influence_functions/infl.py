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
import nnabla as nn
import nnabla.solvers as S
import nnabla.functions as F
import os
import functools
from tqdm import tqdm
from sgd_influence.model import setup_model
from sgd_influence.dataset import get_batch_data, init_dataset, get_data, get_image_size, get_batch_indices
from sgd_influence.utils import ensure_dir, get_indices, save_to_csv


def adjust_batch_size(model, batch_size, loss_fn=None):
    has_loss = loss_fn is not None
    if has_loss:
        loss_d, loss_g = loss_fn.d, loss_fn.g
    pred, loss_fn, input_image = model(batch_size=batch_size)
    if has_loss:
        loss_fn.d = loss_d
        loss_fn.g = loss_g
    return pred, loss_fn, input_image


def save_infl_for_analysis(infl_list, use_all_params, save_dir, infl_filename, epoch, header, data_type):
    dn = os.path.join(save_dir, 'epoch%02d' % (epoch))
    if use_all_params:
        dn = os.path.join(dn, 'infl_original')
    else:
        dn = os.path.join(dn, 'infl_arranged')
    ensure_dir(dn)
    save_to_csv(filename=os.path.join(dn, os.path.basename(infl_filename)),
                header=header, list_to_save=infl_list, data_type=data_type)


def compute_gradient(grad_model, solver, dataset, batch_size, idx_list_to_data, resize_size):
    n = len(idx_list_to_data)
    grad_idx = get_batch_indices(n, batch_size, seed=None)
    u = {}
    loss_fn = None
    for i in tqdm(grad_idx, desc='calc gradient (2/3 steps)'):
        X, y = get_batch_data(dataset, idx_list_to_data,
                              i, resize_size, test=True)
        _, loss_fn, input_image = adjust_batch_size(
            grad_model, len(X), loss_fn)
        input_image["image"].d = X
        input_image["label"].d = y
        loss_fn.forward()
        solver.zero_grad()
        loss_fn.backward(clear_buffer=True)

        for key, param in nn.get_parameters(grad_only=False).items():
            uu = u.get(key, None)
            if uu is None:
                u[key] = nn.Variable(param.shape)
                u[key].data.zero()
            u[key].d += param.g / n
    return u


def infl_icml(model_info_dict, file_dir_dict, use_all_params, need_evaluate, alpha):
    num_epochs = 2
    # params
    lr = 0.005
    seed = model_info_dict['seed']
    net_func = model_info_dict['net_func']
    batch_size = model_info_dict['batch_size']
    test_batch_size = 1000
    target_epoch = model_info_dict['num_epochs']
    # files and dirs
    save_dir = file_dir_dict['save_dir']
    infl_filename = file_dir_dict['infl_filename']
    final_model_name = file_dir_dict['model_filename']
    final_model_path = os.path.join(save_dir, 'epoch%02d' % (
        target_epoch - 1), 'weights', final_model_name)
    input_dir_name = os.path.dirname(file_dir_dict['train_csv'])

    # setup
    trainset, valset, image_shape, n_classes, ntr, nval = init_dataset(
        file_dir_dict['train_csv'], file_dir_dict['val_csv'], seed)
    n_channels, _h, _w = image_shape
    resize_size = get_image_size((_h, _w))
    idx_train = get_indices(ntr, seed)
    idx_val = get_indices(nval, seed)

    nn.load_parameters(final_model_path)
    trained_params = nn.get_parameters(grad_only=False)

    test = True

    grad_model = functools.partial(
        setup_model, net_func=net_func, n_classes=n_classes, n_channels=n_channels, resize_size=resize_size, test=test, reduction='mean')
    solver = S.Momentum(lr=lr, momentum=0.9)
    solver.set_parameters(trained_params)
    # gradient
    u = compute_gradient(grad_model, solver, valset,
                         test_batch_size, idx_val, resize_size)

    # Hinv * u with SGD
    seed_train = 0
    v = dict()
    for key, param in nn.get_parameters(grad_only=False).items():
        v[key] = nn.Variable(param.d.shape, need_grad=True)
        v[key].d = 0
        v[key].g = 0

    solver.set_parameters(v)

    loss_train = []
    loss_fn = None
    for epoch in range(num_epochs):
        # training
        seed_train = 0
        np.random.seed(epoch)
        idx = get_batch_indices(ntr, batch_size, seed=epoch)
        for j, i in enumerate(idx):
            seeds = list(range(seed_train, seed_train + i.size))
            seed_train += i.size
            X, y = get_batch_data(trainset, idx_train, i,
                                  resize_size, test=False, seeds=seeds)
            _, loss_fn, input_image = adjust_batch_size(
                grad_model, len(X), loss_fn)
            input_image["image"].d = X
            input_image["label"].d = y
            loss_fn.forward()

            grad_params = nn.grad(
                loss_fn, [param for param in nn.get_parameters(grad_only=False).values()])
            vg = 0
            for vv, g in zip(v.values(), grad_params):
                vg += F.sum(vv*g)

            for parameters in trained_params.values():
                parameters.grad.zero()

            vgrad_params = nn.grad(
                vg, [param for param in nn.get_parameters(grad_only=False).values()])
            loss_i = 0
            for vgp, vv, uu in zip(vgrad_params, v.values(), u.values()):
                loss_i += 0.5 * F.sum(vgp * vv + alpha *
                                      vv * vv) - F.sum(uu * vv)
            loss_i.forward()

            solver.zero_grad()
            loss_i.backward(clear_buffer=True)
            solver.update()
            loss_train.append(loss_i.d.copy())

    # influence
    infl_dict = dict()
    infl = np.zeros(ntr)
    for i in tqdm(range(ntr), desc='calc influence (3/3 steps)'):
        csv_idx = idx_train[i]
        file_name = trainset.get_filepath_to_data(csv_idx)
        file_name = os.path.join(input_dir_name, file_name)
        file_name = os.path.normpath(file_name)
        X, y = get_data(trainset, idx_train[i], resize_size, True, seed=i)
        _, loss_fn, input_image = adjust_batch_size(
            grad_model, len(X), loss_fn)
        input_image["image"].d = X
        input_image["label"].d = y
        loss_fn.forward()
        for parameters in trained_params.values():
            parameters.grad.zero()
        loss_fn.backward(clear_buffer=True)
        infl_i = 0
        for j, param in enumerate(nn.get_parameters(grad_only=False).values()):
            infl_i += (param.g.copy() * list(v.values())[j].d.copy()).sum()
        infl[i] = -infl_i / ntr
        infl_dict[csv_idx] = [file_name, y, infl[i]]
    infl_list = [val + [key] for key, val in infl_dict.items()]
    infl_list = sorted(infl_list, key=lambda x: (x[-2]))

    # save
    header = ['x:image', 'y:label', 'influence', 'datasource_index']
    data_type = 'object,int,float,int'
    if need_evaluate:
        save_infl_for_analysis(infl_list, use_all_params,
                               save_dir, infl_filename, epoch, header, data_type)
    save_to_csv(filename=infl_filename, header=header,
                list_to_save=infl_list, data_type=data_type)
