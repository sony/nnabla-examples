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
import functools
import numpy as np
from tqdm import tqdm
import nnabla as nn
import nnabla.solvers as S
import nnabla.functions as F
from .model import setup_model, adjust_batch_size
from .dataset import get_batch_data, init_dataset, get_data, get_image_size, get_batch_indices
from .utils import ensure_dir, get_indices, save_to_csv


def select_modelfile_for_infl(use_all_params, final_model_path, save_dir, epoch, step):
    if use_all_params:
        weights_dir = 'weights'
        fn = '%s/epoch%02d/%s/model_step%04d.h5' % (
            save_dir, epoch, weights_dir, step)
    else:
        fn = final_model_path
    return fn


def save_infl_for_analysis(infl_list, use_all_params, save_dir, infl_filename, epoch, header, data_type):
    dn = os.path.join(save_dir, 'epoch%02d' % (epoch))
    if use_all_params:
        dn = os.path.join(dn, 'infl_original')
    else:
        dn = os.path.join(dn, 'infl_arranged')
    ensure_dir(dn)
    save_to_csv(filename=os.path.join(dn, os.path.basename(infl_filename)),
                header=header, list_to_save=infl_list, data_type=data_type)


def infl_sgd(model_info_dict, file_dir_dict, use_all_params, need_evaluate):
    # params
    lr = model_info_dict['lr']
    seed = model_info_dict['seed']
    net_func = model_info_dict['net_func']
    batch_size = model_info_dict['batch_size']
    end_epoch = model_info_dict['end_epoch']
    target_epoch = model_info_dict['num_epochs']
    # files and dirs
    save_dir = file_dir_dict['save_dir']
    info_filename = file_dir_dict['info_filename']
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
        setup_model, net_func=net_func, n_classes=n_classes, n_channels=n_channels, resize_size=resize_size, test=test, reduction='sum')

    solver = S.Sgd(lr=lr)
    solver.set_parameters(trained_params)
    # gradient
    u = compute_gradient(grad_model, solver, valset,
                         batch_size, idx_val, target_epoch, resize_size)

    test = False
    infl_model = functools.partial(
        setup_model, net_func=net_func, n_classes=n_classes, n_channels=n_channels, resize_size=resize_size, test=test)
    # influence
    infl_dict = {}
    info = np.load(os.path.join(save_dir, info_filename), allow_pickle=True)
    loss_fn = None
    for epoch in tqdm(range(target_epoch - 1, end_epoch - 1, -1), desc='calc influence (3/3 steps)'):
        for step_info in info[epoch][::-1]:
            idx, seeds, lr, step = step_info['idx'], step_info['seeds'], step_info['lr'], step_info['step']
            fn = select_modelfile_for_infl(
                use_all_params, final_model_path, save_dir, epoch, step)
            _, loss_fn, input_image = adjust_batch_size(
                infl_model, solver, 1, loss_fn)
            nn.load_parameters(fn)
            params = nn.get_parameters(grad_only=False)
            solver = S.Sgd(lr=lr)
            solver.set_parameters(params)
            X = []
            y = []
            for i, seed in zip(idx, seeds):
                i = int(i)
                image, label = get_data(
                    trainset, idx_train[i], resize_size, test, seed=seed)
                X.append(image)
                y.append(label)
                input_image["image"].d = image
                input_image["label"].d = label
                loss_fn.forward()
                solver.zero_grad()
                loss_fn.backward(clear_buffer=True)

                csv_idx = idx_train[i]
                infl = infl_dict.get(csv_idx, [0.0])[-1]
                for j, (key, param) in enumerate(nn.get_parameters(grad_only=False).items()):
                    infl += lr * (u[key].d * param.g).sum() / idx.size

                # store infl
                file_name = trainset.get_filepath_to_data(csv_idx)
                file_name = os.path.join(input_dir_name, file_name)
                file_name = os.path.normpath(file_name)
                infl_dict[csv_idx] = [file_name, label, infl]

            # update u
            _, loss_fn, input_image = adjust_batch_size(
                infl_model, solver, len(idx), loss_fn)
            input_image["image"].d = X
            input_image["label"].d = np.array(y).reshape(-1, 1)
            loss_fn.forward()
            params = nn.get_parameters(grad_only=False)
            grad_params = {}
            for key, p in zip(params.keys(), nn.grad([loss_fn], params.values())):
                grad_params[key] = p
            ug = 0
            # compute H[t]u[t]
            for key, uu in u.items():
                try:
                    ug += F.sum(uu * grad_params[key])
                except TypeError:
                    # cannot calc grad with batch normalization runnning mean and var
                    pass
            ug.forward()
            solver.zero_grad()
            ug.backward(clear_buffer=True)

            for j, (key, param) in enumerate(nn.get_parameters(grad_only=False).items()):
                u[key].d -= lr * param.g / idx.size

        # sort by influence score
        infl_list = [val + [key] for key, val in infl_dict.items()]
        infl_list = sorted(infl_list, key=lambda x: (x[-2]))

        # save
        header = ['x:image', 'y:label', 'influence', 'datasource_index']
        data_type = 'object,int,float,int'
        if need_evaluate:
            save_infl_for_analysis(
                infl_list, use_all_params, save_dir, infl_filename, epoch, header, data_type)
    save_to_csv(filename=infl_filename, header=header,
                list_to_save=infl_list, data_type=data_type)


def compute_gradient(grad_model, solver, dataset, batch_size, idx_list_to_data, epoch, resize_size):
    n = len(idx_list_to_data)
    grad_idx = get_batch_indices(n, batch_size, seed=None)
    u = {}
    loss_fn = None
    for i in tqdm(grad_idx, desc='calc gradient (2/3 steps)'):
        X, y = get_batch_data(dataset, idx_list_to_data,
                              i, resize_size, test=True)
        _, loss_fn, input_image = adjust_batch_size(
            grad_model, solver, len(X), loss_fn)
        input_image["image"].d = X
        input_image["label"].d = y
        loss_fn.forward()
        solver.zero_grad()
        loss_fn.backward(clear_buffer=True)

        for j, (key, param) in enumerate(nn.get_parameters(grad_only=False).items()):
            uu = u.get(key, None)
            if uu is None:
                u[key] = nn.Variable(param.shape)
                u[key].data.zero()
            u[key].d += param.g / n
    return u
