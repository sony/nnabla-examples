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


import argparse
import functools
import os

import numpy as np
import nnabla as nn
import nnabla.functions as F

from nnabla.utils.data_iterator import data_iterator
from nnabla.ext_utils import get_extension_context

from tqdm import tqdm
from distutils.util import strtobool

from model import resnet23_prediction, resnet56_prediction, loss_function
from cifar10_load import Cifar10NumpySource


CHECKPOINTS_PATH_FORMAT = "params_{}.h5"


def load_ckpt_path():
    checkpoints = [os.path.join(args.checkpoint, CHECKPOINTS_PATH_FORMAT.format(
        str(i))) for i in range(29, 299, 30)]
    return checkpoints


def load_data():
    x = np.load(os.path.join(args.input, 'x_train.npy'))[:50]
    y = np.load(os.path.join(args.input, 'y_train.npy'))[:50]
    y_shuffle = np.load(os.path.join(args.input, 'y_shuffle_train.npy'))[:50]

    data_source = Cifar10NumpySource(x, y, y_shuffle)
    data_num = len(data_source.labels)
    loader = data_iterator(data_source, 1, None, False, False)

    return loader, data_num, x, y, y_shuffle


def calculate_ckpt_score(dataloader, data_num, image_val, label_val, pred_val, hidden, loss_val):
    ckpt_scores = []
    labels, shuffle = [], []

    bar = tqdm(total=data_num)
    for i in range(data_num):
        bar.update(1)
        inputs = dataloader.next()
        for name, param in nn.get_parameters().items():
            param.grad.zero()  # grad initialize
            if 'affine' not in name:
                param.need_grad = False

        grads = []
        images, labels, shuffle = inputs
        image_val.d, label_val.d = images, shuffle

        loss_val.forward()
        loss_val.backward()

        for name, param in nn.get_parameters().items():
            if 'affine' in name:
                grads.append(param.grad)
        grad_mul = [F.sum(grad * grad) for grad in grads]
        score = F.add_n(*grad_mul)
        ckpt_scores.append(score.data)
    return ckpt_scores


def get_scores(dataloader, data_num, ckpt_paths):
    if args.model == 'resnet23':
        model_prediction = resnet23_prediction
    elif args.model == 'resnet56':
        model_prediction = resnet56_prediction

    prediction = functools.partial(model_prediction,
                                   ncls=10,
                                   nmaps=64,
                                   act=F.relu,
                                   seed=args.seed)

    test = True
    image_val = nn.Variable((1, 3, 32, 32))
    label_val = nn.Variable((1, 1))
    pred_val, hidden = prediction(image_val, test)
    loss_val = loss_function(pred_val, label_val)

    infl_scores = np.zeros((data_num, len(ckpt_paths)), dtype=np.float32)
    for ckpt_ind, ckpt_path in enumerate(ckpt_paths):
        epoch = os.path.splitext(os.path.basename(ckpt_path))[0].split('_')[-1]
        print(f'Epoch: {epoch}')
        nn.load_parameters(ckpt_path)

        ckpt_influences = calculate_ckpt_score(
            dataloader, data_num, image_val, label_val, pred_val, hidden, loss_val)
        if args.save_every_epoch:
            np.save(os.path.join(args.output, (epoch+'_influence.npy')),
                    np.array([float(score) for score in ckpt_influences]))

        infl_scores[:, ckpt_ind] = ckpt_influences
    sum_ckpt_scores = infl_scores.sum(axis=-1)

    return {
        'scores': sum_ckpt_scores
    }


def main():
    extension_module = args.context
    ctx = get_extension_context(extension_module,
                                device_id=args.device_id,
                                type_config=args.type_config)
    nn.set_default_context(ctx)

    dataloader, data_num, images, labels, labels_shuffle = load_data()
    ckpt_paths = load_ckpt_path()

    results = get_scores(dataloader, data_num, ckpt_paths)

    np.save(os.path.join(args.output, 'influence_all_epoch.npy'),
            results['scores'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='evaluation classification')
    parser.add_argument('--input', type=str)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--context', '-c', default='cudnn')
    parser.add_argument('--device-id', type=str, default='0')
    parser.add_argument('--model', type=str, choices=['resnet23', 'resnet56'])
    parser.add_argument('--save_every_epoch', type=strtobool, default=False,
                        help='whether save influence score between every epoch or not')
    parser.add_argument("--type_config",
                        "-t",
                        type=str,
                        default='float',
                        help='Type of computation. e.g. "float", "half".')
    parser.add_argument('--seed',
                        '-s',
                        help='random seed number default=0',
                        default=0,
                        type=int)

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    main()
