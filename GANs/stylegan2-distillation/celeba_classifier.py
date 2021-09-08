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
import sys
import numpy as np
import functools
import argparse
from tqdm import trange

stargan_utils_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'stargan'))
sys.path.append(stargan_utils_path)
from dataloader import stargan_load_func, get_data_dict

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
from nnabla.ext_utils import get_extension_context
from nnabla.utils.data_iterator import data_iterator_simple


def get_data_loader(attr_path, image_dir, batch_size, batch_size_valid, image_size, attribute='Bangs'):
    dataset, attr2idx, idx2attr = get_data_dict(attr_path, [attribute])
    np.random.seed(313)
    np.random.shuffle(dataset)
    test_dataset = dataset[-4000:]

    training_dataset = dataset[:-4000]
    print("Use {} images for training.".format(len(training_dataset)))

    # create data iterators.
    load_func = functools.partial(stargan_load_func, dataset=training_dataset,
                                  image_dir=image_dir, image_size=image_size, crop_size=image_size)
    data_iterator = data_iterator_simple(load_func, len(
        training_dataset), batch_size, with_file_cache=False, with_memory_cache=False)

    load_func_test = functools.partial(stargan_load_func, dataset=test_dataset,
                                       image_dir=image_dir, image_size=image_size, crop_size=image_size)
    test_data_iterator = data_iterator_simple(load_func_test, len(
        test_dataset), batch_size_valid, with_file_cache=False, with_memory_cache=False)

    return data_iterator, test_data_iterator


def resnet_prediction(image, test=False, ncls=2, nmaps=128, act=F.relu):
    # Residual Unit
    def res_unit(x, scope_name, dn=False):
        C = x.shape[1]
        with nn.parameter_scope(scope_name):
            # Conv -> BN -> Nonlinear
            with nn.parameter_scope("conv1"):
                h = PF.convolution(x, C, kernel=(1, 1), pad=(0, 0),
                                   with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)
                h = act(h)
            # Conv -> BN -> Nonlinear
            with nn.parameter_scope("conv2"):
                h = PF.convolution(h, C, kernel=(3, 3), pad=(1, 1),
                                   with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)
                h = act(h)
            # Conv -> BN
            with nn.parameter_scope("conv3"):
                h = PF.convolution(h, C, kernel=(1, 1), pad=(0, 0),
                                   with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)
            # Residual -> Nonlinear
            h = act(F.add2(h, x))
            # Maxpooling
            if dn:
                h = F.max_pooling(h, kernel=(2, 2), stride=(2, 2))
            return h
    # Conv -> BN -> Nonlinear
    with nn.parameter_scope("conv1"):
        # Preprocess
        if not test:
            image = F.image_augmentation(image, contrast=1.0,
                                         angle=0.25,
                                         flip_lr=True)
            image.need_grad = False
        h = PF.convolution(image, nmaps, kernel=(3, 3),
                           pad=(1, 1), with_bias=False)
        h = PF.batch_normalization(h, batch_stat=not test)
        h = act(h)

    image_size = image.shape[-1]

    for i in range(int(np.log2(image_size))-1):
        h = res_unit(h, f'conv{i*2+2}', False)
        if i != np.log2(image_size)-2:
            h = res_unit(h, f'conv{i*2+3}', True)

    h = F.average_pooling(h, kernel=(4, 4))  # -> 1x1
    pred = PF.affine(h, ncls)

    return pred


def loss_function(pred, label):
    loss = F.mean(F.softmax_cross_entropy(pred, label))
    return loss


def train(args):

    extension_module = args.context
    ctx = get_extension_context(extension_module)
    nn.set_default_context(ctx)
    prediction = functools.partial(
        resnet_prediction, nmaps=64, act=F.relu)

    # Create training graphs
    test = False
    image_train = nn.Variable(
        (args.batch_size, 3, args.image_size, args.image_size))
    label_train = nn.Variable((args.batch_size, 1))
    pred_train = prediction(image_train, test)
    loss_train = loss_function(pred_train, label_train)
    input_image_train = {"image": image_train, "label": label_train}

    # Create validation graph
    test = True
    image_valid = nn.Variable(
        (args.batch_size_valid, 3, args.image_size, args.image_size))
    pred_valid = prediction(image_valid, test)
    input_image_valid = {"image": image_valid}

    # Solvers
    solver = S.Adam(alpha=args.init_lr)
    solver.set_parameters(nn.get_parameters())
    start_point = 0

    # Data Iterator
    tdata, vdata = get_data_loader(args.attr_path, args.image_dir, args.batch_size,
                                   args.batch_size_valid, args.image_size, args.attribute)

    # Training-loop
    pbar = trange(args.num_iters, desc="Training")
    for i in pbar:
        # Validation
        if i % args.val_step == 0:
            va = 0.
            for j in range(4000//args.batch_size_valid):
                image, label = vdata.next()
                input_image_valid["image"].d = image
                pred_valid.forward()
                va += (pred_valid.d.argmax(1) == label.flat).mean()
            va /= (4000//args.batch_size_valid)

        # Forward/Zerograd/Backward
        image, label = tdata.next()
        input_image_train["image"].d = image
        input_image_train["label"].d = label.astype(np.float32)
        loss_train.forward()
        solver.zero_grad()
        loss_train.backward(clear_buffer=True)

        # Solvers update
        solver.update()

        pbar.set_description(f'Batch Train Loss: {loss_train.d}, VA: {va}')
        if i % args.save_param_step == 0 or i == args.num_iters-1:
            nn.save_parameters(os.path.join(args.model_save_path,
                                            'params_%06d.h5' % (i)))

        if i in args.lr_step_iter:
            solver.set_learning_rate(
                solver.get_learning_rate()*args.lr_step_factor)


def main():

    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--num-iters', type=int, default=15000,
                        help='Number of training iterations')
    parser.add_argument('--image-size', type=int,
                        default=512, help='image resolution')
    parser.add_argument('--batch-size', type=int,
                        default=6, help='batch-size for training')
    parser.add_argument('--batch-size-valid', type=int,
                        default=6, help='batch-size for validation')
    parser.add_argument('--attr-path', type=str,
                        default='celeba-hq-512/list_attr_celeba-hq.txt',
                        help='attr.txt file path for celeba')
    parser.add_argument('--image-dir', type=str,
                        default='celeba-hq-512/images',
                        help='directory containing the images')
    parser.add_argument('--attribute', type=str,
                        default='Bangs',
                        help='One of the 40 attributes for CelebA dataset (Case sensitive)')
    parser.add_argument('--init-lr', type=float,
                        default=0.001,
                        help='Inital learning rate for the solver')
    parser.add_argument('--lr-step-factor', type=float,
                        default=0.1,
                        help='Multiplicative Factor for the learning rate')
    parser.add_argument('--lr-step-iter', type=int, nargs='*',
                        default=[500, 5000],
                        help='Iterations after which to apply the learning rate factor')
    parser.add_argument('--save-param-step', type=int,
                        default=300,
                        help='Number of iterations after which to save model parameters')
    parser.add_argument('--val-step', type=int,
                        default=300,
                        help='Number of iterations after which to check model accuracy on validation set')
    parser.add_argument('--model-save-path', type=str,
                        default='bangs',
                        help='path to save parameters')

    parser.add_argument('--context', type=str, default='cudnn')
    args = parser.parse_args()

    os.makedirs(args.model_save_path, exist_ok=True)

    train(args)


if __name__ == '__main__':
    main()
