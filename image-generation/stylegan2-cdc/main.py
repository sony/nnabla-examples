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

import nnabla as nn
from nnabla.ext_utils import get_extension_context
from nnabla.monitor import Monitor

from argparse import ArgumentParser
import time


from execution import *

common_utils_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..', 'utils'))
sys.path.append(common_utils_path)
from neu.yaml_wrapper import read_yaml, write_yaml
from neu.comm import CommunicatorWrapper

import shutil


def make_parser():
    parser = ArgumentParser(description='StyleGAN2: Nnabla implementation')

    parser.add_argument('--data', type=str, default='ffhq', choices=['ffhq'],
                        help='Model dataset')
    parser.add_argument('--dataset_path', type=str, default='',
                        help='Path to dataset')
    parser.add_argument('--few_shot', type=str, default='few_shot', choices=['few_shot', 'None'],
                        help='Model dataset')

    parser.add_argument('--weights_path', type=str, default='../results/weights',
                        help='Path to trained model weights')
    parser.add_argument('--results_dir', type=str, default='../results/images',
                        help='Path to save results')
    parser.add_argument('--monitor_path', '-mp', type=str, default='../results/monitor',
                        help='Path to save results')

    # # [few-shot learning]
    parser.add_argument('--pre_trained_model', type=str, default='path to pre trained model',
                        help='Path to trained model weights')

    parser.add_argument('--extension_module', type=str, default='cudnn',
                        help='Device context')
    parser.add_argument('--device_id', type=str, default='0',
                        help='Device Id')

    parser.add_argument('--img_size', type=int, default=256,
                        help='Image size to generate')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Image size to generate')

    parser.add_argument('--train', action='store_true', default=False,
                        help='Set this flag to start training')
    parser.add_argument('--auto_forward', action='store_true', default=False,
                        help='Set this flag to execute in dynamic computation mode')
    parser.add_argument('--dali', action='store_true', default=False,
                        help='Set this flag to use DALI data iterator')

    parser.add_argument('--seed_1', type=list, default=[100, 101],
                        help='Seed values 1')
    parser.add_argument('--seed_2', type=list, default=[102, 103],
                        help='Seed values 2')
    parser.add_argument('--test', type=str, choices=['generate', 'latent_space_interpolation', 'style_mixing', 'latent_space_projection', 'ppl'], nargs='*',
                        help='Set this flag for testing')
    parser.add_argument('--batch_size_A', type=int, default=3,
                        help='Only for style mixing: Batch size for style A')
    parser.add_argument('--batch_size_B', type=int, default=3,
                        help='Only for style mixing: Batch size for style B')
    parser.add_argument('--use_tf_weights', action='store_true', default=False,
                        help='Use TF trained weights converted to NNabla')

    parser.add_argument('--img_path', type=str,
                        default='',
                        help='Image path for latent space projection')

    return parser


if __name__ == '__main__':

    parser = make_parser()
    args = parser.parse_args()
    config = read_yaml(os.path.join('configs', f'{args.data}.yaml'))
    ctx = get_extension_context(args.extension_module)
    nn.set_auto_forward(args.auto_forward or args.test)

    comm = CommunicatorWrapper(ctx)
    nn.set_default_context(ctx)

    monitor = None
    if comm is not None:
        if comm.rank == 0:
            monitor = Monitor(args.monitor_path)
            start_time = time.time()

    few_shot_config = None
    if args.few_shot is not None:
        few_shot_config = read_yaml(os.path.join(
            'configs', args.few_shot + '.yaml'))

    if args.train:
        style_gan = Train(monitor, config, args, comm, few_shot_config)
    if args.test:
        style_gan = Evaluate(monitor, config, args, comm, few_shot_config)

    if comm is not None:
        if comm.rank == 0:
            end_time = time.time()
            training_time = (end_time-start_time)/3600

            print('Total running time: {} hours'.format(training_time))
