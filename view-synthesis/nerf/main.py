# Copyright 2020,2021 Sony Corporation.
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
from argparse import ArgumentParser
import yaml
import numpy as np

import nnabla as nn
import nnabla.communicators as C

from train.nerf import train_nerf

common_utils_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'utils'))
sys.path.append(common_utils_path)
from neu.yaml_wrapper import read_yaml
from neu.misc import init_nnabla


def get_args():
    parser = ArgumentParser(description='NeRF: Nnabla implementation')

    parser.add_argument('--config-path', '-c', type=str, default='configs/tiny_data.yaml',
                        required=True,
                        help='model and training configuration file')
    parser.add_argument('--model', type=str, choices=['wild', 'uncertainty', 'appearance', 'vanilla'],
                        required=True,
                        help='Select the model to train')

    parser.add_argument('--data-perturb', type=str, choices=['none', 'color', 'occ', 'both'],
                        default='none',
                        help='Select the model to train')
    parser.add_argument('--dataset', type=str, choices=['blender', 'phototourism'],
                        default='blender',
                        help='Select the dataset')
    parser.add_argument('--save-results-dir', type=str, default='',
                        help='Path to save trained model parameters')

    return parser.parse_args()


def main():

    args = get_args()

    config = read_yaml(args.config_path)

    nn.set_auto_forward(True)
    comm = init_nnabla(ext_name="cuda", device_id='0', type_config='float')

    if args.save_results_dir != '':
        config.log.save_results_dir = args.save_results_dir

    config.data.color_perturb = True if (
        args.data_perturb == 'color' or args.data_perturb == 'both') else False
    config.data.occ_perturb = True if (
        args.data_perturb == 'occ' or args.data_perturb == 'both') else False

    if comm is None or comm.rank == 0:
        if config.data.color_perturb:
            print('Applying color perturbation to the dataset')
        if config.data.occ_perturb:
            print('Applying occlusion perturbation to the dataset')
        if not config.data.color_perturb and not config.data.occ_perturb:
            print('No perturbation will be applied to the data')

    train_nerf(config, comm, args.model, args.dataset)


if __name__ == '__main__':
    main()
