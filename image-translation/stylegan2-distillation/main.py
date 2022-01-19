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

import nnabla as nn
from nnabla.ext_utils import get_extension_context
from nnabla.monitor import Monitor

common_utils_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'utils'))
sys.path.append(common_utils_path)
from neu.comm import CommunicatorWrapper
from neu.yaml_wrapper import read_yaml, write_yaml

from argparse import ArgumentParser
import time

from train import Trainer
from data_iterators.attribute_faces import get_data_iterator_attribute
from data_iterators.mixing_faces import get_data_iterator_mix


def make_parser():
    parser = ArgumentParser(description='StyleGAN2: Nnabla implementation')

    parser.add_argument('--data-root', type=str, required=True,
                        help='Path to the folder containing the images')
    parser.add_argument('--load-path', type=str,
                        help='Path to the saved parameters')
    parser.add_argument('--save-path', type=str,
                        default='results/',
                        help='Path to the save parameters and generation results')
    parser.add_argument('--device-id', type=int,
                        default=0,
                        help='Device ID of the GPU for training')
    parser.add_argument('--face-morph', '--style-mix', action='store_true',
                        default=False,
                        help='Set this flag to train for style mixing data')
    parser.add_argument('--batch-size', type=int,
                        default=4,
                        help='Device ID of the GPU for training')
    parser.add_argument('--g-n-scales', type=int,
                        default=1,
                        help='Number of generator resolution stacks')
    parser.add_argument("--d-n-scales", type=int,
                        default=2,
                        help='Number of layers of discriminator pyramids')

    return parser


if __name__ == '__main__':

    parser = make_parser()
    config = read_yaml(os.path.join('configs', 'gender.yaml'))
    args = parser.parse_args()
    config.nnabla_context.device_id = args.device_id
    config.gender_faces.data_dir = args.data_root
    config.train.save_path = args.save_path
    config.train.batch_size = args.batch_size
    config.model.g_n_scales = args.g_n_scales
    config.model.d_n_scales = args.d_n_scales

    # nn.set_auto_forward(True)

    ctx = get_extension_context(config.nnabla_context.ext_name)
    comm = CommunicatorWrapper(ctx)
    nn.set_default_context(ctx)

    image_shape = tuple(
            x * config.model.g_n_scales for x in config.model.base_image_shape)

    if args.face_morph:
        di = get_data_iterator_mix(
            args.data_root, comm, config.train.batch_size, image_shape)
    else:
        di = get_data_iterator_attribute(
            args.data_root, comm, config.train.batch_size, image_shape)

    if args.load_path:
        nn.load_parameters(args.load_path)

    trainer = Trainer(config.train, config.model, comm,
                      di, face_morph=args.face_morph)

    trainer.train()

    if comm.rank == 0:
        print('Completed!')
