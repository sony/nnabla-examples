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


def get_args(monitor_path='tmp.monitor',
             model_save_path=None,
             description=None):
    import argparse
    import os
    from distutils.util import strtobool

    if model_save_path is None:
        model_save_path = monitor_path
    if description is None:
        parser = argparse.ArgumentParser(
            description='Training TrackIn Base model')

    parser.add_argument('--monitor-path', '-m', type=str, default=monitor_path)

    parser.add_argument('--model_save_path',
                        '-o',
                        type=str,
                        default=model_save_path,
                        help='Directory to save checkpoints and logs.')

    parser.add_argument('--model_save_interval',
                        type=input,
                        default=1)

    parser.add_argument('--shuffle_label',
                        type=strtobool,
                        default=True,
                        help='whether shuffle trainig label or not')

    parser.add_argument('--augmentation', type=strtobool, default=True)

    parser.add_argument('--model', type=str, choices=['resnet23', 'resnet56'])

    parser.add_argument('--input',
                        type=str,
                        default=None,
                        help='input npy file')

    parser.add_argument('--output',
                        type=str,
                        default=None,
                        help='save directory of shuffled numpy array dataset')

    parser.add_argument('--checkpoint',
                        type=str,
                        default=None,
                        help='if resume training, you put your weight')

    parser.add_argument('--resume', type=int, default=0, help='resume epoch')

    parser.add_argument(
        "--device_id",
        "-d",
        type=str,
        default='0',
        help='Device ID the training run on. This is only valid if you specify `-c cudnn`.'
    )

    parser.add_argument("--type_config",
                        "-t",
                        type=str,
                        default='float',
                        help='Type of computation. e.g. "float", "half".')

    parser.add_argument('--context',
                        '-c',
                        type=str,
                        default='cudnn',
                        help="Extension path. ex) cpu, cudnn.")

    parser.add_argument('--train_batch_size',
                        type=int,
                        default=500,
                        help='Batch size for training (per replica).')

    parser.add_argument('--val-iter', type=int, default=100)

    parser.add_argument('--val_batch_size',
                        type=int,
                        default=100,
                        help='Batch size for eval. (per replica)')

    parser.add_argument('--train_epochs',
                        type=int,
                        default=270,
                        help='Number of epochs to train for.')

    parser.add_argument('--seed',
                        '-s',
                        help='random seed number default=0',
                        default=0,
                        type=int)

    args = parser.parse_args()

    if not os.path.isdir(args.model_save_path):
        os.makedirs(args.model_save_path)

    return args
