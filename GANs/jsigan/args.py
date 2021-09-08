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

import argparse
from utils import read_yaml


def get_config():
    """
    Get command line arguments.
    Arguments set the default values of command line arguments
    """
    parser = argparse.ArgumentParser(description='JSIGAN')
    parser.add_argument('--cfg', default="./config.yaml")
    args, _ = parser.parse_known_args()

    conf = read_yaml(args.cfg)

    parser.add_argument('--lr-sdr-test', type=str, default="datasets/",
                        help='The directory of the input data, for testing')
    parser.add_argument('--hr-hdr-test', type=str, default="datasets/",
                        help='The directory of the input data, for testing')
    parser.add_argument('--lr-sdr-train', type=str, default="datasets/",
                        help='The directory of the input data, for training')
    parser.add_argument('--hr-hdr-train', type=str, default="datasets/",
                        help='The directory of the input data, for training')
    parser.add_argument('--scaling-factor', type=int, default=4,
                        help='LR to HR scaling factor')
    parser.add_argument('--pre-trained-model', type=str, default="models/",
                        help='Path of the pre trained weights')
    parser.add_argument('--output-dir', type=str, default="models/",
                        help='Path to save weight files during training')
    parser.add_argument('--jsigan', action='store_true', default=False,
                        help='If True, GAN network will be trained, False otherwise')
    parser.add_argument('--save-images', action='store_true', default=False,
                        help='If True, images will be saved during inference; False to save time')

    args = parser.parse_args()

    conf.data.lr_sdr_test = args.lr_sdr_test
    conf.data.hr_hdr_test = args.hr_hdr_test
    conf.data.lr_sdr_train = args.lr_sdr_train
    conf.data.hr_hdr_train = args.hr_hdr_train
    conf.scaling_factor = args.scaling_factor
    conf.pre_trained_model = args.pre_trained_model
    conf.output_dir = args.output_dir
    conf.jsigan = args.jsigan
    conf.save_images = args.save_images

    return conf
