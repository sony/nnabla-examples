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
    Arguments set the default values of command line arguments.
    """

    parser = argparse.ArgumentParser(
        description='ZoomingSloMo or only Slo-Mo training argument parser')
    parser.add_argument('--cfg', default="./config.yaml")
    args, _ = parser.parse_known_args()
    conf = read_yaml(args.cfg)

    parser.add_argument('--lmdb-data-gt', type=str, default="datasets/",
                        help='Path to HR frames lmdb for training')

    parser.add_argument('--lmdb-data-lq', type=str, default="datasets/",
                        help='Path to LR frames lmdb for training')

    parser.add_argument('--output-dir', type=str, default="models/",
                        help='Path to store trained models')

    parser.add_argument('--batch-size', type=int, default="12",
                        help='Maximum number of iterations for training')

    parser.add_argument('--gt-size', type=int, default=128,
                        help='Ground truth frame size')

    parser.add_argument('--only-slomo', action='store_true', default=False,
                        help='If True, network will train for Slo-Mo only (No Zooming)')

    args = parser.parse_args()

    # Refine config file variables
    conf.data.lmdb_data_gt = args.lmdb_data_gt
    conf.data.lmdb_data_lq = args.lmdb_data_lq
    conf.data.output_dir = args.output_dir
    conf.train.batch_size = args.batch_size
    conf.train.only_slomo = args.only_slomo
    conf.data.gt_size = args.gt_size if not args.only_slomo else args.gt_size // 4
    conf.data.lr_size = args.gt_size // 4

    return conf
