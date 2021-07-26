# Copyright (c) 2021 Sony Group Corporation. All Rights Reserved.
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


def get_args():
    import argparse
    from distutils.util import strtobool

    parser = argparse.ArgumentParser(description="Pretraining Model")
    parser.add_argument("--model_save_interval", type=int, default=30)
    parser.add_argument(
        "--shuffle_label",
        type=strtobool,
        default=True,
        help="whether shuffle trainig label or not",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="if resume training, you put your weight",
    )

    parser.add_argument(
        "--device_id",
        "-d",
        type=str,
        default="0",
        help="Device ID the training run on. This is only valid if you specify `-c cudnn`.",
    )

    parser.add_argument(
        "--type_config",
        "-t",
        type=str,
        default="float",
        help='Type of computation. e.g. "float", "half".',
    )

    parser.add_argument(
        "--context",
        "-c",
        type=str,
        default="cudnn",
        help="Extension path. ex) cpu, cudnn.",
    )

    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=100,
        help="Batch size for training (per replica).",
    )

    parser.add_argument("--val-iter", type=int, default=100)
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=100,
        help="Batch size for eval. (per replica)",
    )

    parser.add_argument(
        "--train_epochs", type=int, default=270, help="Number of epochs to train for."
    )

    parser.add_argument(
        "--seed", "-s", help="random seed number default=0", default=0, type=int
    )

    args = parser.parse_args()

    return args
