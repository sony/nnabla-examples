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


def get_args(batch_size=8, image_size=256, max_iter=100000):
    """
    Get command line arguments.

    Arguments set the default values of command line arguments.
    """
    import argparse
    import os

    description = "Example of Lightweight GAN."
    parser = argparse.ArgumentParser(description)

    parser.add_argument("-d", "--device-id", type=str, default="0",
                        help="Device id.")
    parser.add_argument("-c", "--context", type=str, default="cudnn",
                        help="Context.")
    parser.add_argument("--type-config", "-t", type=str, default='float',
                        help='Type of computation. e.g. "float", "half".')
    parser.add_argument("--img-path", type=str,
                        default="~/AnimalFace-dog",
                        help="Image path.")
    parser.add_argument("--image-size", type=int, default=image_size,
                        help="Image size.")
    parser.add_argument("--batch-size", "-b", type=int, default=batch_size,
                        help="Batch size.")
    parser.add_argument("--max-iter", "-i", type=int, default=max_iter,
                        help="Max iterations.")
    parser.add_argument("--save-interval", type=int, default=50000,
                        help="Interval for saving models.")
    parser.add_argument("--test-interval", type=int, default=5000,
                        help="Interval for testing models.")
    parser.add_argument("--latent", type=int, default=256,
                        help="Number of latent variables.")
    parser.add_argument("--monitor-path", type=str, default="./result/tmp",
                        help="Monitor path.")
    parser.add_argument("--model-load-path", type=str, default=".",
                        help="Path to load parameters from")
    parser.add_argument("--train-samples", type=int, default=-1,
                        help="Number of data to be used. When -1 is set all data is used.")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--aug-list", nargs="+",
                        default=["lrflip", "translation", "color"])

    args = parser.parse_args()

    return args


def save_args(args, mode="train"):
    from nnabla import logger
    import os
    if not os.path.exists(args.monitor_path):
        os.makedirs(args.monitor_path)

    path = "{}/Arguments-{}.txt".format(args.monitor_path, mode)
    logger.info("Arguments are saved to {}.".format(path))
    with open(path, "w") as fp:
        for k, v in sorted(vars(args).items()):
            logger.info("{}={}".format(k, v))
            fp.write("{}={}\n".format(k, v))
