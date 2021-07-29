
# Copyright 2021 Sony Group Corporation
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
import os
from utils import *


def get_config():
    """
    Get command line arguments.
    Arguments set the default values of command line arguments
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default="./config.yaml")
    args, _ = parser.parse_known_args()
    conf = read_yaml(args.cfg)

    parser.add_argument('--device-id', '-d',
                        type=int,
                        default=0,
                        help='device id (default: 0)')
    parser.add_argument(
        '--context',
        '-c',
        type=str,
        default='cudnn',
        help="Extension path: cpu or cudnn.")
    parser.add_argument(
        "--frame_propagation",
        default=False,
        type=bool,
        help="color propogation mode")
    parser.add_argument(
        "--image_size",
        type=int,
        default=[
            216 * 2,
            384 * 2],
        nargs="+",
        help="the image size, eg. 432 768")
    parser.add_argument(
        "--input_path",
        type=str,
        default="./images/input/v32",
        help="path of input clips")
    parser.add_argument(
        "--ref_path",
        type=str,
        default="./images/ref/v32",
        help="path of reference images")
    parser.add_argument(
        "--output_path",
        type=str,
        default="./images/output",
        help="path of output clips")
    parser.add_argument(
        "--output_video",
        type=str,
        default="video.avi",
        help="Video output in *.avi for example, video.avi")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./devc_weights/",
        help="path to checkpoint folder"
    )
    args = parser.parse_args()

    conf.data.image_size = args.image_size
    conf.data.input_path = args.input_path
    conf.data.ref_path = args.ref_path
    conf.data.output_path = args.output_path
    conf.data.output_video = args.output_video
    conf.data.frame_propagation = args.frame_propagation
    conf.nnabla_context.context = args.context
    conf.nnabla_context.device_id: args.device_id
    conf.checkpoint.path: args.checkpoint

    return conf
