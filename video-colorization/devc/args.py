# Copyright (c) 2021 Sony Corporation. All Rights Reserved.
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
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device-id', '-d',
                        type=int,
                        default=0,
                        help='device id (default: 0)')
    parser.add_argument('--context', '-c', type=str,
                        default='cudnn', help="Extension path. ex) cpu, cudnn.")
    parser.add_argument(
        "--frame_propagate", default=False, type=bool, help="propagation mode, , please check the paper")
    parser.add_argument("--image_size", type=int, default=[216 * 2, 384 * 2], nargs ="+", help="the image size, eg. 432 768")
    parser.add_argument("--input_path", type=str, default="./sample_videos/clips/v32", help="path of input clips")
    parser.add_argument("--ref_path", type=str, default="./sample_videos/ref/test", help="path of refernce images")
    parser.add_argument("--output_path", type=str, default="./sample_videos/output", help="path of output clips")
    parser.add_argument("--output_video", type=str, default = "video.avi", help = "Video output in *.avi format\. ex) video.avi")
    args = parser.parse_args()
    return args
