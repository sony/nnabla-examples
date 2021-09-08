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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import cv2
import numpy as np

import nnabla as nn
from nnabla.ext_utils import get_extension_context
import nnabla.functions as F
from nnabla.utils.nnp_graph import NnpLoader

from opts import opts
from models.decode import ctdet_decode
from detectors.detector_factory import detector_factory

image_ext = ['jpg', 'jpeg', 'png', 'webp']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']


def demo(opt):
    '''
    NNabla configuration
    '''

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    if opt.checkpoint == '':
        print("Please provide trained model")
        return
    # opt.extension_module = 'cpu'
    if opt.extension_module != 'cpu':
        if opt.mixed_precision:
            ctx = get_extension_context(
                opt.extension_module, device_id="0", type_config="half")
        else:
            ctx = get_extension_context(opt.extension_module, device_id="0")
        nn.set_default_context(ctx)
    _, ext = os.path.splitext(opt.checkpoint)
    nn.set_auto_forward(True)
    Detector = detector_factory[opt.task]
    detector = Detector(opt)

    if opt.demo is None:
        print("Please provide input image/folder.")
        return

    if os.path.isdir(opt.demo):
        image_names = []
        ls = os.listdir(opt.demo)
        for file_name in sorted(ls):
            ext = file_name[file_name.rfind('.') + 1:].lower()
            if ext in image_ext:
                image_names.append(os.path.join(opt.demo, file_name))
    else:
        image_names = [opt.demo]
    for (image_name) in image_names:
        assert(os.path.exists(image_name)), "{} not found.".format(image_name)
        ret = detector.run(image_name)

        time_str = ''
        for stat in time_stats:
            time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
        print(time_str)


if __name__ == '__main__':
    opt = opts().init()
    demo(opt)
