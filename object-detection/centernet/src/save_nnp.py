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

#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import nnabla as nn

import _init_paths
from models.model import create_model, load_model
from opts import opts
from utils import debugger

########
# Main
########
if __name__ == '__main__':
    nn.set_auto_forward(True)
    opt = opts().init()
    model = create_model(opt.arch, opt.heads, opt.head_conv,
                         opt.num_layers, training=False, channel_last=opt.channel_last)
    if opt.checkpoint != '':
        extension = os.path.splitext(opt.checkpoint)[1]
        assert (extension == '.h5' or
                extension == ".protobuf"), "incorrect file extension, should be .h5 or .protobuf"
        load_model(model, opt.checkpoint, clear=True)

    debugger.save_nnp(opt, model)
