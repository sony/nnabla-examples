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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

import nnabla as nn
import nnabla.experimental.viewers as V
from models.model import create_model, load_nnp_model
from opts import opts


def save_network_graph(filename, pred_dict, format='svg'):
    graph = V.SimpleGraph(format=format, verbose=True)
    graph.save(pred_dict['hm'], filename)
    nn.logger.info("Output network graph {}".format(filename))


def main(opt):
    if opt.checkpoint != '':
        input_variable, output_list = load_nnp_model(
            opt.checkpoint, 1, len(opt.heads))
        filename = os.path.join(opt.save_dir, '{}_graph'.format(
            os.path.splitext(os.path.basename(opt.checkpoint))[0]))
    else:
        model = create_model(opt.arch, opt.heads,
                             opt.head_conv, opt.num_layers)
        input_variable = nn.Variable([1, 3, 512, 512])
        output_list = model(input_variable)
        filename = os.path.join(
            opt.save_dir, '{}_{}_graph'.format(opt.arch, opt.num_layers))

    pred_dict = dict()
    for i, key in enumerate(opt.heads):
        pred_dict[key] = output_list[i]

    save_network_graph(filename, pred_dict)


########
# Main
########
if __name__ == '__main__':
    opt = opts().init()
    main(opt)
