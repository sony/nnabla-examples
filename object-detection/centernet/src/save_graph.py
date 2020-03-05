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
        input_variable, output_list = load_nnp_model(opt.checkpoint, 1, len(opt.heads))
        filename = os.path.join(opt.save_dir, '{}_graph'.format(os.path.splitext(os.path.basename(opt.checkpoint))[0]))
    else:
        model = create_model(opt.arch, opt.heads, opt.head_conv, opt.num_layers)
        input_variable = nn.Variable([1, 3, 512, 512])
        output_list = model(input_variable)
        filename = os.path.join(opt.save_dir, '{}_{}_graph'.format(opt.arch, opt.num_layers))

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
