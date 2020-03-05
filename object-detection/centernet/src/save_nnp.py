#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

###########################################################
#
#       File Name: save_nnp.py
#
#       Description:
#
#       Notes: (C) Copyright 2019 Sony Corporation
#
#       Author: Hsingying Ho
#
###########################################################

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
    model = create_model(opt.arch, opt.heads, opt.head_conv, opt.num_layers, training=False)
    if opt.checkpoint != '':
        extension = os.path.splitext(opt.checkpoint)[1]
        assert (extension == '.h5' or
                extension == ".protobuf"), "incorrect file extension, should be .h5 or .protobuf"
        load_model(model, opt.checkpoint, clear=True)

    debugger.save_nnp(opt, model)
