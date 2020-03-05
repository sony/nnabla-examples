#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# Modified by Xingyi Zhou
# --------------------------------------------------------

# Reval = re-eval. Re-evaluate saved detections.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os.path as osp

sys.path.insert(0, osp.join(osp.dirname(__file__), 'voc_eval_lib'))
print(sys.path)

from voc_datasets.pascal_voc import pascal_voc
import pickle
import os, argparse
import numpy as np
import json


def from_dets(detection_file, pascal_root):
    imdb = pascal_voc('test', '2007', path=pascal_root)
    imdb.competition_mode(False)
    imdb.config['matlab_eval'] = False
    with open(os.path.join(detection_file), 'rb') as f:
        if 'json' in detection_file:
            dets = json.load(f)
        else:
            dets = pickle.load(f, encoding='latin1')
    nms_dets = dets

    print('Evaluating detections')
    return imdb.evaluate_detections(nms_dets)
