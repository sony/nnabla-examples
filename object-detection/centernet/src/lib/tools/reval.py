#!/usr/bin/env python
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

from model.test import apply_nms
from tools.voc_eval_lib.datasets.pascal_voc import pascal_voc
import pickle
import os
import argparse
import numpy as np
import json


def from_dets(imdb_name, detection_file, pascal_root, apply_nms=False):
    imdb = pascal_voc('test', '2007', path=pascal_root)
    imdb.competition_mode(False)
    with open(os.path.join(detection_file), 'rb') as f:
        if 'json' in detection_file:
            dets = json.load(f)
        else:
            dets = pickle.load(f, encoding='latin1')
    # import pdb; pdb.set_trace()
    if apply_nms:
        print('Applying NMS to all detections')
        test_nms = 0.3
        nms_dets = apply_nms(dets, test_nms)
    else:
        nms_dets = dets

    avg_map = (imdb.evaluate_detections(nms_dets))
    return avg_map
