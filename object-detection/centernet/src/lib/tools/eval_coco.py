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
# Reference: https://github.com/xingyizhou/CenterNet
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import sys
import cv2
import numpy as np
import pickle
import os

this_dir = os.path.dirname(__file__)
ANN_PATH = this_dir + '../../data/coco/annotations/instances_val2017.json'
print(ANN_PATH)
if __name__ == '__main__':
    pred_path = sys.argv[1]
    coco = coco.COCO(ANN_PATH)
    dets = coco.loadRes(pred_path)
    img_ids = coco.getImgIds()
    num_images = len(img_ids)
    coco_eval = COCOeval(coco, dets, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
