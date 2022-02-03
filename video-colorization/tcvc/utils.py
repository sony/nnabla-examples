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

from __future__ import absolute_import

import sys
import os

import numpy as np
import nnabla as nn

# Set path to neu
common_utils_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'utils'))
sys.path.append(common_utils_path)

from neu.reporter import Reporter
from neu.variable_utils import set_persistent_all, get_params_startswith
from neu.yaml_wrapper import read_yaml, write_yaml
from neu.misc import init_nnabla, get_current_time, AttrDict
from neu.lr_scheduler import LinearDecayScheduler
from neu.layers import PatchGAN
from neu_utils.losses import get_gan_loss, vgg16_perceptual_loss, vgg16_style_loss, vgg16_get_feat
from data.tcvc_dataset import (create_data_iterator as create_tcvc_iterator,
                                      load_function as tcvc_load_function)

def array2im(image, imtype=np.uint8, normalize=True):
    if isinstance(image, nn.Variable):
        image_numpy = image.data.get_data("r")
    elif isinstance(image, nn.NdArray):
        image_numpy = image.get_data("r")
    else:
        assert isinstance(image, np.ndarray)
        image_numpy = image

    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        #image_numpy = (image_numpy - image_numpy.min()) / (image_numpy.max() - image_numpy.min())
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0      
    #image_numpy = np.clip(image_numpy, 0, 255)
    '''
    if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:        
        image_numpy = image_numpy[:,:,0]
    '''
    return image_numpy.astype(imtype)