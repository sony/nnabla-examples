# Copyright 2022 Sony Group Corporation.
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


import numpy as np
np.set_printoptions(precision=32, floatmode='fixed')

import torch
import nnabla as nn
import nnabla.parametric_functions as PF

print("nnabla version:", nn.__version__)
print("Torch version:", torch.__version__)

from collections import OrderedDict
import pickle


def param_map():
    '''map to tensor name from Nnabla default parameter names
    '''
    return {
        '/W': '/weight',
        '_W': '_weight',
        '/b': '/bias',
        '_b': '_bias',
        '/': '.'
    }


def rename_params(param_name):
    map_dict = param_map()
    for k in map_dict:
        if k in param_name:
            param_name = param_name.replace(k, map_dict[k])
    return param_name


def convert(pth_file, nn_h5, prefix=""):
    nn.clear_parameters()
    nn.load_parameters(nn_h5)
    state_dict = OrderedDict()
    named_params = nn.get_parameters()
    for k, v in named_params.items():
        print(k)
        key = rename_params(k)
        value = torch.from_numpy(v.d)
        state_dict[prefix + key] = value

    torch.save(state_dict, pth_file)


if __name__ == '__main__':
    pth_file = "./convert.pth"  # output file
    # input file to be converted
    nn_h5 = "./tmp.monitor/2022-04-20-09-03-39/param_epoch_00029.h5"
    prefix = "module."  # set prefix for each key

    convert(pth_file, nn_h5, prefix)
