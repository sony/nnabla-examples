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
import nnabla as nn
import os
import numpy as np


def LatentInitialization(
    _latent_code,
    _coef_dict,
    _init_type='average',
    _parameter_scope='Generator'
):
    g_type_list = [
        'average',
        'random'
    ]
    if _init_type == g_type_list[0]:
        mean_latent_val = _coef_dict['{}/dlatent_avg'.format(
            _parameter_scope)].d.copy()
        mean_latent_val = np.reshape(
            mean_latent_val, [1, ]+list(mean_latent_val.shape))
        mean_latent_val = np.broadcast_to(mean_latent_val, _latent_code.shape)
        _latent_code.d = mean_latent_val
    elif _init_type == g_type_list[1]:
        _latent_code.d = np.random.rand(*_latent_code.shape)
    elif os.path.isfile(_init_type) and _init_type.endswith('.h5'):
        nn.load_parameters(_init_type)
    else:
        print('[Initialization.py] Error')
        print('_init_type = {}'.format(_init_type))
        print('Please set _type from below.')
        print('g_type_list = {}'.format(g_type_list))
