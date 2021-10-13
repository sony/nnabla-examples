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

import nnabla as nn
import nnabla.functions as F
import numpy as np


def combination(sample_num, choise_num):
    x = F.rand(shape=(sample_num,))
    x_indices = nn.Variable.from_numpy_array(np.arange(sample_num,) + 1)
    y_top_k = F.top_k_data(x, k=choise_num, reduce=False, base_axis=0)
    y_top_k_sign = F.sign(y_top_k, alpha=0)
    y_top_k_indices = F.top_k_data(
        y_top_k_sign*x_indices, k=choise_num, base_axis=0)
    return y_top_k_indices


class NoiseTop:
    def __init__(self, n_train, latent_dim, batch_size, anch_std=0.05, _PD_switch_var=None):
        # [parameters]
        self.n_train = n_train
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.anch_std = anch_std
        # [variables]
        if type(_PD_switch_var) != type(nn.Variable()):
            self.PD_switch_var = nn.Variable()
        else:
            self.PD_switch_var = _PD_switch_var
        self.PD_switch_var.name = 'Patch_Discriminator_switch'
        self.init_z_var = nn.Variable.from_numpy_array(
            np.random.randn(self.n_train, self.latent_dim))

    def __call__(self):
        # [Z_anchor]
        z_anchor_list = self.generate_z_anchor()
        # [Z_normal]
        z_normal_list = self.generate_z_normal()
        # [integrate]
        z_list = []
        PD_switch_var_reshape = F.reshape(
            self.PD_switch_var, [1, ]*len(z_anchor_list[0].shape))
        for i in range(2):
            z_list.append((1 - PD_switch_var_reshape) *
                          z_normal_list[i] + PD_switch_var_reshape * z_anchor_list[i])
        return z_list

    def generate_z_anchor(self):
        z_anchor_list = []
        for _ in range(2):
            z_anchor_var = F.gather(self.init_z_var, combination(
                self.n_train, self.batch_size)) + F.randn(sigma=self.anch_std, shape=(self.batch_size, self.latent_dim))
            z_anchor_list.append(z_anchor_var)
        return z_anchor_list

    def generate_z_normal(self):
        z_normal_list = [
            F.randn(shape=(self.batch_size, self.latent_dim)) for _ in range(2)]
        return z_normal_list
