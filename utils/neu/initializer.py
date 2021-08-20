# Copyright 2020,2021 Sony Corporation.
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

import numpy as np

import nnabla.initializer as I


def w_init(x, out_dims, gain=0.02, type="xavier"):
    if type == "xavier":
        return I.NormalInitializer(sigma=I.calc_normal_std_glorot(x.shape[1], out_dims) * gain)

    raise ValueError("unsupported init type: {}.".format(type))


def pytorch_conv_init(inmaps, kernel):
    scale = 1 / np.sqrt(inmaps * np.prod(kernel))

    return I.UniformInitializer(lim=(-scale, scale))
