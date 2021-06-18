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

import numpy as np
from nnabla.initializer import NormalInitializer, UniformInitializer


def torch_initializer(inmaps, kernel):
    d = np.sqrt(1. / (np.prod(kernel) * inmaps))
    return UniformInitializer((-d, d))


def he_initializer(ochan, kernel, rng):
    return NormalInitializer(
        sigma=np.sqrt(2/(kernel*kernel*ochan)),
        rng=rng
    )


def bilinear_depthwise_initializer(ichan, kernel):
    factor = (kernel+1)//2
    if kernel % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (np.arange(kernel).reshape(-1, 1), np.arange(kernel).reshape(1, -1))
    filt = (1 - np.abs(og[0] - center) / factor) * \
        (1 - np.abs(og[1] - center) / factor)
    weight = np.zeros((ichan, kernel, kernel))
    weight = np.broadcast_to(filt, (ichan, kernel, kernel))
    # TODO add swap axis for channel last when support comes
    weight = np.expand_dims(weight, axis=1)
    return np.array(weight)


def bilinear_initializer(ichan, kernel):
    factor = (kernel+1)//2
    if kernel % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (np.arange(kernel).reshape(-1, 1), np.arange(kernel).reshape(1, -1))
    filt = (1 - np.abs(og[0] - center) / factor) * \
        (1 - np.abs(og[1] - center) / factor)
    weight = np.zeros((ichan, ichan, kernel, kernel))
    for i in range(ichan):
        weight[i, i] = filt
    return np.array(weight)
