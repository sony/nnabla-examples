# Copyright (c) 2021 Sony Corporation. All Rights Reserved.
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

import nnabla.functions as F
import nnabla.parametric_functions as PF
from utils_nn import *

def vgg_net(input_var, pre_process = True, fix = True):
    if pre_process:
        input_var = vgg_pre_process(input_var)
        
    act1_1 = F.relu(PF.convolution(input_var, outmaps=64, kernel=(
        3, 3), pad=(1, 1), fix_parameters=fix, name="conv1_1"))
    act1_2 = F.relu(PF.convolution(act1_1, outmaps=64, kernel=(
        3, 3), pad=(1, 1), fix_parameters=fix, name="conv1_2"))

    p1 = F.max_pooling(act1_2, kernel=(2, 2), stride=(2, 2))
    act2_1 = F.relu(PF.convolution(p1, outmaps=128, kernel=(
        3, 3), pad=(1, 1), fix_parameters=fix, name="conv2_1"))
    act2_2 = F.relu(PF.convolution(act2_1, outmaps=128, kernel=(
        3, 3), pad=(1, 1), fix_parameters=fix, name="conv2_2"))

    p2 = F.max_pooling(act2_2, kernel=(2, 2), stride=(2, 2))
    act3_1 = F.relu(PF.convolution(p2, outmaps=256, kernel=(
        3, 3), pad=(1, 1), fix_parameters=fix, name="conv3_1"))
    act3_2 = F.relu(PF.convolution(act3_1, outmaps=256, kernel=(
        3, 3), pad=(1, 1), fix_parameters=fix, name="conv3_2"))
    act3_3 = F.relu(PF.convolution(act3_2, outmaps=256, kernel=(
        3, 3), pad=(1, 1), fix_parameters=fix, name="conv3_3"))
    act3_4 = F.relu(PF.convolution(act3_3, outmaps=256, kernel=(
        3, 3), pad=(1, 1), fix_parameters=fix, name="conv3_4"))

    p3 = F.max_pooling(act3_4, kernel=(2, 2), stride=(2, 2))
    act4_1 = F.relu(PF.convolution(p3, outmaps=512, kernel=(
        3, 3), pad=(1, 1), fix_parameters=fix, name="conv4_1"))
    act4_2 = F.relu(PF.convolution(act4_1, outmaps=512, kernel=(
        3, 3), pad=(1, 1), fix_parameters=fix, name="conv4_2"))
    act4_3 = F.relu(PF.convolution(act4_2, outmaps=512, kernel=(
        3, 3), pad=(1, 1), fix_parameters=fix, name="conv4_3"))
    act4_4 = F.relu(PF.convolution(act4_3, outmaps=512, kernel=(
        3, 3), pad=(1, 1), fix_parameters=fix, name="conv4_4"))

    p4 = F.max_pooling(act4_4, kernel=(2, 2), stride=(2, 2))
    act5_1 = F.relu(PF.convolution(p4, outmaps=512, kernel=(
        3, 3), pad=(1, 1), fix_parameters=fix, name="conv5_1"))
    act5_2 = F.relu(PF.convolution(act5_1, outmaps=512, kernel=(
        3, 3), pad=(1, 1), fix_parameters=fix, name="conv5_2"))
    act5_3 = F.relu(PF.convolution(act5_2, outmaps=512, kernel=(
        3, 3), pad=(1, 1), fix_parameters=fix, name="conv5_3"))
    act5_4 = F.relu(PF.convolution(act5_3, outmaps=512, kernel=(
        3, 3), pad=(1, 1), fix_parameters=fix, name="conv5_4"))

    return [act1_2, act2_2, act3_2, act4_2, act5_2]
