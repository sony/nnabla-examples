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
from .VisitUtils import GetNearestFunction
from .VisitUtils import GetFunctionFromInput


def get_feature_list(_fake_var):
    feature_list = []
    # .functions, key=parameter name, value=the outputs of the functions
    GF_class = GetFunctionFromInput(_fake_var)
    GN_class = GetNearestFunction(
        _base_func_list=['Deconvolution', 'Convolution'],
        _need_func_list=['LeakyReLU'],
        _pass_func_list=[
                'Add2', 'AddScalar', 'Mul2', 'MulScalar', 'Div2', 'DivScalar', 'Sub2',
                'Reshape', 'Transpose', 'Identity', 'Slice', 'Broadcast', 'Pad', 'Convolution'
            ],
        _get_type='next'
    )
    # .variables, key=the output name of the function(_base_func_list), value=the next function(_need_func_list)
    _fake_var.visit(GN_class)
    for key in GF_class.functions:
        if 'affine' not in key and '/W' in key and 'ToRGB' not in key:
            # =Convolution or Deconvolution, ex. the weight name is 'Generator/G_synthesis/8x8/Conv0_up/conv/W'
            func = GF_class.functions[key][0]
            # =the LeakyReLU outputs of above convolution
            feature_var = GN_class.variables[func.outputs[0].name]['next'][0]
            feature_list.append(feature_var)
    feature_list.pop(0)                             # Remove resolution 4x4
    return feature_list


def one_hot_combination(sample_num, choise_num):
    x = F.rand(shape=(sample_num,))
    y_top_k = F.top_k_data(x, k=choise_num, reduce=False, base_axis=0)
    y_top_k_sign = F.sign(y_top_k, alpha=0)
    return y_top_k_sign


def make_symmetric_matrix(_x):
    # input
    # _x : type=nn.Variable(), _x.shape=(batch_size, *, *, *)

    # output
    # j_vector : type=nn.Variable(), j_vector.shape=(batch_size, batch_size - 1, *, *, *)

    batch_size = _x.shape[0]
    var_list = F.split(_x)
    concat_list = []
    # --- split & gather components ---
    for i in range(batch_size):
        tmp_list = []
        for j in range(batch_size):
            if i != j:
                tmp_list.append(
                    F.reshape(var_list[j], [1, ] + list(var_list[j].shape)))
        if len(tmp_list) > 1:
            concat_var = F.concatenate(*tmp_list, axis=0)
        else:
            concat_var = tmp_list[0]
        concat_list.append(
            F.reshape(concat_var, [1, ] + list(concat_var.shape)))
    # --- concatenate ---
    j_vector = F.concatenate(*concat_list, axis=0)
    return j_vector


def make_broadcast_matrix(_x):
    # input
    # _x : type=nn.Variable(), _x.shape=(batch_size, *, *, *)

    # output
    # i_vector : type=nn.Variable(), i_vector.shape=(batch_size, batch_size - 1, *, *, *)

    return F.broadcast(F.reshape(_x, [_x.shape[0], 1] + list(_x.shape[1:])), [_x.shape[0], _x.shape[0] - 1] + list(_x.shape[1:]))


def CosineSimilarity(_vector_1, _vector_2, _index=2):
    # input
    # _vector_1 : type=nn.Variable(), shape=(batch_size, batch_size - 1, *, *, *)
    # _vector_2 : type=nn.Variable(), shape=_vector_1.shape

    # output
    # out : type=nn.Variable(), shape=(batch_size, batch_size - 1)

    sum_axis = [i + _index for i in range(len(_vector_1.shape) - _index)]
    inner_product_var = F.sum(_vector_1*_vector_2, axis=sum_axis)
    norm_product_var = F.norm(
        _vector_1, p=2, axis=sum_axis) * F.norm(_vector_2, p=2, axis=sum_axis)
    return inner_product_var / norm_product_var


def CrossDomainCorrespondence(_fake_var_s, _fake_var_t, _choice_num=4, _layer_fix_switch=False):
    # input
    # _fake_var_s : type=nn.Variable(), fake image variable by source model
    # _fake_var_t : type=nn.Variable(), fake image variable by target model

    # output
    # CDC_loss : type=nn.Variable(), shape=()

    # [get feature keys]
    # =list, the len=12, one of components shape is (batch_size, 64, 256, 256)
    feature_list_s = get_feature_list(_fake_var_s)
    feature_list_t = get_feature_list(_fake_var_t)
    if not _layer_fix_switch:
        feature_gate_var = one_hot_combination(
            len(feature_list_s), _choice_num)        # .shape=(12,)
    else:
        feature_gate_var = nn.Variable.from_numpy_array(
            np.array([0, ] * (len(feature_list_s) - _choice_num) + [1, ] * _choice_num))

    # [Cosine Similarity & Integrate KL divergence]
    KL_var_list = []
    for i in range(len(feature_list_s)):
        # --- change shape ---
        i_vector_s = make_broadcast_matrix(feature_list_s[i])
        j_vector_s = make_symmetric_matrix(feature_list_s[i])
        i_vector_t = make_broadcast_matrix(feature_list_t[i])
        j_vector_t = make_symmetric_matrix(feature_list_t[i])
        # --- cosine similarity ---
        # .shape=(batch_size, batch_size - 1)
        CS_var_s = F.softmax(CosineSimilarity(
            i_vector_s, j_vector_s, _index=2), axis=1)
        CS_var_t = F.softmax(CosineSimilarity(
            i_vector_t, j_vector_t, _index=2), axis=1)
        KL_var = F.sum(F.kl_multinomial(CS_var_s, CS_var_t, base_axis=1))
        KL_var_list.append(F.reshape(KL_var, [1, ]))
        # --- name each variables for debug ---
        feature_list_s[i].name = 'Feature/source/{}'.format(i)
        feature_list_t[i].name = 'Feature/target/{}'.format(i)
        i_vector_s.name = 'Feature/source/{}/i_matrix'.format(i)
        j_vector_s.name = 'Feature/source/{}/j_matrix'.format(i)
        i_vector_t.name = 'Feature/target/{}/i_matrix'.format(i)
        j_vector_t.name = 'Feature/target/{}/j_matrix'.format(i)
        CS_var_s.name = 'CosineSimilarity/source/{}'.format(i)
        CS_var_t.name = 'CosineSimilarity/target/{}'.format(i)
        KL_var.name = 'Kullback-Leibler_Divergence/{}'.format(i)
    KL_var_all = F.concatenate(*KL_var_list, axis=0)

    # [Calculate final loss]
    CDC_loss = F.sum(KL_var_all * feature_gate_var)
    CDC_loss.name = 'CrossDomainCorrespondence_Output'

    return CDC_loss
