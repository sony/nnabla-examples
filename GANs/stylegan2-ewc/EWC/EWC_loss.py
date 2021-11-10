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
import sys
import os
from collections import OrderedDict

# --- utils ---
from .utils import GetVariablesOnGraph
from .utils import GetCoefficientOnGraph


class ElasticWeightConsolidation:
    def __init__(
        self,
        _y,
        _out_FI_path=None,
        _iter_num=100,
        _apply_function_type_list=['Convolution'],
        _calc_switch=False
    ):
        # input
        # _y : type=nn.Variable(), The generator output
        # _out_FI_path : type=str, The Fisher Information weights result path
        # _iter_num : type=int, The iteration number of calculation for the Fisher Information.
        # _apply_function_type_list : type=list of str, The function type names which EWC applies to.
        # _calc_switch : type=bool, Whether to calculate the Fisher Information forcely.

        # [parameters]
        self.y = _y
        self.out_FI_path = _out_FI_path
        self.iter_num = _iter_num
        self.apply_function_type_list = _apply_function_type_list
        # [variables]
        self.FisherInformation_val_dict = None
        self.coef_dict_for_FI = OrderedDict()
        self.coef_dict_on_graph = None
        self.FI_save_switch = True
        # [hyper parameters]
        self.FI_scope = 'FisherInformation'
        # [preprocessing]
        self.FisherInformation_val_dict = self._get_initial_FI_dict(
            _calc_switch)
        self._preprocessing()

    def _preprocessing(self):
        # --- all coefficients ---
        GCG_class = GetCoefficientOnGraph()
        self.y.visit(GCG_class)
        self.coef_dict_on_graph = GCG_class.variables
        # --- variables which EWC applies to ---
        GVG_class = GetVariablesOnGraph()
        self.y.visit(GVG_class)
        for key in GVG_class.variables:
            var = GVG_class.variables[key]
            if var.parent.info.type_name in self.apply_function_type_list:
                if len(var.parent.inputs) > 1:
                    for in_var in var.parent.inputs[1:]:
                        use_var = self._get_input_node(in_var)
                        if use_var is not None:
                            self.coef_dict_for_FI[use_var.name] = use_var

    def _get_input_node(self, _var, _already_read_list=[]):
        if _var in self.coef_dict_on_graph.values():
            return _var
        else:
            _already_read_list.append(_var)
            if _var.parent is not None:
                for in_var in _var.parent.inputs:
                    if in_var not in _already_read_list:
                        return self._get_input_node(in_var, _already_read_list)

    def __call__(self, _out_var=None):
        # input
        # _out_var : type=nn.Variable(), The discriminator output
        # --- self ---
        # self.coef_dict : type=OrderedDict(), The coefficient dict of the synthesis network (This needs to be on the graph.)
        # self.data_iterator : type=nnabla data iterator

        # output
        # loss : type=nn.Variable()

        # --- Calculation of the Fisher Information ---
        if _out_var is not None:
            temp_need_grad = self.y.need_grad
            self.y.need_grad = True
            if len(self.FisherInformation_val_dict) == 0:
                log_likelihood_var = F.log(F.sigmoid(_out_var))
                for i in range(self.iter_num):
                    log_likelihood_var.forward(clear_no_need_grad=True)
                    self._zero_grad_all()
                    log_likelihood_var.backward(clear_buffer=True)
                    self._accumulate_grads()
                    sys.stdout.write(
                        '\rFisher Information Accumulating ... {}/{}'.format(i+1, self.iter_num))
                    sys.stdout.flush()
                print('')
                for key in self.FisherInformation_val_dict:
                    self.FisherInformation_val_dict[key] /= self.iter_num
            self.y.need_grad = temp_need_grad
        # --- make loss graph ---
        loss = 0
        for key in self.FisherInformation_val_dict:
            key_source = key.replace(self.FI_scope + '/', '')
            FI_var = nn.Variable.from_numpy_array(
                self.FisherInformation_val_dict[key].copy())
            FI_var.name = key
            coef_source_var = nn.Variable.from_numpy_array(
                self.coef_dict_for_FI[key_source].d.copy())
            coef_source_var.name = key.replace(
                self.FI_scope + '/', 'weight_source/')
            loss += F.mean(FI_var *
                           (self.coef_dict_for_FI[key_source] - coef_source_var)**2)
        # --- save Fisher Information ---
        if self.FI_save_switch:
            self._save_FisherInformation()
        print('[ElasticWeightConsolidation] Success!')
        return loss

    def _save_FisherInformation(self):
        if self.out_FI_path is not None:
            os.makedirs(self.out_FI_path.replace(
                self.out_FI_path.split(os.sep)[-1], ''), exist_ok=True)
            np.savez(self.out_FI_path.replace('.npz', ''),
                     **self.FisherInformation_val_dict)
            print(
                '[ElasticWeightConsolidation] Save the calculated fisher information values to...')
            print('[ElasticWeightConsolidation] {}'.format(self.out_FI_path))

    def _get_initial_FI_dict(self, _calc_switch):
        # input
        # _FI_path : type=string or None, Already calculated fisher information.

        # output
        # FI_dict : type=OrderedDict(), key=parameter name, value=np.ndarray

        FI_dict = OrderedDict()
        if self.out_FI_path is not None and os.path.isfile(self.out_FI_path) and not _calc_switch:
            FI_dict = OrderedDict(np.load(self.out_FI_path))
            self.FI_save_switch = False
            print('[ElasticWeightConsolidation] Load EWC weights ... {}'.format(
                self.out_FI_path))
        return FI_dict

    def _zero_grad_all(self):
        for key in self.coef_dict_for_FI:
            self.coef_dict_for_FI[key].g.fill(0)

    def _accumulate_grads(self):
        for key in self.coef_dict_for_FI:
            if self.FI_scope + '/' + key not in self.FisherInformation_val_dict:
                self.FisherInformation_val_dict[self.FI_scope +
                                                '/' + key] = self.coef_dict_for_FI[key].g.copy()
            else:
                self.FisherInformation_val_dict[self.FI_scope +
                                                '/' + key] += self.coef_dict_for_FI[key].g.copy()
