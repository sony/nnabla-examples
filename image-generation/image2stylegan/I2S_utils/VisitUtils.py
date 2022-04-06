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
from collections import OrderedDict


class GetVariablesOnGraph:
    def __init__(self, _y):
        # [variables]
        self.variables = OrderedDict()
        self.coef_dict_on_graph = OrderedDict()
        self._func_count_outputs = OrderedDict()
        self._func_count_inputs = OrderedDict()
        self.coef_dict_source = nn.get_parameters(grad_only=False)
        self.coef_names_by_value_dict = {
            v: k for k, v in self.coef_dict_source.items()}
        # [preprocess]
        _y.visit(lambda f: self._name_variables(f, _type='Output'))
        _y.visit(lambda f: self._name_variables(f, _type='Input'))
        _y.visit(self._get_coef_dict_on_graph)

    def _name_variables(self, f, _type):
        if _type in ['Input', 'Output']:
            if _type == 'Input':
                count_dict = self._func_count_inputs
                var_list = f.inputs
            elif _type == 'Output':
                count_dict = self._func_count_outputs
                var_list = f.outputs
            for var in var_list:
                if var.name == '':
                    func_name = f.info.type_name
                    if func_name in count_dict:
                        count_dict[func_name] += 1
                        var_name = func_name + \
                            '_{}_{}'.format(count_dict[func_name], _type)
                    else:
                        count_dict[func_name] = 1
                        var_name = func_name + '_{}'.format(_type)
                    var.name = var_name
                if _type == 'Output':
                    self.variables[var.name] = var
        else:
            print('[GetVariablesOnGraph] Error in loss.py')
            print('f.info.type_name = {}'.format(f.info.type_name))

    def _get_coef_dict_on_graph(self, f):
        for in_var in f.inputs:
            if in_var in self.coef_names_by_value_dict:
                in_var.name = self.coef_names_by_value_dict[in_var]
                self.coef_dict_on_graph[in_var.name] = in_var
