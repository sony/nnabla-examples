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
from collections import OrderedDict


class GetVariablesOnGraph:
    def __init__(self):
        self.variables = OrderedDict()
        self._function_count_dict = OrderedDict()

    def __call__(self, f):
        if f.info.type_name in self._function_count_dict:
            self._function_count_dict[f.info.type_name] += 1
        else:
            self._function_count_dict[f.info.type_name] = 1
        function_count = '_{}'.format(
            self._function_count_dict[f.info.type_name]) if self._function_count_dict[f.info.type_name] != 1 else ''
        variable_name = f.info.type_name + function_count + '_Output'
        f.outputs[0].name = variable_name
        self.variables[f.outputs[0].name] = f.outputs[0]


class GetCoefficientOnGraph:
    def __init__(self):
        self.variables = OrderedDict()
        self.coef_dict_source = nn.get_parameters(grad_only=False)
        self.coef_names_by_value_dict = {
            v: k for k, v in self.coef_dict_source.items()}

    def __call__(self, f):
        for in_var in f.inputs:
            if in_var in self.coef_names_by_value_dict:
                in_var.name = self.coef_names_by_value_dict[in_var]
                self.variables[in_var.name] = in_var
