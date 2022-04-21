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
import copy
import pathlib

import nnabla as nn


class Model(object):
    """Model Class
    Args:
        scope_name (str): the scope name of model
    """

    def __init__(self, scope_name: str):
        self._scope_name = scope_name

    @property
    def scope_name(self):
        return self._scope_name

    def get_parameters(self, grad_only=True):
        with nn.parameter_scope(self.scope_name):
            parameters = nn.get_parameters(grad_only=grad_only)
            return parameters

    def save_parameters(self, filepath) -> None:
        if isinstance(filepath, pathlib.Path):
            filepath = str(filepath)
        with nn.parameter_scope(self.scope_name):
            nn.save_parameters(path=filepath)

    def load_parameters(self, filepath) -> None:
        if isinstance(filepath, pathlib.Path):
            filepath = str(filepath)
        with nn.parameter_scope(self.scope_name):
            nn.load_parameters(path=filepath)

    def deepcopy(self, new_scope_name):
        assert new_scope_name != self._scope_name, 'Can not use same scope_name!'
        copied = copy.deepcopy(self)
        copied._scope_name = new_scope_name
        # copy current parameter if is already created
        params = self.get_parameters(grad_only=False)
        with nn.parameter_scope(new_scope_name):
            for param_name, param in params.items():
                if nn.parameter.get_parameter(param_name) is not None:
                    raise RuntimeError(
                        f'Model with scope_name: {new_scope_name} already exists!!')
                nn.parameter.get_parameter_or_create(
                    param_name, shape=param.shape, initializer=param.d)
        return copied
