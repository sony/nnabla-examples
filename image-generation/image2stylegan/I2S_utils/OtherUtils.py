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


def load_latent_var(_latent_path, _name_scope, _key='Embeded_Latent_Code'):
    with nn.parameter_scope(_name_scope):
        nn.load_parameters(_latent_path)
        latent_var = nn.get_parameters(grad_only=False)[_key]
    return latent_var
