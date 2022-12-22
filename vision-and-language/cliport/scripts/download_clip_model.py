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
import pathlib

import nnabla as nn
import nnabla.parametric_functions as PF

import clip


def pytorch_to_nn_param_map():
    '''map from tensor name to Nnabla default parameter names
    '''
    return {
        'running_mean': 'mean',
        'running_var': 'var',
        'weight': 'W',
        'bias': 'b',
        '.': '/'
    }


def rename_params(param_name):
    pytorch_to_nn_dict = pytorch_to_nn_param_map()
    for k in pytorch_to_nn_dict:
        if k in param_name:
            param_name = param_name.replace(k, pytorch_to_nn_dict[k])
    return param_name


if __name__ == '__main__':
    outdir = pathlib.Path('./data')
    outdir.mkdir(exist_ok=True, parents=True)

    for available_model in ['RN50']:
        nn.parameter.clear_parameters()

        print(f'loading: {available_model}')
        model, _ = clip.load(available_model)  # 'ViT-B/16', 'ViT-L/16'
        model = model.cpu()

        out_file = f"./{available_model.replace('/', '-')}.h5"

        for k, v in model.state_dict().items():
            key = rename_params(k)
            print(f'param name (renamed): {key}. shape: {v.shape}')
            params = PF.get_parameter_or_create(key, shape=v.shape)
            params.d = v.detach().numpy()

        nn.parameter.save_parameters(str(outdir/out_file))
