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
import hydra

import os

import pathlib

import nnabla as nn
import nnabla.parametric_functions as PF

import cliport
from cliport import agents
from cliport import dataset
from cliport.utils import utils


cliport_root = pathlib.Path(cliport.__file__).resolve().parent


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


def convert_pytorch_to_nnabla(model):
    for k, v in model.state_dict().items():
        key = rename_params(k)
        # print(f'param name (renamed): {key}. shape: {v.shape}')
        params = PF.get_parameter_or_create(key, shape=v.shape)
        params.d = v.detach().numpy()


@hydra.main(config_path=f'{cliport_root}/cfg', config_name='eval')
def main(vcfg):
    tcfg = utils.load_hydra_config(vcfg['train_config'])
    name = '{}-{}-n{}'.format(vcfg['eval_task'],
                              vcfg['agent'], vcfg['n_demos'])
    mode = vcfg['mode']
    eval_task = vcfg['eval_task']

    # Load eval dataset.
    dataset_type = vcfg['type']
    if 'multi' in dataset_type:
        ds = dataset.RavensMultiTaskDataset(vcfg['data_dir'],
                                            tcfg,
                                            group=eval_task,
                                            mode=mode,
                                            n_demos=vcfg['n_demos'],
                                            augment=False)
    else:
        ds = dataset.RavensDataset(os.path.join(vcfg['data_dir'], f"{eval_task}-{mode}"),
                                   tcfg,
                                   n_demos=vcfg['n_demos'],
                                   augment=False)

    agent = agents.names[vcfg['agent']](name, tcfg, None, ds)

    ckpt = 'steps=400000-val_loss=0.00014655.ckpt'
    model_file = os.path.join(vcfg['model_path'], ckpt)

    # Load checkpoint
    agent.load(model_file)

    outdir = pathlib.Path('./data')
    outdir.mkdir(exist_ok=True, parents=True)

    attention_model = agent.attention.cpu()
    attention_file = "./cliport_attention.h5"
    nn.parameter.clear_parameters()
    convert_pytorch_to_nnabla(attention_model)
    nn.parameter.save_parameters(str(outdir/attention_file))

    transport_model = agent.transport.cpu()
    transport_file = "./cliport_transport.h5"
    nn.parameter.clear_parameters()
    convert_pytorch_to_nnabla(transport_model)
    nn.parameter.save_parameters(str(outdir/transport_file))


if __name__ == '__main__':
    main()
