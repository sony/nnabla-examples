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
from nnabla.ext_utils import get_extension_context
import os
import sys

from models import Generator
from models import Discriminator
common_utils_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..', 'utils'))
sys.path.append(common_utils_path)
from neu.yaml_wrapper import read_yaml

# --- utils ---
from EWC.utils import make_parser

# --- EWC ---
from EWC.EWC_loss import ElasticWeightConsolidation

if __name__ == "__main__":
    # [parameters]
    parser = make_parser()
    args = parser.parse_args()
    config = read_yaml(args.config)
    g_code_name = sys.argv[0]
    # [hyper parameters]
    g_gen_path = os.path.join(args.pre_trained_model,
                              'ffhq-slim-gen-256-config-e.h5')
    g_disc_path = os.path.join(
        args.pre_trained_model, 'ffhq-slim-disc-256-config-e-corrected.h5')
    g_apply_function_type_list = ['Convolution']
    # [calc core]
    ctx = get_extension_context(
        args.extension_module, device_id=args.device_id)
    nn.set_default_context(ctx)
    # [load network]
    print('[{}] Load networks'.format(g_code_name))
    with nn.parameter_scope('Generator'):
        nn.load_parameters(g_gen_path)
    nn.load_parameters(g_disc_path)

    generator = Generator(
        generator_config=config['generator'],
        img_size=args.img_size,
        mix_after=config['train']['mix_after'],
        global_scope='Generator'
    )
    discriminator = Discriminator(
        discriminator_config=config['discriminator'],
        img_size=args.img_size
    )
    noises = [nn.Variable(
        shape=(args.batch_size, config['train']['latent_dim'])) for _ in range(2)]
    fake_img = generator(args.batch_size, noises)
    fake_score = discriminator(fake_img)

    # [EWC loss]
    print('[{}] Calculation the fisher informations'.format(g_code_name))
    EWC_class = ElasticWeightConsolidation(
        _y=fake_img,
        _out_FI_path=args.ewc_weight_path,
        _iter_num=args.ewc_iter,
        _apply_function_type_list=g_apply_function_type_list,
        _calc_switch=True
    )
    _ = EWC_class(fake_score)
