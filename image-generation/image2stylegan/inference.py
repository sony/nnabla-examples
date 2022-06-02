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
from nnabla.ext_utils import get_extension_context
import os

# --- models ---
common_utils_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..', 'utils'))
common_StyleGAN2_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'stylegan2-training'))
import sys
sys.path.append(common_utils_path)
sys.path.append(common_StyleGAN2_path)
from neu.yaml_wrapper import read_yaml
from models import Generator

# --- utils ---
from execution.ops import save_generations
from I2S_utils import load_latent_var

from argparse import ArgumentParser


def make_parser():
    parser = ArgumentParser(
        description='Image2StyleGAN Inference: Nnabla implementation')

    # [Model]
    parser.add_argument('--config_path', type=str, default="../stylegan2-training/configs/ffhq.yaml",
                        help='Path to StyleGAN2 configs')
    parser.add_argument('--gen_path', type=str, default="",
                        help='Path to StyleGAN2 pre-trained generator')
    parser.add_argument('--parameter_scope', type=str, default='Generator',
                        help='Path to StyleGAN2 pre-trained generator')
    # [Base latent code]
    parser.add_argument('--latent_path', type=str, default="",
                        help='Path to latent code')
    # [Driving latent code]
    parser.add_argument('--latent_path_drive2', type=str, default="",
                        help='Path to driving latent code 2')
    parser.add_argument('--latent_path_drive3', type=str, default="",
                        help='Path to driving latent code 3')
    # [Calculation core settings]
    parser.add_argument('--context', type=str, default='cudnn',
                        help='Processing core mode')
    parser.add_argument('--device_id', type=int, default=0,
                        help='Device id')
    # [Data settings]
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image size to generate')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Image size to generate')
    # [File settings]
    parser.add_argument('--save_path', type=str, default="./result/sample.png",
                        help='Path to save the generated images')
    # [Processing settings]
    parser.add_argument('--lambda_value', type=float, default=0.5,
                        help='Lambda value for mixing the driving latent codes')
    parser.add_argument('--w_1', type=float, default=1,
                        help='Coefficient of driving latent code 2')
    parser.add_argument('--w_2', type=float, default=1,
                        help='Coefficient of driving latent code 3')
    parser.add_argument('--sign', type=float, default=1,
                        help='Direction of the driving')
    return parser


if __name__ == "__main__":
    # [hyper parameters]
    parser = make_parser()
    args = parser.parse_args()
    # [parameters]
    config = read_yaml(args.config_path)
    drive_switch = os.path.isfile(
        args.latent_path_drive2) and os.path.isfile(args.latent_path_drive3)
    if not os.path.isdir(os.path.dirname(args.save_path)):
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    # [calc core]
    ctx = get_extension_context(
        ext_name=args.context, device_id=args.device_id)
    nn.set_default_context(ctx)
    # [make network]
    generator = Generator(
        generator_config=config['generator'],
        img_size=args.img_size,
        mix_after=config['train']['mix_after'],
        global_scope=args.parameter_scope
    )
    noises = [nn.Variable(
        shape=(args.batch_size, config['train']['latent_dim'])) for _ in range(2)]
    fake_img, latent_var = generator(
        args.batch_size, noises, return_latent=True)
    # --- latent code ---
    latent_var._clear_parent()
    latent_var.need_grad = True
    latent_var.name = 'Embeded_Latent_Code'
    # [load model]
    # --- latent codes ---
    latent_var_base = load_latent_var(args.latent_path, _name_scope='base')
    if drive_switch:
        latent_var_drive2 = load_latent_var(
            args.latent_path_drive2, _name_scope='drive2')
        latent_var_drive3 = load_latent_var(
            args.latent_path_drive3, _name_scope='drive3')
    # --- main network ---
    with nn.parameter_scope(args.parameter_scope):
        nn.load_parameters(args.gen_path)
    # [synthesis]
    latent_var.d = latent_var_base.d
    if drive_switch:
        latent_var.d += args.lambda_value * \
            (latent_var_drive3.d * args.w_1 -
             latent_var_drive2.d * args.w_2) * args.sign
    fake_img.forward(clear_buffer=True)
    save_generations(fake_img, args.save_path)
