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
from nnabla.monitor import Monitor, MonitorSeries, MonitorTimeElapsed
import nnabla.solvers as S
from nnabla.utils.image_utils import imread, imresize
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
from loss import Loss
from I2S_utils import LatentInitialization
from I2S_utils import InputPreprocess

# --- pre trained model ---
from I2S_utils import VGG16_custom, MobileNet_custom

# --- utils ---
from execution.ops import save_generations

from argparse import ArgumentParser


def make_parser():
    parser = ArgumentParser(
        description='Image2StyleGAN Optimization: nnabla implementation')

    # [Model]
    parser.add_argument('--config_path', type=str, default="../stylegan2-training/configs/ffhq.yaml",
                        help='Path to StyleGAN2 configs')
    parser.add_argument('--gen_path', type=str, default="",
                        help='Path to StyleGAN2 pre-trained generator')
    parser.add_argument('--parameter_scope', type=str, default='GeneratorEMA',
                        help='Coefficient name header')
    # [Image name and path]
    parser.add_argument('--img_path', type=str, default="./obama.png",
                        help='Path to input image')
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
    parser.add_argument('--save_dir', type=str, default="./result",
                        help='Path to save the generated images')
    # [Processing settings]
    parser.add_argument('--monitor_interval', type=int, default=10,
                        help='Monitor interval during the optimization roop.')
    parser.add_argument('--save_interval', type=str, default=100,
                        help='Save interval during the optimization roop.')
    parser.add_argument('--feature_extractor_model', type=str, default='VGG16',
                        help='Model name for perceptual loss')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='Beta1 for Adam')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='Beta2 for Adam')
    parser.add_argument('--eps', type=float, default=1e-08,
                        help='Epsilon for Adam')
    parser.add_argument('--iter', type=int, default=5000,
                        help='Iteration number')
    parser.add_argument('--init_type', type=str, default="average",
                        help='Latent code initialization mode')
    parser.add_argument('--mse_lambda', type=float, default=1,
                        help='Lambda value for MSE')
    parser.add_argument('--pl_lambda_list', type=list, default=[1, 1, 1, 1],
                        help='Lambda value for perceptual loss')
    return parser


if __name__ == "__main__":
    # [hyper parameters]
    parser = make_parser()
    args = parser.parse_args()
    # [parameters]
    config = read_yaml(args.config_path)
    if args.feature_extractor_model == 'VGG16':
        feature_model = VGG16_custom
    elif args.feature_extractor_model == 'MobileNet':
        feature_model = MobileNet_custom
    else:
        print('[Error] : args.feature_extractor_model')
        print('--feature_extractor_model has to be chose from bellow:')
        print('[VGG16, MobileNet]')
        exit(1)
    # [calc core]
    ctx = get_extension_context(
        ext_name=args.context, device_id=args.device_id)
    nn.set_default_context(ctx)
    # [load model]
    with nn.parameter_scope(args.parameter_scope):
        nn.load_parameters(args.gen_path)
    coef_dict = nn.get_parameters()
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
    real_img = nn.Variable(fake_img.shape)
    # --- latent code ---
    latent_var._clear_parent()
    latent_var.need_grad = True
    latent_var.name = 'Embeded_Latent_Code'
    # [loss functions]
    loss = Loss(
        _fake_img_var=fake_img,
        _real_img_var=real_img,
        _mse_lambda=args.mse_lambda,
        _pl_lambda_list=args.pl_lambda_list,
        _pre_trained_network=feature_model
    )
    # [optimizer]
    solver = S.Adam(
        alpha=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
        eps=args.eps
    )
    nn.clear_parameters()
    nn.parameter.set_parameter(latent_var.name, latent_var)
    latent_dict = {latent_var.name: latent_var}
    solver.set_parameters(latent_dict)
    # [load data]
    img_val = imread(args.img_path, channel_first=True, num_channels=3)
    if img_val.shape[:2] != (args.img_size, args.img_size):
        img_val = imresize(
            img_val, (args.img_size, args.img_size), channel_first=True)
    img_val = InputPreprocess(img_val)
    # [monitor]
    monitor_loss = Monitor(args.save_dir)
    monitor_loss_log = MonitorSeries(
        'loss', monitor_loss, interval=args.monitor_interval)
    monitor_loss_time = MonitorTimeElapsed(
        'time', monitor_loss, interval=args.monitor_interval)
    # [iteration]
    LatentInitialization(latent_var, coef_dict, _init_type=args.init_type,
                         _parameter_scope=args.parameter_scope)
    real_img.d = img_val.copy()
    for i in range(args.iter):
        # --- train ---
        solver.zero_grad()
        loss.forward(clear_no_need_grad=True)
        loss.backward(clear_buffer=True)
        solver.update()
        # --- monitor ---
        monitor_loss_log.add(i, loss.d.copy())
        monitor_loss_time.add(i)
        # --- save parameters and inference ---
        if (i+1) % args.save_interval == 0:
            nn.save_parameters(args.save_dir + os.sep +
                               'latent_code_iter{:05d}.h5'.format(i+1))
            fake_img.forward(clear_buffer=True)
            save_generations(fake_img, args.save_dir + os.sep +
                             'image_iter{:05d}.png'.format(i+1))
    # [save parameters and inference]
    nn.save_parameters(args.save_dir + os.sep +
                       'latent_code_iter{:05d}.h5'.format(i+1))
    fake_img.forward(clear_buffer=True)
    save_generations(fake_img, args.save_dir + os.sep +
                     'image_iter{:05d}.png'.format(i+1))
