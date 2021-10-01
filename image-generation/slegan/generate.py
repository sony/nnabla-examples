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


import os
import numpy as np
import nnabla as nn
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.parametric_functions as PF
from nnabla.monitor import Monitor, MonitorImageTile
import nnabla.utils.save as save
from nnabla.ext_utils import get_extension_context

from args import get_args, save_args
from generator import Generator


def generate(args):
    ctx = get_extension_context(
        args.context, device_id=args.device_id, type_config=args.type_config)
    nn.set_default_context(ctx)

    scope_gen = "Generator"
    scope_gen_ema = "Generator_EMA"
    gen_param_path = args.model_load_path + '/Gen_iter100000.h5'
    gen_ema_param_path = args.model_load_path + '/GenEMA_iter100000.h5'
    with nn.parameter_scope(scope_gen):
        nn.load_parameters(gen_param_path)
    with nn.parameter_scope(scope_gen_ema):
        nn.load_parameters(gen_ema_param_path)

    monitor = Monitor(args.monitor_path)
    monitor_image_tile_test = MonitorImageTile("Image Tile", monitor,
                                               num_images=args.batch_size,
                                               interval=1,
                                               normalize_method=lambda x: (x + 1.) / 2.)
    monitor_image_tile_test_ema = MonitorImageTile("Image Tile with EMA", monitor,
                                                   num_images=args.batch_size,
                                                   interval=1,
                                                   normalize_method=lambda x: (x + 1.) / 2.)

    z_test = nn.Variable([args.batch_size, args.latent, 1, 1])
    x_test = Generator(z_test, scope_name=scope_gen,
                       train=True, img_size=args.image_size)[0]
    x_test_ema = Generator(z_test, scope_name=scope_gen_ema,
                           train=True, img_size=args.image_size)[0]
    z_test.d = np.random.randn(args.batch_size, args.latent, 1, 1)

    x_test.forward(clear_buffer=True)
    x_test_ema.forward(clear_buffer=True)
    monitor_image_tile_test.add(0, x_test)
    monitor_image_tile_test_ema.add(0, x_test_ema)


def main():
    args = get_args()
    save_args(args, "generate")

    generate(args)


if __name__ == '__main__':
    main()
