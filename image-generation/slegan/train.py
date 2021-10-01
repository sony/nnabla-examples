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


import os
import numpy as np
import nnabla as nn
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
from nnabla.monitor import Monitor, MonitorSeries, MonitorTimeElapsed, MonitorImageTile
import nnabla.utils.save as save
from nnabla.ext_utils import get_extension_context

from args import get_args, save_args
from generator import Generator
from discriminator import Discriminator
from loss import (loss_gen, loss_dis_real, loss_dis_fake)
from datasets import data_iterator
from augment import augment


def make_ema_updater(scope_ema, scope_cur, ema_decay):
    with nn.parameter_scope(scope_cur):
        params_cur = nn.get_parameters()
    with nn.parameter_scope(scope_ema):
        params_ema = nn.get_parameters()
    update_ema_list = []
    for name in params_ema.keys():
        params_ema_updated = ema_decay * \
            params_ema[name] + (1.0 - ema_decay) * params_cur[name]
        update_ema_list.append(F.assign(params_ema[name], params_ema_updated))
    return F.sink(*update_ema_list)


def copy_params(scope_from, scope_to):
    with nn.parameter_scope(scope_from):
        params_from = nn.get_parameters(grad_only=False)
    with nn.parameter_scope(scope_to):
        params_to = nn.get_parameters(grad_only=False)
    for name in params_to.keys():
        params_to[name].d = params_from[name].d


def train(args):
    # Context
    ctx = get_extension_context(
        args.context, device_id=args.device_id, type_config=args.type_config)
    nn.set_default_context(ctx)

    aug_list = args.aug_list

    # Model
    scope_gen = "Generator"
    scope_dis = "Discriminator"
    # generator loss
    z = nn.Variable([args.batch_size, args.latent, 1, 1])
    x_fake = Generator(z, scope_name=scope_gen, img_size=args.image_size)
    p_fake = Discriminator([augment(xf, aug_list)
                            for xf in x_fake], label="fake", scope_name=scope_dis)
    lossG = loss_gen(p_fake)
    # discriminator loss
    x_real = nn.Variable(
        [args.batch_size, 3, args.image_size, args.image_size])
    x_real_aug = augment(x_real, aug_list)
    p_real, rec_imgs, part = Discriminator(
        x_real_aug, label="real", scope_name=scope_dis)
    lossD_fake = loss_dis_fake(p_fake)
    lossD_real = loss_dis_real(p_real, rec_imgs, part, x_real_aug)
    lossD = lossD_fake + lossD_real
    # generator with fixed latent values for test
    # Use train=True even in an inference phase
    z_test = nn.Variable.from_numpy_array(
        np.random.randn(args.batch_size, args.latent, 1, 1))
    x_test = Generator(z_test, scope_name=scope_gen,
                       train=True, img_size=args.image_size)[0]

    # Exponential Moving Average (EMA) model
    # Use train=True even in an inference phase
    scope_gen_ema = "Generator_EMA"
    x_test_ema = Generator(z_test, scope_name=scope_gen_ema,
                           train=True, img_size=args.image_size)[0]
    copy_params(scope_gen, scope_gen_ema)
    update_ema_var = make_ema_updater(scope_gen_ema, scope_gen, 0.999)

    # Solver
    solver_gen = S.Adam(args.lr, beta1=0.5)
    solver_dis = S.Adam(args.lr, beta1=0.5)
    with nn.parameter_scope(scope_gen):
        params_gen = nn.get_parameters()
        solver_gen.set_parameters(params_gen)
    with nn.parameter_scope(scope_dis):
        params_dis = nn.get_parameters()
        solver_dis.set_parameters(params_dis)

    # Monitor
    monitor = Monitor(args.monitor_path)
    monitor_loss_gen = MonitorSeries(
        "Generator Loss", monitor, interval=10)
    monitor_loss_dis_real = MonitorSeries(
        "Discriminator Loss Real", monitor, interval=10)
    monitor_loss_dis_fake = MonitorSeries(
        "Discriminator Loss Fake", monitor, interval=10)
    monitor_time = MonitorTimeElapsed(
        "Training Time", monitor, interval=10)
    monitor_image_tile_train = MonitorImageTile("Image Tile Train", monitor,
                                                num_images=args.batch_size,
                                                interval=1,
                                                normalize_method=lambda x: (x + 1.) / 2.)
    monitor_image_tile_test = MonitorImageTile("Image Tile Test", monitor,
                                               num_images=args.batch_size,
                                               interval=1,
                                               normalize_method=lambda x: (x + 1.) / 2.)
    monitor_image_tile_test_ema = MonitorImageTile("Image Tile Test EMA", monitor,
                                                   num_images=args.batch_size,
                                                   interval=1,
                                                   normalize_method=lambda x: (x + 1.) / 2.)

    # Data Iterator
    rng = np.random.RandomState(141)
    di = data_iterator(args.img_path, args.batch_size,
                       imsize=(args.image_size, args.image_size),
                       num_samples=args.train_samples, rng=rng)

    # Train loop
    for i in range(args.max_iter):
        # Train discriminator
        x_fake[0].need_grad = False  # no need backward to generator
        x_fake[1].need_grad = False  # no need backward to generator
        solver_dis.zero_grad()
        x_real.d = di.next()[0]
        z.d = np.random.randn(args.batch_size, args.latent, 1, 1)
        lossD.forward()
        lossD.backward()
        solver_dis.update()

        # Train generator
        x_fake[0].need_grad = True  # need backward to generator
        x_fake[1].need_grad = True  # need backward to generator
        solver_gen.zero_grad()
        lossG.forward()
        lossG.backward()
        solver_gen.update()

        # Update EMA model
        update_ema_var.forward()

        # Monitor
        monitor_loss_gen.add(i, lossG.d)
        monitor_loss_dis_real.add(i, lossD_real.d)
        monitor_loss_dis_fake.add(i, lossD_fake.d)
        monitor_time.add(i)

        # Save
        if (i+1) % args.save_interval == 0:
            with nn.parameter_scope(scope_gen):
                nn.save_parameters(os.path.join(
                    args.monitor_path, "Gen_iter{}.h5".format(i+1)))
            with nn.parameter_scope(scope_gen_ema):
                nn.save_parameters(os.path.join(
                    args.monitor_path, "GenEMA_iter{}.h5".format(i+1)))
            with nn.parameter_scope(scope_dis):
                nn.save_parameters(os.path.join(
                    args.monitor_path, "Dis_iter{}.h5".format(i+1)))
        if (i+1) % args.test_interval == 0:
            x_test.forward(clear_buffer=True)
            x_test_ema.forward(clear_buffer=True)
            monitor_image_tile_train.add(i+1, x_fake[0])
            monitor_image_tile_test.add(i+1, x_test)
            monitor_image_tile_test_ema.add(i+1, x_test_ema)

    # Last
    x_test.forward(clear_buffer=True)
    x_test_ema.forward(clear_buffer=True)
    monitor_image_tile_train.add(args.max_iter, x_fake[0])
    monitor_image_tile_test.add(args.max_iter, x_test)
    monitor_image_tile_test_ema.add(args.max_iter, x_test_ema)
    with nn.parameter_scope(scope_gen):
        nn.save_parameters(os.path.join(args.monitor_path,
                                        "Gen_iter{}.h5".format(args.max_iter)))
    with nn.parameter_scope(scope_gen_ema):
        nn.save_parameters(os.path.join(args.monitor_path,
                                        "GenEMA_iter{}.h5".format(args.max_iter)))
    with nn.parameter_scope(scope_dis):
        nn.save_parameters(os.path.join(args.monitor_path,
                                        "Dis_iter{}.h5".format(args.max_iter)))


def main():
    args = get_args()
    save_args(args, "train")

    train(args)


if __name__ == '__main__':
    main()
