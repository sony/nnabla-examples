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
import nnabla.functions as F
import nnabla.solvers as S
from nnabla.monitor import Monitor, MonitorSeries, MonitorImageTile
from nnabla.utils.data_iterator import data_iterator_simple, data_iterator

import os
import subprocess as sp
from tqdm import trange
from collections import namedtuple

from .base import BaseExecution
from models import *
from .ops import *
from .losses import *
from data import *


class Train(BaseExecution):
    """
    Execution model for StyleGAN training and testing
    """

    def __init__(self, monitor, config, args, comm):
        super(Train, self).__init__(monitor, config, args, comm)

        # Initialize Monitor
        self.monitor_train_loss, self.monitor_train_gen = None, None
        self.monitor_val_loss, self.monitor_val_gen = None, None
        if comm is not None:
            if comm.rank == 0:
                self.monitor_train_gen_loss = MonitorSeries(
                        config['monitor']['train_loss'], monitor,
                        interval=self.config['logger_step_interval']
                        )
                self.monitor_train_gen = MonitorImageTile(
                        config['monitor']['train_gen'], monitor,
                        interval=self.config['logger_step_interval'], num_images=self.config['batch_size']
                        )
                self.monitor_train_disc_loss = MonitorSeries(
                        config['monitor']['train_loss'], monitor,
                        interval=self.config['logger_step_interval']
                        )

        os.makedirs(self.config['saved_weights_dir'], exist_ok=True)
        self.results_dir = args.results_dir
        self.save_weights_dir = args.weights_path

        # Initialize Discriminator
        self.discriminator = Discriminator(
            config['discriminator'], self.img_size)
        self.gen_exp_weight = 0.5 ** (32 / (10 * 1000))
        self.generator_ema = Generator(
            config['generator'], self.img_size, config['train']['mix_after'], global_scope='GeneratorEMA')

        # Initialize Solver
        if 'gen_solver' not in dir(self):
            if self.config['solver'] == 'Adam':
                self.gen_solver = S.Adam(beta1=0, beta2=0.99)
                self.disc_solver = S.Adam(beta1=0, beta2=0.99)
            else:
                self.gen_solver = eval('S.'+self.config['solver'])()
                self.disc_solver = eval('S.'+self.config['solver'])()

        self.gen_solver.set_learning_rate(self.config['learning_rate'])
        self.disc_solver.set_learning_rate(self.config['learning_rate'])

        self.gen_mean_path_length = 0.0

        self.args = args
        # Initialize Dataloader
        if args.data == 'ffhq':
            if args.dali:
                self.train_loader = get_dali_iterator_ffhq(
                    config['data'], self.img_size, self.batch_size, self.comm)
            else:
                self.train_loader = get_data_iterator_ffhq(
                    config['data'], self.batch_size, self.img_size, self.comm)
        else:
            print('Dataset not recognized')
            exit(1)

        # Start training
        self.train()

    def build_static_graph(self):
        real_img = nn.Variable(
            shape=(self.batch_size, 3, self.img_size, self.img_size))
        noises = [F.randn(shape=(self.batch_size, self.config['latent_dim']))
                  for _ in range(2)]

        if self.config['regularize_gen']:
            fake_img, dlatents = self.generator(
                self.batch_size, noises, return_latent=True)
        else:
            fake_img = self.generator(self.batch_size, noises)
        fake_img_test = self.generator_ema(self.batch_size, noises)

        gen_loss = gen_nonsaturating_loss(self.discriminator(fake_img))

        fake_disc_out = self.discriminator(fake_img)
        real_disc_out = self.discriminator(real_img)
        disc_loss = disc_logistic_loss(real_disc_out, fake_disc_out)

        var_name_list = ['real_img', 'noises', 'fake_img', 'gen_loss',
                         'disc_loss', 'fake_disc_out', 'real_disc_out', 'fake_img_test']
        var_list = [real_img, noises, fake_img, gen_loss,
                    disc_loss, fake_disc_out, real_disc_out, fake_img_test]

        if self.config['regularize_gen']:
            dlatents.need_grad = True
            mean_path_length = nn.Variable()
            pl_reg, path_mean, _ = gen_path_regularize(
                    fake_img=fake_img,
                    latents=dlatents,
                    mean_path_length=mean_path_length
            )
            path_mean_update = F.assign(mean_path_length, path_mean)
            path_mean_update.name = 'path_mean_update'
            pl_reg += 0*path_mean_update
            gen_loss_reg = gen_loss + pl_reg
            var_name_list.append('gen_loss_reg')
            var_list.append(gen_loss_reg)

        if self.config['regularize_disc']:
            real_img.need_grad = True
            real_disc_out = self.discriminator(real_img)
            disc_loss_reg = disc_loss + self.config['r1_coeff']*0.5*disc_r1_loss(
                real_disc_out, real_img)*self.config['disc_reg_step']
            real_img.need_grad = False
            var_name_list.append('disc_loss_reg')
            var_list.append(disc_loss_reg)

        Parameters = namedtuple('Parameters', var_name_list)
        self.parameters = Parameters(*var_list)

    def forward_backward_pass(self, i, real_img=None, noises=None):
        """1 training step: forward and backward propagation

        Args:
                real_img (nn.Variable): [description]
                noises (list of nn.Variable): [description]

        Returns:
                [type]: [description]
        """		"""
		Graph construction for forward pass for training
		"""

        # Update Discriminator
        if self.auto_forward:

            real_disc_out = self.discriminator(real_img)

            fake_img = self.generator(real_img.shape[0], noises)
            fake_disc_out = self.discriminator(
                fake_img.get_unlinked_variable(need_grad=True))
            disc_loss = disc_logistic_loss(real_disc_out, fake_disc_out)

            if self.config['regularize_disc'] and i % self.config['disc_reg_step'] == 0:
                real_img.need_grad = True
                real_disc_out = self.discriminator(real_img)
                r1_loss = disc_r1_loss(real_disc_out, real_img)
                reg_loss = self.config['r1_coeff']/2 * \
                    r1_loss*self.config['disc_reg_step']

                disc_loss += reg_loss

            disc_params = {k: v for k, v in nn.get_parameters(
            ).items() if 'Discriminator' in k}
            self.disc_solver.set_parameters(
                disc_params, reset=False, retain_state=True)

            self.disc_solver.zero_grad()
            disc_loss.backward(clear_buffer=True)
            if self.comm is not None:
                params = [
                    x.grad for x in self.disc_solver.get_parameters().values()]
                self.comm.all_reduce(params, division=False, inplace=True)

        else:
            self.disc_solver.zero_grad()

            self.parameters.fake_img.need_grad = False
            self.parameters.fake_img.forward()

            if self.config['regularize_disc'] and i % self.config['disc_reg_step'] == 0:
                self.parameters.disc_loss_reg.forward(clear_no_need_grad=True)
                self.parameters.disc_loss_reg.backward(clear_buffer=True)
            else:
                self.parameters.disc_loss.forward(clear_no_need_grad=True)
                self.parameters.disc_loss.backward(clear_buffer=True)
            if self.comm is not None:
                params = [
                    x.grad for x in self.disc_solver.get_parameters().values()]
                self.comm.all_reduce(params, division=False, inplace=True)

        self.disc_solver.update()

        # Update Generator

        if self.auto_forward:
            fake_img = self.generator(real_img.shape[0], noises)
            fake_disc_out = self.discriminator(fake_img)
            gen_loss = gen_nonsaturating_loss(fake_disc_out)

            if self.config['regularize_gen']:
                path_batch_size = max(
                    1, fake_img.shape[0]//self.config['path_batch_shrink'])
                fake_img, latents = self.generator(
                    real_img.shape[0], noises, return_latent=True)

                path_loss, self.gen_mean_path_length, _ = gen_path_regularize(
                    fake_img, latents, self.gen_mean_path_length)

                weighted_path_loss = self.config['path_regularize'] * \
                    self.config['gen_reg_step']*path_loss
                gen_loss += weighted_path_loss

            gen_params = {k: v for k, v in nn.get_parameters(
            ).items() if 'Discriminator' not in k}
            self.gen_solver.set_parameters(
                gen_params, reset=False, retain_state=True)
            fake_img.grad.zero()
            self.gen_solver.zero_grad()
            gen_loss.backward(clear_buffer=True)
            if self.comm is not None:
                params = [
                    x.grad for x in self.gen_solver.get_parameters().values()]
                self.comm.all_reduce(params, division=False, inplace=True)
            self.gen_solver.update()

            return gen_loss, disc_loss, fake_img, real_disc_out, fake_disc_out

        else:
            self.gen_solver.zero_grad()
            self.parameters.fake_img.need_grad = True
            if self.config['regularize_gen'] and i % self.config['gen_reg_step'] == 0:
                self.parameters.gen_loss_reg.forward(clear_no_need_grad=True)
                self.parameters.gen_loss_reg.backward(clear_buffer=True)
            else:
                self.parameters.gen_loss.forward(clear_no_need_grad=True)
                self.parameters.gen_loss.backward(clear_buffer=True)
            if self.comm is not None:
                params = [
                    x.grad for x in self.gen_solver.get_parameters().values()]
                self.comm.all_reduce(params, division=False, inplace=True)
            self.gen_solver.update()

    def ema_update(self):
        with nn.parameter_scope('Generator'):
            g_params = nn.get_parameters(grad_only=False)
        with nn.parameter_scope('GeneratorEMA'):
            g_ema_params = nn.get_parameters(grad_only=False)
        update_ema_list = []
        for name in g_ema_params.keys():
            params_ema_updated = self.gen_exp_weight * \
                    g_ema_params[name] + \
                        (1.0 - self.gen_exp_weight) * g_params[name]
            update_ema_list.append(
                F.assign(g_ema_params[name], params_ema_updated))
        return F.sink(*update_ema_list)

    def copy_params(self, scope_from, scope_to):
        with nn.parameter_scope(scope_from):
            params_from = nn.get_parameters(grad_only=False)
        with nn.parameter_scope(scope_to):
            params_to = nn.get_parameters(grad_only=False)
        for name in params_to.keys():
            params_to[name].d = params_from[name].d

    def train(self):
        """
        Training loop: Runs forward_backward pass for the specified number of iterations and stores the generated images, model weigths and solver states
        """
        # n_procs = 1 if self.comm is None else self.comm.n_procs
        iterations_per_epoch = int(np.ceil(
                self.train_loader.size/self.train_loader.batch_size))

        if not self.auto_forward:
            self.build_static_graph()
            disc_params = {k: v for k, v in nn.get_parameters(
            ).items() if k.startswith('Discriminator')}
            self.disc_solver.set_parameters(disc_params)
            gen_params = {k: v for k, v in nn.get_parameters().items() if (
                k.startswith('Generator') and not k.startswith('GeneratorEMA'))}
            self.gen_solver.set_parameters(gen_params)
            if os.path.isfile(os.path.join(self.args.weights_path, 'gen_solver.h5')):
                self.gen_solver.load_states(os.path.join(
                    self.args.weights_path, 'gen_solver.h5'))
                self.disc_solver.load_states(os.path.join(
                    self.args.weights_path, 'disc_solver.h5'))

            self.copy_params('Generator', 'GeneratorEMA')

            ema_updater = self.ema_update()

        for epoch in range(self.config['num_epochs']):
            pbar = trange(iterations_per_epoch, desc='Epoch ' +
                          str(epoch), disable=self.comm.rank > 0)

            epoch_gen_loss, epoch_disc_loss = 0.0, 0.0
            print(
                f'Iterations per epoch: {iterations_per_epoch}, Number of processes: {self.comm.n_procs}, Data Loader size: {self.train_loader.size}')

            for i in pbar:

                data = self.train_loader.next()

                if self.auto_forward:
                    real_img = nn.Variable(data[0].shape)
                    if isinstance(data[0], nn.NdArray):
                        real_img.data = data[0]
                    else:
                        real_img.d = data[0]
                    noises = [nn.Variable(
                        shape=(self.batch_size, self.config['latent_dim'])).apply(d=noises_data[0])]
                    noises += [nn.Variable(shape=(self.batch_size,
                                                  self.config['latent_dim'])).apply(d=noises_data[1])]
                    gen_loss, disc_loss, fake_img, real_disc_out, fake_disc_out = self.forward_backward_pass(
                        i, real_img, noises)
                else:
                    if isinstance(data[0], nn.NdArray):
                        self.parameters.real_img.data = data[0]
                    else:
                        self.parameters.real_img.d = data[0]

                    self.forward_backward_pass(i)
                    gen_loss = self.parameters.gen_loss
                    disc_loss = self.parameters.disc_loss
                    real_img = self.parameters.real_img
                    fake_img = self.parameters.fake_img
                    real_disc_out = self.parameters.real_disc_out
                    fake_disc_out = self.parameters.fake_disc_out

                    ema_updater.forward()

                epoch_gen_loss += gen_loss.d
                epoch_disc_loss += disc_loss.d

                pbar.set_description(
                    f'Gen Loss: {gen_loss.d}, Disc Loss: {disc_loss.d}')

                if np.isnan(gen_loss.d) or np.isnan(disc_loss.d):
                    for k, v in nn.get_parameters().items():
                        if v.d.max() < 1e-3 or np.any(np.isnan(v.d)):
                            print(k)

                if self.comm.rank == 0 and (i == iterations_per_epoch-1 and (epoch % self.config['save_param_step_interval'] == 0 or epoch == self.config['num_epochs']-1)):
                    self.save_weights(
                            self.save_weights_dir, epoch)
                    if not self.auto_forward:
                        self.parameters.fake_img_test.forward(
                            clear_buffer=True)
                        fake_img.forward(clear_buffer=True)
                    save_generations(self.parameters.fake_img_test, os.path.join(
                        self.results_dir, f'fake_ema_{epoch}'))
                    save_generations(real_img, os.path.join(
                        self.results_dir, f'real_{epoch}'))
                    save_generations(fake_img, os.path.join(
                        self.results_dir, f'fake_{epoch}'))

            epoch_gen_loss /= iterations_per_epoch
            epoch_disc_loss /= iterations_per_epoch

            if self.comm is not None:
                if self.comm.rank == 0:
                    self.monitor_train_gen_loss.add(epoch, epoch_gen_loss)
                    self.monitor_train_gen_loss.add(epoch, epoch_disc_loss)
