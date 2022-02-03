# Copyright 2021 Sony Group Corporation
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
from tqdm import trange

import nnabla as nn
import nnabla.solvers as S
import nnabla.functions as F
from nnabla.logger import logger

from utils import *
from models import TCVCGenerator, TCVCGenerator_feat
from nnabla.utils.image_utils import imsave

from tensorboardX import SummaryWriter
from nnabla.models.imagenet import VGG16

class Trainer(object):
    def __init__(self, tconf, mconf, comm, dataset_path):
        rng = np.random.RandomState(tconf.random_seed)

        self.train_conf = tconf
        self.model_conf = mconf

        self.bs = tconf.batch_size

        self.no_prev = self.train_conf.no_prev
        self.gen_feat = self.train_conf.gen_feat
        self.image_shape = tuple(x * mconf.g_n_scales for x in mconf.base_image_shape)
        self.data_iter = create_tcvc_iterator(self.bs, dataset_path,
                                                    image_shape=self.image_shape,
                                                    rng=rng, flip=tconf.flip, dataset_mode="train")
        if comm.n_procs > 1:
            self.data_iter = self.data_iter.slice(
                rng, num_of_slices=comm.n_procs, slice_pos=comm.rank)

        self.comm = comm
        self.fix_global_epoch = max(tconf.fix_global_epoch, 0)
        self.use_encoder = False  # currently encoder is not supported.

        self.load_path = tconf.load_path
        self.save_path = self.train_conf.save_path

        self.log_freq = 10
        #self.save_latest_freq = 2500
        self.total_steps = 0
        if self.comm.rank == 0:
            self.tb_writer = SummaryWriter(self.save_path + "/tb_logs")
        self.vgg = VGG16()

    def train(self):
        real = nn.Variable(shape=(self.bs, 3) + self.image_shape)
        real_prev = nn.Variable(shape=(self.bs, 3) + self.image_shape)
        sketch = nn.Variable(shape=(self.bs, 1) + self.image_shape)

        if self.no_prev or self.gen_feat:
            x = sketch
        else:
            x = F.concatenate(sketch, real_prev, axis=1)
            
        # generator
        if self.gen_feat:
            print("using TCVC_feat generator")
            generator = TCVCGenerator_feat()
            prev_feat = vgg16_get_feat(real, self.vgg)
            fake, _ = generator(x, prev_feat,
                                channels=self.model_conf.gg_channels,
                                downsample_input=False, 
                                n_residual_layers=self.model_conf.gg_num_residual_loop)
        else:
            print("using orginal TCVC generator")
            generator = TCVCGenerator()
            fake, _ = generator(x,
                                channels=self.model_conf.gg_channels,
                                downsample_input=False, 
                             n_residual_layers=self.model_conf.gg_num_residual_loop)
        unlinked_fake = fake.get_unlinked_variable(need_grad=True)

        # discriminator
        discriminator = PatchGAN(n_scales=self.model_conf.d_n_scales, use_spectral_normalization=True)
        d_real_out, d_real_feats = discriminator(F.concatenate(x, real, axis=1))
        d_fake_out, d_fake_feats = discriminator(F.concatenate(x, unlinked_fake, axis=1))
        g_gan, _, d_real, d_fake = discriminator.get_loss(d_real_out, d_real_feats,
                                                               d_fake_out, d_fake_feats,
                                                               use_fm=False,
                                                               fm_lambda=self.train_conf.lambda_feat,
                                                               gan_loss_type="ls")

        g_vgg = vgg16_perceptual_loss(unlinked_fake, real, self.vgg) * self.train_conf.lambda_perceptual

        g_style= vgg16_style_loss(unlinked_fake, real, self.vgg) * self.train_conf.lambda_style

        g_l1= F.mean(F.absolute_error(unlinked_fake, real)) * self.train_conf.lambda_l1

        set_persistent_all(fake, fake, g_gan, g_vgg, g_style, g_l1, d_real, d_fake)

        g_loss = g_gan + g_vgg + g_style + g_l1
        d_loss = 0.5 * (d_real + d_fake)

        # load parameters
        if self.load_path:
            if not os.path.exists(self.load_path):
                logger.warn("Path to load params is not found."
                            " Loading params is skipped. ({})".format(self.load_path))
            else:
                nn.load_parameters(self.load_path)

        # Setup Solvers
        g_solver = S.Adam(beta1=0.)
        g_solver.set_parameters(get_params_startswith("generator/tcvc"))

        d_solver = S.Adam(beta1=0.)
        d_solver.set_parameters(get_params_startswith("discriminator"))

        # lr scheduler
        G_lr_schduler = LinearDecayScheduler(self.train_conf.G_base_lr, 0.,
                                           start_iter=self.train_conf.lr_decay_starts,
                                           end_iter=self.train_conf.max_epochs)
        D_lr_schduler = LinearDecayScheduler(self.train_conf.D_base_lr, 0.,
                                           start_iter=self.train_conf.lr_decay_starts,
                                           end_iter=self.train_conf.max_epochs)

        # Setup Reporter
        losses = {"g_gan": g_gan, "g_style": g_style, "g_vgg": g_vgg, "g_l1": g_l1, "d_real": d_real, "d_fake": d_fake}
        reporter = Reporter(self.comm, losses, self.train_conf.save_path)

        
        for epoch in range(self.train_conf.max_epochs):
            if epoch == self.fix_global_epoch:
                g_solver.set_parameters(get_params_startswith(
                    "generator"), reset=False, retain_state=True)

            # update learning rate for current epoch
            G_lr = G_lr_schduler(epoch)
            D_lr = D_lr_schduler(epoch)
            g_solver.set_learning_rate(G_lr)
            d_solver.set_learning_rate(D_lr)

            progress_iterator = trange(self.data_iter._size // self.bs,
                                       desc="[epoch {}]".format(epoch), disable=self.comm.rank > 0)

            reporter.start(progress_iterator)

            for i in progress_iterator:
                image, image_prev, sketch_id = self.data_iter.next()

                real.d = image
                real_prev.d = image_prev
                sketch.d = sketch_id


                # create fake
                fake.forward()

                # update discriminator
                d_solver.zero_grad()
                d_loss.forward(clear_no_need_grad=True)
                d_loss.backward(clear_buffer=True)

                if self.comm.n_procs > 1:
                    params = [
                        x.grad for x in d_solver.get_parameters().values()]
                    self.comm.all_reduce(params, division=False, inplace=False)
                '''
                d_solver.update()
                '''
                if i % 12 == 0 :    
                    d_solver.update()
                
                # update generator
                unlinked_fake.grad.zero()
                g_solver.zero_grad()
                g_loss.forward(clear_no_need_grad=True)
                g_loss.backward(clear_buffer=True)

                # backward generator
                fake.backward(grad=None, clear_buffer=True)


                if self.comm.n_procs > 1:
                    params = [
                        x.grad for x in g_solver.get_parameters().values()]
                    self.comm.all_reduce(params, division=False, inplace=False)
                g_solver.update()

                

                # report iteration progress
                reporter()

                # tensorboard logger
                if (self.total_steps % self.log_freq) == 0 and self.comm.rank == 0:
                    self.tb_writer.add_scalar('d_loss/d_real', d_real.data.get_data("r"), self.total_steps)
                    self.tb_writer.add_scalar('d_loss/d_fake', d_fake.data.get_data("r"), self.total_steps)
                    self.tb_writer.add_scalar('g_loss/g_gan', g_gan.data.get_data("r"), self.total_steps)
                    self.tb_writer.add_scalar('g_loss/g_vgg', g_vgg.data.get_data("r"), self.total_steps)
                    self.tb_writer.add_scalar('g_loss/g_style', g_style.data.get_data("r"), self.total_steps)
                    self.tb_writer.add_scalar('g_loss/g_l1', g_l1.data.get_data("r"), self.total_steps)
                    self.tb_writer.add_image('Images/sketch', array2im(sketch.data.get_data("r")[0]), self.total_steps, dataformats='HWC')
                    self.tb_writer.add_image('Images/real', array2im(real.data.get_data("r")[0]), self.total_steps, dataformats='HWC')
                    self.tb_writer.add_image('Images/real_prev', array2im(real_prev.data.get_data("r")[0]), self.total_steps, dataformats='HWC')
                    self.tb_writer.add_image('Images/fake', array2im(fake.data.get_data("r")[0]), self.total_steps, dataformats='HWC')
                
                '''
                # save latest model every self.save_latest_freq iters
                if (self.total_steps % self.save_latest_freq==0):
                    nn.save_parameters(os.path.join(self.train_conf.save_path, 'param_latest_{:03d}.h5'.format(self.total_steps)))    
                '''
                self.total_steps +=1

            # report epoch progress
            show_images = {"GeneratedImage": fake.data.get_data("r").transpose((0, 2, 3, 1)),
                           "RealImage": real.data.get_data("r").transpose((0, 2, 3, 1)),
                           "Real_Prev": real_prev.data.get_data("r").transpose((0, 2, 3, 1))}
            reporter.step(epoch, show_images)

            if (epoch % 1) == 0 and self.comm.rank == 0:
                nn.save_parameters(os.path.join(
                    self.train_conf.save_path, 'param_{:03d}.h5'.format(epoch)))

        if self.comm.rank == 0:
            nn.save_parameters(os.path.join(
                self.train_conf.save_path, 'param_final.h5'))
