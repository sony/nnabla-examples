# Copyright (c) 2017 Sony Corporation. All Rights Reserved.
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

import json
from pathlib import Path

import nnabla as nn
import nnabla.functions as F
import soundfile as sf
from librosa.filters import mel as librosa_mel_fn
from neu.tts.logger import ProgressMeter
from neu.variable_utils import set_persistent_all

from utils import compute_mel


def discriminator_loss(fea, scale):
    loss = sum([F.mean((out[-1] - scale)**2) for out in fea])
    return loss


def feature_loss(fea_real, fea_fake):
    loss = list()
    for o1, o2 in zip(fea_real, fea_fake):
        for f1, f2 in zip(o1[:-1], o2[:-1]):
            loss.append(F.mean(F.absolute_error(f1, f2)))
    return sum(loss)


class HiFiGANTrainer:
    r"""Trainer for HiFi-GAN."""

    def __init__(self, gen, dis, gen_optim, dis_optim, dataloader, hp):
        self.gen = gen
        self.dis = dis
        self.gen_optim = gen_optim
        self.dis_optim = dis_optim
        self.dataloader = dataloader
        self.hp = hp

        # compute mel basis
        mel_basis = librosa_mel_fn(hp.sr, hp.n_fft, n_mels=hp.n_mels,
                                   fmin=hp.mel_fmin, fmax=hp.mel_fmax)
        self.mel_basis = nn.Variable.from_numpy_array(mel_basis[None, ...])

        self.one_epoch_train = dataloader['train'].size // hp.batch_size
        self.one_epoch_valid = dataloader['valid'].size // hp.batch_size
        self.placeholder = dict()

        self.monitor = ProgressMeter(
            self.one_epoch_train, hp.output_path, quiet=hp.comm.rank > 0)
        hp.save(Path(hp.output_path) / 'settings.json')

    def callback_on_start(self):
        r"""Calls this on starting the training."""
        self.cur_epoch = 0
        checkpoint = Path(self.hp.output_path) / 'checkpoint.json'
        if checkpoint.is_file():
            self.load_checkpoint_model(str(checkpoint))

        self.update_graph('train')
        params = self.gen.get_parameters(grad_only=True)
        self.gen_optim.set_parameters(params)

        dis_params = self.dis.get_parameters(grad_only=True)
        self.dis_optim.set_parameters(dis_params)

        if checkpoint.is_file():
            self.load_checkpoint_optim(str(checkpoint))

        self._grads = [x.grad for x in params.values()]
        self._discs = [x.grad for x in dis_params.values()]

        self.update_graph('valid')

        self.log_variables = [
            'g_loss_avd', 'g_loss_mel',
            'g_loss', 'd_loss', 'g_loss_fea'
        ]

    def run(self):
        r"""Run the training process."""
        self.callback_on_start()

        for cur_epoch in range(self.cur_epoch + 1, self.hp.epoch + 1):
            self.monitor.reset()
            lr = self.gen_optim.get_learning_rate()
            self.monitor.info(f'Running epoch={cur_epoch}\tlr={lr:.5f}\n')
            self.cur_epoch = cur_epoch

            for i in range(self.one_epoch_train):
                self.train_on_batch(i)
                if i % (self.hp.print_frequency) == 0:
                    self.monitor.display(i, self.log_variables)

            for i in range(self.one_epoch_valid):
                self.valid_on_batch(i)

            self.callback_on_epoch_end()

        self.callback_on_finish()
        self.monitor.close()

    def _zero_grads(self):
        self.gen_optim.zero_grad()
        self.dis_optim.zero_grad()

    def update_graph(self, key='train'):
        r"""Builds the graph and update the placeholder.
        Args:
            key (str, optional): Type of computational graph.
                Defaults to 'train'.
        """
        assert key in ('train', 'valid')
        hp = self.hp

        self.gen.training = key == 'train'
        self.dis.training = key == 'train'

        # define input variables
        x_real = nn.Variable((hp.batch_size, 1, hp.segment_length))
        x_real_mel = compute_mel(x_real, self.mel_basis, hp)

        x_fake = self.gen(x_real_mel)
        x_fake_mel = compute_mel(x_fake, self.mel_basis, hp)

        dis_real_x = self.dis(x_real)
        dis_fake_x = self.dis(x_fake)

        # ------------------------------ Discriminator -----------------------
        d_loss = (
            discriminator_loss(dis_real_x, 1.0)
            + discriminator_loss(dis_fake_x, 0.0)
        )
        # -------------------------------- Generator -------------------------
        g_loss_avd = discriminator_loss(dis_fake_x, 1.0)
        g_loss_mel = F.mean(F.absolute_error(x_real_mel, x_fake_mel))
        g_loss_fea = feature_loss(dis_real_x, dis_fake_x)
        g_loss = g_loss_avd + 45*g_loss_mel + 2*g_loss_fea

        set_persistent_all(
            g_loss_mel, g_loss_avd, g_loss_fea,
            d_loss, x_fake, g_loss,
        )

        self.placeholder[key] = dict(
            x_real=x_real, x_fake=x_fake, d_loss=d_loss,
            g_loss_avd=g_loss_avd, g_loss_mel=g_loss_mel,
            g_loss_fea=g_loss_fea, g_loss=g_loss,
        )

    def train_on_batch(self, i):
        r"""Updates the model parameters."""
        hp = self.hp
        bs, p = hp.batch_size, self.placeholder['train']
        dl = self.dataloader['train']
        p['x_real'].d = dl.next()[0]

        # ----------------------------- train discriminator ----------------
        self._zero_grads()
        p['x_fake'].need_grad = False
        p['d_loss'].forward()
        p['d_loss'].backward(clear_buffer=True)
        self.monitor.update('d_loss', p['d_loss'].d.copy(), bs)
        hp.comm.all_reduce(self._discs, division=True, inplace=False)
        self.dis_optim.update()
        p['x_fake'].need_grad = True

        # ------------------------------ train generator -------------------
        self._zero_grads()
        p['g_loss'].forward()
        p['g_loss'].backward(clear_buffer=True)
        self.monitor.update('g_loss', p['g_loss'].d.copy(), bs)
        self.monitor.update('g_loss_avd', p['g_loss_avd'].d.copy(), bs)
        self.monitor.update('g_loss_mel', p['g_loss_mel'].d.copy(), bs)
        self.monitor.update('g_loss_fea', p['g_loss_fea'].d.copy(), bs)

        hp.comm.all_reduce(self._grads, division=True, inplace=False)
        self.gen_optim.update()

    def valid_on_batch(self, i):
        r"""Performs validation."""
        p, dl = self.placeholder['valid'], self.dataloader['valid']
        p['x_real'].d = dl.next()[0]
        p['x_fake'].forward(clear_buffer=True)

    def callback_on_epoch_end(self):
        hp = self.hp
        if (hp.comm.rank == 0):
            path = Path(hp.output_path) / 'artifacts'
            path.joinpath('states').mkdir(parents=True, exist_ok=True)
            path.joinpath('samples').mkdir(parents=True, exist_ok=True)
            self.save_checkpoint(path / 'states')
            self.write_samples(path / 'samples')
            if self.cur_epoch % hp.epochs_per_checkpoint == 0:
                path = path / f"epoch_{self.cur_epoch}"
                path.mkdir(parents=True, exist_ok=True)
                self.gen.save_parameters(str(path / 'model.h5'))
                self.dis.save_parameters(str(path / 'cls.h5'))

    def callback_on_finish(self):
        if self.hp.comm.rank == 0:
            path = Path(self.hp.output_path)
            self.gen.save_parameters(str(path / 'model.h5'))
            self.dis.save_parameters(str(path / 'cls.h5'))

    def save_checkpoint(self, path):
        r"""Save the current states of the trainer."""
        if self.hp.comm.rank == 0:
            path = Path(path)
            self.gen.save_parameters(str(path / 'model.h5'))
            self.dis.save_parameters(str(path / 'cls.h5'))
            self.gen_optim._solver.save_states(str(path / 'gen_optim.h5'))
            self.dis_optim._solver.save_states(str(path / 'dis_optim.h5'))
            with open(Path(self.hp.output_path) / 'checkpoint.json', 'w') as f:
                json.dump(
                    dict(cur_epoch=self.cur_epoch,
                         params_path=str(path),
                         gen_optim_n_iters=self.gen_optim._iter,
                         dis_optim_n_iters=self.dis_optim._iter,),
                    f
                )
            self.monitor.info(f"Checkpoint saved: {str(path)}\n")

    def load_checkpoint_model(self, checkpoint):
        r"""Load the last states of the trainer."""
        with open(checkpoint, 'r') as file:
            info = json.load(file)
            path = Path(info['params_path'])
        self.gen.load_parameters(str(path / 'model.h5'), raise_if_missing=True)
        self.dis.load_parameters(str(path / 'cls.h5'), raise_if_missing=True)

    def load_checkpoint_optim(self, checkpoint):
        r"""Load the last states of the trainer."""
        with open(checkpoint, 'r') as file:
            info = json.load(file)
            path = Path(info['params_path'])
            self.gen_optim._iter = info['gen_optim_n_iters']
            self.dis_optim._iter = info['dis_optim_n_iters']
            self.cur_epoch = info['cur_epoch']
        self.gen_optim._solver.load_states(str(path / 'gen_optim.h5'))
        self.dis_optim._solver.load_states(str(path / 'dis_optim.h5'))

    def write_samples(self, path):
        r"""write a few samples."""
        hp = self.hp
        p = self.placeholder['valid']
        p['x_fake'].forward(clear_buffer=True)
        X, Z = p['x_real'].d.copy(), p['x_fake'].d.copy()
        for i, (x, z) in enumerate(zip(X, Z)):
            sf.write(str(path / f'real_{i}.wav'), x[0], hp.sr)
            sf.write(str(path / f'output_{i}.wav'), z[0], hp.sr)
