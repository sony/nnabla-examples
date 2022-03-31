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

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import nnabla as nn
import nnabla.functions as F
from tqdm import tqdm

from neu.tts.logger import ProgressMeter
from neu.variable_utils import set_persistent_all


def mae(x, target, len_target=None):
    """Mean absolute error."""
    if len_target is None:
        return F.mean(F.absolute_error(x, target))
    out = F.sum(F.absolute_error(x, target), axis=1) / len_target
    return F.mean(out)


def mse(x, target, len_target=None):
    """Mean squared error."""
    if len_target is None:
        return F.mean(F.squared_error(x, target))
    out = F.sum(F.squared_error(x, target), axis=1) / len_target
    return F.mean(out)


class Trainer:
    r"""Trainer is a basic class for training a model."""

    def __init__(self, model, optim, dataloader, rng, hp):
        self.model = model
        self.optim = optim

        self.dataloader = dataloader
        self.rng = rng
        self.hp = hp
        self.one_epoch_train = max(
            1, dataloader['train'].size // hp.batch_size)
        self.one_epoch_valid = max(
            1, dataloader['valid'].size // hp.batch_size)

        # create saved directory
        path = Path(hp.output_path) / 'artifacts'
        path.joinpath('states').mkdir(parents=True, exist_ok=True)

        self.placeholder = dict()
        self.monitor = ProgressMeter(
            self.one_epoch_train, hp.output_path,
            quiet=hp.comm.rank > 0
        )
        hp.save(os.path.join(hp.output_path, 'settings.json'))

    def update_graph(self, key='train'):
        r"""Builds the graph and update the placeholder.

        Args:
            training (bool, optional): Type of the graph. Defaults to `train`.
        """
        assert key in ('train', 'valid')

        self.model.training = key == 'train'
        hp = self.hp

        # define input variables
        inp_pho = nn.Variable((hp.batch_size, hp.max_len_phone))
        inp_pit = nn.Variable((hp.batch_size, hp.max_len_phone))
        inp_ene = nn.Variable((hp.batch_size, hp.max_len_phone))
        inp_dur = nn.Variable((hp.batch_size, hp.max_len_phone))
        inp_mel = nn.Variable((hp.batch_size, hp.max_len_mel, hp.n_mels))
        len_mel = F.sum(inp_dur, axis=1, keepdims=True)
        len_pho = nn.Variable((hp.batch_size, 1))

        mel, mel_pos, out_dur, out_pit, out_ene = self.model(
            inp_pho, len_phone=len_pho, target_pitch=inp_pit,
            target_energy=inp_ene, target_duration=inp_dur,
        )

        # define losses
        loss_mel = mae(mel, inp_mel, len_mel)
        loss_mel_pos = mae(mel_pos, inp_mel, len_mel)
        loss_dur = mse(out_dur, F.log(inp_dur + 1), len_pho.reshape((-1,)))
        loss_pit = mse(out_pit, inp_pit, len_pho.reshape((-1,)))
        loss_ene = mse(out_ene, inp_ene, len_pho.reshape((-1,)))
        loss = loss_mel + loss_dur + loss_pit + loss_ene + loss_mel_pos

        set_persistent_all(mel, mel_pos, loss_mel, loss_mel_pos,
                           loss_dur, loss_pit, loss_ene, loss)

        self.placeholder[key] = dict(
            inp_pho=inp_pho, len_pho=len_pho, inp_mel=inp_mel,
            inp_pit=inp_pit, inp_ene=inp_ene,
            inp_dur=inp_dur, mel=mel, mel_pos=mel_pos,
            loss_mel=loss_mel, loss_dur=loss_dur,
            loss_pit=loss_pit, loss_ene=loss_ene, loss=loss,
            loss_mel_pos=loss_mel_pos,
        )

    def callback_on_start(self):
        self.cur_epoch = 0
        checkpoint = Path(self.hp.output_path) / 'checkpoint.json'
        if checkpoint.is_file():
            self.load_checkpoint_model(str(checkpoint))
        self.update_graph('train')
        params = self.model.get_parameters(grad_only=True)
        self.optim.set_parameters(params)
        self.update_graph('valid')
        if checkpoint.is_file():
            self.load_checkpoint_optim(str(checkpoint))

        self._grads = [x.grad for x in params.values()]
        self.log_variables = [
            'loss_mel', 'loss_mel_pos', 'loss_dur',
            'loss_pit', 'loss_ene',
        ]

    def run(self):
        r"""Run the training process."""
        hp = self.hp
        self.callback_on_start()

        for cur_epoch in range(self.cur_epoch + 1, self.hp.epoch + 1):
            self.monitor.reset()
            lr = self.optim.get_learning_rate()
            self.monitor.info(f'Running epoch={cur_epoch}\tlr={lr:.5f}\n')
            self.cur_epoch = cur_epoch

            for i in range(self.one_epoch_train):
                self.train_on_batch(i)
                if i % (self.hp.print_frequency) == 0:
                    key = ['train/' + k for k in self.log_variables]
                    self.monitor.display(i, key)

            for i in tqdm(range(self.one_epoch_valid), disable=hp.comm.rank):
                self.valid_on_batch(i)

            self.callback_on_epoch_end()

        self.callback_on_finish()
        self.monitor.close()

    def getdata(self, key='train'):
        pl = self.placeholder[key]
        (inp_pho, len_pho, mel, _,
         pitch, _, energy, _, duration, _) = self.dataloader[key].next()
        pl['inp_pho'].d = inp_pho
        pl['len_pho'].d = len_pho.reshape(pl['len_pho'].shape)
        pl['inp_pit'].d = pitch
        pl['inp_ene'].d = energy
        pl['inp_dur'].d = duration
        pl['inp_mel'].d = mel

    def train_on_batch(self, i):
        r"""Train on one mini-batch."""
        hp, p = self.hp, self.placeholder['train']
        self.getdata('train')
        self.optim.zero_grad()
        p['loss'].forward()
        p['loss'].backward(clear_buffer=True)
        for key in self.log_variables:
            self.monitor.update('train/' + key, p[key].d.copy(), hp.batch_size)
        hp.comm.all_reduce(self._grads, division=True, inplace=False)
        self.optim.update()

    def valid_on_batch(self, i):
        r"""Validate on one mini-batch."""
        hp, p = self.hp, self.placeholder['valid']
        self.getdata('valid')
        p['loss'].forward(clear_buffer=True)
        for key in self.log_variables:
            self.monitor.update('valid/' + key, p[key].d.copy(), hp.batch_size)

    def callback_on_epoch_end(self):
        hp = self.hp
        path = Path(hp.output_path) / 'artifacts'
        if (hp.comm.rank == 0):
            # save all training states
            self.save_checkpoint(path / 'states')
            # save all model parameters
            if self.cur_epoch % hp.epochs_per_checkpoint == 0:
                path = path / f"epoch_{self.cur_epoch}"
                path.mkdir(parents=True, exist_ok=True)
                self.model.save_parameters(str(path / 'model.h5'))
                self.write_output('train', path)
                self.write_output('valid', path)

    def callback_on_finish(self):
        if self.hp.comm.rank == 0:
            path = Path(self.hp.output_path)
            self.model.save_parameters(str(path / 'model.h5'))

    def save_checkpoint(self, path):
        r"""Save the current states of the trainer."""
        if self.hp.comm.rank == 0:
            path = Path(path)
            self.model.save_parameters(str(path / 'model.h5'))
            self.optim._solver.save_states(str(path / 'optim.h5'))
            with open(Path(self.hp.output_path) / 'checkpoint.json', 'w') as f:
                json.dump(dict(cur_epoch=self.cur_epoch,
                               params_path=str(path),
                               optim_n_iters=self.optim._iter), f)
            self.monitor.info(f"Checkpoint saved: {str(path)}\n")

    def load_checkpoint_model(self, checkpoint):
        r"""Load the last states of the trainer."""
        with open(checkpoint, 'r') as file:
            info = json.load(file)
            path = Path(info['params_path'])
        self.model.load_parameters(
            str(path / 'model.h5'), raise_if_missing=True)

    def load_checkpoint_optim(self, checkpoint):
        r"""Load the last states of the trainer."""
        with open(checkpoint, 'r') as file:
            info = json.load(file)
            path = Path(info['params_path'])
            self.optim._iter = info['optim_n_iters']
            self.cur_epoch = info['cur_epoch']
        self.optim._solver.load_states(str(path / 'optim.h5'))

    def write_output(self, key, path):
        r"""write a few samples to tensorboard."""
        p = self.placeholder[key]
        p['loss'].forward(clear_buffer=True)

        # valiate data
        batch_mel = p['mel_pos'].d.copy()
        _, axs = plt.subplots(1, 4, figsize=(18, 3.5))
        _min, _max = batch_mel.min(), batch_mel.max()

        for _, (ax, m) in enumerate(zip(axs.flat, batch_mel)):
            ax.set(xlabel='Frame', ylabel='Frequency')
            ax.imshow(m.T, aspect='auto', origin='lower',
                      vmin=_min, vmax=_max)

        plt.savefig(path/(f'{key}.png'), bbox_inches='tight')
        plt.close()
