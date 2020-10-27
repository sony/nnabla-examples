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

from abc import ABC
from pathlib import Path

import matplotlib.pyplot as plt
import nnabla as nn
import nnabla.functions as F
import numpy as np
from scipy.io import wavfile
from tqdm import trange

from utils.audio import synthesize_from_spec
from utils.logger import ProgressMeter


def save_image(data, path, label, title, figsize=(6, 5)):
    r"""Saves an image to file."""
    plt.figure(figsize=figsize)
    plt.imshow(data.copy(), origin='lower', aspect='auto')
    plt.xlabel(label[0])
    plt.ylabel(label[1])
    plt.title(title)
    plt.colorbar()
    plt.savefig(path, bbox_inches='tight')
    plt.close()


class TacotronTrainer(ABC):
    r"""Trainer for Tacotron.

    Args:
        model (model.module.Module): Tacotron model.
        dataloader (dict): A dataloader.
        optimizer (Optimizer): An optimizer used to update the parameters.
        hparams (HParams): Hyper-parameters.
    """

    def __init__(self, model, dataloader, optimizer, hparams):
        self.model = model
        self.dataloader = dataloader
        self.hparams = hparams
        self.one_epoch_train = dataloader['train'].size // hparams.batch_size
        self.one_epoch_valid = dataloader['valid'].size // hparams.batch_size
        self.placeholder = dict()
        self.optimizer = optimizer
        self.monitor = ProgressMeter(
            self.one_epoch_train, hparams.output_path, quiet=hparams.comm.rank > 0)
        hparams.save(Path(hparams.output_path) / 'settings.json')

    def update_graph(self, key='train'):
        r"""Builds the graph and update the placeholder.

        Args:
            key (str, optional): Type of computational graph. Defaults to 'train'.
        """
        assert key in ('train', 'valid')

        self.model.training = key != 'valid'
        hp = self.hparams

        # define input variables
        x_txt = nn.Variable([hp.batch_size, hp.text_len])
        x_mel = nn.Variable([hp.batch_size, hp.n_frames, hp.n_mels*hp.r])
        t_mag = nn.Variable([hp.batch_size, hp.n_frames*hp.r, hp.n_fft//2+1])

        # output variables
        o_mel, o_mag, o_att = self.model(x_txt, x_mel)
        o_mel = o_mel.apply(persistent=True)
        o_mag = o_mag.apply(persistent=True)
        o_att = o_att.apply(persistent=True)

        # loss functions
        def criteria(x, t):
            return F.mean(F.absolute_error(x, t))

        n_prior = int(3000 / (hp.sr * 0.5) * (hp.n_fft//2 + 1))

        l_mel = criteria(o_mel, x_mel).apply(persistent=True)
        l_mag = 0.5*criteria(o_mag, t_mag) + 0.5 * \
            criteria(o_mag[..., :n_prior], t_mag[..., :n_prior])
        l_mag.persistent = True

        l_net = (l_mel + l_mag).apply(persistent=True)

        self.placeholder[key] = {
            'x_mel': x_mel, 't_mag': t_mag, 'x_txt': x_txt,
            'o_mel': o_mel, 'o_mag': o_mag, 'o_att': o_att,
            'l_mel': l_mel, 'l_mag': l_mag, 'l_net': l_net
        }

    def callback_on_start(self):
        self.update_graph('train')
        params = self.model.get_parameters(grad_only=True)
        self.optimizer.set_parameters(params)
        self.update_graph('valid')
        self.loss = nn.NdArray.from_numpy_array(np.zeros((1,)))
        if self.hparams.comm.n_procs > 1:
            self._grads = [x.grad for x in params.values()]

    def run(self):
        r"""Run the training process."""
        self.callback_on_start()
        for cur_epoch in range(self.hparams.epoch):
            self.monitor.reset()
            lr = self.optimizer.get_learning_rate()
            self.monitor.info(f'Running epoch={cur_epoch}\tlr={lr:.5f}\n')
            self.cur_epoch = cur_epoch
            for i in range(self.one_epoch_train):
                self.train_on_batch()
                if i % (self.hparams.print_frequency) == 0:
                    self.monitor.display(
                        i, ['train/l_mel', 'train/l_mag', 'train/l_net'])
            for i in trange(self.one_epoch_valid, disable=self.hparams.comm.rank > 0):
                self.valid_on_batch()
            self.callback_on_epoch_end()
        self.callback_on_finish()
        self.monitor.close()

    def train_on_batch(self):
        r"""Updates the model parameters."""
        batch_size = self.hparams.batch_size
        p, dl = self.placeholder['train'], self.dataloader['train']
        self.optimizer.zero_grad()
        if self.hparams.comm.n_procs > 1:
            self.hparams.event.default_stream_synchronize()
        p['x_mel'].d, p['t_mag'].d, p['x_txt'].d = dl.next()
        p['l_net'].forward(clear_no_need_grad=True)
        p['l_net'].backward(clear_buffer=True)
        self.monitor.update('train/l_mel', p['l_mel'].d.copy(), batch_size)
        self.monitor.update('train/l_mag', p['l_mag'].d.copy(), batch_size)
        self.monitor.update('train/l_net', p['l_net'].d.copy(), batch_size)
        if self.hparams.comm.n_procs > 1:
            self.hparams.comm.all_reduce(
                self._grads, division=True, inplace=False)
            self.hparams.event.add_default_stream_event()
        self.optimizer.update()

    def valid_on_batch(self):
        r"""Performs validation."""
        batch_size = self.hparams.batch_size
        p, dl = self.placeholder['valid'], self.dataloader['valid']
        if self.hparams.comm.n_procs > 1:
            self.hparams.event.default_stream_synchronize()
        p['x_mel'].d, p['t_mag'].d, p['x_txt'].d = dl.next()
        p['l_net'].forward(clear_buffer=True)
        self.loss.data += p['l_net'].d.copy() * batch_size
        self.monitor.update('valid/l_mel', p['l_mel'].d.copy(), batch_size)
        self.monitor.update('valid/l_mag', p['l_mag'].d.copy(), batch_size)
        self.monitor.update('valid/l_net', p['l_net'].d.copy(), batch_size)

    def callback_on_epoch_end(self):
        if self.hparams.comm.n_procs > 1:
            self.hparams.comm.all_reduce(
                [self.loss], division=True, inplace=False)
        self.loss.data /= self.dataloader['valid'].size
        if self.hparams.comm.rank == 0:
            p, hp = self.placeholder['valid'], self.hparams
            self.monitor.info(f'valid/loss={self.loss.data[0]:.5f}\n')
            if self.cur_epoch % hp.epochs_per_checkpoint == 0:
                path = Path(hp.output_path) / 'output' / f'epoch_{self.cur_epoch}'
                path.mkdir(parents=True, exist_ok=True)
                # write attention and spectrogram outputs
                for k in ('o_att', 'o_mel', 'o_mag'):
                    p[k].forward(clear_buffer=True)
                    data = p[k].d[0].copy()
                    save_image(
                        data=data.reshape(
                            (-1, hp.n_mels)).T if k == 'o_mel' else data.T,
                        path=path / (k + '.png'),
                        label=('Decoder timestep', 'Encoder timestep') if k == 'o_att' else (
                            'Frame', 'Channel'),
                        title={
                            'o_att': 'Attention', 'o_mel': 'Mel spectrogram', 'o_mag': 'Spectrogram'}[k],
                        figsize=(6, 5) if k == 'o_att' else (6, 3)
                    )
                wave = synthesize_from_spec(p['o_mag'].d[0].copy(), hp)
                wavfile.write(path / 'sample.wav', rate=hp.sr, data=wave)
                self.model.save_parameters(str(path / f'model_{self.cur_epoch}.h5'))
        self.loss.zero()

    def callback_on_finish(self):
        r"""Calls this on finishing the run method."""
        if self.hparams.comm.rank == 0:
            path = str(Path(self.hparams.output_path) / 'model.h5')
            self.model.save_parameters(path)
