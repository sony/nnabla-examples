# Copyright 2020,2021 Sony Corporation.
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

from abc import ABC
from pathlib import Path

import nnabla as nn
import nnabla.functions as F
import numpy as np
from scipy.io import wavfile
from tqdm import trange

from neu.tts.trainer import Trainer


class WaveGlowTrainer(Trainer):
    r"""Trainer for WaveGlow."""

    def update_graph(self, key='train'):
        r"""Builds the graph and update the placeholder.

        Args:
            key (str, optional): Type of computational graph. Defaults to 'train'.
        """
        assert key in ('train', 'valid')

        self.model.training = key != 'valid'
        hp = self.hparams

        # define input variables
        x_aud = nn.Variable([hp.batch_size, hp.segment_length])

        # output variables
        o_aud, log_s, log_det = self.model(x_aud)
        o_aud = o_aud.apply(persistent=True)
        l_det = sum([F.sum(x) for x in log_det])
        l_log = sum([F.sum(x) for x in log_s])

        l_net = (F.sum(o_aud*o_aud)/(2*hp.sigma**2) -
                 l_det - l_log) / np.prod(o_aud.shape)
        l_net = l_net.apply(persistent=True)

        x_mel = self.model.compute_mel(x_aud)
        s_aud = self.model.infer(x_mel)

        self.placeholder[key] = {'x_aud': x_aud,
                                 'o_aud': o_aud, 'l_net': l_net, 's_aud': s_aud}
        self.out_variables = ['train/l_net']

    def train_on_batch(self):
        r"""Updates the model parameters."""
        batch_size = self.hparams.batch_size
        p, dl = self.placeholder['train'], self.dataloader['train']
        self.optimizer.zero_grad()
        if self.hparams.comm.n_procs > 1:
            self.hparams.event.default_stream_synchronize()
        p['x_aud'].d = dl.next()[0]
        p['l_net'].forward(clear_no_need_grad=True)
        p['l_net'].backward(clear_buffer=True)
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
        p['x_aud'].d = dl.next()[0]
        p['l_net'].forward(clear_buffer=True)
        self.loss.data += p['l_net'].d.copy() * batch_size
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
                path = Path(hp.output_path) / 'output' / \
                            f'epoch_{self.cur_epoch}'
                path.mkdir(parents=True, exist_ok=True)
                p['s_aud'].forward(clear_buffer=True)
                wavfile.write(path / 'sample.wav', rate=hp.sr,
                              data=p['s_aud'].d[0].copy())
                self.model.save_parameters(
                    str(path / f'model_{self.cur_epoch}.h5'))

        self.loss.zero()
