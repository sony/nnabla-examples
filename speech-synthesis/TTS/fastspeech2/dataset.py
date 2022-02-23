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

from pathlib import Path

import numpy as np
from nnabla.utils.data_source import DataSource

from utils import text
from utils.text.wdict import ___VALID_SYMBOLS___

pad_value = ___VALID_SYMBOLS___.index('EOL')


class LJSpeechDataSource(DataSource):
    r""" Data source for the LJSpeech dataset."""

    def __init__(self, metadata, hp, shuffle=False, rng=None):
        if rng is None:
            rng = np.random.RandomState(hp.seed)
        super().__init__(shuffle=shuffle, rng=rng)

        with open(Path(hp.precomputed_path) / metadata) as f:
            files = [line.split('|')[0] for line in f.readlines()]

        n = len(files)
        index = self._rng.permutation(n) if shuffle else np.arange(n)
        if hasattr(hp, 'comm'):  # distributed learning
            num = n // hp.comm.n_procs
            index = index[num * hp.comm.rank:num * (hp.comm.rank + 1)]

        self._files = [files[i] for i in index]
        self._path = Path(hp.precomputed_path) / "data"
        self._size = len(self._files)
        self._variables = ["phone", "len_phone",
                           "mel", "len_mel",
                           "pitch", "len_pitch",
                           "energy", "len_energy",
                           "duration", "len_duration"]
        self.hp = hp
        self.stat = np.load(Path(hp.precomputed_path) / 'statistics.npz')

        self.reset()

    def reset(self):
        if self._shuffle:
            self._indexes = self._rng.permutation(self._size)
        else:
            self._indexes = np.arange(self._size)
        super().reset()

    def _normalize(self, x, target):
        return (x - self.stat[f'mean_{target}']) / self.stat[f'std_{target}']

    def _get_data(self, position):
        r"""Return a tuple of data."""
        index = self._indexes[position]
        name = self._files[index]
        hp = self.hp
        data = np.load(self._path / (name + '.npz'))

        # load phoneme
        phone = text.phonemes_to_ids(data["phone"])
        len_phone = len(phone)
        assert len_phone <= hp.max_len_phone
        phone = np.pad(phone, [0, hp.max_len_phone - len_phone],
                       constant_values=pad_value)

        # load mel-spectrogram
        mel = data['mel']
        len_mel = mel.shape[1]
        assert len_mel <= hp.max_len_mel
        if hp.vocoder == 'MelGAN':
            mel = mel / np.log(10)  # change to log10
        mel = np.pad(
            mel, [[0, 0], [0, hp.max_len_mel - len_mel]],
            constant_values=hp.lower_bound
        )
        mel = np.transpose(mel, (1, 0))

        # load pitch
        pitch = data['pitch']
        len_pitch = len(pitch)
        assert len_pitch <= hp.max_len_phone
        pitch = self._normalize(pitch, 'pitch')
        pitch = np.pad(pitch, [0, hp.max_len_phone - len_pitch])

        # load energy
        energy = data['energy']
        len_energy = len(energy)
        assert len_energy <= hp.max_len_phone
        energy = self._normalize(energy, 'energy')
        energy = np.pad(energy, [0, hp.max_len_phone - len_energy])

        # load duration
        duration = data['duration']
        len_duration = len(duration)
        assert len_duration <= hp.max_len_phone
        duration = np.pad(duration, [0, hp.max_len_phone - len_duration])

        return (phone, len_phone,
                mel, len_mel,
                pitch, len_pitch,
                energy, len_energy,
                duration, len_duration)
