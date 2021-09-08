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

from pathlib import Path
import sys
sys.path.append(str(Path().cwd().parents[1] / 'utils'))

import librosa as lr
from nnabla.utils.data_source import DataSource
import numpy as np
from tqdm import tqdm


class LJSpeechDataSource(DataSource):
    r"""Data source for LJ Speech dataset.

    Args:
        metadata (str): The metadata containing text and audio indices.
        hparams (HParams): A container containing all hyper parameters.
        shuffle (bool, optional): If `shuffle==True`, data will be loaded in
            a shuffled order. Defaults to False.
        rng (:obj:`numpy.random.RandomState`, optional): Numpy random number
            generator. Defaults to None.
    """

    def __init__(self, metadata, hparams, shuffle=False, rng=None):
        if rng is None:
            rng = np.random.RandomState(hparams.seed)
        super().__init__(shuffle=shuffle, rng=rng)

        # read text and wave files
        texts, waves = list(), list()
        path = Path(hparams.data_dir)
        with open(path / metadata, encoding='utf-8') as f:
            for line in f:
                inputs = line.strip().split('|')
                waves.append(path / 'wavs' / f'{inputs[0]}.wav')
                texts.append(inputs[2])
        # split data
        n = len(waves)
        index = self._rng.permutation(n) if shuffle else np.arange(n)
        if hasattr(hparams, 'comm'):  # distributed learning
            num = n // hparams.comm.n_procs
            index = index[num*hparams.comm.rank:num*(hparams.comm.rank + 1)]

        self._waves = [waves[i] for i in index]
        self._texts = [texts[i] for i in index]
        self._path = Path(hparams.save_data_dir)
        self._size = len(self._waves)
        self._variables = hparams.out_variables
        self.hparams = hparams

        self.reset()

    def reset(self):
        if self._shuffle:
            self._indexes = self._rng.permutation(self._size)
        else:
            self._indexes = np.arange(self._size)
        super().reset()

    def _get_data(self, position):
        r"""Return a tuple of data."""
        hp = self.hparams
        index = self._indexes[position]
        basename = self._waves[index].with_suffix(".npy").name
        w = np.load(self._path / 'audio' / basename)
        if len(w) > hp.segment_length:
            idx = self._rng.randint(0, len(w) - hp.segment_length)
            w = w[idx:idx + hp.segment_length]
        else:
            w = np.pad(w, (0, hp.segment_length - len(w)), mode='constant')
        return [w]

    def _preprocess(self):
        r"""Precomputes waveform of training data."""
        hp = self.hparams
        for f in self._variables:
            self._path.joinpath(f).mkdir(parents=True, exist_ok=True)

        for i in tqdm(range(self._size)):
            file = self._waves[i]
            w, _ = lr.load(file, sr=hp.sr)
            w, _ = lr.effects.trim(w)  # triming
            basename = self._waves[i].with_suffix(".npy").name
            np.save(self._path / 'audio' / basename, w)


def data_split(path):
    r"""Split the LJ dataset into train and validation sets."""
    with open(Path(path) / 'metadata.csv', encoding='utf-8') as f:
        lines = [line.strip() for line in f]
        n = len(lines)
        index = np.random.RandomState(313).permutation(n)
        split = dict(train=[lines[i] for i in index[:-int(0.1 * n)]],
                     valid=[lines[i] for i in index[-int(0.1 * n):]])
        for key in ['train', 'valid']:
            with open(Path(path) / f'metadata_{key}.csv', 'w') as writer:
                writer.write('\n'.join(split[key]))


if __name__ == '__main__':
    from hparams import hparams as hp
    rng = np.random.RandomState(hp.seed)
    data_split(hp.data_dir)
    LJSpeechDataSource('metadata.csv', hp, False, rng)._preprocess()
