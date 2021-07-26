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

from pathlib import Path

import numpy as np
from nnabla.utils.data_source import DataSource


class LJSpeechDataSource(DataSource):
    r"""Data source for LJ Speech dataset.
    Args:
        metadata (str): The metadata containing text and audio indices.
        hp (HParams): A container containing all hyper parameters.
        shuffle (bool, optional): If `shuffle==True`, data will be loaded in
            a shuffled order. Defaults to False.
        rng (:obj:`numpy.random.RandomState`, optional): Numpy random number
            generator. Defaults to None.
    """

    def __init__(self, metadata, hp, shuffle=False, rng=None):

        if rng is None:
            rng = np.random.RandomState(hp.seed)
        super().__init__(shuffle=shuffle, rng=rng)

        with open(Path(hp.precomputed_path) / 'meta_train.csv') as f:
            files = [line.split('|')[0] for line in f.readlines()]

        n = len(files)
        index = self._rng.permutation(n) if shuffle else np.arange(n)
        if hasattr(hp, 'comm'):  # distributed learning
            num = n // hp.comm.n_procs
            index = index[num * hp.comm.rank:num * (hp.comm.rank + 1)]

        self._files = [files[i] for i in index]
        self._path = Path(hp.precomputed_path) / "data"
        self._size = len(self._files)
        self._variables = ["wave"]
        self.hp = hp

        self.reset()

    def reset(self):
        if self._shuffle:
            self._indexes = self._rng.permutation(self._size)
        else:
            self._indexes = np.arange(self._size)
        super().reset()

    def _get_data(self, position):
        r"""Return a tuple of data."""
        hp = self.hp
        index = self._indexes[position]
        name = self._files[index]
        w = np.load(self._path / (name + '.npz'))['wave']
        if len(w) > hp.segment_length:
            idx = self._rng.randint(0, len(w) - hp.segment_length)
            w = w[idx:idx + hp.segment_length]
        else:
            w = np.pad(w, (0, hp.segment_length - len(w)), mode='constant')
        return w[None, None, ...]
