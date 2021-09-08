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

import os
from pathlib import Path
import sys
sys.path.append(str(Path().cwd().parents[2] / 'utils'))

import librosa as lr
from librosa.filters import mel as librosa_mel_fn
from nnabla.utils.data_source import DataSource
import numpy as np
from tqdm import tqdm

from neu.tts import audio
from neu.tts.text import text_normalize


class LJSpeechDataSource(DataSource):
    r""" Data source for LJ Speech dataset.

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
                waves.append(str(path / 'wavs' / f'{inputs[0]}.wav'))
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
        self._char2idx = {ch: i for i, ch in enumerate(hparams.vocab)}
        self._idx2char = {i: ch for i, ch in enumerate(hparams.vocab)}

        # compute the mel basis
        self.mel_basis = librosa_mel_fn(
            hparams.sr, hparams.n_fft, n_mels=hparams.n_mels,
            fmin=hparams.mel_fmin, fmax=hparams.mel_fmax
        )
        self.reset()

    def reset(self):
        if self._shuffle:
            self._indexes = self._rng.permutation(self._size)
        else:
            self._indexes = np.arange(self._size)
        super().reset()

    def _get_data(self, position):
        r"""Return a tuple of data."""
        index = self._indexes[position]
        basename = Path(self._waves[index]).with_suffix(".npy").name
        return tuple(np.load(self._path / x / basename) for x in self._variables)

    def _store_entry(self, index, linear, w):
        hp = self.hparams
        basename = Path(self._waves[index]).with_suffix(".npy").name

        seq_len = hp.n_frames * hp.r
        lin_len = linear.shape[1]
        assert lin_len <= seq_len  # sanitary check

        # compute linear spectrogram
        post_linear = self._padding(linear.T, (seq_len, linear.shape[0]))
        post_linear = audio.normalize(post_linear, hp)
        np.save(self._path / 'linear' / basename, post_linear)

        # compute mel spectrogram
        mel = np.dot(self.mel_basis, linear)  # transform to mel scales
        mel = self._padding(mel.T, (seq_len, hp.n_mels))
        mel = audio.normalize(mel, hp)
        mel = mel.reshape(-1, hp.n_mels*hp.r)
        np.save(self._path / 'mel' / basename, mel)

        # compute text sequence
        text = text_normalize(self._texts[index], self.hparams.vocab)
        assert len(text) < hp.text_len
        text = text.ljust(hp.text_len, '~')
        text = [self._char2idx[ch] for ch in text]
        np.save(self._path / 'text' / basename, text)

    def _get_spectrograms(self, index):
        r"""Return the corresponding spectrogram and waveform."""
        file = self._waves[index]

        # get hyper-parameters
        hp = self.hparams

        w, _ = lr.load(file, sr=hp.sr)
        w, _ = lr.effects.trim(w)  # triming

        if hasattr(hp, "preemphasis"):
            # apply a pre-emphasis filter to amplify the high frequencies
            w = audio.preemphasis(w, factor=hp.preemphasis)
        linear = audio.wave2spec(w, hp)

        return linear, w

    def _preprocess(self):
        r"""Precomputes mel spectrograms, linear spectrograms, and texts."""
        for f in self._variables:
            self._path.joinpath(f).mkdir(parents=True, exist_ok=True)

        for i in tqdm(range(self._size)):
            linear, w = self._get_spectrograms(i)
            self._store_entry(i, linear, w)

    def _padding(self, x, shape, value=0):
        r"""Add padding to have a desired shape.

        Args:
            x (numpy.ndarray): Input array.
            shape (tuple of int): The desired shape.
            value (int, optional): Value to be padded. Defaults to 0.

        Returns:
            numpy.ndarray: An output array with a desired shape.
        """
        row_padding = shape[0] - x.shape[0]
        col_padding = shape[1] - x.shape[1]
        return np.pad(x, [[0, row_padding], [0, col_padding]], mode="constant", constant_values=value)


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
