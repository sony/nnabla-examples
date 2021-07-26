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

import nnabla as nn
import nnabla.functions as F
import numpy as np


def stft(x, window_size, stride, fft_size,
         window_type='hanning', center=True, pad_mode='reflect'):
    if window_type == 'hanning':
        window_func = np.hanning(window_size + 1)[:-1]
    elif window_type == 'hamming':
        window_func = np.hamming(window_size + 1)[:-1]
    elif window_type == 'rectangular' or window_type is None:
        window_func = np.ones(window_size)
    else:
        raise ValueError("Unknown window type {}.".format(window_type))

    # pad window if `fft_size > window_size`
    if fft_size > window_size:
        diff = fft_size - window_size
        window_func = np.pad(
            window_func, (diff // 2, diff - diff // 2), mode='constant')
    elif fft_size < window_size:
        raise ValueError(
            "FFT size has to be as least as large as window size.")

    # compute STFT filter coefficients
    mat_r = np.zeros((fft_size // 2 + 1, 1, fft_size))
    mat_i = np.zeros((fft_size // 2 + 1, 1, fft_size))

    for w in range(fft_size // 2 + 1):
        for t in range(fft_size):
            mat_r[w, 0, t] = np.cos(2. * np.pi * w * t / fft_size)
            mat_i[w, 0, t] = -np.sin(2. * np.pi * w * t / fft_size)

    conv_r = nn.Variable.from_numpy_array(mat_r * window_func)
    conv_i = nn.Variable.from_numpy_array(mat_i * window_func)

    if center:
        # pad at begin/end (per default this is a reflection padding)
        p = (fft_size - stride) // 2
        x = F.pad(x, (p, p), mode=pad_mode)

    # compute STFT

    y_r = F.convolution(x, conv_r, stride=(stride,))
    y_i = F.convolution(x, conv_i, stride=(stride,))

    return y_r, y_i


def compute_mel(wave, basis, hp):
    r"""Compute the mel-spectrogram from the waveform.

    Args:
        wave (nn.Variable): Wavefrom variable of shape (B, 1, L).
        basis (nn.Variable): Basis for mel-spectrogram computation.
        hp (HParams): Hyper-parameters.

    Returns:
        nn.Variable: Output variable.
    """
    reals, imags = stft(wave, window_size=hp.win_length,
                        stride=hp.hop_length, fft_size=hp.n_fft)
    linear = (reals**2 + imags**2)**0.5
    mels = F.batch_matmul(basis, linear)
    mels = F.log(F.clip_by_value(mels, 1e-5, np.inf))

    return mels
