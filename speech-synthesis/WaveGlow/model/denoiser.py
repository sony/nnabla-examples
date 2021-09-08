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

import librosa as lr
import nnabla.functions as F
import numpy as np


class Denoiser(object):
    r"""Removes model bias from audio produced with waveglow

    Args:
        waveglow (model.module.Module): WaveGlow model.
        hp (HParams): Hyper-parameters.
    """

    def __init__(self, waveglow, hp):
        mel_input = F.constant(shape=[1, hp.n_mels, 88])
        wave = waveglow.infer(mel_input, sigma=0)
        real, imag = F.stft(wave, window_size=hp.win_length,
                            stride=hp.hop_length, fft_size=hp.n_fft)
        bias_spec = F.pow_scalar(real**2 + imag**2, 0.5)
        bias_spec.forward(clear_buffer=True)

        self.bias_spec = bias_spec.d.copy()[:, :, 0][0, :, None]
        self.hparams = hp

    def __call__(self, audio, strength=0.01):
        r"""Generates a clean audio.

        Args:
            audio (numpy.ndarray): Input audio.
            strength (float, optional): Strength level of noise. Defaults to 0.01.

        Returns:
            numpy.ndarray: Clean audio.
        """
        hp = self.hparams
        linear = lr.stft(audio, n_fft=hp.n_fft,
                         hop_length=hp.hop_length, win_length=hp.win_length)
        spect = np.abs(linear) - self.bias_spec * strength
        spect = np.clip(spect, 0, None)
        matrix = spect * np.exp(np.angle(linear) * 1j)
        wave = lr.istft(matrix, hop_length=hp.hop_length,
                        win_length=hp.win_length)
        wave, _ = lr.effects.trim(wave)
        return wave
