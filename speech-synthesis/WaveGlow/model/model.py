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

from librosa.filters import mel as librosa_mel_fn
import nnabla as nn
import nnabla.functions as F
from nnabla.initializer import ConstantInitializer
from nnabla.initializer import NormalInitializer
import nnabla.parametric_functions as PF
import numpy as np

from neu.tts.module import Module

from .ops import fused_add_tanh_sigmoid_multiply
from .ops import invertible_conv


class WN(Module):
    def __init__(self, hparams):
        self.hparams = hparams
        assert hparams.wn_kernel_size % 2 == 1
        assert hparams.wn_n_channels % 2 == 0

    def call(self, audio, spec):
        r"""Return a variable.

        Args:
            audio (nn.Variable): A variable represents audio of shape (B, n_groups, L).
            spec (nn.Variable): A variable represents spectrogram of shape (B, n_mels, L).

        Returns:
            nn.Variable: An output variable.
        """
        hp = self.hparams
        in_channels = audio.shape[1]
        n_channels = hp.wn_n_channels
        n_layers = hp.wn_n_layers
        kernel_size = hp.wn_kernel_size

        with nn.parameter_scope('start'):
            audio = PF.convolution(
                audio, n_channels, kernel=(1,),
                apply_w=PF.weight_normalization,
                w_init=NormalInitializer(0.05)
            )

        for i in range(n_layers):

            with nn.parameter_scope(f'layer_{i}'):
                dilation = 2 ** i
                padding = ((kernel_size - 1)*dilation) // 2

                with nn.parameter_scope('fused'):

                    with nn.parameter_scope('in_layer'):
                        audio_inp = PF.convolution(
                            audio, 2*n_channels, kernel=(kernel_size,),
                            dilation=(dilation,), pad=(padding,),
                            apply_w=PF.weight_normalization,
                            w_init=NormalInitializer(0.05)
                        )

                    with nn.parameter_scope('cond_layer'):
                        spec_inp = PF.convolution(
                            spec, 2*n_channels, kernel=(1,),
                            apply_w=PF.weight_normalization,
                            w_init=NormalInitializer(0.05)
                        )

                    acts = fused_add_tanh_sigmoid_multiply(
                        audio_inp, spec_inp, n_channels)

                with nn.parameter_scope('res_skip'):
                    channels = 2*n_channels if i < n_layers - 1 else n_channels
                    res_skip_acts = PF.convolution(
                        acts, channels, kernel=(1,),
                        apply_w=PF.weight_normalization,
                        w_init=NormalInitializer(0.05)
                    )

                if i < n_layers - 1:
                    audio += res_skip_acts[:, :n_channels, :]
                    skip_acts = res_skip_acts[:, n_channels:, :]
                else:
                    skip_acts = res_skip_acts

                if i == 0:
                    output = skip_acts
                else:
                    output += skip_acts

        with nn.parameter_scope('end'):
            # Initializing last layer to 0
            output = PF.convolution(output, 2*in_channels, kernel=(1,),
                                    w_init=ConstantInitializer(0.0),
                                    b_init=ConstantInitializer(0.0))
        return output


class WaveGlow(Module):

    def __init__(self, hparams):
        self.hparams = hparams
        # Module does not support ModuleList yet.
        for i in range(hparams.n_flows):
            setattr(self, f'WN_{i}', WN(hparams))

        n_half = hparams.n_samples_per_group // 2
        n_remaining_channels = hparams.n_samples_per_group
        for k in range(hparams.n_flows):
            if k % hparams.n_early_every == 0 and k > 0:
                n_half = n_half - hparams.n_early_size // 2
                n_remaining_channels = n_remaining_channels - hparams.n_early_size
        self.n_remaining_channels = n_remaining_channels

        mel_basis = librosa_mel_fn(hparams.sr, hparams.n_fft, n_mels=hparams.n_mels,
                                   fmin=hparams.mel_fmin, fmax=hparams.mel_fmax)
        self.basis = nn.Variable.from_numpy_array(mel_basis[None, ...])
        self.rng = np.random.RandomState(hparams.seed)

    def compute_mel(self, wave):
        hp = self.hparams
        reals, imags = F.stft(wave, window_size=hp.win_length,
                              stride=hp.hop_length, fft_size=hp.n_fft)
        linear = F.pow_scalar(
            F.add2(F.pow_scalar(reals, 2), F.pow_scalar(imags, 2)), 0.5)
        mels = F.batch_matmul(self.basis, linear)
        mels = F.log(F.clip_by_value(mels, 1e-5, np.inf)
                     ).apply(need_grad=False)
        return mels

    def call(self, wave):
        hp = self.hparams
        batch_size = hp.batch_size

        # compute mel-spectrogram from waveform
        with nn.parameter_scope('stft'):
            mels = self.compute_mel(wave)

        #  Upsample spectrogram to the size of audio
        with nn.parameter_scope('upsample'):
            with nn.parameter_scope('deconv'):
                mels = PF.deconvolution(
                    mels, hp.n_mels, kernel=(1024, ), stride=(256,))

            # make sure mels having the same length as wave
            if mels.shape[2] > wave.shape[1]:
                mels = mels[..., :wave.shape[1]]  # (B, L, n_mels)

            # transforming to correct shape
            mels = F.reshape(
                mels, mels.shape[:2] + (-1, hp.n_samples_per_group))
            mels = F.transpose(mels, (0, 2, 1, 3))
            mels = F.reshape(mels, mels.shape[:2] + (-1,))
            # (B, n_mels * n_groups, L/n_groups)
            mels = F.transpose(mels, (0, 2, 1))

        # reshape audio
        wave = F.reshape(wave, (batch_size, -1, hp.n_samples_per_group))
        wave = F.transpose(wave, (0, 2, 1))  # (B, n_groups, L/n_groups)

        output_audio, log_s_list, log_det_W_list = [], [], []

        for k in range(hp.n_flows):
            if k % hp.n_early_every == 0 and k > 0:
                output_audio.append(wave[:, :hp.n_early_size, :])
                wave = wave[:, hp.n_early_size:, :]

            # apply invertible convolution
            wave, log_det_W = invertible_conv(
                wave, reverse=False, rng=self.rng, scope=f'inv_{k}')
            log_det_W_list.append(log_det_W)

            n_half = wave.shape[1] // 2
            audio_0 = wave[:, :n_half, :]
            audio_1 = wave[:, n_half:, :]

            with nn.parameter_scope(f'wn_{k}'):
                output = getattr(self, f'WN_{k}')(audio_0, mels)
                log_s = output[:, n_half:, :]  # (B, n_half, L/n_groups)
                b = output[:, :n_half, :]      # (B, n_half, L/n_groups)
                audio_1 = F.add2(F.exp(log_s) * audio_1, b)
                log_s_list.append(log_s)

            # (B, n_half*2, L/n_groups)
            wave = F.concatenate(audio_0, audio_1, axis=1)

        output_audio.append(wave)

        return F.concatenate(*output_audio, axis=1), log_s_list, log_det_W_list

    def infer(self, mels, sigma=0.9):
        r"""Returns the generated audio.

        Args:
            mels (nn.Variable): Inputs containing mel-spectrograms of shape(B, n_mels, Ty).
                Defaults to None. If None, the mel spectrograms are infferred from data.
            sigma (float, optional): Sigma used to infer audio. Defaults to 0.9.

        Returns:
            nn.Variable: A synthetic audio.
        """

        hp = self.hparams
        with nn.parameter_scope('', self.parameter_scope):

            #  Upsample spectrogram to size of audio
            with nn.parameter_scope('upsample'):
                with nn.parameter_scope('deconv'):
                    mels = PF.deconvolution(
                        mels, hp.n_mels, kernel=(1024, ), stride=(256,))
                # cutout conv artifacts
                mels = mels[..., :-(1024 - 256)]  # kernel - stride

                # transforming to correct shape
                mels = F.reshape(
                    mels, mels.shape[:2]+(-1, hp.n_samples_per_group))
                mels = F.transpose(mels, (0, 2, 1, 3))
                mels = F.reshape(mels, mels.shape[:2] + (-1,))
                # (B, n_mels * n_groups, L/n_groups)
                mels = F.transpose(mels, (0, 2, 1))

            wave = F.randn(
                shape=(mels.shape[0], self.n_remaining_channels, mels.shape[2])) * sigma

            for k in reversed(range(hp.n_flows)):
                n_half = wave.shape[1] // 2
                audio_0 = wave[:, :n_half, :]
                audio_1 = wave[:, n_half:, :]

                with nn.parameter_scope(f'wn_{k}'):
                    output = getattr(self, f'WN_{k}')(audio_0, mels)
                    s = output[:, n_half:, :]
                    b = output[:, :n_half, :]
                    audio_1 = (audio_1 - b) / F.exp(s)
                    wave = F.concatenate(audio_0, audio_1, axis=1)

                wave = invertible_conv(
                    wave, reverse=True, rng=self.rng, scope=f'inv_{k}')

                if k % hp.n_early_every == 0 and k > 0:
                    z = F.randn(
                        shape=(mels.shape[0], hp.n_early_size, mels.shape[2]))
                    wave = F.concatenate(sigma * z, wave, axis=1)

            wave = F.transpose(wave, (0, 2, 1))
            wave = F.reshape(wave, (wave.shape[0], -1))

        return wave
