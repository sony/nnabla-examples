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
from neu.tts.module import Module

from .ops import UpBlock, sn_conv, wn_conv


class Generator(Module):
    def __init__(self, hp):
        self.hp = hp
        for i in range(len(hp.upsample_rates)):
            setattr(self, f"upblock_{i}", UpBlock(hp))

    def call(self, x):
        r"""Returns waveform from mel-spectrogram.

        Args:
            x (nn.Variable): Input mel-spectrogram variable of shape
                (B, n_mels, L).
        """
        hp = self.hp
        init_c = hp.upsample_initial_channel
        with nn.parameter_scope("init_conv"):
            x = wn_conv(x, init_c, kernel=(7,), pad=(3,))

        for i in range(len(hp.upsample_rates)):
            x = getattr(self, f"upblock_{i}")(x, i)

        with nn.parameter_scope("post_conv"):
            x = F.leaky_relu(x, 0.01)
            x = wn_conv(x, 1, kernel=(7,), pad=(3,))
            x = F.tanh(x)

        return x


class DiscriminatorPeriod(Module):

    def __init__(self, hp):
        self.hp = hp

    def call(self, x, p):
        r"""Returns discriminator period.

        Args:
            x (nn.Variable): Input variable of shape (B, 1, L).
            p (int): Period.

        Returns:
            List[nn.Variable]: List of feature maps.
        """
        results = list()

        b, c, t = x.shape
        if t % p:
            x = F.pad(x, (0, 0, 0, p - (t % p)), 'reflect')
            t = x.shape[-1]

        x = F.reshape(x, (b, c, t//p, p))

        for i, c in enumerate([32, 128, 512, 1024, 1024]):
            with nn.parameter_scope(f"conv_{i}"):
                x = wn_conv(x, c, (5, 1), stride=(3, 1)
                            if i < 4 else (1, 1), pad=(2, 0))
                x = F.leaky_relu(x, 0.1)
                results.append(x)

        with nn.parameter_scope("post_conv"):
            x = wn_conv(x, 1, (3, 1), pad=(1, 0))
            x = F.leaky_relu(x, 0.1)
            results.append(x)

        return results


class DiscriminatorScale(Module):

    def __init__(self, hp, spec_norm=False):
        self.hp = hp
        self.spec_norm = spec_norm

    def call(self, x):
        r"""Returns discriminator period.

        Args:
            x (nn.Variable): Input variable of shape (B, 1, L).

        Returns:
            List[nn.Variable]: List of feature maps.
        """
        nf = 128
        results = list()
        conv = sn_conv if self.spec_norm else wn_conv

        with nn.parameter_scope('first_conv'):
            x = conv(x, nf, (15,), pad=(7,))
            x = F.leaky_relu(x, 0.1)
            results.append(x)

        for i, s in enumerate([2, 2, 4, 4, 1]):
            with nn.parameter_scope(f'layer_{i}'):
                x = conv(
                    x, nf, (41,), stride=(s,), pad=(20,),
                    group=16 if i > 0 else 4
                )
                x = F.leaky_relu(x, 0.1, inplace=True)
                results.append(x)
                nf = min(nf * 2, 1024)

        with nn.parameter_scope('last_conv'):
            x = conv(x, nf, (5,), pad=(2,))
            x = F.leaky_relu(x, 0.1)
            results.append(x)

        with nn.parameter_scope('post_conv'):
            x = conv(x, 1, (3,), pad=(1,))
            results.append(x)

        return results


class MultiPeriodDiscriminator(Module):
    def __init__(self, hp):
        self.hp = hp
        for p in hp.periods:
            setattr(self, f"Pdisc_{p}", DiscriminatorPeriod(hp))

    def call(self, x):
        hp = self.hp
        results = list()
        for p in hp.periods:
            results.append(getattr(self, f"Pdisc_{p}")(x, p))
        return results


class MultiScaleDiscriminator(Module):
    def __init__(self, hp):
        self.hp = hp
        for i in range(3):
            setattr(self, f'Sdisc_{i}', DiscriminatorScale(hp, i == 0))

    def call(self, x):
        results = []
        for i in range(3):
            results.append(getattr(self, f'Sdisc_{i}')(x))
            x = F.average_pooling(
                x, (1, 4),
                stride=(1, 2),
                pad=(0, 2),
                including_pad=False
            )

        return results


class Discriminator(Module):
    def __init__(self, hp):
        self.hp = hp
        self.discp = MultiPeriodDiscriminator(hp)
        self.discs = MultiScaleDiscriminator(hp)

    def call(self, x):
        return self.discp(x) + self.discs(x)
